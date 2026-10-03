#!/usr/bin/env python3
"""Recompute the archived three-run comparison using CPU-only Python stdlib.

Paths are resolved relative to this file, so the command works from any CWD.
Default: rewrite comparison.json and comparison_metrics.csv in this directory.
--check: compare the computed bytes with the archived outputs, without writes.
No training, inference, new evaluation, GPU imports or confidence intervals.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics


os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUTPUT = Path(__file__).resolve().parent
RUNS = (
    ("optimized_b200", "8xB200 optimized", 8, "taihua-b200-optimized-20261003"),
    ("baseline_b200", "8xB200 original", 8, "taihua-b200-20260930"),
    ("h200", "4xH200 original", 4, "beijing-h200-20260927"),
)
ACCURACY_KEYS = (
    "math_mean_at_16", "aime24_mean_at_16", "aime25_mean_at_16", "macro_mean"
)
TIME_KEYS = {
    "normal_step_mean_s", "normal_step_median_s", "normal_generate_mean_s",
    "normal_update_mean_s", "summed_retained_step_hours",
    "launcher_wall_hours_including_smoke",
}
LIMITATIONS = [
    "These are independent stochastic on-policy trajectories, not paired fixed-input trials; token totals/means are close but per-step length distributions and later policies differ.",
    "Normal-step statistics exclude steps 50,100,...,1700 and1739, include initialization step1, and use1704steps in all three runs; H200 resumed initialization step51 remains included.",
    "timing/train_s includes update and actor-to-rollout weight sync; summed step hours include checkpoints/evaluation but exclude launcher preparation/smoke/model loading/interruption gaps.",
    "Only old/new B200 have complete all-mode launcher wall times; H200 retained summaries do not provide comparable whole-launcher wall duration.",
    "H200 uses4GPUs and FA3 rollout while B200 uses8GPUs and official FA4 rollout; no strict same-GPU-count hardware causal comparison.",
    "GPU-hours equal visible GPU count multiplied by duration, are not active-compute GPU-hours or billed cost.",
    "Final step1739 mean@16 uses500 MATH and30+30 AIME questions with16outputs each, not pass@16;480 AIME outputs are not480 independent questions.",
    "One run per profile without seed repetitions or step0/teacher evaluation; no confidence interval, statistical non-inferiority, accuracy preservation or improvement claim.",
    "Teacher is original non-GRPO Qwen3-8B; these results do not establish complete paper-accuracy reproduction.",
    "Historical best checkpoints are selected on reported benchmarks and must not be merged across benchmarks; comparisons here use the preplanned final step1739.",
    "Worker PyTorch allocated peaks exclude SGLang/device totals; old cumulative peaks versus new per-update-reset peaks have different semantics and are not a strict memory-usage comparison.",
    "GPU2 NVLink was abnormal in prior snapshots; no proof of identical hardware health throughout the three full runs or repair/firmware causality.",
    "Final weight files were inventoried but not reloaded for inference; extra-state and dataloader CPU checks do not constitute weight-content numerical validation.",
    "H200 training completed but wrapper exit5 from post-training overlay guard is retained, not rewritten as launcher success.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_run(key, label, gpu_count, directory):
    folder = RESULTS / directory
    summary = json.loads((folder / "summary.json").read_text())
    rows = load_csv(folder / "training_metrics.csv")
    evaluations = load_csv(folder / "eval_curve.csv")
    require(len(rows) == 1739, f"{key}: expected 1739 training rows")
    require([int(row["global_step"]) for row in rows] == list(range(1, 1740)),
            f"{key}: training steps must be continuous and unique")
    require(len(evaluations) == 35, f"{key}: expected 35 evaluations")
    require([int(row["step"]) for row in evaluations] ==
            list(range(50, 1701, 50)) + [1739], f"{key}: evaluation steps differ")
    normal = [row for row in rows if int(row["global_step"]) % 50
              and int(row["global_step"]) != 1739]
    require(len(normal) == 1704, f"{key}: expected 1704 ordinary steps")

    metrics = {}
    for csv_key, output_key in (
        ("timing/step_s", "normal_step_mean_s"),
        ("timing/generate_s", "normal_generate_mean_s"),
        ("timing/train_s", "normal_update_mean_s"),
    ):
        metrics[output_key] = statistics.mean(float(row[csv_key]) for row in normal)
        require(math.isclose(metrics[output_key], summary[output_key], rel_tol=1e-13),
                f"{key}: CSV and summary disagree for {output_key}")
    metrics["normal_step_median_s"] = statistics.median(
        float(row["timing/step_s"]) for row in normal)
    require(metrics["normal_step_median_s"] == summary["normal_step_median_s"],
            f"{key}: CSV and summary median disagree")
    metrics["summed_retained_step_hours"] = sum(
        float(row["timing/step_s"]) for row in rows) / 3600
    require(math.isclose(metrics["summed_retained_step_hours"],
                         summary["summed_retained_step_hours"], rel_tol=1e-13),
            f"{key}: CSV and summary step-time totals disagree")
    metrics["total_training_response_tokens"] = sum(
        int(float(row["tip/global_response_tokens"])) for row in rows)
    metrics["mean_training_response_tokens"] = (
        metrics["total_training_response_tokens"] / len(rows))
    metrics["total_normal_response_tokens"] = sum(
        int(float(row["tip/global_response_tokens"])) for row in normal)
    metrics["mean_normal_response_tokens"] = (
        metrics["total_normal_response_tokens"] / len(normal))
    metrics["retained_step_gpu_hours"] = (
        metrics["summed_retained_step_hours"] * gpu_count)
    metrics["launcher_wall_hours_including_smoke"] = summary.get(
        "launcher_wall_hours_including_smoke")
    wall = metrics["launcher_wall_hours_including_smoke"]
    metrics["launcher_allocated_gpu_hours"] = (
        wall * gpu_count if wall is not None else None)

    final = {name: float(evaluations[-1][name]) for name in ACCURACY_KEYS}
    for name, value in final.items():
        require(value == summary["final_evaluation"][name],
                f"{key}: final evaluation CSV and summary disagree for {name}")
    return {
        "label": label,
        "gpu_count": gpu_count,
        "sources": [{"file": str((folder / filename).relative_to(REPO)),
                     "sha256": sha256(folder / filename)}
                    for filename in ("summary.json", "training_metrics.csv", "eval_curve.csv")],
        "run_paths": {name: summary[name] for name in
                      ("run", "original_run", "resumed_run") if name in summary},
        "final_step": 1739,
        "normal_step_count": 1704,
        "evaluation_count": 35,
        "outputs_per_evaluation": 8960,
        "rollouts_per_step": 128,
        "total_training_rollouts": 222592,
        "training_completed": summary["training_completed"],
        "full_launcher_success": summary["full_launcher_success"],
        "gpu_wrapper_exit_code": summary["gpu_wrapper_exit_code"],
        "keepalive_restored_in_log": summary["keepalive_restored_in_log"],
        "performance": metrics,
        "final_evaluation_fraction": final,
        "final_evaluation_percent": {name: value * 100 for name, value in final.items()},
        "peak_reported_worker_allocated_gib": summary["peak_reported_torch_allocated_gib"],
        "reported_limitations": summary["limitations"],
    }


def compare(candidate_key, reference_key, runs):
    candidate = runs[candidate_key]
    reference = runs[reference_key]
    performance = {}
    for name, value in candidate["performance"].items():
        baseline = reference["performance"][name]
        if value is None or baseline is None:
            performance[name] = {
                "candidate": value, "reference": baseline, "absolute_delta": None,
                "relative_change_percent": None, "time_reduction_percent": None,
                "speedup_ratio": None,
                "reason": "reference full launcher wall time is not available in the retained H200 summary",
            }
            continue
        is_time = name in TIME_KEYS
        performance[name] = {
            "candidate": value, "reference": baseline,
            "absolute_delta": value - baseline,
            "relative_change_percent": (value / baseline - 1) * 100,
            "time_reduction_percent": (1 - value / baseline) * 100 if is_time else None,
            "speedup_ratio": baseline / value if is_time else None,
        }
    accuracy = {}
    for name, value in candidate["final_evaluation_fraction"].items():
        baseline = reference["final_evaluation_fraction"][name]
        accuracy[name] = {
            "candidate_fraction": value, "reference_fraction": baseline,
            "candidate_percent": value * 100, "reference_percent": baseline * 100,
            "delta_percentage_points": (value - baseline) * 100,
            "relative_change_percent": (value / baseline - 1) * 100,
        }
    return {"candidate": candidate_key, "reference": reference_key,
            "performance": performance, "final_accuracy": accuracy}


def build_report():
    runs = {key: load_run(key, label, count, folder)
            for key, label, count, folder in RUNS}
    comparisons = [compare(candidate, reference, runs) for candidate, reference in (
        ("optimized_b200", "baseline_b200"),
        ("optimized_b200", "h200"),
        ("baseline_b200", "h200"),
    )]
    return {
        "schema_version": 1,
        "scope": "Independent arithmetic comparison of final-step accuracy and CSV-derived performance for three completed training chains",
        "status": "No new training/inference/evaluation and no added confidence intervals",
        "metric_definitions": {
            "absolute_delta": "candidate minus reference, in the metric unit",
            "relative_change_percent": "(candidate/reference - 1) *100",
            "time_reduction_percent": "(1-candidate/reference)*100; positive means less elapsed time",
            "speedup_ratio": "reference elapsed time / candidate elapsed time",
            "accuracy_delta_percentage_points": "(candidate fraction-reference fraction)*100",
            "normal_step_filter": "global_step %50 !=0 and global_step !=1739; initialization step1 included",
            "normal_step_count": 1704,
            "retained_step_gpu_hours": "summed retained step hours * GPU count",
            "launcher_allocated_gpu_hours": "all-mode launcher wall hours including smoke * GPU count; unavailable for H200",
            "macro_mean": "unweighted arithmetic mean of MATH-500,AIME24,AIME25 mean@16 fractions",
        },
        "runs": runs, "comparisons": comparisons, "limitations": LIMITATIONS,
    }


def build_csv(report):
    specs = [
        ("normal_step_mean_s", "seconds", "performance"),
        ("normal_step_median_s", "seconds", "performance"),
        ("normal_generate_mean_s", "seconds", "performance"),
        ("normal_update_mean_s", "seconds", "performance"),
        ("summed_retained_step_hours", "hours", "performance"),
        ("launcher_wall_hours_including_smoke", "hours", "performance"),
        ("total_training_response_tokens", "tokens", "performance"),
        ("mean_training_response_tokens", "tokens_per_step", "performance"),
        ("total_normal_response_tokens", "tokens", "performance"),
        ("mean_normal_response_tokens", "tokens_per_normal_step", "performance"),
        ("retained_step_gpu_hours", "gpu_hours", "performance"),
        ("launcher_allocated_gpu_hours", "gpu_hours", "performance"),
        ("math_mean_at_16", "percent", "accuracy"),
        ("aime24_mean_at_16", "percent", "accuracy"),
        ("aime25_mean_at_16", "percent", "accuracy"),
        ("macro_mean", "percent", "accuracy"),
    ]
    fields = ["metric", "unit", "optimized_b200", "baseline_b200", "h200"]
    for prefix in ("optimized_vs_baseline", "optimized_vs_h200"):
        fields.extend(prefix + suffix for suffix in (
            "_absolute_delta", "_relative_change_percent", "_time_reduction_percent",
            "_speedup_ratio", "_accuracy_delta_pp",
        ))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for name, unit, kind in specs:
        row = {"metric": name, "unit": unit}
        for run_key, run in report["runs"].items():
            row[run_key] = (run["performance"][name] if kind == "performance"
                            else run["final_evaluation_percent"][name])
        for prefix, comparison in zip(
            ("optimized_vs_baseline", "optimized_vs_h200"), report["comparisons"][:2]
        ):
            if kind == "performance":
                values = comparison["performance"][name]
                for suffix in ("absolute_delta", "relative_change_percent",
                               "time_reduction_percent", "speedup_ratio"):
                    row[prefix + "_" + suffix] = values[suffix]
            else:
                values = comparison["final_accuracy"][name]
                row[prefix + "_absolute_delta"] = values["delta_percentage_points"]
                row[prefix + "_accuracy_delta_pp"] = values["delta_percentage_points"]
                row[prefix + "_relative_change_percent"] = values["relative_change_percent"]
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify archived output bytes without writing files")
    args = parser.parse_args()
    report = build_report()
    outputs = {
        "comparison.json": (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        "comparison_metrics.csv": build_csv(report),
    }
    if args.check:
        for filename, expected in outputs.items():
            require((OUTPUT / filename).read_bytes() == expected,
                    f"Archived output differs from recomputation: {filename}")
        print("CPU-only comparison check: both archived outputs match byte for byte")
    else:
        for filename, content in outputs.items():
            (OUTPUT / filename).write_bytes(content)
            print(f"Wrote {OUTPUT / filename}")


if __name__ == "__main__":
    main()
