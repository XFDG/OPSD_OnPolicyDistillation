#!/usr/bin/env python3
"""Reproduce the four-panel results figure from local saved evidence.

Reads each run's summary.json, training_metrics.csv, and eval_curve.csv.
Writes only summary.png and summary.svg next to this script; no GPU or SSH use.
"""
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
RUNS = [
    ("taihua-b200-20260930", "8 x B200 baseline", "#6786a4"),
    ("taihua-b200-optimized-20261003", "8 x B200 optimized", "#2f8a73"),
    ("beijing-h200-20260927", "4 x H200", "#ac7c45"),
]
BENCHMARKS = [
    ("math_mean_at_16", "MATH-500"),
    ("aime24_mean_at_16", "AIME 2024"),
    ("aime25_mean_at_16", "AIME 2025"),
    ("macro_mean", "Macro"),
]


def load_run(directory, label, color):
    path = RESULTS / directory
    summary = json.loads((path / "summary.json").read_text())
    with (path / "training_metrics.csv").open(newline="") as handle:
        metrics = list(csv.DictReader(handle))
    step_ids = [int(float(row["global_step"])) for row in metrics]
    if len(step_ids) != 1739 or sorted(step_ids) != list(range(1, 1740)):
        raise ValueError(f"{directory}: expected 1739 unique, continuous training steps")
    normal = [row for row in metrics if int(float(row["global_step"])) % 50 != 0
              and int(float(row["global_step"])) != 1739]
    if len(normal) != 1704:
        raise ValueError(f"{directory}: ordinary-step count mismatch")
    for field, key in [("timing/step_s", "normal_step_mean_s"),
                       ("timing/generate_s", "normal_generate_mean_s"),
                       ("timing/train_s", "normal_update_mean_s")]:
        values = [float(row[field]) for row in normal]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{directory}: nonfinite timing")
        if not math.isclose(statistics.mean(values), summary[key], abs_tol=1e-9):
            raise ValueError(f"{directory}: CSV/summary mean mismatch for {field}")
    recorded_hours = sum(float(row["timing/step_s"]) for row in metrics) / 3600
    if not math.isclose(recorded_hours, summary["summed_retained_step_hours"], abs_tol=1e-9):
        raise ValueError(f"{directory}: recorded step-hour mismatch")

    with (path / "eval_curve.csv").open(newline="") as handle:
        curve = [{key: float(value) for key, value in row.items()}
                 for row in csv.DictReader(handle)]
    if [int(row["step"]) for row in curve] != list(range(50, 1701, 50)) + [1739]:
        raise ValueError(f"{directory}: expected 35 evaluation checkpoints")
    for row in curve:
        macro = statistics.mean(row[key] for key, _ in BENCHMARKS[:3])
        if not math.isclose(macro, row["macro_mean"], abs_tol=1e-12):
            raise ValueError(f"{directory}: macro evaluation mismatch")
    for key, _ in BENCHMARKS:
        if not math.isclose(curve[-1][key], summary["final_evaluation"][key], abs_tol=1e-12):
            raise ValueError(f"{directory}: final CSV/summary evaluation mismatch")
    return {"summary": summary, "curve": curve, "label": label, "color": color}


def main():
    runs = [load_run(*args) for args in RUNS]
    labels = [run["label"] for run in runs]
    colors = [run["color"] for run in runs]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    a, b, c, d = axes.flat

    # A: identically filtered ordinary steps; step 1 is retained for all runs.
    gen_color, update_color, other_color = "#759cba", "#dda56f", "#bbbbbb"
    for i, run in enumerate(runs):
        s = run["summary"]
        generation = s["normal_generate_mean_s"]
        update = s["normal_update_mean_s"]
        residual = s["normal_step_mean_s"] - generation - update
        if residual < -1e-6:
            raise ValueError("Negative ordinary-step timing residual")
        a.barh(i, generation, height=0.50, color=gen_color)
        a.barh(i, update, left=generation, height=0.50, color=update_color)
        a.barh(i, max(0, residual), left=generation+update, height=0.50, color=other_color)
        a.text(generation / 2, i, f"{generation:.2f}", ha="center", va="center")
        a.text(generation + update / 2, i, f"{update:.2f}", ha="center", va="center")
        a.text(s["normal_step_mean_s"] + 1, i,
               f"{s['normal_step_mean_s']:.2f} s", va="center", fontsize=10)
    a.set_yticks(range(3), labels)
    a.invert_yaxis()
    a.set_xlim(0, 125)
    a.set_xlabel("Mean recorded time per ordinary step (seconds)")
    a.set_title("A. Ordinary steps: generation + update (n = 1,704/run)", pad=14)
    a.legend(handles=[Patch(facecolor=gen_color, label="Generation"),
                      Patch(facecolor=update_color, label="Update + weight sync"),
                      Patch(facecolor=other_color, label="Other")],
             loc="upper center", bbox_to_anchor=(0.5, -0.24),
             fontsize=9, frameon=False, ncol=3)
    a.grid(axis="x", color="#dddddd", alpha=0.6)
    a.set_axisbelow(True)

    # B: timer sum includes saved/evaluated boundary steps, not launcher wall time.
    hours = [run["summary"]["summed_retained_step_hours"] for run in runs]
    b.bar(range(3), hours, color=colors, width=0.62)
    for i, value in enumerate(hours):
        b.text(i, value + 1.0, f"{value:.2f} h", ha="center", weight="bold")
    b.set_xticks(range(3), ["8 x B200\nbaseline", "8 x B200\noptimized", "4 x H200"])
    b.set_ylim(0, 62)
    b.set_ylabel("Sum of recorded step timers (hours)")
    b.set_title("B. Recorded training + scheduled evaluation time", pad=14)
    b.text(0.5, 0.94, "1,739 retained step timers/run; excludes startup gaps",
           transform=b.transAxes, ha="center", fontsize=9, color="#555555")
    b.grid(axis="y", color="#dddddd", alpha=0.6)
    b.set_axisbelow(True)

    # C: fixed final checkpoint and mean@16, never best checkpoint or pass@16.
    width = 0.24
    for i, run in enumerate(runs):
        x = [group + (i - 1) * width for group in range(4)]
        values = [100 * run["summary"]["final_evaluation"][key] for key, _ in BENCHMARKS]
        c.bar(x, values, width=width, color=run["color"], label=run["label"])
        for coordinate, value in zip(x, values):
            c.text(coordinate, value + 1.1, f"{value:.2f}", ha="left",
                   rotation=45, va="bottom", fontsize=8)
    c.set_xticks(range(4), [label for _, label in BENCHMARKS])
    c.set_ylim(0, 100)
    c.set_ylabel("Mean@16 accuracy (%)")
    c.set_title("C. Final checkpoint: step 1,739", pad=14)
    c.legend(loc="upper right", frameon=False, fontsize=9)
    c.grid(axis="y", color="#dddddd", alpha=0.6)
    c.set_axisbelow(True)

    # D: label the zoomed vertical scale explicitly; 35 unsmoothed points/run.
    for run in runs:
        x = [row["step"] for row in run["curve"]]
        y = [100 * row["macro_mean"] for row in run["curve"]]
        if any(value < 36 or value > 39 for value in y):
            raise ValueError("Evaluation value falls outside the explicitly zoomed axis")
        d.plot(x, y, color=run["color"], label=run["label"],
               linewidth=1.6, marker="o", markersize=3, alpha=0.9)
    d.set_xlim(0, 1739)
    d.set_xticks([0, 500, 1000, 1500, 1739])
    d.set_ylim(36, 39)
    d.set_xlabel("Training step")
    d.set_ylabel("Macro mean@16 (%) — zoomed axis: 36–39")
    d.set_title("D. Macro evaluation curves: 35 points/run", pad=14)
    d.legend(loc="lower left", fontsize=9, frameon=False, ncol=1)
    d.grid(color="#dddddd", alpha=0.6)

    fig.suptitle("OPSD reproduction: completed-run comparison", fontsize=18, y=0.98)
    fig.text(0.5, 0.045,
             "Ordinary-step filter: exclude scheduled save/evaluation steps (50, 100, ..., 1700, 1739); retain initialization step 1.\n"
             "Generation/update timers include synchronization and orchestration. Mean@16 averages 16 sampled answers per question (not pass@16).\n"
             "GPU counts and execution settings differ across runs. Single runs do not establish statistical non-inferiority or a hardware-only effect.",
             ha="center", va="bottom", fontsize=10, color="#555555")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.16, hspace=0.42, wspace=0.36)
    fig.savefig(ROOT / "summary.png", dpi=180, facecolor="white")
    fig.savefig(ROOT / "summary.svg", facecolor="white")
    plt.close(fig)
    print("Verified 1,739 training steps and 35 evaluation checkpoints per run; saved summary.png/.svg.")


if __name__ == "__main__":
    main()
