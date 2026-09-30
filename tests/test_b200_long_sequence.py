"""Bounded real-Qwen capacity check, including colocated SGLang engines.

The caller supplies an existing resolved Hydra config and the normal GPU
wrapper (locks, keepalive, persistent caches and cleanup). This test changes
only its own trainer loop: it initializes the production workers, sleeps the
rollout engines, runs one actual TIP update on 128 synthetic maximum-length
sequences, and synchronizes/wakes the engines again. It writes no checkpoint.

  python tests/test_b200_long_sequence.py --config CONFIG.yaml --report REPORT.json

By default a short actual update allocates AdamW state before the single
maximum-length update. The synthetic inputs test capacity, not generation
quality or numerical equivalence to real rollout samples.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from omegaconf import OmegaConf
import ray
import torch

from opd.main_opd import OPDTaskRunner


ROOT = Path("/volume/pt-test/users/zhaoye")
GLOBAL_ROLLOUTS = 128
WORLD_SIZE = 8
PROMPT_LENGTH = 2048
RESPONSE_LENGTH = 8192
MAX_LENGTH = 16384
VOCAB = 151936


def monitor_summary(path):
    summary = {}
    for values in csv.reader(path.read_text(errors="replace").splitlines()):
        if len(values) != 4:
            continue
        try:
            index, total, used, free = [int(value.strip()) for value in values]
        except ValueError:
            continue
        record = summary.setdefault(index, {"index": index, "total_mib": total,
                                             "max_used_mib": used, "min_free_mib": free,
                                             "samples": 0})
        assert total == record["total_mib"]
        record["max_used_mib"] = max(record["max_used_mib"], used)
        record["min_free_mib"] = min(record["min_free_mib"], free)
        record["samples"] += 1
    assert set(summary) == set(range(WORLD_SIZE)), summary
    assert all(record["samples"] >= 2 for record in summary.values()), summary
    return {"csv": str(path), "sample_interval_ms": 1000,
            "scope": "NVML device-wide sampled usage, including colocated SGLang processes; sampled extrema, not exact transient peak",
            "devices": [summary[index] for index in range(WORLD_SIZE)],
            "max_used_mib": max(record["max_used_mib"] for record in summary.values()),
            "min_free_mib": min(record["min_free_mib"] for record in summary.values())}


def gpu_memory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"], text=True)
    rows = []
    for values in csv.reader(output.splitlines()):
        index, name, total, used, free = [value.strip() for value in values]
        rows.append({"index": int(index), "name": name, "total_mib": int(total),
                     "used_mib": int(used), "free_mib": int(free)})
    assert len(rows) == WORLD_SIZE and all("B200" in row["name"] for row in rows), rows
    return rows


def synthetic_batch(tokenizer, response_length):
    from verl.protocol import DataProto
    from opd.batch_builder import build_opd_batch

    # Avoid special-token/EOS handling. Both models consume the original
    # builder's shared prompt/response tensors, including production padding.
    prompt = 1000 + torch.arange(PROMPT_LENGTH).remainder(97)
    response = 1200 + torch.arange(response_length).remainder(127)
    prompts = prompt.expand(GLOBAL_ROLLOUTS, -1).clone()
    responses = response.expand(GLOBAL_ROLLOUTS, -1).clone()
    prompts += torch.arange(GLOBAL_ROLLOUTS)[:, None].remainder(13)
    responses += torch.arange(GLOBAL_ROLLOUTS)[:, None].remainder(17)
    rollout = DataProto.from_single_dict({
        "prompts": prompts, "responses": responses,
        "attention_mask": torch.ones((GLOBAL_ROLLOUTS, PROMPT_LENGTH + response_length), dtype=torch.long),
        "response_mask": torch.ones_like(responses),
    })
    batch = build_opd_batch(rollout, tokenizer, max_length=MAX_LENGTH)
    assert batch is not None and len(batch) == GLOBAL_ROLLOUTS
    for key in ("input_ids", "attention_mask", "position_ids", "loss_mask"):
        assert torch.equal(batch.batch[f"teacher_{key}"], batch.batch[f"student_{key}"])
    assert batch.batch["student_input_ids"].shape == (GLOBAL_ROLLOUTS, MAX_LENGTH)
    assert torch.equal(batch.batch["student_attention_mask"].sum(-1),
                       torch.full((GLOBAL_ROLLOUTS,), PROMPT_LENGTH + response_length))
    assert torch.equal(batch.batch["student_loss_mask"][:, 1:].sum(-1),
                       torch.full((GLOBAL_ROLLOUTS,), response_length, dtype=torch.float32))
    batch.meta_info.update(opd_loss_type="reverse_kl", opd_beta=0.5, opd_chunk_size=512)
    return batch


def capacity_fit(trainer, prime_optimizer):
    """Test-local loop; worker forward/loss/optimizer code stays unchanged."""
    assert trainer.total_training_steps == 1739, trainer.total_training_steps
    phases, updates = {}, []
    phases["initialized"] = gpu_memory()
    trainer.checkpoint_manager.update_weights()
    phases["engines_awake_initial"] = gpu_memory()
    lengths = [64, RESPONSE_LENGTH] if prime_optimizer else [RESPONSE_LENGTH]
    for step, response_length in enumerate(lengths, 1):
        trainer.checkpoint_manager.sleep_replicas()
        phases[f"step_{step}_engines_asleep"] = gpu_memory()
        print(f"[capacity] step={step} response_length={response_length} global_sequences=128 actual_Qwen_teacher_TIP_backward_Adam", flush=True)
        started = time.perf_counter()
        output = trainer.actor_rollout_wg.update_opd(synthetic_batch(trainer.tokenizer, response_length))
        elapsed = time.perf_counter() - started
        metrics = output.meta_info["metrics"]
        assert metrics and all(isinstance(values, list) and len(values) == WORLD_SIZE
                               for values in metrics.values()), "Expected metrics from all eight ranks"
        expected_tokens = GLOBAL_ROLLOUTS * response_length
        expected_selected = GLOBAL_ROLLOUTS * (response_length // 2)
        expected_cache = GLOBAL_ROLLOUTS // WORLD_SIZE * response_length * VOCAB * 2 / 1024**3
        expected_lr = 1e-6 * (1 + math.cos(math.pi * (step - 1) / 1739)) / 2
        for rank in range(WORLD_SIZE):
            for key, value in {"tip/enabled": 1, "tip/global_normalization": 1,
                               "tip/global_rollouts": GLOBAL_ROLLOUTS,
                               "tip/global_response_tokens": expected_tokens,
                               "tip/global_selected_tokens": expected_selected,
                               "tip/selected_fraction": 0.5, "opt/teacher_cache_cuda": 1,
                               "opt/resident_teacher": 1}.items():
                assert metrics[key][rank] == value, (step, rank, key, metrics[key][rank], value)
            assert metrics["opd/num_tokens"][rank] == expected_selected // WORLD_SIZE
            assert math.isclose(metrics["opt/teacher_cache_gib"][rank], expected_cache, rel_tol=1e-12)
            assert math.isclose(metrics["opd/lr"][rank], expected_lr, rel_tol=1e-8)
            assert math.isfinite(metrics["opd/loss"][rank]) and metrics["opd/loss"][rank] >= 0
            assert math.isfinite(metrics["opd/grad_norm"][rank]) and metrics["opd/grad_norm"][rank] > 0
            assert metrics["perf/max_memory_allocated_gb"][rank] > 0
            assert metrics["perf/update_peak_reserved_gib"][rank] > 0
        assert not any(key.startswith("profile/") for key in metrics), "Capacity gate requires profiling disabled"
        phases[f"step_{step}_after_update"] = gpu_memory()
        # This transition tests release of the large teacher cache before the
        # production SGLang pool and updated rollout weights become resident.
        trainer.checkpoint_manager.update_weights()
        phases[f"step_{step}_engines_awake"] = gpu_memory()
        updates.append({"step": step, "prompt_tokens": PROMPT_LENGTH,
                        "response_tokens_per_sequence": response_length,
                        "actual_sequence_length": PROMPT_LENGTH + response_length,
                        "padded_sequence_length": MAX_LENGTH, "global_sequences": GLOBAL_ROLLOUTS,
                        "teacher_cache_gib_per_rank": expected_cache,
                        "update_seconds": elapsed, "rank_metrics": metrics})
        print(f"[capacity] step={step} actual_update_and_engine_wakeup_PASS elapsed_s={elapsed:.3f}", flush=True)
    return {"passed": True, "world_size": WORLD_SIZE, "vocab": VOCAB,
            "optimizer_primed": prime_optimizer, "updates": updates, "memory_phases": phases,
            "scope": "Actual Qwen3-4B/Qwen3-8B eight-rank production worker initialization, colocated SGLang sleep/wake, full-vocabulary TIP scoring/reverse-KL/backward/Adam on synthetic maximum-length inputs; capacity only, no rollout quality/accuracy/long-run stability claim"}


class LongSequenceTaskRunner(OPDTaskRunner):
    def run(self, config, prime_optimizer):
        from opd.opd_trainer import OPDTrainer
        original_fit = OPDTrainer.fit
        result = {}

        def test_fit(trainer):
            result.update(capacity_fit(trainer, prime_optimizer))

        OPDTrainer.fit = test_fit
        try:
            super().run(config)
        finally:
            OPDTrainer.fit = original_fit
        assert result.get("passed") is True
        return result


def checked_config(path, report_parent, rollout_tp=None):
    config = OmegaConf.load(path)
    OmegaConf.set_struct(config, False)
    # Profiling is observation-only. The physical TP override is explicit and
    # must match the wrapper's selected production profile when provided.
    config.actor_rollout_ref.opd_profile = False
    if rollout_tp is not None:
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = rollout_tp
    expected = {
        "trainer.n_gpus_per_node": 8, "trainer.nnodes": 1,
        "data.train_batch_size": 8, "data.max_prompt_length": PROMPT_LENGTH,
        "data.max_response_length": RESPONSE_LENGTH,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.ref.fsdp_config.param_offload": False,
        "actor_rollout_ref.opd_teacher_cache_device": "cuda",
        "actor_rollout_ref.opd_resident_teacher": True,
        "actor_rollout_ref.opd_profile": False,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "cosine",
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.weight_decay": 0.1,
        "actor_rollout_ref.actor.grad_clip": 1.0,
        "actor_rollout_ref.rollout.n": 16,
        "actor_rollout_ref.rollout.free_cache_engine": True,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.6,
        "actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend": "fa4",
        "actor_rollout_ref.tip.enabled": True,
        "actor_rollout_ref.tip.keep_ratio": 0.5,
        "actor_rollout_ref.tip.entropy_clip_quantile": 0.98,
        "opd.loss_type": "reverse_kl", "opd.chunk_size": 512, "opd.max_length": MAX_LENGTH,
    }
    for key, value in expected.items():
        assert OmegaConf.select(config, key) == value, (key, OmegaConf.select(config, key), value)
    effective_tp = OmegaConf.select(config, "actor_rollout_ref.rollout.tensor_model_parallel_size")
    assert effective_tp in (1, 2)
    if "OPD_ROLLOUT_TP" in os.environ:
        assert effective_tp == int(os.environ["OPD_ROLLOUT_TP"]), "Capacity TP differs from wrapper's selected profile"
    assert Path(config.actor_rollout_ref.model.path).name == "Qwen3-4B"
    assert Path(config.actor_rollout_ref.ref.model.path).name == "Qwen3-8B"
    assert not OmegaConf.select(config, "opd.reward_beta")
    # Output/loop settings are test-only; the original worker math and the
    # complete 1739-step optimizer horizon are preserved.
    config.trainer.total_training_steps = 1739
    config.trainer.stop_after_steps = 0
    config.trainer.resume_mode = "disable"
    config.trainer.resume_from_path = None
    config.trainer.val_before_train = False
    config.trainer.default_local_dir = str(report_parent / "long-gate-output")
    config.trainer.validation_data_dir = str(report_parent / "long-gate-output")
    config.trainer.save_freq = -1
    config.trainer.test_freq = -1
    OmegaConf.resolve(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Resolved Hydra config from the candidate launcher")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--prime-optimizer", action=argparse.BooleanOptionalAction, default=True,
                        help="Prime AdamW with a short actual update before the capacity update (default: enabled).")
    parser.add_argument("--rollout-tp", type=int, choices=(1, 2),
                        help="Override only physical rollout parallelism; default uses the loaded config.")
    arguments = parser.parse_args()
    report = arguments.report.resolve()
    assert report.is_relative_to(ROOT), "Capacity reports must stay on Taihua GPFS"
    assert subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(ROOT)], text=True).strip() == "gpfs"
    for name in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "RAY_TMPDIR"):
        value = os.environ.get(name)
        assert value and Path(value).resolve().is_relative_to(ROOT), ("Persistent cache environment missing", name, value)
    assert os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert not ray.is_initialized(), "Run this standalone gate inside the exclusive GPU wrapper"
    gpu_memory()
    report.parent.mkdir(parents=True, exist_ok=True)
    config = checked_config(arguments.config, report.parent, arguments.rollout_tp)
    (report.parent / "long-sequence-config.yaml").write_text(OmegaConf.to_yaml(config))
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    kwargs = OmegaConf.to_container(config.ray_kwargs.get("ray_init", {}), resolve=True)
    runtime = OmegaConf.merge(get_ppo_ray_runtime_env(), kwargs.get("runtime_env", {}))
    kwargs["runtime_env"] = OmegaConf.to_container(runtime, resolve=True)
    runner = None
    result = {"passed": False}
    memory_csv = report.parent / "long-sequence-nvml.csv"
    monitor_stream = memory_csv.open("w")
    monitor = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits", "--loop-ms=1000"],
        stdout=monitor_stream, stderr=subprocess.STDOUT)
    try:
        ray.init(**kwargs)
        runner = ray.remote(num_cpus=1)(LongSequenceTaskRunner).remote()
        result = ray.get(runner.run.remote(config, arguments.prime_optimizer))
        result.update(source_config=str(arguments.config.resolve()),
                      source_config_sha256=hashlib.sha256(arguments.config.read_bytes()).hexdigest(),
                      gate_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      rollout_tp=int(config.actor_rollout_ref.rollout.tensor_model_parallel_size),
                      scheduler_total_training_steps=1739)
    except BaseException as error:
        result = {"passed": False, "error": repr(error)}
        raise
    finally:
        try:
            try:
                if runner is not None:
                    ray.kill(runner, no_restart=True)
            finally:
                ray.shutdown()
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait(timeout=10)
            monitor_stream.close()
            try:
                result["nvml_monitor"] = monitor_summary(memory_csv)
            except Exception as error:
                result.update(passed=False, nvml_error=repr(error))
            report.write_text(json.dumps(result, indent=2) + "\n")
    assert result.get("passed") is True, result
    print(f"Real-Qwen full-length capacity and SGLang wake PASS: {report}", flush=True)


if __name__ == "__main__":
    main()
