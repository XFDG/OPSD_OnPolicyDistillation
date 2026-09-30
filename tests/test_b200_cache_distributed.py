"""Two-rank FSDP check of actual OPD update with CPU/CUDA teacher caches.

Run from the prepared experiment environment with all caches on GPFS:
  torchrun --standalone --nproc_per_node=2 tests/test_b200_cache_distributed.py report.json

The model is a synthetic embedding with the real Qwen3 vocabulary. This checks
the actual update/TIP/FSDP/AdamW/scheduler path, not Qwen attention or accuracy.
Both models and the optimizer stay on GPU in both variants, isolating cache
placement. --zero-rank additionally checks a rank selecting no tokens at all.
"""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import weakref

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from omegaconf import OmegaConf

from verl import DataProto
import opd.opd_worker as worker_module
from opd.opd_worker import OPDWorker


VOCAB = 151936
CHUNK_SIZE = 512
UPDATES = 2


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def snapshot(model, optimizer, scheduler):
    parameters = list(model.parameters())
    return {
        "weights": [p.detach().cpu().clone() for p in parameters],
        "gradients": [None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters],
        "adam": [
            {key: value.detach().cpu().clone() if torch.is_tensor(value) else value
             for key, value in optimizer.state[p].items()}
            for p in parameters
        ],
        "scheduler": scheduler.state_dict(),
    }


def compare_snapshots(first, second):
    for name in ("weights", "gradients"):
        assert len(first[name]) == len(second[name]), name
        for index, (left, right) in enumerate(zip(first[name], second[name], strict=True)):
            if left is None or right is None:
                assert left is right, (name, index)
            else:
                assert torch.equal(left, right), (name, index, float((left - right).abs().max()))
    assert len(first["adam"]) == len(second["adam"])
    for index, (left, right) in enumerate(zip(first["adam"], second["adam"], strict=True)):
        assert left.keys() == right.keys(), ("adam", index)
        for name in left:
            if torch.is_tensor(left[name]):
                assert torch.equal(left[name], right[name]), ("adam", index, name)
            else:
                assert left[name] == right[name], ("adam", index, name)
    assert first["scheduler"] == second["scheduler"], "scheduler"


def make_batch(lengths, rank):
    sequence_length = max(lengths) + 1
    ids = torch.zeros((len(lengths), sequence_length), dtype=torch.long)
    attention = torch.zeros_like(ids)
    loss_mask = torch.zeros_like(ids)
    for row, response_length in enumerate(lengths):
        length = response_length + 1
        ids[row, :length] = (torch.arange(length) + row * 7 + rank * 3) % 32
        attention[row, :length] = 1
        loss_mask[row, 1:length] = 1
    positions = torch.arange(sequence_length).expand_as(ids).clone()
    values = {"valid_row_mask": torch.ones(len(lengths), dtype=torch.bool)}
    for prefix in ("student", "teacher"):
        for key, value in (("input_ids", ids), ("attention_mask", attention),
                           ("position_ids", positions), ("loss_mask", loss_mask)):
            values[f"{prefix}_{key}"] = value.clone()
    batch = DataProto.from_single_dict(values)
    batch.meta_info.update(opd_loss_type="reverse_kl", opd_chunk_size=CHUNK_SIZE)
    return batch


def run_variant(location, lengths, all_lengths, initial_actor, initial_teacher, rank, device):
    actor = torch.nn.Embedding(32, VOCAB, dtype=torch.float32)
    actor.weight.data.copy_(initial_actor)
    teacher = torch.nn.Embedding(32, VOCAB, dtype=torch.bfloat16)
    teacher.weight.data.copy_(initial_teacher)
    teacher.requires_grad_(False)
    precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                               buffer_dtype=torch.float32)
    actor = FSDP(actor, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=precision, sync_module_states=True, use_orig_params=False)
    teacher = FSDP(teacher, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
                   mixed_precision=precision, sync_module_states=True, use_orig_params=False)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6, weight_decay=0.1,
                                  betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    initial_shards = [p.detach().cpu().clone() for p in actor.parameters()]
    assert all(p.dtype == torch.float32 for p in actor.parameters()), "FP32 actor master weights required"

    capture = {"statistics": [], "selection": [], "teacher_refs": [], "backward_tokens": []}
    original_stats = worker_module.compute_tip_token_stats
    original_selection = worker_module.select_tip_soft_or_indices

    def recording_stats(student_logits, teacher_logits, chunk_size):
        assert student_logits.dtype == torch.bfloat16
        assert teacher_logits.dtype == torch.bfloat16
        assert teacher_logits.device.type == location, (location, teacher_logits.device)
        capture["teacher_refs"].append(weakref.ref(teacher_logits))
        entropy, divergence = original_stats(student_logits, teacher_logits, chunk_size)
        capture["statistics"].append((tensor_hash(entropy), tensor_hash(divergence)))
        return entropy, divergence

    def recording_selection(entropy, divergence, response_lengths, keep_ratio, quantile):
        assert response_lengths == all_lengths, (response_lengths, all_lengths)
        indices, scores = original_selection(entropy, divergence, response_lengths, keep_ratio, quantile)
        capture["selection"].append({"indices": tensor_hash(indices), "scores": tensor_hash(scores),
                                      "entropy": tensor_hash(entropy), "KL": tensor_hash(divergence)})
        return indices, scores

    def forward(model, input_ids, attention_mask, position_ids, loss_mask):
        del attention_mask, position_ids
        if model is actor and torch.is_grad_enabled():
            capture["backward_tokens"].append(int(loss_mask[:, 1:].sum()))
        logits = model(input_ids)
        return logits[:, :-1][loss_mask[:, 1:].bool()]

    config = OmegaConf.create({
        "model": {"use_remove_padding": False},
        "actor": {"ppo_micro_batch_size_per_gpu": 1, "grad_clip": 1.0},
        "ref": {"fsdp_config": {"param_offload": False}},
        "tip": {"enabled": True, "keep_ratio": 0.5, "entropy_clip_quantile": 0.98},
        "opd_profile": False, "opd_teacher_cache_device": location,
    })
    worker = SimpleNamespace(
        config=config, _is_actor=True, _is_offload_param=False, _is_offload_optimizer=False,
        ulysses_sequence_parallel_size=1, ulysses_sharding_manager=nullcontext(),
        actor_module_fsdp=actor, ref_module_fsdp=teacher, actor_optimizer=optimizer,
        actor_lr_scheduler=scheduler, _forward_logits_padded=forward,
    )
    worker._tip_training_step = MethodType(OPDWorker._tip_training_step, worker)
    snapshots, records = [], []
    expected_selected = sum(math.floor(0.5 * length) for length in all_lengths)
    expected_local_selected = sum(math.floor(0.5 * length) for length in lengths)
    worker_module.compute_tip_token_stats = recording_stats
    worker_module.select_tip_soft_or_indices = recording_selection
    try:
        for update in range(1, UPDATES + 1):
            for value in capture.values():
                value.clear()
            torch.cuda.reset_peak_memory_stats(device)
            output = OPDWorker.update_opd.__wrapped__(worker, make_batch(lengths, rank))
            torch.cuda.synchronize(device)
            gc.collect()
            assert all(reference() is None for reference in capture["teacher_refs"]), "teacher cache retained after update"
            metrics = output.meta_info["metrics"]
            method_metrics = {key: value for key, value in metrics.items()
                              if key.startswith("opd/") or key.startswith("tip/")}
            assert metrics["tip/global_rollouts"] == len(all_lengths)
            assert metrics["tip/global_response_tokens"] == sum(all_lengths)
            assert metrics["tip/global_selected_tokens"] == expected_selected
            assert metrics["opd/num_tokens"] == expected_local_selected
            assert metrics["tip/global_normalization"] == 1
            assert math.isfinite(metrics["opd/loss"]) and metrics["opd/loss"] >= 0
            assert math.isfinite(metrics["opd/grad_norm"]) and metrics["opd/grad_norm"] > 0
            assert math.isclose(metrics["opd/lr"], (1e-6, 5e-7)[update - 1], rel_tol=1e-12)
            assert scheduler.last_epoch == update, ("scheduler did not advance globally", rank, update, scheduler.last_epoch)
            assert capture["backward_tokens"] == [math.floor(0.5 * length) for length in lengths]
            if 1 in lengths:
                assert 0 in capture["backward_tokens"], "zero-selected microbatch was not exercised"
            assert len(capture["statistics"]) == len(lengths) and len(capture["selection"]) == 1
            current = snapshot(actor, optimizer, scheduler)
            for weight, gradient in zip(current["weights"], current["gradients"], strict=True):
                assert weight.dtype == torch.float32 and torch.isfinite(weight).all()
                assert gradient is not None and gradient.dtype == torch.float32 and torch.isfinite(gradient).all()
            snapshots.append(current)
            records.append({"update": update, "method_metrics": method_metrics,
                            "statistics": list(capture["statistics"]),
                            "selection": list(capture["selection"]),
                            "backward_tokens": list(capture["backward_tokens"]),
                            "cache_released": True,
                            "memory_allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
                            "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3})
        assert any(not torch.equal(before, after) for before, after in
                   zip(initial_shards, snapshots[-1]["weights"], strict=True)), "FP32 actor weights never changed"
    finally:
        worker_module.compute_tip_token_stats = original_stats
        worker_module.select_tip_soft_or_indices = original_selection
    del worker, actor, teacher, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return snapshots, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--zero-rank", action="store_true",
                        help="Rank 0 selects no tokens; verifies the global optimizer/scheduler decision.")
    arguments = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    assert dist.get_world_size() == 2, "This gate requires exactly two ranks"
    device = torch.device("cuda", local_rank)
    lengths_by_rank = [[1, 1, 1], [1027, 7, 4]] if arguments.zero_rank else [[1, 1027, 4], [1, 1033, 7]]
    lengths = lengths_by_rank[rank]
    all_lengths = lengths_by_rank[0] + lengths_by_rank[1]
    assert any(math.floor(0.5 * length) > CHUNK_SIZE for length in all_lengths)
    generator = torch.Generator(device="cpu").manual_seed(20261001)
    initial_actor = torch.randn((32, VOCAB), generator=generator, dtype=torch.float32) * 0.1
    initial_teacher = (torch.randn((32, VOCAB), generator=generator, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    first, cpu_records = run_variant("cpu", lengths, all_lengths, initial_actor, initial_teacher, rank, device)
    second, cuda_records = run_variant("cuda", lengths, all_lengths, initial_actor, initial_teacher, rank, device)
    for update, (cpu_snapshot, cuda_snapshot, cpu_record, cuda_record) in enumerate(
            zip(first, second, cpu_records, cuda_records, strict=True), 1):
        compare_snapshots(cpu_snapshot, cuda_snapshot)
        for key in ("method_metrics", "statistics", "selection", "backward_tokens", "cache_released"):
            assert cpu_record[key] == cuda_record[key], (rank, update, key)
    report = {
        "rank": rank, "passed": True, "GPU": torch.cuda.get_device_name(device),
        "response_lengths": lengths, "zero_selected_rank": arguments.zero_rank and rank == 0,
        "cpu": cpu_records, "cuda": cuda_records,
        "final_weight_hashes": [tensor_hash(value) for value in second[-1]["weights"]],
    }
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, report)
    if rank == 0:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "passed": True, "world_size": 2, "updates_per_variant": UPDATES,
            "vocab": VOCAB, "chunk_size": CHUNK_SIZE, "actor_master_dtype": "float32",
            "forward_dtype": "bfloat16", "loss_dtype": "float32",
            "reduce_dtype": "float32", "manual_param_offload": False,
            "manual_optimizer_offload": False, "teacher_resident": True,
            "exact_equal": ["entropy", "reverse_KL", "Soft_OR_scores", "selected_tokens",
                            "loss", "gradient", "FP32_weights", "AdamW_state", "scheduler"],
            "cache_lifecycle": "All observed teacher cache tensor weakrefs expired after each actual update",
            "math_scope": "Actual OPDWorker.update_opd and TIP with synthetic FSDP embedding, real Qwen3 vocabulary; no Qwen attention, rollout or accuracy claim",
            "zero_selected_rank_case": arguments.zero_rank,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "verl": importlib.metadata.version("verl"),
            "worker_source_sha256": hashlib.sha256(Path(worker_module.__file__).read_bytes()).hexdigest(),
            "gate_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "ranks": reports,
        }
        arguments.report.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Distributed CPU/CUDA cache: actual update, two ranks, two updates, FP32 masters, chunk boundary, zero-selected microbatch, exact state equivalence PASS; report={arguments.report}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
