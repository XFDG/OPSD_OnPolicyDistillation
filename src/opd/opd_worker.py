"""
OPD Worker: extends verl's ActorRolloutRefWorker with divergence-based training.

A separate teacher model (ref) and trainable student model (actor) see the same
input sequences. The teacher produces better distributions naturally (bigger/stronger).

Training step:
  1. Forward teacher (ref model, frozen) on teacher_input_ids -> teacher logits
  2. Forward student (actor model, trainable) on student_input_ids -> student logits
  3. Compute token-wise divergence along response positions
  4. Backward and step optimizer
"""

import logging
import time

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.protocol import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.attention_utils import index_first_axis, rearrange, unpad_input
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.torch_functional import entropy_from_logits_with_chunking
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .losses import LOSS_FN_MAP, compute_teacher_token_stats, compute_tip_token_stats, select_tip_soft_or_indices

logger = logging.getLogger(__name__)


def _profile_clock(enabled):
    if enabled:
        torch.cuda.synchronize()
    return time.perf_counter()



def _shift_loss_mask_right_per_sequence(loss_mask_rmpad: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Shift loss-mask labels within each unpadded sequence only.

    The padded path uses ``loss_mask[:, 1:]`` so each logit at position ``t``
    is trained against whether token ``t + 1`` belongs to the response. After
    unpadding, we must preserve that same next-token alignment without letting
    the final token of one sample spill into the first token of the next.
    """
    shifted = torch.zeros_like(loss_mask_rmpad)
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        start = int(seq_start.item())
        end = int(seq_end.item())
        if end - start > 1:
            shifted[start : end - 1] = loss_mask_rmpad[start + 1 : end]
    return shifted


def _shift_ids_within_sequences(ids_rmpad: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Get next-token target IDs within each unpadded sequence.

    For each sequence, target[t] = ids[t+1]. The last token of each sequence
    gets 0 (won't be selected by loss_mask anyway).
    """
    shifted = torch.zeros_like(ids_rmpad)
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        s, e = int(seq_start.item()), int(seq_end.item())
        if e - s > 1:
            shifted[s : e - 1] = ids_rmpad[s + 1 : e]
    return shifted


class OPDWorker(AsyncActorRolloutRefWorker):
    """Worker with on-policy distillation training on top of verl's actor+rollout+ref.

    Inherits actor_module_fsdp (student) and ref_module_fsdp (teacher) from
    the HybridEngine ActorRolloutRef worker. Both teacher and student see the
    same input sequences (no privileged information).
    """

    def __init__(self, config, role: str, **kwargs):
        # The parent trainer's init_workers() creates hybrid-engine workers with
        # role="actor_rollout", which sets _is_ref=False and skips building
        # ref_module_fsdp.  OPD needs the colocated ref model as the teacher,
        # so we promote the role to "actor_rollout_ref".
        if role == "actor_rollout":
            role = "actor_rollout_ref"
        super().__init__(config=config, role=role, **kwargs)

    def _build_model_optimizer(self, *args, **kwargs):
        if kwargs.get("role") == "ref" and self.config.get("opd_resident_teacher", False):
            if args:
                raise ValueError("Resident reference requires the pinned keyword builder API")
            from .resident_ref import build_resident_ref
            return build_resident_ref(super()._build_model_optimizer, self, kwargs)
        return super()._build_model_optimizer(*args, **kwargs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_opd(self, data: DataProto) -> DataProto:
        """One OPD training step: divergence between teacher and student.

        Uses a two-phase approach to avoid holding both models on GPU simultaneously:
          Phase 1: Load teacher → forward → cache BF16 logits at configured location → optional offload
          Phase 2: Load student + optimizer → train with cached teacher logits → optional offload

        Input DataProto batch keys:
          teacher_input_ids, teacher_attention_mask, teacher_position_ids, teacher_loss_mask
          student_input_ids, student_attention_mask, student_position_ids, student_loss_mask

        Meta info:
          opd_loss_type: "reverse_kl" | "forward_kl" | "jsd"
          opd_beta: JSD interpolation (only used for jsd)
          opd_chunk_size: tokens per chunk for loss computation
        """
        assert self._is_actor, "update_opd requires actor role"

        def _mem(tag):
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info("[OPD-MEM] %s: allocated=%.2f GB, reserved=%.2f GB", tag, alloc, reserved)

        profiling = bool(self.config.get("opd_profile", False))
        profile_start = _profile_clock(profiling)
        torch.cuda.reset_peak_memory_stats()
        cache_device = self.config.get("opd_teacher_cache_device", "cpu")
        if cache_device not in ("cpu", "cuda"):
            raise ValueError("opd_teacher_cache_device must be cpu or cuda")
        ref_needs_offload = self.config.ref.fsdp_config.get("param_offload", False)

        data = data.to("cpu")
        loss_type = data.meta_info.get("opd_loss_type", "reverse_kl")
        beta = data.meta_info.get("opd_beta", 0.5)
        chunk_size = data.meta_info.get("opd_chunk_size", 512)

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        micro_batch_size = self.config.actor.get(
            "ppo_micro_batch_size_per_gpu",
            self.config.actor.get("micro_batch_size_per_gpu", 2),
        )
        device = get_device_id()

        batch_size = data.batch["student_input_ids"].shape[0]
        if batch_size == 0:
            return DataProto(meta_info={"metrics": {"opd/loss": 0.0, "opd/num_tokens": 0}})

        micro_batches = list(data.split(micro_batch_size))
        logger.info("[OPD-MEM] batch_size=%d, micro_batches=%d, micro_batch_size=%d, ref_needs_offload=%s",
                     batch_size, len(micro_batches), micro_batch_size, ref_needs_offload)
        _mem("before-ulysses")

        with self.ulysses_sharding_manager:
            # ------------------------------------------------------------------
            # Phase 1: Teacher forward — only ref model on GPU
            # ------------------------------------------------------------------
            _mem("phase1-before-ref-load")
            if hasattr(self, "ref_module_fsdp") and self.ref_module_fsdp is not None and ref_needs_offload:
                load_fsdp_model_to_gpu(self.ref_module_fsdp)
            _mem("phase1-after-ref-load")

            self.ref_module_fsdp.eval()
            forward_fn = self._forward_logits_unpadded if use_remove_padding else self._forward_logits_padded
            teacher_logits_cache = []  # list of (teacher_logits, is_valid)
            teacher_loaded = _profile_clock(profiling)
            cache_copy_seconds = 0.0

            for i, micro_batch in enumerate(micro_batches):
                micro_batch = micro_batch.to(device)
                valid_row_mask = micro_batch.batch.get("valid_row_mask")
                if valid_row_mask is not None:
                    valid_row_mask = valid_row_mask.bool()
                    if not valid_row_mask.any():
                        teacher_logits_cache.append((None, False))
                        continue

                t_input_ids = micro_batch.batch["teacher_input_ids"]
                t_attention_mask = micro_batch.batch["teacher_attention_mask"]
                t_position_ids = micro_batch.batch["teacher_position_ids"]
                t_loss_mask = micro_batch.batch["teacher_loss_mask"]

                if valid_row_mask is not None:
                    t_input_ids = t_input_ids[valid_row_mask]
                    t_attention_mask = t_attention_mask[valid_row_mask]
                    t_position_ids = t_position_ids[valid_row_mask]
                    t_loss_mask = t_loss_mask[valid_row_mask]

                logger.info("[OPD-MEM] phase1 micro_batch[%d]: teacher_ids shape=%s, loss_mask response_tokens=%d",
                            i, list(t_input_ids.shape), int(t_loss_mask[:, 1:].sum().item()))
                _mem(f"phase1-mb{i}-before-teacher-fwd")

                with torch.no_grad():
                    teacher_logits = forward_fn(
                        self.ref_module_fsdp, t_input_ids, t_attention_mask, t_position_ids, t_loss_mask
                    )
                logger.info("[OPD-MEM] phase1 micro_batch[%d]: teacher_logits shape=%s",
                            i, list(teacher_logits.shape))
                _mem(f"phase1-mb{i}-after-teacher-fwd")

                copy_start = time.perf_counter()
                teacher_logits_cache.append((teacher_logits.to(device if cache_device == "cuda" else "cpu"), True))
                cache_copy_seconds += time.perf_counter() - copy_start
                del teacher_logits
                _mem(f"phase1-mb{i}-after-cache-to-{cache_device}")

            teacher_done = _profile_clock(profiling)
            _mem("phase1-before-ref-offload")
            if ref_needs_offload:
                offload_fsdp_model_to_cpu(self.ref_module_fsdp)
            _mem("phase1-after-ref-offload")

            torch.cuda.empty_cache()
            _mem("phase1-after-empty-cache")

            # ------------------------------------------------------------------
            # Phase 2: Student forward + loss + backward — actor + optimizer on GPU
            # ------------------------------------------------------------------
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            _mem("phase2-after-actor-load")
            if self._is_offload_optimizer:
                load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=device)
            _mem("phase2-after-optimizer-load")

            student_loaded = _profile_clock(profiling)
            use_sample_weights = "sample_weights" in data.batch
            train_fn = self._tip_training_step if self.config.get("tip", {}).get("enabled", False) else self._opd_training_step
            metrics = train_fn(
                micro_batches, teacher_logits_cache,
                loss_type=loss_type, beta=beta, chunk_size=chunk_size,
                use_remove_padding=use_remove_padding, device=device, batch_size=batch_size,
                use_sample_weights=use_sample_weights,
            )

            student_done = _profile_clock(profiling)
            metrics["opt/teacher_cache_cuda"] = float(cache_device == "cuda")
            metrics["opt/resident_teacher"] = float(self.config.get("opd_resident_teacher", False))
            metrics["opt/teacher_cache_gib"] = sum(t.numel()*t.element_size() for t,valid in teacher_logits_cache if valid) / 1024**3
            # Cache has no consumer beyond this update; release before rollout resumes.
            del teacher_logits_cache
            if profiling:
                metrics.update({"profile/teacher_load_s": teacher_loaded-profile_start,
                                "profile/teacher_forward_cache_s": teacher_done-teacher_loaded,
                                "profile/cache_copy_host_s": cache_copy_seconds,
                                "profile/switch_models_s": student_loaded-teacher_done,
                                "profile/student_total_s": student_done-student_loaded})
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["opd/lr"] = lr.item() if torch.is_tensor(lr) else lr
            if metrics.get("tip/global_selected_tokens", metrics["opd/num_tokens"]) > 0:
                self.actor_lr_scheduler.step()
            else:
                metrics["opd/skipped_step"] = 1.0

            metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            metrics["perf/update_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
            print(f"[update-memory] rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else 0} peak_allocated_gib={metrics['perf/max_memory_allocated_gb']:.3f} peak_reserved_gib={metrics['perf/update_peak_reserved_gib']:.3f}", flush=True)
            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        if profiling:
            output.meta_info["metrics"]["profile/final_offload_s"] = _profile_clock(True)-student_done
            output.meta_info["metrics"]["profile/update_total_s"] = _profile_clock(True)-profile_start
        return output

    def _tip_training_step(
        self, micro_batches, teacher_logits_cache, loss_type="reverse_kl", beta=0.5,
        chunk_size=512, use_remove_padding=False, device=0, batch_size=0,
        use_sample_weights=False,
    ):
        """Global-batch Soft-OR ranking, then selected-token reverse-KL update.

        A detached scoring pass keeps the whole batch's normalization independent
        of GPU count and microbatch size. Only small score vectors are gathered.
        The second forward rebuilds the graph at selected response positions.
        """
        import torch.distributed as dist
        if loss_type != "reverse_kl" or use_sample_weights:
            raise ValueError("TIP requires unweighted reverse KL")
        if self.ulysses_sequence_parallel_size != 1:
            raise ValueError("This TIP implementation requires sequence parallel size 1")
        forward = self._forward_logits_unpadded if use_remove_padding else self._forward_logits_padded
        self.actor_module_fsdp.train()
        # Qwen3 dropout is zero. Reject stochastic dropout, which would make the
        # detached scoring pass and differentiated pass use different policies.
        if any(isinstance(m, torch.nn.Dropout) and m.p for m in self.actor_module_fsdp.modules()):
            raise ValueError("TIP two-pass scoring requires zero dropout")
        profiling = bool(self.config.get("opd_profile", False))
        scoring_start = _profile_clock(profiling)
        entropies, divergences, lengths, masks = [], [], [], []
        for micro, (teacher, valid) in zip(micro_batches, teacher_logits_cache, strict=True):
            if not valid:
                raise ValueError("TIP cannot dispatch an entirely invalid microbatch")
            micro = micro.to(device)
            rows = micro.batch.get("valid_row_mask")
            rows = rows.bool() if rows is not None else slice(None)
            ids, attn, pos, mask = [micro.batch["student_" + k][rows] for k in
                                  ("input_ids", "attention_mask", "position_ids", "loss_mask")]
            with torch.no_grad():
                logits = forward(self.actor_module_fsdp, ids, attn, pos, mask)
                entropy, divergence = compute_tip_token_stats(logits, teacher, chunk_size)
            entropies.append(entropy.cpu())
            divergences.append(divergence.cpu())
            lengths.extend(mask[:, 1:].sum(-1).long().tolist())
            masks.append(mask.cpu())
            del logits, entropy, divergence
        scoring_done = _profile_clock(profiling)
        local_h, local_d = torch.cat(entropies), torch.cat(divergences)
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        gathered = [None] * world
        payload = (local_h, local_d, lengths)
        if world > 1:
            dist.all_gather_object(gathered, payload)
        else:
            gathered[0] = payload
        all_h = torch.cat([x[0] for x in gathered])
        all_d = torch.cat([x[1] for x in gathered])
        all_lengths = [n for x in gathered for n in x[2]]
        chosen, _ = select_tip_soft_or_indices(
            all_h, all_d, all_lengths, self.config.tip.keep_ratio,
            self.config.tip.entropy_clip_quantile,
        )
        denominator = chosen.numel()
        start = sum(x[0].numel() for x in gathered[:rank])
        local_selected = chosen[(chosen >= start) & (chosen < start + local_h.numel())] - start
        del gathered, all_d
        if denominator == 0:
            raise ValueError("TIP selected no tokens in the global batch")
        self.actor_optimizer.zero_grad()
        selection_done = _profile_clock(profiling)
        offset, loss_sum, selected_count = 0, 0.0, 0
        for micro, (teacher, _), mask in zip(micro_batches, teacher_logits_cache, masks, strict=True):
            n_response = int(mask[:, 1:].sum())
            selected = local_selected[(local_selected >= offset) & (local_selected < offset + n_response)] - offset
            offset += n_response
            # Flat response logits follow row order, with each label mask shifted
            # to its preceding logit position by the forward helpers.
            selected_mask = torch.zeros_like(mask)
            positions = mask[:, 1:].nonzero()
            selected_positions = positions[selected]
            selected_mask[selected_positions[:, 0], selected_positions[:, 1] + 1] = 1
            micro = micro.to(device)
            rows = micro.batch.get("valid_row_mask")
            rows = rows.bool() if rows is not None else slice(None)
            ids, attn, pos = [micro.batch["student_" + k][rows] for k in
                             ("input_ids", "attention_mask", "position_ids")]
            logits = forward(self.actor_module_fsdp, ids, attn, pos, selected_mask.to(device))
            n = selected.numel()
            if n:
                teacher_selected = teacher[selected].to(device)
                loss, _ = LOSS_FN_MAP["reverse_kl"](teacher_selected, logits, chunk_size=chunk_size)
                # FSDP averages gradients across ranks; undo that average to
                # obtain the global selected-token mean, including all microbatches.
                scaled_loss = loss * (n * world / denominator)
                loss_sum += float(loss.detach()) * n
                selected_count += n
                del teacher_selected
            else:
                scaled_loss = logits.sum() * 0.0
            scaled_loss.backward()
            del logits, scaled_loss
        backward_done = _profile_clock(profiling)
        if isinstance(self.actor_module_fsdp, FSDP):
            norm = self.actor_module_fsdp.clip_grad_norm_(self.config.actor.get("grad_clip", 1.0))
        else:
            norm = torch.nn.utils.clip_grad_norm_(self.actor_module_fsdp.parameters(), self.config.actor.get("grad_clip", 1.0))
        if hasattr(norm, "full_tensor"):
            norm = norm.full_tensor()
        if not torch.isfinite(norm):
            self.actor_optimizer.zero_grad()
            raise FloatingPointError("TIP gradient norm is not finite")
        self.actor_optimizer.step()
        totals = torch.tensor([loss_sum, selected_count], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(totals)
        result = {
            "opd/loss": float(totals[0] / totals[1]),
            "opd/entropy": float(all_h.mean()),
            "opd/grad_norm": float(norm),
            "opd/num_tokens": selected_count,
            "opd/batch_size": batch_size,
            "opd/valid_rows": len(lengths),
            "tip/enabled": 1.0,
            "tip/global_response_tokens": all_h.numel(),
            "tip/global_selected_tokens": denominator,
            "tip/selected_fraction": denominator / max(1, all_h.numel()),
            "tip/global_rollouts": len(all_lengths),
            "tip/global_normalization": 1.0,
        }

        if profiling:
            result.update({"profile/tip_scoring_s": scoring_done-scoring_start,
                           "profile/tip_global_selection_s": selection_done-scoring_done,
                           "profile/student_forward_backward_s": backward_done-selection_done,
                           "profile/optimizer_s": _profile_clock(True)-backward_done})
        return result

    def _opd_training_step(
        self,
        micro_batches: list,
        teacher_logits_cache: list,
        loss_type: str = "reverse_kl",
        beta: float = 0.5,
        chunk_size: int = 512,
        use_remove_padding: bool = False,
        device: int = 0,
        batch_size: int = 0,
        use_sample_weights: bool = False,
    ) -> dict:
        """Core OPD training with pre-computed teacher logits.

        Teacher logits are already cached on CPU from phase 1. This phase only
        has the student (actor) model + optimizer on GPU, avoiding OOM from
        holding both models simultaneously.

        For each micro-batch:
          1. Move cached teacher logits to GPU
          2. Student forward (trainable actor) on student_input_ids -> logits
          3. Compute divergence loss (chunk-wise for memory efficiency)
          4. Backward with gradient accumulation scaling
        """
        self.actor_module_fsdp.train()

        active_micro_batches = sum(1 for _, is_valid in teacher_logits_cache if is_valid)
        grad_accum = max(1, active_micro_batches)

        loss_fn = LOSS_FN_MAP[loss_type]
        forward_fn = self._forward_logits_unpadded if use_remove_padding else self._forward_logits_padded

        self.actor_optimizer.zero_grad()
        total_loss = 0.0
        total_entropy_sum = 0.0
        total_tokens = 0
        total_rows = 0

        def _mem2(tag):
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info("[OPD-MEM] %s: allocated=%.2f GB, reserved=%.2f GB", tag, alloc, reserved)

        for i, (micro_batch, (cached_teacher_logits, is_valid)) in enumerate(
            zip(micro_batches, teacher_logits_cache, strict=True)
        ):
            if not is_valid:
                continue

            micro_batch = micro_batch.to(device)

            valid_row_mask = micro_batch.batch.get("valid_row_mask")
            if valid_row_mask is not None:
                valid_row_mask = valid_row_mask.bool()

            s_input_ids = micro_batch.batch["student_input_ids"]
            s_attention_mask = micro_batch.batch["student_attention_mask"]
            s_position_ids = micro_batch.batch["student_position_ids"]
            s_loss_mask = micro_batch.batch["student_loss_mask"]

            if valid_row_mask is not None:
                s_input_ids = s_input_ids[valid_row_mask]
                s_attention_mask = s_attention_mask[valid_row_mask]
                s_position_ids = s_position_ids[valid_row_mask]
                s_loss_mask = s_loss_mask[valid_row_mask]

            _mem2(f"phase2-mb{i}-before-teacher-to-gpu")
            teacher_logits = cached_teacher_logits.to(device)
            logger.info("[OPD-MEM] phase2 micro_batch[%d]: student_ids shape=%s, cached_teacher shape=%s",
                        i, list(s_input_ids.shape), list(teacher_logits.shape))
            _mem2(f"phase2-mb{i}-after-teacher-to-gpu")

            student_logits = forward_fn(
                self.actor_module_fsdp, s_input_ids, s_attention_mask, s_position_ids, s_loss_mask
            )
            logger.info("[OPD-MEM] phase2 micro_batch[%d]: student_logits shape=%s", i, list(student_logits.shape))
            _mem2(f"phase2-mb{i}-after-student-fwd")

            if teacher_logits.shape[0] != student_logits.shape[0]:
                raise RuntimeError(
                    "Teacher and student response token counts diverged. "
                    "This usually means prompt truncation dropped response tokens."
                )
            if teacher_logits.shape[0] == 0:
                del teacher_logits
                continue

            _mem2(f"phase2-mb{i}-before-loss")

            # Per-sample reward weighting: compute loss per sample, weight, then average
            mb_sample_weights = None
            if use_sample_weights and "sample_weights" in micro_batch.batch:
                mb_sample_weights = micro_batch.batch["sample_weights"]
                if valid_row_mask is not None:
                    mb_sample_weights = mb_sample_weights[valid_row_mask]
                mb_sample_weights = mb_sample_weights.to(device)

            if mb_sample_weights is not None:
                # Compute per-sample loss using loss_mask to identify sample boundaries
                # s_loss_mask: (n_valid_rows, seq_len), teacher/student_logits: (N_response_tokens, V)
                # We need to map response tokens back to samples
                per_sample_token_counts = s_loss_mask[:, 1:].sum(dim=1).long()  # tokens per sample
                sample_losses = []
                token_offset = 0
                for si in range(per_sample_token_counts.shape[0]):
                    n_tok = per_sample_token_counts[si].item()
                    if n_tok == 0:
                        sample_losses.append(torch.tensor(0.0, device=device))
                        continue
                    t_slice = teacher_logits[token_offset:token_offset + n_tok]
                    s_slice = student_logits[token_offset:token_offset + n_tok]
                    if loss_type == "jsd":
                        sl, _ = loss_fn(t_slice, s_slice, beta=beta, chunk_size=chunk_size)
                    else:
                        sl, _ = loss_fn(t_slice, s_slice, chunk_size=chunk_size)
                    sample_losses.append(sl)
                    token_offset += n_tok
                sample_losses = torch.stack(sample_losses)
                loss = (sample_losses * mb_sample_weights).sum() / mb_sample_weights.sum()
                n_tokens = int(per_sample_token_counts.sum().item())
            else:
                if loss_type == "jsd":
                    loss, n_tokens = loss_fn(teacher_logits, student_logits, beta=beta, chunk_size=chunk_size)
                else:
                    loss, n_tokens = loss_fn(teacher_logits, student_logits, chunk_size=chunk_size)

            del teacher_logits
            _mem2(f"phase2-mb{i}-after-loss-del-teacher")

            # Compute per-token entropy of the student policy (no grad needed)
            with torch.no_grad():
                token_entropy = entropy_from_logits_with_chunking(student_logits.float(), chunk_size=chunk_size)
                total_entropy_sum += token_entropy.sum().item()

            scaled_loss = loss / grad_accum
            scaled_loss.backward()
            _mem2(f"phase2-mb{i}-after-backward")

            total_loss += loss.detach().item()
            total_tokens += n_tokens
            total_rows += s_input_ids.shape[0]

        if total_tokens == 0:
            self.actor_optimizer.zero_grad()
            return {
                "opd/loss": 0.0,
                "opd/entropy": 0.0,
                "opd/grad_norm": 0.0,
                "opd/num_tokens": 0,
                "opd/batch_size": int(batch_size),
                "opd/valid_rows": int(total_rows),
            }

        # Gradient clipping and optimizer step
        grad_clip = self.config.actor.get("grad_clip", 1.0)
        if isinstance(self.actor_module_fsdp, FSDP):
            grad_norm = self.actor_module_fsdp.clip_grad_norm_(max_norm=grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module_fsdp.parameters(), max_norm=grad_clip
            )

        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

        if torch.isfinite(grad_norm):
            self.actor_optimizer.step()
        else:
            logger.warning("Non-finite grad_norm (%.4f), skipping step", grad_norm.item())
            self.actor_optimizer.zero_grad()

        return {
            "opd/loss": total_loss / max(1, active_micro_batches),
            "opd/entropy": total_entropy_sum / max(1, total_tokens),
            "opd/grad_norm": grad_norm.detach().item(),
            "opd/num_tokens": int(total_tokens),
            "opd/batch_size": batch_size,
            "opd/valid_rows": int(total_rows),
        }

    def _extract_response_target_ids_padded(self, input_ids, loss_mask):
        """Extract next-token target IDs at response positions (padded path)."""
        target_ids = input_ids[:, 1:]
        shift_mask = loss_mask[:, 1:]
        flat_target = target_ids.reshape(-1)
        flat_mask = shift_mask.reshape(-1)
        return flat_target[flat_mask.nonzero(as_tuple=True)[0]]

    def _extract_response_target_ids_unpadded(self, input_ids, attention_mask, loss_mask):
        """Extract next-token target IDs at response positions (unpadded path)."""
        _, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        ids_rmpad = input_ids.reshape(-1)[indices]
        target_ids_rmpad = _shift_ids_within_sequences(ids_rmpad, cu_seqlens)
        loss_mask_rmpad = loss_mask.reshape(-1)[indices]
        shifted_mask = _shift_loss_mask_right_per_sequence(loss_mask_rmpad, cu_seqlens)
        return target_ids_rmpad[shifted_mask.nonzero(as_tuple=True)[0]]

    def _forward_logits_padded(self, model, input_ids, attention_mask, position_ids, loss_mask):
        """Forward pass returning response-position logits (padded path).

        Returns: (N_response_tokens, vocab_size)
        """
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = outputs.logits
            del outputs

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_loss_mask = loss_mask[:, 1:]
        del logits

        B, S, V = shift_logits.shape
        flat_logits = shift_logits.reshape(B * S, V)
        flat_mask = shift_loss_mask.reshape(B * S)
        del shift_logits

        response_indices = flat_mask.nonzero(as_tuple=True)[0]
        return flat_logits[response_indices]

    def _forward_logits_unpadded(self, model, input_ids, attention_mask, position_ids, loss_mask):
        """Forward pass returning response-position logits (unpadded/flash_attn path).

        Returns: (N_response_tokens, vocab_size)
        """
        input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        use_ulysses = self.ulysses_sequence_parallel_size > 1
        if use_ulysses:
            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad,
                position_ids_rmpad=position_ids_rmpad,
                sp_size=self.ulysses_sequence_parallel_size,
            )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False,
            )
            logits_rmpad = outputs.logits.squeeze(0)
            del outputs

        if use_ulysses:
            logits_rmpad = gather_outputs_and_unpad(
                logits_rmpad,
                gather_dim=0,
                unpad_dim=0,
                padding_size=pad_size,
            )

        loss_mask_flat = loss_mask.reshape(-1)
        loss_mask_rmpad = loss_mask_flat[indices]

        shifted_loss_mask = _shift_loss_mask_right_per_sequence(loss_mask_rmpad, cu_seqlens)
        response_indices = shifted_loss_mask.nonzero(as_tuple=True)[0]
        return logits_rmpad[response_indices]
