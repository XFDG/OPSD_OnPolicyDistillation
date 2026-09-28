# Taihua 8×B200 TIP migration

Branch: `experiments/taihua-b200x8-tip`; baseline: H200 results commit `4abc896`.

**Status: official FA4 end-to-end smoke PASS, 2026-09-28 05:46 UTC. Full training has not been started.**

## Scope

Fresh independent reproduction from the original Qwen3-4B student and original
Qwen3-8B teacher. This does not resume or overwrite the completed H200 run.
An H200 world-size-4 sharded checkpoint is not accepted by the B200 resume gate.
The teacher remains different from the paper's GRPO teacher.

| Setting | H200 run | B200 profile |
|---|---|---|
| Devices / FSDP ranks | 4×H200 / 4 | 8×B200 / 8 |
| Global batch | 8 prompts ×16 =128 responses | Same |
| Per-rank microbatch | 1 | 1 |
| Rollout TP | 2 (2 replicas) | 2 (4 replicas) |
| TIP / loss | Soft-OR50%, global98% entropy clip/min-max, reverse KL, global selected-token mean | Same source implementation |
| Precision | BF16 forward, FP32 loss | Same, no FP8 |
| Teacher/student revisions | Pinned Qwen3-8B / Qwen3-4B | Same files, SHA256 checked |
| Data | DAPO seed42 80/20, train13918; MATH500/AIME24/AIME25 | Exact processed files copied and hashed |
| Sampling | n16, T1, p1, k−1, thinking=False | Same |
| Length | prompt2048, response8192, OPD16384 | Same |
| Optimizer | AdamW1e−6, cosine, wd0.1, clip1 | Same |
| Training / checkpoints / evaluation | 1epoch1739steps / every50+final / mean@16 | Same |
| Overlap | Experimental transfer overlap disabled | Same |
| Python / torch / SGLang | Python3.12 /2.9.1+cu128 /0.5.6 | Copied isolated venv, installation paths relocated |
| Trainer change | Existing serial TIP | Added flushed phase messages only |
| Training attention | FA2 2.8.3 | Same |
| Rollout attention | SGLang FA3 | Official Dao-AILab FA4, explicitly approved by user |
| SGLang | 0.5.6 | 0.5.6 with the documented FA4 compatibility backport below |

Changing GPU architecture, world size and rollout replica count can change BF16
rounding and sampled trajectories. The objective/settings are preserved; this is
not a claim of bitwise-identical weights or identical benchmark accuracy.

## Paths and lifecycle

On Taihua GPU `zhaoye-taihua-gpu-0`:

- Code: `/volume/pt-test/users/zhaoye/OPSD_B200`
- Assets: `/volume/pt-test/users/zhaoye/OPSD-B200-assets`
- Runs/caches/logs: `/volume/pt-test/users/zhaoye/OPSD-B200-runtime`
- Isolated venv: `/volume/pt-test/users/zhaoye/envs/opsd-py312-cu128`
- Pinned Verl source: `/volume/pt-test/users/zhaoye/OPSD-runtime/src/verl-0ddd289`

`run_b200_full.sh` supports `prepare`, `preflight`, `smoke`, `train`, `all`, `resume [checkpoint]`,
`resume-smoke [checkpoint]`, `status`, `follow`. From Beijing the same script forwards to Taihua over SSH. Default is **prepare**. Only an explicit `train` or
`all` starts full training. `all` verifies assets, runs the bounded smoke, and only
on success starts full training. Resume selects complete B200 production
checkpoints; it never selects H200 or smoke checkpoints.

During CPU preparation the known keepalive remains active. Before GPU work the
controller verifies the exact main script and GPU child processes, stops only
that keepalive parent, waits for idle GPUs, then starts the experiment. Exit paths
restore keepalive after GPU processes have exited. Foreign GPU work causes refusal.
No blanket `pkill`, `ray stop`, or forced termination of unrelated jobs is used.

All workload caches/temp/output paths are GPFS. Directory-wide `/root` and `/tmp`
growth checks remain enabled and may detect unrelated node activity; such a guard
failure is reported separately from optimizer completion.

## Progress

Console and GPFS logs display preparation stages, kernel checks, loading, rollout,
scoring, teacher/student update, weight synchronization, saving, and validation.
A Ray worker output follower bypasses stale driver forwarding. A 10-second
heartbeat shows the last completed update, checkpoint step and eight GPU stats.
Some Ray output may appear twice because both the driver and direct follower print it.

## Acceptance

Kernel preflight passed on all 8 B200s (2026-09-28): BF16 FA2 versus FP32 SDPA relative L2=0.00189861, finite backward gradients. Full preflight exited0; keepalive restored; /root growth208024bytes and /tmp0. The two CPU TIP oracle checks passed.

End-to-end smoke **FAILED before any optimizer update**, in SGLang FA3 initialization: SM100 is outside its SM80–SM90 gate. Exit1; keepalive restored; /root and /tmp growth0. This first attempt was **not accepted for full training**. After the official-source audit below, the user explicitly approved “用官方 FA4 继续 B200 smoke”. FlashInfer has not been enabled as the attention backend.

Official Dao-AILab source was downloaded and synchronized to Taihua at commit `e9cf2c1651d2303191eb40a739a3c135fda00999`. Its FA3 build still enumerates SM80/SM90 kernels and dispatches architectures ≥90 to SM90; the official README names FA4 for B200. No unsupported kernel build was installed and no architecture assertion bypassed. [Audit](results/b200-smoke-20260928/official-fa3-audit.json).

Sources: [official pinned README](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/README.md), [FA3 build](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/hopper/setup.py), [architecture dispatch](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/hopper/static_switch.h).
Smoke uses two real updates with the full batch and response limits, checkpoints
at steps1/2, two validation prompts ×16, global selection/gradient/LR checks and
eight-rank checkpoint checks. Its two-step cosine schedule is a test-only shortened
schedule; full training retains the 1739-step schedule. No full training is launched
by the migration work.

## Official FA4 compatibility adaptation

FA4 is built from the synchronized official source at
`e9cf2c1651d2303191eb40a739a3c135fda00999`, installed as
`flash-attn-4==0.0.0+e9cf2c1`. The built wheel SHA256 is
`0d13718a012608a3f05045f576d7dcac8e9952eed281a3516e1f940ad92f659f`.
The source is under `OPSD-B200-runtime/src/flash-attention-official`; offline wheels
are under `OPSD-B200-runtime/migration/fa4-wheels` on Taihua.
`bash scripts/b200/install_fa4_offline.sh` can recreate this FA4 layer on the
already copied venv while idle; normal launches verify the prepared environment.
This helper does not recreate the entire baseline venv or fetch assets.

SGLang 0.5.6 originally admits FA4 only for prefill. `scripts/b200/apply_fa4.py`
applies a narrow local compatibility patch, preserving original files and
recording original/patched SHA256 values:

- Admit FA4 decode on SM100-class hardware; use KV page size128.
- Route `sgl_kernel` FA4 calls to the official installed FA4 via
  `src/opd/fa4_adapter.py`: rename the LSE flag and adapt the return container.
- Update the local SGLang dependency metadata from Cutlass DSL4.2.1 to4.7.1.
  This is a locally patched 0.5.6 environment, not an untouched upstream release.
- FA3 architecture guards are retained. The FA4 decode admission follows
  [upstream SGLang v0.5.8](https://github.com/sgl-project/sglang/blob/v0.5.8/python/sglang/srt/layers/attention/flashattention_backend.py).

Dependencies added/updated: Cutlass DSL and base/core/cu12 libraries4.7.1,
quack-kernels0.6.5, torch-c-dlpack-ext0.1.5, setuptools-scm8.3.1, and
nvidia-cuda-nvdisasm13.4.92 (compiler tooling dependency, not a replacement of
the PyTorch CUDA12.8 runtime). All other installed package versions are recorded
in the environment artifact. `pip check` passes after the explicit backport.

FA4 kernel preflight passed on 2026-09-28: paged prefix prefill relative L2
0.00219515, paged decode0.00218118, CUDA graph replay with changed query/lengths
0.00226174 against FP32 math SDPA. Maximum absolute error was0.002612 or less.
The eight-GPU FA2 forward/backward gate passed again. The wrapper exited0,
restored all eight keepalive workers, and recorded zero `/root` and `/tmp` growth.
These gates validate tested kernel cases; end-to-end smoke acceptance is separate.
The first FA4 end-to-end attempt stopped before updates because the local SM100
check referenced an unimported `torch`; this was fixed to use SGLang’s existing
`is_sm100_supported()` helper. Its exit1 and successful keepalive restoration
are retained in `fa4-first-attempt-summary.log`.

## End-to-end acceptance: PASS

Run: `/volume/pt-test/users/zhaoye/OPSD-B200-runtime/runs/tip-20260928T053554Z-1134576`.
The launcher exited0, restored eight keepalive workers, and recorded zero growth
under `/root` and `/tmp`. No production training was launched.

| Check | Result |
|---|---|
| Real optimizer updates | 2/2 |
| Rollouts / update | 8 prompts ×16 =128 |
| Global TIP / selected fraction | Enabled /49.987%,49.986% |
| Loss | 0.0633666,0.0620515; finite |
| Gradient norm | 1.06545,1.16127; finite and nonzero |
| Smoke cosine LR | 1e−6,5e−7 |
| Checkpoints | Steps1/2; model/optim/extra-state on all8ranks |
| State checks | Scheduler step, RNG presence, dataloader cursor; pass |
| Validation | 2 prompts ×16 =32 outputs |
| Rollout time | 48.49s,30.74s |
| Train + weight synchronization | 80.37s,74.02s |
| Total step time | 138.30s,130.27s; includes smoke saving/evaluation |

Evidence: [structured checks](results/b200-smoke-20260928/smoke-verification.json),
[console summary](results/b200-smoke-20260928/smoke-summary.log),
[official installed-source verification](results/b200-smoke-20260928/official-fa4-source-verification.json),
[environment](results/b200-smoke-20260928/environment.json),
[local compatibility patch hashes](results/b200-smoke-20260928/fa4-backport.json).
These two steps are a functional test, not an accuracy benchmark or reliable
full-run speed estimate. Full evaluation uses many more prompts; sequence lengths
and checkpoint/evaluation overhead vary. Production resume after interruption
has not been exercised on B200.

## Manual full launch

From Beijing:

```bash
bash /volume/pt-train/users/zhaoye/OPSD_B200/run_b200_full.sh all
```

From the B200 GPU Pod:

```bash
bash /volume/pt-test/users/zhaoye/OPSD_B200/run_b200_full.sh all
```

`all` runs another bounded smoke before fresh full training; full training reloads
original Qwen3 weights and does not inherit smoke updates. `train` performs asset
verification and fresh full training without another smoke. In another terminal,
use the same entrypoint with `follow` for continuous console output or `status`
for a single progress snapshot. Exiting a separate `follow` viewer does not stop
the training process. Keep the training terminal alive (or use your normal durable
terminal session); the script does not claim recovery after a Pod restart.

For an interrupted B200 production run, use `resume` to discover a complete
checkpoint, or `resume /absolute/path/to/global_step_N` to select one. Completed
1739-step runs are excluded. This restoration path is retained from H200 with
8-rank validation; a separate B200 production resume smoke has not been run.
