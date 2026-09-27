# Beijing TIP run

## Final result (audited 2026-09-27)

Training reached step 1739 with complete final evaluation/checkpoint. The GPU
wrapper exited 5 on its node-wide overlay-growth guard after training success
and keepalive restoration. See [the full results report](results/beijing-h200-20260927/RESULTS.md).
Final mean@16: MATH-500 **79.0375%**, AIME24 **13.1250%**, AIME25 **19.1667%**.
This is not a claim of full paper-result reproduction. Do not restart training
for result inspection.


## Model choice and reproduction scope

The user explicitly selects the original `Qwen/Qwen3-4B` student and original
`Qwen/Qwen3-8B` teacher. The latter is not TIP's unpublished GRPO teacher.
The downloaded `pb09204048` GRPO/DAPO checkpoint is not used.

Student revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
Teacher revision: `b968826d9c46dd6066d109eabc6255188de91218`.
Upstream code base: `39b3b0a5dccbc02b8f1908468bb685d2ea386d9a`.
Assets remain in `../OPSD_OnPolicyDistillation`; code runs from this worktree.

## Full experiment configuration

| Setting | Value and rationale |
|---|---|
| Method | TIP Soft-OR top 50% of response tokens per rollout |
| Score | Student entropy / log(vocab), reverse KL; entropy clipped at global-batch 98th percentile; both axes min-max normalized; h+d-h*d |
| Loss | Reverse KL averaged over all selected tokens in the global update; gradients correctly weighted across microbatches and four FSDP ranks |
| Scoring implementation | Detached scoring forward, gather small score vectors across ranks, select, then differentiated forward at selected positions; extra forward is an engineering cost |
| GPUs | 4 H200 rather than the paper's 8 H200 |
| Prompt batch | 8 prompts; interpret the paper's ambiguous “batch size (rollouts): 8” as prompt count |
| Rollouts per prompt | 16; 128 generated responses per update; checked at runtime |
| Parallelism | Rollout TP=2; FSDP across 4 GPUs; sequence parallel=1 |
| Memory | Microbatch 1 per GPU, gradient accumulation; teacher and actor parameter offload; optimizer offload; SGLang utilization 0.6 |
| Optimizer | AdamW, lr=1e-6, cosine schedule, no warmup, weight decay=0.1, gradient clip=1 |
| Lengths | Prompt 2048, response 8192, OPD limit 16384, KL chunk 512 |
| Train sampling | temperature=1, top_p=1, top_k=-1 |
| Thinking | False, following the upstream math launcher's explicit Qwen setting; the paper does not specify this for math |
| Data | Existing pinned DAPO-Math-17k-dedup; 13,918 train / 3,480 held out, seed 42; unchanged 80/20 project split |
| Duration | One epoch, nominally 1,739 updates before any filtering changes. This gives about 222k training responses, comparable to the old 15-epoch n=1 run's 207k. Paper math epoch count is unspecified |
| Validation | MATH-500, AIME24, AIME25; 16 samples/question, temperature=1, top_p=1, top_k=-1; every 50 steps and final step |
| Checkpoints | Every 50 steps and final step, saved before evaluation; all checkpoints retained |
| Policy freshness | Generate → update → synchronize actor weights → next rollout; no stale-policy pipeline overlap |

## Entrypoint

`bash run_beijing_tip_full.sh prepare|smoke|train|all`

The script runs from Beijing CPU `yzhao04-0` or GPU `zhaoye-gpu-0`, checks the
pinned environment/models/data, logs to shared GPFS with live console output,
checks/starts keepalive during preparation, stops it for GPU work, and restores
it after completion or handled failure. Existing OPSD node locks prevent overlap.

`smoke` uses the full training batch/rollout/length/TP settings for two updates,
with 16 training prompts and 2 validation prompts. It saves at each update and
evaluates at step 2. Full training always starts from the original student;
smoke checkpoints are under a separate output directory.

## Verified smoke (2026-09-24)

Run: `/volume/pt-train/users/zhaoye/OPSD-runtime/runs/tip-20260924T152808Z-3601158`.
The complete launcher exited 0 at 15:35:55 UTC; GPU cleanup exited 0 and
automatically restored keepalive (PID 541070 at verification time).
Two optimizer updates each generated 128 responses, selected approximately
49.985% / 49.987% of response tokens, and had finite loss and gradients.
Both four-rank model/optimizer/state checkpoints and 32 validation outputs
passed the automated gate in `smoke-verification.json`.
Step times were 140.19 s and 136.96 s. These are smoke-sample timings, not
a full-data throughput guarantee. At that historical smoke checkpoint, full training had not yet started; it has since completed as reported above.

## References

- TIP, sections 6/7 and appendix table 9: https://arxiv.org/html/2604.14084
- Upstream math launcher: https://github.com/XFDG/OPSD_OnPolicyDistillation/blob/main/scripts/opd/train_opd.sh
- Verl batch-size convention: https://github.com/verl-project/verl/blob/main/docs/perf/best_practices.rst
- Verl memory tuning: https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst
- SGLang weight-sync memory discussion: https://github.com/verl-project/verl/issues/6733

Forum/issue reports inform conservative memory settings; correctness is decided
by the local smoke and numerical oracle tests rather than copied anecdotes.

## Checkpoint recovery verified

The legacy entrypoint now accepts `resume [CHECKPOINT]` and
`resume-smoke [CHECKPOINT]`. Without an explicit path it reads the verified
original step-50 path from `OPSD-runtime/state/tip-resume.path`.
Model, optimizer, scheduler, RNG and dataloader state are restored; outputs go
to a new run directory. The LR horizon remains 1739, with save/eval every 50.

Recovery smoke `tip-20260924T172523Z-3619724` passed with launcher exit 0:
steps 51 and 52, initial LR 9.97961619330696e-7, data cursor advanced to 416
prompts, both full checkpoints saved, 32 validation outputs, keepalive restored.
This is the verified fallback if the experimental overlap path fails.

## Latest user instruction: manual launch only

The automatically launched recovery run `tip-20260924T174236Z-3628953` was
stopped at the user's request (exit 143). No training workers remained and
keepalive was restored. Do not automatically restart training.
The complete manual entrypoint is `../OPSD_OnPolicyDistillation/old.sh`;
the currently deployed `old.sh` discovers the highest complete serial production checkpoint. The original `run_beijing_tip_full.sh` still uses the recovery pointer if no path is supplied. The production run is now complete; do not relaunch it for status inspection.
