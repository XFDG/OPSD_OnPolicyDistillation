# 8×B200 优化全量结果与审计（2026-10-03）

## 结论

**本次优化全量训练成功：1739 步、35 次完整评测、35 套八 rank checkpoint 均通过审计，GPU wrapper 和总启动器均 exit=0。** 总启动器从准备、smoke 到退出清理耗时 **37.6353 小时**，比此前原 B200 全量的 51.8386 小时减少 **27.3991%（14.2033 小时）**。普通步从 102.1982 秒降至 72.7804 秒，更新与权重同步从 71.2799 秒降至 41.4186 秒。

最终 step1739 的 mean@16 为 **MATH-500 79.4125%、AIME24 13.7500%、AIME25 18.1250%**。相对原 B200，分别变化 −0.1250、−0.4167、+0.2083 个百分点；三项等权宏平均下降 0.1111 个百分点。只有单次随机训练的观测，尚不能认定精度严格保持或发生显著退化。

训练退出后自动恢复保活。2026-10-03 **02:06:23 UTC（北京时间 10:06:23）**，既有控制器只读检查返回 exit=0、**8 个 GPU 保活 worker**；CPU 进程审计未见本次训练 worker。结果整理期间没有启动新训练或模型评测。

完整对照见 [优化 B200 / 原 B200 / H200](B200_OPT_VS_BASELINE_AND_H200.md)。这次完成的是既定原始 Qwen3-8B → Qwen3-4B 教师配置的优化实验；此前记录的论文差异继续存在，不等于论文全部精度条件已复现。

## 1. 实验身份与结束时间

| 项目 | 本次结果 |
|---|---|
| 分支 | `experiments/taihua-b200-optimization` |
| 生产源码基线 | `030c09d`；实际运行的37份文件以profile SHA256逐项核验 |
| Run ID | `tip-20261001T003630Z-3211352` |
| 硬件 / 分布式 | 8×B200；FSDP world size=8；rollout TP=2、4 个副本 |
| 正式更新 | step1–1739，连续且唯一；每步 8 prompts × 16 rollouts = 128 |
| 训练 rollout 总数 | 222592；不含 smoke 与验证生成 |
| 完整评测 | 每 50 步及最终 1739，共 35 次；每次 8960 条输出 |
| checkpoint | step50,100,…,1700,1739，共 35 套 |
| 总启动器开始 / 结束（UTC） | 2026-10-01 00:36:30 → 2026-10-02 14:14:37 |
| 总启动器开始 / 结束（北京） | 2026-10-01 08:36:30 → 2026-10-02 22:14:37 |
| 训练脚本开始 / PASS（UTC） | 2026-10-01 00:48:32 → 2026-10-02 14:13:40 |
| 原始运行目录 | `/volume/pt-test/users/zhaoye/OPSD-B200-opt-runtime/runs/tip-20261001T003630Z-3211352` |
| 日志基准 | 该目录的 `ray-live.log`，辅以 `driver.log`、`train-train.log`、`gpu-train.log` |

最终 checkpoint 的完整位置：

```text
/volume/pt-test/users/zhaoye/OPSD-B200-opt-runtime/runs/tip-20261001T003630Z-3211352/train-output/Qwen3-4B-b200-opt-full-TIP-rho0.5-reverse_kl-teacher-Qwen3-8B-nothink-lr1e-6-bs8-n16/global_step_1739
```

## 2. 成功判定与核验范围

本次审计在配对 GPU Pod 上以 `CUDA_VISIBLE_DEVICES=""` 执行，只读取 GPFS 上的日志、JSONL、文件清单以及小型 extra-state/dataloader 状态。完整权重和 optimizer shard 未被重新加载到模型做推理。

| 核验 | 结果 | 证据 |
|---|---|---|
| 训练步完整性 | 恰有 1739 个唯一连续 step，无缺口 | [逐步指标](training_metrics.csv) |
| 更新约束 | 所有步 rollout=128、global TIP normalization 生效；loss/grad finite；保留比例符合 gate | [汇总](summary.json)、逐步指标 |
| 优化配置确实执行 | 所有 1739 步 CUDA teacher cache / resident teacher 指标均为 1；profile=False、overlap=False | [运行 profile](run_profile.json) |
| 源码一致性 | 执行 profile 绑定的 37 份生产文件与审计时部署一致，无 mismatch | 运行 profile、[原始证据 hashes](source_manifest.json) |
| checkpoint 完整性 | 35 套；每套各 8 rank model/optimizer/extra-state 文件非空 | [清单](checkpoint_inventory.json) |
| 最终训练状态 | 8 rank scheduler 均为 1739，下一步 LR 均为 0，CPU/CUDA/NumPy/Python RNG 状态存在 | [状态复核](checkpoint_state_verification.json) |
| 数据进度 | cursor=1739，已消费 13912 prompts；13918 行数据按 batch8 留下末尾 6 行 | 状态复核 |
| 全量评测 | 35 份 JSONL 各 8960 条；重算得分与日志一致 | [评测证据](evaluation_evidence.json)、[35 点曲线](eval_curve.csv) |
| 正常收尾 | 训练完成标记、GPU train exit=0、launcher exit=0 | [退出摘录](completion-evidence.log)、[时间事件](completion_events.json) |
| 文件系统 guard | 正式训练 `/root` 增长 39963 bytes、`/tmp` 0，guard 通过；smoke 为 283 / 0 bytes | 退出摘录 |
| 保活 | 退出日志确认恢复；最新既有控制器 status=0、GPU_workers=8 | [只读状态](keepalive_status.json)、[进程摘要](process_state.json) |

文件存在与大小核验保证了保存产物结构完整。最终权重内容的重新加载推理不在本次 CPU 审计范围内；不能把结构核验写成独立的模型重载验证。

## 3. 模型、方法和 rollout 设置

| 设置 | 正式全量取值 |
|---|---|
| 学生 | 原始 `Qwen/Qwen3-4B`，revision `1cfa9a7208912126459214e8b04321603b3df60c` |
| 教师 | 原始 `Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Verl | `0ddd28933f2d06fbf06d2d4b2cec7da547d596fc` |
| 方法 / loss | global TIP，Soft-OR 50%，reverse KL，按全局入选 token 数归一化 |
| 选择规则 | entropy clip quantile=0.98；全局 min-max Soft-OR；每 response 按 floor(0.5×有效长度) 选择 |
| 前向 / 统计与 loss | BF16 / FP32；未使用 FP8 |
| 数据 / schedule | 固定资产 13918 行；1 epoch；batch8；1739 次更新 |
| 并行 / batch | FSDP8；rollout TP2；microbatch1；全局 8 prompts × n16 |
| 序列上限 | prompt2048 / response8192 / OPD padded16384；scoring/loss chunk512 |
| 训练优化器 | AdamW；LR=1e-6；cosine horizon1739；warmup0；weight decay0.1；grad clip1.0 |
| 采样 | temperature1.0、top_p1.0、top_k−1；thinking=False |
| 训练 attention | FA2 2.8.3；学生 gradient checkpointing 开启 |
| Rollout attention | 用户已批准的官方 FA4；memory utilization=0.6 |
| 验证 | MATH-500 500题、AIME24/25 各30题；每题 n16；同采样设置 |
| 保存 / 验证周期 | 每50步，加最终1739 |
| 本次优化 | teacher logits 存 CUDA；teacher/actor 参数与 AdamW 状态留 GPU；关闭相应 CPU offload |
| 正式计时 | profiling=False；同步 on-policy 更新和 rollout 权重同步；未使用 overlap |
| 固定迁移环境 | Python3.12、Torch2.9.1+cu128；来源及配置见 [环境记录](../../B200_MIGRATION.md)，本次未额外采样驱动版本 |

本次没有为了提速降低 rollout 数量、精度、response 上限、TIP 保留率或调度步数。实现中另有全局零 token rank 的 scheduler 一致性修复：以全局是否执行更新推进 scheduler，避免某 rank 本地零选择时 LR 落后。这个边界修复需要明确记录，不能声称代码除存储位置外完全相同。

缓存数值等价 gate 使用真实词表大小的合成 FSDP Embedding 更新路径；真实 Qwen smoke、最长序列和恢复提供另外的集成证据。它们没有证明两次独立随机训练全轨迹逐位相同。方法和实现证据见 [优化设计](../../OPD_OPTIMIZATION_DESIGN.md)、[有界验收](../../OPTIMIZATION_RESULTS.md)；教师类型与论文差异见 [复现设置](../../BEIJING_TIP_REPRO.md)。

## 4. 实际耗时

| 计时口径 | 时长 | 包含范围 |
|---|---:|---|
| 1739 个 `timing/step_s` 求和 | **37.3526 h** | rollout、更新/同步、步内保存和完整评测；不含启动器准备、smoke 和训练加载 |
| `starting train` → `train training PASS` | **37.4189 h** | 正式训练脚本边界，含训练初始化；终点在退出清理之前 |
| `mode=all` → `launcher exit=0` | **37.6353 h** | 准备、kernel 检查、smoke、正式训练和退出清理 |

普通步统一排除 step50,100,…,1700,1739，剩余 **1704 步**；为和旧两组同口径，保留初始化 step1，因此不称为严格纯稳态统计。平均整步 **72.7804 s**，中位数 **72.6533 s**；生成平均 **31.2929 s**，更新与权重同步平均 **41.4186 s**。两阶段计时与整步存在微小其他开销，不强行令二者求和等于整步。

生产 `tip/global_response_tokens` 合计 **436303728**，每步平均 **250893.46**。该指标统计蒸馏使用的有效 LLM response token，不包含所有 padding/tool token，也不是词表元素总数。与原 B200 总量仅相差 −0.06075%，但随机响应的逐步长度和策略轨迹仍不同。

![完整实验性能和精度概览](summary.png)

图同时标注 GPU 数量和时间口径；宏平均曲线纵轴截取 36–39%，用于观察 checkpoint 波动。原始数字见 [对照 JSON](comparison.json)、[对照 CSV](comparison_metrics.csv)，图片可用 [绘图脚本](plot_results.py) 重绘。

## 5. 完整评测结果

mean@16 是每题 16 次采样的平均正确率，再按题平均；这里每个 benchmark 固定 n16，因此等于正确输出数除以总输出数。**它不等于 pass@16。** AIME 的 480 个输出来自 30 道题，不能当成 480 道独立问题计算精度置信区间。

| 最终 step1739 | 正确 / 输出数 | mean@16 | 相对原 B200（百分点） |
|---|---:|---:|---:|
| MATH-500 | 6353 / 8000 | **79.4125%** | −0.1250 |
| AIME24 | 66 / 480 | **13.7500%** | −0.4167 |
| AIME25 | 87 / 480 | **18.1250%** | +0.2083 |
| 三项等权宏平均 | 不按题数混合 | **37.0958%** | −0.1111 |

本次宏平均最高的**单一 checkpoint 是 step450**：MATH 79.0000%、AIME24 14.1667%、AIME25 22.7083%，宏平均 **38.6250%**。它是在上述验证 benchmark 上事后选出的，不能作为独立测试集结果；主对照固定使用最后一步1739。不得把不同 checkpoint 的各单项最高值拼成一个模型。

最终更新 loss=0.06141701、grad norm=0.40723890、选中144042 / 有效288146 tokens，比例49.9892%。日志 `opd/lr=8.15906648e-13` 是最后一次更新所用的 scheduler 前 LR；checkpoint 中推进至1739后的下一步 LR=0，与完整 cosine schedule 一致。

## 6. 结论的范围

- **速度收益已由全量确认。** 与此前原 B200 比，完整 all 流程少27.40%、普通步少28.79%、更新与同步少41.89%。Rollout 生成平均反而多1.44%，观测收益主要位于更新路径。
- **准确率是单次运行观测。** 没有 seed 重复、step0 或教师评测，也没有统计非劣效验证；当前结果不支持“严格无损”或“显著退化”的表述。
- **论文对齐仍有边界。** 原始非 GRPO 教师等既有差异延续，此结果不证明论文全部精度复现。优化主要改变数据驻留和搬运，方法参数、rollout 及训练 schedule 保留。
- **显存统计不能混比。** 本次日志最大 worker Torch allocated 指标为60.6023 GiB；不覆盖 SGLang 和整卡总占用。旧组累计峰值和新组每 update 重置峰值语义不同，不能据此给出严格的显存增量。此前119.8564 GiB整卡占用来自长序列容量 smoke，不是本次全量峰值。
- **互联异常未视为修复。** GPU2 NVLink 异常见此前只读采样；本次整理没有做 reset、固件或驱动更改，也没有重新检查端口。缺乏三组全程同硬件健康状态的证明，不能把差距写成纯硬件对比，也不能将任务定性为单一 HBM 带宽瓶颈。
- **checkpoint 的验证层次明确。** 本次核验结构、extra-state 和数据游标；此前短恢复测试已通过，但没有对最终完整权重做新一轮推理重载。

## 7. 证据与复算

| 文件 | 用途 |
|---|---|
| [summary.json](summary.json) | 全量总览、固定配置、最终/最好 checkpoint、计时与限制 |
| [training_metrics.csv](training_metrics.csv) / [eval_curve.csv](eval_curve.csv) | 全部1739步与35点评测 |
| [evaluation_evidence.json](evaluation_evidence.json) | 每份原始评测JSONL的 SHA256、数量和重算得分 |
| [final_question_scores.csv](final_question_scores.csv) | 最终560题的16次采样得分聚合与输入SHA256；不附原始题文或模型回答 |
| [checkpoint_inventory.json](checkpoint_inventory.json) / [checkpoint_state_verification.json](checkpoint_state_verification.json) | 保存结构与最终八rank状态 |
| [run_profile.json](run_profile.json) / [source_manifest.json](source_manifest.json) | 实际执行配置、37份源码hash与原始日志来源 |
| [completion-evidence.log](completion-evidence.log) / [completion_events.json](completion_events.json) | 原始退出与时间事件摘录 |
| [process_state.json](process_state.json) / [keepalive_status.json](keepalive_status.json) | 只读收尾/保活状态 |
| [comparison.json](comparison.json) / [comparison_metrics.csv](comparison_metrics.csv) / [compare_runs.py](compare_runs.py) | 三组统一口径数据、公式与复算脚本 |
| [summary.png](summary.png) / [summary.svg](summary.svg) / [plot_results.py](plot_results.py) | 四联图和独立重绘脚本 |
| [artifact_manifest.json](artifact_manifest.json) | 本目录交付文件SHA256及审计脚本hash |

原始全量 JSONL 和 checkpoint 留在 Taihua GPFS，没有将权重、完整模型输出或私有硬件排查材料提交到本分支。新增审计入口为 [summarize_b200_optimized_run.py](../../scripts/summarize_b200_optimized_run.py)，复用已有的全量 B200 审计代码。

三方数据复算和图表重绘（普通 CPU，无需 GPU）：

```bash
cd /volume/pt-train/users/zhaoye/OPSD_B200_Optimize
python3 results/taihua-b200-optimized-20261003/compare_runs.py
python3 results/taihua-b200-optimized-20261003/plot_results.py
```

如需重新读取远端原始日志与全部评测，可以在 Taihua GPU Pod 使用已配置环境做 CPU 审计。输出、临时目录和缓存均指定 GPFS；此命令读取数据并生成审计文件，不启动模型训练：

```bash
OPSD_AUDIT_DIR=/volume/pt-test/users/zhaoye/OPSD-B200-opt-runtime/audit-20261003
mkdir -p "$OPSD_AUDIT_DIR"
env CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  TMPDIR="$OPSD_AUDIT_DIR" TMP="$OPSD_AUDIT_DIR" TEMP="$OPSD_AUDIT_DIR" \
  XDG_CACHE_HOME="$OPSD_AUDIT_DIR" TORCH_HOME="$OPSD_AUDIT_DIR" \
  TORCH_EXTENSIONS_DIR="$OPSD_AUDIT_DIR" TORCHINDUCTOR_CACHE_DIR="$OPSD_AUDIT_DIR" \
  TRITON_CACHE_DIR="$OPSD_AUDIT_DIR" CUDA_CACHE_PATH="$OPSD_AUDIT_DIR" \
  /volume/pt-test/users/zhaoye/envs/opsd-py312-cu128/bin/python \
  /volume/pt-test/users/zhaoye/OPSD_B200_Optimize/scripts/summarize_b200_optimized_run.py \
  --run /volume/pt-test/users/zhaoye/OPSD-B200-opt-runtime/runs/tip-20261001T003630Z-3211352 \
  --ray-log /volume/pt-test/users/zhaoye/OPSD-B200-opt-runtime/runs/tip-20261001T003630Z-3211352/ray-live.log \
  --output-dir "$OPSD_AUDIT_DIR/recheck-results"
```

## 8. 可用于汇报的摘要

完成8×B200优化全量实验，1739步、35次评测和35套checkpoint完整，正常退出并自动恢复八卡保活。相同模型、TIP和rollout设置下，采用GPU缓存与参数/optimizer驻留的本次运行，全流程37.64小时，对比原运行51.84小时减少27.4%；更新加同步耗时减少41.9%。最终三项mean@16为79.4125%/13.7500%/18.1250%，与原B200存在小幅升降，单次实验不能证明精度严格无损。GPU2历史互联异常和原始教师差异仍限制硬件归因与论文精度复现结论。
