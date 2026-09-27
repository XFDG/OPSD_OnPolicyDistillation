# 北京 4×H200 TIP 训练结果（2026-09-27 核验）

## 结论

- **训练与评测成功完成**：逻辑训练链为 1–1739 步，每步 128 条 rollout；最终 checkpoint、35 次评测均齐全。
- **启动脚本全流程不是成功退出**：训练 Python 已正常返回、GPU 日志打印 `resume training PASS`，随后清理阶段的 overlay 增长检查返回 **exit=5**。不能将其掩盖为 exit=0。
- **未全面复现论文结果**：本轮使用原始 Qwen3-8B 教师，而论文使用 GRPO 教师；单次运行、4 卡及其他未明确设置也限制严格可比性。
- GPU 训练进程已退出；2026-09-27 00:07 UTC 实查保活正常。此次整理没有重启训练或评测。

## 最终结果及历史最佳

主指标为 mean@16，即每题 16 次采样的平均准确率。下面历史最佳来自不同 checkpoint，不能拼成一个模型的结果。

| 基准 | 题数 | 每次回答数 | 最终 step 1739 | 本轮该项最高 | 对应 step |
|---|---:|---:|---:|---:|---:|
| MATH-500 | 500 | 8000 | 79.0375% | 79.5250% | 1300 |
| AIME 2024 | 30 | 480 | 13.1250% | 15.2083% | 1050 |
| AIME 2025 | 30 | 480 | 19.1667% | 22.2917% | 700 |

三项等权宏平均：最终 **37.1097%**；本轮最高宏平均为 **step 100 的 38.4667%**（79.15% / 14.1667% / 22.0833%）。
这是对报告基准上的事后 checkpoint 选择，有选择偏差；最终 checkpoint 仍按预定训练结束点报告。
没有 step 0 或教师的独立评测，不能用这些结果证明相对未训练学生的收益。

### 与上游报告的关系

下表上游数值来自基线提交 `39b3b0a` 的 [README](../../README.md) 中 Qwen3 Soft-OR 50% 行。仅供定位差距，不是严格同条件比较。

| 基准 | 上游 Soft-OR 50% | 我们最终 | 差值（百分点） |
|---|---:|---:|---:|
| MATH-500 | 79.1% | 79.0375% | -0.0625 |
| AIME 2024 | 25.7% | 13.1250% | -12.5750 |
| AIME 2025 | 21.9% | 19.1667% | -2.7333 |

教师权重不同是已知实验差异，但没有对照实验，不能将全部差距归因于教师。
AIME 每套只有 30 题，单次采样波动也需要考虑。训练 loss 正常不等于基准准确率必然提高。

## 实验配置

| 项目 | 实际设置 |
|---|---|
| 学生 / 教师 | 原始 Qwen/Qwen3-4B / Qwen/Qwen3-8B；教师冻结，无 GRPO 权重 |
| 模型 revision | 学生 `1cfa9a7208912126459214e8b04321603b3df60c`；教师 `b968826d9c46dd6066d109eabc6255188de91218` |
| 硬件 | 4×H200；rollout TP=2，4 卡 FSDP，sequence parallel=1 |
| 方法 | 学生 on-policy rollout；Reverse KL；全局 batch 熵裁剪 98th percentile、min-max、Soft-OR；每回答选取 top 50% token |
| Loss 归约 | 全局选中 token 的均值，补偿微批次与 FSDP 梯度平均 |
| Batch | 8 prompts × 16 samples = 128 responses/update；微批次每卡 1 |
| 采样 | T=1、top_p=1、top_k=-1；thinking=False |
| 长度 | prompt=2048，response=8192，OPD max=16384，KL chunk=512 |
| 优化器 | AdamW；lr=1e-6；cosine；warmup=0；weight decay=0.1；clip=1 |
| 数据 | DAPO-Math-17k-dedup，seed=42，80/20 分割；13,918 训练问题 |
| 训练预算 | 1 epoch，1739 更新；实际消耗 13,912 prompts，尾部不足一批的 6 条未使用 |
| 保存 / 评测 | 每 50 步和最终 1739；MATH-500 + AIME24 + AIME25，mean@16 |
| 实现 | BF16 模型前向、FP32 loss；教师 logits CPU cache；参数/优化器 offload；额外学生打分前向 |
| Overlap | 正式训练未启用实验性 transfer overlap，无旧策略跨步预生成 |
| 环境 | Python 3.12，torch 2.9.1+cu128，transformers 4.57.1，SGLang 0.5.6，FlashAttention 2.8.3，flashinfer 0.5.3 |
| Verl | `0ddd28933f2d06fbf06d2d4b2cec7da547d596fc` |

完整数据来源、hash 与 AIME24 答案键修正见 [PROVENANCE](../../data/PROVENANCE.md)。论文 batch=8 的口径、数学实验轮数/thinking/具体 split 等没有被完全确认；这些是记录明确的实现选择。

## 训练链与核验

1. 原始生产运行 `tip-20260924T153731Z-542930` 的 **1–50** 步。
2. 用户手动启动 `old.sh` 的运行 `tip-20260924T174814Z-1211426`，恢复模型、Adam、scheduler、RNG、dataloader，完成 **51–1739** 步。
3. 已放弃的原运行 51–53 步、smoke 和 overlap 试验均不计入这条训练链。
4. 共有 35 份生产评测：原运行 step 50，加上恢复运行的 34 份；每份 8960 条，合计 **313,600** 条。
5. 从 JSONL 的 `acc` 字段重算三项 mean@16，全部与原始 worker 日志一致；不是重新执行模型或重新判题。
6. 最终四卡 model/optim/extra_state 文件非空；四卡 scheduler step=1739、next_lr=0，均保存 RNG；data cursor=1739，已读取 13,912 prompts。
7. 本次没有重新载入最终权重做推理，checkpoint 核验范围是文件完整性清单及可读的调度器/RNG/数据状态。

## 实测耗时

| 项目 | 数值 |
|---|---:|
| 正常步骤平均 / 中位数 | 105.58 / 104.97 秒 |
| 正常步骤 rollout 平均 | 27.88 秒 |
| 更新和权重同步平均 | 77.63 秒 |
| 保留训练链 step 时间合计（含保存和评测） | 54.14 小时 |
| 恢复运行 step 时间合计 | 52.55 小时 |
| 日志中 PyTorch allocated 峰值 | 29.91 GiB |

上述 step 时间不含重新加载模型、准备环境及中断期间；PyTorch allocated 指标不代表整个 SGLang/FSDP 作业的总显存峰值。

## 已知运行问题

- **日志转发滞后**：主日志曾停在 step 200，实际 Ray worker 继续训练。因此本报告使用完整 worker 日志核对 51–1739 连续步号。
- **收尾 exit=5**：`/root` 增长 77,902,806 bytes，`/tmp` 增长 109,314,414 bytes，超过脚本 64 MiB 门槛。检查统计的是整个目录，未排除其他进程贡献，不能直接归因于训练。训练完成及保活恢复在此检查之前已确认。
- **Overlap 未采用**：首个 paired update token 选择一致，但权重/梯度/Adam 未逐位一致；学生阶段 58.497→43.714 秒只是未获验收的局部计时。差异大小、FSDP 状态恢复影响和串行自身非确定性尚未隔离，不能宣称精度下降或可安全加速。
- **启动器版本**：提交的 `old.sh` 是整理时现有版本，已带最新完整 checkpoint 发现和原始日志显示辅助；不声称它与本轮启动时的 shell 字节完全一致。训练源码见随附 hash 清单。

## 产物与复核入口

- [summary.json](summary.json)：最终指标、逐项最高值、性能和限制。
- [eval_curve.csv](eval_curve.csv)：完整 35 次评测曲线。
- [training_metrics.csv](training_metrics.csv)：1739 步数值指标。
- [checkpoint_inventory.json](checkpoint_inventory.json)、[checkpoint_state_verification.json](checkpoint_state_verification.json)：最终 checkpoint 清单与状态核验。
- [evaluation_evidence.json](evaluation_evidence.json)：本地评测文件路径、大小、SHA-256；不上传原始问题、回答或模型权重。
- [source_manifest.json](source_manifest.json)：整理时源码和原始日志的 SHA-256。

最终本地 checkpoint：

```text
/volume/pt-train/users/zhaoye/OPSD-runtime/runs/tip-20260924T174814Z-1211426/train-output/Qwen3-4B-beijing-tip-full-TIP-rho0.5-reverse_kl-teacher-Qwen3-8B-nothink-lr1e-6-bs8-n16/global_step_1739
```

复算命令（CPU，只读原始产物）：

```bash
python3 scripts/summarize_h200_run.py \
  --original-run /volume/pt-train/users/zhaoye/OPSD-runtime/runs/tip-20260924T153731Z-542930 \
  --resumed-run /volume/pt-train/users/zhaoye/OPSD-runtime/runs/tip-20260924T174814Z-1211426 \
  --ray-log /volume/pt-train/users/zhaoye/ray/ray/session_2026-09-25_01-48-46_484378_1212905/logs/worker-af8f1e7f571a5aea477ecf937f6ef46c30d9692020ca3151ac867b58-01000000-1218923.out \
  --output-dir results/beijing-h200-20260927
```

训练已经完成，**不要为查看结果再次启动 `old.sh`**。训练启动始终由用户手动执行。本分支保留北京集群专用的绝对路径、CPU/GPU 分工、外部既有保活程序及离线环境依赖；不是任意机器直接运行的通用部署包。
