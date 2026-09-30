# 太华 8×B200 TIP 全量训练结果（2026-09-30）

## 结论

**训练、评估和启动脚本收尾均成功，exit=0。** 独立完整运行 1–1739 步，每步128条 rollout，共222,592条。35次评估共313,600条输出，从原始 JSONL 的 `acc` 重算后与每次 worker 日志一致。此次整理仅执行 CPU 读取和核验，没有重新训练、推理或判题。

最终8卡 checkpoint 文件齐全；所有 rank 的 scheduler step=1739、next_lr=0、RNG存在，数据游标1739、已读取13,912条 prompts。检查未重新加载模型权重执行推理，不代表对权重内容做了完整数值校验。保活已恢复；收尾记录 `/root` 增长31,823 bytes、`/tmp` 增长0。

## 最终准确率与历史最佳

主指标为 mean@16，每题16次采样的平均准确率。

| 基准 | 题数 | 最终 step1739 | 该项历史最佳 | 最佳 step |
|---|---:|---:|---:|---:|
| MATH-500 | 500 | 79.5375% | 80.1125% | 1050 |
| AIME 2024 | 30 | 14.1667% | 15.6250% | 550 |
| AIME 2025 | 30 | 17.9167% | 22.9167% | 1650 |

最终三项等权宏平均 **37.2069%**。最高宏平均为 step1450 的 **38.5681%**，对应 MATH79.6625%、AIME24 13.5417%、AIME25 22.5000%。逐项最佳不能拼成一个 checkpoint 的成绩；历史最佳是基于报告基准的事后选择，有选择偏差。

## 配置与复现边界

- 原始 Qwen3-4B 学生、原始 Qwen3-8B 教师；模型 revision 和数据文件与 H200 相同。
- 8×B200、8个FSDP rank；rollout TP=2、4个副本；8 prompts×16 samples，微批次每卡1。
- BF16前向、FP32 loss；Reverse KL、全局98%熵裁剪/min-max、Soft-OR50%、全局选中token归一化。
- T=1、top_p=1、top_k=-1、thinking=False；prompt2048、response8192、OPD16384、chunk512。
- AdamW lr1e-6、cosine、warmup0、weight decay0.1、clip1；DAPO 13,918题，1epoch1739步，末尾不足batch的6题未使用。
- 每50步及最终保存和评估；35套checkpoint的8个rank的model/optim/extra_state文件均非空。
- 训练 attention 保留FA2；rollout为官方FA4 commit `e9cf2c1651d2303191eb40a739a3c135fda00999`。SGLang0.5.6带已记录的FA4兼容补丁；KV页128；无实验性overlap、无FP8。
- 详细配置、环境版本、源码与适配范围见 [迁移报告](../../B200_MIGRATION.md)。整理时远端部署的18个代码/配置文件均匹配 smoke 时留存的SHA256；这不单独证明整个历史运行期间从未有人改动环境。

教师仍不是论文的GRPO教师；没有step0/教师独立评估，只有单次运行。因此可以确认实现运行成功，不能宣称完整复现论文准确率，也不能证明相对原始学生的训练收益。

## 耗时

| 项目 | 实测 |
|---|---:|
| 普通步平均 / 中位数 | 102.20 / 102.29 秒 |
| 普通步 rollout 平均 | 30.85 秒 |
| 普通步更新及权重同步平均 | 71.28 秒 |
| 全部 step 时间合计（含保存、评估） | 51.56 小时 |
| 日志报告的 PyTorch allocated 峰值 | 22.22 GiB |

普通步排除每50步及最终评估步骤，与H200统计口径一致。step合计不含环境准备、smoke、加载模型；PyTorch allocated 不代表整个作业显存峰值。详细对照见 [B200与H200比较](B200_VS_H200.md)。

## 证据与产物

- [summary.json](summary.json)：最终结果、最佳checkpoint、性能汇总。
- [eval_curve.csv](eval_curve.csv)：完整35点评估曲线。
- [training_metrics.csv](training_metrics.csv)：完整1739步训练指标。
- [checkpoint_inventory.json](checkpoint_inventory.json)、[checkpoint_state_verification.json](checkpoint_state_verification.json)：最终checkpoint文件和状态核验。
- [evaluation_evidence.json](evaluation_evidence.json)：35份原始评估文件的路径、大小和SHA256。
- [source_manifest.json](source_manifest.json)：整理时源码和原始日志SHA256；[completion-evidence.log](completion-evidence.log)：收尾成功证据。

最终 checkpoint：

```text
/volume/pt-test/users/zhaoye/OPSD-B200-runtime/runs/tip-20260928T061500Z-1228422/train-output/Qwen3-4B-b200-tip-full-TIP-rho0.5-reverse_kl-teacher-Qwen3-8B-nothink-lr1e-6-bs8-n16/global_step_1739
```

只读复算命令（太华GPU节点，保持CPU运算）：

```bash
CUDA_VISIBLE_DEVICES="" /volume/pt-test/users/zhaoye/envs/opsd-py312-cu128/bin/python -B \
  /volume/pt-test/users/zhaoye/OPSD_B200/scripts/summarize_b200_run.py \
  --run /volume/pt-test/users/zhaoye/OPSD-B200-runtime/runs/tip-20260928T061500Z-1228422 \
  --ray-log /volume/pt-test/users/zhaoye/bray/ray/session_2026-09-28_14-26-32_336070_1415963/logs/worker-ae07eba7e99dc5a28983b3d5dc63f6e4213bb5c7542e3f0bb9974301-01000000-1433280.out \
  --output-dir /volume/pt-test/users/zhaoye/OPSD_B200/results/taihua-b200-20260930
```

GitHub仅提交小型汇总与审计文件，不上传模型权重、原始题目或生成文本。
