# B200 优化：关闭 profile 的六步稳定性与容量对照

**结果快照：候选六步稳定性、真实Qwen最长序列容量、step6→7/8恢复与最终源码/配置验收均通过；优化全量生产尚未启动。**

## 1. 比较设置与基线审计

两组均为同一台 8×B200、原始 13918 行训练数据、global128 rollout、TP2、microbatch1、BF16 前向/FP32 loss、global TIP Soft-OR50% reverse KL。均关闭阶段 profile 同步计时，cosine horizon 保持 1739 步，仅在六步提前停止；step2/4/6 存八 rank checkpoint，结束时输出 32 条独立小样本验证。正式生产保存/评估仍为每 50 步。

基线使用 CPU logits cache 和原 offload；候选使用 CUDA logits cache、实际 reference FSDP 参数驻留以及 actor/optimizer 驻留。模型、loss、采样策略和 on-policy 权重更新顺序保留。

**基线原 launcher/GPU wrapper exit=1，不记作 exit=0。** 原因是训练后观察器断言一条默认日志级别未输出的 INFO 完成文字，触发了验收断言。训练 shell 本身成功退出，六步指标、step2/4/6 完整八 rank 状态和 32 条验证输出均在 CPU 事后审计中通过，审计 exit=0；25 个执行源 SHA256 与原执行记录完全一致，原 driver/GPU/training/Ray/profile 日志 hashes 和原 exit=1 均保留。详见 [基线复核](baseline-reverification.json) 与 [基线指标/状态验证](baseline-verification.json)。这次复核没有重跑 GPU、写 checkpoint 或更改保活。

候选正常完成六步，launcher/GPU wrapper exit=0，结束后保活恢复 PID3155502。详见 [候选稳定性验证](stability-verification.json)。

## 2. 六步逐项数据

`更新+同步` 为 trainer `timing/train_s`，包含 actor update 和 rollout 权重同步；`整步` 还可能包含保存和结束验证。allocated 是每次 update 重置后的最坏 worker rank PyTorch allocated 峰值；reserved 是日志 metric 的 rank 均值，二者均不等同于整卡显存。

| 方案 / step | 响应 tokens | 生成 s | 更新+同步 s | 整步 s | allocated max GiB | reserved mean GiB | 附加工作 |
|---|---:|---:|---:|---:|---:|---:|---|
| 基线 / 1 | 215,350 | 42.138 | 81.320 | 123.529 | 14.998 | 16.514 | 冷启动 |
| 基线 / 2 | 257,937 | 28.659 | 72.233 | 110.631 | 19.547 | 19.491 | checkpoint |
| 基线 / 3 | 286,057 | 29.854 | 62.063 | 91.993 | 19.690 | 19.888 | 普通步 |
| 基线 / 4 | 338,004 | 40.791 | 86.249 | 136.482 | 22.144 | 21.526 | checkpoint |
| 基线 / 5 | 151,739 | 23.094 | 65.704 | 88.843 | 17.973 | 17.333 | 普通步 |
| 基线 / 6 | 239,653 | 25.083 | 61.440 | 105.860 | 18.313 | 18.601 | checkpoint、32 输出验证 |
| 候选 / 1 | 232,730 | 44.892 | 39.365 | 84.335 | 36.629 | 29.511 | 冷启动 |
| 候选 / 2 | 242,546 | 26.699 | 36.519 | 73.583 | 37.996 | 32.623 | checkpoint |
| 候选 / 3 | 295,770 | 35.100 | 36.938 | 72.119 | 39.953 | 35.170 | 普通步 |
| 候选 / 4 | 317,603 | 32.688 | 39.533 | 82.322 | 43.177 | 36.269 | checkpoint |
| 候选 / 5 | 163,825 | 27.490 | 33.866 | 61.406 | 38.940 | 27.990 | 普通步 |
| 候选 / 6 | 250,163 | 29.783 | 36.074 | 84.684 | 38.868 | 32.921 | checkpoint、32 输出验证 |

## 3. 分开统计更新、普通步和有保存/验证的总时间

| 样本范围 / 方案 | 响应 tokens 均值 | 生成均值 s | 更新+同步均值 s | 整步均值 s | 整步总和 s |
|---|---:|---:|---:|---:|---:|
| step2–6 / 基线 | 254,678.0 | 29.496 | 69.538 | 106.762 | 533.809 |
| step2–6 / 候选 | 253,981.4 | 30.352 | 36.586 | 74.823 | 374.113 |
| 普通 step3/5 / 基线 | 218,898.0 | 26.474 | 63.883 | 90.418 | 180.836 |
| 普通 step3/5 / 候选 | 229,797.5 | 31.295 | 35.402 | 66.762 | 133.525 |
| 全部六步（含保存/验证） / 基线 | 248,123.3 | 31.603 | 71.502 | 109.556 | 657.337 |
| 全部六步（含保存/验证） / 候选 | 250,439.5 | 32.775 | 37.049 | 76.408 | 458.448 |

- 初始化后的五次更新+同步：**69.538 → 36.586 s**，时间减少 **47.39%**（观测速度比约 1.90×）。
- 不含冷启动、保存和验证的两条普通 step：**90.418 → 66.762 s**，时间减少 **26.16%**（观测速度比约 1.35×）。
- step2–6 平均响应 tokens：**254,678.0 / 253,981.4**，总量相近；生成轨迹和各 step 的长度分布仍不同。
- 全部六步的总时长包含三次 checkpoint 和 32 输出验证，保存密度远高于正式生产的每50步；单独报告，不将该总时长直接等比例外推。

普通步候选生成阶段并未在这个小样本中明显变快；主要收益来自 update/offload 路径。六步实测支持候选继续验收，尚不能确认 1739 步所有未来轨迹的速度或完整 benchmark 精度。

## 4. 真实 Qwen 最长序列容量 gate

[长序列验证 JSON](long-sequence-verification.json) 使用实际 Qwen3-4B/Qwen3-8B、八 rank 的生产 worker、共置 SGLang sleep/wake 和完整 vocab 的 TIP scoring/reverse-KL/backward/Adam 路径。先通过短序列更新建立 Adam 状态，再测试合成最大长度输入；这是容量 gate，输入并非模型真实生成的自然语言 trajectory。

| 指标 | 实测 |
|---|---:|
| 全局序列 / 每序列 response | 128 / 8192 token |
| prompt / 实际总长 / padded 长度 | 2048 / 10240 / 16384 token |
| teacher logits cache / rank | 37.09375 GiB |
| 最大长度 worker update | 53.5346 s |
| 最坏 worker allocated / reserved 峰值 | 63.4298 / 67.7676 GiB |
| NVML 采样整卡最大 used | 119.8564 GiB |
| NVML 采样整卡最小 free | 58.4980 GiB |
| NVML 采样周期 | 1000 ms |
| launcher / GPU wrapper | exit=0 / exit=0 |
| 退出保活 | 已恢复 |

NVML 数值覆盖共置 SGLang 等进程；它是每秒采样的 extrema，可能遗漏瞬时峰值。结果证明该真实模型路径在已测合成最大 response 下完成更新并留有容量余量，不能证明所有内容分布的精确最坏峰值、完整 rollout 质量或长期训练稳定性。

## 5. 全量时间粗估

尚无优化候选全量训练时长。用本次普通步与 [历史 B200 基线统计](../taihua-b200-20260930/summary.json) 做两种粗估：

| 估计方法 | 计算 | 约时长 |
|---|---|---:|
| 当前普通步 + 历史保存/评估额外时间 | 66.762×1739/3600 = 32.25 h；再加历史额外约 2.20 h | 34.4 h |
| 按受控普通步比缩放历史总 step 时间 | 51.56 h × 0.7384 | 38.1 h |

可作为排期的粗略范围是 **34–39 小时**。普通步只有 step3/5 两条样本；未来随机生成、长度/准确率变化、完整 MATH/AIME 验证时间、I/O 和启动/清理都可能改变估计，因此不承诺 ETA，也不声称全量精度保持。

## 6. 硬件条件与剩余验收

最终复查仍显示 GPU2 全部 NVLink inactive；GPU2 与其他卡为 NODE/SYS，其他七卡为 NV18。原始证据为 [NVLink 状态](final-gpu2-nvlink.txt) 和 [拓扑](final-topology.txt)。没有 GPU/NVSwitch reset；当前软件速度结果依赖这台机器当时的互联状态，不能证明历史全量一直处于相同状态或把全部性能差距归因于此。

- step6→7/8 的完整 model/optimizer/scheduler/RNG/dataloader 恢复：**已通过**，八rank均加载，data cursor48→56→64、cosine horizon1739；更新39.846/33.620秒，wrapper/launcher exit=0，保活恢复。见 [恢复报告](stability-resume-verification.json)。
- 生产验收 stamp：**已通过**，绑定37份源码和四类报告，拒绝TP1/失败stamp/源码变更；见 [最终CPU验收](acceptance-verification.json)。
- 优化候选正式全量：**尚未启动**；完成验收后由用户手动执行总脚本。

逐步精确值、汇总公式、来源 SHA256 和状态字段见 [stability-comparison.json](stability-comparison.json)。本文件记录已完成有界证据，完整训练时长和准确率仍待生产运行。
