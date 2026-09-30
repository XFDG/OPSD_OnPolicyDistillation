# B200 优化：三组两步 smoke 对照

目录标签为 20261001；机器记录时间为 2026-09-30 UTC。三组均完成两次真实更新、每步 128 个 rollout、step 1/2 的八 rank checkpoint 和 32 条小样本验证，GPU wrapper 与总 launcher 均 exit=0，结束后恢复保活。重复 Ray 输出已按 step 去重；完整数值及日志 SHA256 保存在 [profile-comparison.json](profile-comparison.json)。

## 配置与验证范围

| 设置 | 基线 | GPU 缓存 | GPU 缓存 + 驻留 |
|---|---|---|---|
| 教师 logits 缓存 | CPU | CUDA | CUDA |
| actor 参数/optimizer offload | 开启 | 开启 | 关闭 |
| reference 内部 FSDP CPU offload | 开启 | 开启 | 关闭，guarded resident builder |
| rollout TP / microbatch | 2 / 1 | 2 / 1 | 2 / 1 |
| profile 同步计时 | 开启 | 开启 | 开启 |
| 模型、前向/损失精度、TIP、全局 rollout | Qwen3-8B→4B；BF16/FP32；global Soft-OR50% + reverse KL；128 | 相同 | 相同 |

这组 smoke 使用 16 行训练数据、2 行验证 prompt 和两步 cosine 调度；每步存 checkpoint，step 2 做小样本验证。它验证初始化、更新、保存、验证和退出路径，不代表完整 1739 步训练或完整 benchmark。

## 逐步观测

| 方案/步 | 响应 tokens | 生成 s | 更新+权重同步 s | 整步 s | worker update s | 教师 forward+cache s | TIP scoring s | 学生 forward/backward s | 日志 allocated 峰值 GiB¹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU 缓存 + offload / 1 | 232,831 | 43.332 | 76.264 | 129.578 | 70.386 | 20.963 | 9.653 | 26.965 | 14.525 |
| CPU 缓存 + offload / 2 | 239,236 | 38.624 | 87.626 | 148.790 | 75.796 | 25.956 | 16.008 | 30.362 | 22.147 |
| GPU 缓存 + offload / 1 | 232,407 | 55.156 | 50.851 | 117.364 | 45.450 | 10.495 | 4.972 | 17.026 | 30.312 |
| GPU 缓存 + offload / 2 | 248,236 | 39.454 | 42.334 | 104.323 | 36.157 | 8.730 | 5.295 | 19.422 | 41.945 |
| GPU 缓存 + 参数/优化器驻留 / 1 | 232,011 | 36.625 | 37.902 | 84.772 | 30.825 | 9.021 | 4.889 | 16.012 | 33.267 |
| GPU 缓存 + 参数/优化器驻留 / 2 | 242,776 | 39.840 | 37.610 | 97.566 | 30.826 | 8.100 | 5.360 | 16.932 | 46.157 |

¹ allocated 是各 worker rank 的最大值。前两组为进程累计峰值，驻留组在每次 update 开始重置统计，口径不同。驻留组最坏卡 update allocated 为 **33.267 / 46.157 GiB**，reserved 为 **37.945 / 50.961 GiB**；日志 `perf/update_peak_reserved_gib=29.099 / 32.455` 是 rank 均值，不能用于判断最坏卡。所有这些值只覆盖训练 worker 的 PyTorch allocator，不含独立 SGLang 进程和整卡全部显存。

## 两步均值及观测收益

| 方案 | 更新+同步 s | worker update s | 生成+更新同步 s | 整步 s² | 更新+同步相对基线减少 |
|---|---:|---:|---:|---:|---:|
| CPU 缓存 + offload | 81.945 | 73.091 | 122.923 | 139.184 | 0.00% |
| GPU 缓存 + offload | 46.593 | 40.804 | 93.897 | 110.843 | 43.14% |
| GPU 缓存 + 参数/优化器驻留 | 37.756 | 30.826 | 75.988 | 91.169 | 53.93% |

² 整步含保存，第二步还含验证，不能直接外推全量训练。

| 观测对照 | 更新+同步时间减少 | worker update 时间减少 | 整步时间减少 |
|---|---:|---:|---:|
| CPU 缓存 + offload → GPU 缓存 + offload | 43.14% | 44.17% | 20.36% |
| CPU 缓存 + offload → GPU 缓存 + 参数/优化器驻留 | 53.93% | 57.83% | 34.50% |
| GPU 缓存 + offload → GPU 缓存 + 参数/优化器驻留 | 18.97% | 24.45% | 17.75% |

GPU 缓存候选的教师缓存阶段和学生阶段均缩短；驻留候选进一步消除了第一步约 9 秒 teacher load 与每步约 2–3 秒 final offload。驻留组 teacher load=0.00494/0.00444 秒，final offload=0.00298/0.00229 秒。CUDA 缓存的 Python cache-copy 调用时间接近零；基线 `cache_copy_host_s` 包含前面异步教师算子的等待，不能解释为纯 D2H 带宽。

## 结论与边界

三组 smoke 均通过。GPU 缓存和模型/优化器驻留是有明确初步收益的候选；驻留组两步平均更新+权重同步从 **81.945 秒降到 37.756 秒**，观测减少 **53.93%**，相对只用 GPU 缓存再减少 **18.97%**。这些百分比描述本次两步 profile-on 测量。

- 三组独立随机生成，token 数和轨迹不同。第一步含冷启动，只有一个后续样本，不足以确认长期稳态速度。
- profile 在边界同步 CUDA，阶段时间是 rank 均值，不能逐项相加还原 trainer 的通信关键路径。需要关闭 profile 后做同数据/完整调度的多步稳定性测试。
- 两行 prompt 的 32 输出验证三组均通过且为 100%；这不能证明 MATH/AIME 完整评测精度保持。loss 和梯度也来自不同随机 trajectory，其数值差异不能直接判断数学不等价。
- CUDA 缓存增加显存占用。本次每 rank 平均 logits 缓存约 8.2–8.8 GiB；每 rank 16 条 response、每条 8192 token、vocab=151936、BF16 时，仅 logits 缓存理论上限约 37.1 GiB/rank，还需模型、激活、FP32 loss 临时量。当前两步不能保证最长序列安全。
- 还需验证原始 1739 步调度下的多步运行、最坏卡显存、长序列更新和完整状态 checkpoint 恢复，之后才能批准生产 full.sh。
- GPU2 的 NVLink 被诊断为 inactive，与其他卡拓扑为 NODE/SYS，其他七卡为 NV18。记录见 [OPTIMIZATION_PROGRESS.md](../../OPTIMIZATION_PROGRESS.md) 和 [GPU2 诊断](../optimization-20260930/opt-gpu2-diagnostic.txt)。未重置 GPU/驱动/NVSwitch；没有证据说明此前全量训练一直处于相同状态，也不能把所有性能差距归因于此。

## 运行记录

| 方案 | run ID | smoke 验证 |
|---|---|---|
| CPU 缓存 + offload | `tip-20260930T183746Z-2887574` | [JSON](baseline-smoke-verification.json) |
| GPU 缓存 + offload | `tip-20260930T184911Z-2924752` | [JSON](gpu-cache-smoke-verification.json) |
| GPU 缓存 + 参数/优化器驻留 | `tip-20260930T185936Z-2959773` | [JSON](resident-smoke-verification.json) |

原始日志位于工作区 `OPSD-B200-migration/opt-{baseline,gpu-cache,resident}-20261001.log`；远端 run 目录均为 `/volume/pt-test/users/zhaoye/OPSD-B200-opt-runtime/runs/<run ID>`。
