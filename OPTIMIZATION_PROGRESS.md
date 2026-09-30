# B200 优化实验进度

最新状态：**B200 有界优化验收通过，总脚本已部署，优化全量尚未启动。** 用户已明确目标为B200。以下历史暂停记录保留，当前实验以新增记录为准。机器日志日期为2026-09-30 UTC，用户工作区日期为2026-10-01。

## 继续后的进展

- 完整 CPU 缓存基线两步 smoke 已通过：`tip-20260930T183746Z-2887574`，启动器 exit=0，保活恢复。
- CUDA 缓存候选两步 smoke 已通过：`tip-20260930T184911Z-2924752`，启动器 exit=0，保活恢复。
- CUDA 缓存 + 教师/学生参数/优化器 GPU 驻留候选两步 smoke 已通过：`tip-20260930T185936Z-2959773`。两次更新与权重同步分别 37.90 / 37.61 秒；对应基线 76.26 / 87.63 秒。这是带同步 profiling 的短测试，尚不能当作正式全量速度。
- 驻留实验的八卡 checkpoint、32 条验证输出、全局128 rollout、TIP Soft-OR 与 finite gradient 检查均通过；启动器 exit=0，保活恢复 PID `2993456`，`/root`、`/tmp` 增长均为0。
- 新增两卡实际更新等价性检查已通过：CPU/CUDA logits 缓存的统计、选择、loss、梯度、FP32权重、AdamW状态、scheduler 相等；覆盖大于512的chunk和整卡零选择。检查使用合成 FSDP Embedding，真实 Qwen smoke 提供另外的集成证据。
- 修复原 scheduler 根据本卡 token 数决定是否推进的问题：改为依据全局选中 token 数，避免整卡零选择时各 rank 学习率分歧。
- **TP1 rollout** 两步 smoke 已通过（profiling 关闭），但平均生成时间与 TP2 接近，证据不足以切换；验收候选继续保留 TP2。
- 新总启动器 `full.sh` → `run_b200_optimized_full.sh` 已通过有界验收：实时阶段/每10秒进度、共享锁、自动停启保活、生产每50步保存/评估；正式 all/train/resume 要求37份配置/源码与证据验收记录匹配。默认无参数仅prepare。
- 关闭 profiling 的完整数据基线已完成六步：`tip-20260930T192959Z-3030419`。step2–6 更新+同步均值 69.538 秒；普通 step3/5 整步均值 90.418 秒。原包装器 exit=1 的唯一原因是核验脚本要求一条被默认日志级别过滤的 INFO 结束文字。修复观察器后仅做 CPU 复验，六步/三组八卡 checkpoint/32条验证均通过，原日志/profile/exit=1 未改动；25份实际执行源码 SHA256 完全相同，保活持续运行。见 [CPU复验](results/optimization-20261001/baseline-reverification.json)。
- 真实最长序列容量已通过：`tip-20260930T200309Z-3081555`。先分配 AdamW 状态，再做128条 × 8192 response token 更新（prompt2048、padded16384、实际10240），teacher cache37.094GiB/rank；实际更新53.535秒、权重同步与SGLang唤醒成功。最坏卡 worker allocated63.430GiB/reserved67.768GiB，NVML每秒采样最大整卡占用119.856GiB、最小空闲58.498GiB；NVML是采样极值。启动器/GPU wrapper exit=0、保活恢复，root/tmp增长0。
- 驻留候选六步稳定性已通过：`tip-20260930T201230Z-3113178`，原13918行数据、1739步cosine、TP2、micro1、profileFalse；step2–6更新+同步均值36.586秒，较CPU基线减少47.39%。普通step3/5整步均值66.762秒，减少26.16%；这是短样本速度。
- 从该候选step6恢复到step7/8已通过：`tip-20260930T203205Z-3156633`；八rank完整model/optimizer/RNG/scheduler恢复，data cursor48→56→64，更新39.846/33.620秒，checkpoint7/8和32验证输出通过。GPU wrapper/launcher均exit=0，root/tmp增长0。
- 最终CPU验收于2026-09-30T20:42:02Z通过，绑定37份源码和四类报告；错误TP1、失败stamp和源码变更均被拒绝。GPU2 NVLink复查仍inactive；性能结论依赖当前硬件状态。
- 八卡保活已恢复父PID3189967，最终NVML只见8个已知保活worker；优化全量未启动。估计34–39小时，仅作短样本排期参考。

详见 [完整优化设置和启动说明](OPTIMIZATION_RESULTS.md)、[profile-off稳定性对照](results/optimization-20261001/stability-comparison.md)、[最终验收](results/optimization-20261001/acceptance-verification.json)。此记录与代码/证据一同交付至 `experiments/taihua-b200-optimization` 分支。

手动总启动命令（Taihua GPU或CPU Pod）：

```bash
bash /volume/pt-test/users/zhaoye/OPSD_B200_Optimize/full.sh all
```

另开终端持续观察：

```bash
bash /volume/pt-test/users/zhaoye/OPSD_B200_Optimize/full.sh follow
```

## 历史暂停记录（2026-09-30 14:54 UTC）

### 暂停时机器状态

- 优化训练进程已停止，GPU wrapper 与启动器 exit=143，原因是用户要求主动停止（SIGTERM）。不计作 smoke 验收成功。
- 8卡保活已恢复，父进程 PID `2854278`；控制器核实8个GPU worker均运行，未发现其他GPU作业。
- 收尾 `/root`、`/tmp` 增长均为0。
- 已完成的1739步B200正式训练、H200结果和checkpoint没有被覆盖。

### 暂停时分支与位置

- 分支：`experiments/taihua-b200-optimization`，基于已完成B200结果提交 `2d54d06`。
- 北京工作树：`/volume/pt-train/users/zhaoye/OPSD_B200_Optimize`。
- 太华部署：`/volume/pt-test/users/zhaoye/OPSD_B200_Optimize`。
- 独立实验运行目录：`/volume/pt-test/users/zhaoye/OPSD-B200-opt-runtime`。
- 当前基线：`runs/tip-20260930T144153Z-2815368`。
- 本轮代码为工作树中的实验草稿，尚未提交/推送此优化分支；不可当作已验收生产版本。

### 暂停前已完成工作

1. 建立独立优化工作树、实验启动器 `run_b200_opt.sh`，与原任务共用GPU作业锁。实验启动器禁用全量训练/生产恢复入口。
2. 增加阶段计时：教师加载、教师前向与缓存、模型切换、TIP打分、全局选择、学生反向、优化器、最终offload。
3. 增加可选 `opd_teacher_cache_device=cpu|cuda`；默认CPU。候选只改变教师logits保存位置，保持精度、loss、TIP规则、rollout和优化器设置。
4. 8卡FA2前向/反向及FA4 prefill/decode/CUDA graph检查通过。
5. CPU/GPU教师缓存等价性检查通过：完整151936词表、BF16、实际TIP更新函数、AdamW，token统计、选择、loss、梯度、权重、AdamW状态逐元素相等。该测试使用单GPU合成logits/Embedding，不能代替真实模型多卡或完整准确率验证。
6. 带计时的CPU缓存基线已执行至第2次更新与checkpoint保存/验证输出阶段；主动停止前没有完成最终smoke验收。已提取的完整逐步指标仅step1，不宣称step2完成全部核验。
7. **GPU缓存候选端到端实验尚未启动，无候选速度数据，无优化收益结论。**

### 基线 step1 的阶段计时

这些profile数据为8个rank聚合均值；边界加入CUDA同步。仅一条已留存完整指标，包含冷启动影响，不是稳态性能。子阶段存在包含关系，不能把所有行相加。

| 阶段 | 秒 |
|---|---:|
| rollout（trainer计时） | 48.89 |
| 教师加载及此前准备 | 9.36 |
| 教师前向与缓存 | 19.68 |
| 其中CPU缓存拷贝的host wall时间 | 5.19 |
| 教师offload/学生与优化器load等模型切换 | 0.29 |
| TIP学生打分 | 9.83 |
| TIP全局选择 | 0.47 |
| 学生前向反向 | 25.68 |
| clip/optimizer/all-reduce | 0.39 |
| 最终offload及收尾 | 3.16 |
| worker update总计 | 68.88 |
| trainer更新与权重同步 | 74.28 |
| 整步，含checkpoint保存 | 133.50 |

每rank教师logits缓存平均约7.69GiB。搬运存在可优化成本，但尚不能把任务定性为纯GPU显存带宽受限。

### 硬件线索：GPU2的NVLink异常

2026-09-30本次实查：

- `nvidia-smi topo -m`：GPU2与其他GPU之间为NODE/SYS，其他7张卡互联为NV18；初始化过程中复查结果一致。
- `nvidia-smi nvlink --status -i 2`：`NVML: Unable to retrieve Nvlink information as all links are inActive`。
- GPU0对照：18条链路均报告53.125GB/s。
- 完整 `nvidia-smi -q -i 2` 查询长时间未返回，已仅终止本次创建的诊断进程；保留的诊断文本是不完整输出。
- 未重置GPU、未修改驱动/NVSwitch。GPU2互联需要平台侧进一步确认/修复；没有证据证明此前全量运行始终处于相同状态，也不能直接将全部性能差距归因于此。

### 当时规划的继续顺序（历史记录）

1. 先复查GPU2/NVLink健康状态，记录是否与本次相同。硬件恢复前后的性能应分开报告。
2. 完成并重复CPU缓存基线，获得多步计时、显存峰值与完整退出证据。
3. 运行GPU缓存候选，验证两步更新、checkpoint、评估与保活恢复；检查最坏序列长度的显存余量。
4. 同口径比较profile、序列/token工作量、普通步耗时，区分profiling同步开销及随机生成差异。
5. 再依据计时决定是否测试减少参数/优化器offload、增大微批次等；这些目前仅是待评估方向，没有实施或验证。
6. 只有验证后再考虑生产入口和全量精度实验；当前没有可宣称“充分释放B200潜力”的结论。

### 历史证据文件

- [缓存等价性检查](results/optimization-20260930/cache-equivalence.json)
- [基线已记录指标](results/optimization-20260930/partial-baseline-metrics.json)
- [暂停与保活恢复](results/optimization-20260930/stop-and-keepalive.log)
- [初始拓扑](results/optimization-20260930/opt-topology.txt)
- [初始化期间拓扑](results/optimization-20260930/opt-topology-during-init.txt)
- [NVLink查询与部分诊断](results/optimization-20260930/opt-gpu2-diagnostic.txt)

完整日志在上述太华运行目录，以及北京 `OPSD-B200-migration/opt-baseline-console.log`。
