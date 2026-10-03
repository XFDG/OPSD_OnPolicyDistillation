# Taihua 8×B200 OPSD/TIP 优化实验

更新于2026-10-03。本页保留 `2026-09-30T20:42:02Z` 的有界验收证据，并补入已完成的优化全量结果。短测试和全量计时分别报告；[独立技术分享](TECHNICAL_SHARE_ON_POLICY_DISTILLATION.md)继续保留匿名、短测试的测量范围。

**当前状态：优化全量成功完成1739步、35次完整评测和35套checkpoint，GPU wrapper/launcher均exit=0；八卡保活已自动恢复并通过最新只读检查。**

完整`all`耗时**37.6353小时**，比原B200的51.8386小时少**27.3991% / 14.2033小时**；普通步72.7804秒，更新加权重同步41.4186秒。最终mean@16为**79.4125% / 13.7500% / 18.1250%**（MATH-500/AIME24/AIME25）；相对原B200为−0.1250/−0.4167/+0.2083个百分点，单次运行不能证明严格无损。详见 [全量审计](results/taihua-b200-optimized-20261003/RESULTS.md) 和 [新B200 / 原B200 / H200对照](results/taihua-b200-optimized-20261003/B200_OPT_VS_BASELINE_AND_H200.md)。

分支：`experiments/taihua-b200-optimization`。本实验从已完成的 B200 基线继续优化执行方式。此前三个配置的两步 profile smoke、关闭 profile 的候选六步和真实 Qwen 最长序列容量 gate 已通过；关闭 profile 的基线六步训练产物通过 CPU 事后审计，原观察器导致的 launcher/GPU wrapper exit=1 保留。step6→7/8 恢复 run `tip-20260930T203205Z-3156633` 已通过，八 rank model/optimizer/scheduler/RNG 与数据进度连续；37份源码及四类证据已通过最终验收。用户随后手动启动全量run `tip-20261001T003630Z-3211352`，于2026-10-02 14:14:37 UTC正常结束。

本记录的结果目录标签为 20261001，已完成 smoke 的服务器日志日期为 2026-09-30 UTC。以每个 run 的日志时间和 run ID 为审计依据。

## 1. 候选配置与保留的实验设置

选定候选为 **CUDA teacher logits cache + 教师/学生参数和 AdamW 状态驻留 GPU**。默认 rollout TP=2、每 GPU microbatch=1、profile=False。现阶段没有启用实验性 overlap、FP8 或改变采样策略。

| 项目 | 已完成 B200 基线 | 当前优化候选 | 对齐范围 |
|---|---|---|---|
| 机器 | Taihua 8×B200 | 同一台 8×B200 | 保持；当前硬件限制见后文 |
| 学生 | 原始 `Qwen/Qwen3-4B` | 相同 | 模型 revision 固定 |
| 教师 | 原始 `Qwen/Qwen3-8B` | 相同 | 模型 revision 固定；教师类型限制延续基线 |
| 前向 / loss 精度 | BF16 / FP32 | 相同 | 不使用 FP8 |
| 方法 | global TIP、Soft-OR 50%、reverse KL | 相同 | 分数、全局归一化、选 token 和 loss 公式保留 |
| 熵截断 quantile | 0.98 | 相同 | 保留 |
| 每步 prompt / 每 prompt rollout | 8 / 16 | 相同 | 全局每次更新 128 个 rollout |
| On-policy 时序 | 同步更新当前学生并同步 rollout 权重 | 相同 | 不使用陈旧权重或异步更新 |
| rollout TP / microbatch | 2 / 1 | 2 / 1 | 本次保留 |
| 学生 gradient checkpointing | 开启 | 开启 | 本次保留 |
| prompt / response / OPD 上限 | 2048 / 8192 / 16384 | 相同 | 最长序列须单独验证 |
| loss/scoring token chunk | 512 token | 相同 | 含跨 chunk 的正确性 gate |
| 训练数据 | 已迁移固定的 13918 行训练集 | 相同 | 使用资产 SHA256 校验 |
| epoch / 总调度步数 | 1 / 1739 | 相同 | 稳定性测试仅提前停止，保留完整 horizon |
| optimizer / LR | AdamW / 1e-6 | 相同 | 不改变算法或 LR |
| scheduler / warmup | cosine / 0 | 相同 | 完整 1739 步调度 |
| weight decay / grad clip | 0.1 / 1.0 | 相同 | 保留 |
| 采样 | temperature=1.0、top_p=1.0、top_k=-1 | 相同 | 训练和验证均保留 |
| thinking | False | 相同 | 原始模型非 thinking 模式 |
| 验证 | MATH-500/AIME24/AIME25、n=16 | 相同 | 生产完整验证；有界测试使用独立小样本验证 |
| 生产保存 / 评估频率 | 每 50 步，另存最终步 | 相同 | 原生产 checkpoint 保留 |
| 训练 / rollout attention | FA2 2.8.3 / 官方 FA4 | 相同 | 延续已批准的 B200 内核迁移 |
| rollout memory utilization | 0.6 | 相同 | 保留 |
| teacher logits 存放 | CPU | CUDA | 改存储位置，保留 tensor dtype 和值 |
| teacher 内部 FSDP CPU offload | 开启 | 关闭，实际驻留 | 使用有前置/结果检查的构造 helper |
| actor 参数 / optimizer offload | 开启 / 开启 | 关闭 / 关闭 | 减少模型与状态搬运 |
| profile 同步计时 | 正式基线关闭 | 正式候选关闭 | 两步诊断 smoke 中开启，不能外推为生产速度 |

重要 revision：学生 `1cfa9a7208912126459214e8b04321603b3df60c`，教师 `b968826d9c46dd6066d109eabc6255188de91218`，Verl `0ddd28933f2d06fbf06d2d4b2cec7da547d596fc`。迁移环境和官方 FA4 的来源、版本及测试见 [B200_MIGRATION.md](B200_MIGRATION.md)。

原始教师选择和此前记录的论文差异延续 [BEIJING_TIP_REPRO.md](BEIJING_TIP_REPRO.md)。保持本次模型方法和 rollout 设置，并不等于已经完成论文所有条件或所有 benchmark 精度的复现。

### 1.1 具体实现变化

- `src/opd/opd_worker.py`：teacher logits 留在 CUDA；一次 update 用完后释放缓存，随后恢复 rollout。profile 关闭时不执行阶段同步计时。
- `src/opd/resident_ref.py`：固定版本的 Verl 原实现会强制 reference FSDP 使用 CPU offload。候选通过 guarded builder 构造无 optimizer 的 reference，检查 FSDP 内部 CPU offload 已关闭、参数处于 CUDA、stored dtype 和 worker role 保留。没有修改 pinned Verl 源码。
- `scripts/opd/train_opd.sh`：参数化 actor/reference 参数和 actor optimizer 的 offload 开关，保留原默认行为；优化入口明确选择驻留。
- **scheduler 零 token rank 修复**：当某 rank 本地没有入选 token，但全局存在选中 token 并执行了 optimizer update 时，所有 rank 依据全局更新决定推进 scheduler。旧版本若使用本地 token 数作条件，会导致部分 rank LR 进度不一致。这是分布式边界情况的正确性修复；global selection、loss 和通常的非空 rank 更新公式没有改变。

## 2. 已完成的证据

### 2.1 三组两步 profile smoke

三组均完成两次真实模型更新、每步 128 个 rollout、八 rank 的 step 1/2 checkpoint、32 条小样本验证和退出清理，launcher/GPU wrapper 均 exit=0，结束后恢复保活。完整逐步数据和统计口径见 [profile 对照文档](results/optimization-20261001/profile-comparison.md) 与 [去重指标 JSON](results/optimization-20261001/profile-comparison.json)。

| 配置 | 两步平均更新+权重同步 | 两步平均 worker update | 两步平均整步 | 状态 |
|---|---:|---:|---:|---|
| CPU cache + offload | 81.945 s | 73.091 s | 139.184 s | smoke 通过 |
| GPU cache + offload | 46.593 s | 40.804 s | 110.843 s | smoke 通过 |
| GPU cache + 参数/optimizer 驻留 | 37.756 s | 30.826 s | 91.169 s | smoke 通过 |

这些两步 profile-on 观测中，GPU cache 的更新+同步时间相对基线减少 43.14%；驻留候选减少 53.93%，相对只用 GPU cache 再减少 18.97%。三组随机生成的响应 tokens 不完全相同，第一步有冷启动，第二步还包含小样本验证，整步均包含 checkpoint 保存。**这些数字是初步证据，不是完整训练或关闭 profile 后的速度承诺。**

驻留两步最坏 worker rank 的 update allocated 峰值为 33.267 / 46.157 GiB，reserved 为 37.945 / 50.961 GiB。前两组 allocated 是累计峰值，驻留组是每 update 重置后的峰值，不能直接比较内存增量。worker allocator 也不覆盖其他进程和整张 GPU 的全部显存；后续最长序列测试已补充 NVML 整卡监测，见 2.5。

### 2.2 cache 和分布式边界 gate

resident smoke 的前置 gate 已覆盖：八卡 BF16 FA2 前向/反向，官方 FA4 prefill/decode/CUDA graph，CPU/CUDA cache 数值比较，以及两 rank 的实际 `OPDWorker.update_opd` 路径。两 rank 合成模型 gate 使用真实 vocab 大小，覆盖超过 512 token 的 chunk、Soft-OR 分数和选中索引、loss、梯度、Adam 状态、参数、scheduler 和 cache 释放；另测一个 rank 没有有效 response token 的边界。

这些 cache 等价证据不能替代真实 Qwen 长序列训练、原始数据多步稳定性或完整 benchmark 验证。两行小样本 prompt 的 32 输出验证通过不能证明完整 MATH/AIME 精度保持。

### 2.3 TP=1 探索

另一次 TP=1、profile=False 的两步 smoke 已通过，平均生成耗时约 37.445 s。默认 TP=2 的驻留 profile smoke 平均生成约 38.232 s；两组 profile 和生成轨迹不同，而且样本少，差距不足以作为选型依据。**当前交付候选仍为 TP=2；TP=1 未通过本次稳定性与生产验收。** TP 参数被验收 stamp 绑定，修改它会导致正式启动被拒绝。

### 2.4 关闭 profile 的六步稳定性与速度

两组均使用原始 13918 行数据、1739 步 cosine horizon、TP2/microbatch1，仅在六步提前结束；step2/4/6 保存完整八 rank checkpoint，结束输出32条小样本验证。候选稳定性报告通过、launcher/GPU wrapper exit=0、退出后保活恢复。基线训练产物完成，但原观察器误要求一条 INFO 结束文字，原 launcher/GPU wrapper exit=1；修正观察器后只做 CPU 事后核验并通过，保留所有原日志/hashes，25份实际执行源码与原执行记录一致。这个基线例外不适用于候选、容量和恢复测试。

| 统计范围 | 基线 | 驻留候选 | 观测时间减少 |
|---|---:|---:|---:|
| step2–6 平均更新+权重同步 | 69.538 s | 36.586 s | 47.39% |
| 普通 step3/5 平均整步，不含保存/验证/冷启动 | 90.418 s | 66.762 s | 26.16% |
| 全部六步总 step 时间，含三次保存和32输出验证 | 657.337 s | 458.448 s | 单独报告，不直接外推 |
| step2–6 平均 response tokens | 254678.0 | 253981.4 | 总量相近，轨迹/长度分布仍不同 |

主要收益来自 update/offload 路径。普通步只有两条，未来随机生成、长度和完整评估成本可能改变整体速度。详见 [六步稳定性/容量对照](results/optimization-20261001/stability-comparison.md)、[精确指标及汇总公式](results/optimization-20261001/stability-comparison.json)、[候选状态验证](results/optimization-20261001/stability-verification.json) 与 [基线 CPU 复核](results/optimization-20261001/baseline-reverification.json)。这些结果不能证明完整 benchmark 精度保持。

### 2.5 真实 Qwen 最长序列容量

八 rank 真实 Qwen worker、共置 SGLang sleep/wake、完整 vocab TIP scoring/reverse-KL/backward/Adam 路径已完成容量测试。先通过短序列建立 Adam 状态，再测合成的128条最长 response：每条 prompt2048 + response8192，实际总长10240，padded到16384。最大长度 teacher cache 为37.09375GiB/rank，worker update53.5346秒；最坏 worker allocated/reserved63.4298/67.7676GiB。1秒采样的 NVML 整卡 max used119.8564GiB、min free58.4980GiB；launcher/GPU wrapper exit=0，保活恢复。详见 [长序列验证](results/optimization-20261001/long-sequence-verification.json)。

这是合成输入上的真实模型容量证据，并非自然语言 rollout 质量或长期精度测试。NVML 每秒采样可能遗漏瞬时峰值；该结果不能保证所有内容分布的精确最坏峰值。

## 3. 生产验收状态

| 必需证据 | 测试设置 | 当前状态 | 结果 / 速度 / 显存 |
|---|---|---|---|
| 两步基础 smoke | 真实 Qwen、global128、保存/小样本验证/退出 | 已通过 | 见前述三个 smoke |
| 关闭 profile 的基线六步 | 原始 13918 行训练数据；1739 horizon；CPU cache/offload | 更新/保存/验证完成，修正观察器后 CPU 复验通过；原包装器 exit=1 保留 | step2–6 更新均值69.538秒；普通step3/5均值90.418秒 |
| 关闭 profile 的候选六步 | 同数据/horizon；默认 TP2/micro1 驻留候选；step2/4/6 保存 | 已通过；wrapper/launcher exit=0，保活恢复 | step2–6 更新均值36.586秒；普通step3/5均值66.762秒 |
| 长序列容量 | 真实 Qwen；八 rank；预先初始化 Adam 状态；128×8192 response；NVML 整卡监测 | 已通过；wrapper/launcher exit=0，保活恢复 | cache37.094GiB/rank；worker max allocated63.430/reserved67.768GiB；NVML sampled minfree58.498GiB |
| checkpoint 恢复 | 从已验证的候选 step6 恢复到 step7/8；完整 model/optimizer/scheduler/RNG/dataloader | 已通过；wrapper/launcher exit=0，保活恢复 | 更新39.846/33.620秒；数据cursor48→56→64；cosine仍1739步 |
| 统一证据/source/profile 校验与 stamp | 验证日志退出/保活、配置和源码 hashes；完整证据绑定 | 已通过；CPU核验，无生产启动 | 37份源码与报告SHA绑定；TP1/失败stamp/源码变更均拒绝 |
| 优化生产全量训练 | 用户手动启动；1739步；每50步/最终保存及完整验证 | 已完成；wrapper/launcher exit=0，保活恢复 | all37.6353h；最终MATH/AIME24/AIME25=79.4125%/13.7500%/18.1250% |

目前生产 gate 要求：stability/resume/long-sequence 的独立报告通过、GPU wrapper/launcher exit=0、保活恢复；profile 和源码匹配；从指定 step6 恢复出 step7/8；8192 response 的 cache 超过 37 GiB/rank 且实测最小空闲显存大于 4096 MiB；候选 step2–6 的平均更新+同步时间相对基线至少减少 15%。六步稳定性测试仅提前结束运行，不把 cosine horizon 缩短为六步。

基线原包装器 exit=1 的唯一错误是观察器要求的 INFO 结束文字被默认日志级别过滤；Python训练本身成功，六步、checkpoint和验证输出均已完成。没有重新运行训练或改写日志；修复观察器后仅进行 CPU 事后核验，验证状态、原证据未变化、25份实际执行源码相同、已知8卡保活正常。生产 gate 对该基线单独绑定 [事后核验报告](results/optimization-20261001/baseline-reverification.json) 的 SHA256；原始 exit=1 仍保留，候选与恢复不适用这个例外。

验收由 `/volume/pt-test/users/zhaoye/OPSD-B200-opt-runtime/accepted-profile.json` 绑定所选配置、源码 SHA256 和证据报告 SHA256。缺少必需证据、不符合上述明确基线 CPU 审计例外的失败结果、源码或配置变化都会阻止 `all/train/resume`。最终有界核验在 `2026-09-30T20:42:02Z` 通过；记录见 [acceptance-verification.json](results/optimization-20261001/acceptance-verification.json) 和 [accepted-profile.json](results/optimization-20261001/accepted-profile.json)。有界验收证明其已测条件；后续完整1739步与benchmark结果另见本页开头全量报告。

**历史全量粗估为34–39小时，实际all为37.6353小时。** 当时用普通步66.762秒×1739/3600约32.25小时，加历史 B200 保存/评估额外约2.20小时得到34.4小时；用普通步比0.7384缩放历史51.56小时得到38.1小时。依据是关闭 profile 的六步对照，计算见 [稳定性对照](results/optimization-20261001/stability-comparison.md)。该估算仅有step3/5两条普通步，不作为新的时间承诺；实际完整step求和37.3526小时，train-shell37.4189小时，all37.6353小时，三个口径分别记录。

## 4. 硬件事实与性能判断

2026-09-30有界验收诊断中，GPU2 到其他卡的拓扑为 NODE/SYS，其他七卡互联为 NV18；`nvidia-smi nvlink --status -i 2` 报告所有链路 inactive。该次最终复查仍一致，见 [NVLink 复查](results/optimization-20261001/final-gpu2-nvlink.txt)、[拓扑复查](results/optimization-20261001/final-topology.txt)；此前记录见 [OPTIMIZATION_PROGRESS.md](OPTIMIZATION_PROGRESS.md)。后续只读采样仍见异常，本次结果整理未重查端口，没有执行 GPU、驱动或 NVSwitch reset，不能视为已修复。

NVLink 缺失可能影响 FSDP 通信和 TP2 rollout 的物理传输路径，因此本实验速度与机器当时状态有关。没有证据证明此前完整 B200 基线全程处于同样状态，也不能把所有差距归因于此。恢复互联后的速度应重新测量。

现有 profile 确认数据搬运和 offload 存在明显成本，GPU cache/驻留方案有收益；还不能把整个任务归结为单一 HBM memory-bound。教师/学生计算、缓存搬运、FSDP/TP 通信、权重同步和 rollout 共同决定时间。软件候选仍可验证，但完整性能上限也受机器互联状态影响。

## 5. 总入口、进度与保活

### 5.1 路径与隔离

| 内容 | 路径 |
|---|---|
| Taihua 优化代码 | `/volume/pt-test/users/zhaoye/OPSD_B200_Optimize` |
| 总入口 | `full.sh` → `run_b200_optimized_full.sh` |
| 独立 runtime/日志/实验输出 | `/volume/pt-test/users/zhaoye/OPSD-B200-opt-runtime` |
| 输入模型/数据 | `/volume/pt-test/users/zhaoye/OPSD-B200-assets` |
| 已配置 venv | `/volume/pt-test/users/zhaoye/envs/opsd-py312-cu128` |
| 原生产 runtime | `/volume/pt-test/users/zhaoye/OPSD-B200-runtime` |
| 北京控制台工作树 | `/volume/pt-train/users/zhaoye/OPSD_B200_Optimize` |

缓存、临时文件、模型、数据、日志和 checkpoint 写入 GPFS。优化实验使用独立 runtime/output，与原 B200 入口共享 full/GPU 锁以串行使用同一机器；不会覆盖此前生产输出。自动 resume 只查找优化生产输出下的完整 checkpoint；smoke/stability checkpoint 不会被当作生产恢复点，历史原生产 checkpoint 保留。

### 5.2 实时观察命令

在 Taihua CPU/GPU Pod 上，另开终端持续查看当前 run：

```bash
bash /volume/pt-test/users/zhaoye/OPSD_B200_Optimize/full.sh follow
```

单次读取当前 phase、最新 step、loss、LR、耗时和保存进度：

```bash
bash /volume/pt-test/users/zhaoye/OPSD_B200_Optimize/full.sh status
```

从北京 `yzhao04-0` 控制台使用对应 `/volume/pt-train/users/zhaoye/OPSD_B200_Optimize/full.sh`，入口会转发到 Taihua GPU Pod。实际工作仍在 Taihua 的 `/volume/pt-test` 路径执行。

主命令前台执行时持续输出：环境/模型/数据检查、kernel gate、rollout、teacher forward、global TIP selection、student update、权重同步、checkpoint 和验证阶段。Ray 日志实时转发，每 10 秒输出最新进度与每卡 GPU 利用率/显存；同时写入各 run 的 `driver.log`。默认无参数仅执行 `prepare`。

### 5.3 保活行为

- 准备、下载/校验和 CPU 阶段检查并启动已知保活脚本。
- 获取共享 GPU 锁并确认已知保活进程后，GPU kernel gate、smoke、容量测试和训练前停止保活，等待卡空闲。
- GPU 子任务正常退出、报错或收到正常信号时，trap 清理本任务并恢复保活；driver 有兜底检查。
- 使用已知 PID/身份控制保活，不强杀陌生 GPU 任务。若资源被其他任务占用或控制器状态不明，入口拒绝继续。

### 5.4 生产启动命令：仅验收通过后由用户手动执行

**所选配置已通过验收，并已由用户手动完成一次全量。下面命令保留供后续手动运行；`all/train`会创建新的训练run，本次结果整理未再次启动。**

推荐总命令：检查环境/模型/数据并做两步 smoke，然后启动完整1739步训练：

```bash
bash /volume/pt-test/users/zhaoye/OPSD_B200_Optimize/full.sh all
```

直接开始新的全量训练（仍执行环境/数据和kernel gate）：

```bash
bash /volume/pt-test/users/zhaoye/OPSD_B200_Optimize/full.sh train
```

恢复最新完整的优化生产 checkpoint：

```bash
bash /volume/pt-test/users/zhaoye/OPSD_B200_Optimize/full.sh resume
```

需要明确 checkpoint 时，在 `resume` 后附该优化生产 `global_step_*` 目录的绝对路径。新训练、恢复和 `all` 均由用户手动执行；有界实验通过不自动启动全量。

## 6. 结果索引

- [最新优化全量：1739步/35评测/35checkpoint审计](results/taihua-b200-optimized-20261003/RESULTS.md)
- [优化B200、原B200、H200完整性能与精度对比](results/taihua-b200-optimized-20261003/B200_OPT_VS_BASELINE_AND_H200.md)
- [全量图表](results/taihua-b200-optimized-20261003/summary.png)
- [关闭 profile 的六步稳定性/容量对照](results/optimization-20261001/stability-comparison.md)
- [精确稳定性指标、汇总公式和证据 hashes](results/optimization-20261001/stability-comparison.json)
- [六步候选状态验证](results/optimization-20261001/stability-verification.json)
- [六步基线状态验证](results/optimization-20261001/baseline-verification.json)
- [基线 CPU 事后审计，原 exit=1 保留](results/optimization-20261001/baseline-reverification.json)
- [真实 Qwen 最长序列容量验证](results/optimization-20261001/long-sequence-verification.json)
- [三组优化 profile smoke 对照](results/optimization-20261001/profile-comparison.md)
- [去重 smoke 指标与日志 hashes](results/optimization-20261001/profile-comparison.json)
- [基线 smoke 验证](results/optimization-20261001/baseline-smoke-verification.json)
- [GPU cache smoke 验证](results/optimization-20261001/gpu-cache-smoke-verification.json)
- [驻留 smoke 验证](results/optimization-20261001/resident-smoke-verification.json)
- [TP1 探索 smoke 验证](results/optimization-20261001/tp1-smoke-verification.json)
- [已完成 B200 基线结果](results/taihua-b200-20260930/RESULTS.md)
- [已完成 B200 与 H200 对比](results/taihua-b200-20260930/B200_VS_H200.md)
- [环境/attention 迁移记录](B200_MIGRATION.md)

- [CPU/CUDA缓存等价性](results/optimization-20261001/cache-equivalence.json)
- [两rank实际OPD更新等价性](results/optimization-20261001/cache-distributed.json)
- [整rank零有效token边界](results/optimization-20261001/cache-zero-rank.json)
- [官方FA4 kernel验证](results/optimization-20261001/fa4-kernel-verification.json)
- [候选恢复验证](results/optimization-20261001/stability-resume-verification.json)
- [恢复退出与保活日志](results/optimization-20261001/resume-cleanup.log)
- [最终验收与拒绝错误配置检查](results/optimization-20261001/acceptance-verification.json)
- [配置/源码/证据绑定](results/optimization-20261001/accepted-profile.json)
- [最终八卡保活状态](results/optimization-20261001/final-keepalive-status.log)

八卡保活最终父进程PID3189967；所有GPU compute PID均为已知保活worker。恢复收尾 `/root`、`/tmp` 增长均为0。生产每50步保存/评估，完整精度结果待用户手动全量运行。
