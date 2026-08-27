# Phase 0 账本：ABEGPU 340s 冲刺

> 来源：plan-340s-sprint.md Phase 0 + plan-340s-sprint-review.md。基线 = 部署态 P1+P2+P3.5+P6b+P8+P10。
> 作业：160372（OJ-sim 基线锁定）✅、160427（nsys 账本）✅、160487（ncu 微观，运行中）。
> 权威数据位置：`~/lab4-gpu/evidence/phase0-baseline-20260825-005626/`、`phase0-ledger-20260825-011418/`（nsys.sqlite 1.1G 可再查）。

## 0. 基线不可变快照 ✅

- 快照：`~/lab4-gpu-snapshot-phase0-20260825-005557.tar.gz`（提交包 9 项 + src 全部）。
- 部署态 hash 核对（vs search-memory 记录，全部逐字节吻合）：
  - `bssn_rhs_gpu.cu` 4a4b2aab ✓、`derivatives.h` 3ede4646 ✓（P3.5）、`lopsidediff.h` 5dddaa75 / `kodiss.h` e88f9632 ✓（P2）、`prolongrestrict_cell_gpu.cu` a81cd33e / `sommerfeld_rout_gpu.cu` e9ea7810 ✓、`gpu_manager.cu` 435c9b69（含 cudaDeviceReset，P10）✓。
- 注：`lab4-gpu-submission.sha256` 是 08-22 旧提交包清单（P1-P10 之前），非当前权威；当前权威 = search-memory 逐迭代记录 + 本快照。

## 1. 基线锁定（Job 160372）✅

| 项 | 值 |
|---|---|
| 作业 | 160372（2026-08-25 00:56 UTC，节点 j160372-cmn5s，MIG 1g.10gb）|
| 配置 | MPI=1, OMP=8, GPU=yes, Final=100, Analysis=0.1, Dissipation=0.15, TwoP live（无 cache）|
| **This Program Cost** | **782.593 s**（RUN wall=792s）|
| **Total Evolve Time** | **738.96 s**（≈7.39 s/step）|
| 固定开销（含 TwoP） | 782.59 − 738.96 = **43.63 s**（TwoP ~34s + 分析 0.1 + IO/启动）|
| check.sh | **FINAL PASS**，Trajectory RMS=0（0.000000%）|
| 约束 Ham/Px/Py/Pz | 0.28974817 / 0.039343259 / 0.047298107 / 0.044686547（与 golden 逐位一致）|
| 记录对照 | search-memory "部署节点实测 ~782s" 吻合；记录 743s（job 157367）为另一口径/节点。**本账本以 782.59s 为正式基线** |

**含义**：340s 目标需 **782.59/340 = 2.301×** 总加速（计划表 2.185× 基于 743s，需上调）；330s 工程预算对应 2.37×。节点噪声 ±5%（历史 736.8-782.6s），最终两次 <340s 需同节点验证。

> **2026-08-25 更新（Milestone B 部署后）**：新基线 = **This Program Cost ~702-710s**（job 163800 候选 702.25s / job 163955 部署态 709.50s，F≈1.10-1.11 vs 782.59s，bit-exact，详见 README-lab4.md §17）。340s 目标对应 702/340 = 2.07×（仍 No-Go，RHS 仅 1.27×）。

## 2. 时间账本（Job 160427，nsys 全 100 步 trace）✅

- Total Evolve 739.14s（vs 干净 738.96s，差 0.02%）；This Program Cost 783.37s（nsys 开销 +0.8s）。
- **594.5 万次 kernel 调用**；kernel 总 710.9s / 演化跨度 740.5s（96% 饱和）；host 间隙 29.5s（4.0%）。

### 2.1 kernel→模块映射（归类依据）

| Kernel | 源文件 | 模块组 |
|---|---|---|
| rhs_kernel | bssn_rhs_gpu.cu | RHS |
| prolong3_kernel | prolongrestrict_cell_gpu.cu | Prolong |
| restrict3_kernel | prolongrestrict_cell_gpu.cu | Restrict |
| sommerfeld_rout(_bam)_kernel | sommerfeld_rout_gpu.cu | Sommerfeld |
| rungekutta4_rout_kernel | rungekutta4_rout_gpu.cu | RK4 |
| enforce_ga_kernel | enforce_algebra_gpu.cu | Enforce |
| global_interp(_amr)_kernel, average*, l2normhelper, surf_*, admmass, getnp4, scale_normals | fmisc/surface_integral/fadm/getnp4 | Analysis |
| gpu_pack/unpack_kernel | fmisc_gpu.cu | Ghost |
| lowerboundset, normalize_shellf | fmisc/MPatch | Misc |
| J_times_dv / F_of_v | twopunctures_gpu.cu | TwoP（decouple 后 0 次 ✓）|

### 2.2 全量 kernel 汇总（cuda_gpu_kern_sum）

| Kernel | calls | total_s | avg_ms | share% |
|---|---:|---:|---:|---:|
| rhs_kernel | 38,944 | 460.88 | 11.83 | 64.83 |
| prolong3_kernel | 1,192,683 | 117.52 | 0.099 | 16.53 |
| global_interp_kernel | 34,400 | 42.00 | 1.22 | 5.91 |
| sommerfeld_rout_kernel | 902,976 | 29.94 | 0.033 | 4.21 |
| restrict3_kernel | 294,975 | 24.57 | 0.083 | 3.46 |
| rungekutta4_rout_kernel | 912,576 | 19.44 | 0.021 | 2.73 |
| gpu_unpack_kernel | 1,926,204 | 4.99 | 0.003 | 0.70 |
| enforce_ga_kernel | 38,024 | 4.62 | 0.12 | 0.65 |
| average2_kernel | 93,528 | 2.26 | 0.024 | 0.32 |
| global_interp_amr_kernel | 6,370 | 1.42 | 0.22 | 0.20 |
| gpu_pack_kernel | 438,546 | 1.42 | 0.003 | 0.20 |
| surf_Wave_kernel | 800 | 1.13 | 1.42 | 0.16 |
| lowerboundset_kernel | 38,024 | 0.32 | 0.008 | 0.05 |
| 其余 8 个小 kernel | ~27,000 | 0.54 | — | 0.07 |
| **合计** | **5,945,370** | **710.9** | — | 100 |

### 2.3 模块分组

| 模块 | calls | total_s | 占 kernel | 占 Program Cost |
|---|---:|---:|---:|---:|
| **RHS** | 38,944 | **460.88** | 64.8% | 58.9% |
| **Prolong** | 1,192,683 | **117.52** | 16.5% | 15.0% |
| Analysis | 144,818 | 47.12 | 6.6% | 6.0% |
| Sommerfeld | 912,576 | 30.06 | 4.2% | 3.8% |
| Restrict | 294,975 | 24.57 | 3.5% | 3.1% |
| RK4 | 912,576 | 19.44 | 2.7% | 2.5% |
| Ghost | 2,364,750 | 6.40 | 0.9% | 0.8% |
| Enforce | 38,024 | 4.62 | 0.7% | 0.6% |
| Misc | 46,024 | 0.33 | 0.05% | 0.04% |
| host 间隙 | — | 29.5 | — | 3.8% |
| 固定（TwoP+启动/IO） | — | 43.6 | — | 5.6% |

### 2.4 三窗口（GPU 时间轴三等分；每窗口 247s 墙钟）

| 模块 | EARLY（粗网格）| MID | LATE（细网格）|
|---|---:|---:|---:|
| RHS | 64.9% / 154.7s | 64.8% / 155.2s | 64.7% / 151.0s |
| Prolong | 16.5% / 39.4s | 16.8% / 40.3s | 16.2% / 37.9s |
| Analysis | 6.2% / 14.7s | 6.1% / 14.7s | 7.6% / 17.7s |
| Sommerfeld | 4.3% / 10.3s | 4.3% / 10.2s | 4.1% / 9.5s |
| Restrict | 3.7% / 8.7s | 3.6% / 8.7s | 3.1% / 7.2s |
| RK4 | 2.8% / 6.6s | 2.8% / 6.6s | 2.7% / 6.3s |
| kernel 饱和率 | 97% | 97% | 95% |

**网格演化观察**：占比跨窗口高度稳定，但调用形态有变：prolong 400,731→418,992→372,960 次、restrict 109,935→103,296→81,744 次（后期 coarse 操作减少）、analysis interp 10,550→13,090 次（后期增多）。占比层面"只 profile 前两步"误差 <1%，但绝对时长与调用形态不同，L1 A/B 仍需覆盖多窗口。

### 2.5 host 侧（cuda_api_sum，等待型 API 与 kernel 时间重叠，非增量）

| API | 总时间 | calls | 含义 |
|---|---:|---:|---|
| cudaStreamSynchronize | 481.3s | 111,900 | 等待 kernel（非增量）|
| cudaDeviceSynchronize | 186.2s | 92,323 | 同上 |
| **cudaLaunchKernel** | **24.4s** | 5,945,370 | **launch 开销 = 3.1%，批处理可部分回收** |
| **cudaMalloc + cudaFree** | **16.7s** | 223,313 + 222,513 | **alloc 抖动 = 2.1%，预分配可回收 ~15s** |
| cudaMemcpy（含 Async） | ~8.9s | 92,022 + 7,210 | 1.1% |
| cudaMemset | 1.4s | 203,066 | 0.2% |

### 2.6 预算重算（340s 目标下的硬约束）

基线 782.59s = kernel 710.9 + host 间隙 29.5 + 固定 43.6。340s（工程预算 330s）：固定 ~40s 不可压缩 → 演化+host ≤ 290s → **总 kernel 需 710.9 → ~275-285s（2.5-2.6×）**：

| 模块 | 现状 | 目标加速 | 目标预算 | 备注 |
|---|---:|---:|---:|---|
| RHS | 460.9s | **≥2.7×**（比计划 2.5× 更紧）| ≤170s | 782.59 基线下 340s 的必要条件 |
| Prolong | 117.5s | ≥2.2× | ≤53s | |
| 其余 kernel | 132.5s | ≥1.9× | ≤70s | analysis 47 + sommerfeld 30 + restrict 25 + RK4 19 + ghost/enforce 11 |
| host 间隙 | 29.5s | →≤15s | ≤15s | alloc 复用（~15s）+ launch 批处理 |
| 固定 | 43.6s | ~1.0× | ~40s | TwoP 34 不动；启动/IO 减 ~3s |
| **合计** | **784.0** | — | **≤348s** | 340s 仍需 RHS ≥2.9× 或综合再压 |

**结论：计划表 2.185× 低估了难度。真实基线下 340s 需要 RHS ≥2.7-2.9×（170-159s）+ Prolong ≥2.2-2.5× + 其余 2× + host 减半 + 固定压 40s。门禁（plan §12）必须严格执行；RHS 原型 <2× 时 340s 路线判定不可达，转向 380-430s 诚实预期。**

### 2.7 账本闭合检查 ✅

kernel 710.9 + host 间隙 29.5 + 固定 43.6 = 784.0 vs This Program Cost 782.59 → **误差 0.18% ≤ 3% ✓**。RHS/prolong 占比复认：**64.8% / 16.5%（计划假设 65%/16.5% 精确命中）**。

## 3. 微观指标（Job 160487 + 160541，ncu --set full，早期步样本）✅

> 全部样本 block 均为 (8,8,4)=256 threads；寄存器/占用率为 launch 无关量，mid/late 网格同值（ptxas 静态）。ncu 2026.2，device CC 8.0，MIG 1g.10gb。

| 指标 | rhs_kernel（step~1，grid 5³ blocks，2.52ms）| prolong3_kernel（grid 54，96μs）| restrict3_kernel（grid 29，79μs）|
|---|---:|---:|---:|
| 寄存器/thread | 128 | 100 | 128 |
| spill（local 流量/launch）| **1.39 MB** | 0 | 0 |
| 理论/实际 occupancy | 25% / **22.14%** | 25% / 20.61% | 25% / 21.64% |
| Block Limit Registers | 2（寄存器受限）| 2（寄存器受限）| 2（寄存器受限）|
| eligible warps/scheduler | 0.41 | 0.75 | 0.75 |
| No Eligible 周期 | **74.30%** | 54.05% | 55.46% |
| One+ Eligible 周期 | 25.70% | 45.95% | 44.54% |
| Warp Cycles/Issued Inst | 14.01 | 7.46 | 8.29 |
| IPC（active）| 1.01 | **1.76** | 1.66 |
| Issue Slots Busy | 24.63% | 40.01% | 26.94% |
| 指令/launch | 49.0M | 3.04M | 1.69M |
| DRAM 吞吐 | 26.28% | 3.78% | 5.19% |
| L2 / L1 吞吐 | 24.10% / 14.83% | 13.23% / 15.44% | 11.01% / 24.24% |
| Compute 吞吐 | 27.41% | 40.01% | 27.49% |
| 访存带宽 | 63.57 GB/s | 9.13 GB/s | 12.56 GB/s |

**关键观察**：
1. **三大热点全部寄存器受限**（Block Limit Registers=2 → 理论 25% occupancy），印证 iter1-12 收敛结论：occupancy 墙在寄存器，不在 warp/smem 限制。
2. **rhs 是 latency-bound**：IPC 1.01 + Warp Cycles/Inst 14.01 + No Eligible 74.3% + spill 1.39MB/launch（49M 指令的局部内存流量）。stall 结构历史记录 L1TEX scoreboard 46.6% 仍适用（rhs 未变）。
3. **prolong3/restrict3 是依赖/占用混合**：IPC 1.76/1.66 尚可、0 spill，No Eligible ~55% 主导；P6b 后 prolong3 已无 CALL 开销，剩余头寸在 occupancy（100 regs→理论 25%）与依赖链。
4. DRAM 全部 <30%：**非带宽问题**（与 CPU 侧 k-滚动砍流量的机制不同），验证审查 §0 判断：GPU 侧必须靠 occupancy 逃逸或指令/依赖重构。

> sommerfeld 未采到 ncu（PASS3 的 launch-count 2 被 restrict3 占用）；其结构已由 iter18 P9/P7b 覆盖（55/46 regs、124 div 非杠杆）。

## 4. 结论与预算裁决（Phase 0 输出）✅

1. **340s 中每模块最多秒数**（782.59s 真实基线下，工程预算 330s）：
   - RHS ≤170s（需 ≥2.7×；340s 硬线则 ≤159s 即 2.9×）；
   - Prolong ≤53s（≥2.2×）；
   - 其余 kernel ≤70s（≥1.9×，analysis 47 + sommerfeld 30 + restrict 25 + RK4 19 + ghost/enforce 11）；
   - host 间隙 ≤15s（从 29.5s：alloc 复用 ~15s + launch 批处理）；
   - 固定 ~40s（TwoP 34 不动）。
   - 合计 ≤348s（330 工程预算）→ 340s 硬线需 RHS ≥2.9× 或综合再压。
2. **占比复认**：RHS 64.8% / Prolong 16.5%（kernel 时间基）与计划假设精确一致 ✅；Program Cost 基下 RHS 58.9%、Prolong 15.0%、其余 kernel 16.9%、host 3.8%、固定 5.6%。
3. **预算表修正**（采纳审查建议 + 新数据）：TwoP 34s 固定单列 ✓；"其他" 拆为 kernel 132.5s + host 间隙 29.5s 两桶（原计划 137s 单桶口径不准）；**总目标从 2.185× 上调到 2.30×**。
4. **新发现**：
   - launch 开销 24.4s（3.1%）+ alloc 抖动 16.7s（2.1%）= ~41s host 侧可回收空间（计划 Phase 4.2 与 Phase 1 批处理直接对口）；
   - 三窗口占比稳定 → 早期步 profile 的 L1 结论可外推全演化；
   - 三大热点全部寄存器受限（Block Limit Registers=2），ncu 证实 occupancy 墙。
5. **Go/No-Go 建议**（按 plan §12 + review §3）：RHS interior 原型（R3/R4，跳过 R1）先行；代表步 <1.7× 即判定 340s 路线不可达、转 380-430s 诚实预期；Full RHS 端到端省 <270s 不继续 Phase 3/4。
