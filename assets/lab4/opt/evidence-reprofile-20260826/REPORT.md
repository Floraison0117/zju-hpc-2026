# Reprofile Report — post-iter23 部署态（2026-08-26）

> 作业：165992（nsys 全量 100 步 trace，685.34s，+2.8% 开销）、166024（ncu 早段 8 kernel + SASS 普查）。
> 部署态：rhs interior + prolong3 interior 已部署（job 164356，OJ-sim 666.64s，check FINAL PASS RMS=0，build Aug 25 14:51，与 phase0-baseline-20260825-145025 同一二进制）。
> 本次 nsys 运行本身：**This Program Cost = 685.34s**（含 nsys 开销），Total Evolve 640.6s；check.sh 在作业内因脚本误删输出而 FAIL（非运行失败），位级一致性由同一二进制既有证据（job 145025 FINAL PASS RMS=0）背书。
> 原始证据：`assets/lab4/opt/evidence-reprofile-20260826/`（ledger.txt/ledger2.txt/metrics.txt/sass_stats.txt/stats_*.csv/run.log）。远程：`~/lab4-gpu/evidence/reprofile-nsys-20260826-002832/`、`reprofile-ncu-20260826-005017/`。

## 1. 全量 kernel 时间账本（nsys，kernel 时间基 617.6s / span 641.9s / host gap 24.3s = 3.8%）

| kernel | calls | total_s | avg_ms | share% | blocks/launch min/med/max |
|---|---:|---:|---:|---:|---:|
| **rhs_kernel（boundary）** | 38,944 | **203.72** | 5.23 | **32.98** | 125/512/1120 |
| **rhs_kernel_int（interior）** | 38,944 | **190.54** | 4.89 | **30.85** | 125/512/1120 |
| prolong3_kernel（boundary） | 1,192,683 | **52.10** | 0.044 | **8.44** | 1/43/141 |
| global_interp_kernel | 34,400 | 48.83 | 1.42 | 7.91 | 1/1/144 |
| prolong3_kernel_int（interior） | 1,192,683 | **31.58** | 0.026 | **5.11** | 1/43/141 |
| sommerfeld_rout_kernel | 902,976 | 29.94 | 0.033 | 4.85 | 343/512/1120 |
| restrict3_kernel | 294,975 | 24.58 | 0.083 | 3.98 | 1/29/72 |
| rungekutta4_rout_kernel | 912,576 | 19.76 | 0.022 | 3.20 | 125/422/945 |
| ghost/enforce/misc（9 kernel） | ~2.44M | 12.56 | — | 2.03 | — |

模块合计：RHS 63.84%（int 30.85 + bnd 32.98）、Prolong 13.55%（int 5.11 + bnd 8.44）、Analysis 8.74、Sommerfeld 4.87、Restrict 3.98、RK4 3.20、Ghost 1.04、Enforce 0.74、Misc 0.05。KO dissipation 无独立 kernel（fused 在 rhs 内，见 bssn_rhs_gpu.cu d_kodis_point 调用）。

## 2. 三窗口占比（每窗 214s 墙钟；占比跨窗口高度稳定，与 phase0 结论一致）

| subcategory | EARLY | MID | LATE |
|---|---:|---:|---:|
| rhs_boundary | 33.08% | 33.02% | 32.85% |
| rhs_interior | 31.00% | 30.86% | 30.70% |
| prolong3_boundary | 8.43% | 8.64% | 8.23% |
| analysis | 8.05% | 8.14% | 10.05% |
| prolong3_interior | 5.10% | 5.23% | 5.01% |
| sommerfeld | 4.99% | 4.93% | 4.68% |
| restrict3 | 4.20% | 4.18% | 3.55% |
| rk4 | 3.26% | 3.21% | 3.12% |
| ghost | 1.09% | 0.98% | 1.04% |

四主 kernel blocks/launch 逐窗口（min/med/max）：rhs 双 kernel 125/512/1040-1120 全程一致；prolong3 双 kernel 1/43/134-141 全程一致。**boundary 与 interior 的发射体积完全相同**。

## 3. 四主 kernel 微观指标（ncu 早段 step~1 样本 + 静态 SASS 普查）

### 3.1 ncu（grid 125 rhs / 54 prolong3）

| 指标 | rhs_boundary | rhs_interior | prolong3_boundary | prolong3_interior |
|---|---:|---:|---:|---:|
| duration | 1.43 ms | 1.25 ms | 11.7 μs | 31.2 μs |
| regs/thread | 128 | 128 | 100 | 64 |
| achieved occupancy | 19.0% | 18.7% | 23.0% | 38.0% |
| max warps%（寄存器限制） | 25% | 25% | 25% | 50% |
| issue_active%（有发射周期） | 21.8% | 19.0% | 37.9% | 43.4% |
| stall long_scoreboard | 3.89 | 8.16 | ~0 | 3.82 |
| stall wait(dep) | 2.63 | 2.97 | 2.91 | 2.36 |
| stall no_inst | 4.26 | 2.45 | 0.82 | 0.46 |
| stall icache_miss | 0.03 | 0.04 | 1.71 | 0.52 |
| inst_smsp（动态） | 22.9M | 17.0M | 298K | ~— |
| global_ld sectors | 4.46M | 9.40M | — | 497K |
| local_ld/st（spill） | 3.32M/2.63M | 2.25M/1.33M | 0 | 124K/124K |
| waves_per_sm | 4.46 | 4.46 | 1.93 | — |
| dram_read/write | 48.9/42.2 MB | 37.5/22.9 MB | 51.6/— | 110.1/2.1 MB |

### 3.2 静态 SASS 普查（cuobjdump，指令数）

| 族 | rhs_boundary | rhs_interior | prolong3_boundary | prolong3_interior |
|---|---:|---:|---:|---:|
| **总计** | **123,832** | **30,704（-75.2%）** | **2,912** | **1,104（-62.1%）** |
| ISETP（谓词设置） | 29,274 | 1,033 | 626 | 28 |
| IMAD+IADD（整数索引） | 30,583 | 6,548 | 322 | 182 |
| BSSY/BSYNC | 4,818 | 498 | 188 | 32 |
| SEL/MOV | 9,501 | 2,677 | 102 | 100 |
| **CALL（device call）** | **456** | **456** | **21** | **21** |
| LDL/STL（spill） | 790/728 | 527/312 | 33/6 | 33/6 |
| LDG/STG | 3,602/82 | 3,164/82 | 72/1 | 18/1 |
| DFMA/DMUL/DADD | 15,286 | 9,538 | 433 | 217 |
| MUFU | 456 | 456 | 23 | 23 |

注：CALL 456（rhs）/21（prolong3）= ptxas 双精度除法的库调用（`__cuda_sm20_div_rn_f64_full` / dblrcp slowpath），非跨 TU ABI（P6b/P8 后已无）。rhs 双 kernel CALL 数相同：int 与 bnd 除法点一致。

### 3.3 辅助 kernel 关键异常

- **sommerfeld_rout_kernel：160 regs → occ_limit_regs=1 → 实际占用率仅 9.7%**（max warps 12.5%）。此前 P7b 记录为 46-55 regs（P8 前）；P8 forceinline polint 后膨胀。占 4.85%。
- **global_interp_kernel：33.3M/25.7M local 扇区（栈流量），occ 43.7%，L2 吞吐 86.9%**，占 7.91%（比 phase0 的 5.91% 上升，因 rhs 变快）。
- restrict3：128 regs / 25% max，waves 1.04（grid 29 不足两波）。
- rk4：16 regs / 83.6% occ，但 stall_long_scoreboard 65.4（load 延迟主导），仅 3.2%。

## 4. 关键发现（对决策的直接证据）

1. **boundary kernel 按整个 patch 体积发射（用户假设证实）**：rhs_kernel（boundary）与 rhs_kernel_int、prolong3_kernel 与 prolong3_kernel_int 的 blocks/launch 分布逐窗口完全相同（125/512/1120、1/43/141）。boundary kernel 中 interior 线程仅做索引计算 + 早退，浪费 block 调度与分支执行。
2. **rhs_boundary 是当前第一大单组件（32.98%，203.7s），且每 launch 慢于 interior（5.23 vs 4.89ms avg；动态指令 22.9M vs 17.0M）**。其成本主体是壳点上的掩码 RHS（静态 ISETP 29,274 + BSSY 4,818），非早退本身。
3. **prolong3_boundary（8.44%）同样全体积发射**，且 waves_per_sm=1.93（grid 54）到 min grid=1（大量不足一波的 launch）。
4. 占比跨窗口稳定（±1%），早期 profile 可外推；launch 总数 7.18M（比 phase0 的 5.95M +1.23M，interior 拆分新增）。
5. host 侧：launch 30.0s（4.9% API 时间）+ alloc 16.8s + gap 24.3s（3.8% wall）≈ 10% 级可回收空间；cudaStream/DeviceSynchronize 为等待型（重叠，非增量）。

## 5. 决策门槛评估（用户标准）

| 门槛 | 判定 | 结果 |
|---|---|---|
| **RHS/prolong boundary 合计 ≥10%** | rhs_bnd 32.98 + prolong_bnd 8.44 = **41.42%** | ✅ **触发 → Iter26（boundary compact + face specialization）为主线** |
| RHS interior ≥40% | rhs_int 30.85%（占 RHS 模块 48.3%）| 未触发（按全量口径）；RHS→RK 融合为次线 |
| 大 kernel 不足一个完整 wave 的尾部 | prolong3 grid min=1/med=43（int 并发容量 56 → 0.77 wave）；restrict3 1.04 waves | 部分触发（Iter29 批处理，次线）|
| GPU 空闲/launch gap/同步 ≥5% | gap 3.8 + launch 4.4 + alloc 2.5 ≈ 10% | 触发但为收尾方向（Iter30）|
| 以上均不满足 | — | 不适用 |

**裁决：下一主线 = Iter26（boundary 紧凑化 + face 特化），目标 rhs_boundary + prolong3_boundary（合计 255.8s，41.4%）。**

## 6. Iter26 建议（基于本次数据）

- **收益模型**：rhs_boundary 203.7s 中，(a) 全体积发射浪费（interior-only block 调度 + 早退指令）约 5-15%；(b) 壳点掩码机制（ISETP 29,274 + BSSY 4,818 静态）为主要成本。face 特化用编译期镜像索引替代运行时反射，可显著削减 (b)。prolong3_boundary 52.1s 同理（ISETP 626 静态，216 次/点 d_symmetry_bd_1b 掩码调用）。乐观 30-50% boundary 削减 = 端到端 10-15%。
- **注意**：不能简单"紧凑 1-D 域 + 通用实现"了事——掩码机制削减才是大头（对照 rhs interior -75.2% 静态指令的先例）。face 需要 6 个方向的无分支内核或模板参数化；edge/corner 走通用 fallback（数量少）。
- **验收**：boundary kernel 总时间 -30%（~77s）+ 端到端 ≥3%（~20s）+ 短测 bit-exact；低于 10s 立即停止。
- **前置**：Level-0 ptxas 先验（静态指令/regs/spill），再 L1 A/B，再 L2 100 步。
- **风险**：k 下边距与赤道对称（iter22 教训，kmin=-3）；interior 定义与 boundary 域必须覆盖全集一次、不相交；launch 数可能 +（face 拆多个 launch 的风险），需端到端验证。

## 7. 数据位置与可复现性

- 远程：`~/lab4-gpu/evidence/reprofile-nsys-20260826-002832/`（trace.sqlite 1.3G 可再查）、`~/lab4-gpu/evidence/reprofile-ncu-20260826-005017/`（8 个 .ncu-rep + full.sass 46MB + metrics.txt + sass_stats.txt）。
- 本地：`assets/lab4/opt/evidence-reprofile-20260826/`。
- 工具：`~/reprofile-tools/`（reprofile_ledger.py / ncu_raw_extract.py / sass_stats.py / api_summary.py），本地副本 `tmp/reprofile/`。
- 复现命令：`python3 reprofile_ledger.py <trace.sqlite>`；`ncu --import <rep> --page raw --csv | python3 ncu_raw_extract.py`；`python3 sass_stats.py full.sass`。
