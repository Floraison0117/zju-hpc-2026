# Milestone B：RHS 计算图分析与上限探针（进行中）

> 依据：plan-340s-sprint.md §5 + plan-340s-sprint-review.md §2（三堵墙）+ Phase 0 账本（RHS 460.9s / 64.8%）。
> 方法：源码级 DAG/live-set 分析（bssn_rhs_gpu_deployed.cu，4a4b2aab）+ SASS 指令构成 + L0 探针（job 163419）。

## 1. 结论先行（分析预期，待探针证实/证伪）

**R3/R4 分组+tiled 机制无法达到计划要求的 RHS ≥2.5×（190s→460.9s 需 2.7×）**，因为：

1. **live 下限不可破**：任何计算 Ricci 的 kernel 至少需要 ~66 个 double 同时存活（见 §2），= 132 寄存器 > 50% occupancy 所需的 64 寄存器上限 → **occupancy 被数学钉死在 25%**（128 regs × 2 blocks = 65536 regs 满）。
2. **分组不降 floor**：Ricci 是 gauge（dtSf）、metric（Aij_rhs）、constraints 三组输出的共同依赖（§2.2），按输出分组（R2/R3/R4）不改变任何一组的 live 下限；按层切分（写 scratch）只是把 floor 搬进读回（rhs-split v2 / Lever A 已实测死路）。
3. **rolling window/smem 复用**：iter6 实测 -5.9%（L1 carveout）；寄存器窗口版抬高寄存器 → 掉 occupancy（P6a 教训同型）。
4. **寄存器/spill/occupancy 曲面已穷尽**：natural 255（12.5%）/ lb2 128（25%，部署最优）/ lb3 80（37.5%，spill 杀死）/ noinline 80（-13%）。25% 是曲面顶点。

**唯一未测且有真实空间的机制：interior 特化（去边界谓词）**。SASS 指令构成显示 ~60% 静态指令是边界/索引/选择机制（ISETP 24% + 整数 23% + MOV/SEL 12%），FP64 数学仅 15%。interior 快速路径（fh → 纯加载）预计减指令 ~20-35%、可能减 spill（谓词机器不再活），但**不改变 live 下限 → occupancy 大概率仍 25%** → 预期 RHS 收益 ~1.2-1.5×，低于 Go 门（2.0×）。

**预期裁决**：RHS 无法达到 2.0× Go 门 → 340s 路线 No-Go（按 plan §12），诚实预期 ~500-560s（RHS 1.4× + prolong 2× + host 减半 + 其余 1.3×）。

## 2. DAG / live-set 分析（源码实证）

### 2.1 输出分组与共同依赖

| 输出组 | 成员 | 需要的"重"中间量 |
|---|---|---|
| G1 标量 | chi_rhs, trK_rhs, Lap_rhs | div_beta（小）|
| G2 shift | betax/y/z_rhs, dtSfx/y/z_rhs | **l_R(6) + l_Gam(18) + gup(6) + betaxx..zz(9) + bx/by/bz 二阶(18) + dGam(9)** |
| G3 metric | gij_rhs(6) | shift 一阶(9)（小）|
| G4 curvature | Aij_rhs(6) | **l_R(6) + l_Gam(18) + gup(6) + term_*(6) + fxx..fzz** |
| G5 constraint | ham/mov/Gm_Res | **l_R(6) + l_Gam(18) + gup(6) + d_Aij(18) + DA_*(18)** |

**Ricci（l_R 6 个）被 G2/G4/G5 共同消费** → 任何含 dtSf/Aij_rhs/constraint 的 kernel 都必须持有完整几何层（l_R + l_Gam + gup = 30 doubles）。

### 2.2 live 下限核算（Step 4 Ricci 修正峰值处）

持续存活（从创建到 §8 输出组装/constraints）：
- 场值：Lap, chi, alpn1, chin1, gxx..gzz(6), trK, Axx..Azz(6) = **17**
- shift 一阶：betaxx..betazz(9) + div_beta = **10**
- chi 一阶：chix/y/z = **3**
- gup(6) + l_Gam(18) + l_R(6) + l_Aij(6) = **36**
- val_Gamx/y/z_rhs(3)
- 合计 ≈ **66 doubles ≈ 132 寄存器**（若 1 double = 2 regs；若 64-bit 寄存器则 66）

Step 4 瞬时额外：gxxx..gzzz 收缩(18) + dGam(9) + Gamxa(3) → 峰值 ≈ **105 doubles**。

**推论**：66-double floor > 50% occupancy 的 32-double 上限（64 regs）；128 regs（25%）是任何全功能 RHS kernel 的硬顶。**这一结论与 21 轮实测曲面完全一致**（natural 255 = sm_80 硬件上限，即 ptxas 想要 ~127 doubles）。

### 2.3 指令构成（cuobjdump 部署 binary，静态）

| 类别 | 静态数 | 占比 |
|---|---:|---:|
| ISETP（谓词/边界） | 28,819 | 24% |
| 整数 ALU（IADD3/IMAD/IMAD.MOV/I2F） | 27,338 | 23% |
| MOV/SEL/FSEL/PLOP3（选择/搬移） | 14,736 | 12% |
| DMUL/DFMA/DADD（FP64 数学） | 17,286 | 15% |
| CS2R/BSSY/BSYNC（控制流） | 9,738 | 8% |
| LDG.E.64（全局加载） | 3,602 | 3% |
| STL.64（spill 存储） | 648 | 0.5% |
| 其他/未归类 | ~13,000 | ~11% |

**观察**：动态每线程 ~1531 指令（ncu：49M issued / 32K threads）；静态 107K+ 含大量分支结构（BSSY/BSYNC ~2400 对）。边界/索引/选择机制占静态 ~60%，FP64 数学仅 15% → **指令削减的杠杆在机制层（interior 特化），不在数学层**。

## 3. L0 探针结果（job 163419/163437/163463/163500/163523/163558）✅ 部分证实，部分推翻预期

候选：`~/lab4-gpu-cand-mb-intprobe-20260825-111904`（4 个 fh lambda 加 `RHSPROBE_INTERIOR` 纯加载路径；hash 守卫 3ede4646/5dddaa75/e88f9632 与部署态一致）。

### ptxas + SASS（单 TU apples-to-apples，flags = -O3 -arch=sm_80 -rdc=true -DUSE_GPU -Dfortran3 -Dnewc -DMPI_CUDA_AWARE=0）

| 变体 | 寄存器 | stack | spill S/L | 静态 SASS | 占用率 |
|---|---:|---:|---:|---:|---:|
| base_lb2（部署） | 128 | 1352B | 5504/5996B | 123,824 | 25% |
| base_nat | 255 | 584B | 1052/1148B | 122,536 | 12.5% |
| **int_lb2** | 128 | 656B | **1836/2548B** | **29,952** | 25% |
| int_nat | 255 | 584B | 1052/1148B | 29,952 | 12.5% |
| int_lb3（80 regs） | 80 | 1768B | 13240/15644B | 32,072 | 37.5% → spill 死 |
| int_lb4（64 regs） | 64 | 2072B | 18084/21856B | 33,336 | 50% → spill 更死 |

### 裁决

1. **预期证实（墙仍在）**：interior 自然寄存器仍 255 = live 下限（Ricci 数据流）不可破 → occupancy 25% 硬顶不变；lb3/lb4 在 interior 上同样 spill 死亡。**R3/R4 的 occupancy 路径确认死路。**
2. **预期外大发现（新杠杆）**：interior 在部署 128-reg 预算下**静态指令 -75.8%（123,824→29,952）+ spill -67%/-58%** —— 本项目历史最大单机制指令削减（此前最大 P6b -13.6%），证据库从未测过指令削减机制。边界/谓词机制确占静态 ~76%。
3. **未决**：内核仍 latency-bound（25% occupancy），动态指令削减量与真实加速必须实测 → **L1 决定性测试：真实 interior/boundary 拆分 kernel**。

## 3.5 L1 计划（真实拆分）

- **interior kernel**：rhs_kernel 的 interior 变体（fh→纯加载，逐 token 保留公式），launch_bounds(256,2) 保持；覆盖 interior 子域（每面退 2 格）。
- **boundary 处理**：原 rhs_kernel 以掩码方式仅对边界壳计算（interior 点 early-return），或 shell-only 网格。
- **bit-exact 保证**：interior 点 fh==f[idx] 逐位（P2 已证 in-range 时 fac=1.0 且 x*1.0==x）；boundary 点走原 kernel → 全点位精确。
- **验收**：L0 ptxas（int kernel 128 regs/spill 1836/2548B 已证）→ L1 2-step A/B（.dat IDENTICAL + 每步时间）→ 若 F≥1.3 接 10 步 → F≥1.7 接 100 步。
- **门**：F<1.1 即停（不满足 340s 路线，但可并入诚实路径）；F≥1.3 是保底价值（诚实路径的 RHS 项）。

## 3.6 L1 实测（job 163622，2026-08-25）✅ 机制证实，门未达

**F=1.128 端到端（2-step A/B，bit-exact 8/8 IDENTICAL，RMS=0）**；RHS 模块级 ≈1.27×（460.9→~363s）。RHS 原型 1.27× < 1.7× No-Go 线 → **340s 路线 No-Go（plan §12）**。但这是 P3.5 之后首个 RHS kernel 正收益杠杆，并入诚实路径。

## 3.7 L2 + 部署（2026-08-25）✅ 新基线锁定

- 候选 L2（job 163800，100 步 OJ-sim）：**This Program Cost = 702.25s**（Total Evolve 658.04s），check.sh FINAL PASS RMS=0，约束逐位一致。
- 部署：9 文件（src/bssn_rhs_gpu_int.cu 新增 + bssn_rhs_gpu.cu/bssn_rhs.h/bssn_gpu_class.C/bssn_step_gpu.C/derivatives.h/lopsidediff.h/kodiss.h/CMakeLists.txt）复制至 formal，hash 与候选逐字节一致；快照 `lab4-gpu-snapshot-pre-intsplit-20260825-122924.tar.gz`。
- 部署态 OJ-sim（job 163899 节点失败 → **163955 通过**）：**This Program Cost = 709.50s**（Total Evolve 665.62s），FINAL PASS RMS=0。
- **新基线：782.59 → ~702-710s（F≈1.10-1.11，端到端 -73~80s，bit-exact）**。
- 注：job 163899 在 step~16 静默死亡（无错误输出、shell 未返回、exitCode 1），同二进制候选 L2 已 100 步 PASS，换节点重跑即过 → 判为节点/容器基础设施故障，非代码问题。

真实 interior/boundary 拆分（候选 `~/lab4-gpu-cand-mb-intsplit-20260825-120249`，见 search-memory 迭代22）2 步 OFF/ON×2 交错 A/B：**F=1.128（median 7.2347→6.4150 s/step），8/8 .dat IDENTICAL + check.sh FINAL PASS RMS=0（bit-exact）**。

- **裁决**：interior 指令削减机制**真实生效**（RHS 模块估 1.27×，端到端 -0.82 s/step ≈ -82 s/100 步），可并入诚实路径；但 < 1.3× 保底门 → **340s 路线维持 No-Go**（预期 ~700s 区间，需 prolong Track B + host 侧 + 本杠杆叠加）。
- **k 边距修正（重要，偏离 §3.5 原文）**：equatorial（Symmetry=1）+ z-bbox[0,320] 下 kmin=-3 激活，lopsided/kodis 在 k=2 处访问 fh(k-3) 反射 → 纯加载读负下标 OOB。interior k 下边距须为 3（i/j 保持 2）。已实测 bit-exact 验证。
- **残余**：2 步乐观值可能衰减；部署前建议 10 步/Level-2 确认（主 agent 决策）。

## 4. 若 No-Go：诚实路径预算（Phase 0 实数）

| 模块 | 现状 | 现实目标 | 目标后 |
|---|---:|---:|---:|
| RHS | 460.9s | interior 特化 ~1.4× | ~330s |
| Prolong | 117.5s | Track B 批处理 ~2× | ~59s |
| 其余 kernel | 132.5s | ~1.3× | ~102s |
| host 间隙 | 29.5s | alloc 复用+launch 批处理 | ~15s |
| 固定 | 43.6s | ~40s | ~40s |
| **合计** | **782.6s** | — | **~546s**（约 430-560s 区间，取决于 interior 实测）|

> 注：plan §12 的"380-430s"假设 RHS ≥2×，本分析显示 RHS ≤1.5× → 预期需下修到 ~500s+。诚实交付：优先做 prolong Track B（最高性价比非 RHS 杠杆）+ host 侧 + RHS interior 实测值。
