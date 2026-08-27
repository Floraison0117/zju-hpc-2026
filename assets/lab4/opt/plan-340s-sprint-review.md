# Plan-340s 冲刺：证据库对照审查

> 审查对象：`plan-340s-sprint.md`（2026-08-25）。依据：`assets/lab4/kb/search-memory.md`（iter1-21 全表）、`ABEGPU.md`、`README-lab4.md` §15-§16、`assets/lab4/opt/fitness-gates.md`。
> 结论：计划骨架成立、门禁正确，但 3 处机制与已测死路重叠、1 处预算口径需修正。采纳本文修正后即可执行。

## 0. 总体结论

计划方向与证据库最一致的判断吻合：search-memory iter11/12 已收敛判定"当前内核结构内 500s 不可达，唯一剩余路径是算法级重构"，CPU 侧 Stage 1b k-滚动重写（-23%、120/120 满分）是同类先例。计划的禁止清单（FP32、maxrregcount、LDG/smem 包装、cudaGraph 主导）与已证死路一一对应，§9 三级验证与 fitness-gates.md 一致。

但两点必须清醒：

1. **CPU 先例的机制不机械迁移**。CPU 的 -23% 来自"消除 42 个导数数组物化 → working set 6.6MB→~1MB 入 L2"。GPU rhs_kernel 已点态融合、无数组物化，L1 hit 80.91%/L2 hit 91.14% 已高，瓶颈是**寄存器 live-set（natural 255 = sm_80 硬件上限）+ load 延迟体积（L1TEX scoreboard 46.6%、eligible warps 0.49）**。GPU 必须靠 occupancy 逃逸或指令/依赖重构，这是与 CPU 不同的战场，2.5× RHS 目标没有 CPU 先例背书。
2. **340s 需要 2.19× 总加速，难度高于 CPU 先例的 1.36×**。预算表数学自洽（483→190 / 123→60 / 137→80 = 330s），但 RHS 若只达 2.0×（242s），即使 prolong 2× + other 1.7× 也到 ~382s。§12 门禁是最后防线，必须严格执行。

## 1. 与证据逐项对照

| 计划条目 | 证据（search-memory / README） | 判定 |
|---|---|---|
| 743 s 基线、RMS=0 | job 157367（P10 cudaDeviceReset） | ✓ |
| RHS ~65% / prolong3 ~16.5% | 旧 nsys（job 121872）：rhs 69.1% / prolong3 16.5%；P6b/P8 只砍 non-rhs → 现在 rhs 占比只会更高 | 待 Phase 0 重校准（计划已要求）✓ |
| "强制压寄存器会 spill 变慢" | lb(256,3/4) spill 杀死、P20 noinline F=0.886 | ✓ |
| 禁止 FP32 主路径 | 迭代5：2 步 RMS=0.715% 即 7× 超限，数学必然失败 | ✓ |
| 禁止 fast-math/低阶/少场量 | §11.12、iter5 分析 | ✓ |
| CUDA Graphs 收益有限 | §15.7：已关闭（GPU ~100% 饱和，≤3% 上限） | ✓ 计划已降级为"时间线有空隙才用" |
| streams 消除同步 | per-stream sync 多轮 A/B 纠正为 0 收益（推翻 §15.7 假阳性） | ⚠ Phase 4.2 须先由 Phase 0 trace 证明空隙存在，勿预设收益 |
| 11,821 次 launch 只省 ~1% | ABEGPU.md：GPU kernel 连续忙碌（12.78 vs 12.31 s/step） | ✓ Phase 1 收益必须来自有效工作量提升，而非 launch 本身 |
| R1 单 kernel DAG 重排降 live-set | iter12 P3.5c：F=0.9999，ptxas 吸收源码级重排；iter9 P3：natural 255 纹丝不动 | ⚠ **建议砍掉 R1**，从 R3/R4 开始 |
| R3/R4 tiled rolling window + smem | iter6 SMEM tiling：bit-exact 但 **-5.9%**（L1 164→56KB carveout 是主因） | ⚠ 必须控 smem 足迹（滚动窗口而非整 tile），L0 检查 L1 hit 不塌 |
| R2 分组 + 少量 scratch | rhs split v2：kernel2 spill 5.6×；Lever A：+0.4% 不变 | ⚠ 分组边界不能把 Ricci 33+ 变量整体留在一核；见 §2 墙 3 |
| Phase 3 "展开独立 accumulator 打断依赖链" | P6a unroll：F=0.9235 死路（66→120 regs，占用率 3→2 blocks） | ⚠ 该机制与 P6a 同源；新杠杆是批处理（Track B，未测）+ 分流 + 权重预计算，且须守 ≤66 regs |
| prolong3 目标 2× | P6b forceinline 已 +13.6%（部署，job 155105） | ✓ 剩余 ~1.7× 靠 Track B 兑现；计划自带 "<10s 则停止" 门禁 ✓ |
| "其他" 1.7×（137→80 s） | TwoP ~34 s 固定（TwoP 解耦 + OMP_TUNE 已部署） | ⚠ **TwoP 应单列为固定 34 s**；可演化"其他"实为 ~103 s，1.7× 后 = 60 + 34 = **94 s**，预算表 80 s 口径需修正 |

## 2. 必须打破的三堵墙（Phase 2 成败关键）

过去 21 轮失败不是"机制不对"，而是每次都只达成了一半目标：

1. **P3/P5/P20 从未同时达成 (spill↓ + occupancy↑)**：
   - P5：spill stores -8% / loads -8.3%，但 natural regs 仍 255 → occupancy 零变化，且 61 cvt 开销 > 收益（F=0.9902）；
   - P20 noinline：regs 128→80（occupancy 投影翻倍），但 spill loads 6716B 反增 → F=0.886；
   - P3.5c：源码级重排被 ptxas 寄存器分配吸收（F=0.9999）。
   R3/R4 的假设是"分组 + 空间复用"能同时拿到两者。**验收必须同时看三列：natural regs、spill bytes/point、achieved occupancy**，缺一即视为未达（比计划 §5.3 的"spill ↓≥70%"更严）。
2. **iter6 的 L1 carveout**：任何 smem 使用都缩 L1（latency-bound 内核的命脉）。iter6 用 (12,12,8) 整 tile × 6 场 = 55KB/block → L1 164→56KB → 净 -5.9%。滚动窗口建议 smem 预算 ≤16-24KB/block，并在 L0 记录 L1 hit rate 变化。
3. **fdderivs 61-fh 是 natural-255 的真峰值段**（iter9 定位，非 Ricci 段）：P3 分量级重算 Ricci 后 natural 255 不动。分组调度若只拆 Ricci 段（P3 已证无效），不会动 natural 峰值。必须把 fdderivs 的 61 个 fh 值按输出组拆分（P3.5 的 16→8 模式，但粒度在"单次 d_fdderivs_point 调用内"），且注意 P5 教训：float 精读暂存精度安全但指令开销否决。

## 3. 具体修正建议（执行前采纳）

1. **砍掉 R1**（单 kernel DAG 重排）：P3.5c 已证明同结构源码级重排被 ptxas 吸收；R1 不改变 kernel 边界、无新机制，只花一周复现死路。直接实现 **R3（2 组 + 滚动窗口）与 R4（3 组 + 选择性重算）的 interior 原型**。
2. **R3/R4 原型预门禁**（加在 §12 之前）：interior-only 原型须同时满足 (a) spill bytes/point ↓≥70%、(b) achieved occupancy ↑（不只 regs 数）、(c) L1 hit 不塌，才接边界路径。任一不满足按 §12 No-Go 止损，不扩展通用框架。
3. **复用部署 helper 逐字**：任何重写 kernel 中的 d_fderivs_point / d_fdderivs_point / d_lopsided_point / d_kodis_point / fh（含 P1 clamp、P3.5 分组、fh 语义）必须 verbatim 复制。CPU 先例证明"逐 token 保留 → bit-exact"；CUDA 侧只要每点表达式不变、邻域值相同，同样成立。
4. **prolong3 守住寄存器上限**：只做 (a) 跨 patch/变量批处理（Track B，证据库明示"未测"）、(b) interior/boundary 分流（去分支）、(c) 权重/索引预计算（去整数运算）；**不整段 unroll**（P6a 教训）。目标 ≤66-80 regs/thread 维持 3 blocks/SM。
5. **"其他"预算重算**：TwoP 34 s 固定 + 可演化其他 ~103 s × 1/1.7 ≈ **94 s**（非 80 s）。由此 RHS 目标预算应相应压到 ~185 s 量级（或提高 prolong 目标）。Phase 0 账本必须给出这个数。
6. **Phase 0 的 OJ 提交立即做**（锁定 743 s 得分）✓。提交前确认 TwoP cache 是否随包（CPU 侧有 seeded cache 912e370b84cec7cb；GPU 侧账本要记录 TwoP 34 s 是否含 cache 命中差异）。
7. **A/B 纪律补充**：仓库已有多轮假阳性教训（per-stream sync 假 -30 s、P6b 前 probe 噪声），L1 短跑固定 OFF/ON ×2 交错（≥4 轮）是下限。

## 4. 可行性预期（诚实）

- **最可能的失败点**：R3/R4 原型不能同时达成 spill↓ + occupancy↑（§2 三堵墙之一未破）。届时按 §12 止损，转向"跨变量批处理 prolong3 + 账本内其他"的次优组合，预期 400-500 s 区间。
- **次可能的失败点**：Phase 3 批处理只省 launch/索引开销，per-point 插值数学不变 → 收益 <10 s（计划自带停止门禁 ✓）。
- **可行路径的乐观上界**：RHS 2.5×（193 s）+ prolong 2×（60 s）+ other 94 s = 347 s，仍略超 340 s 硬线；故 **330 s 工程预算不是保守而是必需**，且 any 模块欠账需另一模块超额补偿。

## 5. 建议执行顺序（微调版）

1. Phase 0（账本 + OJ 锁定）— 2 天内，不阻塞后续（计划 Milestone A 原样）；
2. Phase 1（patch descriptor + 分桶）— 与 R3 原型并行（分桶是 R3 的载体）；
3. Phase 2 R3/R4 interior 原型（**跳过 R1**）— Go/No-Go 检查点；
4. Phase 3 批处理 prolong3（Track B）；
5. Phase 4/5 视 Phase 2/3 结果投入。
