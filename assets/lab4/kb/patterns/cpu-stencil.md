# CPU 瓶颈类 → 优化指令 (任务一 ABE CPU)

> KernelPro 语义反馈算子模式：每条 = 触发条件 (trigger) + 诊断逻辑 (analysis) + 处方建议 (recommendation)。
> 剖析器：`bash tmp/sshz.sh arm '<cmd>'`，诊断脚本 `assets/lab4/opt/diagnose/diag_cpu.sh`。
> 主动式（deterministic）：按瓶颈类跑全部相关工具，不让 LLM 随机挑。

## 瓶颈分类（Stage-1，roofline 式，每候选一次）

从 perf-stat 取 IPC、cache-miss、branch-miss；从 perf record flat 取热点函数占比。

- IPC < 1.8 且 L1/LLC miss 低 → **load/issue-bound（数据依赖）**，非带宽。
- cache-miss 高 → memory-bound。
- branch-miss 高 → 控制流分支。
- 分析相位某 rank 显著高于其余 → **straggler**（被 Allreduce barrier 掩盖，需逐 rank 测）。

## 模式库 (Stage-2 微剖析工具)

### M1: load/issue-bound + whole-array F90 风格

- trigger: `compute_rhs_bssn_` 占 >25%，IPC 1.6-1.8，objdump load:FP > 2:1，fopt-info missed 多。
- analysis: whole-array 表达式物化临时数组；但 gfortran -Ofast 已融合单遍循环。
- recommendation: **先查 search-memory**。显式 i/j/k 循环重写在 §13.4 实测 0 收益（编译器已融合）。load:FP 高是 BSSN 80+ 3D 数组 stencil 固有，非可优化 temp 物化。**不要重试显式循环重写**（已证死路）。

### M2: 除法密集（fdiv 多）

- trigger: objdump 见大量 fdiv（如 metric 求逆 6 处/点）。
- analysis: -Ofast 未把除法转倒数乘法（无 recpe）。
- recommendation: **不要**手动 oinv（§12.2 实测 +2.6% 更慢，gfortran 重新折叠回除法，只增物化）。除法延迟非瓶颈。

### M3: 向量化不足

- trigger: fopt-info vec count 低 / missed 多，且热点是可向量化循环（非数据依赖分支）。
- analysis: 区分 "couldn't vectorize"（真不可）vs "clobber memory"（跨过程别名保守）。
- recommendation:
  - clobber 来自 fderivs 跨过程调用 → PURE 属性可解，但需移除 polint 错误路径 I/O（§13.31 判定不可行，改行为）。
  - loop→memset 识别阻断 → 加 `-fno-tree-loop-distribute-patterns`（**已部署，-2%**）。
  - 循环不变量外提 → 加 `-ftree-loop-im`（**已部署，叠加 -0.93%**）。
  - SVE (`-march=native`) 实测 **-11% 更慢**，不要用（§11.12）。

### M4: 分析相位 straggler

- trigger: 某相位（surf_MassPAng.interp 等）逐 rank max 远高于 min，但 Allreduce 使整 rank 报相同墙时（掩盖）。
- analysis: owner-local + Allreduce 模式，分析点集中在少数 rank。
- recommendation: **DIST_INTERP 模式**（多块复制 + 分布式插值，§16.10 已部署）。Bcast level-0 场数据到全 rank，每 rank 插值自己 [Nmin..Nmax] 切片，保留 Allreduce（check.sh PASS）。**已部署，4.56→0.8s/step**。

### M5: 演化负载不均（块点数 max/min 高）

- trigger: GRID_PROFILE 显示 pts_max/pts_min > 1.3。
- analysis: round-robin 块分配，大块集中在少数 rank。
- recommendation: **LOADBAL（least-loaded-first）**（§16.11 已部署）。**注意**：单独测 0 收益（被分析 straggler 掩盖），须在 DIST_INTERP 之后才显 -13s 真收益。**更细 split（AMSS_LOADBAL_SPLIT_FACTOR）0 收益且 psi4 非逐位（check.sh 仍 PASS，非正确性阻断），但无计时收益，不推荐**（§16.23）。

### M6: 同步等待（sync 占步时高）

- trigger: STEP_PROFILE 显示 sync（waitall/allreduce）占 >15%。
- analysis: **sync 本质是 compute straggler 被 barrier 捕获**，非 MPI 通信（§16.19 实测 Waitall 84% 是 barrier 等待）。移除 barrier 只转移等待（SKIP_NAN_ALLREDUCE 0 净收益）。
- recommendation: 通信优化（合并/异步/移除 barrier）无效。唯一途径是减少 compute straggler 本身（见 M5），但演化整步 straggler_ratio≈1.00（已重叠），子步级 straggler 被 barrier 串联无法优化。

### M7: 编译 flag 扫描

- trigger: 无源码改动空间，寻求 codegen 改善。
- analysis: flag 改变循环变换/向量化。
- recommendation: 已穷尽（见 search-memory §13）。可叠加的 FP-safe flag 仅 `-fno-tree-loop-distribute-patterns -ftree-loop-im`。**不要叠加** -ftree-loop-im -fprefetch-loop-arrays -fivopts（§13.25 破坏正确性）、-fmodulo-sched（§13.46 +0.9% 更慢）、-funroll-loops（0）、-fsplit-loops（边际）。LTO 0、PGO 不可行（ld segfault）。

## 死路清单（已实测，勿重试）

显式循环重写、oinv、fdderivs-direct、fderivs-batch、-march/SVE、armclang、LTO、PGO、OpenMP 15×2/10×3、rank 60、NUMA 绑核、Constraint_Out skip、skip 诊断约束残差、SKIP_NAN_ALLREDUCE、更细 loadbal split、modulo-sched、predictive-commoning。详见 `search-memory.md`。
