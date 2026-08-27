# GPU 瓶颈类 → 优化指令 (任务二 ABEGPU)

> KernelPro 语义反馈算子模式：每条 = 触发条件 + 诊断逻辑 + 处方建议。
> 剖析器：`bash tmp/sshz.sh lab2 '<cmd>'`，诊断脚本 `assets/lab4/opt/diagnose/diag_gpu.sh`。
> 主动式：按瓶颈类跑 ncu + nvcc -Xptxas -v + SASS 全部相关工具。

## 瓶颈分类（Stage-1，roofline）

从 ncu roofline 取 compute/memory bound；从 ncu section 取 occupancy、stall reasons。

- GPU 饱和度高（kernel≈wall，无 idle）+ scoreboard stall 高 → **latency-bound**（当前 rhs_kernel 状态）。
- memory-bound → 带 bandwidth 利用率低，可考虑 shared-mem/合并访存。
- launch overhead 占比高 → cudaGraph/批量化。

## 模式库 (Stage-2)

### G1: latency-bound + 跨 TU 调用 + scoreboard stall

- trigger: rhs_kernel 占 >60%，L1TEX scoreboard stall >40%，"No Eligible" >60%，26 次跨 TU `__device__` 调用。
- analysis: 跨 TU 调用边界使编译器无法 hoist/reorder load，串行化 scoreboard stall；RDC ABI 在调用点 spill caller-saved regs。
- recommendation: **split-with-inline-cuh（context.md Section B）**。把 d_fderivs_point/d_fdderivs_point 移到 `.cuh` 作 `__device__ __forceinline__`，在 rhs 拆分点（Christoffel 之后）切两核。期望：rhs_deriv_kernel 25%→75-100% 占用率，spill 消除。
- **但注意**：早前 rhs split（job 122908）实测 -3% 负收益（RDC ABI spill），inline-.cuh 单核实测 spill 688→2964B（4.3× 恶化，负收益）。**重试前必读 search-memory §15.4**，确认拆分点与 inline 范围是否已修正。

### G2: memory-bound + stencil 重读

- trigger: bandwidth 利用率低，同一场被多次全邻居读。
- analysis: context.md §A 统计 ~1187 独立 global load/point（15 场 × stencil）。
- recommendation: shared-mem 7-point stencil。**但单独不可行**：15 场 × halo tile = 120KB > 82KB/block（25% 占用率）。**仅可作为 G1（inline）之后的 follow-up**，且只 tile 5-6 最重场（betax/betay/betaz/chi/Lap），inline 后 `fh` lambda 可重定向到 shared tile。

### G3: 寄存器压力 / 占用率低

- trigger: 128 regs，25% 占用率，或 spill >0。
- analysis: nvcc -Xptxas -v 看 regs/spill。
- recommendation:
  - `__launch_bounds__(256,4)` 降 regs 到 64 → 但 rhs_deriv 可行（54 live），rhs_core 不可行（Ricci 段 72-99 live，§B.3 证 ≤64 不可行）。
  - `__launch_bounds__(256,1)` 降 spill 但占用率 25%→12.5%，实测 -20%（search-memory §15.4）。**不要用**。
  - 真正减 regs 需算法级（Ricci 修正项重构减少存活量），lab 禁止。

### G4: launch / 传输开销

- trigger: H2D/D2H 或 launch 占比高。
- analysis: **当前 nsys 实测 H2D/D2H 仅 0.289s/step 可忽略，GPU 100% 饱和无 idle**。
- recommendation: streams/async/MPI_CUDA_AWARE/fusion/cross-variable batching **当前无收益空间**（已饱和）。cudaGraph 捕获 RK4 消除 33k launches/step 预估 15-30s，不足弥合 888s 差距，可作边际尝试。

### G5: TwoPuncture 路径

- trigger: TwoP 耗时高（>100s）。
- analysis: 正式 lab4-gpu 在 AMSS_ENABLE_GPU=ON 时强制 TwoP 走部分 GPU 路径（~268s）。
- recommendation: **TwoPuncture CMake 解耦（AMSS_ENABLE_TWOP_GPU）+ OMP_TUNE**，让 TwoP 走 CPU-OpenMP ~34s 路径（§15.2 已部署，省 ~234s）。ABEGPU 源码零改动。

## 死路清单（已实测，勿重试）

rhs split v1（-3%）、rhs inline-.cuh 单核（spill 4.3× 恶化）、lb(256,1)（-20%）、lb(256,4)（中性）、memory pool（中性）、block shape sweep（中性）、shared-mem 单独（不可行预算）。详见 `search-memory.md`。
