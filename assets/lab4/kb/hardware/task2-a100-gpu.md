# 任务二 ABEGPU 硬件约束 (zju-hpc-lab2)

## 机器

- 分区：`lab4g10`，单 A100 80GB MIG `1g.10gb` 实例（Nsight Compute 可能报告为 A100，CC 8.0；以分区名为准）。
- 14 SM（MIG 1g.10gb slice）。30 min 墙钟/作业。
- 剖析器：`ncu` (`/usr/local/bin/ncu`)。nsys/nvcc 经 module 加载。
- 主机：Intel Xeon Gold 5418Y（OJ 在 lab2 Intel 上构建，TwoPuncture 走 CPU-OpenMP ~34s 解耦路径）。

## 代码形态

- 语言：CUDA（BSSN GPU 演化）+ C++ host + Fortran TwoPuncture 求解器（解耦后 CPU-OMP）。
- 主热点：`rhs_kernel`（`bssn_rhs_gpu.cu`，单体内核 1075 行），占 GPU kernel 时间 69.1%，8.34 s/step。
- 特性：`__launch_bounds__(256,2)`，128 regs，25% 占用率，latency-bound。
- 26 次跨 TU `__device__` 调用（d_fderivs_point/d_fdderivs_point，diff_new_gpu.cu），需 RDC（`-rdc=true`）。
- L1TEX scoreboard stalls 46.6%，"No Eligible" 64.7%。

## GPU 瓶颈本质

- latency-bound（非 compute/memory-bound）：GPU ~100% 饱和（kernel 12.78s/step ≈ wall 12.31s/step，无 idle gap），H2D/D2H 仅 0.289s/step 可忽略。
- 跨 TU 调用边界使编译器无法跨调用 hoist/reorder load，串行化 L1TEX scoreboard stall。
- shared-mem stencil 单独不可行：15 个场 × halo tile = 120KB > 82KB/block 预算（25% 占用率下）。

## OJ 评测配置

```text
MPI_processes = 1
OMP_threads   = 8
GPU_Calculation = "yes"
Final_Evolution_Time = 100.0
Analysis_Time = 0.1       # 判题机强制
Dissipation   = 0.15
TwoPuncture live（不缓存）
```

- 评分曲线：GPU 满分 120 分（含 20 分 bonus）。100 分点 = 340s。
- 正确性：trajectoryRMS=0（≤0.1%），约束 maxima ≤2，4 个 .dat 全部产出。

## 已部署优化

- TwoPuncture CMake 解耦（`AMSS_ENABLE_TWOP_GPU`）+ OMP_TUNE：TwoP 从 ~268s（部分 GPU 路径）降到 ~34s。
- per-stream sync（`synchronize_all_streams` 替代 cudaDeviceSync）：-30s。
- ABEGPU 源码（bssn_rhs_gpu.cu 等）零改动。

## 当前状态

- 最新 OJ：1044.26s / 64 分（trajectoryRMS=0，约束 maxima ≤2，check.sh PASS）。
- 主瓶颈：rhs_kernel 69.1%（latency-bound，跨 TU 调用 + L1TEX scoreboard stall 46.6%）。
- 已测 rhs 杠杆（split/inline/lb）实测中性或负收益（见 kb/search-memory.md §15.4），待探索新杠杆。

## 路径

- formal 源（只读基线）：`~/lab4-gpu/src`（bssn_rhs_gpu.cu、diff_new_gpu.cu、prolongrestrict_cell_gpu.cu、gpu_manager.cu、bssn_step_gpu.C、bssn_gpu_class.C、CMakeLists.txt）
- 提交包：`~/lab4-gpu`（含 check.sh、golden/、compile.sh、run.sh、src/、scripts/）
- SSH 入口：`bash tmp/sshz.sh lab2 '<cmd>'`
- 剖析用短跑（2 步）采集，别用全量。
