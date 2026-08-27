# 任务一 ABE CPU 硬件约束 (zju-hpc-arm)

## 机器

- 节点：鲲鹏 920B (TaiShan-v120)，60 物理核，4 NUMA node（各 64 逻辑核/32 物理），node 距离 10-37。
- lscpu flags 含 asimd/sve/svei8mm。支持 NEON + SVE。
- 编译器：GNU Fortran 14.2.0 (`/usr/bin/gfortran`)。armclang/armflang 24.10.1 可用但实测 +3.5% 更慢（见 search-memory）。
- 剖析器：`perf` (stat/record/annotate), `objdump` (aarch64), `gfortran -fopt-info-vec-all`。

## 代码形态

- 语言：MPI Fortran（BSSN 引力波演化）+ 少量 C++。非 CUDA。
- 主热点：`compute_rhs_bssn_`（whole-array F90 风格，~100 条数组语句，~978 行），占步时 30.24%。
- 有限差分：15× fderivs + 11× fdderivs（4 阶 stencil），24× lopsided + 24× kodis（6 阶单侧）。
- 物理：BSSN 方程，80+ 3D 数组（phi/trK/gxx.../Sfx...），AMR 9 层，equatorial symmetry。

## 固有特性（决定下限，不可改）

- load:FP ≈ 3.4:1，IPC 1.69（峰值 ~4.0，57% 发射槽空闲因 stencil 数据依赖，非访存延迟）。
- L1 miss 1.67%，LLC miss ~0.9%，NUMA-local（非带宽/延迟受限）。
- gfortran -Ofast 已对 BSSN 全部 whole-array 赋值（含 Ricci 100+ 项）融合+向量化（16B NEON 2×double）。
- lopsidediff/kodis 的数据依赖符号分支（`if Sfx>0`）不可向量化（控制流固有）。

## OJ 评测配置（唯一可行组合，已实测）

```text
MPI_processes = 30
OMP_threads   = 1          # 30×1=30 ≤ 60；pe=1 映射 30 PE；=2 会 Out of resource
Dissipation   = 0.15       # golden；0.0 导致 check FAIL
Analysis_Time = 1000.0     # 提交值，OJ 强制覆盖为 0.1（每步分析必跑）
Final_Evolution_Time = 40.0
```

- 判题机注入 TwoPuncture env：`OMP_NUM_THREADS=60`（求解器 ~15s，TWOP_OMP_TUNE ON）。
- 评分：wall 时间 + 正确性（轨迹 RMS=0，约束 ≤2）。340s→100 分，500s→60 分。

## 已部署编译 flag（最优）

`-Ofast -fno-frontend-optimize -fno-tree-loop-distribute-patterns -ftree-loop-im`（-3.7%，bit-safe，check.sh PASS）。

## 路径

- formal 源（只读基线）：`~/HPC101/src/lab4-abe-cpu-opt/src`
- 提交包：`~/lab4-cpu`（含 check.sh、golden/、compile.sh、run.sh、src/、scripts/、AMSS_NCKU_Input.py）
- SSH 入口：`bash tmp/sshz.sh arm '<cmd>'`
- GPU 不可用，纯 CPU 任务。
