# TRSM 优化实验结果

## 最终状态

当前最快且全 PASS 的正式实现仍是 V9，V9 可恢复快照位于 `checkpoints/final-v9-20260826/`。本轮新增的 SME 原位更新、persistent batch、双缓冲测量和 Case2/Case3 候选均保持宏控，尚未替换默认正式路径，也没有新增鲲鹏计算节点吞吐结果。目标 3000 GFLOPS 未达到，已有 Gate A/B/C 证据表明：FMMLA 路径在本节点不可执行，普通 FMLA 全核上限低于目标；可执行的 SME F64F64 自有内核在包含大矩阵数据流与 pack 后仍低于 3000。

## 环境与不变量

| 项目 | 实测值 |
|---|---|
| 节点 | `cn23035`，队列 `q_kunpeng`，Donau |
| CPU 拓扑 | 608 CPUs，32 个 NUMA 节点，每节点 38 核；`numactl -N 1` 对应 CPU 38--75 |
| 线程绑定 | `OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores` |
| 编译器 | GCC 10.3.1，`gcc -O3 -mcpu=native` |
| KBLAS | `/home/share/shenchao_common/kblas/libkblas.so.25.2.1` |
| KBLAS 线程 | 构造阶段 `BlasSetNumThreads(1)`，早于第一次 KBLAS 调用 |
| bench 哈希 | `bench_trsm.c` MD5 `1c90748eb757e1fd7bf45605f89e30ef`，全程未修改 |
| Git | 本地 `.git` 目录为空，没有可读取的 HEAD 或工作树状态 |

正式构建：

```text
gcc -O3 -mcpu=native bench_trsm.c trsm.c -o trsm_test -lm -lkblas -fopenmp
```

正式脚本：`bash run_all.sh`，脚本只调用官方三个规模，每例 `test_runs=5`，未改计时、误差阈值或问题规模。

## 正式版本表

| Version | 主要变化 | Case1 GFLOPS | Case2 GFLOPS | Case3 GFLOPS | 综合 GFLOPS | 最大误差 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| V9 | 递归 leaf=48/32，Case3 NB=224，外层分块，KBLAS 单线程 | 629.0 | 1251.0 | 1116.4 | 1147 | 1.83e-15 | 全 PASS |
| Persistent candidate | 保持 V9 算法，Case3 使用 persistent team | 624.3 | 1236.1 | 1134.6 | 约 1152 | 1.83e-15 | 候选，未默认启用 |
| V9 rerun 2026-08-26 | `bash run_all.sh`，当前正式文件 | 632.6 | 1241.3 | 1104.7 | 约 1137 | 1.83e-15 | 全 PASS |

最近一次 `run_all.sh` 原始输出：

```text
512 x 19968          8.27 ms   632.6187 GFLOPS   6.66e-16   PASS
2432 x 17024        81.12 ms  1241.2676 GFLOPS   8.88e-16   PASS
17024 x 512        134.32 ms  1104.6839 GFLOPS   1.83e-15   PASS
```

按显示时间计算：总时间 `223.71 ms`，总 FLOPs `254.311137280 GFLOP`，综合 `1136.79 GFLOPS`。原始文件见 `checkpoints/V9-official-run_all-20260826.log`；V9 五次统计原始文件见 `checkpoints/V9-local-20260825-2300/official_test_runs5.log`。

## Gate A

节点 `/proc/cpuinfo` 的 Features 包含 `sve sve2 ... sme smef64f64`，不包含 `svef64mm`。程序输出：

```text
AT_HWCAP2=0xfeb8f3ff
HWCAP2_SVEF64MM=0x800
header_defined=1
has_svef64mm=0
```

`svcntb=64`、`svcntd=8`，确认 SVE VL 为 512 bit。GCC 10.3.1 接受 `-march=armv8.6-a+sve+f64mm -msve-vector-bits=512` 的 FMMLA 探针，反汇编出现：

```text
fmmla z0.d, z1.d, z2.d
```

这只是编译器和汇编器能力证据，不是运行能力证据。`-mcpu=native` 下使用 `svmmla_f64` 被 GCC 拒绝，报错为 ACLE 函数需要 `f64mm`。由于 HWCAP2 明确为 0，本轮没有执行 FMMLA 二进制，也没有在正式源码中使用它。

SME 探针用旧汇编器可接受的 `.inst` 编码生成 `smstart`、`fmopa`、`smstop`，运行结果为 `sme_f64f64_probe_inst=ok`。KBLAS `dgemm_kernel_nn` 反汇编包含 `smstart`、大量 `fmopa za*.d` 和 `smstop`，未发现 FMMLA，确认 KBLAS 走 SME F64F64 路径。

## Gate B

### 普通 SVE FMLA

编译命令：

```text
gcc -O3 -march=armv8.6-a+sve -msve-vector-bits=512 -fopenmp sve_fma_bench.c -o fmla_bench -lm
```

反汇编确认热点含 `fmla z*.d, p0/m, ...`。热缓存、8 个累加器、每线程 1,000,000 次、5 次取中位数的 38 线程结果为 `1857.2 GFLOPS`。完整扩展性见 `checkpoints/GateA-20260825-2313/fmla_scaling.log`。因此普通 FMLA 不能作为 3000 GFLOPS 的充分路径。

### SME 热计算与真实数据流

SME 热循环 38 线程约 `7455 GFLOPS`，但它不含真实 load/store。自有 16×32 FMOPA 内核使用 K×16、K×32 packed A/B，标量参考校验通过，`K=256`、38 线程、5 次取中位数约 `6648 GFLOPS`，仍属于热缓存单 tile。

完整 `m=16800,n=512,k=224` 的 16×32 tile 实验，源码在 `microbench/`，每个 tile 包含真实 A/B load、FMOPA 和 C store：

| 路径 | 中位时间 | GFLOPS | 正确性 |
|---|---:|---:|---|
| 预 pack，单 tile 调用 | 2.218754 ms | 1736.793 | 首 tile 误差 2.22e-16 |
| 共享 B pack，线程内 A pack | 2.527006 ms | 1524.934 | 首 tile 误差 2.22e-16 |
| 批量进入一次 SME | 4.251507 ms | 906.388 | 小规模 tile 通过，未进入正式路径 |

K 扫描的含数据流 GFLOPS：

| K | 64 | 96 | 128 | 160 | 192 | 224 | 256 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| GFLOPS | 1181 | 1452 | 1457 | 1531 | 1640 | 1737 | 1806 |

Case1 形状 `m=512,n=19968` 的含数据流结果在 K=16/32/48/64/96 时为 209/331/385/500/658 GFLOPS，说明小 K 和窄 M 仍受数据搬运与调用开销限制。原始日志分别为 `checkpoints/GateB-20260826-16x32/k_sweep_full.log` 和 `k_sweep_case1.log`。

### KBLAS pack 路径

`cblas_dgemm_pack` 探针确认 A、B 都必须按 KBLAS 布局 pack，raw A + packed B 的结果错误。对 `16800×512×224`，标准 M-split 中位约 `1372.3 GFLOPS`；共享 B pack 的 KBLAS packed compute 端到端约 `491.1 GFLOPS`，因此没有接入共享 B-pack + KBLAS 路径。

## Gate C

V9 的 `perf stat -d -r 3` 是完整 bench 进程统计，包含矩阵生成和正确性准备，不等同于纯计时区间。Case3 代表值：约 25.66B cycles、32.90B instructions、IPC 1.28、L1D miss 35.67%、LLC miss 66.92%，无迁移和上下文切换。Case1/2 原始数据见 `checkpoints/GateC-20260825-2320/perf_case1.log`、`perf_case2.log`、`perf_case3.log`。

Case3 `perf record` 的热点为 `cblas_dgemm/dgemm_nn` 17.23%、`dgemm_oncopy` 8.73%、`dgemm_itcopy` 8.19%，并看到 `solve_diag_block` 和输入矩阵生成。KBLAS 反汇编已证明其 DGEMM 热点为 SME `fmopa`，不是 FMMLA。

自有 16×32 大矩阵实验的 `perf stat` 代表值：约 1.658B cycles、2.460B instructions、IPC 1.48、平均周期频率约 1.43 GHz，L1D load miss 16.76%，LLC load miss 37.09%。这与“热计算远高于真实数据流”的差异一致。

## 决策

- 保留 V9 为默认正式版本，persistent team 仅保留候选。
- 默认不接入自有 SME 内核：新增候选已具备独立 tile/尾块测试和正式候选构建入口，但目标计算节点上的完整 correctness、双缓冲 3% 闸门和正式形状含 pack 吞吐仍待执行。
- 不继续微调 fork/barrier；现有 profile 中 Case3 fork/barrier 不是达到目标的主因，且本轮主要瓶颈已经转为 SME 数据流和 tile 调度。
- 下一条可验证路径是更大范围的 SME 融合更新内核，需同时解决 B panel 的跨 tile 复用、C 的原位累加和 tail 处理；在没有完整随机/边界/正式三例 correctness 之前，不应改写正式 `trsm.c`。

所有 Gate 原始结果保存在 `checkpoints/`，规范化阶段数据在 `profile_stage_b.csv`。
