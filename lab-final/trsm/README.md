# 鲲鹏 920F TRSM

## 当前交付版本

正式 `trsm.c` 保留 V9，未接入本轮未达标的自有 SME 内核。V9 的算法是：

- Case1：递归 TRSM，leaf=48，外层 N-split；
- Case2：递归 TRSM，leaf=32，外层 N-split；
- Case3：固定 NB=224，外层 M-split；
- 构造阶段调用 `BlasSetNumThreads(1)`，每个 OpenMP 线程调用单线程 KBLAS。

当前候选感知源码 MD5：`0d518b1e449a1b1db5ff7bdace3fbe1a`；只读 V9 基线 MD5：`d55205c131f25e5f54f1673ea811b77e`，快照位于 `checkpoints/final-v9-20260826/trsm.c`。

## 构建与运行

服务器工作目录为 `~/trsm_work`，节点为 `cn23035`。环境脚本提供 KBLAS 头文件、库和运行时搜索路径。

```bash
# 当前镜像只提供版本化的 KBLAS 文件名；若存在 libkblas.so 链接名可改回 -lkblas。
gcc -O3 -mcpu=native bench_trsm.c trsm.c -o trsm_test -lm -l:libkblas.so.25.2.1 -fopenmp
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=close
export OMP_PLACES=cores
bash run_all.sh
```

`run_all.sh` 保持官方三个规模和 `test_runs=5`，只设置环境并依次调用 bench，没有修改计时、正确性检查或误差阈值。

## 最近正式结果

| Case | M×N | 时间 | GFLOPS | 最大误差 | 校验 |
|---|---:|---:|---:|---:|---|
| 1 | 512×19968 | 8.27 ms | 632.6187 | 6.66e-16 | PASS |
| 2 | 2432×17024 | 81.12 ms | 1241.2676 | 8.88e-16 | PASS |
| 3 | 17024×512 | 134.32 ms | 1104.6839 | 1.83e-15 | PASS |

显示时间总和为 223.71 ms，三个用例总 FLOPs 为 254.311137280 GFLOP，综合约 1136.79 GFLOPS。3000 GFLOPS 目标尚未达到，完整原因和 Gate A/B/C 证据见 [stage_b_report.md](stage_b_report.md) 与 [benchmark_results.md](benchmark_results.md)。

## 硬件能力结论

节点 Features 含 `sve2`、`sme`、`smef64f64`，但 HWCAP2 的 `HWCAP2_SVEF64MM` 为 0，FMMLA 不能在该节点运行。SVE VL 实测为 512 bit。KBLAS DGEMM 反汇编包含 `fmopa za*.d`，使用的是 SME F64F64 路径。

普通 SVE FMLA 38 线程热计算约 1857 GFLOPS。自有 16×32 SME FMOPA 内核在 `16800×512×224`、真实 load/store 下约 1737 GFLOPS，含共享 B-pack 和每线程 A-pack 约 1525 GFLOPS，因此没有接入正式 TRSM。

## 文件与证据

- `trsm.c`：V9 正式实现；
- `bench_trsm.c`：官方 bench，MD5 `1c90748eb757e1fd7bf45605f89e30ef`，未修改；
- `run_all.sh`：官方运行语义；
- `microbench/`：HWCAP、FMLA、SME FMOPA、KBLAS pack 和自有 tile 微基准；
- `benchmark_results.md`：版本表、原始结果摘要和决策；
- `stage_b_report.md`：Gate A/B/C 证据与目标阻塞结论；
- `profile_stage_b.csv`：规范化阶段数据，旧版逐线程原始 profile 保存在 `checkpoints/GateC-20260825-2320/profile_case3.csv`；
- `checkpoints/`：服务器同步下来的环境、哈希、正式运行、perf 和微基准原始文件。

本地工作区没有有效 Git HEAD，因而使用源码 MD5 和 `checkpoints/` 可恢复快照作为版本对应关系。

## SME 候选实现

V9 仍是默认正式路径。新增 SME 实验文件只有在候选构建命令显式加入
`TRSM_SME_PIPELINE` 和 Case 宏后才会链接，普通命令不会依赖这些新符号。

- `sme_update_16x32.S`：独立 SME 汇编，提供 `C[16,32] -= A_pack * B_pack` 的原位内核，以及一次 `smstart` 覆盖多个 N tile 的 persistent batch 内核；store 前加载旧 C，并用向量减法完成原位合并。
- `sme_update.c`：完整 tile 分发到 SME，任意 M/N 尾块和零尺寸输入走正确的标量回退。
- `trsm_sme_candidate.c`：Case3 共享 B-pack 加 M-split、Case2 共享 A-pack 加 N-split；另有显式宏控制的 Case3 left-looking raw-B 更新候选。每个线程拥有 A 或 B 的双槽缓冲，线程只写自己负责的 B2 行区间。
- `microbench/sme_update_test.c`：K=16 到 256、随机/全零/极端尺度、尾块和多线程 correctness 闸门。
- `microbench/sme_pipeline_bench.c`：单缓冲与双缓冲对照，输出 `pack_current_ms`、`pack_next_overlap_ms`、`kernel_ms`、`store_ms`、`wait_ms` 和 `overlap_ratio`，并逐 panel 比较结果。当前 `store_ms` 包含在 `kernel_ms` 内，记为 0 仅表示没有独立硬件计时点。

候选构建和验证只能在具有 SME F64F64 的 aarch64 计算节点运行：

```bash
bash build_sme_microbench.sh
bash run_sme_validation.sh
bash build_sme_candidates.sh
bash run_sme_candidate_cases.sh
```

Case3 和 Case2 也可以分别构建。默认 `build_sme_candidates.sh` 生成
`trsm_test_case3` 和 `trsm_test_case2`，不会覆盖默认的 `trsm_test`。
left-looking Case3 候选使用独立的显式宏构建：

```bash
CFLAGS='-O3 -mcpu=native -fopenmp -DTRSM_SME_LEFT_LOOKING' bash build_sme_candidates.sh
```

该宏只改变候选二进制，默认构建仍不启用 left-looking 或 SME 更新路径。
开启 `TRSM_PROFILE` 时，V9 记录使用 `TRSM_PROFILE_OUT`，候选额外记录使用
`TRSM_SME_PROFILE_OUT`。两者都使用以下统一表头：

```text
case,version,level_or_step,m,n,k,nb,kernel,instruction_path,active_threads,pack_a_ms,pack_b_ms,kernel_ms,gemm_ms,solve_ms,barrier_ms,fork_ms,join_ms,min_thread_ms,max_thread_ms,bytes_packed,flops,gflops
```

双缓冲只有在同一节点上五次交错测量的端到端中位数稳定优于单缓冲至少 3% 时才具备保留依据。候选在完成独立内核、尾块、线程缩放、真实 Case3/Case2 形状和三正式用例的 correctness 前，不得替换 V9。版本记录入口为：

- `checkpoints/V10-sme-inplace/`
- `checkpoints/V11-sme-double-buffer/`
- `checkpoints/V12-case3-shared-bpack/`
- `checkpoints/V13-case2-shared-apack/`
