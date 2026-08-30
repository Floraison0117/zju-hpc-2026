# 鲲鹏 920F TRSM

## 当前交付版本

正式 `trsm.c` 为 V14：在 V9 的形状混合算法之上，将宽 N 更新从纯 N-split 改为 2D 切分。V9 的基础结构保持不变：

- Case1：递归 TRSM，leaf=48，更新走 N-split；
- Case2：递归 TRSM，leaf=32，更新走 N-split；
- Case3：固定 NB=224，更新走 M-split；
- 构造阶段调用 `BlasSetNumThreads(1)`，每个 OpenMP 线程调用单线程 KBLAS。

V14 新增（`TRSM_WIDE_2D`，默认开启）：当 `n > 4096` 且更新行数不低于 `TRSM_2D_MIN_ROWS`（256）时，38 线程按 2 行组 × 19 列组划分，每个线程只调用一次单线程 `cblas_dgemm`。相对纯 N-split，A 面板（L21）的重复读取从 38 份降到 19 份，B 条带宽度翻倍后流式性更好。行数低于阈值的深层递归更新仍走 N-split；Case3 的 M-split 不受影响。

正式源码 MD5：`53e02355131bcc90cff34f12b890bbce`。只读 V9 基线 MD5：`d55205c131f25e5f54f1673ea811b77e`，快照位于服务器 `checkpoints/final-v9-20260826/trsm.c`。

## 构建与运行

服务器工作目录为 `~/trsm_work`。环境脚本提供 KBLAS 头文件、库和运行时搜索路径。

```bash
# 当前镜像只提供版本化的 KBLAS 文件名；若存在 libkblas.so 链接名可改回 -lkblas。
gcc -O3 -mcpu=native bench_trsm.c trsm.c -o trsm_test -lm -l:libkblas.so.25.2.1 -fopenmp
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=close
export OMP_PLACES=cores
bash run_all.sh
```

`run_all.sh` 保持官方三个规模和 `test_runs=5`，只设置环境并依次调用 bench，没有修改计时、正确性检查或误差阈值。`run.sh` 默认每组运行一次，可通过 `TRSM_TEST_RUNS` 调整重复次数。

## 正式结果

V14 官方记录（2026-08-30 02:45，节点 cn22993，`run_all.sh` 语义，test_runs=5，队列空闲窗口，同批 V9 交错对照）：

| Case | M×N | 时间 | GFLOPS | 最大误差 | 校验 |
|---|---:|---:|---:|---:|---|
| 1 | 512×19968 | 8.23 ms | 636.0313 | 6.66e-16 | PASS |
| 2 | 2432×17024 | 77.24 ms | 1303.6841 | 8.88e-16 | PASS |
| 3 | 17024×512 | 134.89 ms | 1100.0665 | 1.83e-15 | PASS |

显示时间总和 220.36 ms，总 FLOPs 254.311137280 GFLOP，综合 1154.1 GFLOPS。同批 V9 对照综合 1123.4 GFLOPS，V14 提升 +2.7%（Case1 +2.9%，Case2 +5.9%，Case3 持平）。

此前 V9 正式记录（2026-08-26，节点 cn23035）为 632.6 / 1241.3 / 1104.7，综合 1136.8。

3000 GFLOPS 目标未达到。已验证的阻塞：节点无 L3（L1d 32KB + L2 768KB 私有），单 NUMA 域 DRAM 带宽约 137 GB/s；KBLAS 更新 GEMM 数据流天花板约 1500 GFLOPS（Case3 形状 M-split 稳态）；按 Case2 与 Case3 的分段下限求和，1800 GFLOPS 综合在 KBLAS 引擎下数学上不可达。完整证据保存在服务器 `~/trsm_work/` 的 `V10_experiment_log.md` 与 `V11-V13_experiment_log.md`。

## 硬件能力结论

节点 Features 含 `sve2`、`sme`、`smef64f64`，但 HWCAP2 的 `HWCAP2_SVEF64MM` 为 0，FMMLA 不能在该节点运行。SVE VL 实测为 512 bit。KBLAS DGEMM 反汇编包含 `fmopa za*.d`，使用的是 SME F64F64 路径。

普通 SVE FMLA 38 线程热计算约 1857 GFLOPS。自有 16×32 SME FMOPA 内核在 `16800×512×224`、真实 load/store 下约 1737 GFLOPS，含共享 B-pack 和每线程 A-pack 约 1525 GFLOPS；与 KBLAS 实际更新速率（profile 开销修正后约 1500 GFLOPS）相比无实质优势，因此未接入正式 TRSM。

## 已否决的结构变体（全部 correctness PASS，同批交错对照负收益）

- k 聚合（4 块一组，k=896 更新）：Case3 -2.5%，B1 面板 L2 驻留被破坏；
- 扁平分块替代递归（宽 N，NB=128~320 扫描）：Case1 -40%，Case2 -35%；
- 持久区域 + 固定列条带（宽 N 整 case 单 parallel region）：Case2 -60%；
- 强制 M-split 用于宽 N：Case2 -66%，瘦行降低 KBLAS 效率；
- 对角求解双行 ILP（`TRSM_SOLVE_ILP=2`）：Case3 -7.0%；
- Case3 solve/update 流水线重叠（2/4/6/8 线程求解组）：最好 -2%，更新组线程饥饿大于重叠收益。

## 文件清单

- `trsm.c`：V14 正式实现；
- `bench_trsm.c`：官方 bench，MD5 `1c90748eb757e1fd7bf45605f89e30ef`，未修改；
- `env.sh`：KBLAS 与 BiSheng 运行时路径；
- `run_all.sh`：官方运行语义（三规模，test_runs=5）；
- `run.sh`：一键编译并运行（默认每组一次）。

V9 基线与完整实验历史（微基准、SME 候选、perf 记录）保存在服务器 `~/trsm_work/checkpoints/` 与本地 git HEAD（commit `46d950a`）。
