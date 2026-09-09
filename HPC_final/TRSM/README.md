# 鲲鹏 920F TRSM

## 当前交付版本

正式 `trsm.c` 为 V16：在 V14 的基础上，将叶子块对角求解（串行前代消元链）替换为 W-technique（选择性求逆）。V14 的结构保持不变：

- Case1：递归 TRSM，leaf=48，更新走 N-split；
- Case2：递归 TRSM，leaf=32，更新走 N-split；
- Case3：固定 NB=224，更新走 M-split；
- 构造阶段调用 `BlasSetNumThreads(1)`，每个 OpenMP 线程调用单线程 KBLAS。

V14 的宽 N 二维切分（`TRSM_WIDE_2D`，默认开启）：当 `n > 4096` 且更新行数不低于 `TRSM_2D_MIN_ROWS`（256）时，38 线程按 2 行组 × 19 列组划分，每个线程只调用一次单线程 `cblas_dgemm`。相对纯 N-split，A 面板（L21）的重复读取从 38 份降到 19 份。

V16 新增（`TRSM_WTECH`，默认开启）：叶子块（bs ≤ 64）求解改为先以 O(bs³/6) 串行求逆得到 `inv(L_jj)`（转置存储，内层为连续流），再用原地 `cblas_dtrmm` 按 N 分片计算 `B_j := inv(L_jj) × B_j`。前代消元的行串行链被完全消除，求解速率从标量约 110 GF 提升到 dtrmm 480-860 tri-GF（探针实测：(32,17024) 481、(48,19968) 834）。Case3 的 bs=224 块保持标量路径（大块求逆串行成本不划算，宏守卫限制 bs ≤ 64）。

正式源码 MD5：`e05bbf44346eab099c70ba962f9bd1f1`。V9 基线快照位于服务器 `checkpoints/final-v9-20260826/trsm.c`（md5 d55205c131f25e5f54f1673ea811b77e），V14 版本（md5 53e02355131bcc90cff34f12b890bbce）在实验日志中可追溯。

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

V16 官方记录（2026-08-30，节点 cn22994，`run_all.sh` 语义，test_runs=5，同批 V14 交错对照确认窗口干净）：

| Case | M×N | 时间 | GFLOPS | 最大误差 | 校验 |
|---|---:|---:|---:|---:|---|
| 1 | 512×19968 | 6.02 ms | 869.7710 | 2.78e-16 | PASS |
| 2 | 2432×17024 | 71.08 ms | 1416.5514 | 4.44e-16 | PASS |
| 3 | 17024×512 | 138.03 ms | 1075.0646 | 1.83e-15 | PASS |

显示时间总和 215.13 ms，总 FLOPs 254.311137280 GFLOP，综合 1182.1 GFLOPS。同批 V14 对照综合约 1123 GFLOPS（Case1 614.8、Case2 1229.6、Case3 1025.9），V16 相对 V14：Case1 +41%、Case2 +15%、Case3 +4.8%，综合 +5.1%。

版本演进：V9 综合 1136.8（2026-08-26，cn23035）→ V14 综合 1154.1（2026-08-30，cn22993）→ V16 综合 1182.1（2026-08-30，cn22994）。

3000 GFLOPS 目标未达到。已验证的阻塞：节点无 L3（L1d 32KB + L2 768KB 私有），单 NUMA 域 DRAM 带宽约 137 GB/s；KBLAS 更新 GEMM 在 Case2 相关形状族上的历史最佳速率为 1884 GFLOPS（1824×17024×304，2×19 切分），按此上限推算 Case2 端到端天花板约 1841 GFLOPS（全部更新按最佳速率 + 求解归零），Case3 更新段单独就超出共锥所需预算，1800/2000/3000 GFLOPS 目标在 KBLAS 引擎下数学上不可达。完整证据保存在服务器 `~/trsm_work/` 的 `V10_experiment_log.md` 与 `V11-V13_experiment_log.md`。

## 硬件能力结论

节点 Features 含 `sve2`、`sme`、`smef64f64`，但 HWCAP2 的 `HWCAP2_SVEF64MM` 为 0，FMMLA 不能在该节点运行。SVE VL 实测为 512 bit。KBLAS DGEMM 反汇编包含 `fmopa za*.d`，使用的是 SME F64F64 路径。

普通 SVE FMLA 38 线程热计算约 1857 GFLOPS。自有 16×32 SME FMOPA 内核在 `16800×512×224`、真实 load/store 下约 1737 GFLOPS，含共享 B-pack 和每线程 A-pack 约 1525 GFLOPS；与 KBLAS 实际更新速率（profile 开销修正后约 1500 GFLOPS）相比无实质优势，因此未接入正式 TRSM。

## 已否决的结构变体（全部 correctness PASS，同批交错对照负收益）

- k 聚合（4 块一组，k=896 更新）：Case3 -2.5%，B1 面板 L2 驻留被破坏；
- 扁平分块替代递归（宽 N，NB=128~320 扫描）：Case1 -40%，Case2 -35%；
- 持久区域 + 固定列条带（宽 N 整 case 单 parallel region）：Case2 -60%；
- 强制 M-split 用于宽 N：Case2 -66%，瘦行降低 KBLAS 效率；
- 对角求解双行 ILP（`TRSM_SOLVE_ILP=2`）：Case3 -7.0%；
- Case3 solve/update 流水线重叠（2/4/6/8 线程求解组）：最好 -2%，更新组线程饥饿大于重叠收益；
- W-technique 经 dgemm + 临时缓冲（非 dtrmm）：缓冲往返流量（3 倍块流量）吃掉全部计算收益；
- W-technique 下 leaf=64：Case2 1433→1032 GFLOPS；
- 扁平 NB=608 重构：探针显示与树结构同速率族，无增益；
- left-looking 更新族：1399-1672 GF，全面劣于右视 1884。

## 文件清单

- `trsm.c`：V16 正式实现（形状混合递归/分块 + 宽 N 二维切分 + W-technique 叶子求解）；
- `bench_trsm.c`：官方 bench，MD5 `1c90748eb757e1fd7bf45605f89e30ef`，未修改；
- `env.sh`：KBLAS 与 BiSheng 运行时路径；
- `run_all.sh`：官方运行语义（三规模，test_runs=5）；
- `run.sh`：一键编译并运行（默认每组一次）。

V9 基线与完整实验历史（微基准、SME 候选、perf 记录、V10-V16 全部变体及否决证据）保存在服务器 `~/trsm_work/checkpoints/` 与本地 git HEAD（commit `46d950a`）。
