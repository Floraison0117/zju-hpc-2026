# ZGEMM 大作业提交（HPC101 · 深圳超算鲲鹏 920）

**题目**：优化复数双精度矩阵乘 `cblas_zgemm`（`C = α·A·B + β·C`），仅允许修改 `zgemm*.c`。

**最终成绩**：官方三用例合计 **3125.28 GFLOPS**（1009.93 / 1071.82 / 1043.53，全部 PASS，
误差 2.5e-12~4.3e-12，容差 1e-10）。

**目录结构**：

```text
ZGEMM/
├── README.md                 # 本文档
├── run.sh                    # 运行脚本（SVE 检测/回退 + NUMA 绑定纪律）
├── bench_zgemm.c             # 官方基准，原封未改（md5 a6cf1a3b）
├── zgemm.c                   # FCMLA(NEON) 回退实现（zju 阶段调优后的版本）
├── zgemm_nc192.c             # ★ 最终提交实现：SVE MR12 微内核 + NUMA 行分区 + NC=192
├── versions/                 # 优化历程中的关键版本（仅存档供查阅）
│   ├── zgemm_original_backup.c  # 最原始朴素实现（baseline，三重循环 + OpenMP）
│   ├── zgemm_sve12.c         #   SVE MR12 + NC=512（2077 GFLOPS）
│   ├── zgemm_numa.c          #   NUMA 行分区 v4 + NC=512（NC 调参前的版本）
│   ├── zgemm_str1.c          #   1 级 Strassen（实测慢 6.1 倍，已否决，仅存档）
│   └── zgemm_str2.c          #   2 级 Strassen（实测慢 22.9 倍，已否决，仅存档）
└── results/                  # 运行结果存档
    ├── final_run_output.txt  #   最终判题式复测的原始终端输出（全文）
    ├── final_results.txt     #   各阶段成绩文字存档
    └── final_results.png     #   最终成绩截图
```

---

## 1. 编译

官方判题编译命令（**固定，不能加 `-march`**）：

```bash
gcc -O3 bench_zgemm.c zgemm_nc192.c -o zgemm_test -lm -fopenmp
```

要点：

- SVE 指令集在 `zgemm_nc192.c` 内通过 `#pragma GCC target("arch=armv8.2-a+sve")` 打开，
  因此无需（也不允许）在命令行加 `-march`。
- 需要 GCC ≥ 12（SVE intrinsics）与支持 SVE 的硬件（鲲鹏 920，VL=512）。
- 若在**无 SVE 的 aarch64 机器**（如 zju 920B）上编译运行，改用 FCMLA 回退实现：
  ```bash
  gcc -O3 bench_zgemm.c zgemm.c -o zgemm_test -lm -fopenmp
  ```
  （`zgemm.c` 使用 NEON `vcmlaq_f64`，任何 ARMv8.3 以上 CPU 可跑，但性能约为 SVE 版的 1/3。）

## 2. 直接运行

```bash
./zgemm_test M N K reps        # reps = 每个用例重复次数（官方判题用 3）
```

官方三用例（判题即这三条，每条 3 次取均值）：

```bash
./zgemm_test 7427  7427  256  3     # 用例 1
./zgemm_test 14848 14848 256  3     # 用例 2
./zgemm_test 37360 8192  512  3     # 用例 3
```

输出格式：`M N K | 平均耗时 ms | GFLOPS | 最大绝对误差 | PASS/FAIL`
（GFLOPS = 8·M·N·K / 时间；误差容差 tol = 1e-10）。

## 3. 集群提交（推荐方式：直接跑 run.sh）

`run.sh` 会自动完成全部环境处理，直接执行即可：

```bash
cd /home/share/wangmingxiang/zgemm_submit
dsub -q q_kunpeng -R 'cpu=608,mem=64GB' -nn 1 -T 3600 -o OUT.txt bash run.sh
djob -l          # 轮询到 SUCCEEDED
cat OUT.txt      # 查看三用例成绩
```

`run.sh` 内部逻辑（与最终成绩直接相关的关键点）：

1. **工具链**：`module load gcc/compiler12.3.1`（GCC 12.3.1，SVE intrinsics 可用）。
2. **源选择与降级链**：检测 `/proc/cpuinfo` 的 `sve` 标志 → 编译 `zgemm_nc192.c`；
   SVE 编译失败或 8×8×8 冒烟测试 SIGILL → 自动回退 FCMLA 版 `zgemm.c`。打印 `using <src>`。
3. **线程绑定纪律**：`OMP_NUM_THREADS` 取 pinned 进程的 nproc（上限 38），
   `OMP_PROC_BIND=close OMP_PLACES=cores`，并 `unset GOMP_CPU_AFFINITY`（防 HPCKit 覆盖）。
4. **NUMA first-touch 对齐**：`taskset -c node0` 把**整个进程**（含 bench 的矩阵初始化）
   限制在 NUMA 节点 0，保证所有页面本地。
5. 依次跑官方三用例，每条 3 次。

> 关键结论（详见提交根目录总报告 `report.pdf` 题目二部分）：不绑核时线程在 32 个 NUMA 节点间迁移，微内核的
> L1/L2 面板缓存全失效，性能只有 ~10 GFLOPS；绑定纪律带来 37 倍差距。

## 4. 复现检查

```bash
md5sum run.sh bench_zgemm.c zgemm.c zgemm_nc192.c
# run.sh         6e776d90ab0bc9a85e67351b87916dcd
# bench_zgemm.c  a6cf1a3b89a2cfe1114676ac95a51c8e   （官方原封）
# zgemm.c        8b7bb4d93a21d24c1996d414eb54843b
# zgemm_nc192.c  ae6843a2b8cbe62769f47864fa960e05
```

## 5. 关键调参（源码内宏，`zgemm_nc192.c` 顶部）

| 宏 | 取值 | 含义 |
|---|---|---|
| `MR` | 12 | 微内核行数（24 个累加器 z0–z23 占满寄存器预算） |
| `NR` | 8 | 微内核列数（SVE 512-bit = 4 个复数/向量 × 2 组） |
| `KC` | 128 | k 分块（pack 面板深度） |
| `MC` | 384 | 行方向宏块 |
| `NC` | 192 | 列方向宏块（512→192 带来 2077→3073 的跳跃，分析见总报告 `report.pdf`「NUMA 行分区与 NC 收窄」一节） |

`NC=192` 对官方 K=256/512 形状是实测平台高点：tile 数 ×2.6，尾部均衡，
packed B（192 列 = 384 KB）在 L2 流转更顺。
