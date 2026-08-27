# Stage B 报告：920F TRSM 能力闸门与目标可达性

## 结论

本轮目标 3000 GFLOPS 未达成。当前正式版本 V9 三例全 PASS，但综合约 1137--1147 GFLOPS。阻塞不是 FMMLA 编译器探针不足，而是节点运行时明确没有 `SVEF64MM`；可用的普通 SVE FMLA 全核约 1857 GFLOPS。节点有 SME F64F64，KBLAS 也使用 SME `fmopa`，但独立自有 16×32 内核在包含大矩阵 load/store 和 pack 后约 1525 GFLOPS，未形成可替换正式 TRSM 的 3000 GFLOPS 路径。

## Gate A：硬件与指令

节点 `cn23035`：

```text
Features: ... sve ... sve2 ... sme smef64f64 ...
AT_HWCAP2=0xfeb8f3ff
HWCAP2_SVEF64MM=0x800
header_defined=1
has_svef64mm=0
svcntb=64
svcntd=8
```

因此 SVE VL 是 512 bit，但 `HWCAP2_SVEF64MM` 位未置位。GCC 10.3.1 接受以下编译命令并能生成 FMMLA 反汇编：

```text
gcc -O2 -march=armv8.6-a+sve+f64mm -msve-vector-bits=512 fmmla_probe.c -o fmmla_explicit
```

反汇编证据为 `fmmla z0.d, z1.d, z2.d`。这不能推翻运行时 HWCAP 结果，FMMLA 二进制未执行。`-mcpu=native` 下的 ACLE probe 被 GCC 拒绝，错误指出 `svmmla_f64` 需要 `f64mm`。

SME 的 `.inst` probe 运行输出 `sme_f64f64_probe_inst=ok`。旧汇编器不能直接识别 SME mnemonic，但可以接受对应机器码。完整编译错误与二进制保存在 `checkpoints/GateA-20260825-2313/`。

## Gate B：独立上限

普通 SVE FMLA 使用：

```text
gcc -O3 -march=armv8.6-a+sve -msve-vector-bits=512 -fopenmp sve_fma_bench.c -o fmla_bench -lm
```

38 线程热计算中位数：4.864 GFLOP 总量每次重复的累计等价为约 `1857.2 GFLOPS`，1/2/4/8/16/24/32/38 线程完整结果见 `checkpoints/GateA-20260825-2313/fmla_scaling.log`。

SME 热 FMOPA 循环 38 线程约 `7455 GFLOPS`，但不含真实 load/store。8×8 packed tile 在 K=256、38 线程约 `1668 GFLOPS`。随后实现并标量校验了 16×32 packed tile，`K=256` 的热缓存结果约 `6648 GFLOPS`，但完整 `16800×512×224` 形状为：

| 测试 | 时间 | GFLOPS |
|---|---:|---:|
| 16×32，预 pack，真实 load/store | 2.218754 ms | 1736.793 |
| 16×32，共享 B pack，每线程 A pack | 2.527006 ms | 1524.934 |
| KBLAS public packed compute，共享 B pack | 7.847481 ms | 491.051 |

自有 tile 的首 tile 标量参考误差为 `2.22e-16`，K=16/32/48/64/96 的 Case1 形状含 pack 结果为 209/331/385/500/658 GFLOPS。K=64/96/128/160/192/224/256 的 Case3 形状含 pack 结果为 746/1191/1312/1420/1469/1525/1488 GFLOPS。原始日志在 `checkpoints/GateB-20260826-16x32/`。

## Gate C：KBLAS 路径与硬件计数

KBLAS `dgemm_kernel_nn` 反汇编包含 `smstart`、`fmopa za*.d`、`smstop`，没有 FMMLA。Case3 `perf record` 的热点包含：

- `cblas_dgemm/dgemm_nn` 约 17.23%；
- `dgemm_oncopy` 约 8.73%；
- `dgemm_itcopy` 约 8.19%。

完整热点见 `checkpoints/GateC-20260825-2320/perf_report_case3.txt`。

V9 的 Case3 `perf stat -d -r 3` 代表值约为 25.66B cycles、32.90B instructions、IPC 1.28、L1D miss 35.67%、LLC miss 66.92%，没有观察到迁移或上下文切换。自有 16×32 大矩阵实验约为 1.658B cycles、2.460B instructions、IPC 1.48、周期频率约 1.43 GHz、L1D miss 16.76%、LLC miss 37.09%。

这些数据说明：SME 热循环的算力上限不能直接外推到正式 TRSM，正式形状的主要差异来自 packed panel 访问、tile 间复用、C 写回以及每 tile 调度。

## 正式安全闸门

正式默认路径仍保留 V9，因此未把尚未完成节点闸门的 SME 候选放入正式正确性矩阵。当前新增的 `sme_update_16x32.S`、`sme_update.c` 和 `trsm_sme_candidate.c` 只通过候选构建入口启用；独立测试覆盖完整 tile、尾块、随机/全零/极端尺度和多线程，鲲鹏计算节点的指令运行结果仍需补录。现有正式 V9 已通过：

- `bash run_all.sh`，三个官方规模，`test_runs=5`；
- 三例全部 PASS；
- 最大误差 `1.83e-15`；
- `bench_trsm.c` MD5 `1c90748eb757e1fd7bf45605f89e30ef` 未变化。

正式输出见 `checkpoints/V9-official-run_all-20260826.log`，V9 与 persistent candidate 的完整五次原始记录分别见 `checkpoints/V9-local-20260825-2300/` 和 `checkpoints/PERSISTENT-local-20260825-2302/`。

## 下一条可验证路径

如果继续追求 3000，需要继续验证已经加入的 SME 融合更新候选，而不是继续微调 fork/barrier：候选在一个持久 SME 区域内复用 B panel，原位加载并减去 C，并为 Case2/Case3 分别设计 M/N 方向的 pack 生命周期。必须先完成完整 tile、尾块、随机矩阵、极端对角值和三正式用例的 correctness，再接入默认正式路径。
