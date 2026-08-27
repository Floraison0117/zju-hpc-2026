# Lab4.5 优化交付说明

## 当前候选

- 本地源文件：`deliverable/lab4p5-v2/my_int8_fp64.cu`
- 远端正式路径：`~/lab4p5/my_int8_fp64.cu`
- 当前 MD5：`09017b60ca8c00ce2c33f78c0f17db90`
- 部署前快照：`~/lab4p5/my_int8_fp64.cu.snapshot-pre-codex-20260827-095800`
- OJ：未提交，以下是课程计算分区上的 OJ 配置代理测试

## 实现

候选直接使用 CUDA 13.3 cuBLAS 的 FP64 fixed-point emulation 接口。函数每次调用都会：

1. 为 cuBLAS 设置可复用的 workspace；
2. 选择 eager、fixed mantissa 控制；
3. 将 `splits` 映射为 `min(8 * splits, 55)` 个 mantissa bits；
4. 调用 `cublasGemmEx` 完成列主序 FP64 GEMM。

相比此前手动量化、逐 pair INT8 GEMM 和 FP64 重组的版本，这条路径保留了完整的 `splits` 请求，并由 cuBLAS 内部融合量化、INT8 GEMM 与重组，避免了手动实现中的 pair 裁剪和大量 kernel launch。

## OJ 配置代理验证

使用远端 `lab5` H800 MIG 1g.10gb 分区，按官方 `sm_90a -O3 -std=c++17 -Iinclude` 参数 clean build，运行：

```text
hpc submit -p lab5 -g 1 "cd ~/codex-lab4p5-emu-20260827 && export PATH=/usr/local/cuda/bin:$PATH && make clean && make && ./benchmark 4096,8192 2,4,6,8 3 --csv"
```

作业 `177618` 退出码为 0。页面定义的满分阈值为 splits=2/4/6/8 对应 `10000/5000/3000/3000 GFLOPS`，本次结果如下：

| 矩阵 | splits=2 | splits=4 | splits=6 | splits=8 |
|---|---:|---:|---:|---:|
| 4096³ GFLOPS | 10121.15 | 5528.52 | 3264.52 | 3258.52 |
| 8192³ GFLOPS | 12966.55 | 5947.05 | 3328.07 | 3324.25 |

对应的 L2 relative error 为：

| 矩阵 | splits=2 | splits=4 | splits=6 | splits=8 |
|---|---:|---:|---:|---:|
| 4096³ | 5.395e-7 | 9.705e-12 | 2.140e-15 | 2.140e-15 |
| 8192³ | 5.398e-7 | 9.712e-12 | 3.019e-15 | 3.019e-15 |

8 个组合全部超过满分吞吐阈值，且误差与同一 benchmark 中的 `cublas_emulated` 参考实现一致。因此按课程页面给出的评分公式，代理测试为 100/100；这不是 OJ 官方提交成绩。

## 环境备注

远端分区查询显示 `lab4g10` 是 A100 MIG，而课程 Lab4.5 页面描述的 OJ 代理环境是 H800 MIG。A100 候选测试曾因用户的 A100 作业配额未释放而未能重新排队，未将其结果写入验收证据。H800 候选已完成 clean build 和两种官方矩阵规模的全套测试。
