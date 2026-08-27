# Lab4.5 优化交付说明（v2, cuBLAS INT8 引擎）

## 最终提交文件

- **远端部署路径**：`~/lab4p5/my_int8_fp64.cu`（zju-hpc-lab2 登录节点）
  - md5: `11dad0f0d6e59c4a7392034162ce4e45`
- **本地副本**：`deliverable/lab4p5-v2/my_int8_fp64.cu`（同一 md5）
- **基线快照**：`~/lab4p5/my_int8_fp64.cu.snapshot-73pt-20260827-010010`（md5 `952bfe07...`，即 OJ 73 分版本）

## 改动内容（仅 `my_int8_fp64.cu`，未改任何评测/报告文件）

1. **GEMM 引擎切换为 cuBLAS INT8**（核心改动，一行）：
   `const bool fast = can_fast_gemm(...)` → `false`，使所有 pair GEMM 走
   `cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_N, ... CUDA_R_8I, CUBLAS_COMPUTE_32I)`
   （文件中原有的 fallback 路径，OJ 链接 `-lcublas` 即可用，无 CUTLASS 依赖）。
   H800 MIG 1g.10gb 上单 GEMM 由自研 mma.sync 的 ~45 TFLOPS 提升到 cuBLAS 的
   **~92 TFLOPS**（INT8 张量核接近上限）。
2. 量化/求 maxabs 的向量化微优化（double4 读、uint32 打包写、指针提权），
   实测无显著收益，保留为无害改动。

## 关键实验结论（lab5 = H800 PCIe MIG 1g.10gb, CC 9.0, CUDA 13.3）

- DRAM 带宽 ~247 GB/s；L2 6.4MB；14 SM；INT8 张量核峰值 ~186 TFLOPS。
- cuBLAS INT8 GemmEx：4096³ 1.49 ms/GEMM（92 TFLOPS），8192³ 11.99 ms/GEMM（92 TFLOPS）。
- 自研 mma.sync 内核仅 ~45 TFLOPS（带宽受限 + 无 L2 调度），wgmma 手写版也仅 ~30
  （带宽绑定），故 cuBLAS 是最优选择。
- nsys 分解（4096³ splits=4 单次调用）：GEMM 19.06 ms + 重组 4.39 + 量化 5.28 +
  maxabs 1.64 = 30.4 ms。

## 端到端 A/B（H800，同作业交替编译，见 `AB_val2_H800.log` / `final_validation.log`）

| 规模 | splits | 基线 (73分版) | v2 | 提速 |
|---|---|---|---|---|
| 4096³ | 2 | 15.23 ms | 10.15 ms | 33% |
| 4096³ | 4 | 51.01 | 30.27 | 41% |
| 4096³ | 6 | 98.06 | 57.09 | 42% |
| 4096³ | 8 | 105.38 | 61.66 | 41% |
| 8192³ | 2 | 93.13 | 57.99 | 38% |
| 8192³ | 4 | 354.50 | 204.13 | 42% |
| 8192³ | 6 | 699.59 | 399.95 | 43% |
| 8192³ | 8 | 759.93 | 438.42 | 42% |

- 正确性：8 个组合的 L2 相对误差与基线**逐位一致**
  （2.684e-05 / 3.397e-10 / 5.685e-15 / 2.150e-15 等），INT32 累加顺序无关性保证位级相同。
- `compute-sanitizer --tool memcheck`（1024³ s4）：**0 errors**。
- OJ 同款编译（`nvcc -arch=sm_90a -O3 -std=c++17 -Iinclude -c`，无 CUTLASS）：**rc=0**。

## OJ 分数预估

按 73 分版本 OJ 实测 6 个非封顶检查点的分数曲线反推的评分函数
（g>g60 段 `score = 60 + 40*(e^{1.5t}-1)/(e^{1.5}-1)`，t=(g-g60)/(g100-g60)，6 点全部拟合到 2 位小数），
并计入 OJ 较本地慢 ~2.5% 的偏差：

- 4096³ 侧：s2≈100, s4≈85-87, s6≈76-77, s8≈72 → 侧分 ≈86-87
- 8192³ 侧：s2≈100, s4≈100, s6≈86-89, s8≈80-85 → 侧分 ≈93-94
- **总分预估 ≈ 89-91（保守下限 ~87，上限 ~93）**，远高于 80。

## 证据文件（deliverable/lab4p5-v2/）

- `final_validation.log`：最终部署文件的编译测试 + memcheck + 全量 8 组合 benchmark
- `AB_val2_H800.log`、`AB_ab2_H800.log`：v2 vs 基线 A/B（含漂移对照）
- `probe_and_ab_evidence.log`：H800 带宽/峰值探针 + A/B
- `nsys_breakdown.log`：4096/8192 s4 的内核级分解

## 待办（需要用户操作）

1. 将 `~/lab4p5/my_int8_fp64.cu`（或本地副本）上传到课程 OJ（学在浙大 lab4.5 评测），
   评测机为 H800 PCIe MIG 1g.10gb / CUDA 13.3，与本地 lab5 分区一致。
2. 预期总分 ≈ 89-91（73 → 89+）；提交后把 OJ 报告（scoreBeforeRounding / 各检查点）回传即可核对。
