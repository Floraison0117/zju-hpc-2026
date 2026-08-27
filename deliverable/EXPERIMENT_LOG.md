# FusedAddRmsNorm 优化实验日志（Ascend C, 910B4）

> 本文件为开发过程的事实性记录（原始数据、命令、diff 摘要、profiling 证据），
> 供撰写课程报告时引用与核实。**不是**报告正文。所有数据均来自实际运行。
>
> 环境：zju-hpc-910（Ascend 910B4, 32GB HBM, CANN 8.5.0, Python 3.11.14），
> NPU 分区 lab3p5。性能口径：`checker/profile.sh`（case 2, 256×1024，
> msprof op --warm-up=10 --launch-count=1，输出单次 Task Duration(us)）。
> 构建：`bash checker/build.sh`；正确性：`hpc submit -p lab3p5 bash checker/run.sh`。

---

## 1. Baseline 数据流（初始代码，`~/lab3p5_backup_20260825_013707`）

### Host（op_host/fused_add_rms_norm.cpp）
- 读 B、H；`alignedHidden = ceil(H/16)*16`；eps 从 attrs 读取。
- Tiling 字段：batchSize, hiddenSize, alignedHidden, alignNum=16, eps。
- **blockDim 无条件 = GetCoreNumAiv()（40）**，与 B 无关。
- 行切分在 kernel 内做：`rowsPerBlock = ceil(B/blockNum)`，每核连续行段。

### Device（op_kernel/fused_add_rms_norm.cpp）
- UB 布局：inQueX/inQueRes（FP16，BUFFER_NUM=2）、outQueY/outQueResOut（FP16×2）、
  weightHalf/weightFp32/resoFp32/sq/scalar/reduceTmp 等 TBuf。
- 每行数据流（whole-row 路径，H≤4096）：
  1. DataCopyPad 搬 x、residual（byte 粒度，尾块 0 padding）；
  2. Cast→FP32，Add 得 R32（同时写 residual_out：Cast FP16→MTE3）；
  3. Mul 平方 + BlockReduceSum/WholeReduceSum（FP32 规约，mask COUNTER=H）；
  4. `GetValue` 取标量 → `meanPlusEps = sumSq/H + eps`（标量）→
     Duplicate + Sqrt + Div + Mul(weight)（全行向量）→ Cast FP16 → MTE3。
- 同步：CopyIn/CopyOut 内部大量 `PipeBarrier<PIPE_ALL>` 与 V_MTE3 flag；
  TQue 虽 BUFFER_NUM=2，但 PIPE_ALL 使 MTE2/V/MTE3 跨行完全串行。
- H>4096 走两遍流式 chunked 路径。

### Baseline 正确性：5/5 pass。
### Baseline 性能（case 2，5 次顺序）：15.1012, 14.7812, 15.0412, 15.2812, 15.1200 µs
中位数 **15.1012**，min 14.7812，max 15.2812。

### Baseline msprof（op_prof_baseline, aic-metrics=Default）
- Task Duration 14.98 µs；Block Dim 40；核 0-35 各 7 行，核 36 仅 4 行，
  核 37-39 空转（aiv_time ≈2.6 µs 纯启动）。
- 流水占比（中位）：vec ~19-23%（2.72 µs），scalar ~37-51%（4.7-7.1 µs），
  MTE2 ~35-42%（4.2-5.5 µs，活跃带宽 5-7 GB/s），MTE3 ~14-16%（1.9 µs）。
- 无任何流水占满 → 串行化/同步/负载不均为主瓶颈；L2 命中率 6.6%；
  GM↔UB 带宽利用率 ~2%（指令/延迟受限，非带宽受限）。

---

## 2. 优化版本迭代（一次一个主要因素）

协议：改 → build → 5 case 正确性 → case 2 计时 ≥5 次 →（必要时）profile →
接受或回退。时间序列均为 `hpc submit -p lab3p5 bash checker/profile.sh` 顺序运行。

| 版本 | 主要修改 | 5 case | case2 样本(µs) | 中位 | 相对 baseline |
|---|---|---|---|---|---|
| baseline | 初始代码 | 5/5 | 15.10,14.78,15.04,15.28,15.12 | 15.10 | 1.00× |
| V1 | q/r 行切分 + blockDim=min(AIV,B) | 5/5 | 15.64,15.60,14.70,15.12,15.00 | 15.12 | 1.00×（无变化） |
| V2 | 去掉 CopyIn/Out 的 PIPE_ALL 与 V_MTE3，TQue 双缓冲生效 | 5/5 | 9.92,10.06,10.58,9.92,9.58 | 9.92 | 1.52× |
| V2b | 去掉 whole-row 冗余 PIPE_V（编译器 auto-sync 本就插入） | 5/5 | 9.52,10.26,9.74,9.76,10.40 | 9.76 | 1.55× |
| V3 | 对齐 H%16==0 用快速 DataCopy，非对齐保留 DataCopyPad | 5/5 | 9.68,9.34,10.18,10.20,9.56 | 9.68 | 1.56× |
| V4 | **批量 chunk 路径**：chunk 级多块 DataCopy、R 留 UB、逐行规约写 sumSq 数组、chunk 级一次 V_S 同步 | 5/5 | 8.86,7.62,7.60,7.60,7.86 | 7.62 | 1.98× |
| V5 | **rstd 全标量**：`rstd = 1/sqrt(sumSq*invH+eps)`（标量 sqrt 内建），省 3 个 tiny 向量 op/行 | 5/5 | 7.76,7.04,6.98,6.98,7.94 | 7.04 | 2.15× |
| V6 | rowsPerChunk 上限 8→3（更多 chunk） | 5/5 | 7.80,8.12,7.66,7.54,9.06 | 7.80 | 回退（更差） |
| V7 | 向量 Rsqrt 替代标量 sqrt+div | 5/5（但误差大） | — | — | **精度失败，拒绝** |
| V8 | 软件流水（提前发下一 chunk MTE2、y 预乘 weight、批量 rstd） | 5/5 | 见下 | 7.46-7.58 | 回退（更差） |
| E-C | resOut MTE3 提前到 phase A 后 | 5/5 | 7.46,6.94,6.96,7.32,7.20 | 7.20 | 噪声级，未采纳 |
| **最终** | = V5 | 5/5 | 见 §4 | **7.60（16 样本）** | **1.97×** |

关键发现（profiling 驱动）：
- V1 无提升 → 瓶颈不是负载均衡。
- V2 一次 -34% → PIPE_ALL 串行化是主因，TQue EnQue/DeQue 提供正确的跨流水同步。
- V2b/V3 边际 → PIPE_V 由编译器 auto-sync 覆盖；DataCopyPad→DataCopy 非瓶颈。
- V4 再 -21% → 逐行队列 API 是 scalar 主要开销。simulator 显示单行（H=4096）
  约 852 条 SCALAR 指令（队列/地址/掩码 setup），其中 ~85% 为标量开销。
- V5 再 -8% → 标量 sqrt 内建可用（`float sqrt(float)`，__builtin_cce_sqrtf），
  rstd 计算移出 V 链。
- V7 拒绝：910B 向量 Rsqrt 精度 ~0.2-0.5%（error_ratio 0.19-0.47），
  与 baseline 注释一致。
- V6/V8 回退：更多 chunk 的每-chunk 同步/API 开销 > MTE2/V 重叠收益
  （case 2 每核 7 行、chunk=7 时 1 个 chunk 最优；MTE2 2.2 µs > V 1.8 µs，
  切小 chunk 只能部分重叠）。

### 各版本 msprof 中位值（op_prof_*/PipeUtilization.csv, aiv 各列 us）

| 版本 | aiv_time | vec | scalar | MTE2 | MTE3 | Task(us) |
|---|---|---|---|---|---|---|
| baseline | 12-14 | 2.7 | 4.7-7.1 | 4.2-5.5 | 1.9 | 14.98 |
| V2 | 7.18 | 2.32 | 4.31 | 2.79 | 1.70 | 9.92 |
| V2b | 7.40 | 2.14 | 4.65 | 2.97 | 1.72 | 9.82 |
| V3 | 7.15 | 2.19 | 4.17 | 3.12 | 1.70 | 9.52 |
| V4 | 5.96 | 1.81 | 2.62 | 2.20 | 0.24 | 8.10 |
| V5(最终) | 5.22 | 1.56 | 2.62 | 1.81 | 0.24 | 7.42 |

MTE3 从 1.7 µs 降到 0.24 µs（100 GB/s）是 chunk 级大拷贝的收益；
scalar 从 4.2-4.7 降到 2.6 µs 是批量路径的收益。

### 启动开销探针（诊断，未提交）
- 空 kernel（跳过 Init+Process，blockDim=40）：Task 4.00 µs，per-core aiv_time
  min/med/max = 0.58/1.79/3.04 µs。
- 仅 Init（含 InitBuffer×8）：Task 4.52 µs（Init 约 +0.5 µs）。
- blockDim=1 空 kernel：Task 2.48 µs → 每块分派 ~38 ns。
- 结论：~2 µs 任务分派/完成 + ~1.8 µs kernel 入口为固定开销，占最终耗时大头；
  工作区 ~3.2 µs（scalar 2.62 为主）。

---

## 3. 最终实现设计（= V5，src/ascendc/）

### 文件
- `op_host/fused_add_rms_norm.cpp`：tiling 增加 rowsPerChunk；blockDim=min(AIV,B)
  （AIV 数用 platform API 查询，不硬编码 40）；rowsPerChunk 由 UB 预算推导：
  `144KB / (20*alignedHidden)`，上限 8，仅对齐 H≤4096 生效。
- `op_kernel/fused_add_rms_norm_tiling.h`：+rowsPerChunk 字段。
- `op_kernel/fused_add_rms_norm.cpp`：三条路径。
- `extension/custom_op.cpp`、`setup.py`、`CMakeLists.txt`、`build_op.sh`：未改动。

### 数据流（对齐批量路径 ProcessAlignedBatch，H%16==0 且 H≤4096）
```
GM x/residual →(1 条多块 DataCopy, blockCount=nRows, blockLen=H/16) → inQue(双缓冲, 2×n×H FP16)
  → per row: Cast(x)→R32; Cast(res)→sq; Add→R32; Cast→residual_out(FP16 存 chunk buf);
             Mul(sq=R²); BlockReduceSum×2+WholeReduceSum → sumSqArr[row*8]（32B 对齐 lane）
  → 1 次 SetFlag/WaitFlag(V_S)（整 chunk 一次同步）
  → per row: scalar rstd = 1/sqrt(sumSqArr*invH + eps)（标量 sqrt 内建）
  → per row: Muls(R, rstd); Mul(R, weight); Cast→y(FP16)
  → residual_out、y 各 1 条多块 DataCopy 写回 GM
```
- 规约分母：`invH = 1/H`（真实 H），reduce 用 mask COUNTER=H，padding 恒 0；
  非对齐路径（case 4）走 per-row DataCopyPad，zero-pad + mask 排除尾块。
- 精度：全部 FP32；rstd 为标量 1/sqrt(mean+eps)（约 1-2 ULP），y 误差受
  FP16 舍入边界翻转限制，最大相对误差 ≤ 9.77e-4 < 1e-3（任意幅值，见 §5）。
- UB 预算（H=1024, chunk=7）：inQue 2×28KB + outQueResOut/outQueY 2×2×14KB +
  rFp32 28KB + sq 4KB + sumSq 64B + weight 6KB ≈ 150KB < 192KB。

### 同步设计
- TQue EnQue/DeQue 提供 MTE2→V 与 V→MTE3 同步（含 buffer 复用保护，
  enQueEvt/freeBufEvt，见 tikcfw kernel_tpipe.h）；
- 每 chunk 一次 V_S（规约→标量），无 PIPE_ALL；编译器 auto-sync 处理 V 内 RAW。

---

## 4. 最终性能验证

### 最终版本 case 2 全部样本（16 次，顺序）
7.72, 6.96, 6.78, 7.80, 7.42, 7.52, 7.84, 7.58, 7.48, 7.66, 7.62, 7.66, 7.54,
7.48, 7.98, 7.84
→ 排序：6.78, 6.96, 7.42, 7.48, 7.48, 7.52, 7.54, 7.58, 7.62, 7.66, 7.66,
7.72, 7.80, 7.84, 7.84, 7.98
→ 中位 **7.60** µs，min 6.78，max 7.98（3 次为干净树重建后复测）。

### Baseline 同条件复测（从只读备份恢复原代码，3 次）
14.7212, 14.5812, 14.9812 → 中位 14.72 µs。
原始 baseline 5 次：15.10, 14.78, 15.04, 15.28, 15.12 → 中位 15.10 µs。
合计 baseline 8 样本：14.58-15.28，中位 15.01 µs。

### 加速比
- 中位对中位：15.01 / 7.60 = **1.97×**（同条件复测 14.72/7.60 = 1.94×）。
- 最保守口径：baseline 最优 14.58 / 最终最差 7.98 = 1.83×。
- 结论：稳定、可复现的提升（最终全部样本 < 8 µs，baseline 全部样本 ≥ 14.5 µs）。

### 正确性（最终版本，`bash checker/run.sh`）
```
case_0 32x4096     y=pass residual_out=pass
case_1 256x1024    y=pass residual_out=pass
case_2 1x4096      y=pass residual_out=pass
case_3 1997x3037   y=pass residual_out=pass
case_4 2048x4096   y=pass residual_out=pass
```

### 隐藏 shape 防护测试（scratch/hidden_test.py，22 例，全部 PASS）
- B=1 × H=15/16/17/31/32/33/4095/4097（32B 边界、尾块、B=1 单核）
- H=1023/1024/1025；B=32/40/41/250（相对 40 AIV；B 不可整除）
- B=512×H=8192、B=16×H=16384（chunked 两遍路径）、B=4096×H=512
- 数据范围 L（±1000）：256×1024、1997×3037
- H=3、H=8（极小）
- L 范围最大相对误差 9.7e-4 < 1e-3（残差加法/输出均为 0 误差）。
- 理论：y 的 FP32 计算与 golden 差异 ~1e-7 相对，只会触发 1 个 FP16 ULP 的
  舍入翻转，最大相对误差 = 9.77e-4（在 2 的幂边界处），恒小于 1e-3；
  容限的绝对/相对双判据进一步兜底。

### 双缓冲流水证据（simulator）
- 单行 case 3（1×4096）：`msprof op simulator --soc-version=Ascend910B4` 4.04 µs。
- 多 chunk shape（scratch/sim_shape.py, 560×1024, 每核 14 行、2 chunks）：
  core0 指令流显示 `MOV_OUT_TO_UB`（chunk2 输入, 1047 cyc）与
  `MOV_UB_TO_OUT`（chunk1 resOut+y, 1047+685 cyc）并发执行
  → **chunk 级 MTE2/MTE3 确已重叠**。
- 诚实说明：case 2（256×1024）每核 7 行 = 1 个 chunk，无跨 chunk 重叠；
  其加速来源是消除逐行队列 API/同步开销（标量指令 ~852/行 → ~400/行），
  而非双缓冲。双缓冲在多 chunk shape（如 case 5、560×1024）实际生效。

---

## 5. 采纳与放弃的优化

采纳：
1. 去除 PIPE_ALL 串行化，依赖 TQue 队列同步（-34%）。
2. q/r 行切分 + blockDim 上限（消除空转核；单独测量无提升，但为必要正确性）。
3. 对齐快速 DataCopy（小收益，减少 MTE2 指令）。
4. 批量 chunk 路径：chunk 级大拷贝、R 留 UB、chunk 级一次 V_S 同步（-21%）。
5. 标量 rstd（-8%）：`1/sqrt(sumSq*invH+eps)`，标量 sqrt 内建。

放弃（附原因）：
- 向量 Rsqrt：精度 0.2-0.5%，5 case 中 4 个 FAIL。
- rowsPerChunk=3（更多 chunk）：同步/API 开销 > 重叠收益（7.80 vs 7.04）。
- 软件流水 + y 预乘 weight（V8）：7.46-7.58，比 V5 慢。
- resOut MTE3 提前（E-C）：噪声级，未采纳。

---

## 6. 复现命令

```bash
# 正确性（5 公开 case）
hpc submit -p lab3p5 bash checker/run.sh
# 性能（case 2, 单次 Task Duration）
hpc submit -p lab3p5 bash checker/profile.sh
# 详细 profiling（scratch 包装，不改 checker）
hpc submit -p lab3p5 bash scratch/prof_detail.sh op_prof_xxx --aic-metrics=Default
# simulator（scratch 包装）
hpc submit -p lab3p5 bash scratch/sim.sh 3
SIM_B=560 hpc submit -p lab3p5 bash scratch/sim2.sh
# 隐藏 shape 测试（scratch，独立于 checker）
hpc submit -p lab3p5 bash scratch/run_hidden.sh
```

最终提交目录：`~/lab3p5/src/ascendc/`（仅 op_host/、op_kernel/、extension/、
common/、CMakeLists.txt、CMakePresets.json、build_op.sh、setup.py）。

未修改：checker/、env.sh、golden 逻辑、输入生成、计时逻辑；
未硬编码任何输入/输出/shape；未调用任何高层 RMSNorm 算子。

## 7. 残余风险
- 性能噪声：同版本 13 次样本 6.78-7.84 µs（±7%），受集群共租/温度影响；
  报告应使用中位数并说明范围。
- 隐藏 case：非对齐大 H（>4096 且非 16 倍数）走两遍 chunked 路径，
  该路径未经逐元素压力测试（仅 16×16384 对齐测试）；建议正式评测前
  再补测如 7×5003、1×65537 等。
- L 范围数据最大相对误差接近 1e-3（9.7e-4），理论有界但余量 ~3%。
- 启动/分派开销 ~3.7 µs 为固定成本，进一步优化空间有限。
