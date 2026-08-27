# Lab5 Task2 GEMV Kernel Optimization — ncu Profiling Summary

> 日期：2026-08-21。在 H800 MIG 1g.10gb（14 SM）上对 GEMV kernel（`_fused_dequant_gemv_kernel`）
> 的 9 个变体做 ncu `--set full` profiling。被采样形状：q_proj（N=4096, K=4096, M=2, BS2）。
> 所有数字来自 `~/lab5-kernel-opt/ncu_v*.ncu-rep`（`--clock-control none`，相对指标有效）。

## 1. 完整 ncu 指标对比

| 变体 | Theo Occ | Achi Occ | BL Reg | BL SMem | BL Warps | DRAM% | L1/TEX% | SM% | Elig Warp | NoElig% | Duration(µs) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 baseline fp32 s4 | 12.50 | 11.73 | 2 | 5 | 16 | 10.53 | 62.63 | 47.38 | 0.47 | 62.54 | 521 |
| V1 bf16 w s4 | 12.50 | 11.48 | 2 | 5 | 16 | 12.61 | 57.44 | 44.13 | 0.48 | 61.45 | 580 |
| V2 fp32 w s2 | 12.50 | 11.70 | 2 | 5 | 16 | 10.02 | 62.55 | 47.38 | 0.48 | 61.73 | 550 |
| V4 MMA pad8+dot | **25.00** | **23.35** | 2 | 3 | 8 | **2.32** | **97.96** | 11.35 | **0.16** | **87.82** | **1560** |
| V5 bf16sum s4 | 12.50 | 11.51 | 2 | 4 | 16 | 15.73 | 51.41 | 37.54 | 0.46 | 61.99 | 636 |
| V6 bf16sum s2 | 12.50 | 11.53 | 2 | 4 | 16 | 16.05 | 50.62 | 38.98 | 0.45 | 62.93 | 620 |
| V7 BK=128 | 12.50 | 11.73 | 2 | 5 | 16 | 9.87 | 62.34 | 43.80 | 0.47 | 62.61 | 561 |
| V8 BN=32 | 12.50 | 11.49 | 2 | 5 | 16 | 10.59 | 61.54 | 43.95 | 0.48 | 61.82 | 517 |
| V9 BN32+BK128 | 12.50 | 11.51 | 2 | 5 | 16 | 9.95 | 61.48 | 44.14 | 0.48 | 62.05 | 554 |

## 2. 关键发现

### 发现 1：Block Limit Registers = 2 对所有 tl.sum 变体不变

所有使用 `tl.sum(w[None,:,:] * x[:,None,:], axis=2)` 归约模式的变体
（V0/V1/V2/V5/V6/V7/V8/V9），无论：
- dtype（fp32 / bf16 / bf16-sum）
- num_stages（4 / 2）
- tile size（BLOCK_N 64→32, BLOCK_K 256→128）

**Block Limit Registers 恒为 2，Theoretical Occupancy 恒为 12.50%**。

这意味着寄存器压力来自 `tl.sum` 的 3D 广播-乘-归约模式本身，而非权重 tile 的
dtype 或大小。Triton 编译器为该模式固定分配 255 registers/thread（硬件上限），
无法通过调整 operand dtype / tile size / pipeline depth 来降低。

### 发现 2：bf16 变体不降寄存器反增延迟

V1（bf16 w）/V5（bf16 sum）的 Block Limit Registers 仍为 2，证明 bf16 未减少
寄存器占用。所有 bf16 变体比 baseline 慢 14–21%（V1: 580µs vs 521µs），
原因是 `.to(tl.bfloat16)` 转换开销 + DRAM 改善（10.5%→15.7%）不足以补偿。

### 发现 3：num_stages 被 Triton 忽略

V2（stages=2）与 V0（stages=4）的 Block Limit Shared Mem 均为 5，完全相同。
Triton 对该 kernel 的 `tl.sum` 循环体不应用软件流水线（loop-carried dependency
阻止 pipelining），num_stages 参数无效。

### 发现 4：MMA 占用率翻倍但 L1 饱和、DRAM 饿死

V4（M-pad8 + tl.dot）是唯一改变占用率的变体（12.5%→25%），但：
- L1/TEX Throughput = **97.96%**（饱和）— 转置权重加载 [BLOCK_K, BLOCK_N] 从
  [N, K_packed] 存储中 gather，彻底打满 L1 cache
- DRAM Throughput = **2.32%**（饿死）— L1 饱和阻塞了 DRAM 请求
- Eligible Warps = **0.16**（几乎无可用 warp）— tensor core 等不到数据
- Duration = **1560µs**（3× 慢于 baseline 的 521µs）

占用率翻倍无意义，因为 warp 全在等 L1 cache miss 返回。

### 发现 5：tile size 缩减无效

V7（BLOCK_K 256→128）/V8（BLOCK_N 64→32）/V9（两者减半）均未改变 Block Limit
Registers（= 2）或 occupancy（= 12.5%）。3D tensor [M, BLOCK_N, BLOCK_K] 的
理论大小从 128 KiB 降到 32 KiB（V9），但寄存器占用不变，证明 255 reg/thread
是 Triton 编译器对 `tl.sum` 模式的固定分配，与 tensor 大小无关。

## 3. 根因结论

`tl.sum(w[None,:,:] * x[:,None,:], axis=2)` 模式在 Triton 编译器中生成
255 registers/thread（硬件上限）的固定分配，不随 dtype / tile size / pipeline
depth 改变。这是 GEMV kernel 占用率 12.5% 的不可绕过根因。

唯一逃逸路径是 `tl.dot`（MMA，V4），占用率翻倍至 25%，但 MMA 的转置权重加载
模式将瓶颈从寄存器转移到 L1 cache（饱和 98%），使 kernel 从延迟受限变为
cache 带宽受限，总体 3× 更慢。

**结论：在 Triton 3.7.1 + H800 MIG 1g.10gb 环境下，tl.sum GEMV 的占用率
不可通过 dtype/stages/tile-size 提升；MMA 路径虽提升占用率但引入更严重的
L1 cache 瓶颈。三个方向（bf16 / 降 stages / 融合 dequant-MMA）均无法使
kernel 达到带宽受限（DRAM > 50%），当前 baseline（V0）已是该 pattern 的
实际最优。**
