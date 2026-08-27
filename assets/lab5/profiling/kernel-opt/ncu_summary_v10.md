# Lab5 Task2 V10 Kernel Optimization — Profiling Summary

> 日期：2026-08-22。在 H800 MIG 1g.10gb（14 SM）上对 V10 coalesced-MMA kernel
> （`_fused_dequant_mma_kernel`，转置 [K_packed,N] qweight）做 ncu `--set full` profiling。
> 被采样形状：q_proj（N=4096, K=4096, M=2, BS2）。配置 BN=32, BK=128, num_warps=2。

## 1. 关键指标（ncu --set full, q_proj M=2）

| 指标 | V10 (coalesced MMA) | V0 GEMV (对比) |
|---|---|---|
| Achieved Occupancy | **30.99%** | 12.50% |
| Theoretical Occupancy | 37.50% | 12.50% |
| Block Limit | Shared Mem (6) + Warps (16) | Registers (2, 255 reg/thread) |
| Achieved Active Warps/SM | 19.83 | 8 |
| DRAM Active / Elapsed | 57630 / 1205760 = **4.8%** | (延迟受限，离带宽上限 ~55×) |
| L1 Active / Elapsed | 219076 / 3394244 = **6.5%** | 98% (V4 dot 旧版) |
| L2 Active / Elapsed | 220441 / 2405000 = 9.2% | — |
| Uncoalesced Shared Access | 655360 excessive wavefronts (37%) | — |

## 2. 发现

### V10 突破：占用率翻倍 + 非 memory-bound
- V10（转置 qweight [K_packed,N] + tl.dot MMA）将 Achieved Occupancy 从 GEMV 的 12.5%
  提升到 **31%**（理论 37.5%）。Block Limit 从 Registers（255 reg/thread，GEMV 的
  `tl.sum` 3D 广播模式固定分配）变为 Shared Mem + Warps。
- DRAM 利用率仅 4.8%、L1 仅 6.5% —— kernel **非 memory-bound**，而是
  occupancy + shared-mem 访问效率受限。

### V10 的瓶颈：Uncoalesced Shared Access (37%)
- ncu 报告 655360 excessive wavefronts（占总 1788928 的 37%）来自 shared memory
  非合并访问。这是 tl.dot MMA 操作数在 shared memory 中的布局问题（Triton 编译器层面），
  无法通过 dtype/tile-size/num_stages 调参解决。

### 配置扫描结果（num_warps=2 最优）
- BN=32, BK=128, num_warps=**2** → 89.5 ms/step（336 calls, M=2）
  vs num_warps=4 的 97.2 ms/step（8% 提速）。
- BN=48/96 触发 CompilationError（非 2 的幂）。
- 全 step 估算：89.5 ms × 160 = 14.3s decode + 16s prefill ≈ 30s（理论上限）。

## 3. 端到端实测（OJ 等价路径，BS2, async+resident, CUDA Graph）

| 配置 | elapsed_s | tok/s | 正确性 | 峰值显存 |
|---|---|---|---|---|
| V0 GEMV baseline（当前 OJ） | 83.5 | 3.83 | — | 8.09 GiB |
| V10 + graph（dev, warmed） | **42.2** | 7.58 | 320/320 (100%) | 8.53 GiB |
| V10 + graph（cold, OJ 等价单次） | 46.7 | 6.84 | 320/320 (100%) | 8.53 GiB |

- **warmed**（dev 两段：no-graph 预热后 graph）= 42.2s
- **cold**（OJ 单次 generate_continuous）= 46.7s，差 4.5s 为 cuBLAS prefill autotune
  首次开销（5 个 prompt 长度 × ~0.9s）；dummy F.linear 预热无效（neutral），
  全 prefill 预热反而因 allocator 碎片化变慢（53.8s）。

## 4. < 40s 的阻断点

V10 kernel 已到 Triton 编译器对该 pattern 的实际上限：
1. **Occupancy 31%**（理论 37.5%），受 Shared Mem + Warps 限制，非 memory-bound
   （DRAM 4.8%）。无法通过 Triton 调参继续提升。
2. **37% Uncoalesced Shared Access** —— tl.dot 的 shared-mem 布局问题，
   需手写 CUDA（Marlin 风格 fused dequant-MMA + 合并 weight layout）才能突破。
3. cuBLAS prefill（19.7s）已最优；冷启动 autotune 4.5s 无法用 dummy 预热恢复。
4. per-step Python 开销 ~5s（compact tensor fill + sync + done-check）需 graph-loop
   重构（捕获整个 decode 循环而非单步）。

**结论**：V10 + CUDA Graph 将 OJ elapsed 从 83.5s 降至 46.7s（cold）/ 42.2s（warmed），
task2Score 从 62.86 提升到约 91.0（cold）/ 94.9（warmed），100% token-exact 正确。
距 < 40s（S₂≈96.7）还差 ~3-7s，阻断于 Triton kernel 的 occupancy + shared-mem 天花板，
需手写 CUDA kernel（Marlin 风格）才能继续突破。
