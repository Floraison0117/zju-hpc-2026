# Iter38 分析 tap-sharing（跨点 coarse cube 共享）设计 + 风险评估

> 状态：**设计完成，未实现**（工程量 >1h，按 supervisor 指引留待部署后攻）。
> 对应迭代 38.6 记录项。目标：global_interp_multi_kernel 的 800 个 MassPAng 大
> launch（grid 144×17, NN=36,864, 46.5ms avg, 37.2s/100 步）在 A38-1 fused-z
> 后仍占分析大头（L2-pipe-bound，global load 216/(point,var) 是主要 wavefront 源）。

## 机制

- 球面插值点数组 d_XX[j]（j = (θ_i, φ_j) 半网格序，36,864 = N_phi/2 × N_theta）中
  相邻点（φ 相邻）的 6³ coarse cube 高度重叠（相邻点 anchor 差 0-1 cell/dim）。
- 4×4 点 tile（θ×φ）：16 点的 anchor 并集 ~4 cell/dim → union cube ~9³ = 729 taps。
- **按 var 循环处理**（smem 只存 1 个 var 的 cube，5.8KB）：per tile 总 load =
  17 × 729 = 12,393 vs 当前 16×17×216 = 58,752 → **4.7× global load 削减**。
- smem 预算 5.8KB/block（A100 99KB 上限，远安全）；occupancy 由 regs 决定。

## 结构（kernel 草图）

```
grid: (tiles, 1) 或 (tiles, vars)，每 tile 16 点 × 17 var
per (tile, var):
  1. 计算 16 点的 anchor 并集窗口 [min_cxI-2, max_cxI+3]（host 或 device 计算）
  2. 729 taps 按 d_symmetry_bd_1b 语义加载到 smem（含反射处理）
  3. __syncthreads
  4. 每点：读自己的 6³ 窗口（smem）→ d_gi_fused 的 polint 链 → atomicAdd
```

## bit-exactness 论证

- 每 tap 值 = f_at_1b(f, ex, sym(i), sym(j), sym(k)) × factors，与逐点
  d_decide3d 的 per-tap 公式一致（union 加载复用同一公式，点在窗口外的 tap 不用）。
- 每点 polint 序逐 token 不变（d_gi_fused）。
- atomicAdd 地址/贡献集合不变（每 (j,var) 恰一贡献）。
- **风险点**：tile 内点的反射窗口不同（近 θ=0/π 或 equatorial 边界的点），
  union 加载必须逐 tap 独立按各点... 不 —— union 按「tap 位置」加载一次
  （tap 位置在哪个点的哪个窗口内由窗口包含关系决定，值只依赖 tap 位置本身 +
  场数据），所以 union 值 = 该位置在任意点窗口内时的值，逐点读取一致。✓

## 风险

1. **tile 几何正确性**：shell 点数组的 (θ,φ) 排布需从 gpu_scale_normals 的
   d_nx/d_ny/d_nz 推导确认（j 的 θ/φ 步长），错误则窗口覆盖不全 → 错值。
2. **union 窗口边界**：tile 边缘点的 anchor 跨度需保守（max-min 上界），
   若 tile 内 anchor 跨度 > 假设（球面曲率）→ 窗口截断 → 错值。需数值验证。
3. **反射语义**：近 equatorial/θ 边界的点窗口含反射 tap，union 加载的
   d_symmetry_bd_1b 必须逐 tap 正确（与 P33/global_interp 同机制，风险中）。
4. **L2 是否仍饱和**：fused-z 后未重新 ncu，若 global load 不再是主导，
   tap-sharing 收益打折（需先 ncu 确认新瓶颈）。
5. 工程量：tile 枚举 + union 加载 + 每点窗口 + var 循环 + A/B/L2 ≈ 2-4h。

## 预期收益

- 4.7× global load 削减 → MassPAng 若仍 L2-bound 则 ~2-4× 提速 →
  analysis 37.2s → 10-18s → **端到端 -10~25s**（任务/ supervisor 估计上限）。

## 结论

设计可行但工程量大 + 4 项中等风险。本轮时间盒内不实现；建议部署 A38-1 +
p44 后，下一轮先跑一次 fused-z 后 ncu（确认新瓶颈是否仍 L2），再决定实现。
