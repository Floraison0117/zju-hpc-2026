#!/usr/bin/env python3
"""
R5 L0 probe: z-rolling window register-floor test on the thin interior kernel.

Hypothesis (milestone-B): the 66-double live floor includes 17 field values
(Lap, chi, gxx..gzz, trK, Axx..Azz). If these are smem-backed (staged once,
re-read at use sites with __syncthreads invalidation boundaries), the register
live floor drops to ~49 doubles (98 regs) -> potential occupancy >25%.

L0 gate (task): interior currently 128 regs / 1836B spill stores / 2548B spill
loads (lb2) and natural 255 regs. If the new structure shows natural regs < 128
OR lb2 spill significantly down -> promising; if spill explodes -> gate kill.

This patch creates bssn_rhs_gpu_int_zr.cu from the deployed interior kernel:
  1. kernel renamed rhs_kernel_int -> rhs_kernel_int_zr
  2. after idx/dims: stage the 17 raw field values into __shared__ s_f[17][256]
     (per-thread slot), __syncthreads
  3. top 17 point-value loads read from smem instead of global
  4. Step-0 metric/Aij reloads (l_gxx..l_gzz, l_Axx..l_Azz) read from smem
  5. __syncthreads() before Step 8 assembly; assembly + constraints re-read
     alpn1/chin1/val_trK/l_gxx..l_gzz/l_Axx..l_Azz fresh from smem

Bit-exact by construction: same raw values, same arithmetic, only the load
source changes (smem staging holds the exact same doubles). Barriers are the
compiler-invalidation mechanism (conservative smem reload after __syncthreads).

NOTE: the probe kernel keeps the interior early-return (boundary threads skip
the barriers -> UB in a real launch spanning the shell). For L0 compile-only
this is fine; a real wire-up must launch with an interior-only grid.

Usage: python3 patch_r5_zroll_l0.py <src_dir> <dst_dir>
"""
import sys, os, hashlib, re

SRC = os.path.join(sys.argv[1], "bssn_rhs_gpu_int.cu")
DST = os.path.join(sys.argv[2], "bssn_rhs_gpu_int_zr.cu")

with open(SRC) as f:
    src = f.read()

h0 = hashlib.sha256(src.encode()).hexdigest()
EXPECT = "8dc0cf28aef2da184ca219e5e1a817073df6320cf85a87f09148489dd44a3839"
if h0 != EXPECT:
    print(f"HASH_GUARD_FAIL src={h0} expected={EXPECT}", file=sys.stderr)
    sys.exit(2)

# ---- 1. rename kernel (both definition and in-file launcher call) ----
assert "void rhs_kernel_int(" in src
src = src.replace("void rhs_kernel_int(", "void rhs_kernel_int_zr(", 1)
assert "rhs_kernel_int<<<grid, block, 0, stream>>>(" in src
src = src.replace("rhs_kernel_int<<<grid, block, 0, stream>>>(", "rhs_kernel_int_zr<<<grid, block, 0, stream>>>(", 1)

# ---- 2. smem staging + barrier after idx/dims ----
anchor = "    int idx = IDX3D(i, j, k, ex0, ex1, ex2);\n    int dims[3] = {ex0, ex1, ex2}; // 用于传给 device 函数"
assert anchor in src, "idx anchor not found"
stage = anchor + """

    // ==========================================
    // R5 probe: z-rolling smem staging of the 17 point field values
    // (register-floor test: field values live in smem, re-read at use sites;
    //  __syncthreads forces conservative smem reload -> shorter live ranges)
    // ==========================================
    __shared__ double s_f[17][256]; // block (8,8,4)=256 threads
    int ltid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    s_f[0][ltid]  = Lap[idx];
    s_f[1][ltid]  = chi[idx];
    s_f[2][ltid]  = dxx[idx];
    s_f[3][ltid]  = gxy[idx];
    s_f[4][ltid]  = gxz[idx];
    s_f[5][ltid]  = dyy[idx];
    s_f[6][ltid]  = gyz[idx];
    s_f[7][ltid]  = dzz[idx];
    s_f[8][ltid]  = trK[idx];
    s_f[9][ltid]  = Axx[idx];
    s_f[10][ltid] = Axy[idx];
    s_f[11][ltid] = Axz[idx];
    s_f[12][ltid] = Ayy[idx];
    s_f[13][ltid] = Ayz[idx];
    s_f[14][ltid] = Azz[idx];
    __syncthreads();
"""
assert src.count(anchor) == 1
src = src.replace(anchor, stage, 1)

# ---- 3. top point-value loads -> smem reads ----
top_old = """    double val_Lap = Lap[idx];
    double val_chi = chi[idx];
    double alpn1 = val_Lap + ONE;
    double chin1 = val_chi + ONE;

    // Metric (dxx 是偏差量，gxx 是物理量 gxx = dxx + 1)
    double val_gxx = dxx[idx] + ONE;
    double val_gxy = gxy[idx];
    double val_gxz = gxz[idx];
    double val_gyy = dyy[idx] + ONE;
    double val_gyz = gyz[idx];
    double val_gzz = dzz[idx] + ONE;

    double val_trK = trK[idx];

    // Extrinsic Curvature Aij
    double val_Axx = Axx[idx]; double val_Axy = Axy[idx]; double val_Axz = Axz[idx];
    double val_Ayy = Ayy[idx]; double val_Ayz = Ayz[idx]; double val_Azz = Azz[idx];"""
assert top_old in src, "top load block not found"
top_new = """    double val_Lap = s_f[0][ltid];
    double val_chi = s_f[1][ltid];
    double alpn1 = s_f[0][ltid] + ONE;
    double chin1 = s_f[1][ltid] + ONE;

    // Metric (dxx 是偏差量，gxx 是物理量 gxx = dxx + 1)
    double val_gxx = s_f[2][ltid] + ONE;
    double val_gxy = s_f[3][ltid];
    double val_gxz = s_f[4][ltid];
    double val_gyy = s_f[5][ltid] + ONE;
    double val_gyz = s_f[6][ltid];
    double val_gzz = s_f[7][ltid] + ONE;

    double val_trK = s_f[8][ltid];

    // Extrinsic Curvature Aij
    double val_Axx = s_f[9][ltid]; double val_Axy = s_f[10][ltid]; double val_Axz = s_f[11][ltid];
    double val_Ayy = s_f[12][ltid]; double val_Ayz = s_f[13][ltid]; double val_Azz = s_f[14][ltid];"""
src = src.replace(top_old, top_new, 1)

# ---- 4. Step-0 metric reloads -> smem reads ----
# l_gxx..l_gzz block
metric_old = """    double l_gxx = dxx[idx] + ONE; double l_gxy = gxy[idx]; double l_gxz = gxz[idx];
    double l_gyy = dyy[idx] + ONE; double l_gyz = gyz[idx]; double l_gzz = dzz[idx] + ONE;"""
assert metric_old in src, "l_gxx block not found"
metric_new = """    double l_gxx = s_f[2][ltid] + ONE; double l_gxy = s_f[3][ltid]; double l_gxz = s_f[4][ltid];
    double l_gyy = s_f[5][ltid] + ONE; double l_gyz = s_f[6][ltid]; double l_gzz = s_f[7][ltid] + ONE;"""
src = src.replace(metric_old, metric_new, 1)

# l_Axx..l_Azz block
aij_old = """    double l_Axx = Axx[idx]; double l_Axy = Axy[idx]; double l_Axz = Axz[idx];
    double l_Ayy = Ayy[idx]; double l_Ayz = Ayz[idx]; double l_Azz = Azz[idx];"""
assert aij_old in src, "l_Aij block not found"
aij_new = """    double l_Axx = s_f[9][ltid]; double l_Axy = s_f[10][ltid]; double l_Axz = s_f[11][ltid];
    double l_Ayy = s_f[12][ltid]; double l_Ayz = s_f[13][ltid]; double l_Azz = s_f[14][ltid];"""
src = src.replace(aij_old, aij_new, 1)

# ---- 5. barrier before Step 8 + fresh smem re-reads for assembly/constraints ----
asm_anchor = """    // ==========================================
    // Step 8: 组装 Aij_rhs & trK_rhs
    // =========================================="""
assert asm_anchor in src, "Step 8 anchor not found"
asm_new = """    // ==========================================
    // R5 probe: smem invalidation barrier + fresh field-value re-reads
    // ==========================================
    __syncthreads();
    alpn1 = s_f[0][ltid] + ONE;
    chin1 = s_f[1][ltid] + ONE;
    val_trK = s_f[8][ltid];
    l_gxx = s_f[2][ltid] + ONE; l_gxy = s_f[3][ltid]; l_gxz = s_f[4][ltid];
    l_gyy = s_f[5][ltid] + ONE; l_gyz = s_f[6][ltid]; l_gzz = s_f[7][ltid] + ONE;
    l_Axx = s_f[9][ltid]; l_Axy = s_f[10][ltid]; l_Axz = s_f[11][ltid];
    l_Ayy = s_f[12][ltid]; l_Ayz = s_f[13][ltid]; l_Azz = s_f[14][ltid];

    // ==========================================
    // Step 8: 组装 Aij_rhs & trK_rhs
    // =========================================="""
src = src.replace(asm_anchor, asm_new, 1)

os.makedirs(sys.argv[2], exist_ok=True)
with open(DST, "w") as f:
    f.write(src)
print(f"OK wrote {DST}")
print(f"sha256(new)={hashlib.sha256(src.encode()).hexdigest()}")
