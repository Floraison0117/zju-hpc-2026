#!/usr/bin/env python3
"""Iter38 A38-P44 wiring: 4x4x4 group prolong3 (replaces P33 2x2x2 in the
host dispatch). Bit-exact by construction:
  - per-member Z/Y/X accumulation order replicates prolong3_multi_kernel's
    per-parity statements exactly (Z: 6 sequential += parity-flipped
    coefficients; Y/X: single 6-term expressions);
  - the 4x4x4 group spans anchors [cxI, cxI+1]; interior threshold [3, extc-4]
    (vs P33 [3, extc-3]); the groups shifted to boundary use d_symmetry_bd_1b
    masks whose out-of-range taps are 0, matching the original per-point
    masked semantics bit-exactly;
  - interior TU uses the pure-load path (PROLONG3_INTERIOR) identical to P33.

Files changed:
  src/prolongrestrict_cell_gpu.cu     (add prolong3_multi4_kernel; gpu_prolong3_launch -> 4x4x4)
  src/prolongrestrict_cell_gpu_int.cu (add prolong3_multi4_kernel_int; gpu_prolong3_launch_int -> 4x4x4)
"""
import hashlib, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

ENCF = dict(encoding="utf-8")
CU = f"{ROOT}/src/prolongrestrict_cell_gpu.cu"
CUI = f"{ROOT}/src/prolongrestrict_cell_gpu_int.cu"

cur = sha(CU)
cur_i = sha(CUI)
print(f"guard prolongrestrict={cur} int={cur_i}")
assert cur == "68a65584", f"hash drift {cur}"
assert cur_i == "b739a44e", f"int hash drift {cur_i}"

# ================= 1. boundary TU =================
s = open(CU, **ENCF).read()
assert "prolong3_multi4_kernel" not in s.replace("prolong3_multi4_probe_kernel", ""), "reentry guard"

kernel = r'''
// ---------------------------------------------------------------------------
// A38-P44: 4x4x4 group prolong3 kernel (64 outputs/thread, 9^3 smem cube).
// One thread per 4x4x4 parity-aligned fine group (leaders li=lead+4g with
// (li+lbf) even); the group spans coarse anchors [cxI, cxI+1] per dim and
// shares ONE 9^3 coarse cube (taps [cxI-2, cxI+4]). Per-member Z/Y/X
// accumulation order replicates prolong3_multi_kernel per-parity statements.
// ---------------------------------------------------------------------------
__global__ void prolong3_multi4_kernel(
    int Gi, int Gj, int Gk,
    int lead_base0, int lead_base1, int lead_base2,
    int cxI_base0, int cxI_base1, int cxI_base2,
    int i_start, int i_end, int j_start, int j_end, int k_start, int k_end,
    int extc0, int extc1, int extc2,
    const double* __restrict__ d_src_c,
    int extf0, int extf1, int extf2,
    double* __restrict__ d_dst_f,
    double SoA0, double SoA1, double SoA2,
    int Symmetry, int skip_interior, int compact_mode,
    int lo_i, int hi_i, int lo_j, int hi_j, int lo_k, int hi_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int g_i, g_j, g_k;
#ifndef PROLONG3_INTERIOR
    if (compact_mode) {
        int N = p3_bnd_count(Gi, Gj, Gk, lo_i, hi_i, lo_j, hi_j, lo_k, hi_k);
        if (idx >= N) return;
        p3_bnd_map(idx, Gi, Gj, Gk, lo_i, hi_i, lo_j, hi_j, lo_k, hi_k,
                   &g_i, &g_j, &g_k);
    } else {
        int total = Gi * Gj * Gk;
        if (idx >= total) return;
        g_k = idx / (Gi * Gj);
        int rem = idx % (Gi * Gj);
        g_j = rem / Gi;
        g_i = rem % Gi;
    }
#else
    int total = Gi * Gj * Gk;
    if (idx >= total) return;
    g_k = idx / (Gi * Gj);
    int rem = idx % (Gi * Gj);
    g_j = rem / Gi;
    g_i = rem % Gi;
    (void)compact_mode; (void)lo_i; (void)hi_i; (void)lo_j; (void)hi_j;
    (void)lo_k; (void)hi_k;
#endif

    int li = lead_base0 + 4 * g_i;
    int lj = lead_base1 + 4 * g_j;
    int lk = lead_base2 + 4 * g_k;
    int cxI_i = cxI_base0 + 2 * g_i;   // base anchor (group spans cxI_i..cxI_i+1)
    int cxI_j = cxI_base1 + 2 * g_j;
    int cxI_k = cxI_base2 + 2 * g_k;

#ifdef PROLONG3_INTERIOR
    if (cxI_i < 3 || cxI_i > extc0 - 4 ||
        cxI_j < 3 || cxI_j > extc1 - 4 ||
        cxI_k < 3 || cxI_k > extc2 - 4) return;
#else
    if (skip_interior) {
        if (cxI_i >= 3 && cxI_i <= extc0 - 4 &&
            cxI_j >= 3 && cxI_j <= extc1 - 4 &&
            cxI_k >= 3 && cxI_k <= extc2 - 4) return;
    }
#endif

    int extc[3] = {extc0, extc1, extc2};
    const double SoA[3] = {SoA0, SoA1, SoA2};

    // shared 9^3 coarse cube: taps [cxI-2, cxI+4] per dim
    __shared__ double cube[9][9][9];
    for (int t = threadIdx.x; t < 729; t += blockDim.x) {
        int kk = t / 81;
        int rem2 = t % 81;
        int jj = rem2 / 9;
        int ii = rem2 % 9;
        cube[kk][jj][ii] = d_symmetry_bd_1b(3, extc, d_src_c,
                                            cxI_i - 2 + ii, cxI_j - 2 + jj, cxI_k - 2 + kk, SoA);
    }
    __syncthreads();

    for (int dk = 0; dk < 4; ++dk) {
        int k = lk + dk;
        if (k < k_start || k > k_end) continue;
        for (int dj = 0; dj < 4; ++dj) {
            int j = lj + dj;
            if (j < j_start || j > j_end) continue;
            for (int di = 0; di < 4; ++di) {
                int i = li + di;
                if (i < i_start || i > i_end) continue;
                // Z-pass: 36 columns x 6 taps (member anchors di/2, dj/2, dk/2)
                double zrow[6][6];
                for (int m = 0; m < 6; ++m) {
                    for (int n = 0; n < 6; ++n) {
                        double val = 0.0;
                        if ((dk & 1) == 0) {
                            val += C_PROLONG[0] * cube[dk/2 + 0][dj/2 + m][di/2 + n];
                            val += C_PROLONG[1] * cube[dk/2 + 1][dj/2 + m][di/2 + n];
                            val += C_PROLONG[2] * cube[dk/2 + 2][dj/2 + m][di/2 + n];
                            val += C_PROLONG[3] * cube[dk/2 + 3][dj/2 + m][di/2 + n];
                            val += C_PROLONG[4] * cube[dk/2 + 4][dj/2 + m][di/2 + n];
                            val += C_PROLONG[5] * cube[dk/2 + 5][dj/2 + m][di/2 + n];
                        } else {
                            val += C_PROLONG[5] * cube[dk/2 + 0][dj/2 + m][di/2 + n];
                            val += C_PROLONG[4] * cube[dk/2 + 1][dj/2 + m][di/2 + n];
                            val += C_PROLONG[3] * cube[dk/2 + 2][dj/2 + m][di/2 + n];
                            val += C_PROLONG[2] * cube[dk/2 + 3][dj/2 + m][di/2 + n];
                            val += C_PROLONG[1] * cube[dk/2 + 4][dj/2 + m][di/2 + n];
                            val += C_PROLONG[0] * cube[dk/2 + 5][dj/2 + m][di/2 + n];
                        }
                        zrow[m][n] = val;
                    }
                }
                double tmp1[6];
                for (int n = 0; n < 6; ++n) {
                    double val = 0.0;
                    if ((dj & 1) == 0) {
                        val += C_PROLONG[0] * zrow[0][n] + C_PROLONG[1] * zrow[1][n] +
                               C_PROLONG[2] * zrow[2][n] + C_PROLONG[3] * zrow[3][n] +
                               C_PROLONG[4] * zrow[4][n] + C_PROLONG[5] * zrow[5][n];
                    } else {
                        val += C_PROLONG[5] * zrow[0][n] + C_PROLONG[4] * zrow[1][n] +
                               C_PROLONG[3] * zrow[2][n] + C_PROLONG[2] * zrow[3][n] +
                               C_PROLONG[1] * zrow[4][n] + C_PROLONG[0] * zrow[5][n];
                    }
                    tmp1[n] = val;
                }
                double final_val = 0.0;
                if ((di & 1) == 0) {
                    final_val += C_PROLONG[0] * tmp1[0] + C_PROLONG[1] * tmp1[1] +
                                 C_PROLONG[2] * tmp1[2] + C_PROLONG[3] * tmp1[3] +
                                 C_PROLONG[4] * tmp1[4] + C_PROLONG[5] * tmp1[5];
                } else {
                    final_val += C_PROLONG[5] * tmp1[0] + C_PROLONG[4] * tmp1[1] +
                                 C_PROLONG[3] * tmp1[2] + C_PROLONG[2] * tmp1[3] +
                                 C_PROLONG[1] * tmp1[4] + C_PROLONG[0] * tmp1[5];
                }
                int out_idx = get_col_major_idx(i, j, k, extf0, extf1, extf2);
                d_dst_f[out_idx] = final_val;
            }
        }
    }
}
'''

anchor = "// ++++++++++++++ Kernel Implementation ++++++++++++++"
assert s.count(anchor) == 1
s = s.replace(anchor, kernel + "\n" + anchor)

# host launcher: G4 geometry + launch prolong3_multi4_kernel
old_geom = """    // P33: parity-aligned 2x2x2 group geometry.  Fine indices i, i+1 with
    // (i + lbf) even share the coarse anchor cxI, so each group interpolates
    // 8 fine outputs from ONE 6x6x6 coarse cube.  p3_geom's lbf/lbc mirror
    // the device-side formulas (0.4f) exactly (same reliance as 26c).
    int lbf_p3[3], lbc_p3[3];
    p3_geom(llbc, uubc, extc, llbf, uubf, extf, lbf_p3, lbc_p3);
    int st[3] = {i_start, j_start, k_start};
    int en[3] = {i_end, j_end, k_end};
    int lead_base[3], cxI_base[3], G[3];
    for (int d = 0; d < 3; d++) {
        lead_base[d] = st[d] - ((st[d] + lbf_p3[d]) & 1);
        cxI_base[d] = (lead_base[d] + lbf_p3[d]) / 2 - lbc_p3[d] + 1;
        G[d] = (en[d] - lead_base[d]) / 2 + 1;
    }
    int lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    for (int d = 0; d < 3; d++) {
        for (int g = 0; g < G[d]; g++) {
            int cx = cxI_base[d] + g;
            if (cx <= 2) lo[d]++;
            if (cx >= extc[d] - 2) hi[d]++;
        }
    }
    int compact_mode = 1;
    for (int d = 0; d < 3; d++)
        if (lo[d] + hi[d] > G[d]) compact_mode = 0;

    int block = 256;
    int grid;
    if (compact_mode) {
        int N = p3_bnd_count(G[0], G[1], G[2], lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        if (N == 0) return;
        grid = (N + block - 1) / block;
    } else {
        grid = (G[0] * G[1] * G[2] + block - 1) / block;
    }

    prolong3_multi_kernel<<<grid, block, 0, stream>>>(
        G[0], G[1], G[2],
        lead_base[0], lead_base[1], lead_base[2],
        cxI_base[0], cxI_base[1], cxI_base[2],
        i_start, i_end, j_start, j_end, k_start, k_end,
        extc[0], extc[1], extc[2],
        d_src_c,
        extf[0], extf[1], extf[2],
        d_dst_f,
        SoA[0], SoA[1], SoA[2],
        Symmetry, skip_interior, compact_mode,
        lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]
    );
}"""
new_geom = """    // A38-P44: parity-aligned 4x4x4 group geometry.  Fine indices i..i+3 with
    // (i + lbf) even share coarse anchors [cxI, cxI+1], so each group
    // interpolates 64 fine outputs from ONE 9x9x9 coarse cube (729 taps in
    // smem).  p3_geom's lbf/lbc mirror the device-side formulas exactly.
    int lbf_p3[3], lbc_p3[3];
    p3_geom(llbc, uubc, extc, llbf, uubf, extf, lbf_p3, lbc_p3);
    int st[3] = {i_start, j_start, k_start};
    int en[3] = {i_end, j_end, k_end};
    int lead_base[3], cxI_base[3], G[3];
    for (int d = 0; d < 3; d++) {
        lead_base[d] = st[d] - ((st[d] + lbf_p3[d]) & 1);
        cxI_base[d] = (lead_base[d] + lbf_p3[d]) / 2 - lbc_p3[d] + 1;
        G[d] = (en[d] - lead_base[d]) / 4 + 1;
    }
    int lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    for (int d = 0; d < 3; d++) {
        for (int g = 0; g < G[d]; g++) {
            int cx = cxI_base[d] + 2 * g;
            if (cx <= 2) lo[d]++;
            if (cx >= extc[d] - 3) hi[d]++;
        }
    }
    int compact_mode = 1;
    for (int d = 0; d < 3; d++)
        if (lo[d] + hi[d] > G[d]) compact_mode = 0;

    int block = 256;
    int grid;
    if (compact_mode) {
        int N = p3_bnd_count(G[0], G[1], G[2], lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        if (N == 0) return;
        grid = (N + block - 1) / block;
    } else {
        grid = (G[0] * G[1] * G[2] + block - 1) / block;
    }

    prolong3_multi4_kernel<<<grid, block, 0, stream>>>(
        G[0], G[1], G[2],
        lead_base[0], lead_base[1], lead_base[2],
        cxI_base[0], cxI_base[1], cxI_base[2],
        i_start, i_end, j_start, j_end, k_start, k_end,
        extc[0], extc[1], extc[2],
        d_src_c,
        extf[0], extf[1], extf[2],
        d_dst_f,
        SoA[0], SoA[1], SoA[2],
        Symmetry, skip_interior, compact_mode,
        lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]
    );
}"""
assert s.count(old_geom) == 1, "launcher geometry anchor not found"
s = s.replace(old_geom, new_geom)
open(CU, "w", **ENCF).write(s)
print("prolongrestrict_cell_gpu.cu patched:", sha(CU))

# ================= 2. interior TU =================
s = open(CUI, **ENCF).read()
assert "prolong3_multi4_kernel" not in s, "reentry guard"

kernel_int = r'''
// ---------------------------------------------------------------------------
// A38-P44: 4x4x4 group prolong3 INTERIOR kernel (pure-load path). Same group
// geometry as prolong3_multi4_kernel; interior-only classification [3, extc-4].
// ---------------------------------------------------------------------------
__global__ void prolong3_multi4_kernel_int(
    int Gi, int Gj, int Gk,
    int lead_base0, int lead_base1, int lead_base2,
    int cxI_base0, int cxI_base1, int cxI_base2,
    int i_start, int i_end, int j_start, int j_end, int k_start, int k_end,
    int extc0, int extc1, int extc2,
    const double* __restrict__ d_src_c,
    int extf0, int extf1, int extf2,
    double* __restrict__ d_dst_f,
    double SoA0, double SoA1, double SoA2,
    int Symmetry, int skip_interior, int compact_mode,
    int lo_i, int hi_i, int lo_j, int hi_j, int lo_k, int hi_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int g_i, g_j, g_k;
    int total = Gi * Gj * Gk;
    if (idx >= total) return;
    g_k = idx / (Gi * Gj);
    int rem = idx % (Gi * Gj);
    g_j = rem / Gi;
    g_i = rem % Gi;
    (void)compact_mode; (void)lo_i; (void)hi_i; (void)lo_j; (void)hi_j;
    (void)lo_k; (void)hi_k; (void)skip_interior;

    int li = lead_base0 + 4 * g_i;
    int lj = lead_base1 + 4 * g_j;
    int lk = lead_base2 + 4 * g_k;
    int cxI_i = cxI_base0 + 2 * g_i;
    int cxI_j = cxI_base1 + 2 * g_j;
    int cxI_k = cxI_base2 + 2 * g_k;

    if (cxI_i < 3 || cxI_i > extc0 - 4 ||
        cxI_j < 3 || cxI_j > extc1 - 4 ||
        cxI_k < 3 || cxI_k > extc2 - 4) return;

    int extc[3] = {extc0, extc1, extc2};
    const double SoA[3] = {SoA0, SoA1, SoA2};

    __shared__ double cube[9][9][9];
    for (int t = threadIdx.x; t < 729; t += blockDim.x) {
        int kk = t / 81;
        int rem2 = t % 81;
        int jj = rem2 / 9;
        int ii = rem2 % 9;
        cube[kk][jj][ii] = d_symmetry_bd_1b(3, extc, d_src_c,
                                            cxI_i - 2 + ii, cxI_j - 2 + jj, cxI_k - 2 + kk, SoA);
    }
    __syncthreads();

    for (int dk = 0; dk < 4; ++dk) {
        int k = lk + dk;
        if (k < k_start || k > k_end) continue;
        for (int dj = 0; dj < 4; ++dj) {
            int j = lj + dj;
            if (j < j_start || j > j_end) continue;
            for (int di = 0; di < 4; ++di) {
                int i = li + di;
                if (i < i_start || i > i_end) continue;
                double zrow[6][6];
                for (int m = 0; m < 6; ++m) {
                    for (int n = 0; n < 6; ++n) {
                        double val = 0.0;
                        if ((dk & 1) == 0) {
                            val += C_PROLONG[0] * cube[dk/2 + 0][dj/2 + m][di/2 + n];
                            val += C_PROLONG[1] * cube[dk/2 + 1][dj/2 + m][di/2 + n];
                            val += C_PROLONG[2] * cube[dk/2 + 2][dj/2 + m][di/2 + n];
                            val += C_PROLONG[3] * cube[dk/2 + 3][dj/2 + m][di/2 + n];
                            val += C_PROLONG[4] * cube[dk/2 + 4][dj/2 + m][di/2 + n];
                            val += C_PROLONG[5] * cube[dk/2 + 5][dj/2 + m][di/2 + n];
                        } else {
                            val += C_PROLONG[5] * cube[dk/2 + 0][dj/2 + m][di/2 + n];
                            val += C_PROLONG[4] * cube[dk/2 + 1][dj/2 + m][di/2 + n];
                            val += C_PROLONG[3] * cube[dk/2 + 2][dj/2 + m][di/2 + n];
                            val += C_PROLONG[2] * cube[dk/2 + 3][dj/2 + m][di/2 + n];
                            val += C_PROLONG[1] * cube[dk/2 + 4][dj/2 + m][di/2 + n];
                            val += C_PROLONG[0] * cube[dk/2 + 5][dj/2 + m][di/2 + n];
                        }
                        zrow[m][n] = val;
                    }
                }
                double tmp1[6];
                for (int n = 0; n < 6; ++n) {
                    double val = 0.0;
                    if ((dj & 1) == 0) {
                        val += C_PROLONG[0] * zrow[0][n] + C_PROLONG[1] * zrow[1][n] +
                               C_PROLONG[2] * zrow[2][n] + C_PROLONG[3] * zrow[3][n] +
                               C_PROLONG[4] * zrow[4][n] + C_PROLONG[5] * zrow[5][n];
                    } else {
                        val += C_PROLONG[5] * zrow[0][n] + C_PROLONG[4] * zrow[1][n] +
                               C_PROLONG[3] * zrow[2][n] + C_PROLONG[2] * zrow[3][n] +
                               C_PROLONG[1] * zrow[4][n] + C_PROLONG[0] * zrow[5][n];
                    }
                    tmp1[n] = val;
                }
                double final_val = 0.0;
                if ((di & 1) == 0) {
                    final_val += C_PROLONG[0] * tmp1[0] + C_PROLONG[1] * tmp1[1] +
                                 C_PROLONG[2] * tmp1[2] + C_PROLONG[3] * tmp1[3] +
                                 C_PROLONG[4] * tmp1[4] + C_PROLONG[5] * tmp1[5];
                } else {
                    final_val += C_PROLONG[5] * tmp1[0] + C_PROLONG[4] * tmp1[1] +
                                 C_PROLONG[3] * tmp1[2] + C_PROLONG[2] * tmp1[3] +
                                 C_PROLONG[1] * tmp1[4] + C_PROLONG[0] * tmp1[5];
                }
                int out_idx = get_col_major_idx(i, j, k, extf0, extf1, extf2);
                d_dst_f[out_idx] = final_val;
            }
        }
    }
}
'''
# insert the int kernel after the base TU's multi kernel... find an anchor in the int TU:
# the int TU has "// ++++++++++++++ Kernel Implementation ++++++++++++++" too? check
anchor_i = "// ++++++++++++++ Kernel Implementation ++++++++++++++"
if anchor_i in s:
    s = s.replace(anchor_i, kernel_int + "\n" + anchor_i, 1)
else:
    # fallback: insert before "void gpu_prolong3_launch_int("
    a2 = "void gpu_prolong3_launch_int("
    assert s.count(a2) == 1
    s = s.replace(a2, kernel_int + "\n" + a2, 1)

# int launcher: G4 geometry (mirror of the base; int keeps full-box) + launch
old_gi = """    // P33: parity-aligned 2x2x2 group geometry (mirror of the base launcher;
    // lbf/lbc computed with the device-side 0.4f formulas so the group cxI
    // matches the per-point kernel's cxI bit-exactly).
    int lbf_p3[3], lbc_p3[3];
    for (int d = 0; d < 3; d++) {
        double tv = (llbf[d] - base[d]) / FD[d] + 0.4f;
        double tc = (llbc[d] - base[d]) / CD[d] + 0.4f;
        int v = (fabs(tv) < 1.0) ? 0 : (int)tv;
        int c = (fabs(tc) < 1.0) ? 0 : (int)tc;
        lbf_p3[d] = v + 1;
        lbc_p3[d] = c + 1;
    }
    int st[3] = {i_start, j_start, k_start};
    int en[3] = {i_end, j_end, k_end};
    int lead_base[3], cxI_base[3], G[3];
    for (int d = 0; d < 3; d++) {
        lead_base[d] = st[d] - ((st[d] + lbf_p3[d]) & 1);
        cxI_base[d] = (lead_base[d] + lbf_p3[d]) / 2 - lbc_p3[d] + 1;
        G[d] = (en[d] - lead_base[d]) / 2 + 1;
    }

    int block = 256;
    int grid = (G[0] * G[1] * G[2] + block - 1) / block;

    prolong3_multi_kernel_int<<<grid, block, 0, stream>>>("""
new_gi = """    // A38-P44: 4x4x4 group geometry (mirror of the base launcher; int keeps
    // the full-box enumeration).
    int lbf_p3[3], lbc_p3[3];
    for (int d = 0; d < 3; d++) {
        double tv = (llbf[d] - base[d]) / FD[d] + 0.4f;
        double tc = (llbc[d] - base[d]) / CD[d] + 0.4f;
        int v = (fabs(tv) < 1.0) ? 0 : (int)tv;
        int c = (fabs(tc) < 1.0) ? 0 : (int)tc;
        lbf_p3[d] = v + 1;
        lbc_p3[d] = c + 1;
    }
    int st[3] = {i_start, j_start, k_start};
    int en[3] = {i_end, j_end, k_end};
    int lead_base[3], cxI_base[3], G[3];
    for (int d = 0; d < 3; d++) {
        lead_base[d] = st[d] - ((st[d] + lbf_p3[d]) & 1);
        cxI_base[d] = (lead_base[d] + lbf_p3[d]) / 2 - lbc_p3[d] + 1;
        G[d] = (en[d] - lead_base[d]) / 4 + 1;
    }

    int block = 256;
    int grid = (G[0] * G[1] * G[2] + block - 1) / block;

    prolong3_multi4_kernel_int<<<grid, block, 0, stream>>>("""
assert s.count(old_gi) == 1, "int launcher geometry anchor not found"
s = s.replace(old_gi, new_gi)
open(CUI, "w", **ENCF).write(s)
print("prolongrestrict_cell_gpu_int.cu patched:", sha(CUI))
print("DONE")
