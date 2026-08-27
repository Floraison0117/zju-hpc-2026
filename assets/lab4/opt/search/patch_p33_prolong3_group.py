#!/usr/bin/env python3
# P33: prolong3 group multi-output rewrite (ABEGPU.md untested #8/#9).
#
# Mechanism: fine indices i, i+1 with (i+lbf) even share the same coarse
# anchor cxI (floor(ii/2) identical for ii, ii+1).  A parity-aligned
# 2x2x2 group of fine points therefore interpolates from ONE 6x6x6 coarse
# cube: the per-point kernel loads the cube 8 times (8 x 216 = 1728 loads per
# group); the group kernel loads it once (216 loads, 8x fewer global loads),
# shares both k-parity Z-rows in tmp2[2][6][6] (local), and emits all 8
# members.  Per member the accumulation order is EXACTLY the original:
#   Z: val += C[ord] * tap  (t ascending; ord flips with k parity)
#   Y: val += C[...] * tmp2[0..5][n] + ...  (single 6-term expr; flips with j)
#   X: final += C[...] * tmp1[0..5] + ...   (flips with i)
# Group-uniform skip classification (all 8 members share cxI).  The per-point
# geometry prologue (~18 double divisions) is hoisted to the host: the group
# launcher passes lead_base/cxI_base/box bounds (p3_geom mirrors the device
# lbf/lbc formulas with 0.4f exactly; 26c already relies on this).
#
# Files changed (candidate tree only):
#   src/prolongrestrict_cell_gpu.cu     : add prolong3_multi_kernel (boundary,
#         compact 26c-style group enumeration); gpu_prolong3_launch launches it.
#   src/prolongrestrict_cell_gpu_int.cu : add prolong3_multi_kernel_int
#         (interior, full-box group enumeration); gpu_prolong3_launch_int
#         launches it.
import hashlib
import sys

GUARDS = {
    "src/prolongrestrict_cell_gpu.cu": "44b8dc55b7a7421dd6389690c7ca363dd9a476dbce2954f787d840cb895382a0",
    "src/prolongrestrict_cell_gpu_int.cu": "4d0c3acd74de01d3a15b3b55c99730e5f8ff3e8dd91ef4a053b1ca6fa9154dd7",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


KERNEL_TMPL = r'''// ---------------------------------------------------------------------------
// P33: prolong3 group multi-output kernel.
// One thread interpolates a parity-aligned 2x2x2 block of fine points sharing
// the SAME 6x6x6 coarse cube (i and i+1 with (i+lbf) even share cxI).  The 216
# TAPS_COMMENT
// coarse taps are loaded once per group.  Each member's accumulation order is
// exactly the original per-point kernel's (Z: 6 separate += over t ascending
// with parity-flipped coefficients; Y/X: single 6-term expressions).
// ---------------------------------------------------------------------------
__global__ void __NAME__(
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
    // interior kernel: full-box group enumeration only
    int total = Gi * Gj * Gk;
    if (idx >= total) return;
    g_k = idx / (Gi * Gj);
    int rem = idx % (Gi * Gj);
    g_j = rem / Gi;
    g_i = rem % Gi;
    (void)compact_mode; (void)lo_i; (void)hi_i; (void)lo_j; (void)hi_j;
    (void)lo_k; (void)hi_k;
#endif

    // group leader (fine indices) and coarse anchor; cxI is group-uniform
    int li = lead_base0 + 2 * g_i;
    int lj = lead_base1 + 2 * g_j;
    int lk = lead_base2 + 2 * g_k;
    int cxI_i = cxI_base0 + g_i;
    int cxI_j = cxI_base1 + g_j;
    int cxI_k = cxI_base2 + g_k;

#ifdef PROLONG3_INTERIOR
    // interior-only kernel: skip boundary groups entirely (all 8 members
    // share cxI, so the classification is group-uniform)
    if (cxI_i < 3 || cxI_i > extc0 - 3 ||
        cxI_j < 3 || cxI_j > extc1 - 3 ||
        cxI_k < 3 || cxI_k > extc2 - 3) return;
#else
    if (skip_interior) {
        if (cxI_i >= 3 && cxI_i <= extc0 - 3 &&
            cxI_j >= 3 && cxI_j <= extc1 - 3 &&
            cxI_k >= 3 && cxI_k <= extc2 - 3) return;
    }
#endif

    int extc[3] = {extc0, extc1, extc2};
    const double SoA[3] = {SoA0, SoA1, SoA2};

    double tmp2[2][6][6];

    // Z-pass over the shared coarse cube: ONE tap load per (m, n, t);
    // both k-parity rows accumulated with the original statement order.
    for (int m = 0; m < 6; ++m) {
        for (int n = 0; n < 6; ++n) {
            int cur_ic = cxI_i - 2 + n;
            int cur_jc = cxI_j - 2 + m;

            double a0 = d_symmetry_bd_1b(3, extc, d_src_c, cur_ic, cur_jc, cxI_k - 2, SoA);
            double a1 = d_symmetry_bd_1b(3, extc, d_src_c, cur_ic, cur_jc, cxI_k - 1, SoA);
            double a2 = d_symmetry_bd_1b(3, extc, d_src_c, cur_ic, cur_jc, cxI_k    , SoA);
            double a3 = d_symmetry_bd_1b(3, extc, d_src_c, cur_ic, cur_jc, cxI_k + 1, SoA);
            double a4 = d_symmetry_bd_1b(3, extc, d_src_c, cur_ic, cur_jc, cxI_k + 2, SoA);
            double a5 = d_symmetry_bd_1b(3, extc, d_src_c, cur_ic, cur_jc, cxI_k + 3, SoA);

            double v0 = 0.0;
            double v1 = 0.0;
            v0 += C_PROLONG[0] * a0; v1 += C_PROLONG[5] * a0;
            v0 += C_PROLONG[1] * a1; v1 += C_PROLONG[4] * a1;
            v0 += C_PROLONG[2] * a2; v1 += C_PROLONG[3] * a2;
            v0 += C_PROLONG[3] * a3; v1 += C_PROLONG[2] * a3;
            v0 += C_PROLONG[4] * a4; v1 += C_PROLONG[1] * a4;
            v0 += C_PROLONG[5] * a5; v1 += C_PROLONG[0] * a5;

            tmp2[0][m][n] = v0;
            tmp2[1][m][n] = v1;
        }
    }

    // Y/X passes per member parity combination (dj == 0 -> j_even, etc.)
    for (int dk = 0; dk < 2; ++dk) {
        for (int dj = 0; dj < 2; ++dj) {
            double tmp1[6];
            for (int n = 0; n < 6; ++n) {
                double val = 0.0;
                if (dj == 0) {
                    val += C_PROLONG[0] * tmp2[dk][0][n] + C_PROLONG[1] * tmp2[dk][1][n] +
                           C_PROLONG[2] * tmp2[dk][2][n] + C_PROLONG[3] * tmp2[dk][3][n] +
                           C_PROLONG[4] * tmp2[dk][4][n] + C_PROLONG[5] * tmp2[dk][5][n];
                } else {
                    val += C_PROLONG[5] * tmp2[dk][0][n] + C_PROLONG[4] * tmp2[dk][1][n] +
                           C_PROLONG[3] * tmp2[dk][2][n] + C_PROLONG[2] * tmp2[dk][3][n] +
                           C_PROLONG[1] * tmp2[dk][4][n] + C_PROLONG[0] * tmp2[dk][5][n];
                }
                tmp1[n] = val;
            }
            for (int di = 0; di < 2; ++di) {
                int i = li + di;
                int j = lj + dj;
                int k = lk + dk;
                if (i < i_start || i > i_end ||
                    j < j_start || j > j_end ||
                    k < k_start || k > k_end) continue;
                double final_val = 0.0;
                if (di == 0) {
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


def main(root):
    for path, want in GUARDS.items():
        got = sha256(f"{root}/{path}")
        if got != want:
            print(f"GUARD_FAIL {path}: {got} != {want}")
            sys.exit(1)
    print("guards ok")

    # ---------------- 1. base TU ----------------
    p = f"{root}/src/prolongrestrict_cell_gpu.cu"
    s = open(p).read()
    if "prolong3_multi_kernel" in s:
        print("ALREADY_PATCHED base TU")
        sys.exit(1)

    kernel = KERNEL_TMPL.replace("__NAME__", "prolong3_multi_kernel")
    kernel = kernel.replace("# TAPS_COMMENT\n", "")

    anchor = "__global__ void prolong3_kernel("
    assert s.count(anchor) == 1
    s = s.replace(anchor, kernel + anchor)

    # rewrite the launcher tail
    old_tail = """    int total_points = ni * nj * nk;

    int lbf_p3[3], lbc_p3[3];
    p3_geom(llbc, uubc, extc, llbf, uubf, extf, lbf_p3, lbc_p3);
    int st[3] = {i_start, j_start, k_start};
    int nd[3] = {ni, nj, nk};
    int lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    for (int d = 0; d < 3; d++) {
        for (int il = 0; il < nd[d]; il++) {
            int cx = p3_cxI(st[d] + il, lbf_p3[d], lbc_p3[d]);
            if (cx <= 2) lo[d]++;
            if (cx >= extc[d] - 2) hi[d]++;
        }
    }
    int compact_mode = 1;
    for (int d = 0; d < 3; d++)
        if (lo[d] + hi[d] > nd[d]) compact_mode = 0;

    int block = 256;
    int grid;
    if (compact_mode) {
        int N = p3_bnd_count(ni, nj, nk, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        if (N == 0) return;
        grid = (N + block - 1) / block;
    } else {
        grid = (total_points + block - 1) / block;
    }

    prolong3_kernel<<<grid, block, 0, stream>>>(
        ni, nj, nk, i_start, j_start, k_start,
        llbc[0], llbc[1], llbc[2],
        uubc[0], uubc[1], uubc[2],
        extc[0], extc[1], extc[2],
        d_src_c,
        llbf[0], llbf[1], llbf[2],
        uubf[0], uubf[1], uubf[2],
        extf[0], extf[1], extf[2],
        d_dst_f,
        llbt[0], llbt[1], llbt[2],
        uubt[0], uubt[1], uubt[2],
        SoA[0], SoA[1], SoA[2],
        Symmetry, skip_interior, compact_mode,
        lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]
    );
}"""
    assert s.count(old_tail) == 1, "base launcher tail anchor not unique"

    new_tail = """    int total_points = ni * nj * nk;
    (void)total_points;

    // P33: parity-aligned 2x2x2 group geometry.  Fine indices i, i+1 with
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
    s = s.replace(old_tail, new_tail)
    open(p, "w").write(s)
    print("patched prolongrestrict_cell_gpu.cu")

    # ---------------- 2. int TU ----------------
    p = f"{root}/src/prolongrestrict_cell_gpu_int.cu"
    s = open(p).read()
    if "prolong3_multi_kernel_int" in s:
        print("ALREADY_PATCHED int TU")
        sys.exit(1)

    kernel = KERNEL_TMPL.replace("__NAME__", "prolong3_multi_kernel_int")
    kernel = kernel.replace("# TAPS_COMMENT\n", "")

    anchor = "__global__ void prolong3_kernel_int("
    assert s.count(anchor) == 1
    s = s.replace(anchor, kernel + anchor)

    old_tail = """    int total_points = ni * nj * nk;
    int block = 256;
    int grid = (total_points + block - 1) / block;

    prolong3_kernel_int<<<grid, block, 0, stream>>>(
        ni, nj, nk, i_start, j_start, k_start,
        llbc[0], llbc[1], llbc[2],
        uubc[0], uubc[1], uubc[2],
        extc[0], extc[1], extc[2],
        d_src_c,
        llbf[0], llbf[1], llbf[2],
        uubf[0], uubf[1], uubf[2],
        extf[0], extf[1], extf[2],
        d_dst_f,
        llbt[0], llbt[1], llbt[2],
        uubt[0], uubt[1], uubt[2],
        SoA[0], SoA[1], SoA[2],
        Symmetry, skip_interior
    );
}"""
    assert s.count(old_tail) == 1, "int launcher tail anchor not unique"

    new_tail = """    int total_points = ni * nj * nk;
    (void)total_points;

    // P33: parity-aligned 2x2x2 group geometry (mirror of the base launcher;
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

    prolong3_multi_kernel_int<<<grid, block, 0, stream>>>(
        G[0], G[1], G[2],
        lead_base[0], lead_base[1], lead_base[2],
        cxI_base[0], cxI_base[1], cxI_base[2],
        i_start, i_end, j_start, j_end, k_start, k_end,
        extc[0], extc[1], extc[2],
        d_src_c,
        extf[0], extf[1], extf[2],
        d_dst_f,
        SoA[0], SoA[1], SoA[2],
        Symmetry, skip_interior, 0,
        0, 0, 0, 0, 0, 0
    );
}"""
    s = s.replace(old_tail, new_tail)
    open(p, "w").write(s)
    print("patched prolongrestrict_cell_gpu_int.cu")

    print("P33 PATCH DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
