#!/usr/bin/env python3
"""Iter38 A38-P44 probe: add 4x4x4 group prolong3 kernel (uncalled, L0 probe).

Probe stage only: inserts the kernel into prolongrestrict_cell_gpu.cu under
#ifdef A38P44_PROBE (boundary path) - built with -DA38P44_PROBE -Xptxas -v to
read regs/spill/occupancy WITHOUT wiring it into the host dispatch. If the L0
gate passes (no spill explosion, occupancy viable) the wiring follows as a
separate single-variable change.

Per-member structure: Z-pass reads the 6 k-taps for the member's own anchors
(di/2, dj/2, dk/2) and parity; accumulation order replicates
prolong3_multi_kernel's per-parity statements exactly (Z: 6 sequential += with
parity-flipped coefficients; Y/X: single 6-term expressions).
"""
import hashlib, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

ENCF = dict(encoding="utf-8")
CU = f"{ROOT}/src/prolongrestrict_cell_gpu.cu"

cur = sha(CU)
print(f"guard prolongrestrict_cell_gpu.cu={cur}")
assert cur == "68a65584", f"hash drift {cur}"

s = open(CU, **ENCF).read()
assert "A38P44" not in s, "reentry guard"

probe = r'''
// ==========================================
// A38-P44 PROBE: 4x4x4 group prolong3 (64 outputs/thread, 9^3 smem cube).
// Compiled only under -DA38P44_PROBE (L0 gate check); NOT wired into the host
// dispatch. Per-member accumulation order replicates prolong3_multi_kernel's
// per-parity statements exactly.
// ==========================================
#ifdef A38P44_PROBE
__global__ void prolong3_multi4_probe_kernel(
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

    int li = lead_base0 + 4 * g_i;
    int lj = lead_base1 + 4 * g_j;
    int lk = lead_base2 + 4 * g_k;
    int cxI_i = cxI_base0 + 2 * g_i;   // base anchor (group spans cxI_i..cxI_i+1)
    int cxI_j = cxI_base1 + 2 * g_j;
    int cxI_k = cxI_base2 + 2 * g_k;

    if (skip_interior) {
        if (cxI_i >= 3 && cxI_i <= extc0 - 4 &&
            cxI_j >= 3 && cxI_j <= extc1 - 4 &&
            cxI_k >= 3 && cxI_k <= extc2 - 4) return;
    }

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
                // Z-pass: 36 columns x 6 taps (member anchors di/2,dj/2,dk/2)
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
#endif // A38P44_PROBE
'''

anchor = "// ++++++++++++++ Kernel Implementation ++++++++++++++"
assert s.count(anchor) == 1
s = s.replace(anchor, probe + "\n" + anchor)
open(CU, "w", **ENCF).write(s)
print("prolongrestrict_cell_gpu.cu patched:", sha(CU))
print("DONE")
