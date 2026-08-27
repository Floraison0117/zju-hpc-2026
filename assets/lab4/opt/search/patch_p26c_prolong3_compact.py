#!/usr/bin/env python3
# Iter26c: prolong3_boundary compact launch domain (generalized 6-slab).
#
# Evidence (reprofile 2026-08-26): prolong3_kernel (boundary) is 52.1s /
# 8.44% of kernel time, launching over the FULL fine-grid box with interior
# threads early-returning (identical blocks/launch to prolong3_kernel_int:
# 1/43/141 per window). Same waste pattern as rhs_boundary (Iter26a, deployed:
# 666.64 -> 615.04s).
#
# Change (single variable: launch domain; per-point computation untouched ->
# bit-exact by construction):
#   - prolongrestrict_cell_gpu.cu gains __host__ __device__ p3_slab /
#     p3_bnd_count / p3_bnd_map (generalized 6-slab with per-dim boundary
#     thicknesses lo/hi in the launch box, i fastest -> coalesced) plus
#     host-only h_idint / p3_geom / p3_cxI replicating the device geometry
#     EXACTLY (d_idint + 0.4f constants, same formulas).
#   - gpu_prolong3_launch computes lo/hi by iterating the exact device cxI
#     formula over the launch box, then dispatches a 1-D grid over the shell
#     when the partition is exact (lo+hi <= dim in every dim), else legacy
#     full-box launch. N==0 -> skip launch (no boundary points to write).
#   - prolong3_kernel gains trailing params (compact_mode, lo_i..hi_k); the
#     d_prolong3_device call keeps skip_interior=1 (a no-op for mapped points:
#     all mapped points fail the interior test by construction).
#
# Correctness: shell = fine points whose coarse tap cube touches the coarse
# boundary = {cxI_d <= 2 or cxI_d >= extc[d]-2 in some dim}; the host computes
# the exact fine ranges via the same cxI formula (monotonic in the fine index)
# -> the 6-slab partition is exact and disjoint; every boundary point runs the
# identical interpolation code path as before. int TU untouched.
import hashlib, os, sys, shutil

FORMAL = os.path.expanduser("~/lab4-gpu")
EXCLUDE_DIRS = {"evidence", "build", "__pycache__"}

P3_FUNCS = """// ==========================================
// Iter26c: compact boundary-shell 6-slab mapping (host+device)
// ==========================================
// Boundary = fine points whose coarse index is outside [3, extc-3] in some
// dim (see the skip_interior test below). With monotone coarse mapping this
// is exactly the 6-slab shell with per-dim thicknesses lo/hi (local coords).
__host__ __device__ static inline int p3_slab(int s, int ni, int nj, int nk,
                                             int lo_i, int hi_i, int lo_j, int hi_j,
                                             int lo_k, int hi_k) {
    int mi = ni - lo_i - hi_i; if (mi < 0) mi = 0;
    int mj = nj - lo_j - hi_j; if (mj < 0) mj = 0;
    switch (s) {
        case 0: return lo_i * nj * nk;             // i-lo x all j,k
        case 1: return hi_i * nj * nk;             // i-hi
        case 2: return lo_j * mi * nk;             // j-lo (i interior)
        case 3: return hi_j * mi * nk;             // j-hi
        case 4: return lo_k * mi * mj;             // k-lo (i,j interior)
        case 5: return hi_k * mi * mj;             // k-hi
    }
    return 0;
}

__host__ __device__ static inline int p3_bnd_count(int ni, int nj, int nk,
                                                   int lo_i, int hi_i, int lo_j, int hi_j,
                                                   int lo_k, int hi_k) {
    int t = 0;
    for (int s = 0; s < 6; s++) t += p3_slab(s, ni, nj, nk, lo_i, hi_i, lo_j, hi_j, lo_k, hi_k);
    return t;
}

__host__ __device__ static inline void p3_bnd_map(int n, int ni, int nj, int nk,
                                                  int lo_i, int hi_i, int lo_j, int hi_j,
                                                  int lo_k, int hi_k,
                                                  int* il, int* jl, int* kl) {
    // local coords; i fastest within each slab (coalesced)
    int mi = ni - lo_i - hi_i;
    int mj = nj - lo_j - hi_j;
    for (int s = 0; s < 6; s++) {
        int sz = p3_slab(s, ni, nj, nk, lo_i, hi_i, lo_j, hi_j, lo_k, hi_k);
        if (n < sz) {
            int t, di, dj;
            switch (s) {
                case 0: di = lo_i;          dj = nj; *il = n % di;                 t = n / di; *jl = t % dj; *kl = t / dj; return;
                case 1: di = hi_i;          dj = nj; *il = n % di + (ni - hi_i);   t = n / di; *jl = t % dj; *kl = t / dj; return;
                case 2: di = mi;            dj = lo_j; *il = n % di + lo_i;        t = n / di; *jl = t % dj; *kl = t / dj; return;
                case 3: di = mi;            dj = hi_j; *il = n % di + lo_i;        t = n / di; *jl = t % dj + (nj - hi_j); *kl = t / dj; return;
                case 4: di = mi;            dj = mj;   *il = n % di + lo_i;        t = n / di; *jl = t % dj + lo_j; *kl = t / dj; return;
                default: di = mi;           dj = mj;   *il = n % di + lo_i;        t = n / di; *jl = t % dj + lo_j; *kl = t / dj + (nk - hi_k); return;
            }
        }
        n -= sz;
    }
    *il = *jl = *kl = -1; // unreachable when n < p3_bnd_count
}

// host-only exact replicas of the d_prolong3_device geometry formulas
// (same d_idint semantics, same + 0.4f float constants) so that the host-side
// boundary thickness computation agrees bit-for-bit with the device's cxI.
static inline int h_idint(double a) {
    if (fabs(a) < 1.0) return 0;
    return (int)(a);
}

static inline void p3_geom(const double* llbc, const double* uubc, const int* extc,
                           const double* llbf, const double* uubf, const int* extf,
                           int lbf[3], int lbc[3]) {
    double CD[3], FD[3], base[3];
    for (int d = 0; d < 3; d++) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
    }
    for (int d = 0; d < 3; d++) {
        if (llbc[d] <= llbf[d]) {
            base[d] = llbc[d];
        } else {
            int j_val = h_idint((llbc[d] - llbf[d]) / FD[d] + 0.4f);
            if ((j_val / 2) * 2 == j_val) base[d] = llbf[d];
            else base[d] = llbf[d] - CD[d] / 2.0;
        }
    }
    for (int d = 0; d < 3; d++) {
        lbf[d] = h_idint((llbf[d] - base[d]) / FD[d] + 0.4f) + 1;
        lbc[d] = h_idint((llbc[d] - base[d]) / CD[d] + 0.4f) + 1;
    }
}

static inline int p3_cxI(int gi, int lbf_d, int lbc_d) {
    // device: cxI = (i_1b + lbf - 1) / 2 - lbc + 1 with i_1b = gi + 1
    return (gi + 1 + lbf_d - 1) / 2 - lbc_d + 1;
}

"""

KERN_ANCHOR = """    int Symmetry, int skip_interior
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ni * nj * nk;
    if (idx >= total) return;

    // 1D to 3D mapping
    int k_local = idx / (ni * nj);
    int rem     = idx % (ni * nj);
    int j_local = rem / ni;
    int i_local = rem % ni;

    // 0-based Fortran equivalent loop indices
    int i = i_start + i_local;
    int j = j_start + j_local;
    int k = k_start + k_local;
"""

KERN_OUT = """    int Symmetry, int skip_interior, int compact_mode,
    int lo_i, int hi_i, int lo_j, int hi_j, int lo_k, int hi_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_local, j_local, k_local;
    if (compact_mode) {
        // Iter26c: compact boundary-shell mode: 1-D grid over the shell only
        // (6-slab decomposition, local coords). The host computed lo/hi from
        // the exact device cxI formula, so every mapped point fails the
        // interior test -> the d_prolong3_device skip_interior check below is
        // a no-op (bit-exact).
        int N = p3_bnd_count(ni, nj, nk, lo_i, hi_i, lo_j, hi_j, lo_k, hi_k);
        if (idx >= N) return;
        p3_bnd_map(idx, ni, nj, nk, lo_i, hi_i, lo_j, hi_j, lo_k, hi_k,
                   &i_local, &j_local, &k_local);
    } else {
        int total = ni * nj * nk;
        if (idx >= total) return;
        // 1D to 3D mapping
        k_local = idx / (ni * nj);
        int rem     = idx % (ni * nj);
        j_local = rem / ni;
        i_local = rem % ni;
    }

    // 0-based Fortran equivalent loop indices
    int i = i_start + i_local;
    int j = j_start + j_local;
    int k = k_start + k_local;
"""

LAUNCH_ANCHOR = """    int total_points = ni * nj * nk;
    int block = 256;
    int grid = (total_points + block - 1) / block;

    prolong3_kernel<<<grid, block, 0, stream>>>(
"""

LAUNCH_OUT = """    int total_points = ni * nj * nk;

    // Iter26c: compact boundary-shell mode. Compute per-dim boundary
    // thicknesses of the launch box by iterating the exact device cxI
    // formula (monotone -> low/high contiguous ranges); dispatch a 1-D grid
    // over the shell when the 6-slab partition is exact (lo+hi <= dim in
    // every dim), else keep the legacy full-box launch.
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
        if (N == 0) return; // box entirely interior: nothing to write
        grid = (N + block - 1) / block;
    } else {
        grid = (total_points + block - 1) / block;
    }

    prolong3_kernel<<<grid, block, 0, stream>>>(
"""

CALL_ANCHOR = """        SoA[0], SoA[1], SoA[2],
        Symmetry, skip_interior
    );
}"""

CALL_OUT = """        SoA[0], SoA[1], SoA[2],
        Symmetry, skip_interior, compact_mode,
        lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]
    );
}"""


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("usage: patch_p26c_prolong3_compact.py <candidate_root>"); sys.exit(2)
    cand = sys.argv[1]
    if os.path.exists(cand):
        print(f"ERROR: {cand} exists"); sys.exit(3)

    formal_src = os.path.join(FORMAL, "src", "prolongrestrict_cell_gpu.cu")
    fh = sha(formal_src)
    print(f"formal prolongrestrict_cell_gpu.cu {fh[:16]}")
    assert fh.startswith("6aaaf4a6"), "prolongrestrict_cell_gpu.cu drifted from deployed baseline"

    shutil.copytree(FORMAL, cand, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))
    print(f"copied formal -> {cand}")

    p = os.path.join(cand, "src", "prolongrestrict_cell_gpu.cu")
    s = open(p).read()

    # 1. insert p3 helpers before the kernel implementation section
    anchor0 = "// ++++++++++++++ Kernel Implementation ++++++++++++++\n"
    assert s.count(anchor0) == 1, f"kernel impl anchor: {s.count(anchor0)}"
    s = s.replace(anchor0, P3_FUNCS + anchor0)
    assert s.count("p3_bnd_map") >= 1

    # 2. kernel entry: compact mapping
    assert s.count(KERN_ANCHOR) == 1, f"kernel entry anchor: {s.count(KERN_ANCHOR)}"
    s = s.replace(KERN_ANCHOR, KERN_OUT)
    assert "int compact_mode" in s

    # 3. launch dispatch
    assert s.count(LAUNCH_ANCHOR) == 1, f"launch anchor: {s.count(LAUNCH_ANCHOR)}"
    s = s.replace(LAUNCH_ANCHOR, LAUNCH_OUT)
    assert "p3_geom(llbc, uubc, extc, llbf, uubf, extf" in s

    # 4. kernel call tail
    assert s.count(CALL_ANCHOR) == 1, f"call tail anchor: {s.count(CALL_ANCHOR)}"
    s = s.replace(CALL_ANCHOR, CALL_OUT)

    open(p, "w").write(s)
    print(f"patched prolongrestrict_cell_gpu.cu ({len(s)} bytes)")
    print(f"  cand hash {sha(p)[:16]}")
    assert s.count("compact_mode") == 6, f"compact_mode occurrences: {s.count('compact_mode')}"
    assert "if (idx >= total) return;" in s  # legacy path preserved
    # restrict3_kernel untouched
    assert "void gpu_restrict3_launch" in s
    print("PATCH OK")


if __name__ == "__main__":
    main()
