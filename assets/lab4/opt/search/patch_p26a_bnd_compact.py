#!/usr/bin/env python3
# Iter26a: rhs_boundary compact launch domain (6-slab decomposition).
#
# Evidence (reprofile 2026-08-26, jobs 165992/166024): rhs_kernel (boundary)
# is 203.7s / 32.98% of kernel time, and it launches over the FULL patch
# volume with interior threads early-returning (identical blocks/launch to
# rhs_kernel_int: 125/512/1120 per window). This wastes block scheduling and
# per-thread index/branch work for interior-only blocks.
#
# Change (single variable: launch domain; per-point computation untouched ->
# bit-exact by construction):
#   - bssn_rhs_gpu.cu gains __host__ __device__ bnd_slab_size / bnd_count /
#     bnd_map: a 6-slab disjoint partition of the boundary shell:
#       slab 0: i in {0,1}                      (i-lo), all j,k
#       slab 1: i in {ex0-2,ex0-1}              (i-hi), all j,k
#       slab 2: j in {0,1}      , i in [2,ex0-3], all k     (j-lo)
#       slab 3: j in {ex1-2,ex1-1}, i in [2,ex0-3], all k   (j-hi)
#       slab 4: k in {0,1,2}    , i,j interior              (k-lo)
#       slab 5: k in {ex2-2,ex2-1}, i,j interior            (k-hi)
#     (first-slab-wins order; exact for ex0>=4, ex1>=4, ex2>=5; mapping
#     validated numerically for 16 shapes incl. degenerate, see
#     tmp/iter26/validate_bnd_map.py). i fastest -> coalesced loads.
#   - rhs_kernel gains trailing `int compact_mode`: when 1, (i,j,k) come from
#     bnd_map(blockIdx.x*256+threadIdx.x) and the bounds/interior checks are
#     skipped; the body is otherwise token-identical.
#   - gpu_compute_rhs_bssn_launch dispatches: compact 1-D grid over
#     bnd_count points when ex>=8 in every dim, else legacy full-volume launch
#     (generic fallback for rare/small patches; never triggered by real grids,
#     min rhs grid 125 blocks = 40x40x20).
#
# Correctness: every boundary point computed with the exact same code path and
# coordinates as before; interior points never mapped. bit-exact by
# construction (same float ops, same order). Legacy path unchanged.
import hashlib, os, sys, shutil

FORMAL = os.path.expanduser("~/lab4-gpu")
EXCLUDE_DIRS = {"evidence", "build", "__pycache__"}

BND_FUNCS = """__host__ __device__ static inline int bnd_slab_size(int s, int ex0, int ex1, int ex2) {
    // Iter26a: 6-slab disjoint partition of the boundary shell (see header).
    // Interior box: i in [2,ex0-3], j in [2,ex1-3], k in [3,ex2-3].
    int ni = (ex0 - 4 > 0 ? ex0 - 4 : 0); // interior i count
    int nj = (ex1 - 4 > 0 ? ex1 - 4 : 0); // interior j count
    switch (s) {
        case 0: return 2 * ex1 * ex2;             // i-lo: i in {0,1} x all j,k
        case 1: return 2 * ex1 * ex2;             // i-hi
        case 2: return 2 * ni * ex2;              // j-lo (i interior)
        case 3: return 2 * ni * ex2;              // j-hi
        case 4: return 3 * ni * nj;               // k-lo (i,j interior)
        case 5: return 2 * ni * nj;               // k-hi
    }
    return 0;
}

__host__ __device__ static inline int bnd_count(int ex0, int ex1, int ex2) {
    int t = 0;
    for (int s = 0; s < 6; s++) t += bnd_slab_size(s, ex0, ex1, ex2);
    return t;
}

__host__ __device__ static inline void bnd_map(int n, int ex0, int ex1, int ex2,
                                               int* i, int* j, int* k) {
    // map flat shell index -> (i,j,k); i fastest within each slab (coalesced).
    int ni = (ex0 - 4 > 0 ? ex0 - 4 : 0);
    int nj = (ex1 - 4 > 0 ? ex1 - 4 : 0);
    for (int s = 0; s < 6; s++) {
        int sz = bnd_slab_size(s, ex0, ex1, ex2);
        if (n < sz) {
            int t, di, dj;
            switch (s) {
                case 0: di = 2;            dj = ex1; *i = n % di;                 t = n / di; *j = t % dj; *k = t / dj; return;
                case 1: di = 2;            dj = ex1; *i = n % di + (ex0 - 2);     t = n / di; *j = t % dj; *k = t / dj; return;
                case 2: di = ni;           dj = 2;    *i = n % di + 2;            t = n / di; *j = t % dj; *k = t / dj; return;
                case 3: di = ni;           dj = 2;    *i = n % di + 2;            t = n / di; *j = t % dj + (ex1 - 2); *k = t / dj; return;
                case 4: di = ni;           dj = nj;   *i = n % di + 2;            t = n / di; *j = t % dj + 2; *k = t / dj; return;
                default: di = ni;          dj = nj;   *i = n % di + 2;            t = n / di; *j = t % dj + 2; *k = t / dj + (ex2 - 2); return;
            }
        }
        n -= sz;
    }
    *i = *j = *k = -1; // unreachable when n < bnd_count
}

"""

KERN_SIG_ANCHOR = """    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
    int symmetry, int lev, double eps, int co, int skip_interior
) {
    // ------------------------------------------------------------------------------------
    // bssn_derivatives_kernel
    // ------------------------------------------------------------------------------------

    // 计算全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // 越界检查
    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    // boundary-only mode (skip_interior=1): interior points are computed by
    // rhs_kernel_int in bssn_rhs_gpu_int.cu; skip them here (deployed
    // behavior when skip_interior=0: condition never matches, no code change).
    if (skip_interior) { if (i >= 2 && i < ex0 - 2 && j >= 2 && j < ex1 - 2 && k >= 3 && k < ex2 - 2) return; }
"""

KERN_SIG_OUT = """    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
    int symmetry, int lev, double eps, int co, int skip_interior, int compact_mode
) {
    // ------------------------------------------------------------------------------------
    // bssn_derivatives_kernel
    // ------------------------------------------------------------------------------------

    // 计算全局索引（Iter26a compact 模式：一维 grid，6-slab 映射到边界壳点）
    int i, j, k;
    if (compact_mode) {
        int n = blockIdx.x * blockDim.x + threadIdx.x;
        bnd_map(n, ex0, ex1, ex2, &i, &j, &k);
        if (i < 0) return; // safety; unreachable when n < bnd_count
    } else {
        i = blockIdx.x * blockDim.x + threadIdx.x;
        j = blockIdx.y * blockDim.y + threadIdx.y;
        k = blockIdx.z * blockDim.z + threadIdx.z;

        // 越界检查
        if (i >= ex0 || j >= ex1 || k >= ex2) return;

        // boundary-only mode (skip_interior=1): interior points are computed by
        // rhs_kernel_int in bssn_rhs_gpu_int.cu; skip them here (deployed
        // behavior when skip_interior=0: condition never matches, no code change).
        if (skip_interior) { if (i >= 2 && i < ex0 - 2 && j >= 2 && j < ex1 - 2 && k >= 3 && k < ex2 - 2) return; }
    }
"""

LAUNCH_ANCHOR = """    int symmetry, int lev, double eps, int co, int skip_interior
) {
    dim3 block(8, 8, 4); // V1: 256 threads, 2 blocks/SM
    dim3 grid(
        (ex[0] + block.x - 1) / block.x,
        (ex[1] + block.y - 1) / block.y,
        (ex[2] + block.z - 1) / block.z
    );

    // 1. Kernel 1: Derivatives & Connection Coefficients
    rhs_kernel<<<grid, block, 0, stream>>>(
"""

LAUNCH_OUT = """    int symmetry, int lev, double eps, int co, int skip_interior
) {
    // Iter26a: compact boundary-shell mode (1-D grid over the shell points
    // only, 6-slab decomposition); legacy full-volume launch kept for small
    // patches and as the generic fallback.
    int compact_mode = (ex[0] >= 8 && ex[1] >= 8 && ex[2] >= 8) ? 1 : 0;
    dim3 block, grid;
    if (compact_mode) {
        int N = bnd_count(ex[0], ex[1], ex[2]);
        block = dim3(256, 1, 1);
        grid = dim3((N + 255) / 256);
    } else {
        block = dim3(8, 8, 4); // V1: 256 threads, 2 blocks/SM
        grid = dim3(
            (ex[0] + block.x - 1) / block.x,
            (ex[1] + block.y - 1) / block.y,
            (ex[2] + block.z - 1) / block.z
        );
    }

    // 1. Kernel 1: Derivatives & Connection Coefficients
    rhs_kernel<<<grid, block, 0, stream>>>(
"""

CALL_ANCHOR = """        symmetry, lev, eps, co, skip_interior
    );
}"""

CALL_OUT = """        symmetry, lev, eps, co, skip_interior, compact_mode
    );
}"""


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("usage: patch_p26a_bnd_compact.py <candidate_root>"); sys.exit(2)
    cand = sys.argv[1]
    if os.path.exists(cand):
        print(f"ERROR: {cand} exists"); sys.exit(3)

    formal_src = os.path.join(FORMAL, "src", "bssn_rhs_gpu.cu")
    fh = sha(formal_src)
    print(f"formal bssn_rhs_gpu.cu {fh[:16]}")
    assert fh.startswith("807f7b73"), "bssn_rhs_gpu.cu drifted from deployed baseline"

    shutil.copytree(FORMAL, cand, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))
    print(f"copied formal -> {cand}")

    p = os.path.join(cand, "src", "bssn_rhs_gpu.cu")
    s = open(p).read()

    # 1. insert bnd functions before rhs_kernel
    anchor0 = "__global__ __launch_bounds__(256, 2) void rhs_kernel(\n"
    assert s.count(anchor0) == 1, f"rhs_kernel decl anchor: {s.count(anchor0)}"
    s = s.replace(anchor0, BND_FUNCS + anchor0)
    assert s.count("bnd_slab_size") >= 3, "bnd funcs inserted"

    # 2. kernel signature + index computation
    assert s.count(KERN_SIG_ANCHOR) == 1, f"kernel sig anchor: {s.count(KERN_SIG_ANCHOR)}"
    s = s.replace(KERN_SIG_ANCHOR, KERN_SIG_OUT)
    assert "int compact_mode" in s, "compact_mode in kernel"

    # 3. launch dispatch
    assert s.count(LAUNCH_ANCHOR) == 1, f"launch anchor: {s.count(LAUNCH_ANCHOR)}"
    s = s.replace(LAUNCH_ANCHOR, LAUNCH_OUT)
    assert "int compact_mode = (ex[0] >= 8" in s

    # 4. kernel call tail
    assert s.count(CALL_ANCHOR) == 1, f"call tail anchor: {s.count(CALL_ANCHOR)}"
    s = s.replace(CALL_ANCHOR, CALL_OUT)

    open(p, "w").write(s)
    print(f"patched bssn_rhs_gpu.cu ({len(s)} bytes)")
    print(f"  cand hash {sha(p)[:16]}")
    assert s.count("compact_mode") == 5, f"compact_mode occurrences: {s.count('compact_mode')}"
    # legacy path preserved verbatim
    assert "if (i >= ex0 || j >= ex1 || k >= ex2) return;" in s
    assert "if (skip_interior) { if (i >= 2 && i < ex0 - 2" in s
    print("PATCH OK")


if __name__ == "__main__":
    main()
