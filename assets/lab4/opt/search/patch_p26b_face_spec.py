#!/usr/bin/env python3
# Iter26b: rhs_boundary face specialization (compile-time pure/masked paths).
#
# Evidence (reprofile + 26a): after Iter26a (compact 1-D boundary launch,
# deployed 615.04s), rhs_boundary still spends ~35% of its instructions on the
# fh symmetry-mask machinery (static ISETP 29,274 vs interior 1,033). For
# pure-face points (one coordinate in the boundary layer, the other two
# interior) the fh masks are identity for fderivs/fdderivs (all taps in
# [1,ex], non-reflective; same argument as RHSPROBE_INTERIOR), so a
# compile-time pure-load path is bit-exact.
#
# Structure (7 disjoint boundary regions, mapping numerically validated):
#   R0..R3 = x-lo/x-hi/y-lo/y-hi pure faces -> rhs_kernel_facepure
#            (RHSFACE_PURE: fderivs/fdderivs fh = plain load)
#   R4..R5 = z-lo/z-hi pure faces           -> rhs_kernel_facez
#            (RHSFACE_PURE_XY: fh pure in i/j, k masks kept)
#   R6     = REST (edges+corners)           -> rhs_kernel (generic masks,
#            compact region-6 mapping; legacy full-volume fallback retained)
# lopsided/kodis keep full masks everywhere.
#
# Files:
#   - derivatives.h: RHSFACE_PURE / RHSFACE_PURE_XY branches in the two fh
#     lambdas (d_fderivs_point, d_fdderivs_point).
#   - bssn_rhs_gpu.cu: 7-region mapping; rhs_kernel compact_mode maps region 6;
#     gpu_compute_rhs_bssn_launch dispatches 4+2+1 region launches; extern
#     declarations of the face kernels appended.
#   - bssn_rhs_gpu_face.cu / _facez.cu: verbatim copies + define + renamed
#     kernel with region entry; copied launch function deleted (dispatch lives
#     in the generic TU).
#   - CMakeLists.txt: add the two new TUs.
import hashlib, os, sys, shutil

FORMAL = os.path.expanduser("~/lab4-gpu")
EXCLUDE_DIRS = {"evidence", "build", "__pycache__"}

FH_ANCHOR = """#ifdef RHSPROBE_INTERIOR
        // interior fast path: plain load, no in_range/valid/reflection masks
        // (bit-exact for interior points: masks are identity there, P2-verified)
        return f[((kk) * ex[1] + (jj)) * ex[0] + (ii)];
#endif"""

FH_OUT = """#ifdef RHSPROBE_INTERIOR
        // interior fast path: plain load, no in_range/valid/reflection masks
        // (bit-exact for interior points: masks are identity there, P2-verified)
        return f[((kk) * ex[1] + (jj)) * ex[0] + (ii)];
#elif defined(RHSFACE_PURE)
        // 26b x/y pure-face fast path: j,k strictly interior and i taps
        // clamped to [imin,imax] -> every tap in [1,ex] and non-reflective ->
        // plain load is bit-exact (same argument as RHSPROBE_INTERIOR).
        return f[((kk) * ex[1] + (jj)) * ex[0] + (ii)];
#elif defined(RHSFACE_PURE_XY)
        // 26b z-face fast path: i,j interior (pure); k may reflect on the
        // equatorial k-lo rows or sit at the z-max layer -> keep the k masks
        // only (i/j parts are identity at these points).
        {
            int k1b = kk + 1;
            double in_range = 1.0;
            in_range *= (double)((k1b >= -1) & (k1b <= ex[2]));
            int k2 = k1b + (1 - 2*k1b) * (k1b <= 0);
            double fac = (k1b <= 0) ? SoA[2] : 1.0;
            double valid = (double)((k2 >= 1) & (k2 <= ex[2]));
            return f[((k2 - 1) * ex[1] + (jj)) * ex[0] + (ii)] * fac * in_range * valid;
        }
#endif"""

BND26A_START = "__host__ __device__ static inline int bnd_slab_size(int s, int ex0, int ex1, int ex2) {"
BND26A_END = "    *i = *j = *k = -1; // unreachable when n < bnd_count\n}"

BND26B = """__host__ __device__ static inline int bnd_region_size(int r, int ex0, int ex1, int ex2) {
    // Iter26b: 7-region disjoint partition of the boundary shell.
    // R0..R3 = x/y pure faces (face-normal coord in the boundary layer, the
    //          other two interior); R4..R5 = z pure faces; R6 = REST
    //          (edges + corners: >= 2 coords in boundary layers).
    // Interior box: i in [2,ex0-3], j in [2,ex1-3], k in [3,ex2-3].
    int ni = (ex0 - 4 > 0 ? ex0 - 4 : 0);
    int nj = (ex1 - 4 > 0 ? ex1 - 4 : 0);
    int nk = (ex2 - 5 > 0 ? ex2 - 5 : 0);
    switch (r) {
        case 0: return 2 * nj * nk;            // x-lo: i in {0,1}, j,k interior
        case 1: return 2 * nj * nk;            // x-hi
        case 2: return ni * 2 * nk;            // y-lo
        case 3: return ni * 2 * nk;            // y-hi
        case 4: return ni * nj * 3;            // z-lo: k in {0,1,2}, i,j interior
        case 5: return ni * nj * 2;            // z-hi
        case 6: return 16 * ex2 + 4 * nj * 5 + ni * 4 * 5; // REST a+b+c
    }
    return 0;
}

__host__ __device__ static inline int bnd_region_count(int ex0, int ex1, int ex2) {
    int t = 0;
    for (int r = 0; r < 7; r++) t += bnd_region_size(r, ex0, ex1, ex2);
    return t;
}

__host__ __device__ static inline void bnd_map_region(int n, int r, int ex0, int ex1, int ex2,
                                                      int* i, int* j, int* k) {
    // map flat region index -> (i,j,k); i fastest within each region (coalesced)
    int ni = (ex0 - 4 > 0 ? ex0 - 4 : 0);
    int nj = (ex1 - 4 > 0 ? ex1 - 4 : 0);
    int nk = (ex2 - 5 > 0 ? ex2 - 5 : 0);
    int t, di, dj;
    switch (r) {
        case 0: di = 2; dj = nj; *i = n % di;              t = n / di; *j = t % dj + 2; *k = t / dj + 3; return;
        case 1: di = 2; dj = nj; *i = n % di + (ex0 - 2);  t = n / di; *j = t % dj + 2; *k = t / dj + 3; return;
        case 2: di = ni; dj = 2; *i = n % di + 2;          t = n / di; *j = t % dj; *k = t / dj + 3; return;
        case 3: di = ni; dj = 2; *i = n % di + 2;          t = n / di; *j = t % dj + (ex1 - 2); *k = t / dj + 3; return;
        case 4: di = ni; dj = nj; *i = n % di + 2;         t = n / di; *j = t % dj + 2; *k = t / dj; return;
        case 5: di = ni; dj = nj; *i = n % di + 2;         t = n / di; *j = t % dj + 2; *k = t / dj + (ex2 - 2); return;
        default: {
            // R6 REST: a) i,j boundary (4x4 sets), all k; b) i boundary,
            // j interior, k boundary (5 values); c) i interior, j boundary,
            // k boundary.
            int sa = 16 * ex2;
            int sb = 4 * nj * 5;
            if (n < sa) {
                di = 4; dj = 4;
                int isel = n % di; t = n / di; int jsel = t % dj; *k = t / dj;
                *i = (isel < 2) ? isel : (ex0 - 2) + (isel - 2);
                *j = (jsel < 2) ? jsel : (ex1 - 2) + (jsel - 2);
                return;
            }
            n -= sa;
            if (n < sb) {
                di = 4; dj = nj;
                int isel = n % di; t = n / di; *j = t % dj + 2; int ksel = t / dj;
                *i = (isel < 2) ? isel : (ex0 - 2) + (isel - 2);
                *k = (ksel < 3) ? ksel : (ex2 - 2) + (ksel - 3);
                return;
            }
            n -= sb;
            di = ni; dj = 4;
            *i = n % di + 2; t = n / di; int jsel = t % dj; int ksel = t / dj;
            *j = (jsel < 2) ? jsel : (ex1 - 2) + (jsel - 2);
            *k = (ksel < 3) ? ksel : (ex2 - 2) + (ksel - 3);
            return;
        }
    }
}
"""

# 26a generic kernel entry (exact text in the deployed formal)
ENTRY26A = """    // 计算全局索引（Iter26a compact 模式：一维 grid，6-slab 映射到边界壳点）
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
    }"""

ENTRY26B_GENERIC = """    // 计算全局索引（Iter26b：compact 模式只处理 REST 区域 6；纯面由
    // rhs_kernel_facepure / rhs_kernel_facez 处理）
    int i, j, k;
    if (compact_mode) {
        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= bnd_region_size(6, ex0, ex1, ex2)) return;
        bnd_map_region(n, 6, ex0, ex1, ex2, &i, &j, &k);
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
    }"""

ENTRY_FACE = """    // Iter26b face kernel: 1-D grid over a single boundary region.
    int i, j, k;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= bnd_region_size(region, ex0, ex1, ex2)) return;
    bnd_map_region(n, region, ex0, ex1, ex2, &i, &j, &k);
    (void)skip_interior;"""

LAUNCH_DISPATCH_IN = """    // Iter26a: compact boundary-shell mode (1-D grid over the shell points
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

CALL_TAIL_IN = """        symmetry, lev, eps, co, skip_interior, compact_mode
    );
}"""

EXTERN_DECLS = """// Iter26b extern face kernels (defined in bssn_rhs_gpu_face.cu / _facez.cu)
__global__ __launch_bounds__(256, 2) void rhs_kernel_facepure(
    int ex0, int ex1, int ex2, double T, double* X, double* Y, double* Z,
    double* chi, double* trK,
    double* dxx, double* gxy, double* gxz,
    double* dyy, double* gyz, double* dzz,
    double* Axx, double* Axy, double* Axz,
    double* Ayy, double* Ayz, double* Azz,
    double* Gamx, double* Gamy, double* Gamz,
    double* Lap,
    double* betax, double* betay, double* betaz,
    double* dtSfx, double* dtSfy, double* dtSfz,
    double* chi_rhs, double* trK_rhs,
    double* gxx_rhs, double* gxy_rhs, double* gxz_rhs,
    double* gyy_rhs, double* gyz_rhs, double* gzz_rhs,
    double* Axx_rhs, double* Axy_rhs, double* Axz_rhs,
    double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs,
    double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs,
    double* Lap_rhs,
    double* betax_rhs, double* betay_rhs, double* betaz_rhs,
    double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs,
    double* rho, double* Sx, double* Sy, double* Sz,
    double* Sxx, double* Sxy, double* Sxz,
    double* Syy, double* Syz, double* Szz,
    double* Gamxxx, double* Gamxxy, double* Gamxxz,
    double* Gamxyy, double* Gamxyz, double* Gamxzz,
    double* Gamyxx, double* Gamyxy, double* Gamyxz,
    double* Gamyyy, double* Gamyyz, double* Gamyzz,
    double* Gamzxx, double* Gamzxy, double* Gamzxz,
    double* Gamzyy, double* Gamzyz, double* Gamzzz,
    double* Rxx, double* Rxy, double* Rxz,
    double* Ryy, double* Ryz, double* Rzz,
    double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res,
    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
    int symmetry, int lev, double eps, int co, int skip_interior, int region
);
__global__ __launch_bounds__(256, 2) void rhs_kernel_facez(
    int ex0, int ex1, int ex2, double T, double* X, double* Y, double* Z,
    double* chi, double* trK,
    double* dxx, double* gxy, double* gxz,
    double* dyy, double* gyz, double* dzz,
    double* Axx, double* Axy, double* Axz,
    double* Ayy, double* Ayz, double* Azz,
    double* Gamx, double* Gamy, double* Gamz,
    double* Lap,
    double* betax, double* betay, double* betaz,
    double* dtSfx, double* dtSfy, double* dtSfz,
    double* chi_rhs, double* trK_rhs,
    double* gxx_rhs, double* gxy_rhs, double* gxz_rhs,
    double* gyy_rhs, double* gyz_rhs, double* gzz_rhs,
    double* Axx_rhs, double* Axy_rhs, double* Axz_rhs,
    double* Ayy_rhs, double* Ayz_rhs, double* Azz_rhs,
    double* Gamx_rhs, double* Gamy_rhs, double* Gamz_rhs,
    double* Lap_rhs,
    double* betax_rhs, double* betay_rhs, double* betaz_rhs,
    double* dtSfx_rhs, double* dtSfy_rhs, double* dtSfz_rhs,
    double* rho, double* Sx, double* Sy, double* Sz,
    double* Sxx, double* Sxy, double* Sxz,
    double* Syy, double* Syz, double* Szz,
    double* Gamxxx, double* Gamxxy, double* Gamxxz,
    double* Gamxyy, double* Gamxyz, double* Gamxzz,
    double* Gamyxx, double* Gamyxy, double* Gamyxz,
    double* Gamyyy, double* Gamyyz, double* Gamyzz,
    double* Gamzxx, double* Gamzxy, double* Gamzxz,
    double* Gamzyy, double* Gamzyz, double* Gamzzz,
    double* Rxx, double* Rxy, double* Rxz,
    double* Ryy, double* Ryz, double* Rzz,
    double* ham_Res, double* movx_Res, double* movy_Res, double* movz_Res,
    double* Gmx_Res, double* Gmy_Res, double* Gmz_Res,
    int symmetry, int lev, double eps, int co, int skip_interior, int region
);"""

# signature tail of the kernel (unique in each TU after launch deletion)
SIG_IN = "int symmetry, int lev, double eps, int co, int skip_interior, int compact_mode\n) {"
SIG_OUT = "int symmetry, int lev, double eps, int co, int skip_interior, int region\n) {"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("usage: patch_p26b_face_spec.py <candidate_root>"); sys.exit(2)
    cand = sys.argv[1]
    if os.path.exists(cand):
        print(f"ERROR: {cand} exists"); sys.exit(3)

    fb = sha(os.path.join(FORMAL, "src", "bssn_rhs_gpu.cu"))
    fd = sha(os.path.join(FORMAL, "src", "derivatives.h"))
    fc = sha(os.path.join(FORMAL, "CMakeLists.txt"))
    print(f"formal bssn_rhs_gpu.cu {fb[:16]} derivatives.h {fd[:16]} CMakeLists {fc[:16]}")
    assert fb.startswith("6ef7bf1f"), "bssn_rhs_gpu.cu drifted (expect 26a deployed)"
    assert fd.startswith("771f6685"), "derivatives.h drifted"
    assert fc.startswith("40e958e6"), "CMakeLists drifted"

    shutil.copytree(FORMAL, cand, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))
    print(f"copied formal -> {cand}")

    # ---- A. derivatives.h ----
    p = os.path.join(cand, "src", "derivatives.h")
    s = open(p).read()
    n = s.count(FH_ANCHOR)
    assert n == 2, f"fh anchor occurrences: {n}"
    s = s.replace(FH_ANCHOR, FH_OUT)
    assert s.count("RHSFACE_PURE") == 4 and s.count("RHSFACE_PURE_XY") == 2, "fh variant count"
    open(p, "w").write(s)
    print("patched derivatives.h (RHSFACE_PURE / RHSFACE_PURE_XY)")

    # ---- B. bssn_rhs_gpu.cu ----
    p = os.path.join(cand, "src", "bssn_rhs_gpu.cu")
    s = open(p).read()

    # B1: 7-region mapping replaces 6-slab functions
    i0 = s.index(BND26A_START); i1 = s.index(BND26A_END) + len(BND26A_END)
    s = s[:i0] + BND26B + s[i1:]
    assert s.count("bnd_region_size") >= 2 and s.count("bnd_map_region") >= 1

    # B2: generic kernel entry -> region 6
    assert s.count(ENTRY26A) == 1, f"entry26a anchor: {s.count(ENTRY26A)}"
    s = s.replace(ENTRY26A, ENTRY26B_GENERIC)

    # B3: launch dispatch -> 7 region launches. Replace the whole region
    # from the 26a dispatch through the original call tail (args + `);}`)
    # with the new dispatch (which ends with the launch function's closing `}`).
    d0 = s.index(LAUNCH_DISPATCH_IN)
    d1 = s.index(CALL_TAIL_IN) + len(CALL_TAIL_IN)
    call_start = d0 + len(LAUNCH_DISPATCH_IN)
    call_end = d1 - len(CALL_TAIL_IN)
    arglist = s[call_start:call_end].rstrip()
    assert arglist.endswith("d_Gmz_Res,"), arglist[-40:]  # last of the 100 args
    bodyargs = arglist.lstrip()

    def region_launch(kern, region):
        return (f"    if (bnd_region_size({region}, ex[0], ex[1], ex[2]) > 0) {{\n"
                f"        dim3 g{region+1}((bnd_region_size({region}, ex[0], ex[1], ex[2]) + 255) / 256);\n"
                f"        {kern}<<<g{region+1}, block, 0, stream>>>(\n"
                f"{bodyargs}\n            symmetry, lev, eps, co, 1, {region}\n        );\n    }}\n")

    dispatch = """    // Iter26b: compact boundary-shell mode with face specialization.
    // R0..R3 -> rhs_kernel_facepure (RHSFACE_PURE), R4..R5 -> rhs_kernel_facez
    // (RHSFACE_PURE_XY), R6 (REST) -> rhs_kernel (generic masks). Legacy
    // full-volume launch kept for small patches / generic fallback.
    int compact_mode = (ex[0] >= 8 && ex[1] >= 8 && ex[2] >= 8) ? 1 : 0;
    dim3 block(256, 1, 1);
    dim3 block3(8, 8, 4);
    dim3 grid3(
        (ex[0] + 7) / 8,
        (ex[1] + 7) / 8,
        (ex[2] + 3) / 4
    );

    // 1. Kernel 1: Derivatives & Connection Coefficients (per-region launch)
"""
    for r in range(4):
        dispatch += region_launch("rhs_kernel_facepure", r)
    for r in range(4, 6):
        dispatch += region_launch("rhs_kernel_facez", r)
    dispatch += ("    if (bnd_region_size(6, ex[0], ex[1], ex[2]) > 0) {\n"
                 "        dim3 g7((bnd_region_size(6, ex[0], ex[1], ex[2]) + 255) / 256);\n"
                 "        rhs_kernel<<<g7, block, 0, stream>>>(\n"
                 f"{bodyargs}\n            symmetry, lev, eps, co, 1, 1\n        );\n    }} else {{\n"
                 "        // legacy full-volume fallback (tiny patches)\n"
                 "        rhs_kernel<<<grid3, block3, 0, stream>>>(\n"
                 f"{bodyargs}\n            symmetry, lev, eps, co, 1, 0\n        );\n"
                 "    }\n"
                 "}")
    s = s[:d0] + dispatch + s[d1:]
    assert s.count("rhs_kernel_facepure<<<") == 4, "facepure launch count"
    assert s.count("rhs_kernel_facez<<<") == 2, "facez launch count"
    assert s.count("rhs_kernel<<<") == 2, "generic launches (region6 + legacy)"

    # B4: extern face-kernel declarations BEFORE the launch function
    lf = "void gpu_compute_rhs_bssn_launch("
    assert s.count(lf) == 1
    s = s.replace(lf, EXTERN_DECLS + "\n" + lf)
    assert "rhs_kernel_facepure(" in s and "rhs_kernel_facez(" in s
    open(p, "w").write(s)
    print(f"patched bssn_rhs_gpu.cu ({len(s)} bytes) cand {sha(p)[:16]}")

    # ---- C. face TUs (copied from the PATCHED generic TU so they inherit
    # the 7-region bnd functions) ----
    patched_generic = s
    for define, kernname, fname in (("RHSFACE_PURE", "rhs_kernel_facepure", "face"),
                                    ("RHSFACE_PURE_XY", "rhs_kernel_facez", "facez")):
        t = "#define " + define + "\n" + patched_generic
        # kernel entry: generic region-6 mapping -> single-region face mapping
        assert t.count(ENTRY26B_GENERIC) == 1, f"{kernname} generic entry anchor"
        t = t.replace(ENTRY26B_GENERIC, ENTRY_FACE)
        # delete the extern declarations (they mention both face kernels and
        # would be corrupted by the rename below; unused in this TU)
        ex0 = t.find("// Iter26b extern face kernels")
        assert ex0 > 0, "extern block start"
        ex_tail = "int symmetry, int lev, double eps, int co, int skip_interior, int region\n);"
        ex1 = t.rfind(ex_tail)
        assert ex1 > ex0, "extern block end"
        t = t[:ex0] + t[ex1 + len(ex_tail):]
        # delete the copied launch function (last function in the file)
        li = t.index("void gpu_compute_rhs_bssn_launch(")
        t = t[:li].rstrip() + "\n"
        # rename kernel + remaining references
        assert t.count("rhs_kernel") >= 1
        t = t.replace("rhs_kernel", kernname)
        # signature: compact_mode -> region
        assert t.count(SIG_IN) == 1, f"{kernname} sig anchor: {t.count(SIG_IN)}"
        t = t.replace(SIG_IN, SIG_OUT)
        fp = os.path.join(cand, "src", f"bssn_rhs_gpu_{fname}.cu")
        open(fp, "w").write(t)
        print(f"wrote {os.path.basename(fp)} ({len(t)} bytes) {sha(fp)[:16]}")
        assert t.count(kernname) >= 1
        assert "bnd_region_size(region, ex0, ex1, ex2)" in t
        assert "gpu_compute_rhs_bssn_launch(" not in t
        assert "// Iter26b extern face kernels" not in t

    # ---- D. CMakeLists.txt ----
    p = os.path.join(cand, "CMakeLists.txt")
    s = open(p).read()
    old = "src/bssn_rhs_gpu.cu src/bssn_rhs_gpu_int.cu"
    assert s.count(old) == 1, f"cmake anchor: {s.count(old)}"
    s = s.replace(old, "src/bssn_rhs_gpu.cu src/bssn_rhs_gpu_int.cu src/bssn_rhs_gpu_face.cu src/bssn_rhs_gpu_facez.cu")
    open(p, "w").write(s)
    print("patched CMakeLists.txt (face TUs added)")
    print("\nPATCH OK")


if __name__ == "__main__":
    main()
