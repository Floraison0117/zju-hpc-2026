#!/usr/bin/env python3
# Iter26bcd combined: 26b (face specialization) + 26c (prolong3 boundary
# compact) + P28 (matter dedup) in one candidate, applied on the 26a deployed
# formal (615.04s baseline).
#
# Order matters: P28 dedup is applied to bssn_rhs_gpu.cu / _int.cu first; the
# 26b face/facez TUs are then generated from the PATCHED generic TU so they
# inherit the dedup. 26c touches only prolongrestrict_cell_gpu.cu.
#
# All anchors are copied verbatim from the individually validated patches
# (patch_p28_dedup.py, patch_p26b_face_spec.py, patch_p26c_prolong3_compact.py).
import hashlib, os, sys, shutil

FORMAL = os.path.expanduser("~/lab4-gpu")
EXCLUDE_DIRS = {"evidence", "build", "__pycache__", "GW250118"}

# ---------------- P28: matter dedup ----------------
P28_S = """    double S = chin1 * (gupxx * Sxx[idx] + gupyy * Syy[idx] + gupzz * Szz[idx] + 
               TWO * (gupxy * Sxy[idx] + gupxz * Sxz[idx] + gupyz * Syz[idx]));"""
P28_S_OUT = """    // Iter28: load stress-energy arrays once and reuse (in-kernel
    // round-trip elimination; bit-exact, same array values)
    double val_rho = rho[idx];
    double val_Sxx = Sxx[idx]; double val_Sxy = Sxy[idx]; double val_Sxz = Sxz[idx];
    double val_Syy = Syy[idx]; double val_Syz = Syz[idx]; double val_Szz = Szz[idx];

    double S = chin1 * (gupxx * val_Sxx + gupyy * val_Syy + gupzz * val_Szz + 
               TWO * (gupxy * val_Sxy + gupxz * val_Sxz + gupyz * val_Syz));"""
P28_SRC = """    double src_xx = alpn1 * (l_Rxx - EIGHT*PI*Sxx[idx]) - fxx; // fxx is D_i D_j Lap
    double src_yy = alpn1 * (l_Ryy - EIGHT*PI*Syy[idx]) - fyy;
    double src_zz = alpn1 * (l_Rzz - EIGHT*PI*Szz[idx]) - fzz;
    double src_xy = alpn1 * (l_Rxy - EIGHT*PI*Sxy[idx]) - fxy;
    double src_xz = alpn1 * (l_Rxz - EIGHT*PI*Sxz[idx]) - fxz;
    double src_yz = alpn1 * (l_Ryz - EIGHT*PI*Syz[idx]) - fyz;"""
P28_SRC_OUT = """    double src_xx = alpn1 * (l_Rxx - EIGHT*PI*val_Sxx) - fxx; // fxx is D_i D_j Lap
    double src_yy = alpn1 * (l_Ryy - EIGHT*PI*val_Syy) - fyy;
    double src_zz = alpn1 * (l_Rzz - EIGHT*PI*val_Szz) - fzz;
    double src_xy = alpn1 * (l_Rxy - EIGHT*PI*val_Sxy) - fxy;
    double src_xz = alpn1 * (l_Rxz - EIGHT*PI*val_Sxz) - fxz;
    double src_yz = alpn1 * (l_Ryz - EIGHT*PI*val_Syz) - fyz;"""
P28_F = "    double f = F2o3 * val_trK * val_trK - trA2 - F16*PI*rho[idx] + EIGHT*PI*S;"
P28_F_OUT = "    double f = F2o3 * val_trK * val_trK - trA2 - F16*PI*val_rho + EIGHT*PI*S;"
P28_TRK = "    trK_rhs[idx] = -chin1 * trK_rhs_val + alpn1 * (F1o3 * val_trK * val_trK + trA2 + FOUR * PI * (rho[idx] + S));"
P28_TRK_OUT = "    trK_rhs[idx] = -chin1 * trK_rhs_val + alpn1 * (F1o3 * val_trK * val_trK + trA2 + FOUR * PI * (val_rho + S));"
P28_HAM = "    ham_Res[idx] = chin1 * ham_val + F2o3 * val_trK * val_trK - trA2 - F16 * PI * rho[idx];"
P28_HAM_OUT = "    ham_Res[idx] = chin1 * ham_val + F2o3 * val_trK * val_trK - trA2 - F16 * PI * val_rho;"
P28_MOV = """    movx_Res[idx] = movx_Res[idx] - F2o3 * Kx - F8 * PI * Sx[idx];
    movy_Res[idx] = movy_Res[idx] - F2o3 * Ky - F8 * PI * Sy[idx];
    movz_Res[idx] = movz_Res[idx] - F2o3 * Kz - F8 * PI * Sz[idx];"""
P28_MOV_OUT = """    movx_Res[idx] = movx_Res[idx] - F2o3 * Kx - F8 * PI * val_Sx;
    movy_Res[idx] = movy_Res[idx] - F2o3 * Ky - F8 * PI * val_Sy;
    movz_Res[idx] = movz_Res[idx] - F2o3 * Kz - F8 * PI * val_Sz;"""


def apply_p28(path):
    s = open(path).read()
    for anchor, out, label in ((P28_S, P28_S_OUT, "S"), (P28_SRC, P28_SRC_OUT, "src"),
                               (P28_F, P28_F_OUT, "f"), (P28_TRK, P28_TRK_OUT, "trK"),
                               (P28_HAM, P28_HAM_OUT, "ham"), (P28_MOV, P28_MOV_OUT, "mov")):
        n = s.count(anchor)
        assert n == 1, f"{label} anchor: {n}"
        s = s.replace(anchor, out)
    assert s.count("val_rho") == 4 and s.count("rho[idx]") == 1
    open(path, "w").write(s)


# ---------------- 26b: 7-region mapping + face kernels ----------------
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

SIG_IN = "int symmetry, int lev, double eps, int co, int skip_interior, int compact_mode\n) {"
SIG_OUT = "int symmetry, int lev, double eps, int co, int skip_interior, int region\n) {"


def apply_26b_generic(path):
    s = open(path).read()
    i0 = s.index(BND26A_START); i1 = s.index(BND26A_END) + len(BND26A_END)
    s = s[:i0] + BND26B + s[i1:]
    assert s.count(ENTRY26A) == 1
    s = s.replace(ENTRY26A, ENTRY26B_GENERIC)
    d0 = s.index(LAUNCH_DISPATCH_IN)
    d1 = s.index(CALL_TAIL_IN) + len(CALL_TAIL_IN)
    call_start = d0 + len(LAUNCH_DISPATCH_IN)
    arglist = s[call_start:d1 - len(CALL_TAIL_IN)].rstrip()
    assert arglist.endswith("d_Gmz_Res,"), arglist[-40:]
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
    assert s.count("rhs_kernel_facepure<<<") == 4
    assert s.count("rhs_kernel_facez<<<") == 2
    lf = "void gpu_compute_rhs_bssn_launch("
    assert s.count(lf) == 1
    s = s.replace(lf, EXTERN_DECLS + "\n" + lf)
    open(path, "w").write(s)


def make_face_tus(cand, patched_generic):
    for define, kernname, fname in (("RHSFACE_PURE", "rhs_kernel_facepure", "face"),
                                    ("RHSFACE_PURE_XY", "rhs_kernel_facez", "facez")):
        t = "#define " + define + "\n" + patched_generic
        assert t.count(ENTRY26B_GENERIC) == 1
        t = t.replace(ENTRY26B_GENERIC, ENTRY_FACE)
        ex0 = t.find("// Iter26b extern face kernels")
        assert ex0 > 0
        ex_tail = "int symmetry, int lev, double eps, int co, int skip_interior, int region\n);"
        ex1 = t.rfind(ex_tail)
        assert ex1 > ex0
        t = t[:ex0] + t[ex1 + len(ex_tail):]
        li = t.index("void gpu_compute_rhs_bssn_launch(")
        t = t[:li].rstrip() + "\n"
        assert t.count("rhs_kernel") >= 1
        t = t.replace("rhs_kernel", kernname)
        assert t.count(SIG_IN) == 1
        t = t.replace(SIG_IN, SIG_OUT)
        fp = os.path.join(cand, "src", f"bssn_rhs_gpu_{fname}.cu")
        open(fp, "w").write(t)
        assert t.count(kernname) >= 1
        assert "bnd_region_size(region, ex0, ex1, ex2)" in t


# ---------------- 26c: prolong3 compact ----------------
P3_FUNCS = """// ==========================================
// Iter26c: compact boundary-shell 6-slab mapping (host+device)
// ==========================================
__host__ __device__ static inline int p3_slab(int s, int ni, int nj, int nk,
                                             int lo_i, int hi_i, int lo_j, int hi_j,
                                             int lo_k, int hi_k) {
    int mi = ni - lo_i - hi_i; if (mi < 0) mi = 0;
    int mj = nj - lo_j - hi_j; if (mj < 0) mj = 0;
    switch (s) {
        case 0: return lo_i * nj * nk;
        case 1: return hi_i * nj * nk;
        case 2: return lo_j * mi * nk;
        case 3: return hi_j * mi * nk;
        case 4: return lo_k * mi * mj;
        case 5: return hi_k * mi * mj;
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
    *il = *jl = *kl = -1;
}

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
    return (gi + 1 + lbf_d - 1) / 2 - lbc_d + 1;
}

"""

P3_KERN = """    int Symmetry, int skip_interior
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

P3_KERN_OUT = """    int Symmetry, int skip_interior, int compact_mode,
    int lo_i, int hi_i, int lo_j, int hi_j, int lo_k, int hi_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_local, j_local, k_local;
    if (compact_mode) {
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

P3_LAUNCH = """    int total_points = ni * nj * nk;
    int block = 256;
    int grid = (total_points + block - 1) / block;

    prolong3_kernel<<<grid, block, 0, stream>>>(
"""

P3_LAUNCH_OUT = """    int total_points = ni * nj * nk;

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
"""

P3_CALL = """        SoA[0], SoA[1], SoA[2],
        Symmetry, skip_interior
    );
}"""

P3_CALL_OUT = """        SoA[0], SoA[1], SoA[2],
        Symmetry, skip_interior, compact_mode,
        lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]
    );
}"""


def apply_26c(path):
    s = open(path).read()
    a0 = "// ++++++++++++++ Kernel Implementation ++++++++++++++\n"
    assert s.count(a0) == 1
    s = s.replace(a0, P3_FUNCS + a0)
    assert s.count(P3_KERN) == 1
    s = s.replace(P3_KERN, P3_KERN_OUT)
    assert s.count(P3_LAUNCH) == 1
    s = s.replace(P3_LAUNCH, P3_LAUNCH_OUT)
    assert s.count(P3_CALL) == 1
    s = s.replace(P3_CALL, P3_CALL_OUT)
    assert s.count("compact_mode") == 6
    open(path, "w").write(s)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("usage: patch_p26bcd_combined.py <candidate_root>"); sys.exit(2)
    cand = sys.argv[1]
    if os.path.exists(cand):
        print(f"ERROR: {cand} exists"); sys.exit(3)

    for f in ("bssn_rhs_gpu.cu", "bssn_rhs_gpu_int.cu", "derivatives.h",
              "prolongrestrict_cell_gpu.cu"):
        print(f"formal {f}: {sha(os.path.join(FORMAL,'src',f))[:16]}")
    assert sha(os.path.join(FORMAL, "src", "bssn_rhs_gpu.cu")).startswith("6ef7bf1f")
    assert sha(os.path.join(FORMAL, "src", "derivatives.h")).startswith("771f6685")
    assert sha(os.path.join(FORMAL, "src", "prolongrestrict_cell_gpu.cu")).startswith("6aaaf4a6")

    shutil.copytree(FORMAL, cand, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS))
    print(f"copied formal -> {cand}")

    # P28: matter dedup on both rhs TUs
    for f in ("bssn_rhs_gpu.cu", "bssn_rhs_gpu_int.cu"):
        apply_p28(os.path.join(cand, "src", f))
    print("P28 applied (dedup x2 TUs)")

    # 26b: generic TU + derivatives.h + face TUs + CMake
    apply_26b_generic(os.path.join(cand, "src", "bssn_rhs_gpu.cu"))
    d = os.path.join(cand, "src", "derivatives.h")
    s = open(d).read()
    assert s.count(FH_ANCHOR) == 2
    s = s.replace(FH_ANCHOR, FH_OUT)
    assert s.count("RHSFACE_PURE") == 4
    open(d, "w").write(s)
    patched_generic = open(os.path.join(cand, "src", "bssn_rhs_gpu.cu")).read()
    make_face_tus(cand, patched_generic)
    cm = os.path.join(cand, "CMakeLists.txt")
    s = open(cm).read()
    old = "src/bssn_rhs_gpu.cu src/bssn_rhs_gpu_int.cu"
    assert s.count(old) == 1
    s = s.replace(old, "src/bssn_rhs_gpu.cu src/bssn_rhs_gpu_int.cu src/bssn_rhs_gpu_face.cu src/bssn_rhs_gpu_facez.cu")
    open(cm, "w").write(s)
    print("26b applied (7-region + face/facez TUs + CMake)")

    # 26c: prolong3 compact
    apply_26c(os.path.join(cand, "src", "prolongrestrict_cell_gpu.cu"))
    print("26c applied (prolong3 compact)")

    print("\n=== verification ===")
    for f in ("bssn_rhs_gpu.cu", "bssn_rhs_gpu_int.cu", "derivatives.h",
              "prolongrestrict_cell_gpu.cu", "bssn_rhs_gpu_face.cu", "bssn_rhs_gpu_facez.cu"):
        print(f"  {f}: {sha(os.path.join(cand,'src',f))[:16]}")
    # face TUs must inherit the dedup
    for f in ("bssn_rhs_gpu_face.cu", "bssn_rhs_gpu_facez.cu"):
        t = open(os.path.join(cand, "src", f)).read()
        assert "val_rho" in t and "rho[idx]" in t, f"{f} dedup check"
    print("PATCH OK")


if __name__ == "__main__":
    main()
