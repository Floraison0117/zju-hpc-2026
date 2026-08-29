
#ifndef DERIVATIVES
#define DERIVATIVES

#ifdef fortran1
#define f_fderivs fderivs
#define f_fderivs_sh fderivs_sh
#define f_fderivs_shc fderivs_shc
#define f_fdderivs_shc fdderivs_shc
#define f_fdderivs fdderivs
#endif
#ifdef fortran2
#define f_fderivs FDERIVS
#define f_fderivs_sh FDERIVS_SH
#define f_fderivs_shc FDERIVS_SHC
#define f_fdderivs_shc FDDERIVS_SHC
#define f_fdderivs FDDERIVS
#endif
#ifdef fortran3
#define f_fderivs fderivs_
#define f_fderivs_sh fderivs_sh_
#define f_fderivs_shc fderivs_shc_
#define f_fdderivs_shc fdderivs_shc_
#define f_fdderivs fdderivs_
#endif

extern "C"
{
	void f_fderivs(int *, double *,
				   double *, double *, double *,
				   double *, double *, double *,
				   double &, double &, double &, int &, int &);
}

extern "C"
{
	void f_fderivs_sh(int *, double *,
					  double *, double *, double *,
					  double *, double *, double *,
					  double &, double &, double &, int &, int &, int &);
}

extern "C"
{
	void f_fderivs_shc(int *, double *,
					   double *, double *, double *,
					   double *, double *, double *,
					   double &, double &, double &, int &, int &, int &,
					   double *, double *, double *,
					   double *, double *, double *,
					   double *, double *, double *);
}

extern "C"
{
	void f_fdderivs_shc(int *, double *,
						double *, double *, double *, double *, double *, double *,
						double *, double *, double *,
						double &, double &, double &, int &, int &, int &,
						double *, double *, double *,
						double *, double *, double *,
						double *, double *, double *,
						double *, double *, double *, double *, double *, double *,
						double *, double *, double *, double *, double *, double *,
						double *, double *, double *, double *, double *, double *);
}

extern "C"
{
	void f_fdderivs(int *, double *,
					double *, double *, double *, double *, double *, double *,
					double *, double *, double *,
					double &, double &, double &, int &, int &);
}

#ifdef USE_GPU
#include <cuda_runtime.h>
#ifdef RHSFACE_PRED4
#define RHS_LOAD4(U, E) ((U) ? (E) : 0.0)
#else
#define RHS_LOAD4(U, E) (E)
#endif

__device__ __forceinline__ void d_fderivs_point(
    const int ex[3], const double* f,
    double* fx, double* fy, double* fz,
    const double* X, const double* Y, const double* Z,
    double SYM1, double SYM2, double SYM3,
    int symmetry, int onoff,
    int i, int j, int k
) {
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double EIT = 8.0;
    const double F12 = 12.0;
    const double ZEO = 0.0;
    const int NO_SYMM = 0, EQ_SYMM = 1;

    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];

    const int imax = ex[0] - 1;
    const int jmax = ex[1] - 1;
    const int kmax = ex[2] - 1;

    *fx = ZEO;
    *fy = ZEO;
    *fz = ZEO;

    // Fortran 循环范围是 1 到 ex-1，对应 CUDA 0 到 ex-2。
    // 如果 i >= imax (即 i >= ex-1)，直接返回，保持输出为 0。
    if (i >= imax || j >= jmax || k >= kmax) return;

    // --- 修复开始 ---
    // Fortran 中 imin = -1 (1-based index)。
    // 在 CUDA (0-based) 中，为了使边界点 i=0 满足 (i-2 >= imin)，
    // 即 (0-2 >= imin) -> (-2 >= imin)，imin 必须设为 -2。
    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > NO_SYMM && fabs(Z[0]) < dZ) kmin = -2; // 原代码为 -1，修正为 -2
    if (symmetry > EQ_SYMM && fabs(X[0]) < dX) imin = -2; // 原代码为 -1，修正为 -2
    if (symmetry > EQ_SYMM && fabs(Y[0]) < dY) jmin = -2; // 原代码为 -1，修正为 -2

    double SoA[3] = {SYM1, SYM2, SYM3};

    const double d12dx = ONE / F12 / dX;
    const double d12dy = ONE / F12 / dY;
    const double d12dz = ONE / F12 / dZ;

    const double d2dx = ONE / TWO / dX;
    const double d2dy = ONE / TWO / dY;
    const double d2dz = ONE / TWO / dZ;

    // Helper lambda for symmetry boundary access
    const auto fh = [&](int ii, int jj, int kk) -> double {
#ifdef RHSPROBE_INTERIOR
        // interior fast path: plain load, no in_range/valid/reflection masks
        // (bit-exact for interior points: masks are identity there, P2-verified)
        return f[((kk) * ex[1] + (jj)) * ex[0] + (ii)];
#elif defined(RHSFACE_AXIS_X) || defined(RHSFACE_AXIS_Y) || defined(RHSFACE_PURE)
        // x/y face: the two tangential axes are strictly interior.
        return f[((kk) * ex[1] + (jj)) * ex[0] + (ii)];
#elif defined(RHSFACE_AXIS_Z) || defined(RHSFACE_PURE_XY)
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
#endif

        int i1b = ii + 1, j1b = jj + 1, k1b = kk + 1;
        double in_range = 1.0;
        in_range *= (double)((i1b >= -1) & (i1b <= ex[0]));
        in_range *= (double)((j1b >= -1) & (j1b <= ex[1]));
        in_range *= (double)((k1b >= -1) & (k1b <= ex[2]));
        int i2 = i1b + (1 - 2*i1b) * (i1b <= 0);
        int j2 = j1b + (1 - 2*j1b) * (j1b <= 0);
        int k2 = k1b + (1 - 2*k1b) * (k1b <= 0);
        double fac = 1.0;
        fac *= (i1b <= 0) ? SoA[0] : 1.0;
        fac *= (j1b <= 0) ? SoA[1] : 1.0;
        fac *= (k1b <= 0) ? SoA[2] : 1.0;
        double valid = 1.0;
        valid *= (double)((i2 >= 1) & (i2 <= ex[0]));
        valid *= (double)((j2 >= 1) & (j2 <= ex[1]));
        valid *= (double)((k2 >= 1) & (k2 <= ex[2]));
        double result = f[((k2 - 1) * ex[1] + (j2 - 1)) * ex[0] + (i2 - 1)] * fac;
        return result * in_range * valid;
    };
    
    // branchless 4th/2nd order selection (remove warp divergence)
    // fh() args clamped to [imin,imax] to prevent OOB GPU memory access on fine grids.
    // Early-return above already handles boundary points (output stays ZEO).
    // At stencil-valid points, clamped == original (condition guarantees range).
    // At invalid interior points, value masked by m4/m2=0 → no effect.
#if defined(RHSFACE_AXIS_X)
    int ai_m2 = max(imin, i - 2), ai_m1 = max(imin, i - 1), ai_p1 = min(imax, i + 1), ai_p2 = min(imax, i + 2);
    int aj_m2 = j - 2, aj_m1 = j - 1, aj_p1 = j + 1, aj_p2 = j + 2;
    int ak_m2 = k - 2, ak_m1 = k - 1, ak_p1 = k + 1, ak_p2 = k + 2;
#elif defined(RHSFACE_AXIS_Y)
    int ai_m2 = i - 2, ai_m1 = i - 1, ai_p1 = i + 1, ai_p2 = i + 2;
    int aj_m2 = max(jmin, j - 2), aj_m1 = max(jmin, j - 1), aj_p1 = min(jmax, j + 1), aj_p2 = min(jmax, j + 2);
    int ak_m2 = k - 2, ak_m1 = k - 1, ak_p1 = k + 1, ak_p2 = k + 2;
#elif defined(RHSFACE_AXIS_Z)
    int ai_m2 = i - 2, ai_m1 = i - 1, ai_p1 = i + 1, ai_p2 = i + 2;
    int aj_m2 = j - 2, aj_m1 = j - 1, aj_p1 = j + 1, aj_p2 = j + 2;
    int ak_m2 = max(kmin, k - 2), ak_m1 = max(kmin, k - 1), ak_p1 = min(kmax, k + 1), ak_p2 = min(kmax, k + 2);
#else
    int ai_m2 = max(imin, i - 2), ai_m1 = max(imin, i - 1), ai_p1 = min(imax, i + 1), ai_p2 = min(imax, i + 2);
    int aj_m2 = max(jmin, j - 2), aj_m1 = max(jmin, j - 1), aj_p1 = min(jmax, j + 1), aj_p2 = min(jmax, j + 2);
    int ak_m2 = max(kmin, k - 2), ak_m1 = max(kmin, k - 1), ak_p1 = min(kmax, k + 1), ak_p2 = min(kmax, k + 2);
#endif

#ifdef RHSPROBE_INTERIOR
    const bool use4th_load = true;
#else
    bool use4th_load = (i + 2 <= imax && i - 2 >= imin && j + 2 <= jmax && j - 2 >= jmin && k + 2 <= kmax && k - 2 >= kmin);
#endif
    double h_im2 = RHS_LOAD4(use4th_load, fh(ai_m2,j,k)), h_im1 = fh(ai_m1,j,k), h_ip1 = fh(ai_p1,j,k), h_ip2 = RHS_LOAD4(use4th_load, fh(ai_p2,j,k));
    double h_jm2 = fh(i,aj_m2,k), h_jm1 = fh(i,aj_m1,k), h_jp1 = fh(i,aj_p1,k), h_jp2 = fh(i,aj_p2,k);
    double h_km2 = fh(i,j,ak_m2), h_km1 = fh(i,j,ak_m1), h_kp1 = fh(i,j,ak_p1), h_kp2 = fh(i,j,ak_p2);

#if defined(RHSFACE_AXIS_X)
    bool use4th = (i + 2 <= imax && i - 2 >= imin);
    double m4 = use4th ? 1.0 : 0.0;
    double m2 = (i + 1 <= imax && i - 1 >= imin) ? (1.0 - m4) : 0.0;
#elif defined(RHSFACE_AXIS_Y)
    bool use4th = (j + 2 <= jmax && j - 2 >= jmin);
    double m4 = use4th ? 1.0 : 0.0;
    double m2 = (j + 1 <= jmax && j - 1 >= jmin) ? (1.0 - m4) : 0.0;
#elif defined(RHSFACE_AXIS_Z)
    bool use4th = (k + 2 <= kmax && k - 2 >= kmin);
    double m4 = use4th ? 1.0 : 0.0;
    double m2 = (k + 1 <= kmax && k - 1 >= kmin) ? (1.0 - m4) : 0.0;
#else
#ifdef RHSPROBE_INTERIOR
    const bool use4th = true;
    const double m4 = 1.0;
    const double m2 = 0.0;
#else
    bool use4th = (i + 2 <= imax && i - 2 >= imin && j + 2 <= jmax && j - 2 >= jmin && k + 2 <= kmax && k - 2 >= kmin);
    double m4 = use4th ? 1.0 : 0.0;
    double m2 = (i + 1 <= imax && i - 1 >= imin && j + 1 <= jmax && j - 1 >= jmin && k + 1 <= kmax && k - 1 >= kmin) ? (1.0 - m4) : 0.0;
#endif
#endif
    *fx = m4 * d12dx * (h_im2 - EIT*h_im1 + EIT*h_ip1 - h_ip2)
        + m2 * d2dx * (-h_im1 + h_ip1);
    *fy = m4 * d12dy * (h_jm2 - EIT*h_jm1 + EIT*h_jp1 - h_jp2)
        + m2 * d2dy * (-h_jm1 + h_jp1);
    *fz = m4 * d12dz * (h_km2 - EIT*h_km1 + EIT*h_kp1 - h_kp2)
        + m2 * d2dz * (-h_km1 + h_kp1);

    (void)onoff;
}

__device__ __forceinline__ void d_fdderivs_point(
    const int ex[3], const double* f,
    double* fxx, double* fxy, double* fxz,
    double* fyy, double* fyz, double* fzz,
    const double* X, const double* Y, const double* Z,
    double SYM1, double SYM2, double SYM3,
    int symmetry, int onoff,
    int i, int j, int k
) {
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double F1o4 = 0.25;
    const double F1o12 = ONE / 12.0;
    const double F1o144 = ONE / 144.0;
    const double F8 = 8.0;
    const double F16 = 16.0;
    const double F30 = 30.0;
    const double ZEO = 0.0;
    const int NO_SYMM = 0, EQ_SYMM = 1;

    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];

    const int imax = ex[0] - 1;
    const int jmax = ex[1] - 1;
    const int kmax = ex[2] - 1;

    *fxx = ZEO; *fyy = ZEO; *fzz = ZEO;
    *fxy = ZEO; *fxz = ZEO; *fyz = ZEO;

    if (i >= imax || j >= jmax || k >= kmax) return;

    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > NO_SYMM && fabs(Z[0]) < dZ) kmin = -2;
    if (symmetry > EQ_SYMM && fabs(X[0]) < dX) imin = -2;
    if (symmetry > EQ_SYMM && fabs(Y[0]) < dY) jmin = -2;

    double SoA[3] = {SYM1, SYM2, SYM3};

    const double Sdxdx = ONE / (dX * dX);
    const double Sdydy = ONE / (dY * dY);
    const double Sdzdz = ONE / (dZ * dZ);

    const double Fdxdx = F1o12 / (dX * dX);
    const double Fdydy = F1o12 / (dY * dY);
    const double Fdzdz = F1o12 / (dZ * dZ);

    const double Sdxdy = F1o4 / (dX * dY);
    const double Sdxdz = F1o4 / (dX * dZ);
    const double Sdydz = F1o4 / (dY * dZ);

    const double Fdxdy = F1o144 / (dX * dY);
    const double Fdxdz = F1o144 / (dX * dZ);
    const double Fdydz = F1o144 / (dY * dZ);

    const auto fh = [&](int ii, int jj, int kk) -> double {
#ifdef RHSPROBE_INTERIOR
        // interior fast path: plain load, no in_range/valid/reflection masks
        // (bit-exact for interior points: masks are identity there, P2-verified)
        return f[((kk) * ex[1] + (jj)) * ex[0] + (ii)];
#elif defined(RHSFACE_AXIS_X) || defined(RHSFACE_AXIS_Y) || defined(RHSFACE_PURE)
        // x/y face: the two tangential axes are strictly interior.
        return f[((kk) * ex[1] + (jj)) * ex[0] + (ii)];
#elif defined(RHSFACE_AXIS_Z) || defined(RHSFACE_PURE_XY)
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
#endif

        int i1b = ii + 1, j1b = jj + 1, k1b = kk + 1;
        double in_range = 1.0;
        in_range *= (double)((i1b >= -1) & (i1b <= ex[0]));
        in_range *= (double)((j1b >= -1) & (j1b <= ex[1]));
        in_range *= (double)((k1b >= -1) & (k1b <= ex[2]));
        int i2 = i1b + (1 - 2*i1b) * (i1b <= 0);
        int j2 = j1b + (1 - 2*j1b) * (j1b <= 0);
        int k2 = k1b + (1 - 2*k1b) * (k1b <= 0);
        double fac = 1.0;
        fac *= (i1b <= 0) ? SoA[0] : 1.0;
        fac *= (j1b <= 0) ? SoA[1] : 1.0;
        fac *= (k1b <= 0) ? SoA[2] : 1.0;
        double valid = 1.0;
        valid *= (double)((i2 >= 1) & (i2 <= ex[0]));
        valid *= (double)((j2 >= 1) & (j2 <= ex[1]));
        valid *= (double)((k2 >= 1) & (k2 <= ex[2]));
        double result = f[((k2 - 1) * ex[1] + (j2 - 1)) * ex[0] + (i2 - 1)] * fac;
        return result * in_range * valid;
    };
    
    // branchless 4th/2nd order selection (remove warp divergence)
    // fh() args clamped to [imin,imax]x[jmin,jmax]x[kmin,kmax] to prevent
    // OOB GPU memory access on fine grids (BR_ORD lesson: step-28 illegal
    // memory access). Early-return above already zeroes boundary points.
    // At stencil-valid points clamped == original (conditions guarantee range);
    // at invalid points the value is masked by m4/m2=0 -> bit-exact.
#if defined(RHSFACE_AXIS_X)
    int ai_m2 = max(imin, i - 2), ai_m1 = max(imin, i - 1), ai_p1 = min(imax, i + 1), ai_p2 = min(imax, i + 2);
    int aj_m2 = j - 2, aj_m1 = j - 1, aj_p1 = j + 1, aj_p2 = j + 2;
    int ak_m2 = k - 2, ak_m1 = k - 1, ak_p1 = k + 1, ak_p2 = k + 2;
#elif defined(RHSFACE_AXIS_Y)
    int ai_m2 = i - 2, ai_m1 = i - 1, ai_p1 = i + 1, ai_p2 = i + 2;
    int aj_m2 = max(jmin, j - 2), aj_m1 = max(jmin, j - 1), aj_p1 = min(jmax, j + 1), aj_p2 = min(jmax, j + 2);
    int ak_m2 = k - 2, ak_m1 = k - 1, ak_p1 = k + 1, ak_p2 = k + 2;
#elif defined(RHSFACE_AXIS_Z)
    int ai_m2 = i - 2, ai_m1 = i - 1, ai_p1 = i + 1, ai_p2 = i + 2;
    int aj_m2 = j - 2, aj_m1 = j - 1, aj_p1 = j + 1, aj_p2 = j + 2;
    int ak_m2 = max(kmin, k - 2), ak_m1 = max(kmin, k - 1), ak_p1 = min(kmax, k + 1), ak_p2 = min(kmax, k + 2);
#else
    int ai_m2 = max(imin, i - 2), ai_m1 = max(imin, i - 1), ai_p1 = min(imax, i + 1), ai_p2 = min(imax, i + 2);
    int aj_m2 = max(jmin, j - 2), aj_m1 = max(jmin, j - 1), aj_p1 = min(jmax, j + 1), aj_p2 = min(jmax, j + 2);
    int ak_m2 = max(kmin, k - 2), ak_m1 = max(kmin, k - 1), ak_p1 = min(kmax, k + 1), ak_p2 = min(kmax, k + 2);
#endif

#if defined(RHSFACE_AXIS_X)
    bool use4th = (i + 2 <= imax && i - 2 >= imin);
    double m4 = use4th ? 1.0 : 0.0;
    double m2 = (i + 1 <= imax && i - 1 >= imin) ? (1.0 - m4) : 0.0;
#elif defined(RHSFACE_AXIS_Y)
    bool use4th = (j + 2 <= jmax && j - 2 >= jmin);
    double m4 = use4th ? 1.0 : 0.0;
    double m2 = (j + 1 <= jmax && j - 1 >= jmin) ? (1.0 - m4) : 0.0;
#elif defined(RHSFACE_AXIS_Z)
    bool use4th = (k + 2 <= kmax && k - 2 >= kmin);
    double m4 = use4th ? 1.0 : 0.0;
    double m2 = (k + 1 <= kmax && k - 1 >= kmin) ? (1.0 - m4) : 0.0;
#else
#ifdef RHSPROBE_INTERIOR
    const bool use4th = true;
    const double m4 = 1.0;
    const double m2 = 0.0;
#else
    bool use4th = (i + 2 <= imax && i - 2 >= imin && j + 2 <= jmax && j - 2 >= jmin && k + 2 <= kmax && k - 2 >= kmin);
    double m4 = use4th ? 1.0 : 0.0;
    double m2 = (i + 1 <= imax && i - 1 >= imin && j + 1 <= jmax && j - 1 >= jmin && k + 1 <= kmax && k - 1 >= kmin) ? (1.0 - m4) : 0.0;
#endif
#endif

    // fxx: center row, x-stencil
    double h_im2 = RHS_LOAD4(use4th, fh(ai_m2,j,k)), h_im1 = fh(ai_m1,j,k), h_000 = fh(i,j,k), h_ip1 = fh(ai_p1,j,k), h_ip2 = RHS_LOAD4(use4th, fh(ai_p2,j,k));
    *fxx = m4 * Fdxdx * (-h_im2 + F16*h_im1 - F30*h_000 - h_ip2 + F16*h_ip1)
         + m2 * Sdxdx * (h_im1 - TWO*h_000 + h_ip1);

    // fyy: center column, y-stencil
    double h_jm2 = RHS_LOAD4(use4th, fh(i,aj_m2,k)), h_jm1 = fh(i,aj_m1,k), h_jp1 = fh(i,aj_p1,k), h_jp2 = RHS_LOAD4(use4th, fh(i,aj_p2,k));
    *fyy = m4 * Fdydy * (-h_jm2 + F16*h_jm1 - F30*h_000 - h_jp2 + F16*h_jp1)
         + m2 * Sdydy * (h_jm1 - TWO*h_000 + h_jp1);

    // fzz: center line, z-stencil
    double h_km2 = RHS_LOAD4(use4th, fh(i,j,ak_m2)), h_km1 = fh(i,j,ak_m1), h_kp1 = fh(i,j,ak_p1), h_kp2 = RHS_LOAD4(use4th, fh(i,j,ak_p2));
    *fzz = m4 * Fdzdz * (-h_km2 + F16*h_km1 - F30*h_000 - h_kp2 + F16*h_kp1)
         + m2 * Sdzdz * (h_km1 - TWO*h_000 + h_kp1);

    // fxy: xy-plane (k fixed)
    double h_im2_jm2 = RHS_LOAD4(use4th, fh(ai_m2,aj_m2,k)), h_im1_jm2 = RHS_LOAD4(use4th, fh(ai_m1,aj_m2,k)), h_ip1_jm2 = RHS_LOAD4(use4th, fh(ai_p1,aj_m2,k)), h_ip2_jm2 = RHS_LOAD4(use4th, fh(ai_p2,aj_m2,k));
    double r_fxy_0 = (h_im2_jm2 - F8*h_im1_jm2 + F8*h_ip1_jm2 - h_ip2_jm2);
    double h_im2_jm1 = RHS_LOAD4(use4th, fh(ai_m2,aj_m1,k)), h_im1_jm1 = fh(ai_m1,aj_m1,k), h_ip1_jm1 = fh(ai_p1,aj_m1,k), h_ip2_jm1 = RHS_LOAD4(use4th, fh(ai_p2,aj_m1,k));
    double r_fxy_1 = (h_im2_jm1 - F8*h_im1_jm1 + F8*h_ip1_jm1 - h_ip2_jm1);
    double h_im2_jp1 = RHS_LOAD4(use4th, fh(ai_m2,aj_p1,k)), h_im1_jp1 = fh(ai_m1,aj_p1,k), h_ip1_jp1 = fh(ai_p1,aj_p1,k), h_ip2_jp1 = RHS_LOAD4(use4th, fh(ai_p2,aj_p1,k));
    double r_fxy_2 = (h_im2_jp1 - F8*h_im1_jp1 + F8*h_ip1_jp1 - h_ip2_jp1);
    double h_im2_jp2 = RHS_LOAD4(use4th, fh(ai_m2,aj_p2,k)), h_im1_jp2 = RHS_LOAD4(use4th, fh(ai_m1,aj_p2,k)), h_ip1_jp2 = RHS_LOAD4(use4th, fh(ai_p1,aj_p2,k)), h_ip2_jp2 = RHS_LOAD4(use4th, fh(ai_p2,aj_p2,k));
    double r_fxy_3 = (h_im2_jp2 - F8*h_im1_jp2 + F8*h_ip1_jp2 - h_ip2_jp2);
    *fxy = m4 * Fdxdy * (r_fxy_0 - F8*r_fxy_1 + F8*r_fxy_2 - r_fxy_3)
         + m2 * Sdxdy * (h_im1_jm1 - h_ip1_jm1 - h_im1_jp1 + h_ip1_jp1);


    // fxz: xz-plane (j fixed)
    double h_im2_km2 = RHS_LOAD4(use4th, fh(ai_m2,j,ak_m2)), h_im1_km2 = RHS_LOAD4(use4th, fh(ai_m1,j,ak_m2)), h_ip1_km2 = RHS_LOAD4(use4th, fh(ai_p1,j,ak_m2)), h_ip2_km2 = RHS_LOAD4(use4th, fh(ai_p2,j,ak_m2));
    double r_fxz_0 = (h_im2_km2 - F8*h_im1_km2 + F8*h_ip1_km2 - h_ip2_km2);
    double h_im2_km1 = RHS_LOAD4(use4th, fh(ai_m2,j,ak_m1)), h_im1_km1 = fh(ai_m1,j,ak_m1), h_ip1_km1 = fh(ai_p1,j,ak_m1), h_ip2_km1 = RHS_LOAD4(use4th, fh(ai_p2,j,ak_m1));
    double r_fxz_1 = (h_im2_km1 - F8*h_im1_km1 + F8*h_ip1_km1 - h_ip2_km1);
    double h_im2_kp1 = RHS_LOAD4(use4th, fh(ai_m2,j,ak_p1)), h_im1_kp1 = fh(ai_m1,j,ak_p1), h_ip1_kp1 = fh(ai_p1,j,ak_p1), h_ip2_kp1 = RHS_LOAD4(use4th, fh(ai_p2,j,ak_p1));
    double r_fxz_2 = (h_im2_kp1 - F8*h_im1_kp1 + F8*h_ip1_kp1 - h_ip2_kp1);
    double h_im2_kp2 = RHS_LOAD4(use4th, fh(ai_m2,j,ak_p2)), h_im1_kp2 = RHS_LOAD4(use4th, fh(ai_m1,j,ak_p2)), h_ip1_kp2 = RHS_LOAD4(use4th, fh(ai_p1,j,ak_p2)), h_ip2_kp2 = RHS_LOAD4(use4th, fh(ai_p2,j,ak_p2));
    double r_fxz_3 = (h_im2_kp2 - F8*h_im1_kp2 + F8*h_ip1_kp2 - h_ip2_kp2);
    *fxz = m4 * Fdxdz * (r_fxz_0 - F8*r_fxz_1 + F8*r_fxz_2 - r_fxz_3)
         + m2 * Sdxdz * (h_im1_km1 - h_ip1_km1 - h_im1_kp1 + h_ip1_kp1);


    // fyz: yz-plane (i fixed)
    double h_jm2_km2 = RHS_LOAD4(use4th, fh(i,aj_m2,ak_m2)), h_jm1_km2 = RHS_LOAD4(use4th, fh(i,aj_m1,ak_m2)), h_jp1_km2 = RHS_LOAD4(use4th, fh(i,aj_p1,ak_m2)), h_jp2_km2 = RHS_LOAD4(use4th, fh(i,aj_p2,ak_m2));
    double r_fyz_0 = (h_jm2_km2 - F8*h_jm1_km2 + F8*h_jp1_km2 - h_jp2_km2);
    double h_jm2_km1 = RHS_LOAD4(use4th, fh(i,aj_m2,ak_m1)), h_jm1_km1 = fh(i,aj_m1,ak_m1), h_jp1_km1 = fh(i,aj_p1,ak_m1), h_jp2_km1 = RHS_LOAD4(use4th, fh(i,aj_p2,ak_m1));
    double r_fyz_1 = (h_jm2_km1 - F8*h_jm1_km1 + F8*h_jp1_km1 - h_jp2_km1);
    double h_jm2_kp1 = RHS_LOAD4(use4th, fh(i,aj_m2,ak_p1)), h_jm1_kp1 = fh(i,aj_m1,ak_p1), h_jp1_kp1 = fh(i,aj_p1,ak_p1), h_jp2_kp1 = RHS_LOAD4(use4th, fh(i,aj_p2,ak_p1));
    double r_fyz_2 = (h_jm2_kp1 - F8*h_jm1_kp1 + F8*h_jp1_kp1 - h_jp2_kp1);
    double h_jm2_kp2 = RHS_LOAD4(use4th, fh(i,aj_m2,ak_p2)), h_jm1_kp2 = RHS_LOAD4(use4th, fh(i,aj_m1,ak_p2)), h_jp1_kp2 = RHS_LOAD4(use4th, fh(i,aj_p1,ak_p2)), h_jp2_kp2 = RHS_LOAD4(use4th, fh(i,aj_p2,ak_p2));
    double r_fyz_3 = (h_jm2_kp2 - F8*h_jm1_kp2 + F8*h_jp1_kp2 - h_jp2_kp2);
    *fyz = m4 * Fdydz * (r_fyz_0 - F8*r_fyz_1 + F8*r_fyz_2 - r_fyz_3)
         + m2 * Sdydz * (h_jm1_km1 - h_jp1_km1 - h_jm1_kp1 + h_jp1_kp1);


    (void)onoff;
}

#ifdef RHSPROBE_INTERIOR
__device__ __forceinline__ void d_fderivs_point_interior(
    const int ex[3], const double* f,
    double* fx, double* fy, double* fz,
    const double* X, const double* Y, const double* Z,
    double SYM1, double SYM2, double SYM3,
    int symmetry, int onoff,
    int i, int j, int k
) {
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double EIT = 8.0;
    const double F12 = 12.0;
    const int nx = ex[0];
    const int nxy = ex[0] * ex[1];
    const int idx = i + nx * (j + ex[1] * k);
    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];
    const double d12dx = ONE / F12 / dX;
    const double d12dy = ONE / F12 / dY;
    const double d12dz = ONE / F12 / dZ;
    const double d2dx = ONE / TWO / dX;
    const double d2dy = ONE / TWO / dY;
    const double d2dz = ONE / TWO / dZ;

    const double h_im2 = f[idx - 2], h_im1 = f[idx - 1];
    const double h_ip1 = f[idx + 1], h_ip2 = f[idx + 2];
    const double h_jm2 = f[idx - nx * 2], h_jm1 = f[idx - nx];
    const double h_jp1 = f[idx + nx], h_jp2 = f[idx + nx * 2];
    const double h_km2 = f[idx - nxy * 2], h_km1 = f[idx - nxy];
    const double h_kp1 = f[idx + nxy], h_kp2 = f[idx + nxy * 2];

    *fx = d12dx * (h_im2 - EIT * h_im1 + EIT * h_ip1 - h_ip2);
    *fy = d12dy * (h_jm2 - EIT * h_jm1 + EIT * h_jp1 - h_jp2);
    *fz = d12dz * (h_km2 - EIT * h_km1 + EIT * h_kp1 - h_kp2);
    (void)d2dx;
    (void)d2dy;
    (void)d2dz;
    (void)SYM1;
    (void)SYM2;
    (void)SYM3;
    (void)symmetry;
    (void)onoff;
}

__device__ __forceinline__ void d_fderivs_point_interior_fast(
    int idx, int nx, int nxy, const double* f,
    double* fx, double* fy, double* fz,
    double d12dx, double d12dy, double d12dz
) {
    const double h_im2 = f[idx - 2], h_im1 = f[idx - 1];
    const double h_ip1 = f[idx + 1], h_ip2 = f[idx + 2];
    const double h_jm2 = f[idx - nx * 2], h_jm1 = f[idx - nx];
    const double h_jp1 = f[idx + nx], h_jp2 = f[idx + nx * 2];
    const double h_km2 = f[idx - nxy * 2], h_km1 = f[idx - nxy];
    const double h_kp1 = f[idx + nxy], h_kp2 = f[idx + nxy * 2];
    *fx = d12dx * (h_im2 - 8.0 * h_im1 + 8.0 * h_ip1 - h_ip2);
    *fy = d12dy * (h_jm2 - 8.0 * h_jm1 + 8.0 * h_jp1 - h_jp2);
    *fz = d12dz * (h_km2 - 8.0 * h_km1 + 8.0 * h_kp1 - h_kp2);
}
#endif

#endif

#endif /* DERIVATIVES */
