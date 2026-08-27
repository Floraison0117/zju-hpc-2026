
#ifndef KODISS_H
#define KODISS_H

#ifdef USE_GPU
#include <cuda_runtime.h>

__device__ __forceinline__ double d_kodis_point(
    const int ex[3], const double* f,
    const double* X, const double* Y, const double* Z,
    double SYM1, double SYM2, double SYM3,
    int symmetry, double eps,
    int i, int j, int k // 0-based
) {
    const double ONE = 1.0;
    const double SIX = 6.0;
    const double FIT = 15.0;
    const double TWT = 20.0;
    const double cof = 64.0;
    const int NO_SYMM = 0, OCTANT = 2;

    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];

    const int imax = ex[0] - 1;
    const int jmax = ex[1] - 1;
    const int kmax = ex[2] - 1;

    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > NO_SYMM && fabs(Z[0]) < dZ) kmin = -3;
    if (symmetry == OCTANT && fabs(X[0]) < dX) imin = -3;
    if (symmetry == OCTANT && fabs(Y[0]) < dY) jmin = -3;

    double SoA[3] = {SYM1, SYM2, SYM3};

    const auto fh = [&](int ii, int jj, int kk) -> double {
#ifdef RHSPROBE_INTERIOR
        // interior fast path: plain load, no in_range/valid/reflection masks
        // (bit-exact for interior points: masks are identity there, P2-verified)
        return f[((kk) * ex[1] + (jj)) * ex[0] + (ii)];
#endif

        int i1b = ii + 1, j1b = jj + 1, k1b = kk + 1;
        if (i1b < -2 || i1b > ex[0]) return 0.0;
        if (j1b < -2 || j1b > ex[1]) return 0.0;
        if (k1b < -2 || k1b > ex[2]) return 0.0;
        int i2 = i1b, j2 = j1b, k2 = k1b;
        double fac = 1.0;
        if (i2 <= 0) { i2 = 1 - i2; fac *= SoA[0]; }
        if (j2 <= 0) { j2 = 1 - j2; fac *= SoA[1]; }
        if (k2 <= 0) { k2 = 1 - k2; fac *= SoA[2]; }
        if (i2 < 1 || i2 > ex[0]) return 0.0;
        if (j2 < 1 || j2 > ex[1]) return 0.0;
        if (k2 < 1 || k2 > ex[2]) return 0.0;
        return f[((k2 - 1) * ex[1] + (j2 - 1)) * ex[0] + (i2 - 1)] * fac;
    };

    double rhs_add = 0.0;

    if (i - 3 >= imin && i + 3 <= imax &&
        j - 3 >= jmin && j + 3 <= jmax &&
        k - 3 >= kmin && k + 3 <= kmax) {

        // P2 (iter8): field-local reuse. Inside this branch i,j,k are interior
        // (i>=imin+3, i<=imax-3), so the center stencil value == f[idx] exactly
        // (no reflection, fac=1.0, in-range). Load center once, share across
        // the three direction stencils.
        const double h_000 = f[((k) * ex[1] + (j)) * ex[0] + (i)]; // == center fh value
        rhs_add = eps / cof * (
            ((fh(i-3,j,k) + fh(i+3,j,k)) - SIX*(fh(i-2,j,k) + fh(i+2,j,k)) +
             FIT*(fh(i-1,j,k) + fh(i+1,j,k)) - TWT*h_000) / dX +
            ((fh(i,j-3,k) + fh(i,j+3,k)) - SIX*(fh(i,j-2,k) + fh(i,j+2,k)) +
             FIT*(fh(i,j-1,k) + fh(i,j+1,k)) - TWT*h_000) / dY +
            ((fh(i,j,k-3) + fh(i,j,k+3)) - SIX*(fh(i,j,k-2) + fh(i,j,k+2)) +
             FIT*(fh(i,j,k-1) + fh(i,j,k+1)) - TWT*h_000) / dZ
        );
    }

    return rhs_add;
}
#endif

#endif /* KODISS_H */
