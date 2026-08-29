#define RHSPROBE_INTERIOR
#include "bssn_rhs.h"

#include "fmisc.h"
#include "derivatives.h"
#include "kodiss.h"
#include "lopsidediff.h"
#include "gpu_manager.h"

#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <iostream>

// ==========================================
// 宏定义 (对应 macrodef.fh 和 bssn_rhs.f90)
// ==========================================
// Fortran Column-Major Layout: x varies fastest
#define IDX3D(i, j, k, nx, ny, nz) ((i) + (nx) * ((j) + (ny) * (k)))



// Advection-only interior helpers.  The caller proves the interior domain,
// so retain the original stencil branches but share the velocity, coordinate
// scales, bounds, and center index across all 24 field updates.
__device__ __forceinline__ double d_lopsided_point_int(
    const int ex[3], const double* f, double vx, double vy, double vz,
    double d12dx, double d12dy, double d12dz,
    int imax, int jmax, int kmax, int imin, int jmin, int kmin,
    int idx, int i, int j, int k
) {
    const double F3 = 3.0, F6 = 6.0, F18 = 18.0, F10 = 10.0, EIT = 8.0;
    const int sy = ex[0];
    const int sz = ex[0] * ex[1];
    const auto fh = [&](int ii, int jj, int kk) -> double {
        return f[ii + sy * (jj + ex[1] * kk)];
    };
    const double h_000 = f[idx];
    double rhs_add = 0.0;
    if (vx > 0.0) {
        if (i + 3 <= imax) {
            rhs_add += vx * d12dx * (-F3*fh(i-1,j,k) - F10*h_000 + F18*fh(i+1,j,k)
                                     -F6*fh(i+2,j,k) + fh(i+3,j,k));
        } else if (i + 2 <= imax) {
            rhs_add += vx * d12dx * (fh(i-2,j,k) - EIT*fh(i-1,j,k) + EIT*fh(i+1,j,k) - fh(i+2,j,k));
        } else if (i + 1 <= imax) {
            rhs_add -= vx * d12dx * (-F3*fh(i+1,j,k) - F10*h_000 + F18*fh(i-1,j,k)
                                     -F6*fh(i-2,j,k) + fh(i-3,j,k));
        }
    } else if (vx < 0.0) {
        if (i - 3 >= imin) {
            rhs_add -= vx * d12dx * (-F3*fh(i+1,j,k) - F10*h_000 + F18*fh(i-1,j,k)
                                     -F6*fh(i-2,j,k) + fh(i-3,j,k));
        } else if (i - 2 >= imin) {
            rhs_add += vx * d12dx * (fh(i-2,j,k) - EIT*fh(i-1,j,k) + EIT*fh(i+1,j,k) - fh(i+2,j,k));
        } else if (i - 1 >= imin) {
            rhs_add += vx * d12dx * (-F3*fh(i-1,j,k) - F10*h_000 + F18*fh(i+1,j,k)
                                     -F6*fh(i+2,j,k) + fh(i+3,j,k));
        }
    }
    if (vy > 0.0) {
        if (j + 3 <= jmax) {
            rhs_add += vy * d12dy * (-F3*fh(i,j-1,k) - F10*h_000 + F18*fh(i,j+1,k)
                                     -F6*fh(i,j+2,k) + fh(i,j+3,k));
        } else if (j + 2 <= jmax) {
            rhs_add += vy * d12dy * (fh(i,j-2,k) - EIT*fh(i,j-1,k) + EIT*fh(i,j+1,k) - fh(i,j+2,k));
        } else if (j + 1 <= jmax) {
            rhs_add -= vy * d12dy * (-F3*fh(i,j+1,k) - F10*h_000 + F18*fh(i,j-1,k)
                                     -F6*fh(i,j-2,k) + fh(i,j-3,k));
        }
    } else if (vy < 0.0) {
        if (j - 3 >= jmin) {
            rhs_add -= vy * d12dy * (-F3*fh(i,j+1,k) - F10*h_000 + F18*fh(i,j-1,k)
                                     -F6*fh(i,j-2,k) + fh(i,j-3,k));
        } else if (j - 2 >= jmin) {
            rhs_add += vy * d12dy * (fh(i,j-2,k) - EIT*fh(i,j-1,k) + EIT*fh(i,j+1,k) - fh(i,j+2,k));
        } else if (j - 1 >= jmin) {
            rhs_add += vy * d12dy * (-F3*fh(i,j-1,k) - F10*h_000 + F18*fh(i,j+1,k)
                                     -F6*fh(i,j+2,k) + fh(i,j+3,k));
        }
    }
    if (vz > 0.0) {
        if (k + 3 <= kmax) {
            rhs_add += vz * d12dz * (-F3*fh(i,j,k-1) - F10*h_000 + F18*fh(i,j,k+1)
                                     -F6*fh(i,j,k+2) + fh(i,j,k+3));
        } else if (k + 2 <= kmax) {
            rhs_add += vz * d12dz * (fh(i,j,k-2) - EIT*fh(i,j,k-1) + EIT*fh(i,j,k+1) - fh(i,j,k+2));
        } else if (k + 1 <= kmax) {
            rhs_add -= vz * d12dz * (-F3*fh(i,j,k+1) - F10*h_000 + F18*fh(i,j,k-1)
                                     -F6*fh(i,j,k-2) + fh(i,j,k-3));
        }
    } else if (vz < 0.0) {
        if (k - 3 >= kmin) {
            rhs_add -= vz * d12dz * (-F3*fh(i,j,k+1) - F10*h_000 + F18*fh(i,j,k-1)
                                     -F6*fh(i,j,k-2) + fh(i,j,k-3));
        } else if (k - 2 >= kmin) {
            rhs_add += vz * d12dz * (fh(i,j,k-2) - EIT*fh(i,j,k-1) + EIT*fh(i,j,k+1) - fh(i,j,k+2));
        } else if (k - 1 >= kmin) {
            rhs_add += vz * d12dz * (-F3*fh(i,j,k-1) - F10*h_000 + F18*fh(i,j,k+1)
                                     -F6*fh(i,j,k+2) + fh(i,j,k+3));
        }
    }
    return rhs_add;
}

__device__ __forceinline__ double d_lopsided_point_int_fast(
    const double* f, double vx, double vy, double vz,
    double d12dx, double d12dy, double d12dz,
    int nx, int nxy,
    int imax, int jmax, int kmax, int imin, int jmin, int kmin,
    int idx, int i, int j, int k
) {
    const double F3 = 3.0, F6 = 6.0, F18 = 18.0, F10 = 10.0, EIT = 8.0;
    const double h_000 = f[idx];
    double rhs_add = 0.0;
    if (vx > 0.0) {
        if (i + 3 <= imax) {
            rhs_add += vx * d12dx * (-F3*f[idx-1] - F10*h_000 + F18*f[idx+1]
                                     -F6*f[idx+2] + f[idx+3]);
        } else if (i + 2 <= imax) {
            rhs_add += vx * d12dx * (f[idx-2] - EIT*f[idx-1] + EIT*f[idx+1] - f[idx+2]);
        } else if (i + 1 <= imax) {
            rhs_add -= vx * d12dx * (-F3*f[idx+1] - F10*h_000 + F18*f[idx-1]
                                     -F6*f[idx-2] + f[idx-3]);
        }
    } else if (vx < 0.0) {
        if (i - 3 >= imin) {
            rhs_add -= vx * d12dx * (-F3*f[idx+1] - F10*h_000 + F18*f[idx-1]
                                     -F6*f[idx-2] + f[idx-3]);
        } else if (i - 2 >= imin) {
            rhs_add += vx * d12dx * (f[idx-2] - EIT*f[idx-1] + EIT*f[idx+1] - f[idx+2]);
        } else if (i - 1 >= imin) {
            rhs_add += vx * d12dx * (-F3*f[idx-1] - F10*h_000 + F18*f[idx+1]
                                     -F6*f[idx+2] + f[idx+3]);
        }
    }
    if (vy > 0.0) {
        if (j + 3 <= jmax) {
            rhs_add += vy * d12dy * (-F3*f[idx-nx] - F10*h_000 + F18*f[idx+nx]
                                     -F6*f[idx+2*nx] + f[idx+3*nx]);
        } else if (j + 2 <= jmax) {
            rhs_add += vy * d12dy * (f[idx-2*nx] - EIT*f[idx-nx] + EIT*f[idx+nx] - f[idx+2*nx]);
        } else if (j + 1 <= jmax) {
            rhs_add -= vy * d12dy * (-F3*f[idx+nx] - F10*h_000 + F18*f[idx-nx]
                                     -F6*f[idx-2*nx] + f[idx-3*nx]);
        }
    } else if (vy < 0.0) {
        if (j - 3 >= jmin) {
            rhs_add -= vy * d12dy * (-F3*f[idx+nx] - F10*h_000 + F18*f[idx-nx]
                                     -F6*f[idx-2*nx] + f[idx-3*nx]);
        } else if (j - 2 >= jmin) {
            rhs_add += vy * d12dy * (f[idx-2*nx] - EIT*f[idx-nx] + EIT*f[idx+nx] - f[idx+2*nx]);
        } else if (j - 1 >= jmin) {
            rhs_add += vy * d12dy * (-F3*f[idx-nx] - F10*h_000 + F18*f[idx+nx]
                                     -F6*f[idx+2*nx] + f[idx+3*nx]);
        }
    }
    if (vz > 0.0) {
        if (k + 3 <= kmax) {
            rhs_add += vz * d12dz * (-F3*f[idx-nxy] - F10*h_000 + F18*f[idx+nxy]
                                     -F6*f[idx+2*nxy] + f[idx+3*nxy]);
        } else if (k + 2 <= kmax) {
            rhs_add += vz * d12dz * (f[idx-2*nxy] - EIT*f[idx-nxy] + EIT*f[idx+nxy] - f[idx+2*nxy]);
        } else if (k + 1 <= kmax) {
            rhs_add -= vz * d12dz * (-F3*f[idx+nxy] - F10*h_000 + F18*f[idx-nxy]
                                     -F6*f[idx-2*nxy] + f[idx-3*nxy]);
        }
    } else if (vz < 0.0) {
        if (k - 3 >= kmin) {
            rhs_add -= vz * d12dz * (-F3*f[idx+nxy] - F10*h_000 + F18*f[idx-nxy]
                                     -F6*f[idx-2*nxy] + f[idx-3*nxy]);
        } else if (k - 2 >= kmin) {
            rhs_add += vz * d12dz * (f[idx-2*nxy] - EIT*f[idx-nxy] + EIT*f[idx+nxy] - f[idx+2*nxy]);
        } else if (k - 1 >= kmin) {
            rhs_add += vz * d12dz * (-F3*f[idx-nxy] - F10*h_000 + F18*f[idx+nxy]
                                     -F6*f[idx+2*nxy] + f[idx+3*nxy]);
        }
    }
    return rhs_add;
}

__device__ __forceinline__ double d_kodis_point_int_fast(
    const double* f, double dX, double dY, double dZ, double eps_over_cof,
    int nx, int nxy,
    int imax, int jmax, int kmax, int imin, int jmin, int kmin,
    int idx, int i, int j, int k
) {
    if (i - 3 < imin || i + 3 > imax ||
        j - 3 < jmin || j + 3 > jmax ||
        k - 3 < kmin || k + 3 > kmax) return 0.0;
    const double h_000 = f[idx];
    return eps_over_cof * (
        ((f[idx-3] + f[idx+3]) - 6.0*(f[idx-2] + f[idx+2]) +
         15.0*(f[idx-1] + f[idx+1]) - 20.0*h_000) / dX +
        ((f[idx-3*nx] + f[idx+3*nx]) - 6.0*(f[idx-2*nx] + f[idx+2*nx]) +
         15.0*(f[idx-nx] + f[idx+nx]) - 20.0*h_000) / dY +
        ((f[idx-3*nxy] + f[idx+3*nxy]) - 6.0*(f[idx-2*nxy] + f[idx+2*nxy]) +
         15.0*(f[idx-nxy] + f[idx+nxy]) - 20.0*h_000) / dZ);
}

__device__ __forceinline__ double d_kodis_point_int(
    const int ex[3], const double* f,
    double dX, double dY, double dZ, double eps_over_cof,
    int imax, int jmax, int kmax, int imin, int jmin, int kmin,
    int idx, int i, int j, int k
) {
    if (i - 3 < imin || i + 3 > imax ||
        j - 3 < jmin || j + 3 > jmax ||
        k - 3 < kmin || k + 3 > kmax) return 0.0;
    const int sy = ex[0];
    const int sz = ex[0] * ex[1];
    const auto fh = [&](int ii, int jj, int kk) -> double {
        return f[ii + sy * (jj + ex[1] * kk)];
    };
    const double h_000 = f[idx];
    return eps_over_cof * (
        ((fh(i-3,j,k) + fh(i+3,j,k)) - 6.0*(fh(i-2,j,k) + fh(i+2,j,k)) +
         15.0*(fh(i-1,j,k) + fh(i+1,j,k)) - 20.0*h_000) / dX +
        ((fh(i,j-3,k) + fh(i,j+3,k)) - 6.0*(fh(i,j-2,k) + fh(i,j+2,k)) +
         15.0*(fh(i,j-1,k) + fh(i,j+1,k)) - 20.0*h_000) / dY +
        ((fh(i,j,k-3) + fh(i,j,k+3)) - 6.0*(fh(i,j,k-2) + fh(i,j,k+2)) +
         15.0*(fh(i,j,k-1) + fh(i,j,k+1)) - 20.0*h_000) / dZ);
}

constexpr double SYM = 1.0;
constexpr double ANTI = -1.0;
constexpr double ZEO = 0.0;
constexpr double ONE = 1.0;
constexpr double TWO = 2.0;
constexpr double FOUR = 4.0;
constexpr double EIGHT = 8.0;
constexpr double PI = M_PI;
constexpr double F1o3 = 1.0 / 3.0;
constexpr double F2o3 = 2.0 / 3.0;
constexpr double F3o2 = 1.5;
constexpr double HALF = 0.5;
constexpr double FF = 0.75;
constexpr double eta = 2.0;
constexpr double F8 = 8.0;
constexpr double F16 = 16.0;

__global__ __launch_bounds__(128, 1) void rhs_kernel_int_stage1(
    int ex0, int ex1, int ex2, double T, double* __restrict__ X, double* __restrict__ Y, double* __restrict__ Z,
    double* __restrict__ chi, double* __restrict__ trK,
    double* __restrict__ dxx, double* __restrict__ gxy, double* __restrict__ gxz,
    double* __restrict__ dyy, double* __restrict__ gyz, double* __restrict__ dzz,
    double* __restrict__ Axx, double* __restrict__ Axy, double* __restrict__ Axz,
    double* __restrict__ Ayy, double* __restrict__ Ayz, double* __restrict__ Azz,
    double* __restrict__ Gamx, double* __restrict__ Gamy, double* __restrict__ Gamz,
    double* __restrict__ Lap,
    double* __restrict__ betax, double* __restrict__ betay, double* __restrict__ betaz,
    double* __restrict__ dtSfx, double* __restrict__ dtSfy, double* __restrict__ dtSfz,
    double* __restrict__ chi_rhs, double* __restrict__ trK_rhs,
    double* __restrict__ gxx_rhs, double* __restrict__ gxy_rhs, double* __restrict__ gxz_rhs,
    double* __restrict__ gyy_rhs, double* __restrict__ gyz_rhs, double* __restrict__ gzz_rhs,
    double* __restrict__ Axx_rhs, double* __restrict__ Axy_rhs, double* __restrict__ Axz_rhs,
    double* __restrict__ Ayy_rhs, double* __restrict__ Ayz_rhs, double* __restrict__ Azz_rhs,
    double* __restrict__ Gamx_rhs, double* __restrict__ Gamy_rhs, double* __restrict__ Gamz_rhs,
    double* __restrict__ Lap_rhs,
    double* __restrict__ betax_rhs, double* __restrict__ betay_rhs, double* __restrict__ betaz_rhs,
    double* __restrict__ dtSfx_rhs, double* __restrict__ dtSfy_rhs, double* __restrict__ dtSfz_rhs,
    double* __restrict__ rho, double* __restrict__ Sx, double* __restrict__ Sy, double* __restrict__ Sz,
    double* __restrict__ Sxx, double* __restrict__ Sxy, double* __restrict__ Sxz,
    double* __restrict__ Syy, double* __restrict__ Syz, double* __restrict__ Szz,
    double* __restrict__ Gamxxx, double* __restrict__ Gamxxy, double* __restrict__ Gamxxz,
    double* __restrict__ Gamxyy, double* __restrict__ Gamxyz, double* __restrict__ Gamxzz,
    double* __restrict__ Gamyxx, double* __restrict__ Gamyxy, double* __restrict__ Gamyxz,
    double* __restrict__ Gamyyy, double* __restrict__ Gamyyz, double* __restrict__ Gamyzz,
    double* __restrict__ Gamzxx, double* __restrict__ Gamzxy, double* __restrict__ Gamzxz,
    double* __restrict__ Gamzyy, double* __restrict__ Gamzyz, double* __restrict__ Gamzzz,
    double* __restrict__ Rxx, double* __restrict__ Rxy, double* __restrict__ Rxz,
    double* __restrict__ Ryy, double* __restrict__ Ryz, double* __restrict__ Rzz,
    double* __restrict__ ham_Res, double* __restrict__ movx_Res, double* __restrict__ movy_Res, double* __restrict__ movz_Res,
    double* __restrict__ Gmx_Res, double* __restrict__ Gmy_Res, double* __restrict__ Gmz_Res,
    int symmetry, int lev, double eps, int co
) {
    // 计算全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 2;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 3;

    // 越界检查
    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    // interior-only fast path: skip the boundary shell.
    // margins: i,j = 2 (imin=jmin=0 -> no reflection reachable at i,j>=2);
    // k lower = 3 (equatorial symmetry -> kmin=-3 -> lopsided/kodis reflect at
    // k<=2, plain-load fh would read OOB); k upper = 2.
    if (i < 2 || i > ex0 - 3 || j < 2 || j > ex1 - 3 || k < 3 || k > ex2 - 3) return;

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    int dims[3] = {ex0, ex1, ex2}; // 用于传给 device 函数
    const int nx = ex0;
    const int nxy = ex0 * ex1;
    const double d12dx = 1.0 / (12.0 * (X[1] - X[0]));
    const double d12dy = 1.0 / (12.0 * (Y[1] - Y[0]));
    const double d12dz = 1.0 / (12.0 * (Z[1] - Z[0]));

    // Stage 1 keeps only geometry construction.  Its outputs are the raw
    // Christoffel symbols and the fully corrected Ricci tensor; the second
    // stage consumes these arrays on the same CUDA stream.
    double val_Lap = Lap[idx];
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

    double l_gxx = val_gxx; double l_gxy = val_gxy; double l_gxz = val_gxz;
    double l_gyy = val_gyy; double l_gyz = val_gyz; double l_gzz = val_gzz;

    double chix, chiy, chiz;
    d_fderivs_point_interior_fast(idx, nx, nxy, chi, &chix, &chiy, &chiz, d12dx, d12dy, d12dz);

    double gxxx, gxxy, gxxz;
    double gxyx, gxyy, gxyz;
    double gxzx, gxzy, gxzz;
    double gyyx, gyyy, gyyz;
    double gyzx, gyzy, gyzz;
    double gzzx, gzzy, gzzz;

    d_fderivs_point_interior_fast(idx, nx, nxy, dxx, &gxxx, &gxxy, &gxxz, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, gxy, &gxyx, &gxyy, &gxyz, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, gxz, &gxzx, &gxzy, &gxzz, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, dyy, &gyyx, &gyyy, &gyyz, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, gyz, &gyzx, &gyzy, &gyzz, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, dzz, &gzzx, &gzzy, &gzzz, d12dx, d12dy, d12dz);
    double gupzz = val_gxx * val_gyy * val_gzz + val_gxy * val_gyz * val_gxz + val_gxz * val_gxy * val_gyz -
                   val_gxz * val_gyy * val_gxz - val_gxy * val_gxy * val_gzz - val_gxx * val_gyz * val_gyz;
    
    double gupxx = (val_gyy * val_gzz - val_gyz * val_gyz) / gupzz;
    double gupxy = -(val_gxy * val_gzz - val_gyz * val_gxz) / gupzz;
    double gupxz = (val_gxy * val_gyz - val_gyy * val_gxz) / gupzz;
    double gupyy = (val_gxx * val_gzz - val_gxz * val_gxz) / gupzz;
    double gupyz = -(val_gxx * val_gyz - val_gxy * val_gxz) / gupzz;
    gupzz = (val_gxx * val_gyy - val_gxy * val_gxy) / gupzz; // 更新 gupzz 为逆分量
    if (co == 0) {
        // Gmx_Res
        double term_x = gupxx*(gupxx*gxxx+gupxy*gxyx+gupxz*gxzx)
                      + gupxy*(gupxx*gxyx+gupxy*gyyx+gupxz*gyzx)
                      + gupxz*(gupxx*gxzx+gupxy*gyzx+gupxz*gzzx)
                      + gupxx*(gupxy*gxxy+gupyy*gxyy+gupyz*gxzy)
                      + gupxy*(gupxy*gxyy+gupyy*gyyy+gupyz*gyzy)
                      + gupxz*(gupxy*gxzy+gupyy*gyzy+gupyz*gzzy)
                      + gupxx*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)
                      + gupxy*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)
                      + gupxz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
        Gmx_Res[idx] = Gamx[idx] - term_x;

        // Gmy_Res
        double term_y = gupxx*(gupxy*gxxx+gupyy*gxyx+gupyz*gxzx)
                      + gupxy*(gupxy*gxyx+gupyy*gyyx+gupyz*gyzx)
                      + gupxz*(gupxy*gxzx+gupyy*gyzx+gupyz*gzzx)
                      + gupxy*(gupxy*gxxy+gupyy*gxyy+gupyz*gxzy)
                      + gupyy*(gupxy*gxyy+gupyy*gyyy+gupyz*gyzy)
                      + gupyz*(gupxy*gxzy+gupyy*gyzy+gupyz*gzzy)
                      + gupxy*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)
                      + gupyy*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)
                      + gupyz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
        Gmy_Res[idx] = Gamy[idx] - term_y;

        // Gmz_Res
        double term_z = gupxx*(gupxz*gxxx+gupyz*gxyx+gupzz*gxzx)
                      + gupxy*(gupxz*gxyx+gupyz*gyyx+gupzz*gyzx)
                      + gupxz*(gupxz*gxzx+gupyz*gyzx+gupzz*gzzx)
                      + gupxy*(gupxz*gxxy+gupyz*gxyy+gupzz*gxzy)
                      + gupyy*(gupxz*gxyy+gupyz*gyyy+gupzz*gyzy)
                      + gupyz*(gupxz*gxzy+gupyz*gyzy+gupzz*gzzy)
                      + gupxz*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)
                      + gupyz*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)
                      + gupzz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz);
        Gmz_Res[idx] = Gamz[idx] - term_z;
    }
    double l_Gamxxx; double l_Gamxxy; double l_Gamxxz;
    double l_Gamxyy; double l_Gamxyz; double l_Gamxzz;
    double l_Gamyxx; double l_Gamyxy; double l_Gamyxz;
    double l_Gamyyy; double l_Gamyyz; double l_Gamyzz;
    double l_Gamzxx; double l_Gamzxy; double l_Gamzxz;
    double l_Gamzyy; double l_Gamzyz; double l_Gamzzz;

    l_Gamxxx = HALF * (gupxx * gxxx + gupxy * (TWO * gxyx - gxxy) + gupxz * (TWO * gxzx - gxxz));
    l_Gamyxx = HALF * (gupxy * gxxx + gupyy * (TWO * gxyx - gxxy) + gupyz * (TWO * gxzx - gxxz));
    l_Gamzxx = HALF * (gupxz * gxxx + gupyz * (TWO * gxyx - gxxy) + gupzz * (TWO * gxzx - gxxz));

    l_Gamxyy = HALF * (gupxx * (TWO * gxyy - gyyx) + gupxy * gyyy + gupxz * (TWO * gyzy - gyyz));
    l_Gamyyy = HALF * (gupxy * (TWO * gxyy - gyyx) + gupyy * gyyy + gupyz * (TWO * gyzy - gyyz));
    l_Gamzyy = HALF * (gupxz * (TWO * gxyy - gyyx) + gupyz * gyyy + gupzz * (TWO * gyzy - gyyz));

    l_Gamxzz = HALF * (gupxx * (TWO * gxzz - gzzx) + gupxy * (TWO * gyzz - gzzy) + gupxz * gzzz);
    l_Gamyzz = HALF * (gupxy * (TWO * gxzz - gzzx) + gupyy * (TWO * gyzz - gzzy) + gupyz * gzzz);
    l_Gamzzz = HALF * (gupxz * (TWO * gxzz - gzzx) + gupyz * (TWO * gyzz - gzzy) + gupzz * gzzz);

    l_Gamxxy = HALF * (gupxx * gxxy + gupxy * gyyx + gupxz * (gxzy + gyzx - gxyz));
    l_Gamyxy = HALF * (gupxy * gxxy + gupyy * gyyx + gupyz * (gxzy + gyzx - gxyz));
    l_Gamzxy = HALF * (gupxz * gxxy + gupyz * gyyx + gupzz * (gxzy + gyzx - gxyz));

    l_Gamxxz = HALF * (gupxx * gxxz + gupxy * (gxyz + gyzx - gxzy) + gupxz * gzzx);
    l_Gamyxz = HALF * (gupxy * gxxz + gupyy * (gxyz + gyzx - gxzy) + gupyz * gzzx);
    l_Gamzxz = HALF * (gupxz * gxxz + gupyz * (gxyz + gyzx - gxzy) + gupzz * gzzx);

    l_Gamxyz = HALF * (gupxx * (gxyz + gxzy - gyzx) + gupxy * gyyz + gupxz * gzzy);
    l_Gamyyz = HALF * (gupxy * (gxyz + gxzy - gyzx) + gupyy * gyyz + gupyz * gzzy);
    l_Gamzyz = HALF * (gupxz * (gxyz + gxzy - gyzx) + gupyz * gyyz + gupzz * gzzy);

    double l_Rxx, l_Rxy, l_Rxz, l_Ryy, l_Ryz, l_Rzz;
    double fxx, fxy, fxz, fyy, fyz, fzz;
    double Gamxa = gupxx * l_Gamxxx + gupyy * l_Gamxyy + gupzz * l_Gamxzz +
                   TWO * (gupxy * l_Gamxxy + gupxz * l_Gamxxz + gupyz * l_Gamxyz);
    double Gamya = gupxx * l_Gamyxx + gupyy * l_Gamyyy + gupzz * l_Gamyzz +
                   TWO * (gupxy * l_Gamyxy + gupxz * l_Gamyxz + gupyz * l_Gamyyz);
    double Gamza = gupxx * l_Gamzxx + gupyy * l_Gamzyy + gupzz * l_Gamzzz +
                   TWO * (gupxy * l_Gamzxy + gupxz * l_Gamzxz + gupyz * l_Gamzyz);
    
    double dGamxx, dGamxy, dGamxz;
    double dGamyx, dGamyy, dGamyz;
    double dGamzx, dGamzy, dGamzz;
    d_fderivs_point_interior_fast(idx, nx, nxy, Gamx, &dGamxx, &dGamxy, &dGamxz, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, Gamy, &dGamyx, &dGamyy, &dGamyz, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, Gamz, &dGamzx, &dGamzy, &dGamzz, d12dx, d12dy, d12dz);
    gxxx = l_gxx * l_Gamxxx + l_gxy * l_Gamyxx + l_gxz * l_Gamzxx;
    gxyx = l_gxx * l_Gamxxy + l_gxy * l_Gamyxy + l_gxz * l_Gamzxy;
    gxzx = l_gxx * l_Gamxxz + l_gxy * l_Gamyxz + l_gxz * l_Gamzxz;
    gyyx = l_gxx * l_Gamxyy + l_gxy * l_Gamyyy + l_gxz * l_Gamzyy;
    gyzx = l_gxx * l_Gamxyz + l_gxy * l_Gamyyz + l_gxz * l_Gamzyz;
    gzzx = l_gxx * l_Gamxzz + l_gxy * l_Gamyzz + l_gxz * l_Gamzzz;

    gxxy = l_gxy * l_Gamxxx + l_gyy * l_Gamyxx + l_gyz * l_Gamzxx;
    gxyy = l_gxy * l_Gamxxy + l_gyy * l_Gamyxy + l_gyz * l_Gamzxy;
    gxzy = l_gxy * l_Gamxxz + l_gyy * l_Gamyxz + l_gyz * l_Gamzxz;
    gyyy = l_gxy * l_Gamxyy + l_gyy * l_Gamyyy + l_gyz * l_Gamzyy;
    gyzy = l_gxy * l_Gamxyz + l_gyy * l_Gamyyz + l_gyz * l_Gamzyz;
    gzzy = l_gxy * l_Gamxzz + l_gyy * l_Gamyzz + l_gyz * l_Gamzzz;

    gxxz = l_gxz * l_Gamxxx + l_gyz * l_Gamyxx + l_gzz * l_Gamzxx;
    gxyz = l_gxz * l_Gamxxy + l_gyz * l_Gamyxy + l_gzz * l_Gamzxy;
    gxzz = l_gxz * l_Gamxxz + l_gyz * l_Gamyxz + l_gzz * l_Gamzxz;
    gyyz = l_gxz * l_Gamxyy + l_gyz * l_Gamyyy + l_gzz * l_Gamzyy;
    gyzz = l_gxz * l_Gamxyz + l_gyz * l_Gamyyz + l_gzz * l_Gamzyz;
    gzzz = l_gxz * l_Gamxzz + l_gyz * l_Gamyzz + l_gzz * l_Gamzzz;
    
    // Rxx
    d_fdderivs_point(dims, dxx, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    l_Rxx = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;
    // Ryy
    d_fdderivs_point(dims, dyy, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    l_Ryy = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    // Rzz
    d_fdderivs_point(dims, dzz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    l_Rzz = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    // Rxy
    d_fdderivs_point(dims, gxy, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, ANTI, ANTI, SYM, symmetry, lev, i, j, k);
    l_Rxy = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    // Rxz
    d_fdderivs_point(dims, gxz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, ANTI, SYM, ANTI, symmetry, lev, i, j, k);
    l_Rxz = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    // Ryz
    d_fdderivs_point(dims, gyz, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, ANTI, ANTI, symmetry, lev, i, j, k);
    l_Ryz = gupxx * fxx + gupyy * fyy + gupzz * fzz + (gupxy * fxy + gupxz * fxz + gupyz * fyz) * TWO;

    // ==========================================
    // Step 4: Ricci (连接系数项) - 完整展开
    // ==========================================

    // double Gam_dot_dg_xx = Gamxa * gxxx + Gamya * gxyx + Gamza * gxzx;
    // double Gam_dot_dg_yy = Gamxa * gxyy + Gamya * gyyy + Gamza * gyzy;
    // double Gam_dot_dg_zz = Gamxa * gxzz + Gamya * gyzz + Gamza * gzzz;

    // Rxx Correction
    l_Rxx = -HALF * l_Rxx + 
          l_gxx * dGamxx + l_gxy * dGamyx + l_gxz * dGamzx + 
          Gamxa * gxxx + Gamya * gxyx + Gamza * gxzx + 
          gupxx * (TWO*(l_Gamxxx*gxxx + l_Gamyxx*gxyx + l_Gamzxx*gxzx) + l_Gamxxx*gxxx + l_Gamyxx*gxxy + l_Gamzxx*gxxz) +
          gupxy * (TWO*(l_Gamxxx*gxyx + l_Gamyxx*gyyx + l_Gamzxx*gyzx + l_Gamxxy*gxxx + l_Gamyxy*gxyx + l_Gamzxy*gxzx) + l_Gamxxy*gxxx + l_Gamyxy*gxxy + l_Gamzxy*gxxz + l_Gamxxx*gxyx + l_Gamyxx*gxyy + l_Gamzxx*gxyz) + 
          gupxz * (TWO*(l_Gamxxx*gxzx + l_Gamyxx*gyzx + l_Gamzxx*gzzx + l_Gamxxz*gxxx + l_Gamyxz*gxyx + l_Gamzxz*gxzx) + l_Gamxxz*gxxx + l_Gamyxz*gxxy + l_Gamzxz*gxxz + l_Gamxxx*gxzx + l_Gamyxx*gxzy + l_Gamzxx*gxzz) + 
          gupyy * (TWO*(l_Gamxxy*gxyx + l_Gamyxy*gyyx + l_Gamzxy*gyzx) + l_Gamxxy*gxyx + l_Gamyxy*gxyy + l_Gamzxy*gxyz) + 
          gupyz * (TWO*(l_Gamxxy*gxzx + l_Gamyxy*gyzx + l_Gamzxy*gzzx + l_Gamxxz*gxyx + l_Gamyxz*gyyx + l_Gamzxz*gyzx) + l_Gamxxz*gxyx + l_Gamyxz*gxyy + l_Gamzxz*gxyz + l_Gamxxy*gxzx + l_Gamyxy*gxzy + l_Gamzxy*gxzz) + 
          gupzz * (TWO*(l_Gamxxz*gxzx + l_Gamyxz*gyzx + l_Gamzxz*gzzx) + l_Gamxxz*gxzx + l_Gamyxz*gxzy + l_Gamzxz*gxzz);

    // Ryy Correction
    l_Ryy = -HALF * l_Ryy + 
          l_gxy * dGamxy + l_gyy * dGamyy + l_gyz * dGamzy + 
          Gamxa * gxyy + Gamya * gyyy + Gamza * gyzy + 
          gupxx * (TWO*(l_Gamxxy*gxxy + l_Gamyxy*gxyy + l_Gamzxy*gxzy) + l_Gamxxy*gxyx + l_Gamyxy*gxyy + l_Gamzxy*gxyz) + 
          gupxy * (TWO*(l_Gamxxy*gxyy + l_Gamyxy*gyyy + l_Gamzxy*gyzy + l_Gamxyy*gxxy + l_Gamyyy*gxyy + l_Gamzyy*gxzy) + l_Gamxyy*gxyx + l_Gamyyy*gxyy + l_Gamzyy*gxyz + l_Gamxxy*gyyx + l_Gamyxy*gyyy + l_Gamzxy*gyyz) + 
          gupxz * (TWO*(l_Gamxxy*gxzy + l_Gamyxy*gyzy + l_Gamzxy*gzzy + l_Gamxyz*gxxy + l_Gamyyz*gxyy + l_Gamzyz*gxzy) + l_Gamxyz*gxyx + l_Gamyyz*gxyy + l_Gamzyz*gxyz + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) + 
          gupyy * (TWO*(l_Gamxyy*gxyy + l_Gamyyy*gyyy + l_Gamzyy*gyzy) + l_Gamxyy*gyyx + l_Gamyyy*gyyy + l_Gamzyy*gyyz) + 
          gupyz * (TWO*(l_Gamxyy*gxzy + l_Gamyyy*gyzy + l_Gamzyy*gzzy + l_Gamxyz*gxyy + l_Gamyyz*gyyy + l_Gamzyz*gyzy) + l_Gamxyz*gyyx + l_Gamyyz*gyyy + l_Gamzyz*gyyz + l_Gamxyy*gyzx + l_Gamyyy*gyzy + l_Gamzyy*gyzz) + 
          gupzz * (TWO*(l_Gamxyz*gxzy + l_Gamyyz*gyzy + l_Gamzyz*gzzy) + l_Gamxyz*gyzx + l_Gamyyz*gyzy + l_Gamzyz*gyzz);

    // Rzz Correction
    l_Rzz = -HALF * l_Rzz + 
          l_gxz * dGamxz + l_gyz * dGamyz + l_gzz * dGamzz + 
          Gamxa * gxzz + Gamya * gyzz + Gamza * gzzz + 
          gupxx * (TWO*(l_Gamxxz*gxxz + l_Gamyxz*gxyz + l_Gamzxz*gxzz) + l_Gamxxz*gxzx + l_Gamyxz*gxzy + l_Gamzxz*gxzz) + 
          gupxy * (TWO*(l_Gamxxz*gxyz + l_Gamyxz*gyyz + l_Gamzxz*gyzz + l_Gamxyz*gxxz + l_Gamyyz*gxyz + l_Gamzyz*gxzz) + l_Gamxyz*gxzx + l_Gamyyz*gxzy + l_Gamzyz*gxzz + l_Gamxxz*gyzx + l_Gamyxz*gyzy + l_Gamzxz*gyzz) + 
          gupxz * (TWO*(l_Gamxxz*gxzz + l_Gamyxz*gyzz + l_Gamzxz*gzzz + l_Gamxzz*gxxz + l_Gamyzz*gxyz + l_Gamzzz*gxzz) + l_Gamxzz*gxzx + l_Gamyzz*gxzy + l_Gamzzz*gxzz + l_Gamxxz*gzzx + l_Gamyxz*gzzy + l_Gamzxz*gzzz) + 
          gupyy * (TWO*(l_Gamxyz*gxyz + l_Gamyyz*gyyz + l_Gamzyz*gyzz) + l_Gamxyz*gyzx + l_Gamyyz*gyzy + l_Gamzyz*gyzz) + 
          gupyz * (TWO*(l_Gamxyz*gxzz + l_Gamyyz*gyzz + l_Gamzyz*gzzz + l_Gamxzz*gxyz + l_Gamyzz*gyyz + l_Gamzzz*gyzz) + l_Gamxzz*gyzx + l_Gamyzz*gyzy + l_Gamzzz*gyzz + l_Gamxyz*gzzx + l_Gamyyz*gzzy + l_Gamzyz*gzzz) + 
          gupzz * (TWO*(l_Gamxzz*gxzz + l_Gamyzz*gyzz + l_Gamzzz*gzzz) + l_Gamxzz*gzzx + l_Gamyzz*gzzy + l_Gamzzz*gzzz);

    // Rxy Correction
    l_Rxy = HALF * ( - l_Rxy + 
          l_gxx * dGamxy + l_gxy * dGamyy + l_gxz * dGamzy + 
          l_gxy * dGamxx + l_gyy * dGamyx + l_gyz * dGamzx + 
          Gamxa * gxyx + Gamya * gyyx + Gamza * gyzx + 
          Gamxa * gxxy + Gamya * gxyy + Gamza * gxzy) + 
          gupxx * (l_Gamxxx*gxxy + l_Gamyxx*gxyy + l_Gamzxx*gxzy + l_Gamxxy*gxxx + l_Gamyxy*gxyx + l_Gamzxy*gxzx + l_Gamxxx*gxyx + l_Gamyxx*gxyy + l_Gamzxx*gxyz) + 
          gupxy * (l_Gamxxx*gxyy + l_Gamyxx*gyyy + l_Gamzxx*gyzy + l_Gamxxy*gxyx + l_Gamyxy*gyyx + l_Gamzxy*gyzx + l_Gamxxy*gxyx + l_Gamyxy*gxyy + l_Gamzxy*gxyz + l_Gamxxy*gxxy + l_Gamyxy*gxyy + l_Gamzxy*gxzy + l_Gamxyy*gxxx + l_Gamyyy*gxyx + l_Gamzyy*gxzx + l_Gamxxx*gyyx + l_Gamyxx*gyyy + l_Gamzxx*gyyz) + 
          gupxz * (l_Gamxxx*gxzy + l_Gamyxx*gyzy + l_Gamzxx*gzzy + l_Gamxxy*gxzx + l_Gamyxy*gyzx + l_Gamzxy*gzzx + l_Gamxxz*gxyx + l_Gamyxz*gxyy + l_Gamzxz*gxyz + l_Gamxxz*gxxy + l_Gamyxz*gxyy + l_Gamzxz*gxzy + l_Gamxyz*gxxx + l_Gamyyz*gxyx + l_Gamzyz*gxzx + l_Gamxxx*gyzx + l_Gamyxx*gyzy + l_Gamzxx*gyzz) + 
          gupyy * (l_Gamxxy*gxyy + l_Gamyxy*gyyy + l_Gamzxy*gyzy + l_Gamxyy*gxyx + l_Gamyyy*gyyx + l_Gamzyy*gyzx + l_Gamxxy*gyyx + l_Gamyxy*gyyy + l_Gamzxy*gyyz) + 
          gupyz * (l_Gamxxy*gxzy + l_Gamyxy*gyzy + l_Gamzxy*gzzy + l_Gamxyy*gxzx + l_Gamyyy*gyzx + l_Gamzyy*gzzx + l_Gamxxz*gyyx + l_Gamyxz*gyyy + l_Gamzxz*gyyz + l_Gamxxz*gxyy + l_Gamyxz*gyyy + l_Gamzxz*gyzy + l_Gamxyz*gxyx + l_Gamyyz*gyyx + l_Gamzyz*gyzx + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) + 
          gupzz * (l_Gamxxz*gxzy + l_Gamyxz*gyzy + l_Gamzxz*gzzy + l_Gamxyz*gxzx + l_Gamyyz*gyzx + l_Gamzyz*gzzx + l_Gamxxz*gyzx + l_Gamyxz*gyzy + l_Gamzxz*gyzz);

    // Rxz Correction
    l_Rxz = HALF * ( - l_Rxz + 
          l_gxx * dGamxz + l_gxy * dGamyz + l_gxz * dGamzz + 
          l_gxz * dGamxx + l_gyz * dGamyx + l_gzz * dGamzx + 
          Gamxa * gxzx + Gamya * gyzx + Gamza * gzzx + 
          Gamxa * gxxz + Gamya * gxyz + Gamza * gxzz) + 
          gupxx * (l_Gamxxx*gxxz + l_Gamyxx*gxyz + l_Gamzxx*gxzz + l_Gamxxz*gxxx + l_Gamyxz*gxyx + l_Gamzxz*gxzx + l_Gamxxx*gxzx + l_Gamyxx*gxzy + l_Gamzxx*gxzz) + 
          gupxy * (l_Gamxxx*gxyz + l_Gamyxx*gyyz + l_Gamzxx*gyzz + l_Gamxxz*gxyx + l_Gamyxz*gyyx + l_Gamzxz*gyzx + l_Gamxxy*gxzx + l_Gamyxy*gxzy + l_Gamzxy*gxzz + l_Gamxxy*gxxz + l_Gamyxy*gxyz + l_Gamzxy*gxzz + l_Gamxyz*gxxx + l_Gamyyz*gxyx + l_Gamzyz*gxzx + l_Gamxxx*gyzx + l_Gamyxx*gyzy + l_Gamzxx*gyzz) + 
          gupxz * (l_Gamxxx*gxzz + l_Gamyxx*gyzz + l_Gamzxx*gzzz + l_Gamxxz*gxzx + l_Gamyxz*gyzx + l_Gamzxz*gzzx + l_Gamxxz*gxzx + l_Gamyxz*gxzy + l_Gamzxz*gxzz + l_Gamxxz*gxxz + l_Gamyxz*gxyz + l_Gamzxz*gxzz + l_Gamxzz*gxxx + l_Gamyzz*gxyx + l_Gamzzz*gxzx + l_Gamxxx*gzzx + l_Gamyxx*gzzy + l_Gamzxx*gzzz) + 
          gupyy * (l_Gamxxy*gxyz + l_Gamyxy*gyyz + l_Gamzxy*gyzz + l_Gamxyz*gxyx + l_Gamyyz*gyyx + l_Gamzyz*gyzx + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) + 
          gupyz * (l_Gamxxy*gxzz + l_Gamyxy*gyzz + l_Gamzxy*gzzz + l_Gamxyz*gxzx + l_Gamyyz*gyzx + l_Gamzyz*gzzx + l_Gamxxz*gyzx + l_Gamyxz*gyzy + l_Gamzxz*gyzz + l_Gamxxz*gxyz + l_Gamyxz*gyyz + l_Gamzxz*gyzz + l_Gamxzz*gxyx + l_Gamyzz*gyyx + l_Gamzzz*gyzx + l_Gamxxy*gzzx + l_Gamyxy*gzzy + l_Gamzxy*gzzz) + 
          gupzz * (l_Gamxxz*gxzz + l_Gamyxz*gyzz + l_Gamzxz*gzzz + l_Gamxzz*gxzx + l_Gamyzz*gyzx + l_Gamzzz*gzzx + l_Gamxxz*gzzx + l_Gamyxz*gzzy + l_Gamzxz*gzzz);

    // Ryz Correction
    l_Ryz = HALF * ( - l_Ryz + 
          l_gxy * dGamxz + l_gyy * dGamyz + l_gyz * dGamzz + 
          l_gxz * dGamxy + l_gyz * dGamyy + l_gzz * dGamzy + 
          Gamxa * gxzy + Gamya * gyzy + Gamza * gzzy + 
          Gamxa * gxyz + Gamya * gyyz + Gamza * gyzz) + 
          gupxx * (l_Gamxxy*gxxz + l_Gamyxy*gxyz + l_Gamzxy*gxzz + l_Gamxxz*gxxy + l_Gamyxz*gxyy + l_Gamzxz*gxzy + l_Gamxxy*gxzx + l_Gamyxy*gxzy + l_Gamzxy*gxzz) + 
          gupxy * (l_Gamxxy*gxyz + l_Gamyxy*gyyz + l_Gamzxy*gyzz + l_Gamxxz*gxyy + l_Gamyxz*gyyy + l_Gamzxz*gyzy + l_Gamxyy*gxzx + l_Gamyyy*gxzy + l_Gamzyy*gxzz + l_Gamxyy*gxxz + l_Gamyyy*gxyz + l_Gamzyy*gxzz + l_Gamxyz*gxxy + l_Gamyyz*gxyy + l_Gamzyz*gxzy + l_Gamxxy*gyzx + l_Gamyxy*gyzy + l_Gamzxy*gyzz) + 
          gupxz * (l_Gamxxy*gxzz + l_Gamyxy*gyzz + l_Gamzxy*gzzz + l_Gamxxz*gxzy + l_Gamyxz*gyzy + l_Gamzxz*gzzy + l_Gamxyz*gxzx + l_Gamyyz*gxzy + l_Gamzyz*gxzz + l_Gamxyz*gxxz + l_Gamyyz*gxyz + l_Gamzyz*gxzz + l_Gamxzz*gxxy + l_Gamyzz*gxyy + l_Gamzzz*gxzy + l_Gamxxy*gzzx + l_Gamyxy*gzzy + l_Gamzxy*gzzz) + 
          gupyy * (l_Gamxyy*gxyz + l_Gamyyy*gyyz + l_Gamzyy*gyzz + l_Gamxyz*gxyy + l_Gamyyz*gyyy + l_Gamzyz*gyzy + l_Gamxyy*gyzx + l_Gamyyy*gyzy + l_Gamzyy*gyzz) + 
          gupyz * (l_Gamxyy*gxzz + l_Gamyyy*gyzz + l_Gamzyy*gzzz + l_Gamxyz*gxzy + l_Gamyyz*gyzy + l_Gamzyz*gzzy + l_Gamxyz*gyzx + l_Gamyyz*gyzy + l_Gamzyz*gyzz + l_Gamxyz*gxyz + l_Gamyyz*gyyz + l_Gamzyz*gyzz + l_Gamxzz*gxyy + l_Gamyzz*gyyy + l_Gamzzz*gyzy + l_Gamxyy*gzzx + l_Gamyyy*gzzy + l_Gamzyy*gzzz) + 
          gupzz * (l_Gamxyz*gxzz + l_Gamyyz*gyzz + l_Gamzyz*gzzz + l_Gamxzz*gxzy + l_Gamyzz*gyzy + l_Gamzzz*gzzy + l_Gamxyz*gzzx + l_Gamyyz*gzzy + l_Gamzyz*gzzz);
    d_fdderivs_point(dims, chi, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);
    
    // 协变导数修正
    fxx -= l_Gamxxx * chix + l_Gamyxx * chiy + l_Gamzxx * chiz;
    fxy -= l_Gamxxy * chix + l_Gamyxy * chiy + l_Gamzxy * chiz;
    fxz -= l_Gamxxz * chix + l_Gamyxz * chiy + l_Gamzxz * chiz;
    fyy -= l_Gamxyy * chix + l_Gamyyy * chiy + l_Gamzyy * chiz;
    fyz -= l_Gamxyz * chix + l_Gamyyz * chiy + l_Gamzyz * chiz;
    fzz -= l_Gamxzz * chix + l_Gamyzz * chiy + l_Gamzzz * chiz;

    double f_scalar = gupxx * (fxx - F3o2/chin1 * chix * chix) + 
                      gupyy * (fyy - F3o2/chin1 * chiy * chiy) + 
                      gupzz * (fzz - F3o2/chin1 * chiz * chiz) + 
                      TWO * (gupxy * (fxy - F3o2/chin1 * chix * chiy) + 
                             gupxz * (fxz - F3o2/chin1 * chix * chiz) + 
                             gupyz * (fyz - F3o2/chin1 * chiy * chiz));
    
    // Add to Ricci
    l_Rxx += (fxx - chix*chix/chin1/TWO + l_gxx * f_scalar)/chin1/TWO;
    l_Ryy += (fyy - chiy*chiy/chin1/TWO + l_gyy * f_scalar)/chin1/TWO;
    l_Rzz += (fzz - chiz*chiz/chin1/TWO + l_gzz * f_scalar)/chin1/TWO;
    l_Rxy += (fxy - chix*chiy/chin1/TWO + l_gxy * f_scalar)/chin1/TWO;
    l_Rxz += (fxz - chix*chiz/chin1/TWO + l_gxz * f_scalar)/chin1/TWO;
    l_Ryz += (fyz - chiy*chiz/chin1/TWO + l_gyz * f_scalar)/chin1/TWO;

    // Keep the raw connection coefficients.  Stage 2 applies the physical
    // chi correction after using these values for Gam_rhs, matching the
    // ordering of the original monolithic kernel.
    Gamxxx[idx] = l_Gamxxx; Gamxxy[idx] = l_Gamxxy; Gamxxz[idx] = l_Gamxxz;
    Gamxyy[idx] = l_Gamxyy; Gamxyz[idx] = l_Gamxyz; Gamxzz[idx] = l_Gamxzz;
    Gamyxx[idx] = l_Gamyxx; Gamyxy[idx] = l_Gamyxy; Gamyxz[idx] = l_Gamyxz;
    Gamyyy[idx] = l_Gamyyy; Gamyyz[idx] = l_Gamyyz; Gamyzz[idx] = l_Gamyzz;
    Gamzxx[idx] = l_Gamzxx; Gamzxy[idx] = l_Gamzxy; Gamzxz[idx] = l_Gamzxz;
    Gamzyy[idx] = l_Gamzyy; Gamzyz[idx] = l_Gamzyz; Gamzzz[idx] = l_Gamzzz;
    Rxx[idx] = l_Rxx; Rxy[idx] = l_Rxy; Rxz[idx] = l_Rxz;
    Ryy[idx] = l_Ryy; Ryz[idx] = l_Ryz; Rzz[idx] = l_Rzz;
}


__global__ __launch_bounds__(128, 1) void rhs_kernel_int_stage2(
    int ex0, int ex1, int ex2, double T, double* __restrict__ X, double* __restrict__ Y, double* __restrict__ Z,
    double* __restrict__ chi, double* __restrict__ trK,
    double* __restrict__ dxx, double* __restrict__ gxy, double* __restrict__ gxz,
    double* __restrict__ dyy, double* __restrict__ gyz, double* __restrict__ dzz,
    double* __restrict__ Axx, double* __restrict__ Axy, double* __restrict__ Axz,
    double* __restrict__ Ayy, double* __restrict__ Ayz, double* __restrict__ Azz,
    double* __restrict__ Gamx, double* __restrict__ Gamy, double* __restrict__ Gamz,
    double* __restrict__ Lap,
    double* __restrict__ betax, double* __restrict__ betay, double* __restrict__ betaz,
    double* __restrict__ dtSfx, double* __restrict__ dtSfy, double* __restrict__ dtSfz,
    double* __restrict__ chi_rhs, double* __restrict__ trK_rhs,
    double* __restrict__ gxx_rhs, double* __restrict__ gxy_rhs, double* __restrict__ gxz_rhs,
    double* __restrict__ gyy_rhs, double* __restrict__ gyz_rhs, double* __restrict__ gzz_rhs,
    double* __restrict__ Axx_rhs, double* __restrict__ Axy_rhs, double* __restrict__ Axz_rhs,
    double* __restrict__ Ayy_rhs, double* __restrict__ Ayz_rhs, double* __restrict__ Azz_rhs,
    double* __restrict__ Gamx_rhs, double* __restrict__ Gamy_rhs, double* __restrict__ Gamz_rhs,
    double* __restrict__ Lap_rhs,
    double* __restrict__ betax_rhs, double* __restrict__ betay_rhs, double* __restrict__ betaz_rhs,
    double* __restrict__ dtSfx_rhs, double* __restrict__ dtSfy_rhs, double* __restrict__ dtSfz_rhs,
    double* __restrict__ rho, double* __restrict__ Sx, double* __restrict__ Sy, double* __restrict__ Sz,
    double* __restrict__ Sxx, double* __restrict__ Sxy, double* __restrict__ Sxz,
    double* __restrict__ Syy, double* __restrict__ Syz, double* __restrict__ Szz,
    double* __restrict__ Gamxxx, double* __restrict__ Gamxxy, double* __restrict__ Gamxxz,
    double* __restrict__ Gamxyy, double* __restrict__ Gamxyz, double* __restrict__ Gamxzz,
    double* __restrict__ Gamyxx, double* __restrict__ Gamyxy, double* __restrict__ Gamyxz,
    double* __restrict__ Gamyyy, double* __restrict__ Gamyyz, double* __restrict__ Gamyzz,
    double* __restrict__ Gamzxx, double* __restrict__ Gamzxy, double* __restrict__ Gamzxz,
    double* __restrict__ Gamzyy, double* __restrict__ Gamzyz, double* __restrict__ Gamzzz,
    double* __restrict__ Rxx, double* __restrict__ Rxy, double* __restrict__ Rxz,
    double* __restrict__ Ryy, double* __restrict__ Ryz, double* __restrict__ Rzz,
    double* __restrict__ ham_Res, double* __restrict__ movx_Res, double* __restrict__ movy_Res, double* __restrict__ movz_Res,
    double* __restrict__ Gmx_Res, double* __restrict__ Gmy_Res, double* __restrict__ Gmz_Res,
    int symmetry, int lev, double eps, int co
) {
    // 计算全局索引
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 2;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 3;

    // 越界检查
    if (i >= ex0 || j >= ex1 || k >= ex2) return;

    // interior-only fast path: skip the boundary shell.
    // margins: i,j = 2 (imin=jmin=0 -> no reflection reachable at i,j>=2);
    // k lower = 3 (equatorial symmetry -> kmin=-3 -> lopsided/kodis reflect at
    // k<=2, plain-load fh would read OOB); k upper = 2.
    if (i < 2 || i > ex0 - 3 || j < 2 || j > ex1 - 3 || k < 3 || k > ex2 - 3) return;

    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    int dims[3] = {ex0, ex1, ex2}; // 用于传给 device 函数
    const int nx = ex0;
    const int nxy = ex0 * ex1;
    const double d12dx = 1.0 / (12.0 * (X[1] - X[0]));
    const double d12dy = 1.0 / (12.0 * (Y[1] - Y[0]));
    const double d12dz = 1.0 / (12.0 * (Z[1] - Z[0]));

    // Stage 2 assembles the evolution RHS from the geometry written by
    // rhs_kernel_int_stage1.  Keep the original arithmetic order inside
    // each stage so the candidate can be checked for bitwise equivalence.
    double val_Lap = Lap[idx];
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
    double val_Ayy = Ayy[idx]; double val_Ayz = Ayz[idx]; double val_Azz = Azz[idx];
    double betaxx, betaxy, betaxz;
    double betayx, betayy, betayz;
    double betazx, betazy, betazz;

    d_fderivs_point_interior_fast(idx, nx, nxy, betax, &betaxx, &betaxy, &betaxz, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, betay, &betayx, &betayy, &betayz, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, betaz, &betazx, &betazy, &betazz, d12dx, d12dy, d12dz);

    double div_beta = betaxx + betayy + betazz;

    double chix, chiy, chiz;
    d_fderivs_point_interior_fast(idx, nx, nxy, chi, &chix, &chiy, &chiz, d12dx, d12dy, d12dz);
    chi_rhs[idx] = F2o3 * chin1 * (alpn1 * val_trK - div_beta);

    double gxxx, gxxy, gxxz;
    double gxyx, gxyy, gxyz;
    double gxzx, gxzy, gxzz;
    double gyyx, gyyy, gyyz;
    double gyzx, gyzy, gyzz;
    double gzzx, gzzy, gzzz;
    gxx_rhs[idx] = -TWO * alpn1 * val_Axx - F2o3 * val_gxx * div_beta + 
                    TWO * (val_gxx * betaxx + val_gxy * betayx + val_gxz * betazx);

    gyy_rhs[idx] = -TWO * alpn1 * val_Ayy - F2o3 * val_gyy * div_beta + 
                    TWO * (val_gxy * betaxy + val_gyy * betayy + val_gyz * betazy);

    gzz_rhs[idx] = -TWO * alpn1 * val_Azz - F2o3 * val_gzz * div_beta + 
                    TWO * (val_gxz * betaxz + val_gyz * betayz + val_gzz * betazz);

    gxy_rhs[idx] = -TWO * alpn1 * val_Axy + F1o3 * val_gxy * div_beta + 
                    val_gxx * betaxy + val_gxz * betazy + 
                    val_gyy * betayx + val_gyz * betazx - val_gxy * betazz;
    
    gyz_rhs[idx] = -TWO * alpn1 * val_Ayz + F1o3 * val_gyz * div_beta +
                    val_gxy * betaxz + val_gyy * betayz +
                    val_gxz * betaxy + val_gzz * betazy - val_gyz * betaxx;

    gxz_rhs[idx] = -TWO * alpn1 * val_Axz + F1o3 * val_gxz * div_beta + 
                    val_gxx * betaxz + val_gxy * betayz + 
                    val_gyz * betayx + val_gzz * betazx - val_gxz * betayy;
    double gupzz = val_gxx * val_gyy * val_gzz + val_gxy * val_gyz * val_gxz + val_gxz * val_gxy * val_gyz -
                   val_gxz * val_gyy * val_gxz - val_gxy * val_gxy * val_gzz - val_gxx * val_gyz * val_gyz;
    
    double gupxx = (val_gyy * val_gzz - val_gyz * val_gyz) / gupzz;
    double gupxy = -(val_gxy * val_gzz - val_gyz * val_gxz) / gupzz;
    double gupxz = (val_gxy * val_gyz - val_gyy * val_gxz) / gupzz;
    double gupyy = (val_gxx * val_gzz - val_gxz * val_gxz) / gupzz;
    double gupyz = -(val_gxx * val_gyz - val_gxy * val_gxz) / gupzz;
    gupzz = (val_gxx * val_gyy - val_gxy * val_gxy) / gupzz; // 更新 gupzz 为逆分量

    double l_Gamxxx = Gamxxx[idx]; double l_Gamxxy = Gamxxy[idx]; double l_Gamxxz = Gamxxz[idx];
    double l_Gamxyy = Gamxyy[idx]; double l_Gamxyz = Gamxyz[idx]; double l_Gamxzz = Gamxzz[idx];
    double l_Gamyxx = Gamyxx[idx]; double l_Gamyxy = Gamyxy[idx]; double l_Gamyxz = Gamyxz[idx];
    double l_Gamyyy = Gamyyy[idx]; double l_Gamyyz = Gamyyz[idx]; double l_Gamyzz = Gamyzz[idx];
    double l_Gamzxx = Gamzxx[idx]; double l_Gamzxy = Gamzxy[idx]; double l_Gamzxz = Gamzxz[idx];
    double l_Gamzyy = Gamzyy[idx]; double l_Gamzyz = Gamzyz[idx]; double l_Gamzzz = Gamzzz[idx];

    double l_Axx = Axx[idx]; double l_Axy = Axy[idx]; double l_Axz = Axz[idx];
    double l_Ayy = Ayy[idx]; double l_Ayz = Ayz[idx]; double l_Azz = Azz[idx];

    double l_gxx = val_gxx; double l_gxy = val_gxy; double l_gxz = val_gxz;
    double l_gyy = val_gyy; double l_gyz = val_gyz; double l_gzz = val_gzz;
    double l_Rxx, l_Rxy, l_Rxz, l_Ryy, l_Ryz, l_Rzz;

    l_Rxx = gupxx * gupxx * l_Axx + gupxy * gupxy * l_Ayy + gupxz * gupxz * l_Azz + 
          TWO*(gupxx * gupxy * l_Axy + gupxx * gupxz * l_Axz + gupxy * gupxz * l_Ayz);

    l_Ryy = gupxy * gupxy * l_Axx + gupyy * gupyy * l_Ayy + gupyz * gupyz * l_Azz + 
          TWO*(gupxy * gupyy * l_Axy + gupxy * gupyz * l_Axz + gupyy * gupyz * l_Ayz);

    l_Rzz = gupxz * gupxz * l_Axx + gupyz * gupyz * l_Ayy + gupzz * gupzz * l_Azz + 
          TWO*(gupxz * gupyz * l_Axy + gupxz * gupzz * l_Axz + gupyz * gupzz * l_Ayz);

    l_Rxy = gupxx * gupxy * l_Axx + gupxy * gupyy * l_Ayy + gupxz * gupyz * l_Azz + 
          (gupxx * gupyy + gupxy * gupxy)* l_Axy + 
          (gupxx * gupyz + gupxz * gupxy)* l_Axz + 
          (gupxy * gupyz + gupxz * gupyy)* l_Ayz;

    l_Rxz = gupxx * gupxz * l_Axx + gupxy * gupyz * l_Ayy + gupxz * gupzz * l_Azz + 
          (gupxx * gupyz + gupxy * gupxz)* l_Axy + 
          (gupxx * gupzz + gupxz * gupxz)* l_Axz + 
          (gupxy * gupzz + gupxz * gupyz)* l_Ayz;

    l_Ryz = gupxy * gupxz * l_Axx + gupyy * gupyz * l_Ayy + gupyz * gupzz * l_Azz + 
          (gupxy * gupyz + gupyy * gupxz)* l_Axy + 
          (gupxy * gupzz + gupyz * gupxz)* l_Axz + 
          (gupyy * gupzz + gupyz * gupyz)* l_Ayz;
    double Lapx, Lapy, Lapz, Kx, Ky, Kz;
    d_fderivs_point_interior_fast(idx, nx, nxy, Lap, &Lapx, &Lapy, &Lapz, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, trK, &Kx, &Ky, &Kz, d12dx, d12dy, d12dz);

    // double chix = chix_in[idx]; double chiy = chiy_in[idx]; double chiz = chiz_in[idx];
    double val_Sx = Sx[idx]; double val_Sy = Sy[idx]; double val_Sz = Sz[idx];

    // Gamx_rhs
    double val_Gamx_rhs = - TWO * (Lapx * l_Rxx + Lapy * l_Rxy + Lapz * l_Rxz) + 
        TWO * alpn1 * (
        -F3o2/chin1 * (chix * l_Rxx + chiy * l_Rxy + chiz * l_Rxz) - 
        gupxx * (F2o3 * Kx + EIGHT * PI * val_Sx) - 
        gupxy * (F2o3 * Ky + EIGHT * PI * val_Sy) - 
        gupxz * (F2o3 * Kz + EIGHT * PI * val_Sz) + 
        l_Gamxxx * l_Rxx + l_Gamxyy * l_Ryy + l_Gamxzz * l_Rzz + 
        TWO * (l_Gamxxy * l_Rxy + l_Gamxxz * l_Rxz + l_Gamxyz * l_Ryz));

    // Gamy_rhs
    double val_Gamy_rhs = - TWO * (Lapx * l_Rxy + Lapy * l_Ryy + Lapz * l_Ryz) + 
        TWO * alpn1 * (
        -F3o2/chin1 * (chix * l_Rxy + chiy * l_Ryy + chiz * l_Ryz) - 
        gupxy * (F2o3 * Kx + EIGHT * PI * val_Sx) - 
        gupyy * (F2o3 * Ky + EIGHT * PI * val_Sy) - 
        gupyz * (F2o3 * Kz + EIGHT * PI * val_Sz) + 
        l_Gamyxx * l_Rxx + l_Gamyyy * l_Ryy + l_Gamyzz * l_Rzz + 
        TWO * (l_Gamyxy * l_Rxy + l_Gamyxz * l_Rxz + l_Gamyyz * l_Ryz));

    // Gamz_rhs
    double val_Gamz_rhs = - TWO * (Lapx * l_Rxz + Lapy * l_Ryz + Lapz * l_Rzz) + 
        TWO * alpn1 * (
        -F3o2/chin1 * (chix * l_Rxz + chiy * l_Ryz + chiz * l_Rzz) - 
        gupxz * (F2o3 * Kx + EIGHT * PI * val_Sx) - 
        gupyz * (F2o3 * Ky + EIGHT * PI * val_Sy) - 
        gupzz * (F2o3 * Kz + EIGHT * PI * val_Sz) + 
        l_Gamzxx * l_Rxx + l_Gamzyy * l_Ryy + l_Gamzzz * l_Rzz + 
        TWO * (l_Gamzxy * l_Rxy + l_Gamzxz * l_Rxz + l_Gamzyz * l_Ryz));
    // betax 二阶导
    double bx_gxxx, bx_gxyx, bx_gxzx, bx_gyyx, bx_gyzx, bx_gzzx;
    d_fdderivs_point(dims, betax, &bx_gxxx, &bx_gxyx, &bx_gxzx, &bx_gyyx, &bx_gyzx, &bx_gzzx,
                     X, Y, Z, ANTI, SYM, SYM, symmetry, lev, i, j, k);

    // betay 二阶导
    double by_gxxy, by_gxyy, by_gxzy, by_gyyy, by_gyzy, by_gzzy;
    d_fdderivs_point(dims, betay, &by_gxxy, &by_gxyy, &by_gxzy, &by_gyyy, &by_gyzy, &by_gzzy,
                     X, Y, Z, SYM, ANTI, SYM, symmetry, lev, i, j, k);

    // betaz 二阶导
    double bz_gxxz, bz_gxyz, bz_gxzz, bz_gyyz, bz_gyzz, bz_gzzz;
    d_fdderivs_point(dims, betaz, &bz_gxxz, &bz_gxyz, &bz_gxzz, &bz_gyyz, &bz_gyzz, &bz_gzzz,
                     X, Y, Z, SYM, SYM, ANTI, symmetry, lev, i, j, k);

    double fxx, fxy, fxz, fyy, fyz, fzz;

    // fxx/fxy/fxz = Laplacian(beta^i) 的组合 (Fortran line 156)
    fxx = bx_gxxx + by_gxyy + bz_gxzz;
    fxy = bx_gxyx + by_gyyy + bz_gyzz;
    fxz = bx_gxzx + by_gyzy + bz_gzzz;

    double Gamxa = gupxx * l_Gamxxx + gupyy * l_Gamxyy + gupzz * l_Gamxzz +
                   TWO * (gupxy * l_Gamxxy + gupxz * l_Gamxxz + gupyz * l_Gamxyz);
    double Gamya = gupxx * l_Gamyxx + gupyy * l_Gamyyy + gupzz * l_Gamyzz +
                   TWO * (gupxy * l_Gamyxy + gupxz * l_Gamyxz + gupyz * l_Gamyyz);
    double Gamza = gupxx * l_Gamzxx + gupyy * l_Gamzyy + gupzz * l_Gamzzz +
                   TWO * (gupxy * l_Gamzxy + gupxz * l_Gamzxz + gupyz * l_Gamzyz);
    
    double dGamxx, dGamxy, dGamxz;
    double dGamyx, dGamyy, dGamyz;
    double dGamzx, dGamzy, dGamzz;
    d_fderivs_point_interior_fast(idx, nx, nxy, Gamx, &dGamxx, &dGamxy, &dGamxz, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, Gamy, &dGamyx, &dGamyy, &dGamyz, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, Gamz, &dGamzx, &dGamzy, &dGamzz, d12dx, d12dy, d12dz);

    // double betaxx, betaxy, betaxz, betayx, betayy, betayz, betazx, betazy, betazz;
    // d_fderivs_point_interior_fast(idx, nx, nxy, betax, &betaxx, &betaxy, &betaxz, d12dx, d12dy, d12dz);
    // d_fderivs_point_interior_fast(idx, nx, nxy, betay, &betayx, &betayy, &betayz, d12dx, d12dy, d12dz);
    // d_fderivs_point_interior_fast(idx, nx, nxy, betaz, &betazx, &betazy, &betazz, d12dx, d12dy, d12dz);
    // double div_beta = betaxx + betayy + betazz;

    // Gamx_rhs (Fortran line 170)
    val_Gamx_rhs += F2o3 * Gamxa * div_beta
                 - (Gamxa * betaxx + Gamya * betaxy + Gamza * betaxz)
                 + F1o3 * (gupxx * fxx + gupxy * fxy + gupxz * fxz)
                 + gupxx * bx_gxxx + gupyy * bx_gyyx + gupzz * bx_gzzx
                 + TWO * (gupxy * bx_gxyx + gupxz * bx_gxzx + gupyz * bx_gyzx);

    // Gamy_rhs (Fortran line 174)
    val_Gamy_rhs += F2o3 * Gamya * div_beta
                 - (Gamxa * betayx + Gamya * betayy + Gamza * betayz)
                 + F1o3 * (gupxy * fxx + gupyy * fxy + gupyz * fxz)
                 + gupxx * by_gxxy + gupyy * by_gyyy + gupzz * by_gzzy
                 + TWO * (gupxy * by_gxyy + gupxz * by_gxzy + gupyz * by_gyzy);

    // Gamz_rhs (Fortran line 178)
    val_Gamz_rhs += F2o3 * Gamza * div_beta
                 - (Gamxa * betazx + Gamya * betazy + Gamza * betazz)
                 + F1o3 * (gupxz * fxx + gupyz * fxy + gupzz * fxz)
                 + gupxx * bz_gxxz + gupyy * bz_gyyz + gupzz * bz_gzzz
                 + TWO * (gupxy * bz_gxyz + gupxz * bz_gxzz + gupyz * bz_gyzz);
    d_fdderivs_point(dims, Lap, &fxx, &fxy, &fxz, &fyy, &fyz, &fzz, X, Y, Z, SYM, SYM, SYM, symmetry, lev, i, j, k);

    // 计算物理连接系数 (暂存到 Gam 数组中以节省寄存器，最后会写回 Global)
    double gx_phy = (gupxx * chix + gupxy * chiy + gupxz * chiz)/chin1;
    double gy_phy = (gupxy * chix + gupyy * chiy + gupyz * chiz)/chin1;
    double gz_phy = (gupxz * chix + gupyz * chiy + gupzz * chiz)/chin1;
    
    // 更新为物理连接系数 (对应 Fortran 241-258)
    l_Gamxxx -= ((chix + chix)/chin1 - l_gxx * gx_phy)*HALF; // l_Gamxxx = l_Gamxxx;
    l_Gamyxx -= (                            - l_gxx * gy_phy)*HALF; // l_Gamyxx = l_Gamyxx;
    l_Gamzxx -= (                            - l_gxx * gz_phy)*HALF; // l_Gamzxx = l_Gamzxx;
    l_Gamxyy -= (                            - l_gyy * gx_phy)*HALF; // l_Gamxyy = l_Gamxyy;
    l_Gamyyy -= ((chiy + chiy)/chin1 - l_gyy * gy_phy)*HALF; // l_Gamyyy = l_Gamyyy;
    l_Gamzyy -= (                            - l_gyy * gz_phy)*HALF; // l_Gamzyy = l_Gamzyy;
    l_Gamxzz -= (                            - l_gzz * gx_phy)*HALF; // l_Gamxzz = l_Gamxzz;
    l_Gamyzz -= (                            - l_gzz * gy_phy)*HALF; // l_Gamyzz = l_Gamyzz;
    l_Gamzzz -= ((chiz + chiz)/chin1 - l_gzz * gz_phy)*HALF; // l_Gamzzz = l_Gamzzz;

    l_Gamxxy -= (chiy/chin1 - l_gxy * gx_phy)*HALF; // l_Gamxxy = l_Gamxxy;
    l_Gamyxy -= (chix/chin1 - l_gxy * gy_phy)*HALF; // l_Gamyxy = l_Gamyxy;
    l_Gamzxy -= (                 - l_gxy * gz_phy)*HALF; // l_Gamzxy = l_Gamzxy;
    l_Gamxxz -= (chiz/chin1 - l_gxz * gx_phy)*HALF; // l_Gamxxz = l_Gamxxz;
    l_Gamyxz -= (                 - l_gxz * gy_phy)*HALF; // l_Gamyxz = l_Gamyxz;
    l_Gamzxz -= (chix/chin1 - l_gxz * gz_phy)*HALF; // l_Gamzxz = l_Gamzxz;
    l_Gamxyz -= (                 - l_gyz * gx_phy)*HALF; // l_Gamxyz = l_Gamxyz;
    l_Gamyyz -= (chiz/chin1 - l_gyz * gy_phy)*HALF; // l_Gamyyz = l_Gamyyz;
    l_Gamzyz -= (chiy/chin1 - l_gyz * gz_phy)*HALF; // l_Gamzyz = l_Gamzyz;


    // The final Ricci values are produced by stage 1.  The raw connection
    // values above have just been converted to the physical connection.
    l_Rxx = Rxx[idx]; l_Rxy = Rxy[idx]; l_Rxz = Rxz[idx];
    l_Ryy = Ryy[idx]; l_Ryz = Ryz[idx]; l_Rzz = Rzz[idx];

    // Lapse 的协变导数 D_i D_j alpha
    fxx = fxx - l_Gamxxx*Lapx - l_Gamyxx*Lapy - l_Gamzxx*Lapz;
    fyy = fyy - l_Gamxyy*Lapx - l_Gamyyy*Lapy - l_Gamzyy*Lapz;
    fzz = fzz - l_Gamxzz*Lapx - l_Gamyzz*Lapy - l_Gamzzz*Lapz;
    fxy = fxy - l_Gamxxy*Lapx - l_Gamyxy*Lapy - l_Gamzxy*Lapz;
    fxz = fxz - l_Gamxxz*Lapx - l_Gamyxz*Lapy - l_Gamzxz*Lapz;
    fyz = fyz - l_Gamxyz*Lapx - l_Gamyyz*Lapy - l_Gamzyz*Lapz;

    double trK_rhs_val = gupxx * fxx + gupyy * fyy + gupzz * fzz + TWO* (gupxy * fxy + gupxz * fxz + gupyz * fyz);

    // ==========================================
    // Step 8: 组装 Aij_rhs & trK_rhs
    // ==========================================
    // Iter28: load stress-energy arrays once and reuse (in-kernel
    // round-trip elimination; bit-exact, same array values)
    double val_rho = rho[idx];
    double val_Sxx = Sxx[idx]; double val_Sxy = Sxy[idx]; double val_Sxz = Sxz[idx];
    double val_Syy = Syy[idx]; double val_Syz = Syz[idx]; double val_Szz = Szz[idx];

    double S = chin1 * (gupxx * val_Sxx + gupyy * val_Syy + gupzz * val_Szz + 
               TWO * (gupxy * val_Sxy + gupxz * val_Sxz + gupyz * val_Syz));

    double term_xx = gupxx * l_Axx * l_Axx + gupyy * l_Axy * l_Axy + gupzz * l_Axz * l_Axz + TWO * (gupxy * l_Axx * l_Axy + gupxz * l_Axx * l_Axz + gupyz * l_Axy * l_Axz);
    double term_yy = gupxx * l_Axy * l_Axy + gupyy * l_Ayy * l_Ayy + gupzz * l_Ayz * l_Ayz + TWO * (gupxy * l_Axy * l_Ayy + gupxz * l_Axy * l_Ayz + gupyz * l_Ayy * l_Ayz);
    double term_zz = gupxx * l_Axz * l_Axz + gupyy * l_Ayz * l_Ayz + gupzz * l_Azz * l_Azz + TWO * (gupxy * l_Axz * l_Ayz + gupxz * l_Axz * l_Azz + gupyz * l_Ayz * l_Azz);
    double term_xy = gupxx * l_Axx * l_Axy + gupyy * l_Axy * l_Ayy + gupzz * l_Axz * l_Ayz + gupxy * (l_Axx * l_Ayy + l_Axy * l_Axy) + gupxz * (l_Axx * l_Ayz + l_Axz * l_Axy) + gupyz * (l_Axy * l_Ayz + l_Axz * l_Ayy);
    double term_xz = gupxx * l_Axx * l_Axz + gupyy * l_Axy * l_Ayz + gupzz * l_Axz * l_Azz + gupxy * (l_Axx * l_Ayz + l_Axy * l_Axz) + gupxz * (l_Axx * l_Azz + l_Axz * l_Axz) + gupyz * (l_Axy * l_Azz + l_Axz * l_Ayz);
    double term_yz = gupxx * l_Axy * l_Axz + gupyy * l_Ayy * l_Ayz + gupzz * l_Ayz * l_Azz + gupxy * (l_Axy * l_Ayz + l_Ayy * l_Axz) + gupxz * (l_Axy * l_Azz + l_Ayz * l_Axz) + gupyz * (l_Ayy * l_Azz + l_Ayz * l_Ayz);

    double trA2 = gupxx * term_xx + gupyy * term_yy + gupzz * term_zz + TWO * (gupxy * term_xy + gupxz * term_xz + gupyz * term_yz);

    double f = F2o3 * val_trK * val_trK - trA2 - F16*PI*val_rho + EIGHT*PI*S;
    double f_trace = -F1o3 * (trK_rhs_val + alpn1/chin1 * f);

    // 计算 Aij 源项
    double src_xx = alpn1 * (l_Rxx - EIGHT*PI*val_Sxx) - fxx; // fxx is D_i D_j Lap
    double src_yy = alpn1 * (l_Ryy - EIGHT*PI*val_Syy) - fyy;
    double src_zz = alpn1 * (l_Rzz - EIGHT*PI*val_Szz) - fzz;
    double src_xy = alpn1 * (l_Rxy - EIGHT*PI*val_Sxy) - fxy;
    double src_xz = alpn1 * (l_Rxz - EIGHT*PI*val_Sxz) - fxz;
    double src_yz = alpn1 * (l_Ryz - EIGHT*PI*val_Syz) - fyz;

    double Axx_rhs_val = src_xx - l_gxx * f_trace;
    double Ayy_rhs_val = src_yy - l_gyy * f_trace;
    double Azz_rhs_val = src_zz - l_gzz * f_trace;
    double Axy_rhs_val = src_xy - l_gxy * f_trace;
    double Axz_rhs_val = src_xz - l_gxz * f_trace;
    double Ayz_rhs_val = src_yz - l_gyz * f_trace;

    // 添加平流项 (Lie derivative of Aij)
    Axx_rhs[idx] = chin1 * Axx_rhs_val + alpn1 * (val_trK * l_Axx - TWO * term_xx) + TWO * (l_Axx * betaxx + l_Axy * betayx + l_Axz * betazx) - F2o3 * l_Axx * div_beta;
    Ayy_rhs[idx] = chin1 * Ayy_rhs_val + alpn1 * (val_trK * l_Ayy - TWO * term_yy) + TWO * (l_Axy * betaxy + l_Ayy * betayy + l_Ayz * betazy) - F2o3 * l_Ayy * div_beta;
    Azz_rhs[idx] = chin1 * Azz_rhs_val + alpn1 * (val_trK * l_Azz - TWO * term_zz) + TWO * (l_Axz * betaxz + l_Ayz * betayz + l_Azz * betazz) - F2o3 * l_Azz * div_beta;
    
    Axy_rhs[idx] = chin1 * Axy_rhs_val + alpn1 * (val_trK * l_Axy - TWO * term_xy) + l_Axx * betaxy + l_Axz * betazy + l_Ayy * betayx + l_Ayz * betazx - l_Axy * betazz + F1o3 * l_Axy * div_beta;
    Ayz_rhs[idx] = chin1 * Ayz_rhs_val + alpn1 * (val_trK * l_Ayz - TWO * term_yz) + l_Axy * betaxz + l_Ayy * betayz + l_Axz * betaxy + l_Azz * betazy - l_Ayz * betaxx + F1o3 * l_Ayz * div_beta;
    Axz_rhs[idx] = chin1 * Axz_rhs_val + alpn1 * (val_trK * l_Axz - TWO * term_xz) + l_Axx * betaxz + l_Axy * betayz + l_Ayz * betayx + l_Azz * betazx - l_Axz * betayy + F1o3 * l_Axz * div_beta;

    trK_rhs[idx] = -chin1 * trK_rhs_val + alpn1 * (F1o3 * val_trK * val_trK + trA2 + FOUR * PI * (val_rho + S));

    // Gauge vars RHS
    Lap_rhs[idx] = -TWO * alpn1 * val_trK;
    betax_rhs[idx] = FF * dtSfx[idx];
    betay_rhs[idx] = FF * dtSfy[idx];
    betaz_rhs[idx] = FF * dtSfz[idx];
    dtSfx_rhs[idx] = val_Gamx_rhs - eta * dtSfx[idx];
    dtSfy_rhs[idx] = val_Gamy_rhs - eta * dtSfy[idx];
    dtSfz_rhs[idx] = val_Gamz_rhs - eta * dtSfz[idx];

    // 写回 Gam_rhs
    Gamx_rhs[idx] = val_Gamx_rhs;
    Gamy_rhs[idx] = val_Gamy_rhs;
    Gamz_rhs[idx] = val_Gamz_rhs;

    // l_Gamxxx = l_Gamxxx; l_Gamxxy = l_Gamxxy; l_Gamxxz = l_Gamxxz;
    // l_Gamxyy = l_Gamxyy; l_Gamxyz = l_Gamxyz; l_Gamxzz = l_Gamxzz;

    // l_Gamyxx = l_Gamyxx; l_Gamyxy = l_Gamyxy; l_Gamyxz = l_Gamyxz;
    // l_Gamyyy = l_Gamyyy; l_Gamyyz = l_Gamyyz; l_Gamyzz = l_Gamyzz;

    // l_Gamzxx = l_Gamzxx; l_Gamzxy = l_Gamzxy; l_Gamzxz = l_Gamzxz;
    // l_Gamzyy = l_Gamzyy; l_Gamzyz = l_Gamzyz; l_Gamzzz = l_Gamzzz;

    // Rxx[idx] = l_Rxx; Ryy[idx] = l_Ryy; Rzz[idx] = l_Rzz;
    // Rxy[idx] = l_Rxy; Rxz[idx] = l_Rxz; Ryz[idx] = l_Ryz;

    // ------------------------------------------------------------------------------------
    // bssn_constraints_kernel
    // ------------------------------------------------------------------------------------

    if (co != 0) return;

    // ==========================================
    // 0. 加载数据
    // ==========================================

    // double gupxx = gupxx_in[idx]; double gupxy = gupxy_in[idx]; double gupxz = gupxz_in[idx];
    // double gupyy = gupyy_in[idx]; double gupyz = gupyz_in[idx]; double gupzz = gupzz_in[idx];

    // double l_Axx = Axx[idx]; double l_Axy = Axy[idx]; double l_Axz = Axz[idx];
    // double l_Ayy = Ayy[idx]; double l_Ayz = Ayz[idx]; double l_Azz = Azz[idx];

    // double l_Rxx = Rxx_in[idx]; double l_Rxy = Rxy_in[idx]; double l_Rxz = Rxz_in[idx];
    // double l_Ryy = Ryy_in[idx]; double l_Ryz = Ryz_in[idx]; double l_Rzz = Rzz_in[idx];

    // ==========================================
    // 1. Hamiltonian Constraint
    // ==========================================
    // ham_Res = trR + 2/3 K^2 - A_ij A^ij - 16 PI rho

    // 计算 trR (Respect to physical metric)
    // Fortran Line 372
    double ham_val = gupxx * l_Rxx + gupyy * l_Ryy + gupzz * l_Rzz + 
               TWO * (gupxy * l_Rxy + gupxz * l_Rxz + gupyz * l_Ryz);

    // double term_xx = gupxx * l_Axx * l_Axx + gupyy * l_Axy * l_Axy + gupzz * l_Axz * l_Axz + TWO * (gupxy * l_Axx * l_Axy + gupxz * l_Axx * l_Axz + gupyz * l_Axy * l_Axz);
    // double term_yy = gupxx * l_Axy * l_Axy + gupyy * l_Ayy * l_Ayy + gupzz * l_Ayz * l_Ayz + TWO * (gupxy * l_Axy * l_Ayy + gupxz * l_Axy * l_Ayz + gupyz * l_Ayy * l_Ayz);
    // double term_zz = gupxx * l_Axz * l_Axz + gupyy * l_Ayz * l_Ayz + gupzz * l_Azz * l_Azz + TWO * (gupxy * l_Axz * l_Ayz + gupxz * l_Axz * l_Azz + gupyz * l_Ayz * l_Azz);
    
    // double term_xy = gupxx * l_Axx * l_Axy + gupyy * l_Axy * l_Ayy + gupzz * l_Axz * l_Ayz + gupxy * (l_Axx * l_Ayy + l_Axy * l_Axy) + gupxz * (l_Axx * l_Ayz + l_Axz * l_Axy) + gupyz * (l_Axy * l_Ayz + l_Axz * l_Ayy);
    // double term_xz = gupxx * l_Axx * l_Axz + gupyy * l_Axy * l_Ayz + gupzz * l_Axz * l_Azz + gupxy * (l_Axx * l_Ayz + l_Axy * l_Axz) + gupxz * (l_Axx * l_Azz + l_Axz * l_Axz) + gupyz * (l_Axy * l_Azz + l_Axz * l_Ayz);
    // double term_yz = gupxx * l_Axy * l_Axz + gupyy * l_Ayy * l_Ayz + gupzz * l_Ayz * l_Azz + gupxy * (l_Axy * l_Ayz + l_Ayy * l_Axz) + gupxz * (l_Axy * l_Azz + l_Ayz * l_Axz) + gupyz * (l_Ayy * l_Azz + l_Ayz * l_Ayz);

    // double trA2 = gupxx * term_xx + gupyy * term_yy + gupzz * term_zz + TWO * (gupxy * term_xy + gupxz * term_xz + gupyz * term_yz);

    // Final Hamiltonian Calculation
    // Fortran Line 375
    ham_Res[idx] = chin1 * ham_val + F2o3 * val_trK * val_trK - trA2 - F16 * PI * val_rho;
    // ==========================================
    // 2. Momentum Constraint
    // ==========================================
    // mov_Res_j = D_k A^k_j - 2/3 d_j trK - 8 PI S_j

    // 需要 trK 的导数
    // double Kx, Ky, Kz;
    d_fderivs_point_interior_fast(idx, nx, nxy, trK, &Kx, &Ky, &Kz, d12dx, d12dy, d12dz);

    // 需要 Aij 的导数 (Fortran calls fderivs 6 times)
    // 为了节省寄存器和避免创建大数组，我们分量计算并直接应用 Covariant 修正
    
    // double chix = chix_in[idx];
    // double chiy = chiy_in[idx];
    // double chiz = chiz_in[idx];

    // --- Compute D_i A_jk stored in variables named like `dA_xxx` (meaning D_x A_xx) ---
    
    // 1. Axx (SYM, SYM, SYM)
    double d_Axx_x, d_Axx_y, d_Axx_z;
    double d_Axy_x, d_Axy_y, d_Axy_z;
    double d_Axz_x, d_Axz_y, d_Axz_z;
    double d_Ayy_x, d_Ayy_y, d_Ayy_z;
    double d_Ayz_x, d_Ayz_y, d_Ayz_z;
    double d_Azz_x, d_Azz_y, d_Azz_z;
    d_fderivs_point_interior_fast(idx, nx, nxy, Axx, &d_Axx_x, &d_Axx_y, &d_Axx_z, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, Axy, &d_Axy_x, &d_Axy_y, &d_Axy_z, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, Axz, &d_Axz_x, &d_Axz_y, &d_Axz_z, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, Ayy, &d_Ayy_x, &d_Ayy_y, &d_Ayy_z, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, Ayz, &d_Ayz_x, &d_Ayz_y, &d_Ayz_z, d12dx, d12dy, d12dz);
    d_fderivs_point_interior_fast(idx, nx, nxy, Azz, &d_Azz_x, &d_Azz_y, &d_Azz_z, d12dx, d12dy, d12dz);


    double DA_xxx = d_Axx_x - (l_Gamxxx * l_Axx + l_Gamyxx * l_Axy + l_Gamzxx * l_Axz 
                             + l_Gamxxx * l_Axx + l_Gamyxx * l_Axy + l_Gamzxx * l_Axz) - chix * l_Axx / chin1;
    double DA_xyx = d_Axy_x - (l_Gamxxy * l_Axx + l_Gamyxy * l_Axy + l_Gamzxy * l_Axz
                             + l_Gamxxx * l_Axy + l_Gamyxx * l_Ayy + l_Gamzxx * l_Ayz) - chix * l_Axy / chin1;
    double DA_xzx = d_Axz_x - (l_Gamxxz * l_Axx + l_Gamyxz * l_Axy + l_Gamzxz * l_Axz
                             + l_Gamxxx * l_Axz + l_Gamyxx * l_Ayz + l_Gamzxx * l_Azz) - chix * l_Axz / chin1;
    double DA_yyx = d_Ayy_x - (l_Gamxxy * l_Axy + l_Gamyxy * l_Ayy + l_Gamzxy * l_Ayz
                             + l_Gamxxy * l_Axy + l_Gamyxy * l_Ayy + l_Gamzxy * l_Ayz) - chix * l_Ayy / chin1;
    double DA_yzx = d_Ayz_x - (l_Gamxxz * l_Axy + l_Gamyxz * l_Ayy + l_Gamzxz * l_Ayz
                             + l_Gamxxy * l_Axz + l_Gamyxy * l_Ayz + l_Gamzxy * l_Azz) - chix * l_Ayz / chin1;
    double DA_zzx = d_Azz_x - (l_Gamxxz * l_Axz + l_Gamyxz * l_Ayz + l_Gamzxz * l_Azz
                             + l_Gamxxz * l_Axz + l_Gamyxz * l_Ayz + l_Gamzxz * l_Azz) - chix * l_Azz / chin1;
    double DA_xxy = d_Axx_y - (l_Gamxxy * l_Axx + l_Gamyxy * l_Axy + l_Gamzxy * l_Axz
                             + l_Gamxxy * l_Axx + l_Gamyxy * l_Axy + l_Gamzxy * l_Axz) - chiy * l_Axx / chin1;
    double DA_xyy = d_Axy_y - (l_Gamxyy * l_Axx + l_Gamyyy * l_Axy + l_Gamzyy * l_Axz
                             + l_Gamxxy * l_Axy + l_Gamyxy * l_Ayy + l_Gamzxy * l_Ayz) - chiy * l_Axy / chin1;
    double DA_xzy = d_Axz_y - (l_Gamxyz * l_Axx + l_Gamyyz * l_Axy + l_Gamzyz * l_Axz
                             + l_Gamxxy * l_Axz + l_Gamyxy * l_Ayz + l_Gamzxy * l_Azz) - chiy * l_Axz / chin1;
    double DA_yyy = d_Ayy_y - (l_Gamxyy * l_Axy + l_Gamyyy * l_Ayy + l_Gamzyy * l_Ayz
                             + l_Gamxyy * l_Axy + l_Gamyyy * l_Ayy + l_Gamzyy * l_Ayz) - chiy * l_Ayy / chin1; 
    double DA_yzy = d_Ayz_y - (l_Gamxyz * l_Axy + l_Gamyyz * l_Ayy + l_Gamzyz * l_Ayz
                             + l_Gamxyy * l_Axz + l_Gamyyy * l_Ayz + l_Gamzyy * l_Azz) - chiy * l_Ayz / chin1;
    double DA_zzy = d_Azz_y - (l_Gamxyz * l_Axz + l_Gamyyz * l_Ayz + l_Gamzyz * l_Azz
                             + l_Gamxyz * l_Axz + l_Gamyyz * l_Ayz + l_Gamzyz * l_Azz) - chiy * l_Azz / chin1;
    double DA_xxz = d_Axx_z - (l_Gamxxz * l_Axx + l_Gamyxz * l_Axy + l_Gamzxz * l_Axz
                             + l_Gamxxz * l_Axx + l_Gamyxz * l_Axy + l_Gamzxz * l_Axz) - chiz * l_Axx / chin1;
    double DA_xyz = d_Axy_z - (l_Gamxyz * l_Axx + l_Gamyyz * l_Axy + l_Gamzyz * l_Axz
                             + l_Gamxxz * l_Axy + l_Gamyxz * l_Ayy + l_Gamzxz * l_Ayz) - chiz * l_Axy / chin1;
    double DA_xzz = d_Axz_z - (l_Gamxzz * l_Axx + l_Gamyzz * l_Axy + l_Gamzzz * l_Axz
                             + l_Gamxxz * l_Axz + l_Gamyxz * l_Ayz + l_Gamzxz * l_Azz) - chiz * l_Axz / chin1;
    double DA_yyz = d_Ayy_z - (l_Gamxyz * l_Axy + l_Gamyyz * l_Ayy + l_Gamzyz * l_Ayz
                             + l_Gamxyz * l_Axy + l_Gamyyz * l_Ayy + l_Gamzyz * l_Ayz) - chiz * l_Ayy / chin1;
    double DA_yzz = d_Ayz_z - (l_Gamxzz * l_Axy + l_Gamyzz * l_Ayy + l_Gamzzz * l_Ayz
                             + l_Gamxyz * l_Axz + l_Gamyyz * l_Ayz + l_Gamzyz * l_Azz) - chiz * l_Ayz / chin1;
    double DA_zzz = d_Azz_z - (l_Gamxzz * l_Axz + l_Gamyzz * l_Ayz + l_Gamzzz * l_Azz
                             + l_Gamxzz * l_Axz + l_Gamyzz * l_Ayz + l_Gamzzz * l_Azz) - chiz * l_Azz / chin1;


    // ==========================================
    // 3. Contraction (Compute mov_Res)
    // ==========================================
    
    // movx_Res (Fortran Lines 424-426)
    // Note: Use matching DA components. 
    // gupxx*gxxx -> gupxx * DA_xxx
    // gupyy*gxyy -> gupyy * DA_xyy
    // gupzz*gxzz -> gupzz * DA_xzz
    // gupxy*gxyx -> gupxy * DA_xyx
    // gupxz*gxzx -> gupxz * DA_xzx
    // gupyz*gxzy -> gupyz * DA_xzy
    // gupxy*gxxy -> gupxy * DA_xxy
    // gupxz*gxxz -> gupxz * DA_xxz
    // gupyz*gxyz -> gupyz * DA_xyz
    movx_Res[idx] = gupxx * DA_xxx + gupyy * DA_xyy + gupzz * DA_xzz
                  + gupxy * DA_xyx + gupxz * DA_xzx + gupyz * DA_xzy
                  + gupxy * DA_xxy + gupxz * DA_xxz + gupyz * DA_xyz;

    // movy_Res (Fortran Lines 427-429)
    movy_Res[idx] = gupxx * DA_xyx + gupyy * DA_yyy + gupzz * DA_yzz
                  + gupxy * DA_yyx + gupxz * DA_yzx + gupyz * DA_yzy
                  + gupxy * DA_xyy + gupxz * DA_xyz + gupyz * DA_yyz;

    // movz_Res (Fortran Lines 430-432)
    movz_Res[idx] = gupxx * DA_xzx + gupyy * DA_yzy + gupzz * DA_zzz
                  + gupxy * DA_yzx + gupxz * DA_zzx + gupyz * DA_zzy
                  + gupxy * DA_xzy + gupxz * DA_xzz + gupyz * DA_yzz; // Note: last term gupyz*gyzz -> DA_yzz

    // Subtract K derivatives and Matter terms
    // Fortran Lines 434-436
    movx_Res[idx] = movx_Res[idx] - F2o3 * Kx - F8 * PI * val_Sx;
    movy_Res[idx] = movy_Res[idx] - F2o3 * Ky - F8 * PI * val_Sy;
    movz_Res[idx] = movz_Res[idx] - F2o3 * Kz - F8 * PI * val_Sz;
}


__global__ __launch_bounds__(256, 2) void rhs_kernel_int_advection(
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
    int symmetry, int lev, double eps, int co
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 2;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 3;
    if (i >= ex0 || j >= ex1 || k >= ex2) return;
    if (i < 2 || i > ex0 - 3 || j < 2 || j > ex1 - 3 || k < 3 || k > ex2 - 3) return;
    int idx = IDX3D(i, j, k, ex0, ex1, ex2);
    int dims[3] = {ex0, ex1, ex2};
    const int adv_nx = ex0;
    const int adv_nxy = ex0 * ex1;

    const double adv_dX = X[1] - X[0];
    const double adv_dY = Y[1] - Y[0];
    const double adv_dZ = Z[1] - Z[0];
    const double adv_d12dx = 1.0 / 12.0 / adv_dX;
    const double adv_d12dy = 1.0 / 12.0 / adv_dY;
    const double adv_d12dz = 1.0 / 12.0 / adv_dZ;
    const double adv_eps_over_cof = eps / 64.0;
    const int adv_imax = ex0 - 1, adv_jmax = ex1 - 1, adv_kmax = ex2 - 1;
    const int adv_imin = (symmetry > 1 && fabs(X[0]) < adv_dX) ? -3 : 0;
    const int adv_jmin = (symmetry > 1 && fabs(Y[0]) < adv_dY) ? -3 : 0;
    const int adv_kmin = (symmetry > 0 && fabs(Z[0]) < adv_dZ) ? -3 : 0;
    const double adv_vx = betax[idx], adv_vy = betay[idx], adv_vz = betaz[idx];
    // ------------------------------------------------------------------------------------
    // bssn_advection_dissipation_kernel
    // ------------------------------------------------------------------------------------

    // 准备平流所需的速度场 (Shift)
    // lopsided 需要传入 shift 的指针来判断上风方向
    // device 函数内部会根据 i,j,k 读取 betax[idx] 等

    // 定义对称性常量 (对应 Fortran 的 array 定义)
    // SSS: (1, 1, 1)
    // AAS: (-1, -1, 1)
    // ASA: (-1, 1, -1)
    // SAA: (1, -1, -1)
    // ASS: (-1, 1, 1)
    // SAS: (1, -1, 1)
    // SSA: (1, 1, -1)

    // =========================================================
    // Block 1: Metric Variables (gxx, gxy, gxz, gyy, gyz, gzz)
    // =========================================================
    
    // gxx (SSS)
    // Fortran: call lopsided(..., gxx, gxx_rhs, ..., SSS)
    // Note: Passing dxx for derivative calculation is equivalent to gxx
    gxx_rhs[idx] += d_lopsided_point_int_fast(dxx, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) gxx_rhs[idx] += d_kodis_point_int_fast(dxx, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // gxy (AAS)
    gxy_rhs[idx] += d_lopsided_point_int_fast(gxy, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) gxy_rhs[idx] += d_kodis_point_int_fast(gxy, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // gxz (ASA)
    gxz_rhs[idx] += d_lopsided_point_int_fast(gxz, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) gxz_rhs[idx] += d_kodis_point_int_fast(gxz, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // gyy (SSS)
    gyy_rhs[idx] += d_lopsided_point_int_fast(dyy, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) gyy_rhs[idx] += d_kodis_point_int_fast(dyy, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // gyz (SAA)
    gyz_rhs[idx] += d_lopsided_point_int_fast(gyz, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) gyz_rhs[idx] += d_kodis_point_int_fast(gyz, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // gzz (SSS)
    gzz_rhs[idx] += d_lopsided_point_int_fast(dzz, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) gzz_rhs[idx] += d_kodis_point_int_fast(dzz, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // =========================================================
    // Block 2: Extrinsic Curvature (Axx ... Azz)
    // =========================================================

    // Axx (SSS)
    Axx_rhs[idx] += d_lopsided_point_int_fast(Axx, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) Axx_rhs[idx] += d_kodis_point_int_fast(Axx, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // Axy (AAS)
    Axy_rhs[idx] += d_lopsided_point_int_fast(Axy, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) Axy_rhs[idx] += d_kodis_point_int_fast(Axy, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // Axz (ASA)
    Axz_rhs[idx] += d_lopsided_point_int_fast(Axz, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) Axz_rhs[idx] += d_kodis_point_int_fast(Axz, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // Ayy (SSS)
    Ayy_rhs[idx] += d_lopsided_point_int_fast(Ayy, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) Ayy_rhs[idx] += d_kodis_point_int_fast(Ayy, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // Ayz (SAA)
    Ayz_rhs[idx] += d_lopsided_point_int_fast(Ayz, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) Ayz_rhs[idx] += d_kodis_point_int_fast(Ayz, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // Azz (SSS)
    Azz_rhs[idx] += d_lopsided_point_int_fast(Azz, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) Azz_rhs[idx] += d_kodis_point_int_fast(Azz, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // =========================================================
    // Block 3: Scalar Variables (chi, trK)
    // =========================================================

    // chi (SSS)
    chi_rhs[idx] += d_lopsided_point_int_fast(chi, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) chi_rhs[idx] += d_kodis_point_int_fast(chi, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // trK (SSS)
    trK_rhs[idx] += d_lopsided_point_int_fast(trK, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) trK_rhs[idx] += d_kodis_point_int_fast(trK, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // =========================================================
    // Block 4: Gauge Variables - Conformal Connection (Gam)
    // =========================================================

    // Gamx (ASS)
    Gamx_rhs[idx] += d_lopsided_point_int_fast(Gamx, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) Gamx_rhs[idx] += d_kodis_point_int_fast(Gamx, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // Gamy (SAS)
    Gamy_rhs[idx] += d_lopsided_point_int_fast(Gamy, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) Gamy_rhs[idx] += d_kodis_point_int_fast(Gamy, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // Gamz (SSA)
    Gamz_rhs[idx] += d_lopsided_point_int_fast(Gamz, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) Gamz_rhs[idx] += d_kodis_point_int_fast(Gamz, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // =========================================================
    // Block 5: Gauge Variables - Lapse & Shift
    // =========================================================

    // Lap (SSS) - Note: bam code does not apply dissipation on gauge vars usually, but Fortran logic here DOES for Lap
    Lap_rhs[idx] += d_lopsided_point_int_fast(Lap, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) Lap_rhs[idx] += d_kodis_point_int_fast(Lap, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // betax (ASS)
    betax_rhs[idx] += d_lopsided_point_int_fast(betax, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) betax_rhs[idx] += d_kodis_point_int_fast(betax, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // betay (SAS)
    betay_rhs[idx] += d_lopsided_point_int_fast(betay, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) betay_rhs[idx] += d_kodis_point_int_fast(betay, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // betaz (SSA)
    betaz_rhs[idx] += d_lopsided_point_int_fast(betaz, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) betaz_rhs[idx] += d_kodis_point_int_fast(betaz, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // =========================================================
    // Block 6: Gauge Variables - Time derivative of Shift (dtSf)
    // =========================================================

    // dtSfx (ASS)
    dtSfx_rhs[idx] += d_lopsided_point_int_fast(dtSfx, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) dtSfx_rhs[idx] += d_kodis_point_int_fast(dtSfx, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // dtSfy (SAS)
    dtSfy_rhs[idx] += d_lopsided_point_int_fast(dtSfy, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) dtSfy_rhs[idx] += d_kodis_point_int_fast(dtSfy, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);

    // dtSfz (SSA)
    dtSfz_rhs[idx] += d_lopsided_point_int_fast(dtSfz, adv_vx, adv_vy, adv_vz, adv_d12dx, adv_d12dy, adv_d12dz, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);
    if (eps > 0.0) dtSfz_rhs[idx] += d_kodis_point_int_fast(dtSfz, adv_dX, adv_dY, adv_dZ, adv_eps_over_cof, adv_nx, adv_nxy, adv_imax, adv_jmax, adv_kmax, adv_imin, adv_jmin, adv_kmin, idx, i, j, k);


}

void gpu_compute_rhs_bssn_launch_int( // launch kernel with device pointers
    cudaStream_t &stream,
    int* ex, double T, double* d_X, double* d_Y, double* d_Z,
    double* d_chi, double* d_trK,
    double* d_dxx, double* d_gxy, double* d_gxz,
    double* d_dyy, double* d_gyz, double* d_dzz,
    double* d_Axx, double* d_Axy, double* d_Axz,
    double* d_Ayy, double* d_Ayz, double* d_Azz,
    double* d_Gamx, double* d_Gamy, double* d_Gamz,
    double* d_Lap,
    double* d_betax, double* d_betay, double* d_betaz,
    double* d_dtSfx, double* d_dtSfy, double* d_dtSfz,
    double* d_chi_rhs, double* d_trK_rhs,
    double* d_gxx_rhs, double* d_gxy_rhs, double* d_gxz_rhs,
    double* d_gyy_rhs, double* d_gyz_rhs, double* d_gzz_rhs,
    double* d_Axx_rhs, double* d_Axy_rhs, double* d_Axz_rhs,
    double* d_Ayy_rhs, double* d_Ayz_rhs, double* d_Azz_rhs,
    double* d_Gamx_rhs, double* d_Gamy_rhs, double* d_Gamz_rhs,
    double* d_Lap_rhs,
    double* d_betax_rhs, double* d_betay_rhs, double* d_betaz_rhs,
    double* d_dtSfx_rhs, double* d_dtSfy_rhs, double* d_dtSfz_rhs,
    double* d_rho, double* d_Sx, double* d_Sy, double* d_Sz,
    double* d_Sxx, double* d_Sxy, double* d_Sxz,
    double* d_Syy, double* d_Syz, double* d_Szz,
    double* d_Gamxxx, double* d_Gamxxy, double* d_Gamxxz,
    double* d_Gamxyy, double* d_Gamxyz, double* d_Gamxzz,
    double* d_Gamyxx, double* d_Gamyxy, double* d_Gamyxz,
    double* d_Gamyyy, double* d_Gamyyz, double* d_Gamyzz,
    double* d_Gamzxx, double* d_Gamzxy, double* d_Gamzxz,
    double* d_Gamzyy, double* d_Gamzyz, double* d_Gamzzz,
    double* d_Rxx, double* d_Rxy, double* d_Rxz,
    double* d_Ryy, double* d_Ryz, double* d_Rzz,
    double* d_ham_Res, double* d_movx_Res, double* d_movy_Res, double* d_movz_Res,
    double* d_Gmx_Res, double* d_Gmy_Res, double* d_Gmz_Res,
    int symmetry, int lev, double eps, int co
) {
    dim3 stage_block(8, 8, 2); // 128 threads for staged kernels
    dim3 block(8, 8, 4); // 256 threads for the advection kernel
    // The three kernels only update the strict interior.  Keep the same
    // 8x8x4 point tile, but start it at (2,2,3) and omit boundary tiles.
    const int ni = (ex[0] > 4 ? ex[0] - 4 : 0);
    const int nj = (ex[1] > 4 ? ex[1] - 4 : 0);
    const int nk = (ex[2] > 5 ? ex[2] - 5 : 0);
    if (ni <= 0 || nj <= 0 || nk <= 0) return;
    dim3 stage_grid(
        (ni + stage_block.x - 1) / stage_block.x,
        (nj + stage_block.y - 1) / stage_block.y,
        (nk + stage_block.z - 1) / stage_block.z
    );
    dim3 grid(
        (ni + block.x - 1) / block.x,
        (nj + block.y - 1) / block.y,
        (nk + block.z - 1) / block.z
    );

    // 1. Kernel 1: Derivatives & Connection Coefficients
    rhs_kernel_int_stage1<<<stage_grid, stage_block, 0, stream>>>(
        ex[0], ex[1], ex[2], T, d_X, d_Y, d_Z,
        d_chi, d_trK,
        d_dxx, d_gxy, d_gxz,
        d_dyy, d_gyz, d_dzz,
        d_Axx, d_Axy, d_Axz,
        d_Ayy, d_Ayz, d_Azz,
        d_Gamx, d_Gamy, d_Gamz,
        d_Lap,
        d_betax, d_betay, d_betaz,
        d_dtSfx, d_dtSfy, d_dtSfz,
        d_chi_rhs, d_trK_rhs,
        d_gxx_rhs, d_gxy_rhs, d_gxz_rhs,
        d_gyy_rhs, d_gyz_rhs, d_gzz_rhs,
        d_Axx_rhs, d_Axy_rhs, d_Axz_rhs,
        d_Ayy_rhs, d_Ayz_rhs, d_Azz_rhs,
        d_Gamx_rhs, d_Gamy_rhs, d_Gamz_rhs,
        d_Lap_rhs,
        d_betax_rhs, d_betay_rhs, d_betaz_rhs,
        d_dtSfx_rhs, d_dtSfy_rhs, d_dtSfz_rhs,
        d_rho, d_Sx, d_Sy, d_Sz,
        d_Sxx, d_Sxy, d_Sxz,
        d_Syy, d_Syz, d_Szz,
        d_Gamxxx, d_Gamxxy, d_Gamxxz,
        d_Gamxyy, d_Gamxyz, d_Gamxzz,
        d_Gamyxx, d_Gamyxy, d_Gamyxz,
        d_Gamyyy, d_Gamyyz, d_Gamyzz,
        d_Gamzxx, d_Gamzxy, d_Gamzxz,
        d_Gamzyy, d_Gamzyz, d_Gamzzz,
        d_Rxx, d_Rxy, d_Rxz,
        d_Ryy, d_Ryz, d_Rzz,
        d_ham_Res, d_movx_Res, d_movy_Res, d_movz_Res,
        d_Gmx_Res, d_Gmy_Res, d_Gmz_Res,
        symmetry, lev, eps, co
    );
    rhs_kernel_int_stage2<<<stage_grid, stage_block, 0, stream>>>(
        ex[0], ex[1], ex[2], T, d_X, d_Y, d_Z,
        d_chi, d_trK,
        d_dxx, d_gxy, d_gxz,
        d_dyy, d_gyz, d_dzz,
        d_Axx, d_Axy, d_Axz,
        d_Ayy, d_Ayz, d_Azz,
        d_Gamx, d_Gamy, d_Gamz,
        d_Lap,
        d_betax, d_betay, d_betaz,
        d_dtSfx, d_dtSfy, d_dtSfz,
        d_chi_rhs, d_trK_rhs,
        d_gxx_rhs, d_gxy_rhs, d_gxz_rhs,
        d_gyy_rhs, d_gyz_rhs, d_gzz_rhs,
        d_Axx_rhs, d_Axy_rhs, d_Axz_rhs,
        d_Ayy_rhs, d_Ayz_rhs, d_Azz_rhs,
        d_Gamx_rhs, d_Gamy_rhs, d_Gamz_rhs,
        d_Lap_rhs,
        d_betax_rhs, d_betay_rhs, d_betaz_rhs,
        d_dtSfx_rhs, d_dtSfy_rhs, d_dtSfz_rhs,
        d_rho, d_Sx, d_Sy, d_Sz,
        d_Sxx, d_Sxy, d_Sxz,
        d_Syy, d_Syz, d_Szz,
        d_Gamxxx, d_Gamxxy, d_Gamxxz,
        d_Gamxyy, d_Gamxyz, d_Gamxzz,
        d_Gamyxx, d_Gamyxy, d_Gamyxz,
        d_Gamyyy, d_Gamyyz, d_Gamyzz,
        d_Gamzxx, d_Gamzxy, d_Gamzxz,
        d_Gamzyy, d_Gamzyz, d_Gamzzz,
        d_Rxx, d_Rxy, d_Rxz,
        d_Ryy, d_Ryz, d_Rzz,
        d_ham_Res, d_movx_Res, d_movy_Res, d_movz_Res,
        d_Gmx_Res, d_Gmy_Res, d_Gmz_Res,
        symmetry, lev, eps, co
    );

    rhs_kernel_int_advection<<<grid, block, 0, stream>>>(
        ex[0], ex[1], ex[2], T, d_X, d_Y, d_Z,
        d_chi, d_trK,
        d_dxx, d_gxy, d_gxz,
        d_dyy, d_gyz, d_dzz,
        d_Axx, d_Axy, d_Axz,
        d_Ayy, d_Ayz, d_Azz,
        d_Gamx, d_Gamy, d_Gamz,
        d_Lap,
        d_betax, d_betay, d_betaz,
        d_dtSfx, d_dtSfy, d_dtSfz,
        d_chi_rhs, d_trK_rhs,
        d_gxx_rhs, d_gxy_rhs, d_gxz_rhs,
        d_gyy_rhs, d_gyz_rhs, d_gzz_rhs,
        d_Axx_rhs, d_Axy_rhs, d_Axz_rhs,
        d_Ayy_rhs, d_Ayz_rhs, d_Azz_rhs,
        d_Gamx_rhs, d_Gamy_rhs, d_Gamz_rhs,
        d_Lap_rhs,
        d_betax_rhs, d_betay_rhs, d_betaz_rhs,
        d_dtSfx_rhs, d_dtSfy_rhs, d_dtSfz_rhs,
        d_rho, d_Sx, d_Sy, d_Sz,
        d_Sxx, d_Sxy, d_Sxz,
        d_Syy, d_Syz, d_Szz,
        d_Gamxxx, d_Gamxxy, d_Gamxxz,
        d_Gamxyy, d_Gamxyz, d_Gamxzz,
        d_Gamyxx, d_Gamyxy, d_Gamyxz,
        d_Gamyyy, d_Gamyyz, d_Gamyzz,
        d_Gamzxx, d_Gamzxy, d_Gamzxz,
        d_Gamzyy, d_Gamzyz, d_Gamzzz,
        d_Rxx, d_Rxy, d_Rxz,
        d_Ryy, d_Ryz, d_Rzz,
        d_ham_Res, d_movx_Res, d_movy_Res, d_movz_Res,
        d_Gmx_Res, d_Gmy_Res, d_Gmz_Res,
        symmetry, lev, eps, co
    );
}
