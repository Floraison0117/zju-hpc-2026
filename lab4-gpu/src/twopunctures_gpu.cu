#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <thrust/complex.h>

using thrust::complex;

extern "C" void gpu_twop_free(void);

#ifndef PI
#define PI 3.14159265358979323846264338328
#endif
#ifndef PIH
#define PIH 1.57079632679489661923132169164
#endif

struct gpu_derivs {
    double *d0, *d1, *d2, *d3, *d11, *d12, *d13, *d22, *d23, *d33;
};

struct gpu_params {
    double par_b;
    double par_m_plus, par_m_minus;
    double par_P_plus[3], par_P_minus[3];
    double par_S_plus[3], par_S_minus[3];
};

static void set_params(gpu_params *p,
                       double par_b, double par_m_plus, double par_m_minus,
                       const double par_P_plus[3], const double par_P_minus[3],
                       const double par_S_plus[3], const double par_S_minus[3])
{
    p->par_b = par_b;
    p->par_m_plus = par_m_plus;
    p->par_m_minus = par_m_minus;
    for (int i = 0; i < 3; i++) {
        p->par_P_plus[i] = par_P_plus[i];
        p->par_P_minus[i] = par_P_minus[i];
        p->par_S_plus[i] = par_S_plus[i];
        p->par_S_minus[i] = par_S_minus[i];
    }
}

/* ===========================================================================
 * Host-side buffer management and host<->device transfer.
 *
 * Two implementations selected at compile time:
 *   - default (no macro):              per-array cudaMalloc + synchronous
 *                                     cudaMemcpy + cudaDeviceSynchronize.
 *                                     Byte-identical to the deployed baseline.
 *   - USE_GPU_ASYNC defined:           one CUDA stream + a single pinned-host
 *                                     staging buffer per derivs struct; the 10
 *                                     per-call H2D/D2H copies are coalesced
 *                                     into one cudaMemcpyAsync and ordered on
 *                                     the stream, replacing cudaDeviceSynchronize
 *                                     with cudaStreamSynchronize.  Kernel
 *                                     arithmetic is unchanged.
 * ========================================================================== */
#ifndef USE_GPU_ASYNC

/* --------------------------- original synchronous path ------------------- */
static gpu_derivs g_d_work = {0};
static gpu_derivs g_d_u    = {0};
static double *g_d_Jdv     = 0;
static double *g_d_F       = 0;
static double *g_d_sources = 0;
static int    g_ntotal     = 0;
static bool   g_u_valid    = false;
static gpu_params g_params_cache;

static void alloc_derivs(gpu_derivs *d, int n)
{
    size_t bytes = n * sizeof(double);
    cudaMalloc(&d->d0,  bytes); cudaMalloc(&d->d1,  bytes);
    cudaMalloc(&d->d2,  bytes); cudaMalloc(&d->d3,  bytes);
    cudaMalloc(&d->d11, bytes); cudaMalloc(&d->d12, bytes);
    cudaMalloc(&d->d13, bytes); cudaMalloc(&d->d22, bytes);
    cudaMalloc(&d->d23, bytes); cudaMalloc(&d->d33, bytes);
}

static void free_derivs(gpu_derivs *d)
{
    if (d->d0)  { cudaFree(d->d0);  cudaFree(d->d1);  cudaFree(d->d2);  cudaFree(d->d3); }
    if (d->d11) { cudaFree(d->d11); cudaFree(d->d12); cudaFree(d->d13); }
    if (d->d22) { cudaFree(d->d22); cudaFree(d->d23); cudaFree(d->d33); }
    d->d0 = d->d1 = d->d2 = d->d3 = 0;
    d->d11 = d->d12 = d->d13 = 0;
    d->d22 = d->d23 = d->d33 = 0;
}

static void upload_derivs(gpu_derivs *d_dst,
                          const double *h_d0,  const double *h_d1,
                          const double *h_d2,  const double *h_d3,
                          const double *h_d11, const double *h_d12,
                          const double *h_d13, const double *h_d22,
                          const double *h_d23, const double *h_d33,
                          int n)
{
    size_t bytes = n * sizeof(double);
    cudaMemcpy(d_dst->d0,  h_d0,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst->d1,  h_d1,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst->d2,  h_d2,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst->d3,  h_d3,  bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst->d11, h_d11, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst->d12, h_d12, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst->d13, h_d13, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst->d22, h_d22, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst->d23, h_d23, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst->d33, h_d33, bytes, cudaMemcpyHostToDevice);
}

static void download_derivs(double *h_d0,  double *h_d1,
                            double *h_d2,  double *h_d3,
                            double *h_d11, double *h_d12,
                            double *h_d13, double *h_d22,
                            double *h_d23, double *h_d33,
                            const gpu_derivs *d_src, int n)
{
    size_t bytes = n * sizeof(double);
    cudaMemcpy(h_d0,  d_src->d0,  bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d1,  d_src->d1,  bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d2,  d_src->d2,  bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d3,  d_src->d3,  bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d11, d_src->d11, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d12, d_src->d12, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d13, d_src->d13, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d22, d_src->d22, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d23, d_src->d23, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d33, d_src->d33, bytes, cudaMemcpyDeviceToHost);
}

extern "C" void gpu_twop_init(int nvar, int n1, int n2, int n3)
{
    int ntotal = n1 * n2 * n3 * nvar;
    if (g_ntotal == ntotal && g_d_work.d0) return;
    gpu_twop_free();
    g_ntotal = ntotal;
    alloc_derivs(&g_d_work, ntotal);
    alloc_derivs(&g_d_u,    ntotal);
    cudaMalloc(&g_d_Jdv,     ntotal * sizeof(double));
    cudaMalloc(&g_d_F,       ntotal * sizeof(double));
    cudaMalloc(&g_d_sources, ntotal * sizeof(double));
    cudaMemset(g_d_sources, 0, ntotal * sizeof(double));
    g_u_valid = false;
}

extern "C" void gpu_twop_free()
{
    free_derivs(&g_d_work);
    free_derivs(&g_d_u);
    if (g_d_Jdv)     { cudaFree(g_d_Jdv);     g_d_Jdv     = 0; }
    if (g_d_F)       { cudaFree(g_d_F);       g_d_F       = 0; }
    if (g_d_sources) { cudaFree(g_d_sources); g_d_sources = 0; }
    g_ntotal  = 0;
    g_u_valid = false;
}

extern "C" void gpu_upload_u(int ntotal,
                              const double *h_u_d0,  const double *h_u_d1,
                              const double *h_u_d2,  const double *h_u_d3,
                              const double *h_u_d11, const double *h_u_d12,
                              const double *h_u_d13, const double *h_u_d22,
                              const double *h_u_d23, const double *h_u_d33)
{
    upload_derivs(&g_d_u, h_u_d0, h_u_d1, h_u_d2, h_u_d3,
                  h_u_d11, h_u_d12, h_u_d13,
                  h_u_d22, h_u_d23, h_u_d33, ntotal);
    g_u_valid = true;
}

#else  /* USE_GPU_ASYNC — stream + coalesced pinned-staged transfers */

static gpu_derivs g_d_work = {0};
static gpu_derivs g_d_u    = {0};
static double *g_d_Jdv     = 0;
static double *g_d_F       = 0;
static double *g_d_sources = 0;
static int    g_ntotal     = 0;
static bool   g_u_valid    = false;
static gpu_params g_params_cache;

static cudaStream_t g_stream = 0;
/* One contiguous device buffer and one pinned host buffer per derivs struct.
 * g_d_work.d0..d33 are aliases into g_d_stage_work[0..10*ntotal);
 * g_d_u.d0..d33 are aliases into g_d_stage_u[0..10*ntotal). */
static double *g_d_stage_work = 0;
static double *g_d_stage_u    = 0;
static double *g_h_stage_work = 0;
static double *g_h_stage_u    = 0;

static void set_stage_aliases()
{
    double *w = g_d_stage_work;
    g_d_work.d0 = w;  g_d_work.d1 = w + g_ntotal;  g_d_work.d2 = w + 2*g_ntotal;
    g_d_work.d3 = w + 3*g_ntotal;  g_d_work.d11 = w + 4*g_ntotal;
    g_d_work.d12 = w + 5*g_ntotal; g_d_work.d13 = w + 6*g_ntotal;
    g_d_work.d22 = w + 7*g_ntotal; g_d_work.d23 = w + 8*g_ntotal;
    g_d_work.d33 = w + 9*g_ntotal;
    double *u = g_d_stage_u;
    g_d_u.d0 = u;  g_d_u.d1 = u + g_ntotal;  g_d_u.d2 = u + 2*g_ntotal;
    g_d_u.d3 = u + 3*g_ntotal;  g_d_u.d11 = u + 4*g_ntotal;
    g_d_u.d12 = u + 5*g_ntotal; g_d_u.d13 = u + 6*g_ntotal;
    g_d_u.d22 = u + 7*g_ntotal; g_d_u.d23 = u + 8*g_ntotal;
    g_d_u.d33 = u + 9*g_ntotal;
}

/* Gather 10 host source arrays into the contiguous pinned staging buffer, then
 * issue a single async H2D copy on the stream.  The device destination is the
 * base of the stage buffer (g_d_stage_work or g_d_stage_u); the kernel reads
 * the same data through the d0..d33 aliases. */
static void upload_derivs_async(double *d_stage_base, double *h_stage,
                                const double *h_d0,  const double *h_d1,
                                const double *h_d2,  const double *h_d3,
                                const double *h_d11, const double *h_d12,
                                const double *h_d13, const double *h_d22,
                                const double *h_d23, const double *h_d33,
                                int n)
{
    size_t bytes = (size_t)n * sizeof(double);
    memcpy(h_stage + 0*n, h_d0,  bytes);
    memcpy(h_stage + 1*n, h_d1,  bytes);
    memcpy(h_stage + 2*n, h_d2,  bytes);
    memcpy(h_stage + 3*n, h_d3,  bytes);
    memcpy(h_stage + 4*n, h_d11, bytes);
    memcpy(h_stage + 5*n, h_d12, bytes);
    memcpy(h_stage + 6*n, h_d13, bytes);
    memcpy(h_stage + 7*n, h_d22, bytes);
    memcpy(h_stage + 8*n, h_d23, bytes);
    memcpy(h_stage + 9*n, h_d33, bytes);
    cudaMemcpyAsync(d_stage_base, h_stage, 10 * bytes,
                    cudaMemcpyHostToDevice, g_stream);
}

/* Async D2H of one derivs struct into the pinned staging buffer, then wait and
 * scatter into the 10 host destination arrays. */
static void download_derivs_async(const double *d_stage_base, double *h_stage,
                                  double *h_d0,  double *h_d1,
                                  double *h_d2,  double *h_d3,
                                  double *h_d11, double *h_d12,
                                  double *h_d13, double *h_d22,
                                  double *h_d23, double *h_d33,
                                  int n)
{
    size_t bytes = (size_t)n * sizeof(double);
    cudaMemcpyAsync(h_stage, d_stage_base, 10 * bytes,
                    cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
    memcpy(h_d0,  h_stage + 0*n, bytes);
    memcpy(h_d1,  h_stage + 1*n, bytes);
    memcpy(h_d2,  h_stage + 2*n, bytes);
    memcpy(h_d3,  h_stage + 3*n, bytes);
    memcpy(h_d11, h_stage + 4*n, bytes);
    memcpy(h_d12, h_stage + 5*n, bytes);
    memcpy(h_d13, h_stage + 6*n, bytes);
    memcpy(h_d22, h_stage + 7*n, bytes);
    memcpy(h_d23, h_stage + 8*n, bytes);
    memcpy(h_d33, h_stage + 9*n, bytes);
}

extern "C" void gpu_twop_init(int nvar, int n1, int n2, int n3)
{
    int ntotal = n1 * n2 * n3 * nvar;
    if (g_ntotal == ntotal && g_d_stage_work) return;
    gpu_twop_free();
    g_ntotal = ntotal;
    size_t bytes10 = (size_t)10 * ntotal * sizeof(double);
    cudaStreamCreate(&g_stream);
    cudaMalloc(&g_d_stage_work, bytes10);
    cudaMalloc(&g_d_stage_u,    bytes10);
    cudaMallocHost(&g_h_stage_work, bytes10);
    cudaMallocHost(&g_h_stage_u,    bytes10);
    set_stage_aliases();
    cudaMalloc(&g_d_Jdv,     ntotal * sizeof(double));
    cudaMalloc(&g_d_F,       ntotal * sizeof(double));
    cudaMalloc(&g_d_sources, ntotal * sizeof(double));
    cudaMemsetAsync(g_d_sources, 0, ntotal * sizeof(double), g_stream);
    g_u_valid = false;
}

extern "C" void gpu_twop_free()
{
    if (g_d_stage_work) { cudaFree(g_d_stage_work); g_d_stage_work = 0; }
    if (g_d_stage_u)    { cudaFree(g_d_stage_u);    g_d_stage_u    = 0; }
    if (g_h_stage_work) { cudaFreeHost(g_h_stage_work); g_h_stage_work = 0; }
    if (g_h_stage_u)    { cudaFreeHost(g_h_stage_u);    g_h_stage_u    = 0; }
    if (g_d_Jdv)     { cudaFree(g_d_Jdv);     g_d_Jdv     = 0; }
    if (g_d_F)       { cudaFree(g_d_F);       g_d_F       = 0; }
    if (g_d_sources) { cudaFree(g_d_sources); g_d_sources = 0; }
    if (g_stream)    { cudaStreamDestroy(g_stream); g_stream = 0; }
    g_ntotal  = 0;
    g_u_valid = false;
    g_d_work = gpu_derivs{0,0,0,0,0,0,0,0,0,0};
    g_d_u    = gpu_derivs{0,0,0,0,0,0,0,0,0,0};
}

extern "C" void gpu_upload_u(int ntotal,
                              const double *h_u_d0,  const double *h_u_d1,
                              const double *h_u_d2,  const double *h_u_d3,
                              const double *h_u_d11, const double *h_u_d12,
                              const double *h_u_d13, const double *h_u_d22,
                              const double *h_u_d23, const double *h_u_d33)
{
    /* Not invoked by the current host driver; kept for ABI parity with the
     * synchronous path.  Requires a prior gpu_twop_init (done by every
     * gpu_J_times_dv / gpu_F_of_v call). */
    if (g_d_stage_u && g_h_stage_u && g_stream) {
        upload_derivs_async(g_d_stage_u, g_h_stage_u,
                            h_u_d0, h_u_d1, h_u_d2, h_u_d3,
                            h_u_d11, h_u_d12, h_u_d13,
                            h_u_d22, h_u_d23, h_u_d33, ntotal);
        cudaStreamSynchronize(g_stream);
        g_u_valid = true;
    }
}

#endif /* USE_GPU_ASYNC */

/* ============================ shared device code ========================= */

__device__ double dev_BY_KKofxyz(double x, double y, double z, const gpu_params *p)
{
    int i, j;
    double r_plus, r2_plus, r3_plus, r_minus, r2_minus, r3_minus, np_Pp, nm_Pm,
        Aij, AijAij, n_plus[3], n_minus[3], np_Sp[3], nm_Sm[3];

    r2_plus = (x - p->par_b) * (x - p->par_b) + y * y + z * z;
    r2_minus = (x + p->par_b) * (x + p->par_b) + y * y + z * z;
    r_plus = sqrt(r2_plus);
    r_minus = sqrt(r2_minus);
    r3_plus = r_plus * r2_plus;
    r3_minus = r_minus * r2_minus;

    n_plus[0] = (x - p->par_b) / r_plus;
    n_minus[0] = (x + p->par_b) / r_minus;
    n_plus[1] = y / r_plus;
    n_minus[1] = y / r_minus;
    n_plus[2] = z / r_plus;
    n_minus[2] = z / r_minus;

    np_Pp = 0; nm_Pm = 0;
    for (i = 0; i < 3; i++) {
        np_Pp += n_plus[i] * p->par_P_plus[i];
        nm_Pm += n_minus[i] * p->par_P_minus[i];
    }
    np_Sp[0] = n_plus[1] * p->par_S_plus[2] - n_plus[2] * p->par_S_plus[1];
    np_Sp[1] = n_plus[2] * p->par_S_plus[0] - n_plus[0] * p->par_S_plus[2];
    np_Sp[2] = n_plus[0] * p->par_S_plus[1] - n_plus[1] * p->par_S_plus[0];
    nm_Sm[0] = n_minus[1] * p->par_S_minus[2] - n_minus[2] * p->par_S_minus[1];
    nm_Sm[1] = n_minus[2] * p->par_S_minus[0] - n_minus[0] * p->par_S_minus[2];
    nm_Sm[2] = n_minus[0] * p->par_S_minus[1] - n_minus[1] * p->par_S_minus[0];
    AijAij = 0;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            Aij = +1.5 * (p->par_P_plus[i] * n_plus[j] + p->par_P_plus[j] * n_plus[i] + np_Pp * n_plus[i] * n_plus[j]) / r2_plus
                + 1.5 * (p->par_P_minus[i] * n_minus[j] + p->par_P_minus[j] * n_minus[i] + nm_Pm * n_minus[i] * n_minus[j]) / r2_minus
                - 3.0 * (np_Sp[i] * n_plus[j] + np_Sp[j] * n_plus[i]) / r3_plus
                - 3.0 * (nm_Sm[i] * n_minus[j] + nm_Sm[j] * n_minus[i]) / r3_minus;
            if (i == j) Aij -= +1.5 * (np_Pp / r2_plus + nm_Pm / r2_minus);
            AijAij += Aij * Aij;
        }
    }
    return AijAij;
}

__device__ void dev_AB_To_XR(int nvar, double A, double B, double *X, double *R,
                             double *U_d0, double *U_d1, double *U_d2, double *U_d3,
                             double *U_d11, double *U_d12, double *U_d13,
                             double *U_d22, double *U_d23, double *U_d33)
{
    double At = 0.5 * (A + 1), A_X, A_XX, B_R, B_RR;
    *X = 2.0 * atanh(At);
    *R = PIH + 2.0 * atan(B);
    A_X = 1.0 - At * At;
    A_XX = -At * A_X;
    B_R = 0.5 * (1.0 + B * B);
    B_RR = B * B_R;

    for (int ivar = 0; ivar < nvar; ivar++) {
        U_d11[ivar] = A_X * A_X * U_d11[ivar] + A_XX * U_d1[ivar];
        U_d12[ivar] = A_X * B_R * U_d12[ivar];
        U_d13[ivar] = A_X * U_d13[ivar];
        U_d22[ivar] = B_R * B_R * U_d22[ivar] + B_RR * U_d2[ivar];
        U_d23[ivar] = B_R * U_d23[ivar];
        U_d1[ivar] = A_X * U_d1[ivar];
        U_d2[ivar] = B_R * U_d2[ivar];
    }
}

__device__ void dev_C_To_c(int nvar, double X, double R, double *x, double *r,
                           double par_b,
                           double *U_d0, double *U_d1, double *U_d2, double *U_d3,
                           double *U_d11, double *U_d12, double *U_d13,
                           double *U_d22, double *U_d23, double *U_d33)
{
    double C_c2, U_cb, U_CB;
    complex<double> C, C_c, C_cc, c, c_C, c_CC, U_c, U_cc, U_C, U_CC;

    C = complex<double>(X, R);
    c = cosh(C) * par_b;
    c_C = sinh(C) * par_b;
    c_CC = c;
    C_c = complex<double>(1.0, 0.0) / c_C;
    C_cc = -C_c * C_c * C_c * c_CC;
    C_c2 = abs(C_c);
    C_c2 = C_c2 * C_c2;

    for (int ivar = 0; ivar < nvar; ivar++) {
        U_C = complex<double>(0.5 * U_d13[ivar], -0.5 * U_d23[ivar]);
        U_c = U_C * C_c;
        U_d13[ivar] = 2.0 * U_c.real();
        U_d23[ivar] = -2.0 * U_c.imag();

        U_C = complex<double>(0.5 * U_d1[ivar], -0.5 * U_d2[ivar]);
        U_c = U_C * C_c;
        U_d1[ivar] = 2.0 * U_c.real();
        U_d2[ivar] = -2.0 * U_c.imag();

        U_CC = complex<double>(0.25 * (U_d11[ivar] - U_d22[ivar]), -0.5 * U_d12[ivar]);
        U_CB = 0.25 * (U_d11[ivar] + U_d22[ivar]);

        U_cb = U_CB * C_c2;
        U_cc = C_cc * U_C + C_c * C_c * U_CC;

        U_d11[ivar] = 2.0 * (U_cb + U_cc.real());
        U_d22[ivar] = 2.0 * (U_cb - U_cc.real());
        U_d12[ivar] = -2.0 * U_cc.imag();
    }
    *x = c.real();
    *r = c.imag();
}

__device__ void dev_rx3_To_xyz(int nvar, double x, double r, double phi,
                               double *y, double *z,
                               double *U_d0, double *U_d1, double *U_d2, double *U_d3,
                               double *U_d11, double *U_d12, double *U_d13,
                               double *U_d22, double *U_d23, double *U_d33)
{
    double sin_phi = sin(phi), cos_phi = cos(phi);
    double sin2_phi = sin_phi * sin_phi;
    double cos2_phi = cos_phi * cos_phi;
    double sin_2phi = 2.0 * sin_phi * cos_phi;
    double cos_2phi = cos2_phi - sin2_phi;
    double r_inv = 1.0 / r;
    double r_inv2 = r_inv * r_inv;

    *y = r * cos_phi;
    *z = r * sin_phi;

    for (int jvar = 0; jvar < nvar; jvar++) {
        double U_x = U_d1[jvar], U_r = U_d2[jvar], U_3 = U_d3[jvar];
        double U_xx = U_d11[jvar], U_xr = U_d12[jvar], U_x3 = U_d13[jvar];
        double U_rr = U_d22[jvar], U_r3 = U_d23[jvar], U_33 = U_d33[jvar];
        U_d1[jvar] = U_x;
        U_d2[jvar] = U_r * cos_phi - U_3 * r_inv * sin_phi;
        U_d3[jvar] = U_r * sin_phi + U_3 * r_inv * cos_phi;
        U_d11[jvar] = U_xx;
        U_d12[jvar] = U_xr * cos_phi - U_x3 * r_inv * sin_phi;
        U_d13[jvar] = U_xr * sin_phi + U_x3 * r_inv * cos_phi;
        U_d22[jvar] = U_rr * cos2_phi + r_inv2 * sin2_phi * (U_33 + r * U_r)
                      + sin_2phi * r_inv2 * (U_3 - r * U_r3);
        U_d23[jvar] = 0.5 * sin_2phi * (U_rr - r_inv * U_r - r_inv2 * U_33)
                      - cos_2phi * r_inv2 * (U_3 - r * U_r3);
        U_d33[jvar] = U_rr * sin2_phi + r_inv2 * cos2_phi * (U_33 + r * U_r)
                      - sin_2phi * r_inv2 * (U_3 - r * U_r3);
    }
}

__device__ void dev_LinEquations(double A, double B, double X, double R,
                                 double x, double r, double phi, double y, double z,
                                 double *dU_d0, double *dU_d11, double *dU_d22, double *dU_d33,
                                 double *U_d0, const gpu_params *p, double *values)
{
    double r_plus = sqrt((x - p->par_b) * (x - p->par_b) + y * y + z * z);
    double r_minus = sqrt((x + p->par_b) * (x + p->par_b) + y * y + z * z);
    double psi = 1.0 + 0.5 * p->par_m_plus / r_plus + 0.5 * p->par_m_minus / r_minus + U_d0[0];
    double psi2 = psi * psi;
    double psi4 = psi2 * psi2;
    double psi8 = psi4 * psi4;
    values[0] = dU_d11[0] + dU_d22[0] + dU_d33[0]
                - 0.875 * dev_BY_KKofxyz(x, y, z, p) / psi8 * dU_d0[0];
}

__device__ void dev_NonLinEquations(double rho_adm, double A, double B, double X, double R,
                                    double x, double r, double phi, double y, double z,
                                    double *U_d0, double *U_d11, double *U_d22, double *U_d33,
                                    const gpu_params *p, double *values)
{
    double r_plus = sqrt((x - p->par_b) * (x - p->par_b) + y * y + z * z);
    double r_minus = sqrt((x + p->par_b) * (x + p->par_b) + y * y + z * z);
    double psi = 1.0 + 0.5 * p->par_m_plus / r_plus + 0.5 * p->par_m_minus / r_minus + U_d0[0];
    double psi2 = psi * psi;
    double psi4 = psi2 * psi2;
    double psi7 = psi * psi2 * psi4;
    values[0] = U_d11[0] + U_d22[0] + U_d33[0]
                + 0.125 * dev_BY_KKofxyz(x, y, z, p) / psi7
                + 2.0 * PI / psi2 / psi * rho_adm;
}

__global__ void J_times_dv_kernel(int nvar, int n1, int n2, int n3,
                                   const gpu_derivs dv, const gpu_derivs u,
                                   double *Jdv, const gpu_params params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n1 * n2 * n3;
    if (idx >= total) return;

    int i = idx / (n2 * n3);
    int j = (idx / n3) % n2;
    int k = idx % n3;

    double al = PIH * (2 * i + 1) / n1;
    double A = -cos(al);
    double be = PIH * (2 * j + 1) / n2;
    double B = -cos(be);
    double phi = 2.0 * PI * k / n3;
    double Am1 = A - 1.0;

    double dU_d0[1], dU_d1[1], dU_d2[1], dU_d3[1];
    double dU_d11[1], dU_d12[1], dU_d13[1];
    double dU_d22[1], dU_d23[1], dU_d33[1];

    int indx = i + n1 * (j + n2 * k);
    dU_d0[0] = Am1 * dv.d0[indx];
    dU_d1[0] = dv.d0[indx] + Am1 * dv.d1[indx];
    dU_d2[0] = Am1 * dv.d2[indx];
    dU_d3[0] = Am1 * dv.d3[indx];
    dU_d11[0] = 2.0 * dv.d1[indx] + Am1 * dv.d11[indx];
    dU_d12[0] = dv.d2[indx] + Am1 * dv.d12[indx];
    dU_d13[0] = dv.d3[indx] + Am1 * dv.d13[indx];
    dU_d22[0] = Am1 * dv.d22[indx];
    dU_d23[0] = Am1 * dv.d23[indx];
    dU_d33[0] = Am1 * dv.d33[indx];

    double U_d0[1], U_d1[1], U_d2[1], U_d3[1];
    double U_d11[1], U_d12[1], U_d13[1];
    double U_d22[1], U_d23[1], U_d33[1];

    U_d0[0] = u.d0[indx];
    U_d1[0] = u.d1[indx];
    U_d2[0] = u.d2[indx];
    U_d3[0] = u.d3[indx];
    U_d11[0] = u.d11[indx];
    U_d12[0] = u.d12[indx];
    U_d13[0] = u.d13[indx];
    U_d22[0] = u.d22[indx];
    U_d23[0] = u.d23[indx];
    U_d33[0] = u.d33[indx];

    double X, R, xx, rr, y, z;
    dev_AB_To_XR(nvar, A, B, &X, &R,
                 dU_d0, dU_d1, dU_d2, dU_d3, dU_d11, dU_d12, dU_d13, dU_d22, dU_d23, dU_d33);
    dev_C_To_c(nvar, X, R, &xx, &rr, params.par_b,
               dU_d0, dU_d1, dU_d2, dU_d3, dU_d11, dU_d12, dU_d13, dU_d22, dU_d23, dU_d33);
    dev_rx3_To_xyz(nvar, xx, rr, phi, &y, &z,
                   dU_d0, dU_d1, dU_d2, dU_d3, dU_d11, dU_d12, dU_d13, dU_d22, dU_d23, dU_d33);

    double values[1];
    dev_LinEquations(A, B, X, R, xx, rr, phi, y, z,
                     dU_d0, dU_d11, dU_d22, dU_d33, U_d0, &params, values);
    double fac = sin(al) * sin(be); fac = fac * fac * fac;
    Jdv[indx] = values[0] * fac;
}

__global__ void F_of_v_kernel(int nvar, int n1, int n2, int n3,
                              const gpu_derivs v, double *F, gpu_derivs u,
                              const double *sources, const gpu_params params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n1 * n2 * n3;
    if (idx >= total) return;

    int i = idx / (n2 * n3);
    int j = (idx / n3) % n2;
    int k = idx % n3;

    double al = PIH * (2 * i + 1) / n1;
    double A = -cos(al);
    double be = PIH * (2 * j + 1) / n2;
    double B = -cos(be);
    double phi = 2.0 * PI * k / n3;
    double Am1 = A - 1.0;

    double U_d0[1], U_d1[1], U_d2[1], U_d3[1];
    double U_d11[1], U_d12[1], U_d13[1];
    double U_d22[1], U_d23[1], U_d33[1];

    int indx = i + n1 * (j + n2 * k);
    U_d0[0] = Am1 * v.d0[indx];
    U_d1[0] = v.d0[indx] + Am1 * v.d1[indx];
    U_d2[0] = Am1 * v.d2[indx];
    U_d3[0] = Am1 * v.d3[indx];
    U_d11[0] = 2.0 * v.d1[indx] + Am1 * v.d11[indx];
    U_d12[0] = v.d2[indx] + Am1 * v.d12[indx];
    U_d13[0] = v.d3[indx] + Am1 * v.d13[indx];
    U_d22[0] = Am1 * v.d22[indx];
    U_d23[0] = Am1 * v.d23[indx];
    U_d33[0] = Am1 * v.d33[indx];

    double X, R, xx, rr, y, z;
    dev_AB_To_XR(nvar, A, B, &X, &R,
                 U_d0, U_d1, U_d2, U_d3, U_d11, U_d12, U_d13, U_d22, U_d23, U_d33);
    dev_C_To_c(nvar, X, R, &xx, &rr, params.par_b,
               U_d0, U_d1, U_d2, U_d3, U_d11, U_d12, U_d13, U_d22, U_d23, U_d33);
    dev_rx3_To_xyz(nvar, xx, rr, phi, &y, &z,
                   U_d0, U_d1, U_d2, U_d3, U_d11, U_d12, U_d13, U_d22, U_d23, U_d33);

    double rho_adm = sources[indx];
    double values[1];
    dev_NonLinEquations(rho_adm, A, B, X, R, xx, rr, phi, y, z,
                        U_d0, U_d11, U_d22, U_d33, &params, values);
    double fac = sin(al) * sin(be); fac = fac * fac * fac;
    F[indx] = values[0] * fac;

    u.d0[indx] = U_d0[0];
    u.d1[indx] = U_d1[0];
    u.d2[indx] = U_d2[0];
    u.d3[indx] = U_d3[0];
    u.d11[indx] = U_d11[0];
    u.d12[indx] = U_d12[0];
    u.d13[indx] = U_d13[0];
    u.d22[indx] = U_d22[0];
    u.d23[indx] = U_d23[0];
    u.d33[indx] = U_d33[0];
}

extern "C" void gpu_J_times_dv(int nvar, int n1, int n2, int n3,
                               const double *h_dv_d0, const double *h_dv_d1,
                               const double *h_dv_d2, const double *h_dv_d3,
                               const double *h_dv_d11, const double *h_dv_d12,
                               const double *h_dv_d13, const double *h_dv_d22,
                               const double *h_dv_d23, const double *h_dv_d33,
                               const double *h_u_d0, const double *h_u_d1,
                               const double *h_u_d2, const double *h_u_d3,
                               const double *h_u_d11, const double *h_u_d12,
                               const double *h_u_d13, const double *h_u_d22,
                               const double *h_u_d23, const double *h_u_d33,
                               double *h_Jdv,
                               double par_b, double par_m_plus, double par_m_minus,
                               const double par_P_plus[3], const double par_P_minus[3],
                               const double par_S_plus[3], const double par_S_minus[3])
{
    int ntotal = n1 * n2 * n3 * nvar;
    gpu_twop_init(nvar, n1, n2, n3);

#ifdef USE_GPU_ASYNC
    upload_derivs_async(g_d_stage_work, g_h_stage_work,
                        h_dv_d0, h_dv_d1, h_dv_d2, h_dv_d3,
                        h_dv_d11, h_dv_d12, h_dv_d13,
                        h_dv_d22, h_dv_d23, h_dv_d33, ntotal);
    if (!g_u_valid) {
        upload_derivs_async(g_d_stage_u, g_h_stage_u,
                            h_u_d0, h_u_d1, h_u_d2, h_u_d3,
                            h_u_d11, h_u_d12, h_u_d13,
                            h_u_d22, h_u_d23, h_u_d33, ntotal);
        g_u_valid = true;
    }
    set_params(&g_params_cache, par_b, par_m_plus, par_m_minus,
               par_P_plus, par_P_minus, par_S_plus, par_S_minus);
    int total = n1 * n2 * n3;
    int block = 256;
    int grid = (total + block - 1) / block;
    J_times_dv_kernel<<<grid, block, 0, g_stream>>>(nvar, n1, n2, n3, g_d_work, g_d_u, g_d_Jdv, g_params_cache);
    cudaMemcpyAsync(h_Jdv, g_d_Jdv, ntotal * sizeof(double),
                    cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
#else
    upload_derivs(&g_d_work, h_dv_d0, h_dv_d1, h_dv_d2, h_dv_d3,
                  h_dv_d11, h_dv_d12, h_dv_d13,
                  h_dv_d22, h_dv_d23, h_dv_d33, ntotal);

    if (!g_u_valid) {
        upload_derivs(&g_d_u, h_u_d0, h_u_d1, h_u_d2, h_u_d3,
                      h_u_d11, h_u_d12, h_u_d13,
                      h_u_d22, h_u_d23, h_u_d33, ntotal);
        g_u_valid = true;
    }

    set_params(&g_params_cache, par_b, par_m_plus, par_m_minus,
               par_P_plus, par_P_minus, par_S_plus, par_S_minus);

    int total = n1 * n2 * n3;
    int block = 256;
    int grid = (total + block - 1) / block;
    J_times_dv_kernel<<<grid, block>>>(nvar, n1, n2, n3, g_d_work, g_d_u, g_d_Jdv, g_params_cache);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Jdv, g_d_Jdv, ntotal * sizeof(double), cudaMemcpyDeviceToHost);
#endif
}

extern "C" void gpu_F_of_v(int nvar, int n1, int n2, int n3,
                            const double *h_v_d0, const double *h_v_d1,
                            const double *h_v_d2, const double *h_v_d3,
                            const double *h_v_d11, const double *h_v_d12,
                            const double *h_v_d13, const double *h_v_d22,
                            const double *h_v_d23, const double *h_v_d33,
                            double *h_F,
                            double *h_u_d0, double *h_u_d1,
                            double *h_u_d2, double *h_u_d3,
                            double *h_u_d11, double *h_u_d12,
                            double *h_u_d13, double *h_u_d22,
                            double *h_u_d23, double *h_u_d33,
                            double par_b, double par_m_plus, double par_m_minus,
                            const double par_P_plus[3], const double par_P_minus[3],
                            const double par_S_plus[3], const double par_S_minus[3])
{
    int ntotal = n1 * n2 * n3 * nvar;
    gpu_twop_init(nvar, n1, n2, n3);

#ifdef USE_GPU_ASYNC
    upload_derivs_async(g_d_stage_work, g_h_stage_work,
                        h_v_d0, h_v_d1, h_v_d2, h_v_d3,
                        h_v_d11, h_v_d12, h_v_d13,
                        h_v_d22, h_v_d23, h_v_d33, ntotal);
    set_params(&g_params_cache, par_b, par_m_plus, par_m_minus,
               par_P_plus, par_P_minus, par_S_plus, par_S_minus);
    int total = n1 * n2 * n3;
    int block = 256;
    int grid = (total + block - 1) / block;
    F_of_v_kernel<<<grid, block, 0, g_stream>>>(nvar, n1, n2, n3, g_d_work, g_d_F, g_d_u, g_d_sources, g_params_cache);
    cudaMemcpyAsync(h_F, g_d_F, ntotal * sizeof(double),
                    cudaMemcpyDeviceToHost, g_stream);
    download_derivs_async(g_d_stage_u, g_h_stage_u,
                          h_u_d0, h_u_d1, h_u_d2, h_u_d3,
                          h_u_d11, h_u_d12, h_u_d13,
                          h_u_d22, h_u_d23, h_u_d33, ntotal);
    g_u_valid = true;
#else
    upload_derivs(&g_d_work, h_v_d0, h_v_d1, h_v_d2, h_v_d3,
                  h_v_d11, h_v_d12, h_v_d13,
                  h_v_d22, h_v_d23, h_v_d33, ntotal);

    set_params(&g_params_cache, par_b, par_m_plus, par_m_minus,
               par_P_plus, par_P_minus, par_S_plus, par_S_minus);

    int total = n1 * n2 * n3;
    int block = 256;
    int grid = (total + block - 1) / block;
    F_of_v_kernel<<<grid, block>>>(nvar, n1, n2, n3, g_d_work, g_d_F, g_d_u, g_d_sources, g_params_cache);
    cudaDeviceSynchronize();

    cudaMemcpy(h_F, g_d_F, ntotal * sizeof(double), cudaMemcpyDeviceToHost);
    download_derivs(h_u_d0, h_u_d1, h_u_d2, h_u_d3,
                    h_u_d11, h_u_d12, h_u_d13,
                    h_u_d22, h_u_d23, h_u_d33, &g_d_u, ntotal);
    g_u_valid = true;
#endif
}
