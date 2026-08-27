#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "kblas.h"

static double* alloc64(size_t count)
{
    void* p = NULL;
    if (posix_memalign(&p, 64, count * sizeof(double)) != 0) return NULL;
    return p;
}

int main(int argc, char** argv)
{
    const int m = argc > 1 ? atoi(argv[1]) : 64;
    const int n = argc > 2 ? atoi(argv[2]) : 96;
    const int k = argc > 3 ? atoi(argv[3]) : 32;
    BlasSetNumThreads(1);
    const size_t as = cblas_dgemm_pack_get_size(CblasA, m, n, k);
    const size_t bs = cblas_dgemm_pack_get_size(CblasB, m, n, k);
    double* a = alloc64((size_t)m * k);
    double* b = alloc64((size_t)k * n);
    double* c = alloc64((size_t)m * n);
    double* c_raw_a = alloc64((size_t)m * n);
    double* ref = alloc64((size_t)m * n);
    double* pa = alloc64((as + sizeof(double) - 1) / sizeof(double));
    double* pb = alloc64((bs + sizeof(double) - 1) / sizeof(double));
    if (!a || !b || !c || !c_raw_a || !ref || !pa || !pb) return 2;
    for (int i = 0; i < m * k; ++i) a[i] = (double)((i * 17) % 31) / 31.0 - 0.5;
    for (int i = 0; i < k * n; ++i) b[i] = (double)((i * 13) % 29) / 29.0 - 0.5;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, a, k, b, n, 0.0, ref, n);
    cblas_dgemm_pack(CblasRowMajor, CblasA, CblasNoTrans, m, n, k, a, k, pa);
    cblas_dgemm_pack(CblasRowMajor, CblasB, CblasNoTrans, m, n, k, b, n, pb);
    cblas_dgemm_compute(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0, pa, k, pb, n, 0.0, c, n);
    cblas_dgemm_compute(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0, a, k, pb, n, 0.0, c_raw_a, n);
    double maxerr = 0.0;
    double maxerr_raw_a = 0.0;
    for (int i = 0; i < m * n; ++i) {
        const double e = fabs(c[i] - ref[i]);
        if (e > maxerr) maxerr = e;
        const double e_raw_a = fabs(c_raw_a[i] - ref[i]);
        if (e_raw_a > maxerr_raw_a) maxerr_raw_a = e_raw_a;
    }
    printf("m=%d n=%d k=%d pack_a_bytes=%zu pack_b_bytes=%zu "
           "maxerr_both=%.17g maxerr_raw_a=%.17g\n",
           m, n, k, as, bs, maxerr, maxerr_raw_a);
    free(a); free(b); free(c); free(c_raw_a); free(ref); free(pa); free(pb);
    return (maxerr <= 1e-12 && maxerr_raw_a <= 1e-12) ? 0 : 1;
}
