#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kblas.h"

static double* alloc64(size_t bytes)
{
    void* p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return (double*)p;
}

static void fill_matrix(double* x, size_t count, unsigned seed)
{
    unsigned state = seed;
    for (size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        x[i] = ((double)(state >> 8) / 16777216.0) - 0.5;
    }
}

static void standard_msplit(int m, int n, int k, const double* a,
                            const double* b, double* c, int threads)
{
#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const int base = m / nt;
        const int rem = m % nt;
        const int start = tid * base + (tid < rem ? tid : rem);
        const int rows = base + (tid < rem ? 1 : 0);
        if (rows > 0) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        rows, n, k, -1.0,
                        a + (size_t)start * k, k, b, n, 1.0,
                        c + (size_t)start * n, n);
        }
    }
}

static void packed_b_msplit(int m, int n, int k, const double* a,
                            const double* b, double* c, int threads,
                            double** packed_a, double* packed_b)
{
    cblas_dgemm_pack(CblasRowMajor, CblasB, CblasNoTrans,
                     m, n, k, b, n, packed_b);
#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const int base = m / nt;
        const int rem = m % nt;
        const int start = tid * base + (tid < rem ? tid : rem);
        const int rows = base + (tid < rem ? 1 : 0);
        if (rows > 0) {
            cblas_dgemm_pack(CblasRowMajor, CblasA, CblasNoTrans,
                             rows, n, k, a + (size_t)start * k, k,
                             packed_a[tid]);
            cblas_dgemm_compute(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                rows, n, k, -1.0, packed_a[tid], k,
                                packed_b, n, 1.0,
                                c + (size_t)start * n, n);
        }
    }
}

static void packed_compute_msplit(int m, int n, int k, double* c, int threads,
                                  double** packed_a, double* packed_b)
{
#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const int base = m / nt;
        const int rem = m % nt;
        const int start = tid * base + (tid < rem ? tid : rem);
        const int rows = base + (tid < rem ? 1 : 0);
        if (rows > 0) {
            cblas_dgemm_compute(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                rows, n, k, -1.0, packed_a[tid], k,
                                packed_b, n, 1.0,
                                c + (size_t)start * n, n);
        }
    }
}

static double max_diff(int m, int n, const double* x, const double* y)
{
    double emax = 0.0;
    for (size_t i = 0; i < (size_t)m * n; ++i)
        if (fabs(x[i] - y[i]) > emax) emax = fabs(x[i] - y[i]);
    return emax;
}

static int cmp_double(const void* lhs, const void* rhs)
{
    const double a = *(const double*)lhs;
    const double b = *(const double*)rhs;
    return (a > b) - (a < b);
}

int main(int argc, char** argv)
{
    const int m = argc > 1 ? atoi(argv[1]) : 16800;
    const int n = argc > 2 ? atoi(argv[2]) : 512;
    const int k = argc > 3 ? atoi(argv[3]) : 224;
    const int threads = argc > 4 ? atoi(argv[4]) : 38;
    const int trials = argc > 5 ? atoi(argv[5]) : 5;
    const size_t a_count = (size_t)m * k;
    const size_t b_count = (size_t)k * n;
    const size_t c_count = (size_t)m * n;
    const size_t pb_bytes = cblas_dgemm_pack_get_size(CblasB, m, n, k);
    const int max_rows = (m + threads - 1) / threads;
    const size_t pa_bytes = cblas_dgemm_pack_get_size(CblasA, max_rows, n, k);
    double* a = alloc64(a_count * sizeof(double));
    double* b = alloc64(b_count * sizeof(double));
    double* c_standard = alloc64(c_count * sizeof(double));
    double* c_packed = alloc64(c_count * sizeof(double));
    double* packed_b = alloc64(pb_bytes);
    double** packed_a = calloc((size_t)threads, sizeof(*packed_a));
    double* t_standard = calloc((size_t)trials, sizeof(*t_standard));
    double* t_packed = calloc((size_t)trials, sizeof(*t_packed));
    double* t_compute = calloc((size_t)trials, sizeof(*t_compute));
    if (!a || !b || !c_standard || !c_packed || !packed_b || !packed_a ||
        !t_standard || !t_packed || !t_compute) return 2;
    for (int t = 0; t < threads; ++t)
        packed_a[t] = alloc64(pa_bytes);
    fill_matrix(a, a_count, 17u);
    fill_matrix(b, b_count, 29u);
    memset(c_standard, 0, c_count * sizeof(double));
    memset(c_packed, 0, c_count * sizeof(double));
    BlasSetNumThreads(1);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    standard_msplit(m, n, k, a, b, c_standard, threads);
    packed_b_msplit(m, n, k, a, b, c_packed, threads, packed_a, packed_b);
    const double initial_error = max_diff(m, n, c_standard, c_packed);
    for (int t = 0; t < trials; ++t) {
        memset(c_packed, 0, c_count * sizeof(double));
        double start = omp_get_wtime();
        packed_compute_msplit(m, n, k, c_packed, threads, packed_a, packed_b);
        t_compute[t] = omp_get_wtime() - start;
    }
    for (int t = 0; t < trials; ++t) {
        memset(c_standard, 0, c_count * sizeof(double));
        double start = omp_get_wtime();
        standard_msplit(m, n, k, a, b, c_standard, threads);
        t_standard[t] = omp_get_wtime() - start;
        memset(c_packed, 0, c_count * sizeof(double));
        start = omp_get_wtime();
        packed_b_msplit(m, n, k, a, b, c_packed, threads, packed_a, packed_b);
        t_packed[t] = omp_get_wtime() - start;
    }
    qsort(t_standard, (size_t)trials, sizeof(*t_standard), cmp_double);
    qsort(t_packed, (size_t)trials, sizeof(*t_packed), cmp_double);
    qsort(t_compute, (size_t)trials, sizeof(*t_compute), cmp_double);
    const double flops = 2.0 * (double)m * n * k;
    printf("m=%d n=%d k=%d threads=%d trials=%d pack_a_bytes=%zu pack_b_bytes=%zu\n",
           m, n, k, threads, trials, pa_bytes * (size_t)threads, pb_bytes);
    printf("standard_msplit_median_ms=%.6f gflops=%.6f\n",
           t_standard[trials / 2] * 1e3,
           flops / t_standard[trials / 2] / 1e9);
    printf("shared_b_pack_e2e_median_ms=%.6f gflops=%.6f\n",
           t_packed[trials / 2] * 1e3,
           flops / t_packed[trials / 2] / 1e9);
    printf("packed_compute_only_median_ms=%.6f gflops=%.6f\n",
           t_compute[trials / 2] * 1e3,
           flops / t_compute[trials / 2] / 1e9);
    printf("initial_max_abs_diff=%.17g final_max_abs_diff=%.17g\n",
           initial_error, max_diff(m, n, c_standard, c_packed));
    free(t_standard); free(t_packed); free(t_compute); free(packed_b); free(c_standard);
    free(c_packed); free(a); free(b);
    for (int t = 0; t < threads; ++t) free(packed_a[t]);
    free(packed_a);
    return initial_error <= 1e-12 ? 0 : 1;
}
