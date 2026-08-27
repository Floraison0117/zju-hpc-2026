#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void sme_gemm_8x8(const double* a, const double* b, double* c,
                         long k, long ldc_bytes);

static double max_error(const double* a, const double* b, int n)
{
    double emax = 0.0;
    for (int i = 0; i < n; ++i) emax = fmax(emax, fabs(a[i] - b[i]));
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
    const int threads = argc > 1 ? atoi(argv[1]) : 38;
    const long k = argc > 2 ? atol(argv[2]) : 256;
    const int reps = argc > 3 ? atoi(argv[3]) : 10000;
    const int trials = argc > 4 ? atoi(argv[4]) : 5;
    double* a = aligned_alloc(64, (size_t)k * 8 * sizeof(double));
    double* b = aligned_alloc(64, (size_t)k * 8 * sizeof(double));
    double* c = aligned_alloc(64, (size_t)threads * 64 * sizeof(double));
    double* ref = calloc(64, sizeof(double));
    double* times = calloc((size_t)trials, sizeof(double));
    if (!a || !b || !c || !ref || !times) return 2;
    for (long i = 0; i < k * 8; ++i) {
        a[i] = ((double)((i * 17) % 31) / 31.0) - 0.5;
        b[i] = ((double)((i * 13) % 29) / 29.0) - 0.5;
    }
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            for (long q = 0; q < k; ++q)
                ref[i * 8 + j] += a[q * 8 + i] * b[q * 8 + j];
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    memset(c, 0, (size_t)threads * 64 * sizeof(double));
#pragma omp parallel num_threads(threads)
    sme_gemm_8x8(a, b, c + (size_t)omp_get_thread_num() * 64, k, 8 * sizeof(double));
    const double error = max_error(c, ref, 64);
    printf("debug_c");
    for (int i = 0; i < 16; ++i) printf(" %.6f", c[i]);
    printf("\ndebug_ref");
    for (int i = 0; i < 16; ++i) printf(" %.6f", ref[i]);
    putchar('\n');
    for (int t = 0; t < trials; ++t) {
        memset(c, 0, (size_t)threads * 64 * sizeof(double));
        const double start = omp_get_wtime();
#pragma omp parallel num_threads(threads)
        {
            for (int r = 0; r < reps; ++r)
                sme_gemm_8x8(a, b, c + (size_t)omp_get_thread_num() * 64,
                             k, 8 * sizeof(double));
        }
        times[t] = omp_get_wtime() - start;
    }
    qsort(times, (size_t)trials, sizeof(*times), cmp_double);
    const double flops = (double)threads * reps * 128.0 * k;
    printf("path=sme-gemm-8x8-packed threads=%d k=%ld reps=%d trials=%d\n",
           threads, k, reps, trials);
    printf("max_error=%.17g median_ms=%.6f gflops=%.6f\n",
           error, times[trials / 2] * 1e3,
           flops / times[trials / 2] / 1e9);
    printf("trial_ms");
    for (int t = 0; t < trials; ++t) printf(" %.6f", times[t] * 1e3);
    putchar('\n');
    free(a); free(b); free(c); free(ref); free(times);
    return error <= 1e-12 ? 0 : 1;
}
