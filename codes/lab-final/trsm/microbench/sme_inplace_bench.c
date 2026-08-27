#include "../sme_update.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double median5(double x[5])
{
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            if (x[j] < x[i]) {
                const double t = x[i];
                x[i] = x[j];
                x[j] = t;
            }
        }
    }
    return x[2];
}

static void fill(double *x, size_t count, int salt)
{
    for (size_t i = 0; i < count; ++i)
        x[i] = 0.001 * (double)(1 + (int)((i * 17u + (size_t)salt) % 97u));
}

int main(int argc, char **argv)
{
    const int m = argc > 1 ? atoi(argv[1]) : 16800;
    const int n = argc > 2 ? atoi(argv[2]) : 512;
    const int k = argc > 3 ? atoi(argv[3]) : 224;
    if (m <= 0 || n <= 0 || k <= 0 || (m & 15) != 0 || (n & 31) != 0) {
        fprintf(stderr, "m and n must be positive tile multiples\n");
        return 2;
    }

    const int mt = m / 16;
    const int nt = n / 32;
    const size_t a_count = (size_t)mt * (size_t)k * 16u;
    const size_t b_count = (size_t)nt * (size_t)k * 32u;
    const size_t c_count = (size_t)m * (size_t)n;
    double *a = aligned_alloc(64, a_count * sizeof(double));
    double *b = aligned_alloc(64, b_count * sizeof(double));
    double *c = aligned_alloc(64, c_count * sizeof(double));
    if (a == NULL || b == NULL || c == NULL) {
        free(a); free(b); free(c);
        return 3;
    }
    fill(a, a_count, 11);
    fill(b, b_count, 29);

    double trials[5];
    for (int trial = 0; trial < 5; ++trial) {
        memset(c, 0, c_count * sizeof(double));
        const double start = now_sec();
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int threads = omp_get_num_threads();
            const int per = mt / threads;
            const int rem = mt % threads;
            const int first = tid * per + (tid < rem ? tid : rem);
            const int local_mt = per + (tid < rem ? 1 : 0);
            if (local_mt > 0) {
                sme_update_16x32_grid_batch(
                    a + (size_t)first * (size_t)k * 16u, b,
                    c + (size_t)first * 16u * (size_t)n,
                    k, n * (int)sizeof(double), nt, local_mt);
            }
        }
        trials[trial] = now_sec() - start;
    }
    const double elapsed = median5(trials);
    const double flops = 2.0 * (double)m * (double)n * (double)k;
    printf("m=%d n=%d k=%d mt=%d nt=%d threads=%d\n",
           m, n, k, mt, nt, omp_get_max_threads());
    printf("inplace_median_ms=%.6f inplace_gflops=%.3f\n",
           elapsed * 1e3, flops / elapsed / 1e9);
    free(a); free(b); free(c);
    return 0;
}
