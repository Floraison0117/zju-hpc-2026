#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "kblas.h"

/* Compute-ceiling test: 38 threads, EACH with a private high-intensity GEMM.
 * If aggregate approaches 38*131 = 5000 GFLOPS, compute ceiling is reachable
 * when data layout permits. Usage: priv_gemm <N> <reps> */
int main(int argc, char** argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 2048;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    BlasSetNumThreads(1);

    /* each thread gets its own N x N x N GEMM on private buffers */
    size_t per = (size_t)n * n * 3;
    double* pool = malloc(per * 8);
    if (!pool) { printf("malloc fail (%.1fGB)\n", per * 8 / 1e9); return 1; }
#pragma omp parallel for
    for (size_t i = 0; i < per; i++) pool[i] = 0.5;

    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();
#pragma omp parallel
        {
            int t = omp_get_thread_num();
            double* A = pool + (size_t)t * n * n;
            double* B = A + (size_t)n * n;
            double* C = B + (size_t)n * n;
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        n, n, n, 1.0, A, n, B, n, 0.0, C, n);
        }
        double t = omp_get_wtime() - t0;
        if (t < best) best = t;
        printf("  rep%d: %.4f s  aggregate %.1f GFLOPS\n", r, t, 2.0 * n * n * n * 38 / t / 1e9);
    }
    printf("BEST aggregate: %.1f GFLOPS (per-thread %.1f)\n",
           2.0 * n * n * n * 38 / best / 1e9, 2.0 * n * n * n / best / 1e9);
    volatile double s = pool[0];
    printf("sink=%f\n", s);
    free(pool);
    return 0;
}
