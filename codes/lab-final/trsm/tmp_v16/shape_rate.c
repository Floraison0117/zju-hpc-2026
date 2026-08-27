#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "kblas.h"

/* In-situ per-thread KBLAS serial rate for the shapes used by the
 * recursive/blocked TRSM updates.  Outer 38 libgomp threads, each calls
 * cblas_dgemm on its own C slice (M-split), or on its own B slice (N-split).
 * Usage: shape_rate <M> <N> <K> <reps> <mode>  (mode m = M-split, n = N-split) */

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv)
{
    if (argc < 6) { printf("usage: shape_rate M N K reps mode\n"); return 1; }
    int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
    int reps = atoi(argv[4]);
    char mode = argv[5][0];

    BlasSetNumThreads(1);

    double* A = malloc((size_t)m * k * 8);
    double* B = malloc((size_t)k * n * 8);
    double* C = malloc((size_t)m * n * 8);
    for (size_t i = 0; i < (size_t)m * k; i++) A[i] = 0.5;
    for (size_t i = 0; i < (size_t)k * n; i++) B[i] = 0.25;
    for (size_t i = 0; i < (size_t)m * n; i++) C[i] = 0.0;

    printf("shape_rate M=%d N=%d K=%d mode=%c reps=%d OMP_NUM_THREADS=%s\n",
           m, n, k, mode, reps, getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "?");

    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        if (mode == 'm') {
#pragma omp parallel
            {
                int t = omp_get_thread_num(), nt = omp_get_num_threads();
                int per = m / nt, rem = m % nt;
                int start = t * per + (t < rem ? t : rem);
                int len = per + (t < rem ? 1 : 0);
                if (len > 0)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                len, n, k, -1.0,
                                A + (size_t)start * k, k, B, n,
                                1.0, C + (size_t)start * n, n);
            }
        } else {
#pragma omp parallel
            {
                int t = omp_get_thread_num(), nt = omp_get_num_threads();
                int per = n / nt, rem = n % nt;
                int start = t * per + (t < rem ? t : rem);
                int len = per + (t < rem ? 1 : 0);
                if (len > 0)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                m, len, k, -1.0, A, k,
                                B + (size_t)start, n,
                                1.0, C + (size_t)start, n);
            }
        }
        double t = now_s() - t0;
        double fl = 2.0 * m * n * k;
        printf("  rep%d: %.4f ms aggregate %.1f GFLOPS\n", r, t * 1e3, fl / t / 1e9);
    }
    free(A); free(B); free(C);
    return 0;
}
