#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void sme_gemm_16x32(const double *a, const double *b, double *c,
                           int k, int ldc_bytes);
extern void sme_gemm_16x32_batch(const double *a, const double *b, double *c,
                                 int k, int ldc_bytes, int nt);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double median5(double values[5]) {
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            if (values[j] < values[i]) {
                double t = values[i];
                values[i] = values[j];
                values[j] = t;
            }
    return values[2];
}

static void fill_matrix(double *x, int rows, int cols, int salt) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            x[(size_t)i * cols + j] =
                0.001 * (double)(1 + ((i * (salt + 7) + j * (salt + 3)) % 97));
}

static void pack_a_tile(const double *a, int lda, int row, int k, double *pa) {
    for (int p = 0; p < k; ++p)
        for (int i = 0; i < 16; ++i)
            pa[(size_t)p * 16 + i] = a[(size_t)(row + i) * lda + p];
}

static void pack_b_tile(const double *b, int ldb, int col, int k, double *pb) {
    for (int p = 0; p < k; ++p)
        for (int j = 0; j < 32; ++j)
            pb[(size_t)p * 32 + j] = b[(size_t)p * ldb + col + j];
}

static double check_first_tile(const double *a, int lda, const double *b, int ldb,
                               const double *c, int ldc, int k) {
    double result = 0.0;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 32; ++j) {
            double ref = 0.0;
            for (int p = 0; p < k; ++p)
                ref += a[(size_t)i * lda + p] * b[(size_t)p * ldb + j];
            double e = fabs(ref - c[(size_t)i * ldc + j]);
            if (e > result) result = e;
        }
    }
    return result;
}

int main(int argc, char **argv) {
    const int m = argc > 1 ? atoi(argv[1]) : 16800;
    const int n = argc > 2 ? atoi(argv[2]) : 512;
    const int k = argc > 3 ? atoi(argv[3]) : 224;
    const int reps = argc > 4 ? atoi(argv[4]) : 1;
    if (m <= 0 || n <= 0 || k <= 0 || m % 16 != 0 || n % 32 != 0) {
        fprintf(stderr, "m and n must be positive tile multiples\n");
        return 2;
    }

    const int mt = m / 16;
    const int nt = n / 32;
    const int threads = omp_get_max_threads();
    double *a = aligned_alloc(64, (size_t)m * k * sizeof(double));
    double *b = aligned_alloc(64, (size_t)k * n * sizeof(double));
    double *c = aligned_alloc(64, (size_t)m * n * sizeof(double));
    double *pa = aligned_alloc(64, (size_t)mt * k * 16 * sizeof(double));
    double *pb = aligned_alloc(64, (size_t)nt * k * 32 * sizeof(double));
    double *scratch = aligned_alloc(64, (size_t)threads * k * 16 * sizeof(double));
    if (!a || !b || !c || !pa || !pb || !scratch) {
        fprintf(stderr, "allocation failed\n");
        return 3;
    }
    fill_matrix(a, m, k, 11);
    fill_matrix(b, k, n, 19);

    for (int ti = 0; ti < mt; ++ti) pack_a_tile(a, k, ti * 16, k, pa + (size_t)ti * k * 16);
    for (int tj = 0; tj < nt; ++tj) pack_b_tile(b, n, tj * 32, k, pb + (size_t)tj * k * 32);

    memset(c, 0, (size_t)m * n * sizeof(double));
    sme_gemm_16x32_batch(pa, pb, c, k, n * (int)sizeof(double), nt);
    double error = check_first_tile(a, k, b, n, c, n, k);
    printf("m=%d n=%d k=%d reps=%d max_error_sample=%.17g threads=%d\n",
           m, n, k, reps, error, threads);
    if (error > 1e-12) {
        for (int i = 0; i < 32; ++i) {
            double row_error = 0.0;
            int col = 0;
            for (int j = 0; j < 32; ++j) {
                double ref = 0.0;
                for (int p = 0; p < k; ++p)
                    ref += a[(size_t)i * k + p] * b[(size_t)p * n + j];
                double e = fabs(c[(size_t)i * n + j] - ref);
                if (e > row_error) { row_error = e; col = j; }
            }
            printf("row%d maxerr=%.6g col=%d got=%.6g\n", i, row_error, col,
                   c[(size_t)i * n + col]);
        }
        printf("row0 got/ref:");
        for (int j = 0; j < 8; ++j) {
            double ref = 0.0;
            for (int p = 0; p < k; ++p)
                ref += a[p] * b[(size_t)p * n + j];
            printf(" %.6g/%.6g", c[j], ref);
        }
        printf("\n");
        return 4;
    }

    double kernel_trials[5];
    for (int trial = 0; trial < 5; ++trial) {
        memset(c, 0, (size_t)m * n * sizeof(double));
        double start = now_sec();
#pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < reps; ++r) {
            for (int ti = 0; ti < mt; ++ti) {
                sme_gemm_16x32_batch(pa + (size_t)ti * k * 16,
                                     pb,
                                     c + (size_t)(ti * 16) * n,
                                     k, n * (int)sizeof(double), nt);
            }
        }
        kernel_trials[trial] = now_sec() - start;
    }
    double kernel_s = median5(kernel_trials);

    double e2e_trials[5];
    for (int trial = 0; trial < 5; ++trial) {
        memset(c, 0, (size_t)m * n * sizeof(double));
        double start = now_sec();
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double *local_pa = scratch + (size_t)tid * k * 16;
#pragma omp single
            {
                for (int tj = 0; tj < nt; ++tj)
                    pack_b_tile(b, n, tj * 32, k, pb + (size_t)tj * k * 32);
            }
#pragma omp barrier
#pragma omp for collapse(2) schedule(static)
            for (int r = 0; r < reps; ++r) {
                for (int ti = 0; ti < mt; ++ti) {
                    pack_a_tile(a, k, ti * 16, k, local_pa);
                    sme_gemm_16x32_batch(local_pa, pb,
                                         c + (size_t)(ti * 16) * n,
                                         k, n * (int)sizeof(double), nt);
                }
            }
        }
        e2e_trials[trial] = now_sec() - start;
    }
    double e2e_s = median5(e2e_trials);
    double flops = (double)reps * 2.0 * (double)m * n * k;
    printf("kernel_only_ms=%.6f kernel_only_gflops=%.3f\n",
           kernel_s * 1e3, flops / kernel_s / 1e9);
    printf("shared_b_pack_e2e_ms=%.6f shared_b_pack_e2e_gflops=%.3f\n",
           e2e_s * 1e3, flops / e2e_s / 1e9);
    free(a); free(b); free(c); free(pa); free(pb); free(scratch);
    return 0;
}
