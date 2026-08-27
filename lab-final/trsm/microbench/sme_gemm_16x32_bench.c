#include <arm_sve.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void sme_gemm_16x32(const double *a, const double *b, double *c,
                           int k, int ldc_bytes);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_inputs(double *a, double *b, int k) {
    for (int p = 0; p < k; ++p) {
        for (int i = 0; i < 16; ++i)
            a[p * 16 + i] = 0.001 * (double)(1 + ((p * 17 + i * 3) % 97));
        for (int j = 0; j < 32; ++j)
            b[p * 32 + j] = -0.002 * (double)(1 + ((p * 11 + j * 5) % 89));
    }
}

static void reference(const double *a, const double *b, double *c, int k) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 32; ++j) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p)
                sum += a[p * 16 + i] * b[p * 32 + j];
            c[i * 32 + j] = sum;
        }
    }
}

static double max_abs_diff(const double *x, const double *y) {
    double result = 0.0;
    for (int i = 0; i < 16 * 32; ++i) {
        double error = fabs(x[i] - y[i]);
        if (error > result) result = error;
    }
    return result;
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

int main(int argc, char **argv) {
    int k = argc > 1 ? atoi(argv[1]) : 256;
    int reps = argc > 2 ? atoi(argv[2]) : 5000;
    if (k <= 0 || reps <= 0) {
        fprintf(stderr, "usage: %s K REPS\n", argv[0]);
        return 2;
    }

    double *a = aligned_alloc(64, (size_t)k * 16 * sizeof(double));
    double *b = aligned_alloc(64, (size_t)k * 32 * sizeof(double));
    double *reference_c = aligned_alloc(64, 16 * 32 * sizeof(double));
    double *thread_c = aligned_alloc(64, (size_t)omp_get_max_threads() * 16 * 32 * sizeof(double));
    if (!a || !b || !reference_c || !thread_c) {
        fprintf(stderr, "allocation failed\n");
        return 3;
    }
    fill_inputs(a, b, k);
    reference(a, b, reference_c, k);
    memset(thread_c, 0, (size_t)omp_get_max_threads() * 16 * 32 * sizeof(double));

    sme_gemm_16x32(a, b, thread_c, k, 32 * (int)sizeof(double));
    double error = max_abs_diff(thread_c, reference_c);
    printf("k=%d reps=%d max_error=%.17g svcntd=%zu threads=%d\n",
           k, reps, error, svcntd(), omp_get_max_threads());
    if (error > 1e-12) {
        fprintf(stderr, "correctness failure\n");
        return 4;
    }

    int threads = omp_get_max_threads();
    double trials[5];
    for (int trial = 0; trial < 5; ++trial) {
        double start = now_sec();
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double *c = thread_c + (size_t)tid * 16 * 32;
            for (int r = 0; r < reps; ++r)
                sme_gemm_16x32(a, b, c, k, 32 * (int)sizeof(double));
        }
        trials[trial] = now_sec() - start;
    }
    double elapsed = median5(trials);
    double flops = (double)threads * (double)reps * 2.0 * 16.0 * 32.0 * (double)k;
    printf("median_ms=%.6f gflops=%.3f total_flops=%.0f\n",
           elapsed * 1e3, flops / elapsed / 1e9, flops);
    free(a);
    free(b);
    free(reference_c);
    free(thread_c);
    return 0;
}
