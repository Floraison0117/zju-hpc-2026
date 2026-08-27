#define _GNU_SOURCE
#include <arm_sve.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    double cpu_ms;
    double sink;
} thread_result;

static double thread_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static inline svfloat64_t do_fmla(svbool_t pg, svfloat64_t acc,
                                  svfloat64_t a, svfloat64_t b)
{
    return svmla_f64_x(pg, acc, a, b);
}

#ifdef ROOF_FMMLA
static inline svfloat64_t do_fmmla(svfloat64_t acc, svfloat64_t a,
                                   svfloat64_t b)
{
    return svmmla_f64(acc, a, b);
}
#endif

static double run_once(int threads, long reps, thread_result* results)
{
    const svbool_t pg = svptrue_b64();
    const svfloat64_t a = svdup_f64(1.0000001192092896);
    const svfloat64_t b = svdup_f64(0.9999998807907104);
    const double start = omp_get_wtime();

#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const double cpu_start = thread_seconds();
        svfloat64_t c0 = svdup_f64(0.01 + tid * 1e-5);
        svfloat64_t c1 = svdup_f64(0.02 + tid * 1e-5);
        svfloat64_t c2 = svdup_f64(0.03 + tid * 1e-5);
        svfloat64_t c3 = svdup_f64(0.04 + tid * 1e-5);
        svfloat64_t c4 = svdup_f64(0.05 + tid * 1e-5);
        svfloat64_t c5 = svdup_f64(0.06 + tid * 1e-5);
        svfloat64_t c6 = svdup_f64(0.07 + tid * 1e-5);
        svfloat64_t c7 = svdup_f64(0.08 + tid * 1e-5);

        for (long r = 0; r < reps; ++r) {
#ifdef ROOF_FMMLA
            c0 = do_fmmla(c0, a, b);
            c1 = do_fmmla(c1, a, b);
            c2 = do_fmmla(c2, a, b);
            c3 = do_fmmla(c3, a, b);
            c4 = do_fmmla(c4, a, b);
            c5 = do_fmmla(c5, a, b);
            c6 = do_fmmla(c6, a, b);
            c7 = do_fmmla(c7, a, b);
#else
            c0 = do_fmla(pg, c0, a, b);
            c1 = do_fmla(pg, c1, a, b);
            c2 = do_fmla(pg, c2, a, b);
            c3 = do_fmla(pg, c3, a, b);
            c4 = do_fmla(pg, c4, a, b);
            c5 = do_fmla(pg, c5, a, b);
            c6 = do_fmla(pg, c6, a, b);
            c7 = do_fmla(pg, c7, a, b);
#endif
        }

        const svfloat64_t s0 = svadd_x(pg, c0, c1);
        const svfloat64_t s1 = svadd_x(pg, c2, c3);
        const svfloat64_t s2 = svadd_x(pg, c4, c5);
        const svfloat64_t s3 = svadd_x(pg, c6, c7);
        const svfloat64_t s4 = svadd_x(pg, s0, s1);
        const svfloat64_t s5 = svadd_x(pg, s2, s3);
        const svfloat64_t sum = svadd_x(pg, s4, s5);
        results[tid].sink = svlastb_f64(pg, sum);
        results[tid].cpu_ms = (thread_seconds() - cpu_start) * 1e3;
    }
    return (omp_get_wtime() - start) * 1e3;
}

static int compare_double(const void* lhs, const void* rhs)
{
    const double a = *(const double*)lhs;
    const double b = *(const double*)rhs;
    return (a > b) - (a < b);
}

int main(int argc, char** argv)
{
    const int threads = argc > 1 ? atoi(argv[1]) : 38;
    const long reps = argc > 2 ? atol(argv[2]) : 1000000;
    const int trials = argc > 3 ? atoi(argv[3]) : 5;
    if (threads < 1 || threads > 256 || reps < 1 || trials < 1 || trials > 100) {
        fprintf(stderr, "usage: %s THREADS REPS TRIALS\n", argv[0]);
        return 2;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    thread_result* results = calloc((size_t)threads, sizeof(*results));
    double* times = calloc((size_t)trials, sizeof(*times));
    if (results == NULL || times == NULL) return 3;

    (void)run_once(threads, reps / 10 + 1, results);
    for (int t = 0; t < trials; ++t)
        times[t] = run_once(threads, reps, results);
    qsort(times, (size_t)trials, sizeof(*times), compare_double);
    const double median_ms = times[trials / 2];
#ifdef ROOF_FMMLA
    const double flops_per_instruction = (double)svcntd() * 8.0;
    const char* path = "fmmla";
#else
    const double flops_per_instruction = (double)svcntd() * 2.0;
    const char* path = "fmla";
#endif
    const double flops = (double)threads * (double)reps * 8.0 * flops_per_instruction;
    double cpu_min = results[0].cpu_ms;
    double cpu_max = results[0].cpu_ms;
    double cpu_sum = 0.0;
    double sink = 0.0;
    for (int t = 0; t < threads; ++t) {
        cpu_min = fmin(cpu_min, results[t].cpu_ms);
        cpu_max = fmax(cpu_max, results[t].cpu_ms);
        cpu_sum += results[t].cpu_ms;
        sink += results[t].sink;
    }
    printf("path=%s threads=%d sve_d=%lu reps=%ld trials=%d\n",
           path, threads, (unsigned long)svcntd(), reps, trials);
    printf("flops=%.0f median_ms=%.6f gflops=%.6f cpu_min_ms=%.6f "
           "cpu_max_ms=%.6f cpu_sum_ms=%.6f sink=%.17g\n",
           flops, median_ms, flops / (median_ms * 1e6),
           cpu_min, cpu_max, cpu_sum, sink);
    printf("trial_ms");
    for (int t = 0; t < trials; ++t) printf(" %.6f", times[t]);
    putchar('\n');
    free(times);
    free(results);
    return 0;
}
