#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

extern void sme_fmopa_hot(long iterations);

static int cmp_double(const void* lhs, const void* rhs)
{
    const double a = *(const double*)lhs;
    const double b = *(const double*)rhs;
    return (a > b) - (a < b);
}

static double run_once(int threads, long iterations)
{
    const double start = omp_get_wtime();
#pragma omp parallel num_threads(threads)
    sme_fmopa_hot(iterations);
    return omp_get_wtime() - start;
}

int main(int argc, char** argv)
{
    const int threads = argc > 1 ? atoi(argv[1]) : 38;
    const long iterations = argc > 2 ? atol(argv[2]) : 1000000;
    const int trials = argc > 3 ? atoi(argv[3]) : 5;
    double* times = calloc((size_t)trials, sizeof(*times));
    if (times == NULL || threads < 1 || trials < 1) return 2;
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    (void)run_once(threads, iterations / 10 + 1);
    for (int i = 0; i < trials; ++i)
        times[i] = run_once(threads, iterations);
    qsort(times, (size_t)trials, sizeof(*times), cmp_double);
    const double flops = (double)threads * (double)iterations * 8.0 * 128.0;
    printf("path=sme-fmopa threads=%d iterations=%ld trials=%d flops=%.0f\n",
           threads, iterations, trials, flops);
    printf("median_ms=%.6f gflops=%.6f\n",
           times[trials / 2] * 1e3,
           flops / times[trials / 2] / 1e9);
    printf("trial_ms");
    for (int i = 0; i < trials; ++i) printf(" %.6f", times[i] * 1e3);
    putchar('\n');
    free(times);
    return 0;
}
