#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* Corrected STREAM probe. Usage: stream_bench <MB> [mode]
 * mode 0 = copy (read a, write c), 2 = triad (read a+b, write c), 3 = read-only */
int main(int argc, char** argv)
{
    long mb = argc > 1 ? atol(argv[1]) : 1024;
    int mode = argc > 2 ? atoi(argv[2]) : 0;
    size_t n = (size_t)mb * 1024 * 1024 / 8;
    double* a = malloc(n * 8);
    double* b = malloc(n * 8);
    double* c = malloc(n * 8);
    if (!a || !b || !c) { printf("malloc fail\n"); return 1; }

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }

    int reps = 5;
    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();
        if (mode == 0) {
#pragma omp parallel for
            for (size_t i = 0; i < n; i++) c[i] = a[i];
        } else if (mode == 2) {
#pragma omp parallel for
            for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < n; i++) { if (a[i] > 1e30) c[0] = 1.0; }
        }
        double t = omp_get_wtime() - t0;
        double bytes = (mode == 2) ? 3.0 * n * 8 : ((mode == 0) ? 2.0 * n * 8 : 1.0 * n * 8);
        printf("  rep%d: %.4f ms  %.1f GB/s\n", r, t * 1e3, bytes / t / 1e9);
    }
    volatile double sink = a[0] + b[0] + c[0];
    printf("sink=%f\n", sink);
    free(a); free(b); free(c);
    return 0;
}
