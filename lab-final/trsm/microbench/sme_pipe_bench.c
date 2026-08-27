#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Compare sme_update_16x32 (original) vs sme_update_16x32_pipe (pipelined)
 * on the full tile grid, 38 threads. Packs are padded by one k-chunk so the
 * pipelined kernel's look-ahead loads stay in bounds.
 * Usage: sme_pipe_bench <m> <n> <k> [reps] */

extern void sme_update_16x32(const double *a_pack, const double *b_pack,
                             double *c, int k, int ldc_bytes);
extern void sme_update_16x32_pipe(const double *a_pack, const double *b_pack,
                                  double *c, int k, int ldc_bytes);
extern void sme_update_16x32_kblock(const double *a_pack, const double *b_pack,
                                    double *c, int k, int ldc_bytes);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double median5(double values[5]) {
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            if (values[j] < values[i]) { double t = values[i]; values[i] = values[j]; values[j] = t; }
    return values[2];
}

static void fill_matrix(double *x, size_t n, int salt) {
    for (size_t i = 0; i < n; ++i)
        x[i] = 0.001 * (double)(1 + ((i * (salt + 7)) % 97));
}

int main(int argc, char **argv) {
    int m = argc > 1 ? atoi(argv[1]) : 16800;
    int n = argc > 2 ? atoi(argv[2]) : 512;
    int k = argc > 3 ? atoi(argv[3]) : 224;
    int reps = argc > 4 ? atoi(argv[4]) : 1;
    if (m <= 0 || n <= 0 || k <= 0 || m % 16 != 0 || n % 32 != 0) {
        fprintf(stderr, "m%16 or n%32\n"); return 2;
    }
    const int mt = m / 16, nt = n / 32;
    /* pads: +8 doubles per A tile (half chunk), +16 per B tile */
    const int apad = 16, bpad = 32;
    double *a = aligned_alloc(64, (size_t)m * k * 8);
    double *b = aligned_alloc(64, (size_t)k * n * 8);
    double *c  = aligned_alloc(64, (size_t)m * n * 8);
    double *c2 = aligned_alloc(64, (size_t)m * n * 8);
    double *pa = aligned_alloc(64, (size_t)mt * (k + apad) * 16 * 8);
    double *pb = aligned_alloc(64, (size_t)nt * (k + bpad) * 32 * 8);
    if (!a || !b || !c || !c2 || !pa || !pb) { fprintf(stderr, "alloc fail\n"); return 3; }
    fill_matrix(a, (size_t)m * k, 11);
    fill_matrix(b, (size_t)k * n, 19);

    /* pack with padding (a tile: [k+apad][16], b tile: [k+bpad][32]) */
    for (int ti = 0; ti < mt; ++ti)
        for (int p = 0; p < k; ++p)
            for (int i = 0; i < 16; ++i)
                pa[((size_t)ti * (k + apad) + p) * 16 + i] = a[(size_t)(ti * 16 + i) * k + p];
    for (int tj = 0; tj < nt; ++tj)
        for (int p = 0; p < k; ++p)
            for (int j = 0; j < 32; ++j)
                pb[((size_t)tj * (k + bpad) + p) * 32 + j] = b[(size_t)p * n + tj * 32 + j];

    memset(c, 0, (size_t)m * n * 8);
    memset(c2, 0, (size_t)m * n * 8);

    /* correctness: single thread reference tile */
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 32; ++j) {
            double ref = 0;
            for (int p = 0; p < k; ++p) ref += a[(size_t)i * k + p] * b[(size_t)p * n + j];
            c[(size_t)i * n + j] -= ref;
        }
    double tmp[16 * 32];
    sme_update_16x32(pa, pb, tmp, k, 32 * 8);
    double emax = 0;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 32; ++j) {
            double e = fabs(tmp[i * 32 + j] - (c[(size_t)i * n + j] - c[(size_t)i * n + j]));
            double got = tmp[i * 32 + j];
            double want = -c[(size_t)i * n + j]; /* kernel subtracted; c holds -ref */
            (void)e;
            double d = fabs(got - want);
            if (d > emax) emax = d;
        }
    /* proper check: fresh zero C, kernel should leave C = -A*B */
    memset(tmp, 0, sizeof(tmp));
    sme_update_16x32(pa, pb, tmp, k, 32 * 8);
    emax = 0;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 32; ++j) {
            double want = 0;
            for (int p = 0; p < k; ++p) want -= a[(size_t)i * k + p] * b[(size_t)p * n + j];
            double d = fabs(tmp[i * 32 + j] - want);
            if (d > emax) emax = d;
        }
    printf("orig_tile_maxerr=%.3g\n", emax);

    memset(tmp, 0, sizeof(tmp));
    sme_update_16x32_pipe(pa, pb, tmp, k, 32 * 8);
    emax = 0;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 32; ++j) {
            double want = 0;
            for (int p = 0; p < k; ++p) want -= a[(size_t)i * k + p] * b[(size_t)p * n + j];
            double d = fabs(tmp[i * 32 + j] - want);
            if (d > emax) emax = d;
        }
    printf("pipe_tile_maxerr=%.3g\n", emax);

    memset(tmp, 0, sizeof(tmp));
    sme_update_16x32_kblock(pa, pb, tmp, k, 32 * 8);
    emax = 0;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 32; ++j) {
            double want = 0;
            for (int p = 0; p < k; ++p) want -= a[(size_t)i * k + p] * b[(size_t)p * n + j];
            double d = fabs(tmp[i * 32 + j] - want);
            if (d > emax) emax = d;
        }
    printf("kblock_tile_maxerr=%.3g\n", emax);

    /* timed full grid */
    double to[5], tp[5], tk[5];
    for (int t = 0; t < 5; ++t) {
        memset(c, 0, (size_t)m * n * 8);
        double s = now_sec();
#pragma omp parallel for schedule(static)
        for (int ti = 0; ti < mt; ++ti)
            for (int tj = 0; tj < nt; ++tj)
                sme_update_16x32(pa + (size_t)ti * (k + apad) * 16,
                                 pb + (size_t)tj * (k + bpad) * 32,
                                 c + (size_t)(ti * 16) * n + tj * 32,
                                 k, n * 8);
        to[t] = now_sec() - s;
        memset(c2, 0, (size_t)m * n * 8);
        s = now_sec();
#pragma omp parallel for schedule(static)
        for (int ti = 0; ti < mt; ++ti)
            for (int tj = 0; tj < nt; ++tj)
                sme_update_16x32_pipe(pa + (size_t)ti * (k + apad) * 16,
                                      pb + (size_t)tj * (k + bpad) * 32,
                                      c2 + (size_t)(ti * 16) * n + tj * 32,
                                      k, n * 8);
        tp[t] = now_sec() - s;
        memset(c, 0, (size_t)m * n * 8);
        s = now_sec();
#pragma omp parallel for schedule(static)
        for (int ti = 0; ti < mt; ++ti)
            for (int tj = 0; tj < nt; ++tj)
                sme_update_16x32_kblock(pa + (size_t)ti * (k + apad) * 16,
                                        pb + (size_t)tj * (k + bpad) * 32,
                                        c + (size_t)(ti * 16) * n + tj * 32,
                                        k, n * 8);
        tk[t] = now_sec() - s;
    }
    double fl = 2.0 * m * n * k;
    printf("m=%d n=%d k=%d reps=%d\n", m, n, k, reps);
    printf("orig  median=%.4f ms %.1f GFLOPS\n", median5(to) * 1e3, fl / median5(to) / 1e9);
    printf("pipe  median=%.4f ms %.1f GFLOPS (x%.3f)\n", median5(tp) * 1e3, fl / median5(tp) / 1e9, median5(to) / median5(tp));
    printf("kblk  median=%.4f ms %.1f GFLOPS (x%.3f)\n", median5(tk) * 1e3, fl / median5(tk) / 1e9, median5(to) / median5(tk));
    free(a); free(b); free(c); free(c2); free(pa); free(pb);
    return 0;
}
