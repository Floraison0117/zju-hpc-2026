#include "../sme_update.h"

#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void *aligned_bytes(size_t bytes)
{
    const size_t rounded = (bytes + 63u) & ~(size_t)63u;
#if defined(_WIN32)
    return rounded == 0 ? NULL : malloc(rounded);
#else
    return rounded == 0 ? NULL : aligned_alloc(64, rounded);
#endif
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_matrix(double *x, int rows, int cols, int salt)
{
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            x[(size_t)i * (size_t)cols + j] =
                0.001 * (double)(1 + ((i * (salt + 7) + j * (salt + 3)) % 97));
}

static void pack_a(const double *a, int lda, int row, int k, double *pa)
{
    for (int p = 0; p < k; ++p)
        for (int i = 0; i < 16; ++i)
            pa[(size_t)p * 16u + i] = a[(size_t)(row + i) * lda + p];
}

static void pack_b(const double *b, int ldb, int n, int k, double *pb)
{
    const int nt = (n + 31) / 32;
    for (int tj = 0; tj < nt; ++tj) {
        const int col0 = tj * 32;
        const int width = n - col0 < 32 ? n - col0 : 32;
        double *tile = pb + (size_t)tj * (size_t)k * 32u;
        for (int p = 0; p < k; ++p) {
            for (int j = 0; j < width; ++j)
                tile[(size_t)p * 32u + j] =
                    b[(size_t)p * (size_t)ldb + col0 + j];
            for (int j = width; j < 32; ++j)
                tile[(size_t)p * 32u + j] = 0.0;
        }
    }
}

static void run_single(const double *a, const double *b, double *c,
                       int m, int n, int k, int panels,
                       double *pack_ms, double *kernel_ms)
{
    const int mt = (m + 15) / 16;
    const int nt = (n + 31) / 32;
    const size_t b_bytes = (size_t)nt * (size_t)k * 32u * sizeof(double);
    double *b_pack = (double *)aligned_bytes(b_bytes);
    double *a_pack = (double *)aligned_bytes(
        (size_t)omp_get_max_threads() * (size_t)k * 16u * sizeof(double));
    if (b_pack == NULL || a_pack == NULL) {
        free(b_pack); free(a_pack);
        return;
    }
    for (int panel = 0; panel < panels; ++panel) {
        double start = now_sec();
        pack_b(b, n, n, k, b_pack);
        pack_ms[panel] = (now_sec() - start) * 1e3;
        start = now_sec();
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double *pa = a_pack + (size_t)tid * (size_t)k * 16u;
#pragma omp for schedule(static)
            for (int ti = 0; ti < mt; ++ti) {
                const int row0 = ti * 16;
                pack_a(a, k, row0, k, pa);
                sme_update_16x32_batch(
                    pa, b_pack,
                    c + (size_t)panel * (size_t)m * (size_t)n +
                        (size_t)row0 * (size_t)n,
                    k, n * (int)sizeof(double), nt);
            }
        }
        kernel_ms[panel] = (now_sec() - start) * 1e3;
    }
    free(b_pack);
    free(a_pack);
}

static void run_double_buffer(const double *a, const double *b, double *c,
                              int m, int n, int k, int panels,
                              double *pack_ms, double *kernel_ms)
{
    const int mt = (m + 15) / 16;
    const int nt = (n + 31) / 32;
    const size_t b_tile_bytes = (size_t)nt * (size_t)k * 32u * sizeof(double);
    const size_t a_tile_bytes = (size_t)k * 16u * sizeof(double);
    double *b_slots[2] = {
        (double *)aligned_bytes(b_tile_bytes),
        (double *)aligned_bytes(b_tile_bytes)};
    double *a_tiles = (double *)aligned_bytes(
        (size_t)mt * a_tile_bytes);
    int ready0 = 0;
    int ready1 = 0;
    (void)ready0;
    (void)ready1;
    if (b_slots[0] == NULL || b_slots[1] == NULL || a_tiles == NULL) {
        free(b_slots[0]); free(b_slots[1]); free(a_tiles);
        return;
    }
    for (int ti = 0; ti < mt; ++ti)
        pack_a(a, k, ti * 16, k,
               a_tiles + (size_t)ti * (size_t)k * 16u);
    memset(pack_ms, 0, (size_t)panels * sizeof(*pack_ms));
    memset(kernel_ms, 0, (size_t)panels * sizeof(*kernel_ms));

    const double total_start = now_sec();
#pragma omp parallel
    {
#pragma omp single
        {
            for (int panel = 0; panel < panels; ++panel) {
                const int slot = panel & 1;
                if (panel >= 2 && (panel & 1) == 0) {
#pragma omp taskwait
                }
                if (slot == 0) {
#pragma omp task firstprivate(panel, slot) depend(out:ready0) shared(pack_ms, b_slots, b, n, k)
                    {
                        const double start = now_sec();
                        pack_b(b, n, n, k, b_slots[slot]);
                        pack_ms[panel] = (now_sec() - start) * 1e3;
                    }
                } else {
#pragma omp task firstprivate(panel, slot) depend(out:ready1) shared(pack_ms, b_slots, b, n, k)
                    {
                        const double start = now_sec();
                        pack_b(b, n, n, k, b_slots[slot]);
                        pack_ms[panel] = (now_sec() - start) * 1e3;
                    }
                }

                for (int ti = 0; ti < mt; ++ti) {
                    if (slot == 0) {
#pragma omp task firstprivate(panel, slot, ti) depend(in:ready0) shared(kernel_ms, a_tiles, c, b_slots, m, n, k)
                    {
                        const int row0 = ti * 16;
                        double *pa = a_tiles + (size_t)ti * (size_t)k * 16u;
                        const double start = now_sec();
                        sme_update_16x32_batch(
                            pa, b_slots[slot],
                            c + (size_t)panel * (size_t)m * (size_t)n +
                                (size_t)row0 * (size_t)n,
                            k, n * (int)sizeof(double), nt);
                        const double elapsed = (now_sec() - start) * 1e3;
#pragma omp atomic update
                        kernel_ms[panel] += elapsed;
                    }
                    } else {
#pragma omp task firstprivate(panel, slot, ti) depend(in:ready1) shared(kernel_ms, a_tiles, c, b_slots, m, n, k)
                    {
                        const int row0 = ti * 16;
                        double *pa = a_tiles + (size_t)ti * (size_t)k * 16u;
                        const double start = now_sec();
                        sme_update_16x32_batch(
                            pa, b_slots[slot],
                            c + (size_t)panel * (size_t)m * (size_t)n +
                                (size_t)row0 * (size_t)n,
                            k, n * (int)sizeof(double), nt);
                        const double elapsed = (now_sec() - start) * 1e3;
#pragma omp atomic update
                        kernel_ms[panel] += elapsed;
                    }
                    }
                }
            }
#pragma omp taskwait
        }
    }
    const double total_ms = (now_sec() - total_start) * 1e3;
    double pack_total = 0.0;
    double kernel_total = 0.0;
    for (int panel = 0; panel < panels; ++panel) {
        pack_total += pack_ms[panel];
        kernel_total += kernel_ms[panel];
    }
    const double serial_span = pack_total + kernel_total;
    const double overlap = serial_span > 0.0
                               ? 1.0 - total_ms / serial_span : 0.0;
    const double wait = total_ms > (pack_total > kernel_total ? pack_total : kernel_total)
                            ? total_ms - (pack_total > kernel_total ? pack_total : kernel_total)
                            : 0.0;
    printf("pack_current_ms=%.6f pack_next_overlap_ms=%.6f "
           "kernel_ms=%.6f store_ms=0.000000 wait_ms=%.6f "
           "overlap_ratio=%.6f total_ms=%.6f\n",
           pack_ms[0], pack_total - pack_ms[0], kernel_total, wait,
           overlap, total_ms);
    free(b_slots[0]); free(b_slots[1]); free(a_tiles);
}

int main(int argc, char **argv)
{
    const int m = argc > 1 ? atoi(argv[1]) : 16800;
    const int n = argc > 2 ? atoi(argv[2]) : 512;
    const int k = argc > 3 ? atoi(argv[3]) : 224;
    const int panels = argc > 4 ? atoi(argv[4]) : 4;
    if (m <= 0 || n <= 0 || k <= 0 || (m & 15) != 0 ||
        (n & 31) != 0 || panels <= 0) {
        fprintf(stderr, "usage: %s M N K PANELS, tile multiples required\n", argv[0]);
        return 2;
    }
    double *a = (double *)aligned_bytes((size_t)m * (size_t)k * sizeof(double));
    double *b = (double *)aligned_bytes((size_t)k * (size_t)n * sizeof(double));
    double *c = (double *)aligned_bytes((size_t)panels * (size_t)m *
                                         (size_t)n * sizeof(double));
    if (a == NULL || b == NULL || c == NULL) {
        free(a); free(b); free(c);
        return 3;
    }
    fill_matrix(a, m, k, 11);
    fill_matrix(b, k, n, 19);
    memset(c, 0, (size_t)panels * (size_t)m * (size_t)n * sizeof(double));
    double *single_pack = (double *)calloc((size_t)panels, sizeof(double));
    double *single_kernel = (double *)calloc((size_t)panels, sizeof(double));
    double *single_result = (double *)aligned_bytes(
        (size_t)panels * (size_t)m * (size_t)n * sizeof(double));
    if (single_pack == NULL || single_kernel == NULL || single_result == NULL)
        return 3;

    run_single(a, b, c, m, n, k, panels, single_pack, single_kernel);
    double single_total = 0.0;
    double single_pack_total = 0.0;
    double single_kernel_total = 0.0;
    for (int p = 0; p < panels; ++p) {
        single_pack_total += single_pack[p];
        single_kernel_total += single_kernel[p];
    }
    single_total = single_pack_total + single_kernel_total;
    printf("single_buffer_pack_ms=%.6f single_buffer_kernel_ms=%.6f "
           "single_buffer_serial_ms=%.6f\n",
           single_pack_total, single_kernel_total,
           single_total);
    memcpy(single_result, c, (size_t)panels * (size_t)m * (size_t)n *
                                  sizeof(double));

    memset(c, 0, (size_t)panels * (size_t)m * (size_t)n * sizeof(double));
    run_double_buffer(a, b, c, m, n, k, panels, single_pack, single_kernel);
    double max_diff = 0.0;
    size_t max_index = 0;
    for (size_t i = 0; i < (size_t)panels * (size_t)m * (size_t)n; ++i) {
        const double e = fabs(c[i] - single_result[i]);
        if (e > max_diff) { max_diff = e; max_index = i; }
    }
    printf("pipeline_max_abs_diff=%.17g index=%llu got=%.17g ref=%.17g\n",
           max_diff, (unsigned long long)max_index,
           c[max_index], single_result[max_index]);
    free(single_pack); free(single_kernel);
    free(single_result);
    free(a); free(b); free(c);
    return max_diff <= 1e-12 ? 0 : 1;
}
