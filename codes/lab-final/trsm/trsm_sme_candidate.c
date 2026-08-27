#include "trsm_sme_candidate.h"
#include "sme_update.h"

#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(TRSM_PROFILE) || defined(TRSM_SME_DEBUG)
#include <stdio.h>
#endif

#ifdef TRSM_PROFILE
static FILE *sme_profile_fp;
static int sme_profile_header_written;

static FILE *sme_profile_stream(void)
{
    if (sme_profile_fp == NULL) {
        const char *path = getenv("TRSM_SME_PROFILE_OUT");
        sme_profile_fp = (path != NULL && path[0] != '\0')
                             ? fopen(path, "a") : stderr;
        if (sme_profile_fp == NULL) sme_profile_fp = stderr;
    }
    if (!sme_profile_header_written) {
        fprintf(sme_profile_fp,
                "case,version,level_or_step,m,n,k,nb,kernel,instruction_path,"
                "active_threads,pack_a_ms,pack_b_ms,kernel_ms,gemm_ms,solve_ms,"
                "barrier_ms,fork_ms,join_ms,min_thread_ms,max_thread_ms,"
                "bytes_packed,flops,gflops\n");
        sme_profile_header_written = 1;
    }
    return sme_profile_fp;
}

static void sme_profile_emit(int case_id, const char *level,
                             int m, int n, int k, int nb,
                             const char *kernel, const char *path,
                             int active_threads,
                             double pack_a_ms, double pack_b_ms,
                             double kernel_ms, double gemm_ms,
                             double solve_ms, double barrier_ms,
                             double fork_ms, double join_ms,
                             double min_thread_ms, double max_thread_ms,
                             size_t bytes_packed, double flops)
{
    FILE *fp = sme_profile_stream();
    const double gflops = kernel_ms > 0.0 ? flops / (kernel_ms * 1e6) : 0.0;
    fprintf(fp,
            "%d,SME_PIPELINE,%s,%d,%d,%d,%d,%s,%s,%d,"
            "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%llu,%.0f,%.3f\n",
            case_id, level, m, n, k, nb, kernel, path, active_threads,
            pack_a_ms, pack_b_ms, kernel_ms, gemm_ms, solve_ms, barrier_ms,
            fork_ms, join_ms, min_thread_ms, max_thread_ms,
            (unsigned long long)bytes_packed, flops, gflops);
    fflush(fp);
}
#else
#define SME_PROFILE_EMIT(...) ((void)0)
#endif

#ifndef TRSM_SME_MIN_K
#define TRSM_SME_MIN_K 32
#endif

static void *sme_alloc(size_t bytes)
{
    const size_t rounded = (bytes + 63u) & ~(size_t)63u;
#if defined(_WIN32)
    return rounded == 0 ? NULL : malloc(rounded);
#else
    return rounded == 0 ? NULL : aligned_alloc(64, rounded);
#endif
}

static void pack_a_tiles(const double *src, int lda,
                         int rows, int k, double *dst)
{
    const int mt = (rows + 15) / 16;
    for (int ti = 0; ti < mt; ++ti) {
        const int row0 = ti * 16;
        const int height = rows - row0 < 16 ? rows - row0 : 16;
        double *tile = dst + (size_t)ti * (size_t)k * 16u;
        for (int p = 0; p < k; ++p) {
            for (int i = 0; i < height; ++i)
                tile[(size_t)p * 16u + i] =
                    src[(size_t)(row0 + i) * (size_t)lda + p];
            for (int i = height; i < 16; ++i)
                tile[(size_t)p * 16u + i] = 0.0;
        }
    }
}

static void pack_b_tiles(const double *src, int ldb,
                         int n, int k, double *dst)
{
    const int nt = (n + 31) / 32;
    for (int tj = 0; tj < nt; ++tj) {
        const int col0 = tj * 32;
        const int width = n - col0 < 32 ? n - col0 : 32;
        double *tile = dst + (size_t)tj * (size_t)k * 32u;
        for (int p = 0; p < k; ++p) {
            for (int j = 0; j < width; ++j)
                tile[(size_t)p * 32u + j] =
                    src[(size_t)p * (size_t)ldb + col0 + j];
            for (int j = width; j < 32; ++j)
                tile[(size_t)p * 32u + j] = 0.0;
        }
    }
}

/* Run a complete packed tile grid with one persistent SME region per worker.
 * A flat task list lets the scheduler use all workers even when the matrix
 * has fewer row tiles than OpenMP threads. */
static int sme_update_packed_grid(const double *a_pack,
                                  const double *b_pack, double *c,
                                  int k, int ldc, int mt, int nt)
{
    const int count = mt * nt;
    const size_t a_tile_elems = (size_t)k * 16u;
    const size_t b_tile_elems = (size_t)k * 32u;
    const double **a_tiles = (const double **)malloc(
        (size_t)count * sizeof(*a_tiles));
    const double **b_tiles = (const double **)malloc(
        (size_t)count * sizeof(*b_tiles));
    double **c_tiles = (double **)malloc(
        (size_t)count * sizeof(*c_tiles));
    if (a_tiles == NULL || b_tiles == NULL || c_tiles == NULL) {
        free(a_tiles);
        free(b_tiles);
        free(c_tiles);
        return 0;
    }

    int q = 0;
    for (int ti = 0; ti < mt; ++ti) {
        for (int tj = 0; tj < nt; ++tj, ++q) {
            a_tiles[q] = a_pack + (size_t)ti * a_tile_elems;
            b_tiles[q] = b_pack + (size_t)tj * b_tile_elems;
            c_tiles[q] = c + (size_t)ti * 16u * (size_t)ldc +
                         (size_t)tj * 32u;
        }
    }

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int active_threads = omp_get_num_threads();
        const int per = count / active_threads;
        const int rem = count % active_threads;
        const int first = tid * per + (tid < rem ? tid : rem);
        const int local_count = per + (tid < rem ? 1 : 0);
        if (local_count > 0) {
            sme_update_16x32_ptr_batch(
                a_tiles + first, b_tiles + first, c_tiles + first,
                k, ldc * (int)sizeof(double), local_count);
        }
    }

    free(a_tiles);
    free(b_tiles);
    free(c_tiles);
    return 1;
}

static int candidate_shape_ok(int rows, int n, int bs)
{
    return rows >= 16 && n >= 32 && bs >= TRSM_SME_MIN_K;
}

int trsm_sme_case3_update(int rows, int n, int bs,
                          const double *l21, int lda,
                          const double *bi, int ldb,
                          double *b2, int ldb2)
{
    if (!candidate_shape_ok(rows, n, bs)) return 0;

    const int mt = (rows + 15) / 16;
    const int nt = (n + 31) / 32;
    const size_t a_tile_elems = (size_t)bs * 16u;
    const size_t a_tile_bytes = (size_t)bs * 16u * sizeof(double);
    const size_t b_bytes = (size_t)nt * (size_t)bs * 32u * sizeof(double);
    double *b_pack = (double *)sme_alloc(b_bytes);
    double *a_pack = (double *)sme_alloc((size_t)mt * a_tile_bytes);
    if (b_pack == NULL || a_pack == NULL) {
        free(b_pack);
        free(a_pack);
        return 0;
    }

#ifdef TRSM_PROFILE
    const double b_start = omp_get_wtime();
#endif
    pack_b_tiles(bi, ldb, n, bs, b_pack);
#ifdef TRSM_PROFILE
    const double pack_b_ms = (omp_get_wtime() - b_start) * 1e3;
#else
    const double pack_b_ms = 0.0;
#endif
    (void)pack_b_ms;

#ifdef TRSM_PROFILE
    const double kernel_start = omp_get_wtime();
#endif
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int active_threads = omp_get_num_threads();
        const int per = mt / active_threads;
        const int rem = mt % active_threads;
        const int first = tid * per + (tid < rem ? tid : rem);
        const int local_mt = per + (tid < rem ? 1 : 0);
        double *a_tiles = a_pack + (size_t)first * a_tile_elems;
        double pack_a_total = 0.0;

        for (int local = 0; local < local_mt; ++local) {
            const int ti = first + local;
            const int row0 = ti * 16;
            const int height = rows - row0 < 16 ? rows - row0 : 16;
            double *a_tile = a_tiles + (size_t)local * a_tile_elems;
#ifdef TRSM_PROFILE
            const double pack_a_start = omp_get_wtime();
#endif
            for (int p = 0; p < bs; ++p) {
                for (int i = 0; i < height; ++i)
                    a_tile[(size_t)p * 16u + i] =
                        l21[(size_t)(row0 + i) * (size_t)lda + p];
                for (int i = height; i < 16; ++i)
                    a_tile[(size_t)p * 16u + i] = 0.0;
            }
#ifdef TRSM_PROFILE
            pack_a_total += (omp_get_wtime() - pack_a_start) * 1e3;
#else
            (void)height;
#endif
        }
#ifndef TRSM_PROFILE
        (void)pack_a_total;
#endif

        double kernel_ms = 0.0;
        if (local_mt > 0 && (rows & 15) == 0 && (n & 31) == 0) {
#ifdef TRSM_SME_DEBUG
            if (tid == 0) {
                fprintf(stderr,
                        "grid rows=%d n=%d bs=%d first=%d mt=%d a_mod64=%llu b_mod64=%llu c_mod64=%llu ldc_bytes=%d\n",
                        rows, n, bs, first, local_mt,
                        (unsigned long long)((uintptr_t)a_tiles & 63u),
                        (unsigned long long)((uintptr_t)b_pack & 63u),
                        (unsigned long long)((uintptr_t)(b2 +
                            (size_t)first * 16u * (size_t)ldb2) & 63u),
                        ldb2 * (int)sizeof(double));
                fflush(stderr);
            }
#endif
#ifdef TRSM_PROFILE
            const double start = omp_get_wtime();
#endif
#ifdef TRSM_SME_ZA_INIT
            sme_update_16x32_za_init_grid_batch(
                a_tiles, b_pack,
                b2 + (size_t)first * 16u * (size_t)ldb2,
                bs, ldb2 * (int)sizeof(double), nt, local_mt);
#else
            sme_update_16x32_grid_batch(
                a_tiles, b_pack,
                b2 + (size_t)first * 16u * (size_t)ldb2,
                bs, ldb2 * (int)sizeof(double), nt, local_mt);
#endif
#ifdef TRSM_PROFILE
            kernel_ms = (omp_get_wtime() - start) * 1e3;
#endif
        } else if (local_mt > 0) {
#ifdef TRSM_PROFILE
            const double start = omp_get_wtime();
#endif
            for (int local = 0; local < local_mt; ++local) {
                const int ti = first + local;
                const int row0 = ti * 16;
                const int height = rows - row0 < 16 ? rows - row0 : 16;
                double *a_tile = a_tiles + (size_t)local * a_tile_elems;
                for (int tj = 0; tj < nt; ++tj) {
                    const int width = n - tj * 32 < 32 ? n - tj * 32 : 32;
                    sme_update_tile(
                        a_tile, 16,
                        b_pack + (size_t)tj * (size_t)bs * 32u, 32,
                        b2 + (size_t)row0 * (size_t)ldb2 + tj * 32,
                        ldb2 * (int)sizeof(double), height, width, bs);
                }
            }
#ifdef TRSM_PROFILE
            kernel_ms = (omp_get_wtime() - start) * 1e3;
#endif
        }
#ifndef TRSM_PROFILE
        (void)kernel_ms;
#endif
#ifdef TRSM_PROFILE
#pragma omp critical(trsm_sme_profile)
        {
            sme_profile_emit(3, "CASE3_M_BLOCK", local_mt * 16, n, bs, 16,
                             "inplace_update", "SME_FMOPA_or_scalar",
                             active_threads, pack_a_total, pack_b_ms,
                             kernel_ms, kernel_ms, 0.0, 0.0, 0.0, 0.0,
                             kernel_ms, kernel_ms,
                             (size_t)bs * (size_t)(local_mt * 16 + 32) *
                                 sizeof(double),
                             2.0 * (double)local_mt * 16.0 * n * bs);
        }
#endif
    }
#ifdef TRSM_PROFILE
    const double kernel_ms = (omp_get_wtime() - kernel_start) * 1e3;
    const int active_threads = omp_get_max_threads();
    sme_profile_emit(3, "CASE3_TOTAL", rows, n, bs, 16,
                     "inplace_update", "SME_FMOPA_or_scalar",
                     active_threads, 0.0, pack_b_ms, kernel_ms, kernel_ms,
                     0.0, 0.0, 0.0, 0.0, 0.0, kernel_ms,
                     b_bytes + (size_t)rows * (size_t)bs * sizeof(double),
                     2.0 * (double)rows * n * bs);
#endif
    free(b_pack);
    free(a_pack);
    return 1;
}

int trsm_sme_case2_update(int rows, int n, int bs,
                          const double *l21, int lda,
                          const double *bi, int ldb,
                          double *b2, int ldb2)
{
    if (!candidate_shape_ok(rows, n, bs) || rows < 512) return 0;

    const int mt = (rows + 15) / 16;
    const int nt = (n + 31) / 32;
    const int max_threads = omp_get_max_threads();
    const size_t a_bytes = (size_t)mt * (size_t)bs * 16u * sizeof(double);
    const size_t b_tile_bytes = (size_t)bs * 32u * sizeof(double);
    double *a_pack = (double *)sme_alloc(a_bytes);
    double *b_pack = (double *)sme_alloc(
        (size_t)max_threads * 2u * b_tile_bytes);
    if (a_pack == NULL || b_pack == NULL) {
        free(a_pack);
        free(b_pack);
        return 0;
    }

#ifdef TRSM_PROFILE
    const double a_start = omp_get_wtime();
#endif
    pack_a_tiles(l21, lda, rows, bs, a_pack);
#ifdef TRSM_PROFILE
    const double pack_a_ms = (omp_get_wtime() - a_start) * 1e3;
#else
    const double pack_a_ms = 0.0;
#endif
    (void)pack_a_ms;

#ifdef TRSM_PROFILE
    const double kernel_start = omp_get_wtime();
#endif
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int active_threads = omp_get_num_threads();
        (void)active_threads;
        double *b_buffers = b_pack + (size_t)tid * 2u * (size_t)bs * 32u;
#pragma omp for schedule(static)
        for (int tj = 0; tj < nt; ++tj) {
            const int col0 = tj * 32;
            const int width = n - col0 < 32 ? n - col0 : 32;
            const int slot = tj & 1;
            double *b_tile = b_buffers + (size_t)slot * (size_t)bs * 32u;
#ifdef TRSM_PROFILE
            const double pack_b_start = omp_get_wtime();
#endif
            for (int p = 0; p < bs; ++p) {
                for (int j = 0; j < width; ++j)
                    b_tile[(size_t)p * 32u + j] =
                        bi[(size_t)p * (size_t)ldb + col0 + j];
                for (int j = width; j < 32; ++j)
                    b_tile[(size_t)p * 32u + j] = 0.0;
            }
#ifdef TRSM_PROFILE
            const double pack_b_ms =
                (omp_get_wtime() - pack_b_start) * 1e3;
            const double kernel_start_tile = omp_get_wtime();
#else
            const double pack_b_ms = 0.0;
#endif
            (void)pack_b_ms;
            if ((rows & 15) == 0 && width == 32) {
                sme_update_16x32_rows_batch(
                    a_pack, b_tile, b2 + col0, bs,
                    ldb2 * (int)sizeof(double), mt);
            } else {
                for (int ti = 0; ti < mt; ++ti) {
                    const int row0 = ti * 16;
                    const int height = rows - row0 < 16 ? rows - row0 : 16;
                    sme_update_tile(
                        a_pack + (size_t)ti * (size_t)bs * 16u, 16,
                        b_tile, 32,
                        b2 + (size_t)row0 * (size_t)ldb2 + col0,
                        ldb2 * (int)sizeof(double), height, width, bs);
                }
            }
#ifdef TRSM_PROFILE
            const double kernel_ms =
                (omp_get_wtime() - kernel_start_tile) * 1e3;
            #pragma omp critical(trsm_sme_profile)
            {
                sme_profile_emit(2, "CASE2_N_PANEL", rows, width, bs, 16,
                                 "inplace_update", "SME_FMOPA_or_scalar",
                                 active_threads, 0.0, pack_b_ms, kernel_ms,
                                 kernel_ms, 0.0, 0.0, 0.0, 0.0,
                                 kernel_ms, kernel_ms,
                                 (size_t)bs * (size_t)(width + 16) * sizeof(double),
                                 2.0 * (double)rows * width * bs);
            }
#endif
        }
    }
#ifdef TRSM_PROFILE
    const double kernel_ms = (omp_get_wtime() - kernel_start) * 1e3;
    sme_profile_emit(2, "CASE2_TOTAL", rows, n, bs, 16,
                     "inplace_update", "SME_FMOPA_or_scalar",
                     omp_get_max_threads(), pack_a_ms, 0.0, kernel_ms,
                     kernel_ms, 0.0, 0.0, 0.0, 0.0, 0.0, kernel_ms,
                     a_bytes + (size_t)n * (size_t)bs * sizeof(double),
                     2.0 * (double)rows * n * bs);
#endif
    free(a_pack);
    free(b_pack);
    return 1;
}

/*
 * Left-looking Case-3 update.  The source B rows are already solved.  Pack
 * that source panel once per target block, then reuse it across all target
 * row tiles.  This avoids rereading the same raw B rows once per row tile,
 * which is particularly costly as k grows through the left-looking sweep.
 */
int trsm_sme_case3_left_update(int rows, int n, int k,
                               const double *lblock, int lda,
                               const double *b, int ldb,
                               double *c, int ldc)
{
    if (!candidate_shape_ok(rows, n, k) ||
        (rows & 15) != 0 || (n & 31) != 0)
        return 0;

    const int mt = rows / 16;
    const int nt = n / 32;
    const size_t tile_elems = (size_t)k * 16u;
    const size_t b_bytes = (size_t)nt * (size_t)k * 32u * sizeof(double);
    double *a_pack = (double *)sme_alloc((size_t)mt * tile_elems *
                                          sizeof(double));
    double *b_pack = (double *)sme_alloc(b_bytes);
    if (a_pack == NULL || b_pack == NULL) {
        free(a_pack);
        free(b_pack);
        return 0;
    }

#ifdef TRSM_PROFILE
    const double pack_b_start = omp_get_wtime();
#endif
    pack_b_tiles(b, ldb, n, k, b_pack);
#ifdef TRSM_PROFILE
    const double pack_b_ms = (omp_get_wtime() - pack_b_start) * 1e3;
    const double pack_a_start = omp_get_wtime();
#else
    const double pack_b_ms = 0.0;
#endif
#ifndef TRSM_PROFILE
    (void)pack_b_ms;
#endif
    pack_a_tiles(lblock, lda, rows, k, a_pack);
#ifdef TRSM_PROFILE
    const double pack_a_ms = (omp_get_wtime() - pack_a_start) * 1e3;
    const double kernel_start = omp_get_wtime();
#else
    const double pack_a_ms = 0.0;
#endif
#ifndef TRSM_PROFILE
    (void)pack_a_ms;
#endif

    if (!sme_update_packed_grid(a_pack, b_pack, c, k, ldc, mt, nt)) {
        free(a_pack);
        free(b_pack);
        return 0;
    }

#ifdef TRSM_PROFILE
    {
        const double kernel_ms = (omp_get_wtime() - kernel_start) * 1e3;
        sme_profile_emit(3, "CASE3_LEFT_UPDATE", rows, n, k, 16,
                         "inplace_update", "SME_FMOPA_packed_B",
                         omp_get_max_threads(), pack_a_ms, pack_b_ms, kernel_ms,
                         kernel_ms, 0.0, 0.0, 0.0, 0.0, 0.0, kernel_ms,
                         (size_t)rows * (size_t)k * sizeof(double) + b_bytes,
                         2.0 * (double)rows * n * k);
    }
#endif
    free(a_pack);
    free(b_pack);
    return 1;
}
