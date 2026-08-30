#include <stddef.h>
#include <omp.h>
#include "kblas.h"

#ifdef TRSM_PROFILE
#include <stdio.h>
#include <stdlib.h>
#endif

/*
 * V9: shape-hybrid TRSM.  TRSM_PROFILE adds diagnostic-only timing records.
 *   Case1 (m <= 1024): recursive, leaf TRSM_LEAF1 (~32-48)
 *   Case2 (m <= 4096): recursive, leaf TRSM_LEAF2 (~32)
 *   Case3 (m > 4096):  fixed-NB right-looking, NB TRSM_NB3 (~256), M-split
 * KBLAS forced single-threaded (BlasSetNumThreads(1), constructor) and all
 * trailing updates parallelised with outer libgomp regions (N-split for large
 * n, M-split for small n); tiny GEMMs run serial (threshold).
 */
#ifndef TRSM_LEAF1
#define TRSM_LEAF1 48
#endif
#ifndef TRSM_LEAF2
#define TRSM_LEAF2 32
#endif
#ifndef TRSM_NB3
#define TRSM_NB3 224
#endif
#ifndef TRSM_SPLIT_ALIGN
#define TRSM_SPLIT_ALIGN 32
#endif
#ifndef TRSM_N_SPLIT_THR
#define TRSM_N_SPLIT_THR 4096
#endif
#ifndef TRSM_WIDE_2D
#define TRSM_WIDE_2D 1
#endif
#ifndef TRSM_2D_MIN_ROWS
#define TRSM_2D_MIN_ROWS 256
#endif
#ifndef TRSM_2D_MG
#define TRSM_2D_MG 2
#endif
#ifndef TRSM_2D_NG
#define TRSM_2D_NG 19
#endif
#ifndef TRSM_SERIAL_MINFLOP
#define TRSM_SERIAL_MINFLOP 8388608
#endif
#ifndef TRSM_W
#define TRSM_W 32
#endif

__attribute__((constructor))
static void trsm_force_serial_kblas(void)
{
    BlasSetNumThreads(1);
}

#ifdef TRSM_PROFILE

#define TRSM_PROFILE_MAX_THREADS 256

typedef struct {
    double start_offset;
    double end_offset;
    double work_ms;
    int has_work;
    int m;
    int n;
    int k;
    long long flops;
} trsm_profile_thread;

static FILE* trsm_profile_fp;
static int trsm_profile_headers_written;
static unsigned long trsm_profile_invocation;
static int trsm_profile_case;

static FILE* trsm_profile_stream(void)
{
    if (trsm_profile_fp == NULL) {
        const char* path = getenv("TRSM_PROFILE_OUT");
        trsm_profile_fp = (path != NULL && path[0] != '\0') ? fopen(path, "w") : stderr;
        if (trsm_profile_fp == NULL) trsm_profile_fp = stderr;
    }
    if (!trsm_profile_headers_written) {
        fprintf(trsm_profile_fp,
                "case,level_or_step,m,n,k,active_threads,gemm_ms,solve_ms,"
                "barrier_ms,min_thread_ms,max_thread_ms\n");
        fprintf(trsm_profile_fp,
                "# thread_case,level_or_step,tid,m,n,k,actual_flops,gemm_ms\n");
        fprintf(trsm_profile_fp,
                "# region_case,level_or_step,m,n,k,active_threads,region_ms,"
                "fork_ms,barrier_ms,join_ms,max_thread_ms\n");
        fprintf(trsm_profile_fp,
                "# step_case,step,remaining,active_threads,nb,gemm_ms,barrier_ms\n");
        trsm_profile_headers_written = 1;
    }
    return trsm_profile_fp;
}

static void trsm_profile_begin(int m, int n)
{
    ++trsm_profile_invocation;
    trsm_profile_case = (m <= 1024) ? 1 : ((m <= 4096) ? 2 : 3);
    FILE* fp = trsm_profile_stream();
    fprintf(fp, "# invocation=%lu case=%d m=%d n=%d\n",
            trsm_profile_invocation, trsm_profile_case, m, n);
}

static void trsm_profile_label(char* dst, size_t cap, int kind, int index)
{
    const char prefix = (kind == 0) ? 'L' : ((kind == 1) ? 'S' : 'D');
    (void)snprintf(dst, cap, "%c%d", prefix, index);
}

static void trsm_profile_emit_summary(int kind, int index,
                                      int m, int n, int k, int active_threads,
                                      double gemm_ms, double solve_ms,
                                      double barrier_ms, double min_thread_ms,
                                      double max_thread_ms)
{
    char label[32];
    trsm_profile_label(label, sizeof(label), kind, index);
    FILE* fp = trsm_profile_stream();
    fprintf(fp, "%d,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            trsm_profile_case, label, m, n, k, active_threads,
            gemm_ms, solve_ms, barrier_ms, min_thread_ms, max_thread_ms);
}

static void trsm_profile_emit_region(int kind, int index,
                                     int m, int n, int k, int active_threads,
                                     double region_ms, double fork_ms,
                                     double barrier_ms, double join_ms,
                                     double max_thread_ms)
{
    char label[32];
    trsm_profile_label(label, sizeof(label), kind, index);
    FILE* fp = trsm_profile_stream();
    fprintf(fp, "region,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            label, m, n, k, active_threads,
            region_ms, fork_ms, barrier_ms, join_ms, max_thread_ms);
}

static void trsm_profile_emit_thread(int kind, int index, int tid,
                                     const trsm_profile_thread* stat)
{
    char label[32];
    trsm_profile_label(label, sizeof(label), kind, index);
    FILE* fp = trsm_profile_stream();
    fprintf(fp, "thread,%s,%d,%d,%d,%d,%lld,%.6f\n",
            label, tid, stat->m, stat->n, stat->k,
            stat->flops, stat->work_ms);
}

static void trsm_profile_emit_step(int step, int remaining, int active_threads,
                                   int nb, double gemm_ms, double barrier_ms)
{
    FILE* fp = trsm_profile_stream();
    fprintf(fp, "step,%d,%d,%d,%d,%.6f,%.6f\n",
            step, remaining, active_threads, nb, gemm_ms, barrier_ms);
}

#endif

static void solve_diag_block(int bs, int n, int ii,
                             const double* L, int lda, double* B, int ldb,
                             int profile_index)
{
    const int W = (n <= 1024) ? 16 : 32;
    if (bs <= 0 || n <= 0) return;

#ifdef TRSM_PROFILE
    trsm_profile_thread stats[TRSM_PROFILE_MAX_THREADS];
    const double region_start = omp_get_wtime();
    int observed_threads = 0;
#endif

#pragma omp parallel
    {
#ifdef TRSM_PROFILE
        const int tid = omp_get_thread_num();
        const double thread_start = omp_get_wtime();
        if (tid == 0) observed_threads = omp_get_num_threads();
        if (tid < TRSM_PROFILE_MAX_THREADS) {
            stats[tid].start_offset = (thread_start - region_start) * 1e3;
            stats[tid].has_work = 0;
        }
#endif
#pragma omp for schedule(static)
        for (int jb = 0; jb < n; jb += W) {
            int je = jb + W;
            if (je > n) je = n;
            const int w = je - jb;

#ifdef TRSM_PROFILE
            if (tid < TRSM_PROFILE_MAX_THREADS) stats[tid].has_work = 1;
#endif

            const double* restrict Lbase = L + (size_t)ii * lda + ii;
            double* restrict Bbase = B + (size_t)ii * ldb + jb;
            for (int i = 0; i < bs; ++i) {
                const double* restrict Li = Lbase + (size_t)i * lda;
                double* restrict Bi = Bbase + (size_t)i * ldb;
                double s[32];
                for (int jj = 0; jj < w; ++jj) s[jj] = Bi[jj];

                const double* restrict Bk = Bbase;
                for (int k = 0; k < i; ++k) {
                    const double lik = Li[k];
#pragma omp simd
                    for (int jj = 0; jj < w; ++jj) {
                        s[jj] -= lik * Bk[jj];
                    }
                    Bk += ldb;
                }

                const double inv = 1.0 / Li[i];
#pragma omp simd
                for (int jj = 0; jj < w; ++jj) {
                    Bi[jj] = s[jj] * inv;
                }
            }
        }
#ifdef TRSM_PROFILE
        if (tid < TRSM_PROFILE_MAX_THREADS) {
            stats[tid].end_offset = (omp_get_wtime() - region_start) * 1e3;
            stats[tid].work_ms = stats[tid].end_offset - stats[tid].start_offset;
        }
#endif
    }

#ifdef TRSM_PROFILE
    {
        const double region_ms = (omp_get_wtime() - region_start) * 1e3;
        double max_start = 0.0;
        double min_end = 0.0;
        double max_end = 0.0;
        double min_thread_ms = 0.0;
        double max_thread_ms = 0.0;
        int active_threads = 0;
        const int limit = (observed_threads < TRSM_PROFILE_MAX_THREADS)
                              ? observed_threads : TRSM_PROFILE_MAX_THREADS;
        for (int tid = 0; tid < limit; ++tid) {
            if (stats[tid].start_offset > max_start) max_start = stats[tid].start_offset;
            if (tid == 0 || stats[tid].end_offset < min_end) min_end = stats[tid].end_offset;
            if (stats[tid].end_offset > max_end) max_end = stats[tid].end_offset;
            if (stats[tid].has_work) {
                if (active_threads == 0 || stats[tid].work_ms < min_thread_ms)
                    min_thread_ms = stats[tid].work_ms;
                if (stats[tid].work_ms > max_thread_ms)
                    max_thread_ms = stats[tid].work_ms;
                ++active_threads;
            }
        }
        const double barrier_ms = (max_end > min_end) ? (max_end - min_end) : 0.0;
        const double join_ms = (region_ms > max_end) ? (region_ms - max_end) : 0.0;
        trsm_profile_emit_summary(2, profile_index, bs, n, 0, active_threads,
                                  0.0, region_ms,
                                  barrier_ms, min_thread_ms, max_thread_ms);
        trsm_profile_emit_region(2, profile_index, bs, n, 0, active_threads,
                                 region_ms, max_start, barrier_ms, join_ms,
                                 max_thread_ms);
    }
#endif
}

static void dgemm_update(int rows, int n, int bs,
                         const double* L21, int lda,
                         const double* Bi, int ldb,
                         double* B2, int ldb2,
                         int profile_kind, int profile_index)
{
    if (rows <= 0 || n <= 0) return;
    if ((double)rows * n * bs < TRSM_SERIAL_MINFLOP) {
#ifdef TRSM_PROFILE
        const double gemm_start = omp_get_wtime();
#endif
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    rows, n, bs, -1.0, L21, lda, Bi, ldb, 1.0, B2, ldb2);
#ifdef TRSM_PROFILE
        const double gemm_ms = (omp_get_wtime() - gemm_start) * 1e3;
        trsm_profile_emit_summary(profile_kind, profile_index,
                                  rows, n, bs, 1, gemm_ms, 0.0, 0.0,
                                  gemm_ms, gemm_ms);
        trsm_profile_emit_thread(profile_kind, profile_index, 0,
                                 &(trsm_profile_thread){
                                     .m = rows, .n = n, .k = bs,
                                     .flops = (long long)2 * rows * n * bs,
                                     .work_ms = gemm_ms});
#endif
        return;
    }

#ifdef TRSM_PROFILE
    trsm_profile_thread stats[TRSM_PROFILE_MAX_THREADS];
    const double region_start = omp_get_wtime();
    int observed_threads = 0;
#endif
    if (n > TRSM_N_SPLIT_THR) {
#if TRSM_WIDE_2D
        if (rows >= TRSM_2D_MIN_ROWS) {
            /* 2D split: TRSM_2D_MG row groups x TRSM_2D_NG col groups.
             * Probe evidence (split2d): 2x19 beats pure N-split by 9-15% on
             * the Case2 tree update shapes (fat-row panels reuse B better). */
#pragma omp parallel
            {
                const int t = omp_get_thread_num();
                const int mg = TRSM_2D_MG, ng = TRSM_2D_NG;
                const int rg = t / ng, cg = t % ng;
                if (rg < mg) {
                    const int r0 = (int)((long)rows * rg / mg);
                    const int r1 = (int)((long)rows * (rg + 1) / mg);
                    const int c0 = (int)((long)n * cg / ng);
                    const int c1 = (int)((long)n * (cg + 1) / ng);
                    if (r1 > r0 && c1 > c0) {
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    r1 - r0, c1 - c0, bs,
                                    -1.0, L21 + (size_t)r0 * lda, lda,
                                    Bi + c0, ldb,
                                    1.0, B2 + (size_t)r0 * ldb2 + c0, ldb2);
                    }
                }
            }
            return;
        }
#endif
#pragma omp parallel
        {
            const int t = omp_get_thread_num();
            const int nt = omp_get_num_threads();
#ifdef TRSM_PROFILE
            const double thread_start = omp_get_wtime();
            if (t == 0) observed_threads = nt;
            if (t < TRSM_PROFILE_MAX_THREADS) {
                stats[t].start_offset = (thread_start - region_start) * 1e3;
                stats[t].has_work = 0;
            }
#endif
            const int per = n / nt, rem = n % nt;
            const int start = t * per + (t < rem ? t : rem);
            const int len = per + (t < rem ? 1 : 0);
            if (len > 0) {
#ifdef TRSM_PROFILE
                if (t < TRSM_PROFILE_MAX_THREADS) {
                    stats[t].has_work = 1;
                    stats[t].m = rows;
                    stats[t].n = len;
                    stats[t].k = bs;
                    stats[t].flops = (long long)2 * rows * len * bs;
                }
                const double gemm_start = omp_get_wtime();
#endif
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            rows, len, bs,
                            -1.0, L21, lda,
                            Bi + start, ldb,
                            1.0, B2 + start, ldb2);
#ifdef TRSM_PROFILE
                if (t < TRSM_PROFILE_MAX_THREADS)
                    stats[t].work_ms = (omp_get_wtime() - gemm_start) * 1e3;
#endif
            }
#ifdef TRSM_PROFILE
            if (t < TRSM_PROFILE_MAX_THREADS)
                stats[t].end_offset = (omp_get_wtime() - region_start) * 1e3;
#endif
        }
    } else {
#pragma omp parallel
        {
            const int t = omp_get_thread_num();
            const int nt = omp_get_num_threads();
#ifdef TRSM_PROFILE
            const double thread_start = omp_get_wtime();
            if (t == 0) observed_threads = nt;
            if (t < TRSM_PROFILE_MAX_THREADS) {
                stats[t].start_offset = (thread_start - region_start) * 1e3;
                stats[t].has_work = 0;
            }
#endif
            const int per = rows / nt, rem = rows % nt;
            const int start = t * per + (t < rem ? t : rem);
            const int len = per + (t < rem ? 1 : 0);
            if (len > 0) {
#ifdef TRSM_PROFILE
                if (t < TRSM_PROFILE_MAX_THREADS) {
                    stats[t].has_work = 1;
                    stats[t].m = len;
                    stats[t].n = n;
                    stats[t].k = bs;
                    stats[t].flops = (long long)2 * len * n * bs;
                }
                const double gemm_start = omp_get_wtime();
#endif
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            len, n, bs,
                            -1.0, L21 + (size_t)start * lda, lda,
                            Bi, ldb,
                            1.0, B2 + (size_t)start * ldb2, ldb2);
#ifdef TRSM_PROFILE
                if (t < TRSM_PROFILE_MAX_THREADS)
                    stats[t].work_ms = (omp_get_wtime() - gemm_start) * 1e3;
#endif
            }
#ifdef TRSM_PROFILE
            if (t < TRSM_PROFILE_MAX_THREADS)
                stats[t].end_offset = (omp_get_wtime() - region_start) * 1e3;
#endif
        }
    }

#ifdef TRSM_PROFILE
    {
        const double region_ms = (omp_get_wtime() - region_start) * 1e3;
        double max_start = 0.0;
        double min_end = 0.0;
        double max_end = 0.0;
        double min_thread_ms = 0.0;
        double max_thread_ms = 0.0;
        int active_threads = 0;
        const int limit = (observed_threads < TRSM_PROFILE_MAX_THREADS)
                              ? observed_threads : TRSM_PROFILE_MAX_THREADS;
        for (int tid = 0; tid < limit; ++tid) {
            if (stats[tid].start_offset > max_start) max_start = stats[tid].start_offset;
            if (tid == 0 || stats[tid].end_offset < min_end) min_end = stats[tid].end_offset;
            if (stats[tid].end_offset > max_end) max_end = stats[tid].end_offset;
            if (stats[tid].has_work) {
                if (active_threads == 0 || stats[tid].work_ms < min_thread_ms)
                    min_thread_ms = stats[tid].work_ms;
                if (stats[tid].work_ms > max_thread_ms)
                    max_thread_ms = stats[tid].work_ms;
                ++active_threads;
            }
        }
        const double barrier_ms = (max_end > min_end) ? (max_end - min_end) : 0.0;
        const double join_ms = (region_ms > max_end) ? (region_ms - max_end) : 0.0;
        trsm_profile_emit_summary(profile_kind, profile_index,
                                  rows, n, bs, active_threads, region_ms, 0.0,
                                  barrier_ms, min_thread_ms, max_thread_ms);
        trsm_profile_emit_region(profile_kind, profile_index,
                                 rows, n, bs, active_threads, region_ms,
                                 max_start, barrier_ms, join_ms, max_thread_ms);
        for (int tid = 0; tid < limit; ++tid) {
            if (stats[tid].has_work)
                trsm_profile_emit_thread(profile_kind, profile_index, tid, &stats[tid]);
        }
        if (trsm_profile_case == 3 && profile_kind == 1)
            trsm_profile_emit_step(profile_index, rows, active_threads, bs,
                                   region_ms, barrier_ms);
    }
#endif
}

static void trsm_rec(int m, int n, int leaf,
                     const double* L, int lda, double* B, int ldb,
                     int profile_level)
{
    if (m <= leaf) {
        solve_diag_block(m, n, 0, L, lda, B, ldb, profile_level);
        return;
    }
    int m1 = (m / 2 / TRSM_SPLIT_ALIGN) * TRSM_SPLIT_ALIGN;
    if (m1 < TRSM_SPLIT_ALIGN) m1 = TRSM_SPLIT_ALIGN;
    if (m1 >= m - TRSM_SPLIT_ALIGN) m1 = m - TRSM_SPLIT_ALIGN;
    if (m1 <= 0) m1 = m / 2;
    const int m2 = m - m1;

    trsm_rec(m1, n, leaf, L, lda, B, ldb, profile_level + 1);
    const double* L21 = L + (size_t)m1 * lda;
    double* B2 = B + (size_t)m1 * ldb;
    dgemm_update(m2, n, m1, L21, lda, B, ldb, B2, ldb,
                 0, profile_level);
    trsm_rec(m2, n, leaf, L21 + m1, lda, B2, ldb, profile_level + 1);
}

static void trsm_blocked(int m, int n, int nb,
                         const double* L, int lda, double* B, int ldb)
{
    for (int ii = 0; ii < m; ii += nb) {
        int bs = nb;
        if (ii + bs > m) bs = m - ii;
        solve_diag_block(bs, n, ii, L, lda, B, ldb, ii / nb);
        const int remaining = m - (ii + bs);
        if (remaining > 0) {
            const double* L21 = L + (size_t)(ii + bs) * lda + ii;
            const double* Bi  = B + (size_t)ii * ldb;
            double*       B2  = B + (size_t)(ii + bs) * ldb;
            dgemm_update(remaining, n, bs, L21, lda, Bi, ldb, B2, ldb,
                         1, ii / nb);
        }
    }
}

#ifdef TRSM_PERSISTENT_TEAM

/*
 * Experimental Stage-A candidate, disabled by default.  This path is kept
 * separate from V9 until a real-node profile shows that region fork/join is
 * material.  It is deliberately limited to the Case-3 M-split shape.
 */
#ifndef TRSM_PERSISTENT_THREADS
#define TRSM_PERSISTENT_THREADS 38
#endif
#ifndef TRSM_PERSISTENT_M_THRESHOLD
#define TRSM_PERSISTENT_M_THRESHOLD 4097
#endif

static void solve_diag_block_persistent(int bs, int n, int ii,
                                        const double* L, int lda, double* B, int ldb,
                                        int tid, int nt)
{
    const int W = (n <= 1024) ? 16 : 32;
    for (int jb = tid * W; jb < n; jb += nt * W) {
        int je = jb + W;
        if (je > n) je = n;
        const int w = je - jb;

        const double* restrict Lbase = L + (size_t)ii * lda + ii;
        double* restrict Bbase = B + (size_t)ii * ldb + jb;
        for (int i = 0; i < bs; ++i) {
            const double* restrict Li = Lbase + (size_t)i * lda;
            double* restrict Bi = Bbase + (size_t)i * ldb;
            double s[32];
            for (int jj = 0; jj < w; ++jj) s[jj] = Bi[jj];

            const double* restrict Bk = Bbase;
            for (int k = 0; k < i; ++k) {
                const double lik = Li[k];
#pragma omp simd
                for (int jj = 0; jj < w; ++jj) s[jj] -= lik * Bk[jj];
                Bk += ldb;
            }

            const double inv = 1.0 / Li[i];
#pragma omp simd
            for (int jj = 0; jj < w; ++jj) Bi[jj] = s[jj] * inv;
        }
    }
}

static void dgemm_update_persistent_msplit(int rows, int n, int bs,
                                           const double* L21, int lda,
                                           const double* Bi, int ldb,
                                           double* B2, int ldb2,
                                           int tid, int nt)
{
    const int per = rows / nt;
    const int rem = rows % nt;
    const int start = tid * per + (tid < rem ? tid : rem);
    const int len = per + (tid < rem ? 1 : 0);
    if (len > 0) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    len, n, bs,
                    -1.0, L21 + (size_t)start * lda, lda,
                    Bi, ldb,
                    1.0, B2 + (size_t)start * ldb2, ldb2);
    }
}

static void trsm_blocked_persistent(int m, int n, int nb,
                                    const double* L, int lda, double* B, int ldb)
{
#pragma omp parallel num_threads(TRSM_PERSISTENT_THREADS)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        for (int ii = 0; ii < m; ii += nb) {
            int bs = nb;
            if (ii + bs > m) bs = m - ii;
            solve_diag_block_persistent(bs, n, ii, L, lda, B, ldb, tid, nt);
#pragma omp barrier

            const int remaining = m - (ii + bs);
            if (remaining > 0) {
                const double* L21 = L + (size_t)(ii + bs) * lda + ii;
                const double* Bi  = B + (size_t)ii * ldb;
                double*       B2  = B + (size_t)(ii + bs) * ldb;
                dgemm_update_persistent_msplit(remaining, n, bs,
                                                L21, lda, Bi, ldb, B2, ldb,
                                                tid, nt);
            }
#pragma omp barrier
        }
    }
}

#endif

void l_trsm(int m, int n, const double* L, int lda, double* B, int ldb)
{
    if (m <= 0 || n <= 0) return;
#ifdef TRSM_PROFILE
    trsm_profile_begin(m, n);
#endif
#ifdef TRSM_PERSISTENT_TEAM
    if (m >= TRSM_PERSISTENT_M_THRESHOLD && n <= TRSM_N_SPLIT_THR) {
        trsm_blocked_persistent(m, n, TRSM_NB3, L, lda, B, ldb);
        return;
    }
#endif
    if (m <= 1024) {
        trsm_rec(m, n, TRSM_LEAF1, L, lda, B, ldb, 0);
    } else if (m <= 4096) {
        trsm_rec(m, n, TRSM_LEAF2, L, lda, B, ldb, 0);
    } else {
        trsm_blocked(m, n, TRSM_NB3, L, lda, B, ldb);
    }
}
