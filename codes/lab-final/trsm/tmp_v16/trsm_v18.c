#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdint.h>
#include <omp.h>
#ifdef __aarch64__
#include <arm_sve.h>
#endif
#include "kblas.h"

#if defined(TRSM_SME_PIPELINE) || defined(TRSM_SME_LEFT_LOOKING)
#include "trsm_sme_candidate.h"
#if defined(TRSM_SME_PIPELINE)
static int trsm_sme_active_case;
#endif
#endif

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
#ifndef TRSM_NB2
#define TRSM_NB2 224
#endif
#ifndef TRSM_SPLIT_ALIGN
#define TRSM_SPLIT_ALIGN 32
#endif
#ifndef TRSM_N_SPLIT_THR
#define TRSM_N_SPLIT_THR 4096
#endif
#ifndef TRSM_SME_LEFT_LOOKING_M_THRESHOLD
#define TRSM_SME_LEFT_LOOKING_M_THRESHOLD 4097
#endif
#ifndef TRSM_SERIAL_MINFLOP
#define TRSM_SERIAL_MINFLOP 8388608
#endif
#ifndef TRSM_W
#define TRSM_W 32
#endif
#ifndef TRSM_KBLAS_THREADS
#define TRSM_KBLAS_THREADS 1
#endif

__attribute__((constructor))
static void trsm_force_serial_kblas(void)
{
    BlasSetNumThreads(TRSM_KBLAS_THREADS);
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
                "case,version,level_or_step,m,n,k,nb,kernel,instruction_path,"
                "active_threads,pack_a_ms,pack_b_ms,kernel_ms,gemm_ms,solve_ms,"
                "barrier_ms,fork_ms,join_ms,min_thread_ms,max_thread_ms,"
                "bytes_packed,flops,gflops\n");
        trsm_profile_headers_written = 1;
    }
    return trsm_profile_fp;
}

static void trsm_profile_emit_row(const char* version, const char* level,
                                  int m, int n, int k, int nb,
                                  const char* kernel, const char* path,
                                  int active_threads,
                                  double pack_a_ms, double pack_b_ms,
                                  double kernel_ms, double gemm_ms,
                                  double solve_ms, double barrier_ms,
                                  double fork_ms, double join_ms,
                                  double min_thread_ms, double max_thread_ms,
                                  size_t bytes_packed, double flops)
{
    FILE* fp = trsm_profile_stream();
    const double gflops = kernel_ms > 0.0 ? flops / (kernel_ms * 1e6) : 0.0;
    fprintf(fp,
            "%d,%s,%s,%d,%d,%d,%d,%s,%s,%d,"
            "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%llu,%.0f,%.3f\n",
            trsm_profile_case, version, level, m, n, k, nb, kernel, path,
            active_threads, pack_a_ms, pack_b_ms, kernel_ms, gemm_ms,
            solve_ms, barrier_ms, fork_ms, join_ms, min_thread_ms,
            max_thread_ms, (unsigned long long)bytes_packed, flops, gflops);
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
    trsm_profile_emit_row("V9", label, m, n, k, 0,
                          kind == 2 ? "solve" : "update",
                          kind == 2 ? "scalar" : "KBLAS",
                          active_threads, 0.0, 0.0,
                          gemm_ms > 0.0 ? gemm_ms : solve_ms,
                          gemm_ms, solve_ms, barrier_ms, 0.0, 0.0,
                          min_thread_ms, max_thread_ms, 0,
                          (double)2 * m * n * k);
}

static void trsm_profile_emit_region(int kind, int index,
                                     int m, int n, int k, int active_threads,
                                     double region_ms, double fork_ms,
                                     double barrier_ms, double join_ms,
                                     double max_thread_ms)
{
    char label[32];
    trsm_profile_label(label, sizeof(label), kind, index);
    trsm_profile_emit_row("V9", label, m, n, k, 0, "region", "runtime",
                          active_threads, 0.0, 0.0, region_ms, region_ms,
                          0.0, barrier_ms, fork_ms, join_ms, 0.0,
                          max_thread_ms, 0, 0.0);
}

static void trsm_profile_emit_thread(int kind, int index, int tid,
                                     const trsm_profile_thread* stat)
{
    char label[32];
    trsm_profile_label(label, sizeof(label), kind, index);
    (void)tid;
    trsm_profile_emit_row("V9", label, stat->m, stat->n, stat->k, 0,
                          "thread", "runtime", 1, 0.0, 0.0,
                          stat->work_ms, stat->work_ms, 0.0, 0.0, 0.0,
                          0.0, stat->work_ms, stat->work_ms, 0,
                          (double)stat->flops);
}

static void trsm_profile_emit_step(int step, int remaining, int active_threads,
                                   int nb, double gemm_ms, double barrier_ms)
{
    char label[32];
    (void)snprintf(label, sizeof(label), "step%d", step);
    trsm_profile_emit_row("V9", label, remaining, 0, 0, nb, "step",
                          "runtime", active_threads, 0.0, 0.0, gemm_ms,
                          gemm_ms, 0.0, barrier_ms, 0.0, 0.0, 0.0,
                          gemm_ms, 0, 0.0);
}

#endif

static void solve_diag_block(int bs, int n, int ii,
                             const double* L, int lda, double* B, int ldb,
                             int profile_index)
{
    const int W = (n <= 1024) ? 16 : 32;
    if (bs <= 0 || n <= 0) return;
#ifndef TRSM_PROFILE
    (void)profile_index;
#endif

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

#ifdef TRSM_USE_SME_UPDATE
static int sme_update_block(int rows, int n, int bs,
                            const double *l21, int lda,
                            const double *bi, int ldb,
                            double *b2, int ldb2);
static int sme_nslice_update(int rows, int n, int bs,
                             const double *l21, int lda,
                             const double *bi, int ldb,
                             double *b2, int ldb2);
#ifdef TRSM_CASE2_SME_NSLICE
static void trsm_blocked_nslice_sme(int m, int n, int nb,
                                    const double *L, int lda,
                                    double *B, int ldb);
#endif
#endif

static void dgemm_update(int rows, int n, int bs,
                         const double* L21, int lda,
                         const double* Bi, int ldb,
                         double* B2, int ldb2,
                         int profile_kind, int profile_index)
{
    if (rows <= 0 || n <= 0) return;
#ifndef TRSM_PROFILE
    (void)profile_kind;
    (void)profile_index;
#endif

#if defined(TRSM_SME_PIPELINE) && defined(TRSM_SME_CASE3) && \
    !defined(TRSM_KBLAS_INTERNAL)
    if (trsm_sme_active_case == 3 &&
        trsm_sme_case3_update(rows, n, bs, L21, lda, Bi, ldb, B2, ldb2))
        return;
#elif defined(TRSM_SME_PIPELINE) && defined(TRSM_SME_CASE2) && \
      !defined(TRSM_KBLAS_INTERNAL)
    if (trsm_sme_active_case == 2 &&
        trsm_sme_case2_update(rows, n, bs, L21, lda, Bi, ldb, B2, ldb2))
        return;
#endif

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

#ifdef TRSM_USE_SME_UPDATE
    /* SME grid update: M-split shared-B-pack for narrow-n (Case3);
     * per-thread N-slice for wide-n (Case1/2, cheap A re-read). */
    if (n <= 1024) {
        if (n % 32 == 0 && rows % 16 == 0) {
            if (sme_update_block(rows, n, bs, L21, lda, Bi, ldb, B2, ldb2))
                return;
        }
    } else if (bs % 2 == 0 && rows % 16 == 0 && n % 32 == 0) {
        if (sme_nslice_update(rows, n, bs, L21, lda, Bi, ldb, B2, ldb2))
            return;
    }
#endif

#ifdef TRSM_KBLAS_INTERNAL
    /* KBLAS owns the full parallel DGEMM region in this isolated
     * experiment; the default V9 path remains outer-OpenMP plus one thread. */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, n, bs, -1.0, L21, lda,
                Bi, ldb, 1.0, B2, ldb2);
    return;
#endif

#ifdef TRSM_PROFILE
    trsm_profile_thread stats[TRSM_PROFILE_MAX_THREADS];
    const double region_start = omp_get_wtime();
    int observed_threads = 0;
#endif
    if (n > TRSM_N_SPLIT_THR) {
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

/*
 * V17: blocked right-looking with the diagonal block solved recursively
 * (GEMM-based).  A large outer nb cuts the trailing C re-read traffic
 * (C is rewritten once per panel), while the recursive diag solve keeps
 * the panel-solve flops inside the parallel GEMM engine instead of the
 * slow scalar chain.  TRSM_BLOCKED_REC_DIAG_LEAF controls the leaf size.
 */
#ifndef TRSM_BLOCKED_REC_DIAG_LEAF
#define TRSM_BLOCKED_REC_DIAG_LEAF 64
#endif
static void trsm_blocked_rec_diag(int m, int n, int nb,
                                  const double* L, int lda, double* B, int ldb)
{
#ifdef TRSM_DOUBLE_PANEL
    /* Two panels are solved per outer step; the far update then covers both
     * (K = 2*nb) so each trailing C element is read/written once per two
     * panels, halving the C traffic.  The two L column blocks are contiguous
     * in L and the two X row blocks are contiguous in B, so the same
     * sme_update_block call with bs = 2*nb works directly. */
    for (int ii = 0; ii < m; ii += 2 * nb) {
        int bs1 = nb, bs2 = nb;
        if (ii + bs1 > m) bs1 = m - ii;
        const int j2 = ii + bs1;
        if (j2 + bs2 > m) bs2 = m - j2;
        /* solve panel ii (its B rows already updated by prior steps) */
        trsm_rec(bs1, n, TRSM_BLOCKED_REC_DIAG_LEAF,
                 L + (size_t)ii * lda + ii, lda,
                 B + (size_t)ii * ldb, ldb, 0);
        /* intra-pair: update panel j2 from panel ii */
        if (bs2 > 0 && bs1 > 0) {
            const double* L21p = L + (size_t)j2 * lda + ii;
            const double* Bip  = B + (size_t)ii * ldb;
            double*       B2p  = B + (size_t)j2 * ldb;
            dgemm_update(bs2, n, bs1, L21p, lda, Bip, ldb, B2p, ldb,
                         1, ii / nb);
            /* solve panel j2 */
            trsm_rec(bs2, n, TRSM_BLOCKED_REC_DIAG_LEAF,
                     L + (size_t)j2 * lda + j2, lda,
                     B + (size_t)j2 * ldb, ldb, 0);
        }
        const int remaining = m - (ii + bs1 + bs2);
        if (remaining > 0 && bs1 + bs2 > 0) {
            const double* L21 = L + (size_t)(ii + bs1 + bs2) * lda + ii;
            const double* Bi  = B + (size_t)ii * ldb;
            double*       B2  = B + (size_t)(ii + bs1 + bs2) * ldb;
            dgemm_update(remaining, n, bs1 + bs2, L21, lda, Bi, ldb, B2, ldb,
                         1, ii / nb);
        }
    }
#else
    for (int ii = 0; ii < m; ii += nb) {
        int bs = nb;
        if (ii + bs > m) bs = m - ii;
        trsm_rec(bs, n, TRSM_BLOCKED_REC_DIAG_LEAF, L + (size_t)ii * lda + ii,
                 lda, B + (size_t)ii * ldb, ldb, 0);
        const int remaining = m - (ii + bs);
        if (remaining > 0) {
            const double* L21 = L + (size_t)(ii + bs) * lda + ii;
            const double* Bi  = B + (size_t)ii * ldb;
            double*       B2  = B + (size_t)(ii + bs) * ldb;
            dgemm_update(remaining, n, bs, L21, lda, Bi, ldb, B2, ldb,
                         1, ii / nb);
        }
    }
#endif
}

#ifdef TRSM_SME_LEFT_LOOKING

/*
 * Case-3 left-looking schedule.  Once block ii is reached, all rows above
 * it have already been solved.  Update this block from those rows, then
 * solve its diagonal block.  The candidate writes only the current B block,
 * avoiding the repeated read/modify/write traffic of right-looking updates.
 */
static void trsm_blocked_left_looking_sme(int m, int n, int nb,
                                          const double* L, int lda,
                                          double* B, int ldb)
{
    for (int ii = 0; ii < m; ii += nb) {
        int bs = nb;
        if (ii + bs > m) bs = m - ii;

        if (ii > 0) {
            const double* lblock = L + (size_t)ii * (size_t)lda;
            double* c = B + (size_t)ii * (size_t)ldb;
#ifndef TRSM_SME_LEFT_LOOKING_KBLAS_ONLY
            if (!trsm_sme_case3_left_update(bs, n, ii,
                                            lblock, lda, B, ldb,
                                            c, ldb)) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            bs, n, ii, -1.0, lblock, lda,
                            B, ldb, 1.0, c, ldb);
            }
#else
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        bs, n, ii, -1.0, lblock, lda,
                        B, ldb, 1.0, c, ldb);
#endif
        }

        solve_diag_block(bs, n, ii, L, lda, B, ldb, ii / nb);
    }
}

#endif

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
#ifdef TRSM_SME_PIPELINE
    trsm_sme_active_case = (m <= 1024) ? 1 : ((m <= 4096) ? 2 : 3);
#endif
#ifdef TRSM_PROFILE
    trsm_profile_begin(m, n);
#endif
#ifdef TRSM_SME_LEFT_LOOKING
    if (m >= TRSM_SME_LEFT_LOOKING_M_THRESHOLD &&
        n <= TRSM_N_SPLIT_THR) {
        trsm_blocked_left_looking_sme(m, n, TRSM_NB3, L, lda, B, ldb);
        return;
    }
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
#ifdef TRSM_CASE2_BLOCKED
        trsm_blocked_rec_diag(m, n, TRSM_NB2, L, lda, B, ldb);
#elif defined(TRSM_CASE2_SME_NSLICE)
        trsm_blocked_nslice_sme(m, n, TRSM_NB2, L, lda, B, ldb);
#else
        trsm_rec(m, n, TRSM_LEAF2, L, lda, B, ldb, 0);
#endif
    } else {
#ifdef TRSM_CASE3_RECURSIVE
        trsm_rec(m, n, TRSM_LEAF3, L, lda, B, ldb, 0);
#else
#ifdef TRSM_CASE3_REC_DIAG
        trsm_blocked_rec_diag(m, n, TRSM_NB3, L, lda, B, ldb);
#else
        trsm_blocked(m, n, TRSM_NB3, L, lda, B, ldb);
#endif
#endif
    }
}

/* ================= V18: SME-kernel update path ================= */
#ifndef TRSM_V18_PIPE
#define TRSM_V18_PIPE 1
#endif

#if TRSM_V18_PIPE
extern void sme_update_16x32_pipe(const double *a_pack, const double *b_pack,
                                  double *c, int k, int ldc_bytes);
#else
extern void sme_update_16x32(const double *a_pack, const double *b_pack,
                             double *c, int k, int ldc_bytes);
#endif

/*
 * Full-tile SME update: C(rows x n) -= L21(rows x bs) * Bi(bs x n).
 * Handles row/col tails with scalar fallbacks.  Shared B pack; per-thread
 * A pack; 38-thread M-split over the row tiles.
 */
/* persistent pack buffers (avoid per-step mmap + first-touch faults) */
static double *sme_pack_abuf;
static double *sme_pack_bbuf;
static size_t  sme_pack_abuf_n;
static size_t  sme_pack_bbuf_n;

static int sme_pack_reserve(size_t a_doubles, size_t b_doubles)
{
    if (a_doubles > sme_pack_abuf_n) {
        free(sme_pack_abuf);
        sme_pack_abuf = (double *)malloc(a_doubles * sizeof(double));
        if (!sme_pack_abuf) return 0;
        sme_pack_abuf_n = a_doubles;
    }
    if (b_doubles > sme_pack_bbuf_n) {
        free(sme_pack_bbuf);
        sme_pack_bbuf = (double *)malloc(b_doubles * sizeof(double));
        if (!sme_pack_bbuf) return 0;
        sme_pack_bbuf_n = b_doubles;
    }
    return 1;
}

static int sme_update_block(int rows, int n, int bs,
                            const double *l21, int lda,
                            const double *bi, int ldb,
                            double *b2, int ldb2)
{
    if (rows <= 0 || n <= 0 || bs <= 0) return 0;
    const int mt = rows / 16;       /* full row tiles */
    const int nt = n / 32;          /* full col tiles */
    const int rtail = rows - mt * 16;
    const int ctail = n - nt * 32;
    if (mt == 0) return 0;          /* nothing for the SME grid */

    const size_t arow = (size_t)bs + 16;   /* padded A tile rows */
    const size_t brow = (size_t)bs + 32;   /* padded B tile rows */
    double *b_pack, *a_pack;
    if (!sme_pack_reserve((size_t)mt * arow * 16, (size_t)nt * brow * 32))
        return 0;
    b_pack = sme_pack_bbuf;
    a_pack = sme_pack_abuf;

    /* shared B pack: nt x (bs+32) x 32 */
#pragma omp parallel for schedule(static)
    for (int tj = 0; tj < nt; ++tj)
        for (int p = 0; p < bs; ++p)
            for (int j = 0; j < 32; ++j)
                b_pack[((size_t)tj * brow + p) * 32 + j] =
                    bi[(size_t)p * ldb + (size_t)tj * 32 + j];

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const int per = mt / nthreads;
        const int rem = mt % nthreads;
        const int first = tid * per + (tid < rem ? tid : rem);
        const int local_mt = per + (tid < rem ? 1 : 0);

        for (int local = 0; local < local_mt; ++local) {
            const int ti = first + local;
            const int row0 = ti * 16;
            double *a_tile = a_pack + (size_t)ti * arow * 16;
#ifdef __aarch64__
            /* vectorised transpose: one SVE gather per k-value */
            {
                svbool_t pg = svptrue_b64();
                uint64_t base0 = (uint64_t)(uintptr_t)(l21 + (size_t)row0 * lda);
                for (int p = 0; p < bs; ++p) {
                    /* 16 rows = two 8-lane gathers */
                    svuint64_t a0 = svindex_u64(base0 + (uint64_t)p * 8,
                                                (uint64_t)(lda * 8));
                    svuint64_t a1 = svindex_u64(base0 + (uint64_t)(8 * lda + p) * 8,
                                                (uint64_t)(lda * 8));
                    svfloat64_t v0 = svld1_gather_u64base_f64(pg, a0);
                    svfloat64_t v1 = svld1_gather_u64base_f64(pg, a1);
                    svst1_f64(pg, a_tile + (size_t)p * 16, v0);
                    svst1_f64(pg, a_tile + (size_t)p * 16 + 8, v1);
                }
            }
#else
            for (int p = 0; p < bs; ++p) {
                for (int i = 0; i < 16; ++i)
                    a_tile[(size_t)p * 16 + i] = l21[(size_t)(row0 + i) * lda + p];
            }
#endif
            for (int tj = 0; tj < nt; ++tj) {
#if TRSM_V18_PIPE
                sme_update_16x32_pipe(a_tile,
                                      b_pack + (size_t)tj * brow * 32,
                                      b2 + (size_t)row0 * ldb2 + (size_t)tj * 32,
                                      bs, ldb2 * (int)sizeof(double));
#else
                sme_update_16x32(a_tile,
                                 b_pack + (size_t)tj * brow * 32,
                                 b2 + (size_t)row0 * ldb2 + (size_t)tj * 32,
                                 bs, ldb2 * (int)sizeof(double));
#endif
            }
        }
    }
    /* tails: remaining rows and/or cols via serial KBLAS, touching only the
     * tail regions (full-tile area already updated by the SME grid) */
    if (rtail > 0) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    rtail, n, bs, -1.0,
                    l21 + (size_t)mt * 16 * lda, lda,
                    bi, ldb, 1.0,
                    b2 + (size_t)mt * 16 * ldb2, ldb2);
    }
    if (ctail > 0) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    mt * 16, ctail, bs, -1.0, l21, lda,
                    bi + (size_t)nt * 32, ldb, 1.0,
                    b2 + (size_t)nt * 32, ldb2);
    }
    return 1;
}

/*
 * V18 Case2 path: right-looking with N-split updates (each thread owns a
 * column slice of the wide n=17024 matrix) executed by the SME grid.  The
 * B pack is per-thread (its own column slice), so the shared B-re-read that
 * kills M-split is avoided; the A operand is re-read per thread but the
 * SME tile engine keeps it cheap.  Only active for wide-n cases under
 * TRSM_CASE2_SME_NSLICE.
 */
#if defined(TRSM_CASE2_SME_NSLICE) || defined(TRSM_USE_SME_UPDATE)

static int sme_nslice_update(int rows, int n, int bs,
                             const double *l21, int lda,
                             const double *bi, int ldb,
                             double *b2, int ldb2)
{
    if (rows <= 0 || n <= 0 || bs <= 0) return 0;
    if (bs % 2 != 0) return 0;              /* pipe kernel needs even k */
    if (rows % 16 != 0) return 0;           /* full row tiles only */
    const int nt_total = n / 32;
    if (nt_total == 0) return 0;
    const int ctail = n - nt_total * 32;

    const size_t arow = (size_t)bs + 16;
    const size_t brow = (size_t)bs + 32;

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const int per = nt_total / nthreads;
        const int rem = nt_total % nthreads;
        const int first = tid * per + (tid < rem ? tid : rem);
        const int local_nt = per + (tid < rem ? 1 : 0);
        if (local_nt > 0) {
        /* per-thread B pack for its own column tiles (persistent buffer) */
        static double *b_pack_tls[64];
        static size_t  b_pack_tls_n[64];
        static double *a_pack_shared;
        static size_t  a_pack_shared_n;
        static int     a_pack_shared_ok;
        double *a_pack, *b_pack;
#pragma omp single
        {
            size_t need_a = (size_t)(rows / 16) * arow * 16;
            if (need_a > a_pack_shared_n) {
                free(a_pack_shared);
                a_pack_shared = (double *)malloc(need_a * sizeof(double));
                a_pack_shared_n = need_a;
            }
            a_pack_shared_ok = (a_pack_shared != NULL);
        }
        if (a_pack_shared_ok && tid < 64) {
        if ((size_t)local_nt * brow * 32 > b_pack_tls_n[tid]) {
            free(b_pack_tls[tid]);
            b_pack_tls[tid] = (double *)malloc((size_t)local_nt * brow * 32 * sizeof(double));
            b_pack_tls_n[tid] = (size_t)local_nt * brow * 32;
        }
        a_pack = a_pack_shared;
        b_pack = b_pack_tls[tid];
        if (a_pack && b_pack) {

        for (int l = 0; l < local_nt; ++l) {
            const int tj = first + l;
            for (int p = 0; p < bs; ++p)
                for (int j = 0; j < 32; ++j)
                    b_pack[((size_t)l * brow + p) * 32 + j] =
                        bi[(size_t)p * ldb + (size_t)tj * 32 + j];
        }
        for (int ti = 0; ti < rows / 16; ++ti) {
            double *a_tile = a_pack + (size_t)ti * arow * 16;
#ifdef __aarch64__
            {
                svbool_t pg = svptrue_b64();
                uint64_t base0 = (uint64_t)(uintptr_t)(l21 + (size_t)ti * 16 * lda);
                for (int p = 0; p < bs; ++p) {
                    /* 16 rows = two 8-lane gathers */
                    svuint64_t a0 = svindex_u64(base0 + (uint64_t)p * 8,
                                                (uint64_t)(lda * 8));
                    svuint64_t a1 = svindex_u64(base0 + (uint64_t)(8 * lda + p) * 8,
                                                (uint64_t)(lda * 8));
                    svfloat64_t v0 = svld1_gather_u64base_f64(pg, a0);
                    svfloat64_t v1 = svld1_gather_u64base_f64(pg, a1);
                    svst1_f64(pg, a_tile + (size_t)p * 16, v0);
                    svst1_f64(pg, a_tile + (size_t)p * 16 + 8, v1);
                }
            }
#else
            for (int p = 0; p < bs; ++p)
                for (int i = 0; i < 16; ++i)
                    a_tile[(size_t)p * 16 + i] =
                        l21[(size_t)(ti * 16 + i) * lda + p];
#endif
            for (int l = 0; l < local_nt; ++l) {
                const int tj = first + l;
                sme_update_16x32_pipe(a_tile,
                                      b_pack + (size_t)l * brow * 32,
                                      b2 + (size_t)ti * 16 * ldb2 + (size_t)tj * 32,
                                      bs, ldb2 * (int)sizeof(double));
            }
        }
        }
        }
        }
    }
    /* column tail */
    if (ctail > 0) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    rows, ctail, bs, -1.0, l21, lda,
                    bi + (size_t)nt_total * 32, ldb, 1.0,
                    b2 + (size_t)nt_total * 32, ldb2);
    }
    return 1;
}

static void trsm_blocked_nslice_sme(int m, int n, int nb,
                                    const double *L, int lda,
                                    double *B, int ldb)
{
    for (int ii = 0; ii < m; ii += nb) {
        int bs = nb;
        if (ii + bs > m) bs = m - ii;
        trsm_rec(bs, n, TRSM_BLOCKED_REC_DIAG_LEAF,
                 L + (size_t)ii * lda + ii, lda,
                 B + (size_t)ii * ldb, ldb, 0);
        const int remaining = m - (ii + bs);
        if (remaining > 0) {
            if (!sme_nslice_update(remaining, n, bs,
                                   L + (size_t)(ii + bs) * lda + ii, lda,
                                   B + (size_t)ii * ldb, ldb,
                                   B + (size_t)(ii + bs) * ldb, ldb))
                dgemm_update(remaining, n, bs,
                             L + (size_t)(ii + bs) * lda + ii, lda,
                             B + (size_t)ii * ldb, ldb,
                             B + (size_t)(ii + bs) * ldb, ldb,
                             1, ii / nb);
        }
    }
}
#endif
