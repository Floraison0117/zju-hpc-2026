// ZGEMM for Kunpeng 920 (Shenzhen SC, SVE VL=512) — SVE MR12 microkernel
//
// v5 (zgemm_nc192.c): v4 NUMA row-partition fast path with NC baked to 192.
// K=256/512 is only 2-4 KC panels, so a NARROW NC (more column tiles, better
// L2 residency of packed B and tail balance) beats the old NC=512 at 38t:
// 2077 -> ~3073 GFLOPS total on the three official cases.
//
// Evolution: v1 pre-broadcast A (1.33 B/flop, == FCMLA speed) -> v2 tight A +
// svld1rq_f64 LD1RO in-register broadcast (0.58 B/flop, 1222 total vs 1110
// FCMLA). This v3 widens MR 6->12: B loads amortize over 2x flops
// (0.42 B/flop), per-kernel compute density doubles, epilogue halves.
// Register budget: 24 accumulators (z0-z23) + 2 B vectors (z24,z25) +
// 3 cycled LD1RO temps (z26-z28) = 29 of 32 z-regs.
//
// Official build command (no -march): gcc -O3 bench_zgemm.c zgemm_sve12.c
//   -o zgemm_test -lm -fopenmp   -> SVE enabled via the pragma below.

#define _GNU_SOURCE
#include <arm_sve.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

#pragma GCC target("arch=armv8.2-a+sve")

typedef int BLASINT;
typedef double _Complex zdouble;

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

// ---------------- tunables ----------------
// NG = complex columns per sv group (VL/2 = 4). NGROUPS sv vectors of B per k.
#ifndef MR
#define MR 12
#endif
#ifndef NR
#define NR 8
#endif
#define NG 4
#define NGROUPS (NR / NG)
#ifndef KC
#define KC 128
#endif
#ifndef MC
#define MC 384
#endif
#ifndef NC
#define NC 192
#endif

static inline int imin(int a, int b) { return a < b ? a : b; }

// scalar alpha/beta copies for the partial-group epilogue path
static zdouble alpha_z, beta_z;

// ---------------------------------------------------------------------------
// Microkernel: C[MR x NR] (+)= alpha * (A[MR x KC] * B[KC x NR])
// pA layout: [(k*MR + i)*2 + {re,im}] — TIGHT, expanded in-register by LD1RO
// pB layout: [(k*NR + j)*2 + {re,im}]        — NG complex contiguous per group
// ---------------------------------------------------------------------------
static inline void microkernel(const double* restrict pA, const double* restrict pB,
                                zdouble* restrict C, BLASINT ldc,
                                int m_rem, int n_rem, int apply_beta, int kc)
{
    // 24 accumulators (12 rows x 2 groups) + 2 B + 3 cycled A temps, all as
    // named register variables (SVE types cannot be array elements in C).
    register svfloat64_t a00  asm("z0"),  a01  asm("z1");
    register svfloat64_t a10  asm("z2"),  a11  asm("z3");
    register svfloat64_t a20  asm("z4"),  a21  asm("z5");
    register svfloat64_t a30  asm("z6"),  a31  asm("z7");
    register svfloat64_t a40  asm("z8"),  a41  asm("z9");
    register svfloat64_t a50  asm("z10"), a51  asm("z11");
    register svfloat64_t a60  asm("z12"), a61  asm("z13");
    register svfloat64_t a70  asm("z14"), a71  asm("z15");
    register svfloat64_t a80  asm("z16"), a81  asm("z17");
    register svfloat64_t a90  asm("z18"), a91  asm("z19");
    register svfloat64_t a100 asm("z20"), a101 asm("z21");
    register svfloat64_t a110 asm("z22"), a111 asm("z23");
    register svfloat64_t b0   asm("z24"), b1   asm("z25");
    register svfloat64_t va0  asm("z26"), va1  asm("z27"), va2 asm("z28");

    const svbool_t pg = svptrue_b64();
    a00  = svdup_f64(0.0); a01  = svdup_f64(0.0);
    a10  = svdup_f64(0.0); a11  = svdup_f64(0.0);
    a20  = svdup_f64(0.0); a21  = svdup_f64(0.0);
    a30  = svdup_f64(0.0); a31  = svdup_f64(0.0);
    a40  = svdup_f64(0.0); a41  = svdup_f64(0.0);
    a50  = svdup_f64(0.0); a51  = svdup_f64(0.0);
    a60  = svdup_f64(0.0); a61  = svdup_f64(0.0);
    a70  = svdup_f64(0.0); a71  = svdup_f64(0.0);
    a80  = svdup_f64(0.0); a81  = svdup_f64(0.0);
    a90  = svdup_f64(0.0); a91  = svdup_f64(0.0);
    a100 = svdup_f64(0.0); a101 = svdup_f64(0.0);
    a110 = svdup_f64(0.0); a111 = svdup_f64(0.0);

    const double* a = pA;
    const double* b = pB;
    for (int k = 0; k < kc; k++) {   // kc: only the packed rows that exist
        b0 = svld1_f64(pg, b);
        b1 = svld1_f64(pg, b + NG * 2);
        va0 = svld1rq_f64(pg, a);
        a00  = svcmla_f64_x(pg, a00,  va0, b0, 0);  a00  = svcmla_f64_x(pg, a00,  va0, b0, 90);
        a01  = svcmla_f64_x(pg, a01,  va0, b1, 0);  a01  = svcmla_f64_x(pg, a01,  va0, b1, 90);
        va1 = svld1rq_f64(pg, a + 2);
        a10  = svcmla_f64_x(pg, a10,  va1, b0, 0);  a10  = svcmla_f64_x(pg, a10,  va1, b0, 90);
        a11  = svcmla_f64_x(pg, a11,  va1, b1, 0);  a11  = svcmla_f64_x(pg, a11,  va1, b1, 90);
        va2 = svld1rq_f64(pg, a + 4);
        a20  = svcmla_f64_x(pg, a20,  va2, b0, 0);  a20  = svcmla_f64_x(pg, a20,  va2, b0, 90);
        a21  = svcmla_f64_x(pg, a21,  va2, b1, 0);  a21  = svcmla_f64_x(pg, a21,  va2, b1, 90);
        va0 = svld1rq_f64(pg, a + 6);
        a30  = svcmla_f64_x(pg, a30,  va0, b0, 0);  a30  = svcmla_f64_x(pg, a30,  va0, b0, 90);
        a31  = svcmla_f64_x(pg, a31,  va0, b1, 0);  a31  = svcmla_f64_x(pg, a31,  va0, b1, 90);
        va1 = svld1rq_f64(pg, a + 8);
        a40  = svcmla_f64_x(pg, a40,  va1, b0, 0);  a40  = svcmla_f64_x(pg, a40,  va1, b0, 90);
        a41  = svcmla_f64_x(pg, a41,  va1, b1, 0);  a41  = svcmla_f64_x(pg, a41,  va1, b1, 90);
        va2 = svld1rq_f64(pg, a + 10);
        a50  = svcmla_f64_x(pg, a50,  va2, b0, 0);  a50  = svcmla_f64_x(pg, a50,  va2, b0, 90);
        a51  = svcmla_f64_x(pg, a51,  va2, b1, 0);  a51  = svcmla_f64_x(pg, a51,  va2, b1, 90);
        va0 = svld1rq_f64(pg, a + 12);
        a60  = svcmla_f64_x(pg, a60,  va0, b0, 0);  a60  = svcmla_f64_x(pg, a60,  va0, b0, 90);
        a61  = svcmla_f64_x(pg, a61,  va0, b1, 0);  a61  = svcmla_f64_x(pg, a61,  va0, b1, 90);
        va1 = svld1rq_f64(pg, a + 14);
        a70  = svcmla_f64_x(pg, a70,  va1, b0, 0);  a70  = svcmla_f64_x(pg, a70,  va1, b0, 90);
        a71  = svcmla_f64_x(pg, a71,  va1, b1, 0);  a71  = svcmla_f64_x(pg, a71,  va1, b1, 90);
        va2 = svld1rq_f64(pg, a + 16);
        a80  = svcmla_f64_x(pg, a80,  va2, b0, 0);  a80  = svcmla_f64_x(pg, a80,  va2, b0, 90);
        a81  = svcmla_f64_x(pg, a81,  va2, b1, 0);  a81  = svcmla_f64_x(pg, a81,  va2, b1, 90);
        va0 = svld1rq_f64(pg, a + 18);
        a90  = svcmla_f64_x(pg, a90,  va0, b0, 0);  a90  = svcmla_f64_x(pg, a90,  va0, b0, 90);
        a91  = svcmla_f64_x(pg, a91,  va0, b1, 0);  a91  = svcmla_f64_x(pg, a91,  va0, b1, 90);
        va1 = svld1rq_f64(pg, a + 20);
        a100 = svcmla_f64_x(pg, a100, va1, b0, 0);  a100 = svcmla_f64_x(pg, a100, va1, b0, 90);
        a101 = svcmla_f64_x(pg, a101, va1, b1, 0);  a101 = svcmla_f64_x(pg, a101, va1, b1, 90);
        va2 = svld1rq_f64(pg, a + 22);
        a110 = svcmla_f64_x(pg, a110, va2, b0, 0);  a110 = svcmla_f64_x(pg, a110, va2, b0, 90);
        a111 = svcmla_f64_x(pg, a111, va2, b1, 0);  a111 = svcmla_f64_x(pg, a111, va2, b1, 90);
        a += MR * 2;
        b += NR * 2;
    }

    // epilogue — spill all accumulators once, then finish rows/cols in scalar C.
    double acc[MR][NGROUPS * NG * 2];
    svst1_f64(pg, &acc[0][0],   a00);  svst1_f64(pg, &acc[0][8],   a01);
    svst1_f64(pg, &acc[1][0],   a10);  svst1_f64(pg, &acc[1][8],   a11);
    svst1_f64(pg, &acc[2][0],   a20);  svst1_f64(pg, &acc[2][8],   a21);
    svst1_f64(pg, &acc[3][0],   a30);  svst1_f64(pg, &acc[3][8],   a31);
    svst1_f64(pg, &acc[4][0],   a40);  svst1_f64(pg, &acc[4][8],   a41);
    svst1_f64(pg, &acc[5][0],   a50);  svst1_f64(pg, &acc[5][8],   a51);
    svst1_f64(pg, &acc[6][0],   a60);  svst1_f64(pg, &acc[6][8],   a61);
    svst1_f64(pg, &acc[7][0],   a70);  svst1_f64(pg, &acc[7][8],   a71);
    svst1_f64(pg, &acc[8][0],   a80);  svst1_f64(pg, &acc[8][8],   a81);
    svst1_f64(pg, &acc[9][0],   a90);  svst1_f64(pg, &acc[9][8],   a91);
    svst1_f64(pg, &acc[10][0],  a100); svst1_f64(pg, &acc[10][8],  a101);
    svst1_f64(pg, &acc[11][0],  a110); svst1_f64(pg, &acc[11][8],  a111);

    for (int i = 0; i < m_rem; i++) {
        zdouble* crow = C + (long)i * ldc;
        for (int j = 0; j < n_rem; j++) {
            int g = j / NG, o = (j % NG) * 2;
            zdouble av = acc[i][g * NG * 2 + o] + acc[i][g * NG * 2 + o + 1] * I;
            zdouble t = alpha_z * av;
            zdouble ccur = crow[j];
            crow[j] = (apply_beta ? beta_z * ccur + t : ccur + t);
        }
    }
}

// pack one MR x kc block of A, zero-padded to MR rows, TIGHT layout:
// dst[(k*MR + i)*2 + {re,im}] (LD1RO expands it in-register). Asrc = &A[i0][pc].
static void pack_A_block(double* restrict dst, const zdouble* restrict Asrc,
                         BLASINT lda, int kc, int m)
{
    for (int k = 0; k < kc; k++) {
        for (int i = 0; i < MR; i++) {
            zdouble v = (i < m) ? Asrc[(long)i * lda + k] : 0.0;
            dst[(k * MR + i) * 2]     = creal(v);
            dst[(k * MR + i) * 2 + 1] = cimag(v);
        }
    }
    // k-tail: NO zero-fill needed. The microkernel is kc-parameterized (its
    // k-loop runs only to kc), so rows [kc,KC) of the final partial panel are
    // never read — stale bytes from the previous panel's pack are harmless.
    // This halves pack traffic when kc == KC/2 (the K=64 sub-problems of the
    // 2-level recursion), where pack cost otherwise equals compute cost.
}

// pack one kc x NR block of B, zero-padded to NR cols. Bsrc = &B[pc][j0].
static void pack_B_block(double* restrict dst, const zdouble* restrict Bsrc,
                         BLASINT ldb, int kc, int n)
{
    for (int k = 0; k < kc; k++) {
        for (int j = 0; j < NR; j++) {
            zdouble v = (j < n) ? Bsrc[(long)k * ldb + j] : 0.0;
            dst[(k * NR + j) * 2]     = creal(v);
            dst[(k * NR + j) * 2 + 1] = cimag(v);
        }
    }
    // k-tail: no zero-fill needed (see pack_A_block).
}

// ---------------- core (extracted) ----------------
static void gemm_core(const zdouble* Ap, const zdouble* Bp, zdouble* Cp,
                      BLASINT M, BLASINT N, BLASINT K,
                      zdouble alpha_val, zdouble beta_val,
                      BLASINT lda, BLASINT ldb, BLASINT ldc)
{
    alpha_z = alpha_val; beta_z = beta_val;

        // ---- NUMA-aware ROW partition (v4) ----
        // The bench first-touches A and C row-parallel (libgomp static for =>
        // thread t owns the contiguous row range [t*chunk,(t+1)*chunk)).
        // Threads are close-bound over the allowed CPUs in ascending order, so
        // each NUMA node owns a contiguous row range of every matrix. Assigning
        // MC row-blocks to their first-touch node keeps A packs and C epilogue
        // writes node-local; B is read-only shared. Threads steal tiles from
        // other nodes once their own list runs dry (fixes orphan tiles at small
        // N and tail imbalance).
        static int cpu_node[608];
        static int ncpu[32];
        static int allowed_cpus[608];
        static int nallowed = 0;
        static int nnodes = 0;
        static int topo_done = 0;
        if (!topo_done) {
            cpu_set_t allowed;
            CPU_ZERO(&allowed);
            if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
                for (int nd = 0; nd < 32; nd++) {
                    char path[96];
                    snprintf(path, sizeof(path),
                             "/sys/devices/system/node/node%d/cpulist", nd);
                    FILE* g = fopen(path, "r");
                    if (!g) break;
                    char buf[512];
                    if (fgets(buf, sizeof(buf), g)) {
                        char* s = buf;
                        for (;;) {
                            int a = -1, b = -1, consumed = 0;
                            if (sscanf(s, "%d-%d%n", &a, &b, &consumed) == 2) {
                            } else if (sscanf(s, "%d%n", &a, &consumed) == 1) {
                                b = a;
                            } else {
                                break;
                            }
                            for (int c = a; c <= b && c < 608; c++)
                                if (CPU_ISSET(c, &allowed)) cpu_node[c] = nd;
                            s += consumed;
                            if (*s == ',') s++; else break;
                        }
                    }
                    fclose(g);
                }
                for (int c = 0; c < 608; c++)
                    if (cpu_node[c] >= 0) {
                        ncpu[cpu_node[c]]++;
                        allowed_cpus[nallowed++] = c;   // ascending CPU order
                    }
                for (int nd = 0; nd < 32; nd++)
                    if (ncpu[nd] > 0) nnodes++;
            }
            topo_done = 1;
        }

        const int njc = (N + NC - 1) / NC;
        const int nic = (M + MC - 1) / MC;

        // contiguous per-node ic-block ranges (indexed by RAW node id)
        int bstart[32], bcnt[32];
        for (int q = 0; q < 32; q++) { bstart[q] = 0; bcnt[q] = 0; }
        int usable = 0;
        if (nnodes > 1 && nallowed > 0) {
            // first-touch node of each block: mirror the bench's libgomp
            // static-for chunking (thread t owns rows [t*chunk,(t+1)*chunk))
            int home[nic];
            int ok = 1;
            int T = omp_get_max_threads();
            if (T < 1) T = 1;
            if (T > nallowed) T = nallowed;
            long chunk = ((long)M + T - 1) / T;
            if (chunk < 1) chunk = 1;
            for (int b = 0; b < nic; b++) {
                long t = ((long)b * MC) / chunk;
                if (t >= T) t = T - 1;
                int nd = cpu_node[allowed_cpus[t]];
                if (nd < 0) { ok = 0; break; }
                home[b] = nd;
            }
            for (int b = 1; b < nic && ok; b++)
                if (home[b] < home[b - 1]) ok = 0;   // first-touch is contiguous
            if (ok) {
                int cur = home[0];
                bstart[cur] = 0;
                for (int b = 0; b < nic; b++) {
                    if (home[b] != cur) {
                        bcnt[cur] = b - bstart[cur];
                        cur = home[b];
                        bstart[cur] = b;
                    }
                }
                bcnt[cur] = nic - bstart[cur];
                usable = 1;
            }
        }
        if (!usable) {
            // proportional contiguous split by allowed CPU count per node
            int total = 0;
            for (int nd = 0; nd < 32; nd++) total += ncpu[nd];
            if (nnodes <= 0 || total <= 0) {
                bstart[0] = 0; bcnt[0] = nic;    // no topology: one global list
            } else {
                int idx = 0, placed = 0;
                for (int nd = 0; nd < 32; nd++) {
                    if (ncpu[nd] == 0) continue;
                    int share = (placed == nnodes - 1) ? nic - idx
                              : (int)((long)nic * ncpu[nd] / total);
                    if (share < 0) share = 0;
                    bstart[nd] = idx; bcnt[nd] = share;
                    idx += share; placed++;
                }
                if (idx < nic) {
                    for (int nd = 31; nd >= 0; nd--)
                        if (ncpu[nd] > 0) { bcnt[nd] += nic - idx; break; }
                }
            }
        }

        // per-node atomic tile counters (tile = (ic block, jc) pair)
        _Atomic int tile_next[32];
        for (int q = 0; q < 32; q++) atomic_init(&tile_next[q], 0);

        #pragma omp parallel
        {
            const size_t a_blk = (size_t)KC * MR * 2;
            const size_t b_blk = (size_t)KC * NR * 2;
            const int na = (MC + MR - 1) / MR;
            const int nb = (NC + NR - 1) / NR;
            double* pA = (double*)aligned_alloc(64, na * a_blk * sizeof(double));
            double* pB = (double*)aligned_alloc(64, nb * b_blk * sizeof(double));

            // this thread's node (RAW id); fallback: split threads by count
            int mynode = -1;
            {
                int c = sched_getcpu();
                if (c >= 0 && c < 608 && cpu_node[c] >= 0)
                    mynode = cpu_node[c];
                if (mynode < 0) {
                    if (nnodes <= 0) {
                        mynode = 0;
                    } else {
                        int per = (omp_get_num_threads() + nnodes - 1) / nnodes;
                        if (per < 1) per = 1;
                        int p = omp_get_thread_num() / per;
                        int nd = -1;
                        for (int q = 0; q < 32; q++) {
                            if (ncpu[q] > 0) {
                                if (p == 0) { nd = q; break; }
                                p--;
                            }
                        }
                        mynode = (nd >= 0) ? nd : 0;
                    }
                }
            }

            for (;;) {
                int h = mynode;
                long idx = atomic_fetch_add(&tile_next[h], 1);
                if (idx >= (long)bcnt[h] * njc) {
                    // own list dry -> steal from any other node with tiles
                    int got = 0;
                    for (int q = 0; q < 32 && !got; q++) {
                        if (q == h || bcnt[q] == 0) continue;
                        long j = atomic_fetch_add(&tile_next[q], 1);
                        if (j < (long)bcnt[q] * njc) { h = q; idx = j; got = 1; }
                    }
                    if (!got) break;   // all lists dry
                }
                const int blk = bstart[h] + (int)(idx / njc);
                const int jc_i = (int)(idx % njc);
                const BLASINT ic = (BLASINT)blk * MC;
                const BLASINT jc = (BLASINT)jc_i * NC;
                const int ncb = imin(N - jc, NC);
                const int mcb = imin(M - ic, MC);
                const int nib = (mcb + MR - 1) / MR;
                const int njb = (ncb + NR - 1) / NR;

                for (BLASINT pc = 0; pc < K; pc += KC) {
                    const int kc = imin(K - pc, KC);
                    const int first = (pc == 0);

                    for (int jb = 0; jb < njb; jb++)
                        pack_B_block(pB + jb * b_blk,
                                     Bp + (long)pc * ldb + jc + (BLASINT)jb * NR,
                                     ldb, kc, ncb - jb * NR);

                    for (int ib = 0; ib < nib; ib++)
                        pack_A_block(pA + ib * a_blk,
                                     Ap + (long)(ic + ib * MR) * lda + pc,
                                     lda, kc, mcb - ib * MR);

                    for (int ib = 0; ib < nib; ib++) {
                        const int m_rem = imin(mcb - ib * MR, MR);
                        const double* pa = pA + ib * a_blk;
                        zdouble* cbase = Cp + (long)(ic + ib * MR) * ldc + jc;
                        for (int jb = 0; jb < njb; jb++) {
                            const int n_rem = imin(ncb - jb * NR, NR);
                            microkernel(pa, pB + jb * b_blk,
                                        cbase + (long)jb * NR, ldc, m_rem, n_rem,
                                        first, kc);
                        }
                    }
                }
            }
            free(pA); free(pB);
        }
        return;
}

// ---------------- Strassen level 1 ----------------
// One Strassen step: 7 half-size products instead of 8, via gemm_core.
// Odd dims: quadrants are ceil/floor halves; shorter operands read as 0 via
// clamped reads, so every shape combination is correct by construction.
// alpha/beta of the caller are folded into the merge.
#ifndef STRASSEN_DEPTH
#define STRASSEN_DEPTH 2     // 2 = two levels (49/64 work), 1 = one level
#endif
#define STRASSEN_MIN_M 2048
#define STRASSEN_MIN_N 2048
#define STRASSEN_MIN_K 128

// ---- optional per-level profiling (STRASSEN_PROFILE compiles it in) ----
#ifdef STRASSEN_PROFILE
#include <time.h>
static double str_t_gather, str_t_add, str_t_prod[2], str_t_merge;
static int    str_n_prod[2];
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
                            return ts.tv_sec + 1e-9 * ts.tv_nsec; }
void strassen_report(void)
{
    fprintf(stderr, "[str] gather=%.3fs add=%.3fs merge=%.3fs "
                    "prod(L1)=%.3fs/%d prod(L2)=%.3fs/%d\n",
            str_t_gather, str_t_add, str_t_merge,
            str_t_prod[0], str_n_prod[0], str_t_prod[1], str_n_prod[1]);
}
#else
#define strassen_report() ((void)0)
#endif


static void strassen_level(const zdouble* Ap, const zdouble* Bp, zdouble* Cp,
                           BLASINT M, BLASINT N, BLASINT K,
                           zdouble alpha, zdouble beta,
                           BLASINT lda, BLASINT ldb, BLASINT ldc, int depth);

// sub-product dispatch: recurse one more Strassen level when the sub-problem
// is big enough (and depth remains), else plain gemm_core.
static void strassen_product(const zdouble* S, const zdouble* T, zdouble* P,
                             BLASINT pm, BLASINT pn, BLASINT pk,
                             BLASINT lds, BLASINT ldt, BLASINT ldp, int depth)
{
{
#ifdef STRASSEN_PROFILE
    double _tp0 = now_s();
    int _rec = (depth > 0 && pm >= STRASSEN_MIN_M && pn >= STRASSEN_MIN_N
                && pk >= STRASSEN_MIN_K);
#endif
    if (depth > 0 && pm >= STRASSEN_MIN_M && pn >= STRASSEN_MIN_N && pk >= STRASSEN_MIN_K)
        strassen_level(S, T, P, pm, pn, pk, 1.0, 0.0, lds, ldt, ldp, depth - 1);
    else
        gemm_core(S, T, P, pm, pn, pk, 1.0, 0.0, lds, ldt, ldp);
#ifdef STRASSEN_PROFILE
    str_t_prod[_rec ? 1 : 0] += now_s() - _tp0;
    str_n_prod[_rec ? 1 : 0]++;
#endif
}
}

static void strassen_level(const zdouble* Ap, const zdouble* Bp, zdouble* Cp,
                           BLASINT M, BLASINT N, BLASINT K,
                           zdouble alpha, zdouble beta,
                           BLASINT lda, BLASINT ldb, BLASINT ldc, int depth)
{
    const BLASINT mh = (M + 1) / 2, m2 = M - mh;      // mh = ceil(M/2)
    const BLASINT nh = (N + 1) / 2, n2 = N - nh;      // nh = ceil(N/2)
    const BLASINT kh = (K + 1) / 2, k2 = K - kh;      // kh = ceil(K/2)

    // ---- gather the 8 quadrants as dense matrices (row-parallel memcpys).
    // Odd tails: the "second" quadrant of each dim is simply one row/col
    // shorter (m2/nh etc. already encode it); pads appear as clamped reads
    // in the S/T formation below.
    zdouble *A11 = aligned_alloc(64, (size_t)mh * kh * sizeof(zdouble));
    zdouble *A12 = aligned_alloc(64, (size_t)mh * k2 * sizeof(zdouble));
    zdouble *A21 = aligned_alloc(64, (size_t)m2 * kh * sizeof(zdouble));
    zdouble *A22 = aligned_alloc(64, (size_t)m2 * k2 * sizeof(zdouble));
    zdouble *B11 = aligned_alloc(64, (size_t)kh * nh * sizeof(zdouble));
    zdouble *B12 = aligned_alloc(64, (size_t)kh * n2 * sizeof(zdouble));
    zdouble *B21 = aligned_alloc(64, (size_t)k2 * nh * sizeof(zdouble));
    zdouble *B22 = aligned_alloc(64, (size_t)k2 * n2 * sizeof(zdouble));
    if (!A11 || !A12 || !A21 || !A22 || !B11 || !B12 || !B21 || !B22) {
        free(A11); free(A12); free(A21); free(A22);
        free(B11); free(B12); free(B21); free(B22);
        if (depth > 0)
            strassen_level(Ap, Bp, Cp, M, N, K, alpha, beta, lda, ldb, ldc, depth - 1);
        else
            gemm_core(Ap, Bp, Cp, M, N, K, alpha, beta, lda, ldb, ldc);
        return;
    }
#ifdef STRASSEN_PROFILE
    double _tg0 = now_s();
#endif
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (BLASINT i = 0; i < mh; i++)
            memcpy(A11 + (size_t)i * kh, Ap + (size_t)i * lda,
                   (size_t)kh * sizeof(zdouble));
        #pragma omp for nowait
        for (BLASINT i = 0; i < mh; i++)
            memcpy(A12 + (size_t)i * k2, Ap + (size_t)i * lda + kh,
                   (size_t)k2 * sizeof(zdouble));
        #pragma omp for nowait
        for (BLASINT i = 0; i < m2; i++)
            memcpy(A21 + (size_t)i * kh, Ap + (size_t)(i + mh) * lda,
                   (size_t)kh * sizeof(zdouble));
        #pragma omp for nowait
        for (BLASINT i = 0; i < m2; i++)
            memcpy(A22 + (size_t)i * k2, Ap + (size_t)(i + mh) * lda + kh,
                   (size_t)k2 * sizeof(zdouble));
        #pragma omp for nowait
        for (BLASINT i = 0; i < kh; i++)
            memcpy(B11 + (size_t)i * nh, Bp + (size_t)i * ldb,
                   (size_t)nh * sizeof(zdouble));
        #pragma omp for nowait
        for (BLASINT i = 0; i < kh; i++)
            memcpy(B12 + (size_t)i * n2, Bp + (size_t)i * ldb + nh,
                   (size_t)n2 * sizeof(zdouble));
        #pragma omp for nowait
        for (BLASINT i = 0; i < k2; i++)
            memcpy(B21 + (size_t)i * nh, Bp + (size_t)(i + kh) * ldb,
                   (size_t)nh * sizeof(zdouble));
        #pragma omp for nowait
        for (BLASINT i = 0; i < k2; i++)
            memcpy(B22 + (size_t)i * n2, Bp + (size_t)(i + kh) * ldb + nh,
                   (size_t)n2 * sizeof(zdouble));
    }
#ifdef STRASSEN_PROFILE
    str_t_gather += now_s() - _tg0;
#endif

    // ---- 7 products (classic Strassen, formulas verified numerically) ----
    //   P1 = (A11+A22)@(B11+B22)  P2 = (A21+A22)@B11   P3 = A11@(B12-B22)
    //   P4 = A22@(B21-B11)        P5 = (A11+A12)@B22   P6 = (A21-A11)@(B11+B12)
    //   P7 = (A12-A22)@(B21+B22)
    // Shapes (odd-dim safe, validated by numpy transcription):
    //   P1 = mh x nh   P3/P5 = mh x n2   P2/P4/P6 = m2 x nh   P7 = mh x nh
    // P1/P7 must cover the FULL C11 quadrant (mh x nh): with odd N the extra
    // C11 column n2 has no B22 counterpart, with odd M the extra C11 row has
    // no A22 — the clamps below zero those operands instead.
    size_t szP1 = (size_t)mh * nh, szP35 = (size_t)mh * n2, szP2 = (size_t)m2 * nh,
         szP7 = (size_t)mh * nh;
    zdouble *P1 = aligned_alloc(64, szP1 * sizeof(zdouble));
    zdouble *P2 = aligned_alloc(64, szP2 * sizeof(zdouble));
    zdouble *P3 = aligned_alloc(64, szP35 * sizeof(zdouble));
    zdouble *P4 = aligned_alloc(64, szP2 * sizeof(zdouble));
    zdouble *P5 = aligned_alloc(64, szP35 * sizeof(zdouble));
    zdouble *P6 = aligned_alloc(64, szP2 * sizeof(zdouble));
    zdouble *P7 = aligned_alloc(64, szP7 * sizeof(zdouble));
    if (!P1 || !P2 || !P3 || !P4 || !P5 || !P6 || !P7) {
        zdouble* ps[7] = {P1, P2, P3, P4, P5, P6, P7};
        for (int q = 0; q < 7; q++) free(ps[q]);
        free(A11); free(A12); free(A21); free(A22);
        free(B11); free(B12); free(B21); free(B22);
        if (depth > 0)
            strassen_level(Ap, Bp, Cp, M, N, K, alpha, beta, lda, ldb, ldc, depth - 1);
        else
            gemm_core(Ap, Bp, Cp, M, N, K, alpha, beta, lda, ldb, ldc);
        return;
    }

    // P1 = (A11+A22) @ (B11+B22)     [mh x nh, K=kh]
    {
        zdouble* S = aligned_alloc(64, (size_t)mh * kh * sizeof(zdouble));
        zdouble* T = aligned_alloc(64, (size_t)kh * nh * sizeof(zdouble));
        if (S && T) {
            #ifdef STRASSEN_PROFILE
            double _ta0 = now_s();
            #endif
            #pragma omp parallel for schedule(static)
            for (BLASINT i = 0; i < mh; i++) {
                const zdouble* a1r = A11 + (size_t)i * kh;
                const zdouble* a2r = (i < m2) ? A22 + (size_t)i * k2 : NULL;
                zdouble* srow = S + (size_t)i * kh;
                for (BLASINT k = 0; k < kh; k++)
                    srow[k] = a1r[k] + ((a2r && k < k2) ? a2r[k] : 0.0);
            }
            #pragma omp parallel for schedule(static)
            for (BLASINT k = 0; k < kh; k++) {
                const zdouble* b1r = B11 + (size_t)k * nh;
                const zdouble* b2r = (k < k2) ? B22 + (size_t)k * n2 : NULL;
                zdouble* trow = T + (size_t)k * nh;
                for (BLASINT j = 0; j < nh; j++)
                    trow[j] = b1r[j] + ((b2r && j < n2) ? b2r[j] : 0.0);
            }
            #ifdef STRASSEN_PROFILE
            str_t_add += now_s() - _ta0;
            #endif

            strassen_product(S, T, P1, mh, nh, kh, kh, nh, nh, depth);
        }
        free(S); free(T);
    }

    // P2 = (A21+A22) @ B11          [m2 x nh, K=kh]
    {
        zdouble* S = aligned_alloc(64, (size_t)m2 * kh * sizeof(zdouble));
        if (S) {
            #ifdef STRASSEN_PROFILE
            double _ta0 = now_s();
            #endif
            #pragma omp parallel for schedule(static)
            for (BLASINT i = 0; i < m2; i++) {
                const zdouble* a1r = A21 + (size_t)i * kh;
                const zdouble* a2r = (i < m2) ? A22 + (size_t)i * k2 : NULL;
                zdouble* srow = S + (size_t)i * kh;
                for (BLASINT k = 0; k < kh; k++)
                    srow[k] = a1r[k] + ((a2r && k < k2) ? a2r[k] : 0.0);
            }
            #ifdef STRASSEN_PROFILE
            str_t_add += now_s() - _ta0;
            #endif

            strassen_product(S, B11, P2, m2, nh, kh, kh, nh, nh, depth);
        }
        free(S);
    }

    // P3 = A11 @ (B12 - B22)        [mh x n2, K=kh]
    {
        zdouble* T = aligned_alloc(64, (size_t)kh * n2 * sizeof(zdouble));
        if (T) {
            #ifdef STRASSEN_PROFILE
            double _ta0 = now_s();
            #endif
            #pragma omp parallel for schedule(static)
            for (BLASINT k = 0; k < kh; k++) {
                const zdouble* b1r = B12 + (size_t)k * n2;   // B12 has kh rows
                const zdouble* b2r = (k < k2) ? B22 + (size_t)k * n2 : NULL;
                zdouble* trow = T + (size_t)k * n2;
                for (BLASINT j = 0; j < n2; j++)
                    trow[j] = b1r[j] - (b2r ? b2r[j] : 0.0);
            }
            #ifdef STRASSEN_PROFILE
            str_t_add += now_s() - _ta0;
            #endif

            strassen_product(A11, T, P3, mh, n2, kh, kh, n2, n2, depth);
        }
        free(T);
    }

    // P4 = A22 @ (B21 - B11)        [m2 x nh, K=k2]
    {
        zdouble* T = aligned_alloc(64, (size_t)k2 * nh * sizeof(zdouble));
        if (T) {
            #ifdef STRASSEN_PROFILE
            double _ta0 = now_s();
            #endif
            #pragma omp parallel for schedule(static)
            for (BLASINT k = 0; k < k2; k++) {
                const zdouble* b1r = B21 + (size_t)k * nh;
                const zdouble* b2r = (k < kh) ? B11 + (size_t)k * nh : NULL;
                zdouble* trow = T + (size_t)k * nh;
                for (BLASINT j = 0; j < nh; j++)
                    trow[j] = b1r[j] - (b2r ? b2r[j] : 0.0);
            }
            #ifdef STRASSEN_PROFILE
            str_t_add += now_s() - _ta0;
            #endif

            strassen_product(A22, T, P4, m2, nh, k2, k2, nh, nh, depth);
        }
        free(T);
    }

    // P5 = (A11 + A12) @ B22        [mh x n2, K=k2]
    {
        zdouble* S = aligned_alloc(64, (size_t)mh * k2 * sizeof(zdouble));
        if (S) {
            #ifdef STRASSEN_PROFILE
            double _ta0 = now_s();
            #endif
            #pragma omp parallel for schedule(static)
            for (BLASINT i = 0; i < mh; i++) {
                const zdouble* a1r = A11 + (size_t)i * kh;
                const zdouble* a2r = A12 + (size_t)i * k2;
                zdouble* srow = S + (size_t)i * k2;
                for (BLASINT k = 0; k < k2; k++)
                    srow[k] = a1r[k] + a2r[k];
            }
            #ifdef STRASSEN_PROFILE
            str_t_add += now_s() - _ta0;
            #endif

            strassen_product(S, B22, P5, mh, n2, k2, k2, n2, n2, depth);
        }
        free(S);
    }

    // P6 = (A21 - A11) @ (B11 + B12)   [m2 x nh, K=kh]
    {
        zdouble* S = aligned_alloc(64, (size_t)m2 * kh * sizeof(zdouble));
        zdouble* T = aligned_alloc(64, (size_t)kh * nh * sizeof(zdouble));
        if (S && T) {
            #ifdef STRASSEN_PROFILE
            double _ta0 = now_s();
            #endif
            #pragma omp parallel for schedule(static)
            for (BLASINT i = 0; i < m2; i++) {
                const zdouble* a1r = A21 + (size_t)i * kh;
                const zdouble* a2r = A11 + (size_t)i * kh;   // i < m2 <= mh
                zdouble* srow = S + (size_t)i * kh;
                for (BLASINT k = 0; k < kh; k++)
                    srow[k] = a1r[k] - (a2r ? a2r[k] : 0.0);
            }
            #pragma omp parallel for schedule(static)
            for (BLASINT k = 0; k < kh; k++) {
                const zdouble* b1r = B11 + (size_t)k * nh;
                const zdouble* b2r = B12 + (size_t)k * n2;   // B12 has kh rows
                zdouble* trow = T + (size_t)k * nh;
                for (BLASINT j = 0; j < nh; j++)
                    trow[j] = b1r[j] + ((j < n2) ? b2r[j] : 0.0);
            }
            #ifdef STRASSEN_PROFILE
            str_t_add += now_s() - _ta0;
            #endif

            strassen_product(S, T, P6, m2, nh, kh, kh, nh, nh, depth);
        }
        free(S); free(T);
    }

    // P7 = (A12 - A22) @ (B21 + B22)   [mh x nh, K=k2]
    {
        zdouble* S = aligned_alloc(64, (size_t)mh * k2 * sizeof(zdouble));
        zdouble* T = aligned_alloc(64, (size_t)k2 * nh * sizeof(zdouble));
        if (S && T) {
            #ifdef STRASSEN_PROFILE
            double _ta0 = now_s();
            #endif
            #pragma omp parallel for schedule(static)
            for (BLASINT i = 0; i < mh; i++) {
                const zdouble* a1r = A12 + (size_t)i * k2;
                const zdouble* a2r = (i < m2) ? A22 + (size_t)i * k2 : NULL;
                zdouble* srow = S + (size_t)i * k2;
                for (BLASINT k = 0; k < k2; k++)
                    srow[k] = a1r[k] - (a2r ? a2r[k] : 0.0);
            }
            #pragma omp parallel for schedule(static)
            for (BLASINT k = 0; k < k2; k++) {
                const zdouble* b1r = B21 + (size_t)k * nh;
                const zdouble* b2r = B22 + (size_t)k * n2;
                zdouble* trow = T + (size_t)k * nh;
                for (BLASINT j = 0; j < nh; j++)
                    trow[j] = b1r[j] + ((j < n2) ? b2r[j] : 0.0);
            }
            #ifdef STRASSEN_PROFILE
            str_t_add += now_s() - _ta0;
            #endif

            strassen_product(S, T, P7, mh, nh, k2, k2, nh, nh, depth);
        }
        free(S); free(T);
    }

    // ---- merge quadrants into C (alpha/beta folded here) ----
    //   C11 = P1 + P4 - P5 + P7      C12 = P3 + P5
    //   C21 = P2 + P4                C22 = P1 - P2 + P3 + P6
#ifdef STRASSEN_PROFILE
    double _tm0 = now_s();
#endif
    #pragma omp parallel for schedule(static)
    for (BLASINT i = 0; i < mh; i++) {
        zdouble* c11 = Cp + (size_t)i * ldc;
        zdouble* c12 = c11 + nh;
        const zdouble* p1 = P1 + (size_t)i * nh;
        const zdouble* p3 = P3 + (size_t)i * n2;
        const zdouble* p4 = (k2 > 0 && i < m2) ? P4 + (size_t)i * nh : NULL;
        // k2 == 0 (K == 1) leaves P5/P7 unwritten by gemm_core — treat as 0.
        const zdouble* p5 = (k2 > 0) ? P5 + (size_t)i * n2 : NULL;
        const zdouble* p7 = (k2 > 0) ? P7 + (size_t)i * nh : NULL;
        for (BLASINT j = 0; j < nh; j++) {
            zdouble v = p1[j] + (p4 ? p4[j] : 0.0)
                      - ((p5 && j < n2) ? p5[j] : 0.0) + (p7 ? p7[j] : 0.0);
            c11[j] = alpha * v + beta * c11[j];
        }
        for (BLASINT j = 0; j < n2; j++) {
            zdouble v = p3[j] + (p5 ? p5[j] : 0.0);
            c12[j] = alpha * v + beta * c12[j];
        }
    }
    #pragma omp parallel for schedule(static)
    for (BLASINT i = 0; i < m2; i++) {
        zdouble* c21 = Cp + (size_t)(i + mh) * ldc;
        zdouble* c22 = c21 + nh;
        const zdouble* p1 = P1 + (size_t)i * nh;
        const zdouble* p2 = P2 + (size_t)i * nh;
        const zdouble* p3 = P3 + (size_t)i * n2;   // P3 is K=kh: always computed
        const zdouble* p4 = (k2 > 0) ? P4 + (size_t)i * nh : NULL;
        const zdouble* p6 = P6 + (size_t)i * nh;
        for (BLASINT j = 0; j < nh; j++) {
            zdouble v = p2[j] + (p4 ? p4[j] : 0.0);
            c21[j] = alpha * v + beta * c21[j];
        }
        for (BLASINT j = 0; j < n2; j++) {
            zdouble v = p1[j] - p2[j] + p3[j] + p6[j];
            c22[j] = alpha * v + beta * c22[j];
        }
    }

#ifdef STRASSEN_PROFILE
    str_t_merge += now_s() - _tm0;
#endif
    free(P1); free(P2); free(P3); free(P4); free(P5); free(P6); free(P7);
    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
}

void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                 const BLASINT K, const void* alpha, const void* A, const BLASINT lda,
                 const void* B, const BLASINT ldb, const void* beta, void* C,
                 const BLASINT ldc)
{
    zdouble alpha_val = *(const zdouble*)alpha;
    zdouble beta_val  = *(const zdouble*)beta;
    const zdouble* Ap = (const zdouble*)A;
    const zdouble* Bp = (const zdouble*)B;
    zdouble* Cp = (zdouble*)C;

    // ---------- fast path: RowMajor, NoTrans/NoTrans ----------
    if (Order == CblasRowMajor && TransA == CblasNoTrans && TransB == CblasNoTrans) {
        // Strassen level-1 for large problems (7/8 work).  Gate keeps it off
        // for small/medium sizes where the extra traffic would dominate.
        if (M >= STRASSEN_MIN_M && N >= STRASSEN_MIN_N && K >= STRASSEN_MIN_K) {
            strassen_level(Ap, Bp, Cp, M, N, K, alpha_val, beta_val,
                           lda, ldb, ldc, STRASSEN_DEPTH);
            strassen_report();
        } else
            gemm_core(Ap, Bp, Cp, M, N, K, alpha_val, beta_val, lda, ldb, ldc);
        return;
    }

    // ---------- fallback: naive (correctness for other paths) ----------
    if (Order == CblasRowMajor) {
        if (beta_val != 1.0)
            for (BLASINT i = 0; i < M; i++)
                for (BLASINT j = 0; j < N; j++)
                    Cp[i * ldc + j] *= beta_val;
        #pragma omp parallel for
        for (BLASINT i = 0; i < M; i++)
            for (BLASINT j = 0; j < N; j++) {
                zdouble sum = 0.0;
                for (BLASINT k = 0; k < K; k++) {
                    zdouble a = (TransA == CblasNoTrans) ? Ap[i * lda + k]
                              : (TransA == CblasTrans) ? Ap[k * lda + i]
                              : conj(Ap[k * lda + i]);
                    zdouble b = (TransB == CblasNoTrans) ? Bp[k * ldb + j]
                              : (TransB == CblasTrans) ? Bp[j * ldb + k]
                              : conj(Bp[j * ldb + k]);
                    sum += a * b;
                }
                Cp[i * ldc + j] = alpha_val * sum + Cp[i * ldc + j];
            }
    } else {
        if (beta_val != 1.0)
            for (BLASINT i = 0; i < M; i++)
                for (BLASINT j = 0; j < N; j++)
                    Cp[j * ldc + i] *= beta_val;
        #pragma omp parallel for
        for (BLASINT i = 0; i < M; i++)
            for (BLASINT j = 0; j < N; j++) {
                zdouble sum = 0.0;
                for (BLASINT k = 0; k < K; k++) {
                    zdouble a = (TransA == CblasNoTrans) ? Ap[k * lda + i]
                              : (TransA == CblasTrans) ? Ap[i * lda + k]
                              : conj(Ap[i * lda + k]);
                    zdouble b = (TransB == CblasNoTrans) ? Bp[j * ldb + k]
                              : (TransB == CblasTrans) ? Bp[k * ldb + j]
                              : conj(Bp[k * ldb + j]);
                    sum += a * b;
                }
                Cp[j * ldc + i] = alpha_val * sum + Cp[j * ldc + i];
            }
    }
}
