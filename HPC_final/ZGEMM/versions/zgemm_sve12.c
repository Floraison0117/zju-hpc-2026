// ZGEMM for Kunpeng 920 (Shenzhen SC, SVE VL=512) — SVE MR12 microkernel
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

#include <arm_sve.h>
#include <omp.h>
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
#define NC 512
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
                                int m_rem, int n_rem, int apply_beta)
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
    for (int k = 0; k < KC; k++) {
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
}

// ---------------- top level ----------------
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
        alpha_z = alpha_val; beta_z = beta_val;

        const int njc = (N + NC - 1) / NC;
        const int nic = (M + MC - 1) / MC;
        const long ntiles = (long)njc * (long)nic;

        #pragma omp parallel
        {
            const size_t a_blk = (size_t)KC * MR * 2;             // doubles per A i-block
            const size_t b_blk = (size_t)KC * NR * 2;              // doubles per B j-block
            const int na = (MC + MR - 1) / MR;
            const int nb = (NC + NR - 1) / NR;
            double* pA = (double*)aligned_alloc(64, na * a_blk * sizeof(double));
            double* pB = (double*)aligned_alloc(64, nb * b_blk * sizeof(double));

            #pragma omp for schedule(dynamic, 1)
            for (long t = 0; t < ntiles; t++) {
                const BLASINT jc = (BLASINT)(t / nic) * NC;
                const BLASINT ic = (BLASINT)(t % nic) * MC;
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
                                        first);
                        }
                    }
                }
            }
            free(pA); free(pB);
        }
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
