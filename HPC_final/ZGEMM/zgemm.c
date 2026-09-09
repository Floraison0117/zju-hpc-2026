// ZGEMM for Kunpeng 920 (TaiShan-v120, FCMLA) — submission version
//
// Strategy: blocked GEMM with panel packing + FCMLA microkernel.
//   - MR x NR x KC microkernel using vcmlaq_f64 / vcmlaq_rot90_f64
//     (2 FCMLA instructions per complex MAC = 8 flops).
//   - Full-panel packing: A panels [MC x KC], B panels [KC x NC],
//     zero-padded to MR/NR so the microkernel always runs full width.
//   - beta folded into the FIRST k-block epilogue:
//       pc == 0:  C = beta*C + alpha*acc
//       pc >  0:  C = C + alpha*acc
//     (no separate full-matrix beta pass; C is up to 4.9 GB)
//   - 2D tile parallelism (M x N macro tiles), dynamic schedule,
//     per-thread packed buffers.
//   - Epilogue multiplies by alpha/beta with FCMLA pairs.
//
// The official build command is `gcc -O3 bench_zgemm.c zgemm.c -o zgemm_test
// -lm -fopenmp` (no -march flag), so the FCMLA target is enabled here.

#include <arm_neon.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

#pragma GCC target("arch=armv8.3-a")

typedef int BLASINT;
typedef double _Complex zdouble;

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

// ---------------- tunables ----------------
#ifndef MR
#define MR 6
#endif
#ifndef NR
#define NR 2
#endif
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

// ---------------------------------------------------------------------------
// Microkernel: C[MR x NR] (+)= alpha * (A[MR x KC] * B[KC x NR])
//   apply_beta != 0 : C = beta*C + alpha*acc   (first k-block)
//   apply_beta == 0 : C = C + alpha*acc        (later k-blocks)
// pA layout: [(k*MR + i)*2 + {re,im}]
// pB layout: [(k*NR + j)*2 + {re,im}]
// ---------------------------------------------------------------------------
static inline void microkernel(const double* restrict pA, const double* restrict pB,
                               zdouble* restrict C, BLASINT ldc,
                               int m_rem, int n_rem,
                               float64x2_t valpha, float64x2_t vbeta,
                               int apply_beta)
{
    float64x2_t acc[MR][NR];
    for (int i = 0; i < MR; i++)
        for (int j = 0; j < NR; j++)
            acc[i][j] = vdupq_n_f64(0.0);

    const double* a = pA;
    const double* b = pB;
    for (int k = 0; k < KC; k++) {
        float64x2_t va[MR], vb[NR];
        for (int i = 0; i < MR; i++) va[i] = vld1q_f64(a + i * 2);
        for (int j = 0; j < NR; j++) vb[j] = vld1q_f64(b + j * 2);
        for (int i = 0; i < MR; i++)
            for (int j = 0; j < NR; j++) {
                acc[i][j] = vcmlaq_f64(acc[i][j], va[i], vb[j]);
                acc[i][j] = vcmlaq_rot90_f64(acc[i][j], va[i], vb[j]);
            }
        a += MR * 2;
        b += NR * 2;
    }

    for (int i = 0; i < m_rem; i++) {
        zdouble* crow = C + (long)i * ldc;
        for (int j = 0; j < n_rem; j++) {
            // t = alpha * acc  (FCMLA #0 + #90 from zero)
            float64x2_t t = vcmlaq_f64(vdupq_n_f64(0.0), acc[i][j], valpha);
            t = vcmlaq_rot90_f64(t, acc[i][j], valpha);
            float64x2_t c;
            if (apply_beta) {
                // c = beta * C
                float64x2_t cb = vld1q_f64((const double*)(crow + j));
                float64x2_t s = vcmlaq_f64(vdupq_n_f64(0.0), cb, vbeta);
                s = vcmlaq_rot90_f64(s, cb, vbeta);
                c = vaddq_f64(s, t);
            } else {
                c = vaddq_f64(vld1q_f64((const double*)(crow + j)), t);
            }
            vst1q_f64((double*)(crow + j), c);
        }
    }
}

// pack one MR x kc block of A, zero-padded to MR rows.  Asrc = &A[i0][pc].
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

// pack one kc x NR block of B, zero-padded to NR cols.  Bsrc = &B[pc][j0].
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
        double apack[2] = { creal(alpha_val), cimag(alpha_val) };
        double bpack[2] = { creal(beta_val),  cimag(beta_val)  };
        float64x2_t valpha = vld1q_f64(apack);
        float64x2_t vbeta  = vld1q_f64(bpack);

        const int njc = (N + NC - 1) / NC;
        const int nic = (M + MC - 1) / MC;
        const long ntiles = (long)njc * (long)nic;

        #pragma omp parallel
        {
            // rounded-up panel sizes: safe for any MR/NR
            const size_t a_blk = (size_t)KC * MR * 2;          // doubles per A i-block
            const size_t b_blk = (size_t)KC * NR * 2;          // doubles per B j-block
            const int na = (MC + MR - 1) / MR;                  // A i-blocks per panel
            const int nb = (NC + NR - 1) / NR;                  // B j-blocks per panel
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
                                        valpha, vbeta, first);
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
