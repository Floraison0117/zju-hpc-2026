#include <complex.h>

// 类型定义
typedef int BLASINT;
typedef double _Complex zdouble;

// 枚举定义
enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
};
enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};

// 辅助函数：获取转置后的维度
static inline BLASINT get_A_rows(enum CBLAS_TRANSPOSE TransA, BLASINT M, BLASINT K)
{
    return (TransA == CblasNoTrans) ? M : K;
}

static inline BLASINT get_A_cols(enum CBLAS_TRANSPOSE TransA, BLASINT M, BLASINT K)
{
    return (TransA == CblasNoTrans) ? K : M;
}

static inline BLASINT get_B_rows(enum CBLAS_TRANSPOSE TransB, BLASINT K, BLASINT N)
{
    return (TransB == CblasNoTrans) ? K : N;
}

static inline BLASINT get_B_cols(enum CBLAS_TRANSPOSE TransB, BLASINT K, BLASINT N)
{
    return (TransB == CblasNoTrans) ? N : K;
}

// 核心实现：标准三重循环
void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                 const BLASINT K, const void* alpha, const void* A, const BLASINT lda,
                 const void* B, const BLASINT ldb, const void* beta, void* C, const BLASINT ldc)
{

    // 转换为复数指针
    const zdouble* A_ptr = (const zdouble*)A;
    const zdouble* B_ptr = (const zdouble*)B;
    zdouble* C_ptr = (zdouble*)C;
    zdouble alpha_val = *(const zdouble*)alpha;
    zdouble beta_val = *(const zdouble*)beta;

    // 实际维度（考虑转置）
    BLASINT A_rows = get_A_rows(TransA, M, K);
    BLASINT A_cols = get_A_cols(TransA, M, K);
    BLASINT B_rows = get_B_rows(TransB, K, N);
    BLASINT B_cols = get_B_cols(TransB, K, N);

    // 根据存储顺序进行矩阵乘法
    if (Order == CblasRowMajor) {
        // ========== 行主序存储 ==========
        // 首先缩放 C = beta * C
        if (beta_val != 1.0) {
            for (BLASINT i = 0; i < M; i++) {
                for (BLASINT j = 0; j < N; j++) {
                    C_ptr[i * ldc + j] *= beta_val;
                }
            }
        }
#pragma omp parallel for
        // 计算 C = alpha * A * B + C
        for (BLASINT i = 0; i < M; i++) {
            for (BLASINT j = 0; j < N; j++) {
                zdouble sum = 0.0;
                for (BLASINT k_idx = 0; k_idx < K; k_idx++) {
                    // 获取 A[i, k_idx]
                    zdouble a_elem;
                    if (TransA == CblasNoTrans) {
                        a_elem = A_ptr[i * lda + k_idx];
                    } else if (TransA == CblasTrans) {
                        a_elem = A_ptr[k_idx * lda + i];
                    } else { // ConjTrans
                        a_elem = conj(A_ptr[k_idx * lda + i]);
                    }

                    // 获取 B[k_idx, j]
                    zdouble b_elem;
                    if (TransB == CblasNoTrans) {
                        b_elem = B_ptr[k_idx * ldb + j];
                    } else if (TransB == CblasTrans) {
                        b_elem = B_ptr[j * ldb + k_idx];
                    } else { // ConjTrans
                        b_elem = conj(B_ptr[j * ldb + k_idx]);
                    }

                    sum += a_elem * b_elem;
                }
                C_ptr[i * ldc + j] = alpha_val * sum + C_ptr[i * ldc + j];
            }
        }
    } else { // CblasColMajor
        // ========== 列主序存储 ==========
        // 首先缩放 C = beta * C
        if (beta_val != 1.0) {
            for (BLASINT i = 0; i < M; i++) {
                for (BLASINT j = 0; j < N; j++) {
                    C_ptr[j * ldc + i] *= beta_val;
                }
            }
        }

// 计算 C = alpha * A * B + C
#pragma omp parallel for
        for (BLASINT i = 0; i < M; i++) {
            for (BLASINT j = 0; j < N; j++) {
                zdouble sum = 0.0;
                for (BLASINT k_idx = 0; k_idx < K; k_idx++) {
                    // 获取 A[i, k_idx] (列主序)
                    zdouble a_elem;
                    if (TransA == CblasNoTrans) {
                        a_elem = A_ptr[k_idx * lda + i];
                    } else if (TransA == CblasTrans) {
                        a_elem = A_ptr[i * lda + k_idx];
                    } else { // ConjTrans
                        a_elem = conj(A_ptr[i * lda + k_idx]);
                    }

                    // 获取 B[k_idx, j] (列主序)
                    zdouble b_elem;
                    if (TransB == CblasNoTrans) {
                        b_elem = B_ptr[j * ldb + k_idx];
                    } else if (TransB == CblasTrans) {
                        b_elem = B_ptr[k_idx * ldb + j];
                    } else { // ConjTrans
                        b_elem = conj(B_ptr[k_idx * ldb + j]);
                    }

                    sum += a_elem * b_elem;
                }
                C_ptr[j * ldc + i] = alpha_val * sum + C_ptr[j * ldc + i];
            }
        }
    }
}
