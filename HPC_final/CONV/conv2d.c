#include <omp.h>
#include <math.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

typedef float CONVFLOAT;
typedef int CONVINT;

/*
 * The reference implementation accumulates in the order
 *
 *     kernel row -> kernel column
 *
 * for every output element.  The vector path below changes only the
 * independent output columns: each lane still visits the kernel elements in
 * exactly that order.  On the target compiler this produces the same fused
 * multiply-add operation as the reference while exposing four
 * single-precision values per NEON instruction.  Adjacent output rows are
 * handled together so their shared input rows are loaded only once.
 */

#if defined(__aarch64__)

static inline void conv2d_neon_row(const CONVFLOAT *input,
                                   CONVINT inputWidth,
                                   const CONVFLOAT *kernel,
                                   CONVINT kernelHeight,
                                   CONVINT kernelWidth,
                                   CONVINT outputWidth,
                                   CONVINT outputRow,
                                   CONVFLOAT *output)
{
    CONVINT i = 0;

    /* Eight accumulators cover 32 output columns and hide arithmetic latency. */
    for (; i + 31 < outputWidth; i += 32) {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        float32x4_t sum4 = vdupq_n_f32(0.0f);
        float32x4_t sum5 = vdupq_n_f32(0.0f);
        float32x4_t sum6 = vdupq_n_f32(0.0f);
        float32x4_t sum7 = vdupq_n_f32(0.0f);

        for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
            const CONVFLOAT *inputRow = input + (outputRow + jk) * inputWidth + i;
            const CONVFLOAT *kernelRow = kernel + jk * kernelWidth;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                const CONVFLOAT coefficient = kernelRow[ik];
                sum0 = vaddq_f32(sum0, vmulq_n_f32(vld1q_f32(inputRow + ik), coefficient));
                sum1 = vaddq_f32(sum1, vmulq_n_f32(vld1q_f32(inputRow + ik + 4), coefficient));
                sum2 = vaddq_f32(sum2, vmulq_n_f32(vld1q_f32(inputRow + ik + 8), coefficient));
                sum3 = vaddq_f32(sum3, vmulq_n_f32(vld1q_f32(inputRow + ik + 12), coefficient));
                sum4 = vaddq_f32(sum4, vmulq_n_f32(vld1q_f32(inputRow + ik + 16), coefficient));
                sum5 = vaddq_f32(sum5, vmulq_n_f32(vld1q_f32(inputRow + ik + 20), coefficient));
                sum6 = vaddq_f32(sum6, vmulq_n_f32(vld1q_f32(inputRow + ik + 24), coefficient));
                sum7 = vaddq_f32(sum7, vmulq_n_f32(vld1q_f32(inputRow + ik + 28), coefficient));
            }
        }

        CONVFLOAT *outputRowPtr = output + i;
        vst1q_f32(outputRowPtr, sum0);
        vst1q_f32(outputRowPtr + 4, sum1);
        vst1q_f32(outputRowPtr + 8, sum2);
        vst1q_f32(outputRowPtr + 12, sum3);
        vst1q_f32(outputRowPtr + 16, sum4);
        vst1q_f32(outputRowPtr + 20, sum5);
        vst1q_f32(outputRowPtr + 24, sum6);
        vst1q_f32(outputRowPtr + 28, sum7);
    }

    /* Handle the short tail with one vector accumulator at a time. */
    for (; i + 3 < outputWidth; i += 4) {
        float32x4_t sum = vdupq_n_f32(0.0f);

        for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
            const CONVFLOAT *inputRow = input + (outputRow + jk) * inputWidth + i;
            const CONVFLOAT *kernelRow = kernel + jk * kernelWidth;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                sum = vaddq_f32(sum, vmulq_n_f32(vld1q_f32(inputRow + ik), kernelRow[ik]));
            }
        }

        vst1q_f32(output + i, sum);
    }

    /* At most three columns remain after the vector loops. */
    for (; i < outputWidth; ++i) {
        CONVFLOAT sum = 0.0f;

        for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
            const CONVFLOAT *inputRow = input + (outputRow + jk) * inputWidth + i;
            const CONVFLOAT *kernelRow = kernel + jk * kernelWidth;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                /* Match the reference program's fused multiply-add. */
                sum = fmaf(inputRow[ik], kernelRow[ik], sum);
            }
        }

        output[i] = sum;
    }
}

/* Compute two adjacent output rows while reusing every input-row load. */
static inline void conv2d_neon_rows2(const CONVFLOAT *input,
                                     CONVINT inputWidth,
                                     const CONVFLOAT *kernel,
                                     CONVINT kernelHeight,
                                     CONVINT kernelWidth,
                                     CONVINT outputWidth,
                                     CONVINT outputRow,
                                     CONVFLOAT *output0,
                                     CONVFLOAT *output1)
{
    CONVINT i = 0;

    /* Four accumulators per output row keep the reused input vectors live. */
    for (; i + 15 < outputWidth; i += 16) {
        float32x4_t sum00 = vdupq_n_f32(0.0f);
        float32x4_t sum01 = vdupq_n_f32(0.0f);
        float32x4_t sum02 = vdupq_n_f32(0.0f);
        float32x4_t sum03 = vdupq_n_f32(0.0f);
        float32x4_t sum10 = vdupq_n_f32(0.0f);
        float32x4_t sum11 = vdupq_n_f32(0.0f);
        float32x4_t sum12 = vdupq_n_f32(0.0f);
        float32x4_t sum13 = vdupq_n_f32(0.0f);

        {
            const CONVFLOAT *inputRow = input + outputRow * inputWidth + i;
            const CONVFLOAT *kernelRow = kernel;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                const float32x4_t x0 = vld1q_f32(inputRow + ik);
                const float32x4_t x1 = vld1q_f32(inputRow + ik + 4);
                const float32x4_t x2 = vld1q_f32(inputRow + ik + 8);
                const float32x4_t x3 = vld1q_f32(inputRow + ik + 12);
                const CONVFLOAT coefficient = kernelRow[ik];

                sum00 = vaddq_f32(sum00, vmulq_n_f32(x0, coefficient));
                sum01 = vaddq_f32(sum01, vmulq_n_f32(x1, coefficient));
                sum02 = vaddq_f32(sum02, vmulq_n_f32(x2, coefficient));
                sum03 = vaddq_f32(sum03, vmulq_n_f32(x3, coefficient));
            }
        }

        for (CONVINT jk = 1; jk < kernelHeight; ++jk) {
            const CONVFLOAT *inputRow = input + (outputRow + jk) * inputWidth + i;
            const CONVFLOAT *kernelRow0 = kernel + jk * kernelWidth;
            const CONVFLOAT *kernelRow1 = kernel + (jk - 1) * kernelWidth;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                const float32x4_t x0 = vld1q_f32(inputRow + ik);
                const float32x4_t x1 = vld1q_f32(inputRow + ik + 4);
                const float32x4_t x2 = vld1q_f32(inputRow + ik + 8);
                const float32x4_t x3 = vld1q_f32(inputRow + ik + 12);
                const CONVFLOAT coefficient0 = kernelRow0[ik];
                const CONVFLOAT coefficient1 = kernelRow1[ik];

                sum00 = vaddq_f32(sum00, vmulq_n_f32(x0, coefficient0));
                sum01 = vaddq_f32(sum01, vmulq_n_f32(x1, coefficient0));
                sum02 = vaddq_f32(sum02, vmulq_n_f32(x2, coefficient0));
                sum03 = vaddq_f32(sum03, vmulq_n_f32(x3, coefficient0));
                sum10 = vaddq_f32(sum10, vmulq_n_f32(x0, coefficient1));
                sum11 = vaddq_f32(sum11, vmulq_n_f32(x1, coefficient1));
                sum12 = vaddq_f32(sum12, vmulq_n_f32(x2, coefficient1));
                sum13 = vaddq_f32(sum13, vmulq_n_f32(x3, coefficient1));
            }
        }

        {
            const CONVFLOAT *inputRow = input + (outputRow + kernelHeight) * inputWidth + i;
            const CONVFLOAT *kernelRow = kernel + (kernelHeight - 1) * kernelWidth;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                const float32x4_t x0 = vld1q_f32(inputRow + ik);
                const float32x4_t x1 = vld1q_f32(inputRow + ik + 4);
                const float32x4_t x2 = vld1q_f32(inputRow + ik + 8);
                const float32x4_t x3 = vld1q_f32(inputRow + ik + 12);
                const CONVFLOAT coefficient = kernelRow[ik];

                sum10 = vaddq_f32(sum10, vmulq_n_f32(x0, coefficient));
                sum11 = vaddq_f32(sum11, vmulq_n_f32(x1, coefficient));
                sum12 = vaddq_f32(sum12, vmulq_n_f32(x2, coefficient));
                sum13 = vaddq_f32(sum13, vmulq_n_f32(x3, coefficient));
            }
        }

        vst1q_f32(output0 + i, sum00);
        vst1q_f32(output0 + i + 4, sum01);
        vst1q_f32(output0 + i + 8, sum02);
        vst1q_f32(output0 + i + 12, sum03);
        vst1q_f32(output1 + i, sum10);
        vst1q_f32(output1 + i + 4, sum11);
        vst1q_f32(output1 + i + 8, sum12);
        vst1q_f32(output1 + i + 12, sum13);
    }

    /* A four-column vector tail uses the same row-sharing scheme. */
    for (; i + 3 < outputWidth; i += 4) {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);

        {
            const CONVFLOAT *inputRow = input + outputRow * inputWidth + i;
            const CONVFLOAT *kernelRow = kernel;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                sum0 = vaddq_f32(sum0, vmulq_n_f32(vld1q_f32(inputRow + ik), kernelRow[ik]));
            }
        }

        for (CONVINT jk = 1; jk < kernelHeight; ++jk) {
            const CONVFLOAT *inputRow = input + (outputRow + jk) * inputWidth + i;
            const CONVFLOAT *kernelRow0 = kernel + jk * kernelWidth;
            const CONVFLOAT *kernelRow1 = kernel + (jk - 1) * kernelWidth;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                const float32x4_t values = vld1q_f32(inputRow + ik);
                sum0 = vaddq_f32(sum0, vmulq_n_f32(values, kernelRow0[ik]));
                sum1 = vaddq_f32(sum1, vmulq_n_f32(values, kernelRow1[ik]));
            }
        }

        {
            const CONVFLOAT *inputRow = input + (outputRow + kernelHeight) * inputWidth + i;
            const CONVFLOAT *kernelRow = kernel + (kernelHeight - 1) * kernelWidth;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                sum1 = vaddq_f32(sum1, vmulq_n_f32(vld1q_f32(inputRow + ik), kernelRow[ik]));
            }
        }

        vst1q_f32(output0 + i, sum0);
        vst1q_f32(output1 + i, sum1);
    }

    /* Scalar columns retain the reference program's fused multiply-add order. */
    for (; i < outputWidth; ++i) {
        CONVFLOAT sum0 = 0.0f;
        CONVFLOAT sum1 = 0.0f;

        for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
            const CONVFLOAT *inputRow0 = input + (outputRow + jk) * inputWidth + i;
            const CONVFLOAT *inputRow1 = input + (outputRow + jk + 1) * inputWidth + i;
            const CONVFLOAT *kernelRow = kernel + jk * kernelWidth;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                sum0 = fmaf(inputRow0[ik], kernelRow[ik], sum0);
                sum1 = fmaf(inputRow1[ik], kernelRow[ik], sum1);
            }
        }

        output0[i] = sum0;
        output1[i] = sum1;
    }
}

#define CONV_ACCUM_PAIR(sum_a, sum_b, value_a, value_b, coefficient) \
    do { \
        (sum_a) = vaddq_f32((sum_a), vmulq_n_f32((value_a), (coefficient))); \
        (sum_b) = vaddq_f32((sum_b), vmulq_n_f32((value_b), (coefficient))); \
    } while (0)

/* Compute four adjacent output rows with an eight-column vector tile. */
static inline void conv2d_neon_rows4(const CONVFLOAT *input,
                                     CONVINT inputWidth,
                                     const CONVFLOAT *kernel,
                                     CONVINT kernelHeight,
                                     CONVINT kernelWidth,
                                     CONVINT outputWidth,
                                     CONVINT outputRow,
                                     CONVFLOAT *output0,
                                     CONVFLOAT *output1,
                                     CONVFLOAT *output2,
                                     CONVFLOAT *output3)
{
    CONVINT i = 0;

    for (; i + 7 < outputWidth; i += 8) {
        float32x4_t sum00 = vdupq_n_f32(0.0f);
        float32x4_t sum01 = vdupq_n_f32(0.0f);
        float32x4_t sum10 = vdupq_n_f32(0.0f);
        float32x4_t sum11 = vdupq_n_f32(0.0f);
        float32x4_t sum20 = vdupq_n_f32(0.0f);
        float32x4_t sum21 = vdupq_n_f32(0.0f);
        float32x4_t sum30 = vdupq_n_f32(0.0f);
        float32x4_t sum31 = vdupq_n_f32(0.0f);

        for (CONVINT r = 0; r < kernelHeight + 3; ++r) {
            const CONVFLOAT *inputRow = input + (outputRow + r) * inputWidth + i;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                const float32x4_t x0 = vld1q_f32(inputRow + ik);
                const float32x4_t x1 = vld1q_f32(inputRow + ik + 4);

                if (r < kernelHeight) {
                    const CONVFLOAT coefficient = kernel[r * kernelWidth + ik];
                    CONV_ACCUM_PAIR(sum00, sum01, x0, x1, coefficient);
                }
                if (r >= 1 && r < kernelHeight + 1) {
                    const CONVFLOAT coefficient = kernel[(r - 1) * kernelWidth + ik];
                    CONV_ACCUM_PAIR(sum10, sum11, x0, x1, coefficient);
                }
                if (r >= 2 && r < kernelHeight + 2) {
                    const CONVFLOAT coefficient = kernel[(r - 2) * kernelWidth + ik];
                    CONV_ACCUM_PAIR(sum20, sum21, x0, x1, coefficient);
                }
                if (r >= 3) {
                    const CONVFLOAT coefficient = kernel[(r - 3) * kernelWidth + ik];
                    CONV_ACCUM_PAIR(sum30, sum31, x0, x1, coefficient);
                }
            }
        }

        vst1q_f32(output0 + i, sum00);
        vst1q_f32(output0 + i + 4, sum01);
        vst1q_f32(output1 + i, sum10);
        vst1q_f32(output1 + i + 4, sum11);
        vst1q_f32(output2 + i, sum20);
        vst1q_f32(output2 + i + 4, sum21);
        vst1q_f32(output3 + i, sum30);
        vst1q_f32(output3 + i + 4, sum31);
    }

    for (; i + 3 < outputWidth; i += 4) {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        for (CONVINT r = 0; r < kernelHeight + 3; ++r) {
            const CONVFLOAT *inputRow = input + (outputRow + r) * inputWidth + i;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                const float32x4_t values = vld1q_f32(inputRow + ik);

                if (r < kernelHeight) {
                    sum0 = vaddq_f32(sum0, vmulq_n_f32(values, kernel[r * kernelWidth + ik]));
                }
                if (r >= 1 && r < kernelHeight + 1) {
                    sum1 = vaddq_f32(sum1, vmulq_n_f32(values, kernel[(r - 1) * kernelWidth + ik]));
                }
                if (r >= 2 && r < kernelHeight + 2) {
                    sum2 = vaddq_f32(sum2, vmulq_n_f32(values, kernel[(r - 2) * kernelWidth + ik]));
                }
                if (r >= 3) {
                    sum3 = vaddq_f32(sum3, vmulq_n_f32(values, kernel[(r - 3) * kernelWidth + ik]));
                }
            }
        }

        vst1q_f32(output0 + i, sum0);
        vst1q_f32(output1 + i, sum1);
        vst1q_f32(output2 + i, sum2);
        vst1q_f32(output3 + i, sum3);
    }

    for (; i < outputWidth; ++i) {
        CONVFLOAT sum0 = 0.0f;
        CONVFLOAT sum1 = 0.0f;
        CONVFLOAT sum2 = 0.0f;
        CONVFLOAT sum3 = 0.0f;

        for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
            const CONVFLOAT *inputRow0 = input + (outputRow + jk) * inputWidth + i;
            const CONVFLOAT *inputRow1 = input + (outputRow + jk + 1) * inputWidth + i;
            const CONVFLOAT *inputRow2 = input + (outputRow + jk + 2) * inputWidth + i;
            const CONVFLOAT *inputRow3 = input + (outputRow + jk + 3) * inputWidth + i;
            const CONVFLOAT *kernelRow = kernel + jk * kernelWidth;

            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                sum0 = fmaf(inputRow0[ik], kernelRow[ik], sum0);
                sum1 = fmaf(inputRow1[ik], kernelRow[ik], sum1);
                sum2 = fmaf(inputRow2[ik], kernelRow[ik], sum2);
                sum3 = fmaf(inputRow3[ik], kernelRow[ik], sum3);
            }
        }

        output0[i] = sum0;
        output1[i] = sum1;
        output2[i] = sum2;
        output3[i] = sum3;
    }
}


#undef CONV_ACCUM_PAIR
#undef CONV_ACCUM_ONE

#endif

void conv2d(const CONVFLOAT *input,
            CONVINT inputHeight,
            CONVINT inputWidth,
            const CONVFLOAT *kernel,
            CONVINT kernelHeight,
            CONVINT kernelWidth,
            CONVFLOAT *output)
{
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;

#if defined(__aarch64__)
    CONVINT j = 0;
#pragma omp parallel for schedule(static)
    for (j = 0; j < outputHeight; j += 4) {
        if (j + 3 < outputHeight) {
            conv2d_neon_rows4(input, inputWidth, kernel, kernelHeight, kernelWidth,
                              outputWidth, j, output + j * outputWidth,
                              output + (j + 1) * outputWidth,
                              output + (j + 2) * outputWidth,
                              output + (j + 3) * outputWidth);
        } else if (j + 1 < outputHeight) {
            conv2d_neon_rows2(input, inputWidth, kernel, kernelHeight, kernelWidth,
                              outputWidth, j, output + j * outputWidth,
                              output + (j + 1) * outputWidth);
        } else {
            conv2d_neon_row(input, inputWidth, kernel, kernelHeight, kernelWidth,
                            outputWidth, j, output + j * outputWidth);
        }
    }
#else
    /* Portable fallback for local correctness tests on non-Arm machines. */
#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; ++j) {
        CONVFLOAT *outputRow = output + j * outputWidth;

        for (CONVINT i = 0; i < outputWidth; ++i) {
            CONVFLOAT sum = 0.0f;

            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT *inputRow = input + (j + jk) * inputWidth + i;
                const CONVFLOAT *kernelRow = kernel + jk * kernelWidth;

                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    sum += inputRow[ik] * kernelRow[ik];
                }
            }

            outputRow[i] = sum;
        }
    }
#endif
}
