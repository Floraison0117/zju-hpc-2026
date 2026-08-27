#include "sme_update.h"

#include <stdint.h>

static void update_tile_scalar(const double *a_pack, int a_stride,
                               const double *b_pack, int b_stride,
                               double *c, int ldc_bytes,
                               int m, int n, int k)
{
    if (a_pack == NULL || b_pack == NULL || c == NULL ||
        m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    const int ldc = ldc_bytes / (int)sizeof(double);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                sum += a_pack[(size_t)p * (size_t)a_stride + i] *
                       b_pack[(size_t)p * (size_t)b_stride + j];
            }
            c[(size_t)i * (size_t)ldc + j] -= sum;
        }
    }
}

void sme_update_tile_fallback(const double *a_pack, int a_stride,
                              const double *b_pack, int b_stride,
                              double *c, int ldc_bytes,
                              int m, int n, int k)
{
    update_tile_scalar(a_pack, a_stride, b_pack, b_stride, c, ldc_bytes,
                       m, n, k);
}

void sme_update_tile(const double *a_pack, int a_stride,
                     const double *b_pack, int b_stride,
                     double *c, int ldc_bytes,
                     int m, int n, int k)
{
    if (m == 16 && n == 32 && k > 0 &&
        a_stride == 16 && b_stride == 32) {
        sme_update_16x32(a_pack, b_pack, c, k, ldc_bytes);
        return;
    }
    update_tile_scalar(a_pack, a_stride, b_pack, b_stride, c, ldc_bytes,
                       m, n, k);
}
