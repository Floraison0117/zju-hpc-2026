#include "../sme_update.h"

#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *aligned_bytes(size_t bytes)
{
    const size_t rounded = (bytes + 63u) & ~(size_t)63u;
#if defined(_WIN32)
    return rounded == 0 ? NULL : malloc(rounded);
#else
    return rounded == 0 ? NULL : aligned_alloc(64, rounded);
#endif
}

static double value_for(int p, int j, int salt, int mode)
{
    if (mode == 1) return 0.0;
    if (mode == 2) {
        const int inverse = salt == 19;
        const double large = (p & 1) ? 1e100 : 1e-100;
        const double small = (p & 1) ? 1e-100 : 1e100;
        const double a = inverse ? small : large;
        return (j & 1) ? -a : a;
    }
    return 0.001 * (double)(1 + ((p * (salt + 7) + j * (salt + 3)) % 97));
}

static void reference(const double *a, const double *b, const double *c0,
                      double *out, int ldc, int m, int n, int k)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p)
                sum += a[(size_t)p * 16u + i] * b[(size_t)p * 32u + j];
            out[(size_t)i * (size_t)ldc + j] =
                c0[(size_t)i * (size_t)ldc + j] - sum;
        }
    }
}

static double max_error(const double *x, const double *y,
                        int ldc, int m, int n)
{
    double result = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const double e = fabs(x[(size_t)i * (size_t)ldc + j] -
                                  y[(size_t)i * (size_t)ldc + j]);
            if (e > result || isnan(e)) result = e;
        }
    }
    return result;
}

static int one_case(int m, int n, int k, int mode)
{
    const int ldc = 40;
    double *a = (double *)aligned_bytes((size_t)k * 16u * sizeof(double));
    double *b = (double *)aligned_bytes((size_t)k * 32u * sizeof(double));
    double *c = (double *)aligned_bytes((size_t)16u * ldc * sizeof(double));
    double *c_alt = (double *)aligned_bytes((size_t)16u * ldc * sizeof(double));
    double *c_direct = (double *)aligned_bytes((size_t)16u * ldc * sizeof(double));
    double *c0 = (double *)aligned_bytes((size_t)16u * ldc * sizeof(double));
    double *ref = (double *)aligned_bytes((size_t)16u * ldc * sizeof(double));
    if (a == NULL || b == NULL || c == NULL || c_alt == NULL ||
        c_direct == NULL ||
        c0 == NULL || ref == NULL) {
        free(a); free(b); free(c); free(c_alt); free(c_direct);
        free(c0); free(ref);
        return 0;
    }
    for (int p = 0; p < k; ++p) {
        for (int i = 0; i < 16; ++i) a[(size_t)p * 16u + i] =
            value_for(p, i, 11, mode);
        for (int j = 0; j < 32; ++j) b[(size_t)p * 32u + j] =
            value_for(p, j, 19, mode);
    }
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < ldc; ++j)
            c0[(size_t)i * ldc + j] = 0.01 * (double)(i + j + 1);
    memcpy(c, c0, (size_t)16u * ldc * sizeof(double));
    memcpy(c_direct, c0, (size_t)16u * ldc * sizeof(double));
    memcpy(ref, c0, (size_t)16u * ldc * sizeof(double));
    reference(a, b, c0, ref, ldc, m, n, k);
    sme_update_tile(a, 16, b, 32, c, ldc * (int)sizeof(double), m, n, k);
    const double error = max_error(c, ref, ldc, m, n);
    int za_ok = 1;
    int za_direct_ok = 1;
    if (m == 16 && n == 32) {
        memcpy(c_alt, c0, (size_t)16u * ldc * sizeof(double));
        sme_update_16x32_za_init(a, b, c_alt, k,
                                 ldc * (int)sizeof(double));
        const double za_error = max_error(c_alt, ref, ldc, m, n);
        if (za_error > 1e-12 || isnan(za_error)) {
            fprintf(stderr, "FAIL za_init k=%d mode=%d error=%.17g\n",
                    k, mode, za_error);
            za_ok = 0;
        }
        sme_update_16x32_za_ldst(a, b, c_direct, k,
                                 ldc * (int)sizeof(double));
        const double direct_error = max_error(c_direct, ref, ldc, m, n);
        if (direct_error > 1e-12 || isnan(direct_error)) {
            fprintf(stderr, "FAIL za_ldst k=%d mode=%d error=%.17g\n",
                    k, mode, direct_error);
            za_direct_ok = 0;
        }
    }
    free(a); free(b); free(c); free(c_alt); free(c_direct);
    free(c0); free(ref);
    if (error > 1e-12 || isnan(error) || !za_ok || !za_direct_ok) {
        fprintf(stderr, "FAIL m=%d n=%d k=%d mode=%d error=%.17g\n",
                m, n, k, mode, error);
        return 0;
    }
    return 1;
}

static int threaded_case(int k)
{
    const int threads = omp_get_max_threads();
    const int ldc = 32;
    double *a = (double *)aligned_bytes((size_t)k * 16u * sizeof(double));
    double *b = (double *)aligned_bytes((size_t)k * 32u * sizeof(double));
    double *c = (double *)aligned_bytes((size_t)threads * 16u * 32u *
                                         sizeof(double));
    if (a == NULL || b == NULL || c == NULL) {
        free(a); free(b); free(c);
        return 0;
    }
    for (int p = 0; p < k; ++p) {
        for (int i = 0; i < 16; ++i) a[(size_t)p * 16u + i] =
            value_for(p, i, 5, 0);
        for (int j = 0; j < 32; ++j) b[(size_t)p * 32u + j] =
            value_for(p, j, 7, 0);
    }
    memset(c, 0, (size_t)threads * 16u * 32u * sizeof(double));
#pragma omp parallel for schedule(static)
    for (int t = 0; t < threads; ++t) {
        sme_update_16x32(a, b, c + (size_t)t * 16u * 32u,
                         k, ldc * (int)sizeof(double));
    }
    double error = 0.0;
    for (int t = 0; t < threads; ++t) {
        double ref[16 * 32];
        double zero[16 * 32] = {0.0};
        reference(a, b, zero, ref, 32, 16, 32, k);
        const double e = max_error(c + (size_t)t * 16u * 32u,
                                   ref, 32, 16, 32);
        if (e > error) error = e;
    }
    printf("threads=%d threaded_k=%d max_error=%.17g\n", threads, k, error);
    free(a); free(b); free(c);
    return error <= 1e-12 && !isnan(error);
}

static int batch_case(int k, int nt)
{
    const int ldc = nt * 32 + 7;
    const size_t matrix_c = 16u * (size_t)ldc;
    double *a = (double *)aligned_bytes((size_t)k * 16u * sizeof(double));
    double *b = (double *)aligned_bytes((size_t)nt * (size_t)k * 32u *
                                         sizeof(double));
    double *c = (double *)aligned_bytes(matrix_c * sizeof(double));
    double *c0 = (double *)aligned_bytes(matrix_c * sizeof(double));
    double *ref = (double *)aligned_bytes(matrix_c * sizeof(double));
    if (a == NULL || b == NULL || c == NULL || c0 == NULL || ref == NULL) {
        free(a); free(b); free(c); free(c0); free(ref);
        return 0;
    }
    for (int p = 0; p < k; ++p) {
        for (int i = 0; i < 16; ++i)
            a[(size_t)p * 16u + i] = value_for(p, i, 23, 0);
        for (int t = 0; t < nt; ++t) {
            for (int j = 0; j < 32; ++j)
                b[((size_t)t * k + p) * 32u + j] =
                    value_for(p, j, 31 + t, 0);
        }
    }
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < ldc; ++j)
            c0[(size_t)i * ldc + j] = 0.01 * (double)(1 + i + j);
    memcpy(c, c0, matrix_c * sizeof(double));
    memcpy(ref, c0, matrix_c * sizeof(double));
    sme_update_16x32_batch(a, b, c, k, ldc * (int)sizeof(double), nt);
    for (int t = 0; t < nt; ++t)
        reference(a, b + (size_t)t * k * 32u,
                  c0 + (size_t)t * 32u,
                  ref + (size_t)t * 32u,
                  ldc, 16, 32, k);
    double error = 0.0;
    int bad_tile = -1;
    for (int t = 0; t < nt; ++t) {
        const double e = max_error(c + (size_t)t * 32u,
                                   ref + (size_t)t * 32u,
                                   ldc, 16, 32);
        if (e > error) { error = e; bad_tile = t; }
    }
    if (error > 1e-12 || isnan(error)) {
        fprintf(stderr, "FAIL batch nt=%d k=%d tile=%d error=%.17g got0=%.17g ref0=%.17g\n",
                nt, k, bad_tile, error,
                c[(size_t)bad_tile * 32u],
                ref[(size_t)bad_tile * 32u]);
        free(a); free(b); free(c); free(c0); free(ref);
        return 0;
    }
    free(a); free(b); free(c); free(c0); free(ref);
    return 1;
}

static int rows_batch_case(int k, int mt)
{
    const int ldc = 40;
    const size_t tile_a = (size_t)k * 16u;
    const size_t tile_c = 16u * (size_t)ldc;
    double *a = (double *)aligned_bytes((size_t)mt * tile_a * sizeof(double));
    double *b = (double *)aligned_bytes((size_t)k * 32u * sizeof(double));
    double *c = (double *)aligned_bytes((size_t)mt * tile_c * sizeof(double));
    double *c0 = (double *)aligned_bytes((size_t)mt * tile_c * sizeof(double));
    double *ref = (double *)aligned_bytes((size_t)mt * tile_c * sizeof(double));
    if (a == NULL || b == NULL || c == NULL || c0 == NULL || ref == NULL) {
        free(a); free(b); free(c); free(c0); free(ref);
        return 0;
    }
    for (int t = 0; t < mt; ++t) {
        for (int p = 0; p < k; ++p) {
            for (int i = 0; i < 16; ++i)
                a[((size_t)t * k + p) * 16u + i] =
                    value_for(p, i, 43 + t, 0);
        }
    }
    for (int p = 0; p < k; ++p)
        for (int j = 0; j < 32; ++j)
            b[(size_t)p * 32u + j] = value_for(p, j, 53, 0);
    for (int t = 0; t < mt; ++t) {
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < ldc; ++j)
                c0[(size_t)t * tile_c + (size_t)i * ldc + j] =
                    0.02 * (double)(1 + t + i + j);
        }
    }
    memcpy(c, c0, (size_t)mt * tile_c * sizeof(double));
    memcpy(ref, c0, (size_t)mt * tile_c * sizeof(double));
    sme_update_16x32_rows_batch(a, b, c, k, ldc * (int)sizeof(double), mt);
    for (int t = 0; t < mt; ++t)
        reference(a + (size_t)t * tile_a, b,
                  c0 + (size_t)t * tile_c,
                  ref + (size_t)t * tile_c,
                  ldc, 16, 32, k);
    double error = 0.0;
    int bad_tile = -1;
    for (int t = 0; t < mt; ++t) {
        const double e = max_error(c + (size_t)t * tile_c,
                                   ref + (size_t)t * tile_c,
                                   ldc, 16, 32);
        if (e > error) { error = e; bad_tile = t; }
    }
    if (error > 1e-12 || isnan(error)) {
        fprintf(stderr, "FAIL rows_batch mt=%d k=%d tile=%d error=%.17g got0=%.17g ref0=%.17g\n",
                mt, k, bad_tile, error,
                c[(size_t)bad_tile * tile_c],
                ref[(size_t)bad_tile * tile_c]);
        free(a); free(b); free(c); free(c0); free(ref);
        return 0;
    }
    free(a); free(b); free(c); free(c0); free(ref);
    return 1;
}

static int grid_batch_case(int k, int mt, int nt)
{
    const int ldc = nt * 32 + 7;
    const size_t tile_a = (size_t)k * 16u;
    const size_t tile_b = (size_t)k * 32u;
    const size_t matrix_c = (size_t)mt * 16u * (size_t)ldc;
    double *a = (double *)aligned_bytes((size_t)mt * tile_a * sizeof(double));
    double *b = (double *)aligned_bytes((size_t)nt * tile_b * sizeof(double));
    double *c = (double *)aligned_bytes(matrix_c * sizeof(double));
    double *c_ptr = (double *)aligned_bytes(matrix_c * sizeof(double));
    double *c_alt = (double *)aligned_bytes(matrix_c * sizeof(double));
    double *c0 = (double *)aligned_bytes(matrix_c * sizeof(double));
    double *ref = (double *)aligned_bytes(matrix_c * sizeof(double));
    if (a == NULL || b == NULL || c == NULL || c_ptr == NULL || c_alt == NULL ||
        c0 == NULL || ref == NULL) {
        free(a); free(b); free(c); free(c_ptr); free(c_alt); free(c0); free(ref);
        return 0;
    }
    for (int ti = 0; ti < mt; ++ti) {
        for (int p = 0; p < k; ++p) {
            for (int i = 0; i < 16; ++i)
                a[((size_t)ti * k + p) * 16u + i] =
                    value_for(p, i, 61 + ti, 0);
        }
    }
    for (int tj = 0; tj < nt; ++tj) {
        for (int p = 0; p < k; ++p) {
            for (int j = 0; j < 32; ++j)
                b[((size_t)tj * k + p) * 32u + j] =
                    value_for(p, j, 71 + tj, 0);
        }
    }
    for (int i = 0; i < 16 * mt; ++i)
        for (int j = 0; j < ldc; ++j)
            c0[(size_t)i * ldc + j] = 0.03 * (double)(1 + i + j);
    memcpy(c, c0, matrix_c * sizeof(double));
    memcpy(c_ptr, c0, matrix_c * sizeof(double));
    memcpy(c_alt, c0, matrix_c * sizeof(double));
    memcpy(ref, c0, matrix_c * sizeof(double));
    sme_update_16x32_grid_batch(a, b, c, k, ldc * (int)sizeof(double), nt, mt);
    for (int ti = 0; ti < mt; ++ti) {
        for (int tj = 0; tj < nt; ++tj) {
            const double *a_tile = a + (size_t)ti * tile_a;
            const double *b_tile = b + (size_t)tj * tile_b;
            double *c_tile0 = c0 + (size_t)ti * 16u * ldc + tj * 32;
            double *ref_tile = ref + (size_t)ti * 16u * ldc + tj * 32;
            reference(a_tile, b_tile, c_tile0, ref_tile,
                      ldc, 16, 32, k);
        }
    }
    const int task_count = mt * nt;
    const double *a_tiles[task_count];
    const double *b_tiles[task_count];
    double *c_tiles[task_count];
    int task = 0;
    for (int ti = 0; ti < mt; ++ti) {
        for (int tj = 0; tj < nt; ++tj, ++task) {
            a_tiles[task] = a + (size_t)ti * tile_a;
            b_tiles[task] = b + (size_t)tj * tile_b;
            c_tiles[task] = c_ptr + (size_t)ti * 16u * ldc + tj * 32;
        }
    }
    sme_update_16x32_ptr_batch(a_tiles, b_tiles, c_tiles, k,
                               ldc * (int)sizeof(double), task_count);
    double error = 0.0;
    double error_ptr = 0.0;
    double error_alt = 0.0;
    int bad_ti = -1;
    int bad_tj = -1;
    for (int ti = 0; ti < mt; ++ti) {
        for (int tj = 0; tj < nt; ++tj) {
            const double *got = c + (size_t)ti * 16u * ldc + tj * 32;
            const double *want = ref + (size_t)ti * 16u * ldc + tj * 32;
            const double e = max_error(got, want, ldc, 16, 32);
            if (e > error) { error = e; bad_ti = ti; bad_tj = tj; }
            const double e_ptr = max_error(
                c_ptr + (size_t)ti * 16u * ldc + tj * 32,
                want, ldc, 16, 32);
            if (e_ptr > error_ptr) error_ptr = e_ptr;
        }
    }
    sme_update_16x32_za_init_grid_batch(
        a, b, c_alt, k, ldc * (int)sizeof(double), nt, mt);
    for (int ti = 0; ti < mt; ++ti) {
        for (int tj = 0; tj < nt; ++tj) {
            const double *got = c_alt + (size_t)ti * 16u * ldc + tj * 32;
            const double *want = ref + (size_t)ti * 16u * ldc + tj * 32;
            const double e = max_error(got, want, ldc, 16, 32);
            if (e > error_alt) error_alt = e;
        }
    }
    if (error > 1e-12 || isnan(error) ||
        error_ptr > 1e-12 || isnan(error_ptr) ||
        error_alt > 1e-12 || isnan(error_alt)) {
        fprintf(stderr, "FAIL grid_batch mt=%d nt=%d k=%d tile=(%d,%d) error=%.17g got0=%.17g ref0=%.17g\n",
                mt, nt, k, bad_ti, bad_tj, error,
                c[(size_t)bad_ti * 16u * ldc + bad_tj * 32],
                ref[(size_t)bad_ti * 16u * ldc + bad_tj * 32]);
        fprintf(stderr, "FAIL ptr_batch mt=%d nt=%d k=%d error=%.17g\n",
                mt, nt, k, error_ptr);
        fprintf(stderr, "FAIL za_init_grid mt=%d nt=%d k=%d error=%.17g\n",
                mt, nt, k, error_alt);
        free(a); free(b); free(c); free(c_ptr); free(c_alt); free(c0); free(ref);
        return 0;
    }
    free(a); free(b); free(c); free(c_ptr); free(c_alt); free(c0); free(ref);
    return 1;
}

static int strided_grid_case(int k, int mt, int nt)
{
    const int n = nt * 32;
    const int ldb = n + 5;
    const int ldc = n + 7;
    const size_t tile_a = (size_t)k * 16u;
    const size_t matrix_c = (size_t)mt * 16u * (size_t)ldc;
    double *a = (double *)aligned_bytes((size_t)mt * tile_a * sizeof(double));
    double *b = (double *)aligned_bytes((size_t)k * (size_t)ldb * sizeof(double));
    double *c = (double *)aligned_bytes(matrix_c * sizeof(double));
    double *c0 = (double *)aligned_bytes(matrix_c * sizeof(double));
    double *ref = (double *)aligned_bytes(matrix_c * sizeof(double));
    if (a == NULL || b == NULL || c == NULL || c0 == NULL || ref == NULL) {
        free(a); free(b); free(c); free(c0); free(ref);
        return 0;
    }
    for (int ti = 0; ti < mt; ++ti)
        for (int p = 0; p < k; ++p)
            for (int i = 0; i < 16; ++i)
                a[((size_t)ti * k + p) * 16u + i] =
                    value_for(p, i, 83 + ti, 0);
    for (int p = 0; p < k; ++p) {
        for (int j = 0; j < n; ++j)
            b[(size_t)p * ldb + j] = value_for(p, j, 97, 0);
        for (int j = n; j < ldb; ++j)
            b[(size_t)p * ldb + j] = -17.0;
    }
    for (int i = 0; i < 16 * mt; ++i)
        for (int j = 0; j < ldc; ++j)
            c0[(size_t)i * ldc + j] = 0.04 * (double)(1 + i + j);
    memcpy(c, c0, matrix_c * sizeof(double));
    memcpy(ref, c0, matrix_c * sizeof(double));

    for (int ti = 0; ti < mt; ++ti) {
        for (int tj = 0; tj < nt; ++tj) {
            const double *a_tile = a + (size_t)ti * tile_a;
            double *ref_tile = ref + (size_t)ti * 16u * ldc + tj * 32;
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 32; ++j) {
                    double sum = 0.0;
                    for (int p = 0; p < k; ++p)
                        sum += a_tile[(size_t)p * 16u + i] *
                               b[(size_t)p * ldb + tj * 32 + j];
                    ref_tile[(size_t)i * ldc + j] -= sum;
                }
            }
        }
    }
    sme_update_16x32_strided_b_grid(
        a, b, k, ldb * (int)sizeof(double), c,
        ldc * (int)sizeof(double), nt, mt);

    double error = 0.0;
    int bad_ti = -1;
    int bad_tj = -1;
    for (int ti = 0; ti < mt; ++ti) {
        for (int tj = 0; tj < nt; ++tj) {
            const double *got = c + (size_t)ti * 16u * ldc + tj * 32;
            const double *want = ref + (size_t)ti * 16u * ldc + tj * 32;
            const double e = max_error(got, want, ldc, 16, 32);
            if (e > error) {
                error = e;
                bad_ti = ti;
                bad_tj = tj;
            }
        }
    }
    if (error > 1e-12 || isnan(error)) {
        fprintf(stderr,
                "FAIL strided_grid mt=%d nt=%d k=%d tile=(%d,%d) error=%.17g\n",
                mt, nt, k, bad_ti, bad_tj, error);
        free(a); free(b); free(c); free(c0); free(ref);
        return 0;
    }
    free(a); free(b); free(c); free(c0); free(ref);
    return 1;
}

static int grid_stress_case(void)
{
    const int k = 224;
    const int mt = 1050;
    const int nt = 16;
    const int ldc = 576;
    const size_t a_size = (size_t)mt * k * 16u;
    const size_t b_size = (size_t)nt * k * 32u;
    const size_t c_size = (size_t)mt * 16u * (size_t)ldc;
    double *a = (double *)aligned_bytes(a_size * sizeof(double));
    double *b = (double *)aligned_bytes(b_size * sizeof(double));
    double *c = (double *)aligned_bytes((c_size + 64u) * sizeof(double));
    if (a == NULL || b == NULL || c == NULL) {
        free(a); free(b); free(c);
        return 0;
    }
    memset(a, 0, a_size * sizeof(double));
    memset(b, 0, b_size * sizeof(double));
    memset(c, 0, (c_size + 64u) * sizeof(double));
    for (size_t i = c_size; i < c_size + 64u; ++i)
        c[i] = 12345.0 + (double)i;
    sme_update_16x32_grid_batch(a, b, c, k, ldc * (int)sizeof(double), nt, mt);
    int canary_ok = 1;
    for (size_t i = c_size; i < c_size + 64u; ++i)
        if (c[i] != 12345.0 + (double)i) canary_ok = 0;
    free(a); free(b); free(c);
    if (!canary_ok) {
        fprintf(stderr, "FAIL grid_stress canary\n");
        return 0;
    }
    return 1;
}

static int grid_threaded_case(void)
{
    const int k = 224;
    const int mt = 4;
    const int nt = 16;
    const int ldc = 576;
    const int threads = omp_get_max_threads();
    const size_t a_one = (size_t)mt * k * 16u;
    const size_t b_one = (size_t)nt * k * 32u;
    const size_t c_one = (size_t)mt * 16u * (size_t)ldc;
    double *a = (double *)aligned_bytes((size_t)threads * a_one * sizeof(double));
    double *b = (double *)aligned_bytes((size_t)threads * b_one * sizeof(double));
    double *c = (double *)aligned_bytes((size_t)threads * (c_one + 64u) *
                                         sizeof(double));
    if (a == NULL || b == NULL || c == NULL) {
        free(a); free(b); free(c);
        return 0;
    }
    memset(a, 0, (size_t)threads * a_one * sizeof(double));
    memset(b, 0, (size_t)threads * b_one * sizeof(double));
    memset(c, 0, (size_t)threads * (c_one + 64u) * sizeof(double));
    for (int t = 0; t < threads; ++t)
        for (size_t i = c_one; i < c_one + 64u; ++i)
            c[(size_t)t * (c_one + 64u) + i] = 27000.0 + (double)i;
#pragma omp parallel for schedule(static)
    for (int t = 0; t < threads; ++t) {
        sme_update_16x32_grid_batch(
            a + (size_t)t * a_one,
            b + (size_t)t * b_one,
            c + (size_t)t * (c_one + 64u),
            k, ldc * (int)sizeof(double), nt, mt);
    }
    int canary_ok = 1;
    for (int t = 0; t < threads; ++t)
        for (size_t i = c_one; i < c_one + 64u; ++i)
            if (c[(size_t)t * (c_one + 64u) + i] !=
                27000.0 + (double)i) canary_ok = 0;
    free(a); free(b); free(c);
    if (!canary_ok) {
        fprintf(stderr, "FAIL grid_threaded canary threads=%d\n", threads);
        return 0;
    }
    return 1;
}

int main(void)
{
    const int ks[] = {16, 32, 48, 64, 96, 128, 160, 192, 224, 256};
    int ok = 1;
    for (size_t q = 0; q < sizeof(ks) / sizeof(ks[0]); ++q) {
        ok &= one_case(16, 32, ks[q], 0);
        ok &= one_case(16, 32, ks[q], 1);
        ok &= one_case(16, 32, ks[q], 2);
    }
    const int tails[][2] = {{1, 1}, {7, 17}, {15, 31}, {16, 32},
                            {9, 32}, {16, 17}};
    for (size_t q = 0; q < sizeof(tails) / sizeof(tails[0]); ++q)
        ok &= one_case(tails[q][0], tails[q][1], 64, 0);
    const int batch_ks[] = {16, 64, 224};
    const int batch_nt[] = {1, 2, 16};
    for (size_t q = 0; q < sizeof(batch_ks) / sizeof(batch_ks[0]); ++q)
        for (size_t r = 0; r < sizeof(batch_nt) / sizeof(batch_nt[0]); ++r)
            ok &= batch_case(batch_ks[q], batch_nt[r]);
    const int rows_mt[] = {1, 2, 7, 19};
    for (size_t q = 0; q < sizeof(batch_ks) / sizeof(batch_ks[0]); ++q)
        for (size_t r = 0; r < sizeof(rows_mt) / sizeof(rows_mt[0]); ++r)
            ok &= rows_batch_case(batch_ks[q], rows_mt[r]);
    for (size_t q = 0; q < sizeof(batch_ks) / sizeof(batch_ks[0]); ++q)
        for (size_t r = 0; r < sizeof(rows_mt) / sizeof(rows_mt[0]); ++r)
            for (size_t s = 0; s < sizeof(batch_nt) / sizeof(batch_nt[0]); ++s)
                ok &= grid_batch_case(batch_ks[q], rows_mt[r], batch_nt[s]);
    const int strided_mt[] = {1, 2, 7};
    for (size_t q = 0; q < sizeof(batch_ks) / sizeof(batch_ks[0]); ++q)
        for (size_t r = 0; r < sizeof(strided_mt) / sizeof(strided_mt[0]); ++r)
            for (size_t s = 0; s < sizeof(batch_nt) / sizeof(batch_nt[0]); ++s)
                ok &= strided_grid_case(batch_ks[q], strided_mt[r],
                                        batch_nt[s]);
    ok &= grid_stress_case();
    ok &= grid_threaded_case();
    ok &= threaded_case(128);
    printf("sme_update_test=%s threads=%d\n", ok ? "PASS" : "FAIL",
           omp_get_max_threads());
    return ok ? 0 : 1;
}
