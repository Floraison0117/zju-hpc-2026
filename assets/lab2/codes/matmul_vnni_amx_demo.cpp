#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
#include <immintrin.h>

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#include <asm/prctl.h>
#endif

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif

#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

static void fill_random(uint8_t* A, int8_t* B, int M, int N, int K) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_a(0, 5);
    std::uniform_int_distribution<int> dist_b(-5, 5);

    for (int i = 0; i < M * K; ++i) A[i] = static_cast<uint8_t>(dist_a(rng));
    for (int i = 0; i < K * N; ++i) B[i] = static_cast<int8_t>(dist_b(rng));
}

// B:  [K, N] row-major
// Bp: [K/4, N, 4]
// For each output column n, four consecutive K values are packed together.
static void pack_B_vnni_i8(const int8_t* B, int8_t* Bp, int K, int N) {
    for (int kk = 0; kk < K; kk += 4) {
        for (int n = 0; n < N; ++n) {
            for (int r = 0; r < 4; ++r) {
                Bp[static_cast<size_t>(kk / 4) * N * 4 + n * 4 + r] =
                    B[static_cast<size_t>(kk + r) * N + n];
            }
        }
    }
}

static void matmul_u8s8s32_scalar(
    const uint8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K
) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[static_cast<size_t>(m) * K + k]) *
                       static_cast<int32_t>(B[static_cast<size_t>(k) * N + n]);
            }
            C[static_cast<size_t>(m) * N + n] = acc;
        }
    }
}

// AVX-512 VNNI version.
// Computes one row and 16 output columns at a time.
// Requirements for this teaching version: N % 16 == 0, K % 4 == 0.
static void matmul_u8s8s32_avx512_vnni(
    const uint8_t* A, const int8_t* Bp, int32_t* C,
    int M, int N, int K
) {
    for (int m = 0; m < M; ++m) {
        const uint8_t* a_row = A + static_cast<size_t>(m) * K;

        for (int n = 0; n < N; n += 16) {
            __m512i acc = _mm512_setzero_si512();

            for (int kk = 0; kk < K; kk += 4) {
                // A[m, kk : kk + 4], four uint8 values in one 32-bit word.
                uint32_t a4;
                std::memcpy(&a4, a_row + kk, sizeof(uint32_t));

                // Broadcast the same four A bytes to all 16 int32 lanes.
                const __m512i av = _mm512_set1_epi32(static_cast<int>(a4));

                // Load B for 16 columns.
                // Layout per int32 lane: [B[kk+0,n+i], B[kk+1,n+i], B[kk+2,n+i], B[kk+3,n+i]].
                const int8_t* b_ptr =
                    Bp + static_cast<size_t>(kk / 4) * N * 4 + n * 4;
                const __m512i bv = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(b_ptr)
                );

                // VPDPBUSD:
                // acc[i] += sum_{r=0..3} uint8_A[r] * int8_B[i][r]
                acc = _mm512_dpbusd_epi32(acc, av, bv);
            }

            _mm512_storeu_si512(
                reinterpret_cast<void*>(C + static_cast<size_t>(m) * N + n),
                acc
            );
        }
    }
}

#if defined(__linux__)
static bool enable_amx_linux() {
    long ret = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    return ret == 0;
}
#else
static bool enable_amx_linux() { return false; }
#endif

struct TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

/*
[1,1,1,1]
[1,1,1,1]
[1,1,1,1]
[1,1,1,1]
*/

static void init_amx_int8_config() {
    alignas(64) TileConfig cfg{};
    cfg.palette_id = 1;
    cfg.start_row = 0;

    // tmm0: C tile, 16 rows × 16 int32 = 16 rows × 64 bytes.
    cfg.rows[0] = 16;
    cfg.colsb[0] = 64;

    // tmm1: A tile, 16 rows × 64 uint8 = 16 rows × 64 bytes.
    cfg.rows[1] = 16;
    cfg.colsb[1] = 64;

    // tmm2: B tile in VNNI layout.
    // K block is 64, so K/4 = 16 tile rows.
    // N block is 16, each output column occupies 4 bytes in VNNI format.
    cfg.rows[2] = 16;
    cfg.colsb[2] = 64;

    _tile_loadconfig(&cfg);
}

// AMX-INT8 version.
// Computes a 16x16 block of C at a time.
// Requirements for this teaching version: M % 16 == 0, N % 16 == 0, K % 64 == 0.
static bool matmul_u8s8s32_amx_int8(
    const uint8_t* A, const int8_t* Bp, int32_t* C,
    int M, int N, int K
) {
    if (!enable_amx_linux()) {
        return false;
    }

    init_amx_int8_config();

    // 1024 / 4 = 256 = 16 * 16
    

    for (int m = 0; m < M; m += 16) {
        for (int n = 0; n < N; n += 16) {
            // C 矩阵的 16x16 block, tmm0.
            _tile_zero(0);

            // A 矩阵的 16x64 block, tmm1.
            // kk = 128
            // static_cast<size_t>(m) * K
            // |
            // [......, (128, 129, ..., 191),...] ->
            // [......, (128, 129, ..., 191),...]
            // [......, (128, 129, ..., 191),...]
            // B 矩阵的 64x16 block, tmm2.
            // int8: [K / 4, N * 4].         bf16/fp16:[K / 2, N * 2]
            // [......, (128, 129, ..., 191),...]
            // [......, (128, 129, ..., 191),...]


            /*
            32 bit 为单位的转置
            int8: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

            ->
            [1,5,9,13]
            [2,6,10,14]
            [3,7,11,15]
            [4,8,12,16]

            fp16: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

            ->
            [1,3,5,7,9,11,13,15]
            [2,4,6,8,10,12,14,16]
            */

            for (int kk = 0; kk < K; kk += 64) {
                const uint8_t* a_ptr = A + static_cast<size_t>(m) * K + kk;
                const int8_t* b_ptr =
                    Bp + static_cast<size_t>(kk / 4) * N * 4 + n * 4;

                // A tile stride: original row-major A has K bytes per row.
                // stride
                _tile_loadd(1, a_ptr, K);

                // B tile stride: packed B has N * 4 bytes per K/4 row.
                _tile_loadd(2, b_ptr, N * 4);

                // tmm0 += dot_product_u8s8(tmm1, tmm2)
                _tile_dpbusd(0, 1, 2);
            }

        
            _tile_stored(
                0,
                C + static_cast<size_t>(m) * N + n,
                N * sizeof(int32_t)
            );
        }
    }

    _tile_release();
    return true;
}

static int32_t max_abs_diff(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    int32_t max_diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        int32_t d = std::abs(a[i] - b[i]);
        max_diff = std::max(max_diff, d);
    }
    return max_diff;
}

int main(int argc, char** argv) {
    const int M = (argc > 1) ? std::atoi(argv[1]) : 128;
    const int N = (argc > 2) ? std::atoi(argv[2]) : 128;
    const int K = (argc > 3) ? std::atoi(argv[3]) : 256;
    const int repeat = (argc > 4) ? std::atoi(argv[4]) : 20;

    if (M % 16 != 0 || N % 16 != 0 || K % 64 != 0) {
        std::cerr << "This teaching demo requires M % 16 == 0, N % 16 == 0, K % 64 == 0.\n";
        return 1;
    }

    std::vector<uint8_t> A(static_cast<size_t>(M) * K);
    std::vector<int8_t> B(static_cast<size_t>(K) * N);
    std::vector<int8_t> Bp(static_cast<size_t>(K / 4) * N * 4);

    std::vector<int32_t> C_ref(static_cast<size_t>(M) * N);
    std::vector<int32_t> C_vnni(static_cast<size_t>(M) * N);
    std::vector<int32_t> C_amx(static_cast<size_t>(M) * N);

    fill_random(A.data(), B.data(), M, N, K);
    pack_B_vnni_i8(B.data(), Bp.data(), K, N);

    matmul_u8s8s32_scalar(A.data(), B.data(), C_ref.data(), M, N, K);
    matmul_u8s8s32_avx512_vnni(A.data(), Bp.data(), C_vnni.data(), M, N, K);

    const int32_t diff_vnni = max_abs_diff(C_ref, C_vnni);

    std::cout << "INT8 GEMM: C[M,N] = A[M,K] u8 x B[K,N] s8 -> s32\n";
    std::cout << "M=" << M << ", N=" << N << ", K=" << K << ", repeat=" << repeat << "\n";
    std::cout << "AVX-512 VNNI max_abs_diff: " << diff_vnni
              << (diff_vnni == 0 ? " PASS\n" : " FAIL\n");

    bool amx_ok = matmul_u8s8s32_amx_int8(A.data(), Bp.data(), C_amx.data(), M, N, K);
    if (amx_ok) {
        const int32_t diff_amx = max_abs_diff(C_ref, C_amx);
        std::cout << "AMX-INT8 max_abs_diff: " << diff_amx
                  << (diff_amx == 0 ? " PASS\n" : " FAIL\n");
    } else {
        std::cout << "AMX-INT8: SKIP, arch_prctl XTILEDATA permission failed or unsupported OS.\n";
    }

    double t0 = now_ms();
    for (int r = 0; r < repeat; ++r) {
        matmul_u8s8s32_scalar(A.data(), B.data(), C_ref.data(), M, N, K);
    }
    double t1 = now_ms();

    for (int r = 0; r < repeat; ++r) {
        matmul_u8s8s32_avx512_vnni(A.data(), Bp.data(), C_vnni.data(), M, N, K);
    }
    double t2 = now_ms();

    bool amx_ran = false;
    double t3 = t2;
    double t4 = t2;
    if (amx_ok) {
        t3 = now_ms();
        for (int r = 0; r < repeat; ++r) {
            matmul_u8s8s32_amx_int8(A.data(), Bp.data(), C_amx.data(), M, N, K);
        }
        t4 = now_ms();
        amx_ran = true;
    }

    std::cout << "scalar avg: " << (t1 - t0) / repeat << " ms\n";
    std::cout << "vnni   avg: " << (t2 - t1) / repeat << " ms\n";
    if (amx_ran) {
        std::cout << "amx    avg: " << (t4 - t3) / repeat << " ms\n";
    }

    return (diff_vnni == 0) ? 0 : 1;
}
