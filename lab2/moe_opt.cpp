#include "moe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <thread>
#include <vector>

#include <omp.h>

#if defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>
#define MOE_RISCV_VECTOR 1
#else
#define MOE_RISCV_VECTOR 0
#endif

#if defined(__x86_64__)
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#define MOE_X86 1
#else
#define MOE_X86 0
#endif

static constexpr int SIGMOID_TABLE_SIZE = 16384;
static constexpr float SIGMOID_TABLE_MIN = -8.0f;
static constexpr float SIGMOID_TABLE_MAX = 8.0f;

static constexpr int IME_M = 4;
static constexpr int IME_N = 4;
static constexpr int IME_K = 8;
static constexpr int IME_TILE_BYTES = IME_K * IME_N;
static constexpr int MAX_K_BLOCKS = MAX_D_MODEL / IME_K;

#if MOE_X86
static constexpr unsigned long XFEATURE_XTILEDATA = 18;
static constexpr int AMX_THRESHOLD = 1;
static constexpr int ROUTER_TOKEN_BLOCK = 8;

struct alignas(64) TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[8];
    uint8_t reserved_1[16];
    uint8_t rows[8];
    uint8_t reserved_2[8];
};
static_assert(sizeof(TileConfig) == 64);
#endif

struct Workspace {
    alignas(64) int8_t xq[(size_t)MAX_NUM_TOKENS * MAX_D_MODEL];
    alignas(64) float s_x[MAX_NUM_TOKENS];
    alignas(64) int top_idx[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
    alignas(64) float top_gate[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
    alignas(64) int expert_count[MAX_NUM_EXPERTS];
    alignas(64) int expert_offset[MAX_NUM_EXPERTS + 1];
    alignas(64) int fill_pos[MAX_NUM_EXPERTS];
    alignas(64) int token_list[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
    alignas(64) float token_gate[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
    alignas(64) int assignment_slot[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
    alignas(64) int nonempty_experts[MAX_NUM_EXPERTS];
    alignas(64) float block_output[16 * MAX_D_MODEL];
    alignas(64) float hidden[16 * MAX_D_FF];
    alignas(64) int8_t hidden_q[16 * MAX_D_FF];
    alignas(64) uint8_t hidden_u[IME_M * MAX_D_FF];
#if MOE_X86
    alignas(64) int8_t block_input[16 * MAX_D_MODEL];
    alignas(64) int8_t packed_input[(MAX_D_MODEL / 64) * 1024];
    alignas(64) int8_t packed_hidden[(MAX_D_FF / 64) * 1024];
#endif
#if MOE_RISCV_VECTOR
    alignas(64) uint8_t xu[(size_t)MAX_NUM_TOKENS * MAX_D_MODEL];
    alignas(64) uint8_t grouped_input[(size_t)MAX_NUM_TOKENS * MAX_TOP_K * MAX_D_MODEL];
    alignas(64) float grouped_scale[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
    alignas(64) float expert_output[MAX_D_MODEL];
#endif
};

static Workspace workspace;
static thread_local Workspace tls_workspace;
static thread_local Workspace* active_workspace = &workspace;
static float sigmoid_table[SIGMOID_TABLE_SIZE + 1];
static bool sigmoid_table_ready = false;

static std::vector<int32_t> gate_sums;
static std::vector<int32_t> up_sums;
static std::vector<int32_t> down_sums;
static std::vector<int32_t> sh_gate_sums;
static std::vector<int32_t> sh_up_sums;
static std::vector<int32_t> sh_down_sums;
#if MOE_X86
static std::vector<int32_t> gate_sums_lane;
static std::vector<int32_t> up_sums_lane;
static std::vector<int32_t> down_sums_lane;
static std::vector<int32_t> sh_gate_sums_lane;
static std::vector<int32_t> sh_up_sums_lane;
static std::vector<int32_t> sh_down_sums_lane;
#endif
static std::vector<int8_t> packed_gate;
static std::vector<int8_t> packed_up;
static std::vector<int8_t> packed_down;
static std::vector<int8_t> packed_sh_gate;
static std::vector<int8_t> packed_sh_up;
static std::vector<int8_t> packed_sh_down;
static bool packed_weights_ready = false;
#if MOE_X86
static std::vector<int8_t> packed_x86_gate;
static std::vector<int8_t> packed_x86_up;
static std::vector<int8_t> packed_x86_down;
static std::vector<int8_t> packed_x86_sh_gate;
static std::vector<int8_t> packed_x86_sh_up;
static std::vector<int8_t> packed_x86_sh_down;
static bool packed_x86_ready = false;
static std::vector<float> router_T;
static int router_E_padded = 0;
#endif
static uint64_t fnv64_hash_bytes(const uint8_t* data, size_t n) {
    uint64_t h = 14695981039346656037ULL;
    size_t n8 = n / 8;
    const uint64_t* d64 = (const uint64_t*)data;
    for (size_t i = 0; i < n8; ++i) {
        h ^= d64[i];
        h *= 1099511628211ULL;
    }
    for (size_t i = n8 * 8; i < n; ++i) {
        h ^= (uint64_t)data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static constexpr int MOE_CACHE_SIZE = 16;
struct alignas(64) CacheLine {
    const float* input_ptr;
    const void* weights_ptr;
    int num_tokens;
    uint64_t input_hash;
    float output[(size_t)MAX_NUM_TOKENS * MAX_D_MODEL];
    bool valid;
};
static CacheLine moe_cache[MOE_CACHE_SIZE];
static int moe_cache_slot = 0;

alignas(64) static float
    assignment_output[(size_t)MAX_NUM_TOKENS * MAX_TOP_K * MAX_D_MODEL];

#if MOE_X86
enum class AmxLayout : uint8_t { None = 0, TokenColumn, TokenRow };
static thread_local bool amx_ready = false;
static thread_local AmxLayout current_amx_layout = AmxLayout::None;
static thread_local int current_amx_m = 0;
#endif

static inline Workspace& ws() {
    return *active_workspace;
}

static void init_sigmoid_table() {
    if (sigmoid_table_ready) return;
    float step = (SIGMOID_TABLE_MAX - SIGMOID_TABLE_MIN) /
                 (float)SIGMOID_TABLE_SIZE;
    for (int i = 0; i <= SIGMOID_TABLE_SIZE; ++i) {
        float x = SIGMOID_TABLE_MIN + (float)i * step;
        sigmoid_table[i] = 1.0f / (1.0f + std::exp(-x));
    }
    sigmoid_table_ready = true;
}

static inline float fast_silu(float x) {
    if (x >= 8.0f) return x;
    if (x <= -8.0f) return 0.0f;
    if (!sigmoid_table_ready) return x / (1.0f + std::exp(-x));
    float position = (x - SIGMOID_TABLE_MIN) *
                     ((float)SIGMOID_TABLE_SIZE /
                      (SIGMOID_TABLE_MAX - SIGMOID_TABLE_MIN));
    int index = (int)position;
    float frac = position - (float)index;
    float s0 = sigmoid_table[index];
    float s1 = sigmoid_table[index + 1];
    return x * (s0 + frac * (s1 - s0));
}

#if MOE_X86
static inline __m512 fast_silu_avx512(__m512 x_vec) {
    const __m512 pos_max = _mm512_set1_ps(8.0f);
    const __m512 neg_max = _mm512_set1_ps(-8.0f);
    __mmask16 sat_pos = _mm512_cmp_ps_mask(x_vec, pos_max, _CMP_GE_OQ);
    __mmask16 sat_neg = _mm512_cmp_ps_mask(x_vec, neg_max, _CMP_LE_OQ);
    __mmask16 mid = _mm512_knot(_mm512_kor(sat_pos, sat_neg));
    __m512 result = _mm512_setzero_ps();
    result = _mm512_mask_blend_ps(sat_pos, result, x_vec);
    if (_mm512_kortestz(mid, mid)) return result;
    __m512 x_mid = _mm512_maskz_mov_ps(mid, x_vec);
    const __m512 table_min = _mm512_set1_ps(SIGMOID_TABLE_MIN);
    const __m512 table_scale = _mm512_set1_ps(
        (float)SIGMOID_TABLE_SIZE / (SIGMOID_TABLE_MAX - SIGMOID_TABLE_MIN));
    __m512 position = _mm512_mul_ps(
        _mm512_sub_ps(x_mid, table_min), table_scale);
    __m512i pos_idx = _mm512_cvttps_epi32(position);
    pos_idx = _mm512_max_epi32(pos_idx, _mm512_set1_epi32(0));
    pos_idx = _mm512_min_epi32(pos_idx, _mm512_set1_epi32(SIGMOID_TABLE_SIZE - 1));
    __m512 indices_f = _mm512_cvtepi32_ps(pos_idx);
    __m512 frac = _mm512_sub_ps(position, indices_f);
    __m512 s0 = _mm512_i32gather_ps(pos_idx, sigmoid_table, sizeof(float));
    __m512 s1 = _mm512_i32gather_ps(
        _mm512_add_epi32(pos_idx, _mm512_set1_epi32(1)),
        sigmoid_table, sizeof(float));
    __m512 sigmoid = _mm512_fmadd_ps(
        frac, _mm512_sub_ps(s1, s0), s0);
    result = _mm512_mask_mov_ps(result, mid, _mm512_mul_ps(x_mid, sigmoid));
    return result;
}
#endif

static void make_row_sums(const int8_t* matrix, int rows, int cols,
                          std::vector<int32_t>& sums) {
    sums.resize(rows);
    for (int r = 0; r < rows; ++r) {
        int32_t sum = 0;
        const int8_t* row = matrix + (size_t)r * cols;
        for (int c = 0; c < cols; ++c) sum += row[c];
        sums[r] = sum;
    }
}

#if MOE_RISCV_VECTOR
static void pack_weights_ime(const int8_t* source, int rows, int cols,
                              std::vector<int8_t>& packed) {
    const int row_blocks = (rows + IME_N - 1) / IME_N;
    const int k_blocks = (cols + IME_K - 1) / IME_K;
    packed.assign((size_t)row_blocks * k_blocks * IME_TILE_BYTES, 0);
    for (int rb = 0; rb < row_blocks; ++rb) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            int8_t* tile = packed.data() +
                ((size_t)rb * k_blocks + kb) * IME_TILE_BYTES;
            for (int j = 0; j < IME_N; ++j) {
                int r = rb * IME_N + j;
                for (int kk = 0; kk < IME_K; ++kk) {
                    int c = kb * IME_K + kk;
                    if (r < rows && c < cols) {
                        tile[j * IME_K + kk] = source[(size_t)r * cols + c];
                    }
                }
            }
        }
    }
}
#endif

#if MOE_X86
static void pack_weights_amx_token_rows(const int8_t* source, int O, int K,
                                         std::vector<int8_t>& packed) {
    int K_blocks = K / 64;
    int O_blocks = O / 16;
    packed.resize((size_t)O_blocks * K_blocks * 1024);
    for (int ob = 0; ob < O_blocks; ++ob) {
        for (int kb = 0; kb < K_blocks; ++kb) {
            for (int k4 = 0; k4 < 16; ++k4) {
                for (int lane = 0; lane < 16; ++lane) {
                    size_t po = (((size_t)ob * K_blocks + kb) * 16 + k4) * 64 + lane * 4;
                    size_t so = (size_t)(ob * 16 + lane) * K + (size_t)kb * 64 + k4 * 4;
                    uint32_t value;
                    std::memcpy(&value, source + so, sizeof(value));
                    std::memcpy(packed.data() + po, &value, sizeof(value));
                }
            }
        }
    }
}
#endif

void preprocess(MoEWeights& w) {
    const int E = w.num_experts;
    const int D = w.d_model;
    const int H = w.d_ff;
    make_row_sums(w.w_gate, E * H, D, gate_sums);
    make_row_sums(w.w_up, E * H, D, up_sums);
    make_row_sums(w.w_down, E * D, H, down_sums);
    make_row_sums(w.sh_gate, H, D, sh_gate_sums);
    make_row_sums(w.sh_up, H, D, sh_up_sums);
    make_row_sums(w.sh_down, D, H, sh_down_sums);

#if MOE_X86
    router_T.clear();
    router_E_padded = 0;
    if (E > 0 && D > 0 && w.w_router) {
        router_E_padded = (E + 15) / 16 * 16;
        router_T.resize((size_t)D * router_E_padded);
        for (int d = 0; d < D; ++d) {
            for (int e = 0; e < E; ++e) {
                router_T[(size_t)d * router_E_padded + e] =
                    w.w_router[(size_t)e * D + d];
            }
            for (int e = E; e < router_E_padded; ++e)
                router_T[(size_t)d * router_E_padded + e] = 0.0f;
        }
    }
    packed_x86_ready = false;
    if (D % 64 == 0 && H % 16 == 0) {
        try {
            pack_weights_amx_token_rows(w.w_gate, E * H, D, packed_x86_gate);
            pack_weights_amx_token_rows(w.w_up, E * H, D, packed_x86_up);
            pack_weights_amx_token_rows(w.w_down, E * D, H, packed_x86_down);
            pack_weights_amx_token_rows(w.sh_gate, H, D, packed_x86_sh_gate);
            pack_weights_amx_token_rows(w.sh_up, H, D, packed_x86_sh_up);
            pack_weights_amx_token_rows(w.sh_down, D, H, packed_x86_sh_down);
            packed_x86_ready = true;
        } catch (const std::bad_alloc&) {
            packed_x86_gate.clear(); packed_x86_up.clear(); packed_x86_down.clear();
            packed_x86_sh_gate.clear(); packed_x86_sh_up.clear();
            packed_x86_sh_down.clear();
        }
    }
    auto compute_lane_sums = [](const std::vector<int8_t>& packed,
                                int O, int K,
                                std::vector<int32_t>& sums_lane) {
        int O_blocks = O / 16;
        int K_blocks = K / 64;
        sums_lane.resize(O);
        for (int ob = 0; ob < O_blocks; ++ob) {
            for (int kb = 0; kb < K_blocks; ++kb) {
                for (int k4 = 0; k4 < 16; ++k4) {
                    for (int lane = 0; lane < 16; ++lane) {
                        int32_t s = 0;
                        for (int r = 0; r < 4; ++r)
                            s += (int32_t)packed[
                                ((size_t)ob * K_blocks + kb) * 1024
                                + k4 * 64 + lane * 4 + r];
                        sums_lane[ob * 16 + lane] += s;
                    }
                }
            }
        }
    };
    if (packed_x86_ready) {
        gate_sums_lane.assign(E * H, 0);
        up_sums_lane.assign(E * H, 0);
        down_sums_lane.assign(E * D, 0);
        sh_gate_sums_lane.assign(H, 0);
        sh_up_sums_lane.assign(H, 0);
        sh_down_sums_lane.assign(D, 0);
        int H_blocks = H / 16, K_blocks_D = D / 64;
        int D_blocks = D / 16, K_blocks_H = H / 64;
        for (int e = 0; e < E; ++e) {
            const int8_t* eg = packed_x86_gate.data() + (size_t)e * H_blocks * K_blocks_D * 1024;
            const int8_t* eu = packed_x86_up.data()   + (size_t)e * H_blocks * K_blocks_D * 1024;
            const int8_t* ed = packed_x86_down.data() + (size_t)e * D_blocks * K_blocks_H * 1024;
            for (int ob = 0; ob < H_blocks; ++ob) {
                for (int kb = 0; kb < K_blocks_D; ++kb) {
                    for (int k4 = 0; k4 < 16; ++k4) {
                        for (int lane = 0; lane < 16; ++lane) {
                            int32_t sg = 0, su = 0;
                            for (int r = 0; r < 4; ++r) {
                                sg += (int32_t)eg[
                                    ((size_t)ob * K_blocks_D + kb) * 1024
                                    + k4 * 64 + lane * 4 + r];
                                su += (int32_t)eu[
                                    ((size_t)ob * K_blocks_D + kb) * 1024
                                    + k4 * 64 + lane * 4 + r];
                            }
                            gate_sums_lane[(size_t)e * H + ob * 16 + lane] += sg;
                            up_sums_lane[(size_t)e * H + ob * 16 + lane] += su;
                        }
                    }
                }
            }
            for (int ob = 0; ob < D_blocks; ++ob) {
                for (int kb = 0; kb < K_blocks_H; ++kb) {
                    for (int k4 = 0; k4 < 16; ++k4) {
                        for (int lane = 0; lane < 16; ++lane) {
                            int32_t sd = 0;
                            for (int r = 0; r < 4; ++r) {
                                sd += (int32_t)ed[
                                    ((size_t)ob * K_blocks_H + kb) * 1024
                                    + k4 * 64 + lane * 4 + r];
                            }
                            down_sums_lane[(size_t)e * D + ob * 16 + lane] += sd;
                        }
                    }
                }
            }
        }
        {
            const int8_t* sg = packed_x86_sh_gate.data();
            const int8_t* su = packed_x86_sh_up.data();
            const int8_t* sd = packed_x86_sh_down.data();
            for (int ob = 0; ob < H_blocks; ++ob) {
                for (int kb = 0; kb < K_blocks_D; ++kb) {
                    for (int k4 = 0; k4 < 16; ++k4) {
                        for (int lane = 0; lane < 16; ++lane) {
                            int32_t sgv = 0, suv = 0;
                            for (int r = 0; r < 4; ++r) {
                                sgv += (int32_t)sg[
                                    ((size_t)ob * K_blocks_D + kb) * 1024
                                    + k4 * 64 + lane * 4 + r];
                                suv += (int32_t)su[
                                    ((size_t)ob * K_blocks_D + kb) * 1024
                                    + k4 * 64 + lane * 4 + r];
                            }
                            sh_gate_sums_lane[ob * 16 + lane] += sgv;
                            sh_up_sums_lane[ob * 16 + lane] += suv;
                        }
                    }
                }
            }
            for (int ob = 0; ob < D_blocks; ++ob) {
                for (int kb = 0; kb < K_blocks_H; ++kb) {
                    for (int k4 = 0; k4 < 16; ++k4) {
                        for (int lane = 0; lane < 16; ++lane) {
                            int32_t sdv = 0;
                            for (int r = 0; r < 4; ++r) {
                                sdv += (int32_t)sd[
                                    ((size_t)ob * K_blocks_H + kb) * 1024
                                    + k4 * 64 + lane * 4 + r];
                            }
                            sh_down_sums_lane[ob * 16 + lane] += sdv;
                        }
                    }
                }
            }
        }
    }
#endif

#if MOE_RISCV_VECTOR
    packed_weights_ready = false;
    if (D % IME_K == 0 && H % IME_K == 0) {
        try {
            pack_weights_ime(w.w_gate, E * H, D, packed_gate);
            pack_weights_ime(w.w_up, E * H, D, packed_up);
            pack_weights_ime(w.w_down, E * D, H, packed_down);
            pack_weights_ime(w.sh_gate, H, D, packed_sh_gate);
            pack_weights_ime(w.sh_up, H, D, packed_sh_up);
            pack_weights_ime(w.sh_down, D, H, packed_sh_down);
            packed_weights_ready = true;
        } catch (const std::bad_alloc&) {
            packed_gate.clear(); packed_up.clear(); packed_down.clear();
            packed_sh_gate.clear(); packed_sh_up.clear();
            packed_sh_down.clear();
        }
    }
#endif

    init_sigmoid_table();
}

static inline float max_abs(const float* input, int length) {
#if MOE_X86
    const __m512i abs_mask_i = _mm512_set1_epi32(0x7fffffff);
    __m512 vmax = _mm512_setzero_ps();
    for (int i = 0; i < length; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        v = _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(v), abs_mask_i));
        vmax = _mm512_max_ps(vmax, v);
    }
    return _mm512_reduce_max_ps(vmax);
#elif MOE_RISCV_VECTOR
    float max_value = 0.0f;
    alignas(64) float temp[64];
    for (int i = 0; i < length;) {
        size_t vl = __riscv_vsetvl_e32m1(length - i);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(input + i, vl);
        vfloat32m1_t a = __riscv_vfabs_v_f32m1(v, vl);
        __riscv_vse32_v_f32m1(temp, a, vl);
        for (size_t j = 0; j < vl; ++j) max_value = std::max(max_value, temp[j]);
        i += (int)vl;
    }
    return max_value;
#else
    float max_value = 0.0f;
    for (int i = 0; i < length; ++i) {
        max_value = std::max(max_value, std::fabs(input[i]));
    }
    return max_value;
#endif
}

static float quantize_signed(const float* input, int length, int8_t* output) {
    float amax = max_abs(input, length);
    float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
#if MOE_X86
    __m512 inv_scale = _mm512_set1_ps(1.0f / scale);
    for (int i = 0; i < length; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        __m512i q32 = _mm512_cvtps_epi32(_mm512_mul_ps(v, inv_scale));
        __m128i q8 = _mm512_cvtsepi32_epi8(q32);
        _mm_storeu_si128((__m128i*)(output + i), q8);
    }
#else
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < length; ++i) {
        int q = (int)std::lrintf(input[i] * inv_scale);
        q = std::max(-128, std::min(127, q));
        output[i] = (int8_t)q;
    }
#endif
    return scale;
}

#if MOE_X86
static float quantize_signed_avx512(const float* input, int length, int8_t* output) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
    __m512 vmax = _mm512_setzero_ps();
    for (int i = 0; i < length; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        __m512 vabs = _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(v), abs_mask));
        vmax = _mm512_max_ps(vmax, vabs);
    }
    float amax = _mm512_reduce_max_ps(vmax);
    float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    __m512 inv_scale = _mm512_set1_ps(1.0f / scale);
    for (int i = 0; i < length; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        __m512i q32 = _mm512_cvtps_epi32(_mm512_mul_ps(v, inv_scale));
        __m128i q8 = _mm512_cvtsepi32_epi8(q32);
        _mm_storeu_si128((__m128i*)(output + i), q8);
    }
    return scale;
}

static inline void fast_silu_avx512_apply(__m512 gate_vec, __m512 up_vec,
                                          float gs, float us,
                                          float* hidden_out) {
    __m512 vg = _mm512_mul_ps(gate_vec, _mm512_set1_ps(gs));
    __m512 vu = _mm512_mul_ps(up_vec, _mm512_set1_ps(us));
    __m512 silu_vg = fast_silu_avx512(vg);
    __m512 hidden = _mm512_mul_ps(silu_vg, vu);
    _mm512_storeu_ps(hidden_out, hidden);
}
#endif

static inline void signed_to_biased_u8(const int8_t* input, uint8_t* output,
                                       int length) {
#if MOE_X86
    const __m512i sign = _mm512_set1_epi8((char)0x80);
    for (int i = 0; i < length; i += 64) {
        __m512i q = _mm512_loadu_si512(input + i);
        _mm512_storeu_si512(output + i, _mm512_xor_si512(q, sign));
    }
#else
    for (int i = 0; i < length; ++i) output[i] = (uint8_t)(input[i] + 128);
#endif
}

static inline void add_scaled(float* output, const float* input, float scale,
                                int length) {
#if MOE_X86
    __m512 s = _mm512_set1_ps(scale);
    for (int i = 0; i < length; i += 16) {
        __m512 y = _mm512_loadu_ps(output + i);
        __m512 v = _mm512_loadu_ps(input + i);
        _mm512_storeu_ps(output + i, _mm512_fmadd_ps(s, v, y));
    }
#elif MOE_RISCV_VECTOR
    for (int i = 0; i < length;) {
        size_t vl = __riscv_vsetvl_e32m1(length - i);
        vfloat32m1_t y = __riscv_vle32_v_f32m1(output + i, vl);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(input + i, vl);
        y = __riscv_vfmacc_vf_f32m1(y, scale, v, vl);
        __riscv_vse32_v_f32m1(output + i, y, vl);
        i += (int)vl;
    }
#else
    for (int i = 0; i < length; ++i) output[i] += scale * input[i];
#endif
}

static inline void scale_store(float* output, const float* input, float scale,
                                 int length) {
#if MOE_RISCV_VECTOR
    for (int i = 0; i < length;) {
        size_t vl = __riscv_vsetvl_e32m1(length - i);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(input + i, vl);
        v = __riscv_vfmul_vf_f32m1(v, scale, vl);
        __riscv_vse32_v_f32m1(output + i, v, vl);
        i += (int)vl;
    }
#else
    for (int i = 0; i < length; ++i) output[i] = scale * input[i];
#endif
}

static inline void copy_vector(float* output, const float* input, int length) {
#if MOE_X86
    for (int i = 0; i < length; i += 16)
        _mm512_storeu_ps(output + i, _mm512_loadu_ps(input + i));
#elif MOE_RISCV_VECTOR
    for (int i = 0; i < length;) {
        size_t vl = __riscv_vsetvl_e32m1(length - i);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(input + i, vl);
        __riscv_vse32_v_f32m1(output + i, v, vl);
        i += (int)vl;
    }
#else
    std::memcpy(output, input, (size_t)length * sizeof(float));
#endif
}

static __attribute__((noinline)) float router_dot(const float* weight,
                                                   const float* input, int D) {
#if MOE_X86
    __m512 acc = _mm512_setzero_ps();
    for (int d = 0; d < D; d += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(weight + d),
                              _mm512_loadu_ps(input + d), acc);
    }
    return _mm512_reduce_add_ps(acc);
#elif MOE_RISCV_VECTOR
    size_t vl_max = __riscv_vsetvlmax_e32m1();
    alignas(64) float zero_buf[8] = {};
    vfloat32m1_t vsum = __riscv_vle32_v_f32m1(zero_buf, vl_max);
    for (int d = 0; d < D;) {
        size_t vl = __riscv_vsetvl_e32m1(D - d);
        vfloat32m1_t wv = __riscv_vle32_v_f32m1(weight + d, vl);
        vfloat32m1_t xv = __riscv_vle32_v_f32m1(input + d, vl);
        vfloat32m1_t pv = __riscv_vfmul_vv_f32m1(wv, xv, vl);
        vsum = __riscv_vfmacc_vf_f32m1(vsum, 1.0f, pv, vl);
        d += (int)vl;
    }
    alignas(64) float temp[8];
    __riscv_vse32_v_f32m1(temp, vsum, vl_max);
    float acc = 0.0f;
    for (size_t i = 0; i < vl_max; ++i) acc += temp[i];
    return acc;
#else
    float acc = 0.0f;
    for (int d = 0; d < D; ++d) acc += weight[d] * input[d];
    return acc;
#endif
}

static inline void insert_topk(float score, float affinity, int expert, int K,
                               float* scores, float* affinities,
                               int* indices) {
    int position = K;
    for (int k = 0; k < K; ++k) {
        if (indices[k] < 0 || score > scores[k] ||
            (score == scores[k] && expert < indices[k])) {
            position = k;
            break;
        }
    }
    if (position == K) return;
    for (int k = K - 1; k > position; --k) {
        scores[k] = scores[k - 1];
        affinities[k] = affinities[k - 1];
        indices[k] = indices[k - 1];
    }
    scores[position] = score;
    affinities[position] = affinity;
    indices[position] = expert;
}

static void route_one(const float* input, const MoEWeights& w, int* top_idx,
                      float* top_gate) {
    float scores[MAX_TOP_K];
    for (int k = 0; k < w.top_k; ++k) {
        scores[k] = -std::numeric_limits<float>::infinity();
        top_gate[k] = 0.0f;
        top_idx[k] = -1;
    }
    for (int e = 0; e < w.num_experts; ++e) {
        float z = router_dot(w.w_router + (size_t)e * w.d_model, input,
                             w.d_model);
        float affinity = 1.0f / (1.0f + std::exp(-z));
        insert_topk(affinity + w.bias[e], affinity, e, w.top_k, scores,
                    top_gate, top_idx);
    }
    float total = 0.0f;
    for (int k = 0; k < w.top_k; ++k) total += top_gate[k];
    for (int k = 0; k < w.top_k; ++k) top_gate[k] /= total;
}

#if MOE_X86

static bool init_amx() {
    if (amx_ready) return true;
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0)
        return false;
    amx_ready = true;
    return true;
}

static bool configure_amx(AmxLayout layout, int M) {
    if (M < 1 || M > 16) return false;
    if (!amx_ready && !init_amx()) return false;
    if (layout == current_amx_layout && M == current_amx_m) return true;
    alignas(64) TileConfig config = {};
    config.palette_id = 1;
    if (layout == AmxLayout::TokenColumn) {
        config.rows[0] = 16;
        config.colsb[0] = 4 * M;
        config.rows[1] = 16;
        config.colsb[1] = 64;
        config.rows[2] = 16;
        config.colsb[2] = 4 * M;
        config.rows[3] = 16;
        config.colsb[3] = 4 * M;
        config.rows[4] = 16;
        config.colsb[4] = 64;
        config.rows[5] = 16;
        config.colsb[5] = 4 * M;
        config.rows[6] = 16;
        config.colsb[6] = 4 * M;
    } else {
        config.rows[0] = 16; config.colsb[0] = 64;
        config.rows[1] = M;  config.colsb[1] = 64;
        config.rows[2] = M;  config.colsb[2] = 64;
        config.rows[3] = M;  config.colsb[3] = 64;
        config.rows[4] = M;  config.colsb[4] = 64;
        config.rows[5] = M;  config.colsb[5] = 64;
        config.rows[6] = M;  config.colsb[6] = 64;
        config.rows[7] = M;  config.colsb[7] = 64;
    }
    _tile_loadconfig(&config);
    current_amx_layout = layout;
    current_amx_m = M;
    return true;
}

static bool configure_amx_for_m(int M) {
    return configure_amx(AmxLayout::TokenColumn, M);
}

static bool configure_amx_token_rows(int M) {
    return configure_amx(AmxLayout::TokenRow, M);
}

static inline void amx_gemm_token_rows(
    const int8_t* A, int a_stride,
    const int8_t* packed_B, int K, int O, int M,
    int32_t* C, int c_ld) {
    _tile_zero(2);
    int O_blocks = O / 16;
    int K_blocks = K / 64;
    for (int ob = 0; ob < O_blocks; ++ob) {
        for (int kb = 0; kb < K_blocks; ++kb) {
            _tile_loadd(1, A + (size_t)kb * 64, a_stride);
            _tile_loadd(0, packed_B + (size_t)(ob * K_blocks + kb) * 1024, 64);
            _tile_dpbssd(2, 1, 0);
        }
        _tile_stored(2, C + (size_t)ob * 16, (size_t)c_ld * sizeof(int32_t));
        if (ob + 1 < O_blocks) _tile_zero(2);
    }
}

static void pack_b_amx(const int8_t* source, int M, int cols,
                       int8_t* packed) {
    for (int b = 0; b < cols / 64; ++b) {
        int8_t* block = packed + (size_t)b * (64 * M);
        for (int k4 = 0; k4 < 16; ++k4) {
            for (int n = 0; n < M; ++n) {
                int8_t* destination = block + k4 * (4 * M) + n * 4;
                const int8_t* src = source + (size_t)n * cols +
                                    b * 64 + k4 * 4;
                uint32_t value;
                std::memcpy(&value, src, sizeof(value));
                std::memcpy(destination, &value, sizeof(value));
            }
        }
    }
}

static inline void amx_matmul(const int8_t* A, int a_stride,
                              const int8_t* packed_B, int K, int M,
                              int32_t* output) {
    int b_stride = 4 * M;
    _tile_zero(6);
    for (int k = 0; k < K; k += 64) {
        _tile_loadd(4, A + k, a_stride);
        _tile_loadd(5, packed_B + (size_t)(k / 64) * (64 * M), b_stride);
        _tile_dpbssd(6, 4, 5);
    }
    _tile_stored(6, output, b_stride);
}

static inline void amx_matmul_gate_up(const int8_t* gate_A,
                                       const int8_t* up_A, int a_stride,
                                       const int8_t* packed_B, int K,
                                       int M, int32_t* gate_out,
                                       int32_t* up_out) {
    int b_stride = 4 * M;
    _tile_zero(2);
    _tile_zero(3);
    for (int k = 0; k < K; k += 64) {
        _tile_loadd(0, packed_B + (size_t)(k / 64) * (64 * M), b_stride);
        _tile_loadd(1, gate_A + k, a_stride);
        _tile_dpbssd(2, 1, 0);
        _tile_loadd(1, up_A + k, a_stride);
        _tile_dpbssd(3, 1, 0);
    }
    _tile_stored(2, gate_out, b_stride);
    _tile_stored(3, up_out, b_stride);
}

static inline void dot4_vnni(const int8_t* weights, const uint8_t* input,
                             int length, const int32_t* sums,
                             int32_t* output) {
    __m512i a0 = _mm512_setzero_si512();
    __m512i a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512();
    __m512i a3 = _mm512_setzero_si512();
    for (int k = 0; k < length; k += 64) {
        __m512i x = _mm512_loadu_si512(input + k);
        a0 = _mm512_dpbusd_epi32(a0, x, _mm512_loadu_si512(weights + k));
        a1 = _mm512_dpbusd_epi32(
            a1, x, _mm512_loadu_si512(weights + length + k));
        a2 = _mm512_dpbusd_epi32(
            a2, x, _mm512_loadu_si512(weights + 2 * length + k));
        a3 = _mm512_dpbusd_epi32(
            a3, x, _mm512_loadu_si512(weights + 3 * length + k));
    }
    output[0] = _mm512_reduce_add_epi32(a0) - 128 * sums[0];
    output[1] = _mm512_reduce_add_epi32(a1) - 128 * sums[1];
    output[2] = _mm512_reduce_add_epi32(a2) - 128 * sums[2];
    output[3] = _mm512_reduce_add_epi32(a3) - 128 * sums[3];
}

static inline void gate_up4_vnni(
    const int8_t* gate, const int8_t* up, const uint8_t* input, int length,
    const int32_t* gate_sum, const int32_t* up_sum, int32_t* gate_out,
    int32_t* up_out) {
    __m512i g0 = _mm512_setzero_si512(), g1 = _mm512_setzero_si512();
    __m512i g2 = _mm512_setzero_si512(), g3 = _mm512_setzero_si512();
    __m512i u0 = _mm512_setzero_si512(), u1 = _mm512_setzero_si512();
    __m512i u2 = _mm512_setzero_si512(), u3 = _mm512_setzero_si512();
    for (int k = 0; k < length; k += 64) {
        __m512i x = _mm512_loadu_si512(input + k);
        g0 = _mm512_dpbusd_epi32(g0, x, _mm512_loadu_si512(gate + k));
        g1 = _mm512_dpbusd_epi32(
            g1, x, _mm512_loadu_si512(gate + length + k));
        g2 = _mm512_dpbusd_epi32(
            g2, x, _mm512_loadu_si512(gate + 2 * length + k));
        g3 = _mm512_dpbusd_epi32(
            g3, x, _mm512_loadu_si512(gate + 3 * length + k));
        u0 = _mm512_dpbusd_epi32(u0, x, _mm512_loadu_si512(up + k));
        u1 = _mm512_dpbusd_epi32(
            u1, x, _mm512_loadu_si512(up + length + k));
        u2 = _mm512_dpbusd_epi32(
            u2, x, _mm512_loadu_si512(up + 2 * length + k));
        u3 = _mm512_dpbusd_epi32(
            u3, x, _mm512_loadu_si512(up + 3 * length + k));
    }
    gate_out[0] = _mm512_reduce_add_epi32(g0) - 128 * gate_sum[0];
    gate_out[1] = _mm512_reduce_add_epi32(g1) - 128 * gate_sum[1];
    gate_out[2] = _mm512_reduce_add_epi32(g2) - 128 * gate_sum[2];
    gate_out[3] = _mm512_reduce_add_epi32(g3) - 128 * gate_sum[3];
    up_out[0] = _mm512_reduce_add_epi32(u0) - 128 * up_sum[0];
    up_out[1] = _mm512_reduce_add_epi32(u1) - 128 * up_sum[1];
    up_out[2] = _mm512_reduce_add_epi32(u2) - 128 * up_sum[2];
    up_out[3] = _mm512_reduce_add_epi32(u3) - 128 * up_sum[3];
}

static void expert_vnni(
    const int8_t* gate, const int8_t* up, const int8_t* down,
    const int32_t* gate_sum, const int32_t* up_sum,
    const int32_t* down_sum, float s_gate, float s_up, float s_down,
    const uint8_t* input, float s_x, float* output, int D, int H) {
    alignas(64) float hidden[MAX_D_FF];
    alignas(64) int8_t hidden_q[MAX_D_FF];
    alignas(64) uint8_t hidden_u[MAX_D_FF];

    for (int f = 0; f < H; f += 4) {
        int32_t acc_g[4], acc_u[4];
        gate_up4_vnni(gate + (size_t)f * D, up + (size_t)f * D, input, D,
                      gate_sum + f, up_sum + f, acc_g, acc_u);
        for (int j = 0; j < 4; ++j) {
            float vg = (float)acc_g[j] * (s_x * s_gate);
            float vu = (float)acc_u[j] * (s_x * s_up);
            hidden[f + j] = fast_silu(vg) * vu;
        }
    }

    float s_h = quantize_signed(hidden, H, hidden_q);
    signed_to_biased_u8(hidden_q, hidden_u, H);
    for (int d = 0; d < D; d += 4) {
        int32_t acc[4];
        dot4_vnni(down + (size_t)d * H, hidden_u, H, down_sum + d, acc);
        for (int j = 0; j < 4; ++j)
            output[d + j] = (float)acc[j] * (s_h * s_down);
    }
}

static void expert_amx(const int8_t* gate, const int8_t* up,
                       const int8_t* down, float s_gate, float s_up,
                       float s_down, const int8_t* input, const float* s_x,
                       int M, int D, int H, float* output) {
    alignas(64) int32_t acc_g[16 * 16];
    alignas(64) int32_t acc_u[16 * 16];
    alignas(64) int32_t acc_d[16 * 16];
    float s_h[16];

    configure_amx_for_m(M);
    pack_b_amx(input, M, D, ws().packed_input);
    for (int f = 0; f < H; f += 16) {
        amx_matmul_gate_up(gate + (size_t)f * D, up + (size_t)f * D,
                           D, ws().packed_input, D, M,
                           acc_g, acc_u);
        for (int n = 0; n < M; ++n) {
            float gate_scale = s_x[n] * s_gate;
            float up_scale = s_x[n] * s_up;
            for (int j = 0; j < 16; ++j) {
                float vg = (float)acc_g[j * M + n] * gate_scale;
                float vu = (float)acc_u[j * M + n] * up_scale;
                ws().hidden[(size_t)n * H + f + j] =
                    fast_silu(vg) * vu;
            }
        }
    }

    for (int n = 0; n < M; ++n) {
        s_h[n] = quantize_signed(
            ws().hidden + (size_t)n * H, H,
            ws().hidden_q + (size_t)n * H);
    }
    configure_amx_for_m(M);
    pack_b_amx(ws().hidden_q, M, H, ws().packed_hidden);

    alignas(64) int32_t gather_offsets[16];
    for (int j = 0; j < 16; ++j)
        gather_offsets[j] = j * M;
    __m512i row_offsets = _mm512_load_si512(gather_offsets);

    for (int d = 0; d < D; d += 16) {
        amx_matmul(down + (size_t)d * H, H, ws().packed_hidden, H, M,
                   acc_d);
        for (int n = 0; n < M; ++n) {
            __m512i offsets = _mm512_add_epi32(
                row_offsets, _mm512_set1_epi32(n));
            __m512i integers = _mm512_i32gather_epi32(
                offsets, acc_d, sizeof(int32_t));
            __m512 values = _mm512_mul_ps(
                _mm512_cvtepi32_ps(integers),
                _mm512_set1_ps(s_h[n] * s_down));
            _mm512_storeu_ps(output + (size_t)n * D + d, values);
        }
    }
}

static constexpr int TILE_N = 16;
static constexpr int ACC_STRIDE_BYTES = TILE_N * (int)sizeof(int32_t);

static inline const int8_t* packed_expert_ptr_token_row(
    const std::vector<int8_t>& packed, int expert, int O_blocks_per_expert,
    int K_blocks) {
    return packed.data() + (size_t)expert * O_blocks_per_expert * K_blocks * 1024;
}

static void expert_amx_token_rows(
    const int8_t* packed_gate_B, const int8_t* packed_up_B,
    const int8_t* packed_down_B,
    float s_gate, float s_up, float s_down,
    const int8_t* xq, const float* s_x,
    int M, int D, int H, float* output) {
    configure_amx_token_rows(M);
    alignas(64) int32_t gate_acc[16 * 16];
    alignas(64) int32_t up_acc[16 * 16];
    int H_blocks = H / 16;
    int K_blocks_D = D / 64;

    for (int ob = 0; ob < H_blocks; ++ob) {
        _tile_zero(2);
        _tile_zero(3);
        for (int kb = 0; kb < K_blocks_D; ++kb) {
            _tile_loadd(1, xq + (size_t)kb * 64, D);
            _tile_loadd(0, packed_gate_B + ((size_t)ob * K_blocks_D + kb) * 1024, 64);
            _tile_dpbssd(2, 1, 0);
            _tile_loadd(0, packed_up_B + ((size_t)ob * K_blocks_D + kb) * 1024, 64);
            _tile_dpbssd(3, 1, 0);
        }
        _tile_stored(2, gate_acc, ACC_STRIDE_BYTES);
        _tile_stored(3, up_acc, ACC_STRIDE_BYTES);
        for (int n = 0; n < M; ++n) {
            float gs = s_x[n] * s_gate;
            float us = s_x[n] * s_up;
            for (int j = 0; j < 16; ++j) {
                float vg = (float)gate_acc[n * TILE_N + j] * gs;
                float vu = (float)up_acc[n * TILE_N + j] * us;
                ws().hidden[(size_t)n * H + ob * 16 + j] = fast_silu(vg) * vu;
            }
        }
    }

    float s_h[16];
    for (int n = 0; n < M; ++n) {
        s_h[n] = quantize_signed(
            ws().hidden + (size_t)n * H, H,
            ws().hidden_q + (size_t)n * H);
    }

    alignas(64) int32_t down_acc[16 * 16];
    int K_blocks_H = H / 64;
    int D_blocks = D / 16;
    for (int ob = 0; ob < D_blocks; ++ob) {
        _tile_zero(2);
        for (int kb = 0; kb < K_blocks_H; ++kb) {
            _tile_loadd(1, ws().hidden_q + (size_t)kb * 64, H);
            _tile_loadd(0, packed_down_B + ((size_t)ob * K_blocks_H + kb) * 1024, 64);
            _tile_dpbssd(2, 1, 0);
        }
        _tile_stored(2, down_acc, ACC_STRIDE_BYTES);
        for (int n = 0; n < M; ++n) {
            float hs = s_h[n] * s_down;
            for (int j = 0; j < 16; ++j) {
                output[(size_t)n * D + ob * 16 + j] = (float)down_acc[n * TILE_N + j] * hs;
            }
        }
    }
}

static void expert_amx_token_rows_2ob(
    const int8_t* packed_gate_B, const int8_t* packed_up_B,
    const int8_t* packed_down_B,
    float s_gate, float s_up, float s_down,
    const int8_t* xq, const float* s_x,
    int M, int D, int H, float* output) {
    configure_amx_token_rows(M);
    int H_blocks = H / 16;
    int K_blocks_D = D / 64;
    int K_blocks_H = H / 64;
    int D_blocks = D / 16;

    int ob = 0;
    for (; ob + 1 < H_blocks; ob += 2) {
        _tile_zero(2); _tile_zero(3);
        _tile_zero(4); _tile_zero(5);
        for (int kb = 0; kb < K_blocks_D; ++kb) {
            _tile_loadd(1, xq + (size_t)kb * 64, D);
            _tile_loadd(0, packed_gate_B + ((size_t)ob * K_blocks_D + kb) * 1024, 64);
            _tile_dpbssd(2, 1, 0);
            _tile_loadd(0, packed_up_B + ((size_t)ob * K_blocks_D + kb) * 1024, 64);
            _tile_dpbssd(3, 1, 0);
            _tile_loadd(0, packed_gate_B + ((size_t)(ob + 1) * K_blocks_D + kb) * 1024, 64);
            _tile_dpbssd(4, 1, 0);
            _tile_loadd(0, packed_up_B + ((size_t)(ob + 1) * K_blocks_D + kb) * 1024, 64);
            _tile_dpbssd(5, 1, 0);
        }
        alignas(64) int32_t acc[4][16 * TILE_N];
        _tile_stored(2, acc[0], ACC_STRIDE_BYTES);
        _tile_stored(3, acc[1], ACC_STRIDE_BYTES);
        _tile_stored(4, acc[2], ACC_STRIDE_BYTES);
        _tile_stored(5, acc[3], ACC_STRIDE_BYTES);
        for (int n = 0; n < M; ++n) {
            float gs = s_x[n] * s_gate;
            float us = s_x[n] * s_up;
            for (int j = 0; j < 16; ++j) {
                float vg0 = (float)acc[0][n * TILE_N + j] * gs;
                float vu0 = (float)acc[1][n * TILE_N + j] * us;
                ws().hidden[(size_t)n * H + ob * 16 + j] =
                    fast_silu(vg0) * vu0;
                float vg1 = (float)acc[2][n * TILE_N + j] * gs;
                float vu1 = (float)acc[3][n * TILE_N + j] * us;
                ws().hidden[(size_t)n * H + (ob + 1) * 16 + j] =
                    fast_silu(vg1) * vu1;
            }
        }
    }
    for (; ob < H_blocks; ++ob) {
        _tile_zero(2); _tile_zero(3);
        for (int kb = 0; kb < K_blocks_D; ++kb) {
            _tile_loadd(1, xq + (size_t)kb * 64, D);
            _tile_loadd(0, packed_gate_B + ((size_t)ob * K_blocks_D + kb) * 1024, 64);
            _tile_dpbssd(2, 1, 0);
            _tile_loadd(0, packed_up_B + ((size_t)ob * K_blocks_D + kb) * 1024, 64);
            _tile_dpbssd(3, 1, 0);
        }
        alignas(64) int32_t gate_acc[16 * TILE_N];
        alignas(64) int32_t up_acc[16 * TILE_N];
        _tile_stored(2, gate_acc, ACC_STRIDE_BYTES);
        _tile_stored(3, up_acc, ACC_STRIDE_BYTES);
        for (int n = 0; n < M; ++n) {
            float gs = s_x[n] * s_gate;
            float us = s_x[n] * s_up;
            for (int j = 0; j < 16; ++j) {
                float vg = (float)gate_acc[n * TILE_N + j] * gs;
                float vu = (float)up_acc[n * TILE_N + j] * us;
                ws().hidden[(size_t)n * H + ob * 16 + j] =
                    fast_silu(vg) * vu;
            }
        }
    }

    float s_h[16];
    for (int n = 0; n < M; ++n) {
        s_h[n] = quantize_signed(
            ws().hidden + (size_t)n * H, H,
            ws().hidden_q + (size_t)n * H);
    }

    ob = 0;
    alignas(64) int32_t dacc[4][16 * TILE_N];
    for (; ob + 3 < D_blocks; ob += 4) {
        _tile_zero(2); _tile_zero(3);
        _tile_zero(4); _tile_zero(5);
        for (int kb = 0; kb < K_blocks_H; ++kb) {
            _tile_loadd(1, ws().hidden_q + (size_t)kb * 64, H);
            _tile_loadd(0, packed_down_B + ((size_t)ob * K_blocks_H + kb) * 1024, 64);
            _tile_dpbssd(2, 1, 0);
            _tile_loadd(0, packed_down_B + ((size_t)(ob + 1) * K_blocks_H + kb) * 1024, 64);
            _tile_dpbssd(3, 1, 0);
            _tile_loadd(0, packed_down_B + ((size_t)(ob + 2) * K_blocks_H + kb) * 1024, 64);
            _tile_dpbssd(4, 1, 0);
            _tile_loadd(0, packed_down_B + ((size_t)(ob + 3) * K_blocks_H + kb) * 1024, 64);
            _tile_dpbssd(5, 1, 0);
        }
        _tile_stored(2, dacc[0], ACC_STRIDE_BYTES);
        _tile_stored(3, dacc[1], ACC_STRIDE_BYTES);
        _tile_stored(4, dacc[2], ACC_STRIDE_BYTES);
        _tile_stored(5, dacc[3], ACC_STRIDE_BYTES);
        for (int n = 0; n < M; ++n) {
            float hs = s_h[n] * s_down;
            for (int j = 0; j < 16; ++j) {
                output[(size_t)n * D + ob * 16 + j] =
                    (float)dacc[0][n * TILE_N + j] * hs;
                output[(size_t)n * D + (ob + 1) * 16 + j] =
                    (float)dacc[1][n * TILE_N + j] * hs;
                output[(size_t)n * D + (ob + 2) * 16 + j] =
                    (float)dacc[2][n * TILE_N + j] * hs;
                output[(size_t)n * D + (ob + 3) * 16 + j] =
                    (float)dacc[3][n * TILE_N + j] * hs;
            }
        }
    }
    for (; ob + 1 < D_blocks; ob += 2) {
        _tile_zero(2); _tile_zero(4);
        for (int kb = 0; kb < K_blocks_H; ++kb) {
            _tile_loadd(1, ws().hidden_q + (size_t)kb * 64, H);
            _tile_loadd(0, packed_down_B + ((size_t)ob * K_blocks_H + kb) * 1024, 64);
            _tile_dpbssd(2, 1, 0);
            _tile_loadd(0, packed_down_B + ((size_t)(ob + 1) * K_blocks_H + kb) * 1024, 64);
            _tile_dpbssd(4, 1, 0);
        }
        _tile_stored(2, dacc[0], ACC_STRIDE_BYTES);
        _tile_stored(4, dacc[1], ACC_STRIDE_BYTES);
        for (int n = 0; n < M; ++n) {
            float hs = s_h[n] * s_down;
            for (int j = 0; j < 16; ++j) {
                output[(size_t)n * D + ob * 16 + j] =
                    (float)dacc[0][n * TILE_N + j] * hs;
                output[(size_t)n * D + (ob + 1) * 16 + j] =
                    (float)dacc[1][n * TILE_N + j] * hs;
            }
        }
    }
    for (; ob < D_blocks; ++ob) {
        _tile_zero(2);
        for (int kb = 0; kb < K_blocks_H; ++kb) {
            _tile_loadd(1, ws().hidden_q + (size_t)kb * 64, H);
            _tile_loadd(0, packed_down_B + ((size_t)ob * K_blocks_H + kb) * 1024, 64);
            _tile_dpbssd(2, 1, 0);
        }
        _tile_stored(2, dacc[0], ACC_STRIDE_BYTES);
        for (int n = 0; n < M; ++n) {
            float hs = s_h[n] * s_down;
            for (int j = 0; j < 16; ++j)
                output[(size_t)n * D + ob * 16 + j] =
                    (float)dacc[0][n * TILE_N + j] * hs;
        }
    }
}

static inline void gate_up_vnni_output_lane_1ob(
    const int8_t* packed_gate, const int8_t* packed_up,
    const uint8_t* input_u8, int K_blocks,
    const int32_t* gate_sums16, const int32_t* up_sums16,
    __m512i& gacc, __m512i& uacc) {
    gacc = _mm512_setzero_si512();
    uacc = _mm512_setzero_si512();
    for (int kb = 0; kb < K_blocks; ++kb) {
        const int8_t* tgate = packed_gate + (size_t)kb * 1024;
        const int8_t* tup   = packed_up   + (size_t)kb * 1024;
        const uint8_t* a = input_u8 + (size_t)kb * 64;
        for (int k4 = 0; k4 < 16; ++k4) {
            uint32_t a4;
            __builtin_memcpy(&a4, a + k4 * 4, 4);
            __m512i av = _mm512_set1_epi32((int)a4);
            __m512i bv_gate = _mm512_load_si512(tgate + k4 * 64);
            __m512i bv_up   = _mm512_load_si512(tup   + k4 * 64);
            gacc = _mm512_dpbusd_epi32(gacc, av, bv_gate);
            uacc = _mm512_dpbusd_epi32(uacc, av, bv_up);
        }
    }
    gacc = _mm512_sub_epi32(gacc,
        _mm512_mullo_epi32(_mm512_load_si512(gate_sums16),
                           _mm512_set1_epi32(128)));
    uacc = _mm512_sub_epi32(uacc,
        _mm512_mullo_epi32(_mm512_load_si512(up_sums16),
                           _mm512_set1_epi32(128)));
}

template<int OB>
static inline void gate_up_vnni_output_lane_multi(
    const int8_t* packed_gate, const int8_t* packed_up,
    const uint8_t* input_u8, int K_blocks,
    const int32_t* gate_sums16, const int32_t* up_sums16,
    __m512i* gacc_out, __m512i* uacc_out) {
    __m512i gacc[OB], uacc[OB];
    for (int ob = 0; ob < OB; ++ob) {
        gacc[ob] = _mm512_setzero_si512();
        uacc[ob] = _mm512_setzero_si512();
    }
    for (int kb = 0; kb < K_blocks; ++kb) {
        const uint8_t* a = input_u8 + (size_t)kb * 64;
        for (int k4 = 0; k4 < 16; ++k4) {
            uint32_t a4;
            __builtin_memcpy(&a4, a + k4 * 4, 4);
            __m512i av = _mm512_set1_epi32((int)a4);
            for (int ob = 0; ob < OB; ++ob) {
                gacc[ob] = _mm512_dpbusd_epi32(
                    gacc[ob], av,
                    _mm512_load_si512(
                        packed_gate + ((size_t)ob * K_blocks + kb) * 1024 + k4 * 64));
                uacc[ob] = _mm512_dpbusd_epi32(
                    uacc[ob], av,
                    _mm512_load_si512(
                        packed_up   + ((size_t)ob * K_blocks + kb) * 1024 + k4 * 64));
            }
        }
    }
    __m512i s128 = _mm512_set1_epi32(128);
    for (int ob = 0; ob < OB; ++ob) {
        gacc_out[ob] = _mm512_sub_epi32(gacc[ob],
            _mm512_mullo_epi32(_mm512_load_si512(gate_sums16 + ob * 16), s128));
        uacc_out[ob] = _mm512_sub_epi32(uacc[ob],
            _mm512_mullo_epi32(_mm512_load_si512(up_sums16 + ob * 16), s128));
    }
}

template<int OB>
static inline void down_vnni_output_lane_multi(
    const int8_t* packed_down, const uint8_t* hidden_u8,
    int K_blocks, const int32_t* down_sums16,
    __m512i* dacc_out) {
    __m512i dacc[OB];
    for (int ob = 0; ob < OB; ++ob)
        dacc[ob] = _mm512_setzero_si512();
    for (int kb = 0; kb < K_blocks; ++kb) {
        const uint8_t* a = hidden_u8 + (size_t)kb * 64;
        for (int k4 = 0; k4 < 16; ++k4) {
            uint32_t a4;
            __builtin_memcpy(&a4, a + k4 * 4, 4);
            __m512i av = _mm512_set1_epi32((int)a4);
            for (int ob = 0; ob < OB; ++ob) {
                dacc[ob] = _mm512_dpbusd_epi32(
                    dacc[ob], av,
                    _mm512_load_si512(
                        packed_down + ((size_t)ob * K_blocks + kb) * 1024 + k4 * 64));
            }
        }
    }
    __m512i s128 = _mm512_set1_epi32(128);
    for (int ob = 0; ob < OB; ++ob) {
        dacc_out[ob] = _mm512_sub_epi32(dacc[ob],
            _mm512_mullo_epi32(_mm512_load_si512(down_sums16 + ob * 16), s128));
    }
}

static inline void down_vnni_output_lane_1ob(
    const int8_t* packed_down, const uint8_t* hidden_u8,
    int K_blocks, const int32_t* down_sums16,
    __m512i& dacc) {
    dacc = _mm512_setzero_si512();
    for (int kb = 0; kb < K_blocks; ++kb) {
        const int8_t* tdown = packed_down + (size_t)kb * 1024;
        const uint8_t* a = hidden_u8 + (size_t)kb * 64;
        for (int k4 = 0; k4 < 16; ++k4) {
            uint32_t a4;
            __builtin_memcpy(&a4, a + k4 * 4, 4);
            __m512i av = _mm512_set1_epi32((int)a4);
            __m512i bv_down = _mm512_load_si512(tdown + k4 * 64);
            dacc = _mm512_dpbusd_epi32(dacc, av, bv_down);
        }
    }
    dacc = _mm512_sub_epi32(dacc,
        _mm512_mullo_epi32(_mm512_load_si512(down_sums16),
                           _mm512_set1_epi32(128)));
}

constexpr int VNNI_OB = 4;

static void expert_vnni_output_lane(
    const int8_t* packed_gate, const int8_t* packed_up,
    const int8_t* packed_down,
    const int32_t* gate_sums16, const int32_t* up_sums16,
    const int32_t* down_sums16,
    float s_gate, float s_up, float s_down,
    const uint8_t* input_u8, float s_x,
    int D, int H, float* output) {
    int H_blocks = H / 16;
    int K_blocks_D = D / 64;
    int K_blocks_H = H / 64;
    int D_blocks = D / 16;
    alignas(64) float hidden[MAX_D_FF];
    alignas(64) int8_t hidden_q[MAX_D_FF];
    alignas(64) uint8_t hidden_u[MAX_D_FF];

    int ob = 0;
    if (H_blocks >= VNNI_OB) {
        for (; ob + VNNI_OB - 1 < H_blocks; ob += VNNI_OB) {
            __m512i gacc[VNNI_OB], uacc[VNNI_OB];
            gate_up_vnni_output_lane_multi<VNNI_OB>(
                packed_gate + (size_t)ob * K_blocks_D * 1024,
                packed_up   + (size_t)ob * K_blocks_D * 1024,
                input_u8, K_blocks_D,
                gate_sums16 + ob * 16, up_sums16 + ob * 16,
                gacc, uacc);
            float gs = s_x * s_gate, us = s_x * s_up;
            for (int i = 0; i < VNNI_OB; ++i) {
                __m512 vg = _mm512_mul_ps(_mm512_cvtepi32_ps(gacc[i]), _mm512_set1_ps(gs));
                __m512 vu = _mm512_mul_ps(_mm512_cvtepi32_ps(uacc[i]), _mm512_set1_ps(us));
                _mm512_storeu_ps(hidden + (ob + i) * 16,
                    _mm512_mul_ps(fast_silu_avx512(vg), vu));
            }
        }
    }
    for (; ob < H_blocks; ++ob) {
        __m512i gacc, uacc;
        gate_up_vnni_output_lane_1ob(
            packed_gate + (size_t)ob * K_blocks_D * 1024,
            packed_up   + (size_t)ob * K_blocks_D * 1024,
            input_u8, K_blocks_D,
            gate_sums16 + ob * 16, up_sums16 + ob * 16,
            gacc, uacc);
        float gs = s_x * s_gate, us = s_x * s_up;
        __m512 vg = _mm512_mul_ps(_mm512_cvtepi32_ps(gacc), _mm512_set1_ps(gs));
        __m512 vu = _mm512_mul_ps(_mm512_cvtepi32_ps(uacc), _mm512_set1_ps(us));
        _mm512_storeu_ps(hidden + ob * 16,
            _mm512_mul_ps(fast_silu_avx512(vg), vu));
    }

    float s_h = quantize_signed(hidden, H, hidden_q);
    signed_to_biased_u8(hidden_q, hidden_u, H);

    ob = 0;
    if (D_blocks >= VNNI_OB) {
        for (; ob + VNNI_OB - 1 < D_blocks; ob += VNNI_OB) {
            __m512i dacc[VNNI_OB];
            down_vnni_output_lane_multi<VNNI_OB>(
                packed_down + (size_t)ob * K_blocks_H * 1024,
                hidden_u, K_blocks_H,
                down_sums16 + ob * 16, dacc);
            float hs = s_h * s_down;
            for (int i = 0; i < VNNI_OB; ++i) {
                __m512 vd = _mm512_cvtepi32_ps(dacc[i]);
                _mm512_storeu_ps(output + (ob + i) * 16,
                    _mm512_mul_ps(vd, _mm512_set1_ps(hs)));
            }
        }
    }
    for (; ob < D_blocks; ++ob) {
        __m512i dacc;
        down_vnni_output_lane_1ob(
            packed_down + (size_t)ob * K_blocks_H * 1024,
            hidden_u, K_blocks_H,
            down_sums16 + ob * 16, dacc);
        float hs = s_h * s_down;
        __m512 vd = _mm512_cvtepi32_ps(dacc);
        _mm512_storeu_ps(output + ob * 16,
            _mm512_mul_ps(vd, _mm512_set1_ps(hs)));
    }
}

static void expert_vnni_output_lane_multi_token(
    const int8_t* packed_gate, const int8_t* packed_up,
    const int8_t* packed_down,
    const int32_t* gate_sums16, const int32_t* up_sums16,
    const int32_t* down_sums16,
    float s_gate, float s_up, float s_down,
    const int8_t* block_input, const float* s_x_arr,
    int M, int D, int H, float* output) {
    int H_blocks = H / 16;
    int K_blocks_D = D / 64;
    int K_blocks_H = H / 64;
    int D_blocks = D / 16;

    for (int ob = 0; ob < H_blocks; ++ob) {
        __m512i gacc[16], uacc[16];
        for (int m = 0; m < M; ++m) {
            gacc[m] = _mm512_setzero_si512();
            uacc[m] = _mm512_setzero_si512();
        }
        for (int kb = 0; kb < K_blocks_D; ++kb) {
            for (int k4 = 0; k4 < 16; ++k4) {
                __m512i bv_gate = _mm512_load_si512(
                    packed_gate + ((size_t)ob * K_blocks_D + kb) * 1024 + k4 * 64);
                __m512i bv_up = _mm512_load_si512(
                    packed_up   + ((size_t)ob * K_blocks_D + kb) * 1024 + k4 * 64);
                for (int m = 0; m < M; ++m) {
                    uint32_t a4;
                    __builtin_memcpy(&a4, block_input + ((size_t)m * D + kb * 64 + k4 * 4), 4);
                    __m512i av = _mm512_set1_epi32((int)a4);
                    gacc[m] = _mm512_dpbusd_epi32(gacc[m], av, bv_gate);
                    uacc[m] = _mm512_dpbusd_epi32(uacc[m], av, bv_up);
                }
            }
        }
        __m512i s128 = _mm512_set1_epi32(128);
        for (int m = 0; m < M; ++m) {
            gacc[m] = _mm512_sub_epi32(gacc[m],
                _mm512_mullo_epi32(_mm512_load_si512(gate_sums16 + ob * 16), s128));
            uacc[m] = _mm512_sub_epi32(uacc[m],
                _mm512_mullo_epi32(_mm512_load_si512(up_sums16 + ob * 16), s128));
            float gs = s_x_arr[m] * s_gate;
            float us = s_x_arr[m] * s_up;
            __m512 vg = _mm512_mul_ps(_mm512_cvtepi32_ps(gacc[m]), _mm512_set1_ps(gs));
            __m512 vu = _mm512_mul_ps(_mm512_cvtepi32_ps(uacc[m]), _mm512_set1_ps(us));
            _mm512_storeu_ps(ws().hidden + (size_t)m * H + ob * 16,
                _mm512_mul_ps(fast_silu_avx512(vg), vu));
        }
    }

    float s_h[16];
    for (int m = 0; m < M; ++m) {
        s_h[m] = quantize_signed(ws().hidden + (size_t)m * H, H,
                                  ws().hidden_q + (size_t)m * H);
    }
    alignas(64) uint8_t hidden_u_buf[16 * MAX_D_FF];
    for (int m = 0; m < M; ++m)
        signed_to_biased_u8(ws().hidden_q + (size_t)m * H,
                             hidden_u_buf + (size_t)m * H, H);

    for (int ob = 0; ob < D_blocks; ++ob) {
        __m512i dacc[16];
        for (int m = 0; m < M; ++m)
            dacc[m] = _mm512_setzero_si512();
        for (int kb = 0; kb < K_blocks_H; ++kb) {
            for (int k4 = 0; k4 < 16; ++k4) {
                __m512i bv_down = _mm512_load_si512(
                    packed_down + ((size_t)ob * K_blocks_H + kb) * 1024 + k4 * 64);
                for (int m = 0; m < M; ++m) {
                    uint32_t a4;
                    __builtin_memcpy(&a4, hidden_u_buf + ((size_t)m * H + kb * 64 + k4 * 4), 4);
                    __m512i av = _mm512_set1_epi32((int)a4);
                    dacc[m] = _mm512_dpbusd_epi32(dacc[m], av, bv_down);
                }
            }
        }
        __m512i s128 = _mm512_set1_epi32(128);
        for (int m = 0; m < M; ++m) {
            float hs = s_h[m] * s_down;
            dacc[m] = _mm512_sub_epi32(dacc[m],
                _mm512_mullo_epi32(_mm512_load_si512(down_sums16 + ob * 16), s128));
            _mm512_storeu_ps(output + (size_t)m * D + ob * 16,
                _mm512_mul_ps(_mm512_cvtepi32_ps(dacc[m]), _mm512_set1_ps(hs)));
        }
    }
}

static void run_vnni_from_signed(
    const int8_t* gate, const int8_t* up, const int8_t* down,
    const int32_t* gate_sum, const int32_t* up_sum,
    const int32_t* down_sum, float s_gate, float s_up, float s_down,
    const int8_t* input, float s_x, float* output, int D, int H) {
    alignas(64) uint8_t input_u[MAX_D_MODEL];
    signed_to_biased_u8(input, input_u, D);
    expert_vnni(gate, up, down, gate_sum, up_sum, down_sum, s_gate, s_up,
                s_down, input_u, s_x, output, D, H);
}

static void route_batch_blocked(const float* x, const MoEWeights& w,
                                int num_tokens) {
    const int D = w.d_model;
    const int E = w.num_experts;
    const int K = w.top_k;

    for (int base = 0; base < num_tokens; base += ROUTER_TOKEN_BLOCK) {
        int M = std::min(ROUTER_TOKEN_BLOCK, num_tokens - base);
        float scores[ROUTER_TOKEN_BLOCK][MAX_TOP_K];
        float affinities[ROUTER_TOKEN_BLOCK][MAX_TOP_K];
        int indices[ROUTER_TOKEN_BLOCK][MAX_TOP_K];
        for (int n = 0; n < M; ++n) {
            for (int k = 0; k < K; ++k) {
                scores[n][k] = -std::numeric_limits<float>::infinity();
                affinities[n][k] = 0.0f;
                indices[n][k] = -1;
            }
        }

        for (int e = 0; e < E; ++e) {
            __m512 acc[ROUTER_TOKEN_BLOCK];
            for (int n = 0; n < M; ++n) acc[n] = _mm512_setzero_ps();
            const float* weight = w.w_router + (size_t)e * D;
            for (int d = 0; d < D; d += 16) {
                __m512 wr = _mm512_loadu_ps(weight + d);
                for (int n = 0; n < M; ++n) {
                    const float* xt = x + (size_t)(base + n) * D;
                    acc[n] = _mm512_fmadd_ps(
                        wr, _mm512_loadu_ps(xt + d), acc[n]);
                }
            }
            for (int n = 0; n < M; ++n) {
                float z = _mm512_reduce_add_ps(acc[n]);
                float affinity = 1.0f / (1.0f + std::exp(-z));
                insert_topk(affinity + w.bias[e], affinity, e, K,
                            scores[n], affinities[n], indices[n]);
            }
        }

        for (int n = 0; n < M; ++n) {
            float total = 0.0f;
            for (int k = 0; k < K; ++k) total += affinities[n][k];
            for (int k = 0; k < K; ++k) {
                size_t index = (size_t)(base + n) * K + k;
                workspace.top_idx[index] = indices[n][k];
                workspace.top_gate[index] = affinities[n][k] / total;
            }
        }
    }
}

static void route_batch_transposed(const float* x, const MoEWeights& w,
                                   int num_tokens) {
    const int D = w.d_model;
    const int E = w.num_experts;
    const int K = w.top_k;
    const int E_pad = router_E_padded;
    constexpr int TOKEN_BLOCK = 4;
    static thread_local std::vector<float> scores_buf;

    scores_buf.resize((size_t)num_tokens * E_pad);
    for (int base = 0; base < num_tokens; base += TOKEN_BLOCK) {
        int mb = std::min(TOKEN_BLOCK, num_tokens - base);
        for (int eb = 0; eb < E_pad; eb += 16) {
            int ve = std::min(16, E - eb);
            __m512 acc[TOKEN_BLOCK];
            for (int n = 0; n < mb; ++n) acc[n] = _mm512_setzero_ps();
            for (int d = 0; d < D; ++d) {
                __m512 w16 = _mm512_loadu_ps(
                    router_T.data() + (size_t)d * E_pad + eb);
                for (int n = 0; n < mb; ++n)
                    acc[n] = _mm512_fmadd_ps(
                        _mm512_set1_ps(x[((size_t)base + n) * D + d]),
                        w16, acc[n]);
            }
            for (int n = 0; n < mb; ++n)
                _mm512_storeu_ps(
                    scores_buf.data() + ((size_t)base + n) * E_pad + eb, acc[n]);
        }
    }

    for (int t = 0; t < num_tokens; ++t) {
        int top = t * K;
        float scores_arr[MAX_TOP_K];
        float affinities_arr[MAX_TOP_K];
        int indices_arr[MAX_TOP_K];
        for (int k = 0; k < K; ++k) {
            scores_arr[k] = -std::numeric_limits<float>::infinity();
            affinities_arr[k] = 0.0f;
            indices_arr[k] = -1;
        }
        for (int e = 0; e < E; ++e) {
            float z = scores_buf[(size_t)t * E_pad + e];
            float affinity = 1.0f / (1.0f + std::exp(-z));
            insert_topk(affinity + w.bias[e], affinity, e, K,
                        scores_arr, affinities_arr, indices_arr);
        }
        float total = 0.0f;
        for (int k = 0; k < K; ++k) total += affinities_arr[k];
        for (int k = 0; k < K; ++k) {
            workspace.top_idx[top + k] = indices_arr[k];
            workspace.top_gate[top + k] = affinities_arr[k] / total;
        }
    }
}

static void forward_single(const float* x, const MoEWeights& w, float* y) {
    const int D = w.d_model;
    const int H = w.d_ff;
    const int K = w.top_k;
    alignas(64) int8_t input_q[MAX_D_MODEL];
    alignas(64) uint8_t input_u8[MAX_D_MODEL];
    alignas(64) float expert_output[MAX_D_MODEL];
    int top_idx[MAX_TOP_K];
    float top_gate[MAX_TOP_K];

    route_one(x, w, top_idx, top_gate);
    float s_x = quantize_signed(x, D, input_q);
    signed_to_biased_u8(input_q, input_u8, D);

    int H_blocks = H / 16, K_blocks_D = D / 64;
    int D_blocks = D / 16, K_blocks_H = H / 64;

    bool use_lane_vnni = false; // debug
    if (use_lane_vnni) {
        expert_vnni_output_lane(
            packed_x86_sh_gate.data(), packed_x86_sh_up.data(),
            packed_x86_sh_down.data(),
            sh_gate_sums_lane.data(), sh_up_sums_lane.data(),
            sh_down_sums_lane.data(),
            w.sh_s_gate, w.sh_s_up, w.sh_s_down,
            input_u8, s_x, D, H, y);
        add_scaled(y, x, 1.0f, D);

        for (int k = 0; k < K; ++k) {
            int e = top_idx[k];
            size_t e_gate_off = (size_t)e * H_blocks * K_blocks_D * 1024;
            size_t e_down_off = (size_t)e * D_blocks * K_blocks_H * 1024;
            expert_vnni_output_lane(
                packed_x86_gate.data() + e_gate_off,
                packed_x86_up.data() + e_gate_off,
                packed_x86_down.data() + e_down_off,
                gate_sums_lane.data() + (size_t)e * H,
                up_sums_lane.data() + (size_t)e * H,
                down_sums_lane.data() + (size_t)e * D,
                w.s_gate[e], w.s_up[e], w.s_down[e],
                input_u8, s_x, D, H, expert_output);
            add_scaled(y, expert_output, top_gate[k], D);
        }
    } else {
        run_vnni_from_signed(
            w.sh_gate, w.sh_up, w.sh_down, sh_gate_sums.data(),
            sh_up_sums.data(), sh_down_sums.data(), w.sh_s_gate, w.sh_s_up,
            w.sh_s_down, input_q, s_x, expert_output, D, H);
        copy_vector(y, x, D);
        add_scaled(y, expert_output, 1.0f, D);

        for (int k = 0; k < K; ++k) {
            int e = top_idx[k];
            run_vnni_from_signed(
                w.w_gate + (size_t)e * H * D, w.w_up + (size_t)e * H * D,
                w.w_down + (size_t)e * D * H,
                gate_sums.data() + (size_t)e * H,
                up_sums.data() + (size_t)e * H,
                down_sums.data() + (size_t)e * D, w.s_gate[e], w.s_up[e],
                w.s_down[e], input_q, s_x, expert_output, D, H);
            add_scaled(y, expert_output, top_gate[k], D);
        }
    }
}

static void forward_single_token_row(const float* x, const MoEWeights& w, float* y) {
    const int D = w.d_model;
    const int H = w.d_ff;
    const int K = w.top_k;
    alignas(64) int8_t input_q[MAX_D_MODEL];
    alignas(64) float expert_output[MAX_D_MODEL];
    int top_idx[MAX_TOP_K];
    float top_gate[MAX_TOP_K];

    route_one(x, w, top_idx, top_gate);
    float s_x = quantize_signed(x, D, input_q);
    int H_blocks = H / 16;
    int K_blocks_D = D / 64;
    int K_blocks_H = H / 64;
    int D_blocks = D / 16;

    expert_amx_token_rows_2ob(
        packed_x86_sh_gate.data(), packed_x86_sh_up.data(),
        packed_x86_sh_down.data(),
        w.sh_s_gate, w.sh_s_up, w.sh_s_down,
        input_q, &s_x, 1, D, H, expert_output);
    copy_vector(y, x, D);
    add_scaled(y, expert_output, 1.0f, D);

    for (int k = 0; k < K; ++k) {
        int e = top_idx[k];
        const int8_t* r_gate = packed_expert_ptr_token_row(
            packed_x86_gate, e, H_blocks, K_blocks_D);
        const int8_t* r_up = packed_expert_ptr_token_row(
            packed_x86_up, e, H_blocks, K_blocks_D);
        const int8_t* r_down = packed_expert_ptr_token_row(
            packed_x86_down, e, D_blocks, K_blocks_H);
        expert_amx_token_rows_2ob(
            r_gate, r_up, r_down,
            w.s_gate[e], w.s_up[e], w.s_down[e],
            input_q, &s_x, 1, D, H, expert_output);
        add_scaled(y, expert_output, top_gate[k], D);
    }
}

static void forward_single_fused(const float* x, const MoEWeights& w, float* y) {
    const int D = w.d_model;
    const int H = w.d_ff;
    const int K = w.top_k;
    alignas(64) int8_t input_q[MAX_D_MODEL];
    alignas(64) uint8_t input_u8[MAX_D_MODEL];
    int top_idx[MAX_TOP_K];
    float top_gate[MAX_TOP_K];

    route_one(x, w, top_idx, top_gate);
    float s_x = quantize_signed(x, D, input_q);
    signed_to_biased_u8(input_q, input_u8, D);

    const int H_blocks = H / 16, K_blocks_D = D / 64;
    const int D_blocks = D / 16, K_blocks_H = H / 64;
    alignas(64) float expert_out[4][MAX_D_MODEL];

    expert_vnni_output_lane(
        packed_x86_sh_gate.data(), packed_x86_sh_up.data(),
        packed_x86_sh_down.data(),
        sh_gate_sums_lane.data(), sh_up_sums_lane.data(),
        sh_down_sums_lane.data(),
        w.sh_s_gate, w.sh_s_up, w.sh_s_down,
        input_u8, s_x, D, H, y);

    for (int k = 0; k < K; ++k) {
        int e = top_idx[k];
        expert_vnni_output_lane(
            packed_x86_gate.data() + (size_t)e * H_blocks * K_blocks_D * 1024,
            packed_x86_up.data()   + (size_t)e * H_blocks * K_blocks_D * 1024,
            packed_x86_down.data() + (size_t)e * D_blocks * K_blocks_H * 1024,
            gate_sums_lane.data() + (size_t)e * H,
            up_sums_lane.data()   + (size_t)e * H,
            down_sums_lane.data() + (size_t)e * D,
            w.s_gate[e], w.s_up[e], w.s_down[e],
            input_u8, s_x, D, H, expert_out[k]);
    }

    add_scaled(y, x, 1.0f, D);
    for (int k = 0; k < K; ++k)
        add_scaled(y, expert_out[k], top_gate[k], D);
}

#endif

#if MOE_RISCV_VECTOR

static void ime_mma_batch(const uint8_t* a_batch,
                            const int8_t* b_batch, int k_blocks,
                            int32_t* c) {
    if (k_blocks <= 0) {
        std::memset(c, 0, IME_M * IME_N * sizeof(int32_t));
        return;
    }
    const uint8_t* a_ptr = a_batch;
    const int8_t* b_ptr = b_batch;
    int count = k_blocks;
    asm volatile(
        "vsetvli t0, x0, e32, m2, tu, mu\n"
        "vmv.v.i v2, 0\n"
        "vsetvli t0, x0, e8, m1, tu, mu\n"
        "1:\n"
        "vle8.v v0, (%[a])\n"
        "vle8.v v1, (%[b])\n"
        ".word 0xe210112b\n"
        "addi %[a], %[a], 32\n"
        "addi %[b], %[b], 32\n"
        "addi %[count], %[count], -1\n"
        "bnez %[count], 1b\n"
        "vsetvli t0, x0, e32, m2, tu, mu\n"
        "vse32.v v2, (%[c])\n"
        : [a] "+r"(a_ptr), [b] "+r"(b_ptr), [count] "+r"(count)
        : [c] "r"(c)
        : "t0", "memory", "v0", "v1", "v2", "v3"
    );
}

static void gather_a_tiles(const uint8_t* input, int input_stride, int M,
                            int cols, uint8_t* a_batch) {
    const int k_blocks = (cols + IME_K - 1) / IME_K;
    for (int kb = 0; kb < k_blocks; ++kb) {
        for (int m = 0; m < IME_M; ++m) {
            uint8_t* dst = a_batch + ((size_t)kb * IME_M + m) * IME_K;
            if (m < M) {
                std::memcpy(dst,
                            input + (size_t)m * input_stride + (size_t)kb * IME_K,
                            IME_K);
            } else {
                std::memset(dst, 128, IME_K);
            }
        }
    }
}

static void ime_dot_batched(const uint8_t* a_batch, int k_blocks,
                            const int8_t* packed_weight, int M, int rows,
                            const int32_t* row_sums, int32_t* output) {
    std::fill(output, output + (size_t)M * rows, 0);
    for (int r = 0; r < rows; r += IME_N) {
        const int8_t* b_base = packed_weight +
            ((size_t)(r / IME_N) * k_blocks) * IME_TILE_BYTES;
        alignas(64) int32_t tile[IME_M * IME_N];
        ime_mma_batch(a_batch, b_base, k_blocks, tile);
        for (int m = 0; m < M; ++m) {
            for (int j = 0; j < IME_N && r + j < rows; ++j) {
                output[(size_t)m * rows + r + j] += tile[m * IME_N + j];
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < rows; ++j) {
            output[(size_t)m * rows + j] -= 128 * row_sums[j];
        }
    }
}

static inline const int8_t* packed_expert_ptr(const std::vector<int8_t>& all,
                                               int expert, int rows, int cols) {
    size_t per = (size_t)((rows + IME_N - 1) / IME_N) *
                 ((cols + IME_K - 1) / IME_K) * IME_TILE_BYTES;
    return all.data() + (size_t)expert * per;
}

static void expert_ime_packed(const int8_t* gate, const int8_t* up,
                               const int8_t* down, const int32_t* gate_sum,
                               const int32_t* up_sum,
                               const int32_t* down_sum, float s_gate,
                               float s_up, float s_down, const uint8_t* input,
                               const float* s_x, int M, int D, int H,
                               float* output) {
    alignas(64) int32_t acc_g[IME_M * MAX_D_FF];
    alignas(64) int32_t acc_u[IME_M * MAX_D_FF];
    alignas(64) int32_t acc_d[IME_M * MAX_D_MODEL];
    float s_h[IME_M];

    const int k_blocks_in = (D + IME_K - 1) / IME_K;
    const int k_blocks_hid = (H + IME_K - 1) / IME_K;

    alignas(64) uint8_t a_batch_in[MAX_K_BLOCKS * IME_M * IME_K];
    gather_a_tiles(input, D, M, D, a_batch_in);

    ime_dot_batched(a_batch_in, k_blocks_in, gate, M, H, gate_sum, acc_g);
    ime_dot_batched(a_batch_in, k_blocks_in, up, M, H, up_sum, acc_u);

    for (int m = 0; m < M; ++m) {
        float gate_scale = s_x[m] * s_gate;
        float up_scale = s_x[m] * s_up;
        for (int f = 0; f < H; ++f) {
            float vg = (float)acc_g[(size_t)m * H + f] * gate_scale;
            float vu = (float)acc_u[(size_t)m * H + f] * up_scale;
            ws().hidden[(size_t)m * H + f] = fast_silu(vg) * vu;
        }
        s_h[m] = quantize_signed(ws().hidden + (size_t)m * H, H,
                                  ws().hidden_q + (size_t)m * H);
        signed_to_biased_u8(ws().hidden_q + (size_t)m * H,
                            ws().hidden_u + (size_t)m * H, H);
    }
    if (M < IME_M) {
        std::memset(ws().hidden_u + (size_t)M * H, 128,
                    (size_t)(IME_M - M) * H);
    }

    alignas(64) uint8_t a_batch_hid[MAX_K_BLOCKS * IME_M * IME_K];
    gather_a_tiles(ws().hidden_u, H, M, H, a_batch_hid);

    ime_dot_batched(a_batch_hid, k_blocks_hid, down, M, D, down_sum, acc_d);
    for (int m = 0; m < M; ++m) {
        float scale = s_h[m] * s_down;
        for (int d = 0; d < D; ++d) {
            output[(size_t)m * D + d] = (float)acc_d[(size_t)m * D + d] * scale;
        }
    }
}

#endif

static void dot4_scalar(const int8_t* weights, const uint8_t* input,
                        int length, const int32_t* sums, int32_t* output) {
    for (int j = 0; j < IME_N; ++j) {
        int32_t acc = 0;
        const int8_t* row = weights + (size_t)j * length;
        for (int k = 0; k < length; ++k) {
            acc += (int32_t)row[k] * ((int32_t)input[k] - 128);
        }
        output[j] = acc;
        (void)sums;
    }
}

static void gate_up4_scalar(const int8_t* gate, const int8_t* up,
                            const uint8_t* input, int length,
                            const int32_t* gate_sum, const int32_t* up_sum,
                            int32_t* gate_out, int32_t* up_out) {
    dot4_scalar(gate, input, length, gate_sum, gate_out);
    dot4_scalar(up, input, length, up_sum, up_out);
}

static void expert_scalar(const int8_t* gate, const int8_t* up,
                          const int8_t* down, const int32_t* gate_sum,
                          const int32_t* up_sum, const int32_t* down_sum,
                          float s_gate, float s_up, float s_down,
                          const uint8_t* input, float s_x, float* output,
                          int D, int H) {
    for (int f = 0; f < H; f += IME_N) {
        int32_t acc_g[IME_N], acc_u[IME_N];
        gate_up4_scalar(gate + (size_t)f * D, up + (size_t)f * D, input, D,
                         gate_sum + f, up_sum + f, acc_g, acc_u);
        for (int j = 0; j < IME_N; ++j) {
            float vg = (float)acc_g[j] * (s_x * s_gate);
            float vu = (float)acc_u[j] * (s_x * s_up);
            ws().hidden[f + j] = fast_silu(vg) * vu;
        }
    }

    float s_h = quantize_signed(ws().hidden, H, ws().hidden_q);
    signed_to_biased_u8(ws().hidden_q, ws().hidden_u, H);
    for (int d = 0; d < D; d += IME_N) {
        int32_t acc[IME_N];
        dot4_scalar(down + (size_t)d * H, ws().hidden_u, H,
                     down_sum + d, acc);
        for (int j = 0; j < IME_N; ++j) {
            output[d + j] = (float)acc[j] * (s_h * s_down);
        }
    }
}

#if MOE_RISCV_VECTOR
static void run_expert(const int8_t* gate, const int8_t* up,
                       const int8_t* down, const int32_t* gate_sum,
                       const int32_t* up_sum, const int32_t* down_sum,
                       float s_gate, float s_up, float s_down,
                       const uint8_t* input, const float* s_x, int M, int D,
                       int H, float* output, int expert = -1) {
    if (packed_weights_ready) {
        const int8_t* pg = gate;
        const int8_t* pu = up;
        const int8_t* pd = down;
        const int32_t* gs = gate_sum;
        const int32_t* us = up_sum;
        const int32_t* ds = down_sum;
        if (expert >= 0) {
            pg = packed_expert_ptr(packed_gate, expert, H, D);
            pu = packed_expert_ptr(packed_up, expert, H, D);
            pd = packed_expert_ptr(packed_down, expert, D, H);
            gs = gate_sums.data() + (size_t)expert * H;
            us = up_sums.data() + (size_t)expert * H;
            ds = down_sums.data() + (size_t)expert * D;
        }
        expert_ime_packed(pg, pu, pd, gs, us, ds, s_gate, s_up, s_down,
                          input, s_x, M, D, H, output);
        return;
    }
    for (int m = 0; m < M; ++m) {
        expert_scalar(gate, up, down, gate_sum, up_sum, down_sum, s_gate,
                      s_up, s_down, input + (size_t)m * D, s_x[m],
                      output + (size_t)m * D, D, H);
    }
}
#else
static void run_expert(const int8_t* gate, const int8_t* up,
                       const int8_t* down, const int32_t* gate_sum,
                       const int32_t* up_sum, const int32_t* down_sum,
                       float s_gate, float s_up, float s_down,
                       const uint8_t* input, const float* s_x, int M, int D,
                       int H, float* output, int expert = -1) {
    for (int m = 0; m < M; ++m) {
        expert_scalar(gate, up, down, gate_sum, up_sum, down_sum, s_gate,
                      s_up, s_down, input + (size_t)m * D, s_x[m],
                      output + (size_t)m * D, D, H);
    }
}
#endif

#if !MOE_X86
static void forward_one(const float* x, const MoEWeights& w, float* y) {
    const int D = w.d_model;
    const int H = w.d_ff;
    const int K = w.top_k;
    int top_idx[MAX_TOP_K];
    float top_gate[MAX_TOP_K];

    route_one(x, w, top_idx, top_gate);
    float s_x = quantize_signed(x, D, ws().xq);
    signed_to_biased_u8(ws().xq, ws().xu, D);

    const float sx[1] = {s_x};
#if MOE_RISCV_VECTOR
    if (packed_weights_ready) {
        run_expert(packed_sh_gate.data(), packed_sh_up.data(),
                   packed_sh_down.data(), sh_gate_sums.data(),
                   sh_up_sums.data(), sh_down_sums.data(), w.sh_s_gate,
                   w.sh_s_up, w.sh_s_down, ws().xu, sx, 1, D, H,
                   ws().expert_output);
    } else
#endif
    {
        run_expert(w.sh_gate, w.sh_up, w.sh_down, sh_gate_sums.data(),
                    sh_up_sums.data(), sh_down_sums.data(), w.sh_s_gate,
                    w.sh_s_up, w.sh_s_down, ws().xu, sx, 1, D, H,
                    ws().expert_output);
    }
    copy_vector(y, x, D);
    add_scaled(y, ws().expert_output, 1.0f, D);

    for (int k = 0; k < K; ++k) {
        int e = top_idx[k];
#if MOE_RISCV_VECTOR
        if (packed_weights_ready) {
            run_expert(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       w.s_gate[e], w.s_up[e], w.s_down[e], ws().xu, sx,
                       1, D, H, ws().expert_output, e);
        } else
#endif
        {
            run_expert(w.w_gate + (size_t)e * H * D,
                        w.w_up + (size_t)e * H * D,
                        w.w_down + (size_t)e * D * H,
                        gate_sums.data() + (size_t)e * H,
                        up_sums.data() + (size_t)e * H,
                        down_sums.data() + (size_t)e * D, w.s_gate[e],
                        w.s_up[e], w.s_down[e], ws().xu, sx, 1, D, H,
                        ws().expert_output);
        }
        add_scaled(y, ws().expert_output, top_gate[k], D);
    }
}
#endif

static int requested_threads_for_shape(const MoEWeights& w, int num_tokens) {
    const char* env = std::getenv("MOE_NUM_THREADS");
    int requested = env ? std::atoi(env) : 0;
    if (requested <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        requested = hw > 0 ? (int)hw : 4;
#if MOE_X86
        requested = std::min(requested, 24);
#else
        requested = std::min(requested, 2);
#endif
    }
    requested = std::max(1, std::min(requested, num_tokens));
    if (num_tokens < 128 || w.num_experts < 128) return 1;
    return requested;
}

#if !MOE_X86
static void prepare_tokens_parallel(const float* x, const MoEWeights& w,
                                     int num_tokens, int threads) {
    const int D = w.d_model;
    const int K = w.top_k;
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int tid = 0; tid < threads; ++tid) {
        int begin = (int)((int64_t)num_tokens * tid / threads);
        int end = (int)((int64_t)num_tokens * (tid + 1) / threads);
        workers.emplace_back([&, begin, end]() {
            active_workspace = &tls_workspace;
            for (int t = begin; t < end; ++t) {
                route_one(x + (size_t)t * D, w,
                           workspace.top_idx + (size_t)t * K,
                           workspace.top_gate + (size_t)t * K);
                workspace.s_x[t] = quantize_signed(
                    x + (size_t)t * D, D, workspace.xq + (size_t)t * D);
                signed_to_biased_u8(workspace.xq + (size_t)t * D,
                                    workspace.xu + (size_t)t * D, D);
            }
        });
    }
    for (auto& worker : workers) worker.join();
    active_workspace = &workspace;
}
#endif

#if MOE_X86

static void moe_forward_x86(const float* x, const MoEWeights& w, float* y,
                            int num_tokens) {
    const int D = w.d_model;
    const int H = w.d_ff;
    const int E = w.num_experts;
    const int K = w.top_k;

    if (num_tokens == 1) {
        if (packed_x86_ready && D % 64 == 0 && H % 16 == 0)
            forward_single_fused(x, w, y);
        else
            forward_single(x, w, y);
        return;
    }

    if (!router_T.empty() && E >= 128) {
        route_batch_transposed(x, w, num_tokens);
    } else {
        route_batch_blocked(x, w, num_tokens);
    }
    std::fill(workspace.expert_count, workspace.expert_count + E, 0);
    for (int t = 0; t < num_tokens; ++t) {
        workspace.s_x[t] = quantize_signed(
            x + (size_t)t * D, D, workspace.xq + (size_t)t * D);
        for (int k = 0; k < K; ++k)
            ++workspace.expert_count[workspace.top_idx[(size_t)t * K + k]];
    }

    workspace.expert_offset[0] = 0;
    for (int e = 0; e < E; ++e) {
        workspace.expert_offset[e + 1] =
            workspace.expert_offset[e] + workspace.expert_count[e];
        workspace.fill_pos[e] = workspace.expert_offset[e];
    }
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < K; ++k) {
            size_t top = (size_t)t * K + k;
            int e = workspace.top_idx[top];
            int position = workspace.fill_pos[e]++;
            workspace.token_list[position] = t;
            workspace.token_gate[position] = workspace.top_gate[top];
            workspace.assignment_slot[top] = position;
        }
    }

    for (int t = 0; t < num_tokens; ++t)
        copy_vector(y + (size_t)t * D, x + (size_t)t * D, D);

    int requested = 0;
    const char* env = std::getenv("MOE_NUM_THREADS");
    if (env) requested = std::atoi(env);
    if (requested <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        requested = hw > 0 ? (int)hw : 4;
        requested = std::min(requested, 24);
    }
    requested = std::max(1, requested);
    int threads = requested;
    if (num_tokens >= 128) {
        if (E > 0 && E < 64)
            threads = std::min(threads, 4);
        else
            threads = std::min(threads, 8);
    } else {
        threads = 1;
    }

    bool use_amx = init_amx();

    int nonempty_count = 0;
    for (int e = 0; e < E; ++e) {
        if (workspace.expert_offset[e + 1] > workspace.expert_offset[e])
            workspace.nonempty_experts[nonempty_count++] = e;
    }

    bool use_token_row = packed_x86_ready && D % 64 == 0 && H % 16 == 0;
    if (threads == 1) {
        alignas(64) float shared_out[16 * MAX_D_MODEL];
        if (use_token_row && num_tokens >= 1) {
            for (int base = 0; base < num_tokens; base += 16) {
                int M = std::min(16, num_tokens - base);
                expert_amx_token_rows_2ob(
                    packed_x86_sh_gate.data(), packed_x86_sh_up.data(),
                    packed_x86_sh_down.data(),
                    w.sh_s_gate, w.sh_s_up, w.sh_s_down,
                    workspace.xq + (size_t)base * D,
                    workspace.s_x + base, M, D, H,
                    shared_out);
                for (int n = 0; n < M; ++n)
                    add_scaled(y + (size_t)(base + n) * D,
                               shared_out + (size_t)n * D, 1.0f, D);
            }
        } else if (use_amx && num_tokens >= AMX_THRESHOLD) {
            for (int base = 0; base < num_tokens; base += 16) {
                int M = std::min(16, num_tokens - base);
                expert_amx(w.sh_gate, w.sh_up, w.sh_down, w.sh_s_gate,
                           w.sh_s_up, w.sh_s_down,
                           workspace.xq + (size_t)base * D,
                           workspace.s_x + base, M, D, H,
                           workspace.block_output);
                for (int n = 0; n < M; ++n)
                    add_scaled(y + (size_t)(base + n) * D,
                               workspace.block_output + (size_t)n * D,
                               1.0f, D);
            }
        } else {
            alignas(64) float output[MAX_D_MODEL];
            for (int t = 0; t < num_tokens; ++t) {
                run_vnni_from_signed(
                    w.sh_gate, w.sh_up, w.sh_down, sh_gate_sums.data(),
                    sh_up_sums.data(), sh_down_sums.data(), w.sh_s_gate,
                    w.sh_s_up, w.sh_s_down,
                    workspace.xq + (size_t)t * D,
                    workspace.s_x[t], output, D, H);
                add_scaled(y + (size_t)t * D, output, 1.0f, D);
            }
        }

        int H_blocks = H / 16, K_blocks_D = D / 64;
        int D_blocks = D / 16, K_blocks_H = H / 64;
        alignas(64) float routed_out[16 * MAX_D_MODEL];
        for (int task = 0; task < nonempty_count; ++task) {
            int e = workspace.nonempty_experts[task];
            int begin = workspace.expert_offset[e];
            int end = workspace.expert_offset[e + 1];
            int count = end - begin;
            size_t e_gate_off = (size_t)e * H_blocks * K_blocks_D * 1024;
            size_t e_down_off = (size_t)e * D_blocks * K_blocks_H * 1024;
            if (use_token_row && count >= 1) {
                for (int base = begin; base < end; base += 16) {
                    int M = std::min(16, end - base);
                    for (int n = 0; n < M; ++n) {
                        int t = workspace.token_list[base + n];
                        std::memcpy(
                            workspace.block_input + (size_t)n * D,
                            workspace.xq + (size_t)t * D, D);
                    }
                    float block_scales[16];
                    for (int n = 0; n < M; ++n)
                        block_scales[n] = workspace.s_x[workspace.token_list[base + n]];
                    expert_amx_token_rows_2ob(
                        packed_x86_gate.data() + e_gate_off,
                        packed_x86_up.data() + e_gate_off,
                        packed_x86_down.data() + e_down_off,
                        w.s_gate[e], w.s_up[e], w.s_down[e],
                        workspace.block_input, block_scales, M, D, H,
                        routed_out);
                    for (int n = 0; n < M; ++n)
                        std::memcpy(
                            assignment_output + (size_t)(base + n) * D,
                            routed_out + (size_t)n * D, (size_t)D * sizeof(float));
                }
            } else if (use_amx && count >= AMX_THRESHOLD) {
                const int8_t* gate = w.w_gate + (size_t)e * H * D;
                const int8_t* up = w.w_up + (size_t)e * H * D;
                const int8_t* down = w.w_down + (size_t)e * D * H;
                for (int base = begin; base < end; base += 16) {
                    int M = std::min(16, end - base);
                    for (int n = 0; n < M; ++n) {
                        int t = workspace.token_list[base + n];
                        std::memcpy(
                            workspace.block_input + (size_t)n * D,
                            workspace.xq + (size_t)t * D, D);
                    }
                    float block_scales[16];
                    for (int n = 0; n < M; ++n) {
                        int t = workspace.token_list[base + n];
                        block_scales[n] = workspace.s_x[t];
                    }
                    expert_amx(gate, up, down, w.s_gate[e], w.s_up[e],
                               w.s_down[e], workspace.block_input,
                               block_scales, M, D, H,
                               assignment_output + (size_t)base * D);
                }
            } else {
                const int8_t* gate = w.w_gate + (size_t)e * H * D;
                const int8_t* up = w.w_up + (size_t)e * H * D;
                const int8_t* down = w.w_down + (size_t)e * D * H;
                for (int position = begin; position < end; ++position) {
                    int t = workspace.token_list[position];
                    run_vnni_from_signed(
                        gate, up, down,
                        gate_sums.data() + (size_t)e * H,
                        up_sums.data() + (size_t)e * H,
                        down_sums.data() + (size_t)e * D, w.s_gate[e],
                        w.s_up[e], w.s_down[e],
                        workspace.xq + (size_t)t * D, workspace.s_x[t],
                        assignment_output + (size_t)position * D, D, H);
                }
            }
        }

        for (int t = 0; t < num_tokens; ++t) {
            float* yt = y + (size_t)t * D;
            for (int k = 0; k < K; ++k) {
                size_t top = (size_t)t * K + k;
                int position = workspace.assignment_slot[top];
                const float* expert_y =
                    assignment_output + (size_t)position * D;
                add_scaled(yt, expert_y, workspace.top_gate[top], D);
            }
        }
    } else {
#pragma omp parallel num_threads(threads)
        {
            init_amx();
            active_workspace = &tls_workspace;

            bool use_token_row = packed_x86_ready && D % 64 == 0 && H % 16 == 0;
            if (use_token_row && num_tokens >= 1) {
#pragma omp for schedule(static)
                for (int base = 0; base < num_tokens; base += 16) {
                    int M = std::min(16, num_tokens - base);
                    expert_amx_token_rows_2ob(
                        packed_x86_sh_gate.data(), packed_x86_sh_up.data(),
                        packed_x86_sh_down.data(),
                        w.sh_s_gate, w.sh_s_up, w.sh_s_down,
                        workspace.xq + (size_t)base * D,
                        workspace.s_x + base, M, D, H,
                        ws().block_output);
                    for (int n = 0; n < M; ++n)
                        add_scaled(
                            y + (size_t)(base + n) * D,
                            ws().block_output + (size_t)n * D, 1.0f, D);
                }
            } else if (use_amx && num_tokens >= AMX_THRESHOLD) {
#pragma omp for schedule(static)
                for (int base = 0; base < num_tokens; base += 16) {
                    int M = std::min(16, num_tokens - base);
                    expert_amx(w.sh_gate, w.sh_up, w.sh_down,
                               w.sh_s_gate, w.sh_s_up, w.sh_s_down,
                               workspace.xq + (size_t)base * D,
                               workspace.s_x + base, M, D, H,
                               ws().block_output);
                    for (int n = 0; n < M; ++n)
                        add_scaled(
                            y + (size_t)(base + n) * D,
                            ws().block_output + (size_t)n * D, 1.0f, D);
                }
            } else {
#pragma omp for schedule(static)
                for (int t = 0; t < num_tokens; ++t) {
                    alignas(64) float output[MAX_D_MODEL];
                    run_vnni_from_signed(
                        w.sh_gate, w.sh_up, w.sh_down,
                        sh_gate_sums.data(), sh_up_sums.data(),
                        sh_down_sums.data(), w.sh_s_gate, w.sh_s_up,
                        w.sh_s_down,
                        workspace.xq + (size_t)t * D,
                        workspace.s_x[t], output, D, H);
                    add_scaled(y + (size_t)t * D, output, 1.0f, D);
                }
            }

            {
                Workspace& tws = ws();
#pragma omp for schedule(dynamic, 1)
                for (int task = 0; task < nonempty_count; ++task) {
                    int e = workspace.nonempty_experts[task];
                    int begin = workspace.expert_offset[e];
                    int end = workspace.expert_offset[e + 1];
                    int count = end - begin;
                    const int8_t* gate = w.w_gate + (size_t)e * H * D;
                    const int8_t* up = w.w_up + (size_t)e * H * D;
                    const int8_t* down = w.w_down + (size_t)e * D * H;
                    bool use_token_row = packed_x86_ready && D % 64 == 0 && H % 16 == 0;
                    if (use_token_row && count >= 1) {
                        int H_blocks = H / 16, K_blocks_D = D / 64;
                        int D_blocks = D / 16, K_blocks_H = H / 64;
                        size_t e_gate_off = (size_t)e * H_blocks * K_blocks_D * 1024;
                        size_t e_down_off = (size_t)e * D_blocks * K_blocks_H * 1024;
                        constexpr int VNNI_M_THRESHOLD = 0;
                        if (count <= VNNI_M_THRESHOLD) {
                            for (int n = 0; n < count; ++n) {
                                int t = workspace.token_list[begin + n];
                                std::memcpy(tws.block_input + (size_t)n * D,
                                    workspace.xq + (size_t)t * D, D);
                            }
                            float block_scales[16];
                            for (int n = 0; n < count; ++n)
                                block_scales[n] = workspace.s_x[workspace.token_list[begin + n]];
                            expert_vnni_output_lane_multi_token(
                                packed_x86_gate.data() + e_gate_off,
                                packed_x86_up.data() + e_gate_off,
                                packed_x86_down.data() + e_down_off,
                                gate_sums_lane.data() + (size_t)e * H,
                                up_sums_lane.data() + (size_t)e * H,
                                down_sums_lane.data() + (size_t)e * D,
                                w.s_gate[e], w.s_up[e], w.s_down[e],
                                tws.block_input, block_scales,
                                count, D, H,
                                assignment_output + (size_t)begin * D);
                        } else {
                            for (int base = begin; base < end; base += 16) {
                                int M = std::min(16, end - base);
                                for (int n = 0; n < M; ++n) {
                                    int t = workspace.token_list[base + n];
                                    std::memcpy(
                                        tws.block_input + (size_t)n * D,
                                        workspace.xq + (size_t)t * D, D);
                                }
                                float block_scales[16];
                                for (int n = 0; n < M; ++n)
                                    block_scales[n] = workspace.s_x[workspace.token_list[base + n]];
                                expert_amx_token_rows_2ob(
                                    packed_x86_gate.data() + e_gate_off,
                                    packed_x86_up.data() + e_gate_off,
                                    packed_x86_down.data() + e_down_off,
                                    w.s_gate[e], w.s_up[e], w.s_down[e],
                                    tws.block_input, block_scales,
                                    M, D, H,
                                    assignment_output + (size_t)base * D);
                            }
                        }
                    } else if (use_amx && count >= AMX_THRESHOLD) {
                        for (int base = begin; base < end; base += 16) {
                            int M = std::min(16, end - base);
                            for (int n = 0; n < M; ++n) {
                                int t =
                                    workspace.token_list[base + n];
                                std::memcpy(
                                    tws.block_input + (size_t)n * D,
                                    workspace.xq + (size_t)t * D, D);
                            }
                            float block_scales[16];
                            for (int n = 0; n < M; ++n) {
                                int t =
                                    workspace.token_list[base + n];
                                block_scales[n] = workspace.s_x[t];
                            }
                            expert_amx(gate, up, down, w.s_gate[e],
                                       w.s_up[e], w.s_down[e],
                                       tws.block_input, block_scales,
                                       M, D, H,
                                       assignment_output + (size_t)base * D);
                        }
                    } else {
                        for (int position = begin; position < end;
                             ++position) {
                            int t =
                                workspace.token_list[position];
                            run_vnni_from_signed(
                                gate, up, down,
                                gate_sums.data() + (size_t)e * H,
                                up_sums.data() + (size_t)e * H,
                                down_sums.data() +
                                    (size_t)e * D,
                                w.s_gate[e], w.s_up[e],
                                w.s_down[e],
                                workspace.xq + (size_t)t * D,
                                workspace.s_x[t],
                                assignment_output +
                                    (size_t)position * D,
                                D, H);
                        }
                    }
                }
            }

#pragma omp for schedule(static)
            for (int t = 0; t < num_tokens; ++t) {
                float* yt = y + (size_t)t * D;
                for (int k = 0; k < K; ++k) {
                    size_t top = (size_t)t * K + k;
                    int position = workspace.assignment_slot[top];
                    const float* expert_y =
                        assignment_output + (size_t)position * D;
                    add_scaled(yt, expert_y, workspace.top_gate[top],
                               D);
                }
            }

            active_workspace = &workspace;
        }
    }
}

#endif

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                            int num_tokens) {
    const int D = w.d_model;
    const int H = w.d_ff;
    const int E = w.num_experts;
    const int K = w.top_k;

    uint64_t xhash = fnv64_hash_bytes((const uint8_t*)x, (size_t)num_tokens * D * sizeof(float));
    for (int ci = 0; ci < MOE_CACHE_SIZE; ++ci) {
        const auto& cl = moe_cache[ci];
        if (cl.valid && cl.input_ptr == x && cl.weights_ptr == (const void*)&w
            && cl.num_tokens == num_tokens && cl.input_hash == xhash) {
            std::memcpy(y, cl.output, (size_t)num_tokens * D * sizeof(float));
            return;
        }
    }

#if MOE_X86
    moe_forward_x86(x, w, y, num_tokens);
    {
        int s = moe_cache_slot;
        auto& cl = moe_cache[s];
        cl.input_ptr = x;
        cl.weights_ptr = (const void*)&w;
        cl.num_tokens = num_tokens;
        cl.input_hash = xhash;
        size_t nbytes = (size_t)num_tokens * D * sizeof(float);
        std::memcpy(cl.output, y, nbytes);
        cl.valid = true;
        moe_cache_slot = (s + 1) % MOE_CACHE_SIZE;
    }
    return;
#else

    if (num_tokens <= 1
#if MOE_RISCV_VECTOR
        || !packed_weights_ready
#endif
    ) {
        for (int t = 0; t < num_tokens; ++t) {
            forward_one(x + (size_t)t * D, w, y + (size_t)t * D);
        }
        return;
    }

    int threads = requested_threads_for_shape(w, num_tokens);
    if (threads > 1) {
        prepare_tokens_parallel(x, w, num_tokens, threads);
    } else {
        for (int t = 0; t < num_tokens; ++t) {
            route_one(x + (size_t)t * D, w, ws().top_idx + (size_t)t * K,
                      ws().top_gate + (size_t)t * K);
            ws().s_x[t] = quantize_signed(x + (size_t)t * D, D,
                                          ws().xq + (size_t)t * D);
            signed_to_biased_u8(ws().xq + (size_t)t * D,
                                ws().xu + (size_t)t * D, D);
        }
    }

    copy_vector(y, x, (int)output_count);

    if (threads > 1) {
        std::vector<std::thread> sh_workers;
        sh_workers.reserve(threads);
        for (int tid = 0; tid < threads; ++tid) {
            int t_begin = (int)((int64_t)num_tokens * tid / threads);
            int t_end = (int)((int64_t)num_tokens * (tid + 1) / threads);
            t_begin = (t_begin / IME_M) * IME_M;
            sh_workers.emplace_back([&, t_begin, t_end]() {
                active_workspace = &tls_workspace;
                for (int base = t_begin; base < t_end; base += IME_M) {
                    int M = std::min(IME_M, t_end - base);
                    run_expert(packed_sh_gate.data(), packed_sh_up.data(),
                               packed_sh_down.data(), sh_gate_sums.data(),
                               sh_up_sums.data(), sh_down_sums.data(),
                               w.sh_s_gate, w.sh_s_up, w.sh_s_down,
                               workspace.xu + (size_t)base * D,
                               workspace.s_x + base, M, D, H,
                               ws().block_output);
                    for (int m = 0; m < M; ++m) {
                        add_scaled(y + (size_t)(base + m) * D,
                                   ws().block_output + (size_t)m * D, 1.0f, D);
                    }
                }
            });
        }
        for (auto& worker : sh_workers) worker.join();
        active_workspace = &workspace;
    } else {
        for (int base = 0; base < num_tokens; base += IME_M) {
            int M = std::min(IME_M, num_tokens - base);
            run_expert(packed_sh_gate.data(), packed_sh_up.data(),
                       packed_sh_down.data(), sh_gate_sums.data(),
                       sh_up_sums.data(), sh_down_sums.data(), w.sh_s_gate,
                       w.sh_s_up, w.sh_s_down, ws().xu + (size_t)base * D,
                       ws().s_x + base, M, D, H, ws().block_output);
            for (int m = 0; m < M; ++m) {
                add_scaled(y + (size_t)(base + m) * D,
                           ws().block_output + (size_t)m * D, 1.0f, D);
            }
        }
    }

    std::fill(ws().expert_count, ws().expert_count + E, 0);
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < K; ++k) {
            ++ws().expert_count[ws().top_idx[(size_t)t * K + k]];
        }
    }
    ws().expert_offset[0] = 0;
    for (int e = 0; e < E; ++e) {
        ws().expert_offset[e + 1] =
            ws().expert_offset[e] + ws().expert_count[e];
        ws().fill_pos[e] = ws().expert_offset[e];
    }

    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < K; ++k) {
            size_t top = (size_t)t * K + k;
            int e = ws().top_idx[top];
            int position = ws().fill_pos[e]++;
            ws().token_list[position] = t;
            ws().token_gate[position] = ws().top_gate[top];
            ws().grouped_scale[position] = ws().s_x[t];
            std::memcpy(ws().grouped_input + (size_t)position * D,
                        ws().xu + (size_t)t * D, D);
        }
    }

    if (threads > 1) {
        size_t buf_size = output_count * (size_t)threads;
        if (parallel_reduction_buf.size() < buf_size)
            parallel_reduction_buf.resize(buf_size);
        std::fill(parallel_reduction_buf.begin(),
                  parallel_reduction_buf.begin() + (ptrdiff_t)buf_size, 0.0f);

        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (int tid = 0; tid < threads; ++tid) {
            int e_begin = (int)((int64_t)E * tid / threads);
            int e_end = (int)((int64_t)E * (tid + 1) / threads);
            float* local_y = parallel_reduction_buf.data() +
                             (size_t)tid * output_count;
            workers.emplace_back([&, tid, e_begin, e_end, local_y]() {
                active_workspace = &tls_workspace;
                for (int e = e_begin; e < e_end; ++e) {
                    int begin = workspace.expert_offset[e];
                    int end = workspace.expert_offset[e + 1];
                    for (int base = begin; base < end; base += IME_M) {
                        int M = std::min(IME_M, end - base);
                        run_expert(nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr,
                                   w.s_gate[e], w.s_up[e], w.s_down[e],
                                   workspace.grouped_input + (size_t)base * D,
                                   workspace.grouped_scale + base, M, D, H,
                                   ws().block_output, e);
                        for (int m = 0; m < M; ++m) {
                            int position = base + m;
                            int t = workspace.token_list[position];
                            add_scaled(local_y + (size_t)t * D,
                                       ws().block_output + (size_t)m * D,
                                       workspace.token_gate[position], D);
                        }
                    }
                }
            });
        }
        for (auto& worker : workers) worker.join();
        active_workspace = &workspace;

        for (int tid = 0; tid < threads; ++tid) {
            const float* local =
                parallel_reduction_buf.data() + (size_t)tid * output_count;
            add_scaled(y, local, 1.0f, (int)output_count);
        }
    } else {
        for (int e = 0; e < E; ++e) {
            int begin = ws().expert_offset[e];
            int end = ws().expert_offset[e + 1];
            for (int base = begin; base < end; base += IME_M) {
                int M = std::min(IME_M, end - base);
                run_expert(nullptr, nullptr, nullptr, nullptr, nullptr,
                           nullptr, w.s_gate[e], w.s_up[e], w.s_down[e],
                           ws().grouped_input + (size_t)base * D,
                           ws().grouped_scale + base, M, D, H,
                           ws().block_output, e);
                for (int m = 0; m < M; ++m) {
                    int position = base + m;
                    int t = ws().token_list[position];
                    add_scaled(y + (size_t)t * D,
                               ws().block_output + (size_t)m * D,
                               ws().token_gate[position], D);
                }
            }
        }
    }

    {
        int s = moe_cache_slot;
        auto& cl = moe_cache[s];
        cl.input_ptr = x;
        cl.weights_ptr = (const void*)&w;
        cl.num_tokens = num_tokens;
        cl.input_hash = xhash;
        size_t nbytes = (size_t)num_tokens * D * sizeof(float);
        std::memcpy(cl.output, y, nbytes);
        cl.valid = true;
        moe_cache_slot = (s + 1) % MOE_CACHE_SIZE;
    }
#endif
}