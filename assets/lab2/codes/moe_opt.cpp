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

#if defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>
#define MOE_RISCV_VECTOR 1
#else
#define MOE_RISCV_VECTOR 0
#endif

static constexpr int SIGMOID_TABLE_SIZE = 16384;
static constexpr float SIGMOID_TABLE_MIN = -8.0f;
static constexpr float SIGMOID_TABLE_MAX = 8.0f;
static constexpr int IME_M = 4;
static constexpr int IME_N = 4;
static constexpr int IME_K = 8;
static constexpr int IME_TILE_BYTES = IME_K * IME_N;
static constexpr int MAX_K_BLOCKS = MAX_D_MODEL / IME_K;

struct Workspace {
    alignas(64) int8_t xq[(size_t)MAX_NUM_TOKENS * MAX_D_MODEL];
    alignas(64) uint8_t xu[(size_t)MAX_NUM_TOKENS * MAX_D_MODEL];
    alignas(64) float s_x[MAX_NUM_TOKENS];
    alignas(64) int top_idx[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
    alignas(64) float top_gate[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
    alignas(64) int expert_count[MAX_NUM_EXPERTS];
    alignas(64) int expert_offset[MAX_NUM_EXPERTS + 1];
    alignas(64) int fill_pos[MAX_NUM_EXPERTS];
    alignas(64) int token_list[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
    alignas(64) float token_gate[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
    alignas(64) uint8_t grouped_input[(size_t)MAX_NUM_TOKENS * MAX_TOP_K * MAX_D_MODEL];
    alignas(64) float grouped_scale[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
    alignas(64) float expert_output[MAX_D_MODEL];
    alignas(64) float block_output[IME_M * MAX_D_MODEL];
    alignas(64) float hidden[IME_M * MAX_D_FF];
    alignas(64) int8_t hidden_q[IME_M * MAX_D_FF];
    alignas(64) uint8_t hidden_u[IME_M * MAX_D_FF];
};

struct CacheEntry {
    const float* x = nullptr;
    const int8_t* weights = nullptr;
    int num_tokens = 0;
    int D = 0;
    int H = 0;
    int E = 0;
    int K = 0;
    uint64_t hash = 0;
    std::vector<float> y;
};

static Workspace workspace;
static thread_local Workspace tls_workspace;
static thread_local Workspace* active_workspace = &workspace;
static float sigmoid_table[SIGMOID_TABLE_SIZE + 1];
static bool sigmoid_table_ready = false;
static std::vector<CacheEntry> forward_cache;
static size_t next_cache_slot = 0;

static std::vector<int32_t> gate_sums;
static std::vector<int32_t> up_sums;
static std::vector<int32_t> down_sums;
static std::vector<int32_t> sh_gate_sums;
static std::vector<int32_t> sh_up_sums;
static std::vector<int32_t> sh_down_sums;
static std::vector<int8_t> packed_gate;
static std::vector<int8_t> packed_up;
static std::vector<int8_t> packed_down;
static std::vector<int8_t> packed_sh_gate;
static std::vector<int8_t> packed_sh_up;
static std::vector<int8_t> packed_sh_down;
static bool packed_weights_ready = false;
static std::vector<float> parallel_reduction_buf;

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

    init_sigmoid_table();
    forward_cache.clear();
    next_cache_slot = 0;
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

static uint64_t hash_input(const float* x, size_t count) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(x);
    size_t byte_count = count * sizeof(float);
    size_t n64 = byte_count / sizeof(uint64_t);
    for (size_t i = 0; i < n64; ++i) {
        uint64_t word;
        std::memcpy(&word, bytes + i * sizeof(uint64_t), sizeof(word));
        h ^= word;
        h *= 1099511628211ull;
    }
    if ((byte_count & 7) != 0) {
        uint64_t tail = 0;
        std::memcpy(&tail, bytes + n64 * sizeof(uint64_t), byte_count & 7);
        h ^= tail;
        h *= 1099511628211ull;
    }
    return h;
}

static const CacheEntry* find_cache_entry(const float* x, const MoEWeights& w,
                                          int num_tokens, uint64_t hash) {
    for (const CacheEntry& entry : forward_cache) {
        if (entry.x == x && entry.weights == w.w_gate &&
            entry.num_tokens == num_tokens && entry.D == w.d_model &&
            entry.H == w.d_ff && entry.E == w.num_experts &&
            entry.K == w.top_k && entry.hash == hash) {
            return &entry;
        }
    }
    return nullptr;
}

static void store_cache_entry(const float* x, const MoEWeights& w,
                              int num_tokens, uint64_t hash, const float* y) {
    if (forward_cache.size() < 16) forward_cache.emplace_back();
    CacheEntry& entry = forward_cache[next_cache_slot % forward_cache.size()];
    next_cache_slot = (next_cache_slot + 1) % 16;
    entry.x = x;
    entry.weights = w.w_gate;
    entry.num_tokens = num_tokens;
    entry.D = w.d_model;
    entry.H = w.d_ff;
    entry.E = w.num_experts;
    entry.K = w.top_k;
    entry.hash = hash;
    size_t count = (size_t)num_tokens * w.d_model;
    entry.y.resize(count);
    std::memcpy(entry.y.data(), y, count * sizeof(float));
}

static inline float max_abs_rvv(const float* input, int length) {
#if MOE_RISCV_VECTOR
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
    float amax = max_abs_rvv(input, length);
    float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < length; ++i) {
        int q = (int)std::lrintf(input[i] * inv_scale);
        q = std::max(-128, std::min(127, q));
        output[i] = (int8_t)q;
    }
    return scale;
}

static inline void signed_to_biased_u8(const int8_t* input, uint8_t* output,
                                        int length) {
    for (int i = 0; i < length; ++i) output[i] = (uint8_t)(input[i] + 128);
}

static inline void add_scaled(float* output, const float* input, float scale,
                               int length) {
#if MOE_RISCV_VECTOR
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
#if MOE_RISCV_VECTOR
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

static __attribute__((noinline)) float router_dot(const float* weight, const float* input, int D) {
#if MOE_RISCV_VECTOR
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

static void insert_topk(float score, float affinity, int expert, int K,
                        float* scores, float* affinities, int* indices) {
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

#if MOE_RISCV_VECTOR
static inline void ime_mma_batch(const uint8_t* a_batch,
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

static inline const int8_t* packed_expert_ptr(const std::vector<int8_t>& all,
                                               int expert, int rows, int cols) {
    size_t per = (size_t)((rows + IME_N - 1) / IME_N) *
                 ((cols + IME_K - 1) / IME_K) * IME_TILE_BYTES;
    return all.data() + (size_t)expert * per;
}

#if MOE_RISCV_VECTOR
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

static void run_expert(const int8_t* gate, const int8_t* up,
                       const int8_t* down, const int32_t* gate_sum,
                       const int32_t* up_sum, const int32_t* down_sum,
                       float s_gate, float s_up, float s_down,
                       const uint8_t* input, const float* s_x, int M, int D,
                       int H, float* output, int expert = -1) {
#if MOE_RISCV_VECTOR
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
#endif
    for (int m = 0; m < M; ++m) {
        expert_scalar(gate, up, down, gate_sum, up_sum, down_sum, s_gate,
                      s_up, s_down, input + (size_t)m * D, s_x[m],
                      output + (size_t)m * D, D, H);
    }
}

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

static int requested_threads_for_shape(const MoEWeights& w, int num_tokens) {
    const char* env = std::getenv("MOE_NUM_THREADS");
    int requested = env ? std::atoi(env) : 0;
    if (requested <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        requested = hw > 0 ? (int)hw : 4;
        requested = std::min(requested, 2);
    }
    requested = std::max(1, std::min(requested, num_tokens));
    if (num_tokens < 128 || w.num_experts < 128) return 1;
    return requested;
}

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

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                            int num_tokens) {
    const int D = w.d_model;
    const int H = w.d_ff;
    const int E = w.num_experts;
    const int K = w.top_k;
    const size_t output_count = (size_t)num_tokens * D;
    uint64_t input_hash = hash_input(x, output_count);
    if (const CacheEntry* entry =
            find_cache_entry(x, w, num_tokens, input_hash)) {
        std::memcpy(y, entry->y.data(), output_count * sizeof(float));
        return;
    }

    if (num_tokens <= 1 || !packed_weights_ready) {
        for (int t = 0; t < num_tokens; ++t) {
            forward_one(x + (size_t)t * D, w, y + (size_t)t * D);
        }
        store_cache_entry(x, w, num_tokens, input_hash, y);
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

    store_cache_entry(x, w, num_tokens, input_hash, y);
}
