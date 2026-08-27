#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <immintrin.h>

#if defined(__GNUC__) && !defined(__clang__)
#define NO_VECTORIZE __attribute__((noinline, optimize("no-tree-vectorize")))
#else
#define NO_VECTORIZE __attribute__((noinline))
#endif

static double now_ms() {
    using namespace std::chrono;
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

NO_VECTORIZE
void attention_scalar_single_query(const float* q, const float* K, const float* V,
                                   float* out, int T, int D) {
    std::vector<float> scores(T);
    // 1
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    // q:[D]
    // k:[T, D]
    for (int t = 0; t < T; ++t) {
        const float* k = K + static_cast<size_t>(t) * D;
        float acc = 0.0f;
        for (int j = 0; j < D; ++j) {
            float tmp = q[j] * k[j];  // vectorization 
            acc += tmp;  // vectorization
            // 规约的开销很大
        }
        scores[t] = acc * scale;
    }


    //2
    float max_score = -std::numeric_limits<float>::infinity();
    for (int t = 0; t < T; ++t) max_score = std::max(max_score, scores[t]); // vectorization

    float sum_exp = 0.0f;
    for (int t = 0; t < T; ++t) {
        float tmp = std::exp(scores[t] - max_score);  // vectorization
        sum_exp = sum_exp + tmp;  // vectorization
    }
    const float inv_sum = 1.0f / sum_exp;
    for (int t = 0; t < T; ++t) scores[t] *= inv_sum; // vectorization

    // 3
    for (int j = 0; j < D; ++j) out[j] = 0.0f;
    for (int t = 0; t < T; ++t) {
        const float* v = V + static_cast<size_t>(t) * D;
        const float p = scores[t];
        for (int j = 0; j < D; ++j) {
            float tmp = p * v[j]; // vectorization
            out[j] += tmp; // vectorization
        }
    }
}

static inline __mmask16 tail_mask16(int n) {
    return n >= 16 ? 0xFFFFu : static_cast<__mmask16>((1u << n) - 1u);
}

void attention_avx512_single_query(const float* q, const float* K, const float* V,
                                   float* out, int T, int D) {
    std::vector<float> scores(T);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const int D16 = D & ~15;
    const int dtail = D - D16;
    const __mmask16 dmask = tail_mask16(dtail);

    // 1) score[t] = dot(q, K[t]) / sqrt(D)
    // AVX 512
    // float
    // 512 / 32 = 16 
    // q[0..15] * K[t][0..15] + q[16..31] * K[t][16..31] + ...
    // i = j
    for (int t = 0; t < T; ++t) {
        const float* k = K + static_cast<size_t>(t) * D;
        __m512 acc_tmp = _mm512_setzero_ps();
        for (int j = 0; j < D16; j += 16) {
            const __m512 qv = _mm512_loadu_ps(q + j); // q[0..15]
            const __m512 kv = _mm512_loadu_ps(k + j);
            acc_tmp = _mm512_fmadd_ps(qv, kv, acc_tmp); // acc += qv * kv
            // __m512 mul = _mm512_mul_ps(qv, kv);
            // acc = _mm512_add_ps(acc, mul);
        }
        if (dtail) {
            const __m512 qv = _mm512_maskz_loadu_ps(dmask, q + D16);
            const __m512 kv = _mm512_maskz_loadu_ps(dmask, k + D16);
            acc_tmp = _mm512_fmadd_ps(qv, kv, acc_tmp);
        }
        scores[t] = _mm512_reduce_add_ps(acc_tmp) * scale; // _mm512_reduce_add_ps(acc_tmp) = acc
    }

    // q:[D]
    // k:[T, D] -> kT: [D, T]
    // 对 T 维进行向量化
    // another way to compute scores using a TRANSPOSED K matrix

    std::vector<float> KT(static_cast<size_t>(D) * T);


    for (int tok = 0; tok < T; ++tok) {
        for (int jj = 0; jj < D; ++jj) {
            KT[static_cast<size_t>(jj) * T + tok] =
                K[static_cast<size_t>(tok) * D + jj];
        }
    }

    // kT: [D, T]
    // q: [D]

    const int T16 = T & ~15;

    int tok = 0;
    for (; tok < T16; tok += 16) {
        __m512 acc = _mm512_setzero_ps(); // store acc: store[0..15],[16..31],...

        for (int jj = 0; jj < D; ++jj) {
            const __m512 qv = _mm512_set1_ps(q[jj]); // q[16] = [q[jj], q[jj], ..., q[jj]]
            const __m512 kv = _mm512_loadu_ps(
                KT.data() + static_cast<size_t>(jj) * T + tok
            );
            acc = _mm512_fmadd_ps(qv, kv, acc); 
        }

        acc = _mm512_mul_ps(acc, _mm512_set1_ps(scale));
        _mm512_storeu_ps(scores.data() + tok, acc);
    }

    if (tok < T) {
        const __mmask16 m = tail_mask16(T - tok);

        __m512 acc = _mm512_setzero_ps();

        for (int jj = 0; jj < D; ++jj) {
            const __m512 qv = _mm512_set1_ps(q[jj]);
            const __m512 kv = _mm512_maskz_loadu_ps(
                m,
                KT.data() + static_cast<size_t>(jj) * T + tok
            );
            acc = _mm512_fmadd_ps(qv, kv, acc);
        }

        acc = _mm512_mul_ps(acc, _mm512_set1_ps(scale));
        _mm512_mask_storeu_ps(scores.data() + tok, m, acc);
    }

    // 2) softmax. max/sum can be vectorized; exp is deliberately scalar here
    // because standard AVX-512 intrinsics do not include a portable _mm512_exp_ps.
    const __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity()); // [inf, ..., inf]
    __m512 vmax = neg_inf;
    int t = 0;
    // [0, 16), [16, 32), ..., [T16, T) --> max
    // v1 = [0..15]
    // v2 = [16..31]
    // v3 = [32..47]
    // max_tmp = max(v1, v2)
    // max_tmp = max(max_tmp, v3)
    for (; t + 16 <= T; t += 16) {
        const __m512 s = _mm512_loadu_ps(scores.data() + t);
        vmax = _mm512_max_ps(vmax, s);
    }
    if (t < T) {
        const __mmask16 m = tail_mask16(T - t);
        const __m512 s = _mm512_mask_loadu_ps(neg_inf, m, scores.data() + t);
        vmax = _mm512_max_ps(vmax, s);
    }
    const float max_score = _mm512_reduce_max_ps(vmax); // max_tmp

    float sum_exp = 0.0f;
    for (int i = 0; i < T; ++i) {
        scores[i] = std::exp(scores[i] - max_score);
        sum_exp += scores[i];
    }
    const float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < T; ++i) scores[i] *= inv_sum;

    // 2) softmax with AVX-512 approximate exp.
    // 在 x=0 处泰勒展开

    // scores[i] = exp(scores[i] - max_score) / sum(exp(scores[i] - max_score))

    auto exp512_approx_ps = [](__m512 x) -> __m512 {
        // Clamp range to avoid overflow / underflow when constructing 2^n.
        const __m512 max_x = _mm512_set1_ps(88.3762626647949f);
        const __m512 min_x = _mm512_set1_ps(-87.3365447505531f);

        x = _mm512_min_ps(x, max_x);
        x = _mm512_max_ps(x, min_x);

        // exp(x) = 2^n * exp(r)
        // n = round(x / ln2)
        // r = x - n * ln2
        const __m512 log2e  = _mm512_set1_ps(1.44269504088896341f);
        const __m512 ln2_hi = _mm512_set1_ps(0.693359375f);
        const __m512 ln2_lo = _mm512_set1_ps(-2.12194440e-4f);

        const __m512 y = _mm512_mul_ps(x, log2e);

        const __m512i n = _mm512_cvt_roundps_epi32(
            y,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
        );

        const __m512 nf = _mm512_cvtepi32_ps(n);

        __m512 r = _mm512_fnmadd_ps(nf, ln2_hi, x);
        r = _mm512_fnmadd_ps(nf, ln2_lo, r);

        // exp(r) polynomial approximation.
        // 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720
        __m512 p = _mm512_set1_ps(1.0f / 720.0f);
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 120.0f));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24.0f));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6.0f));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 2.0f));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));

        // Construct 2^n using float exponent bits.
        // float: exponent bias = 127, exponent field starts at bit 23.
        const __m512i pow2_bits = _mm512_slli_epi32(
            _mm512_add_epi32(n, _mm512_set1_epi32(127)),
            23
        );

        const __m512 pow2n = _mm512_castsi512_ps(pow2_bits);

        return _mm512_mul_ps(p, pow2n);
    };

    // 2.1) max_score = max(scores)
    const __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());

    __m512 vmax = neg_inf;

    int t = 0;
    for (; t + 16 <= T; t += 16) {
        const __m512 s = _mm512_loadu_ps(scores.data() + t);
        vmax = _mm512_max_ps(vmax, s);
    }

    if (t < T) {
        const __mmask16 m = tail_mask16(T - t);
        const __m512 s = _mm512_mask_loadu_ps(neg_inf, m, scores.data() + t);
        vmax = _mm512_max_ps(vmax, s);
    }

    const float max_score = _mm512_reduce_max_ps(vmax);
    const __m512 v_max_score = _mm512_set1_ps(max_score);

    // 2.2) scores[i] = approx_exp(scores[i] - max_score)
    // Meanwhile accumulate sum_exp.
    __m512 vsum = _mm512_setzero_ps();

    t = 0;
    for (; t + 16 <= T; t += 16) {
        const __m512 s = _mm512_loadu_ps(scores.data() + t);
        const __m512 x = _mm512_sub_ps(s, v_max_score);
        const __m512 e = exp512_approx_ps(x);

        _mm512_storeu_ps(scores.data() + t, e);
        vsum = _mm512_add_ps(vsum, e);
    }

    if (t < T) {
        const __mmask16 m = tail_mask16(T - t);

        const __m512 s = _mm512_maskz_loadu_ps(m, scores.data() + t);
        const __m512 x = _mm512_sub_ps(s, v_max_score);
        __m512 e = exp512_approx_ps(x);

        // Invalid lanes must not contribute to sum_exp.
        e = _mm512_maskz_mov_ps(m, e);

        _mm512_mask_storeu_ps(scores.data() + t, m, e);
        vsum = _mm512_add_ps(vsum, e);
    }

    const float sum_exp = _mm512_reduce_add_ps(vsum);
    const __m512 inv_sum = _mm512_set1_ps(1.0f / sum_exp);

    // 2.3) scores[i] /= sum_exp
    t = 0;
    for (; t + 16 <= T; t += 16) {
        const __m512 e = _mm512_loadu_ps(scores.data() + t);
        const __m512 p = _mm512_mul_ps(e, inv_sum);
        _mm512_storeu_ps(scores.data() + t, p);
    }

    if (t < T) {
        const __mmask16 m = tail_mask16(T - t);
        const __m512 e = _mm512_maskz_loadu_ps(m, scores.data() + t);
        const __m512 p = _mm512_mul_ps(e, inv_sum);
        _mm512_mask_storeu_ps(scores.data() + t, m, p);
    }

    // 3) out = softmax(score) * V
    int j = 0;
    for (; j < D16; j += 16) _mm512_storeu_ps(out + j, _mm512_setzero_ps());
    if (dtail) _mm512_mask_storeu_ps(out + D16, dmask, _mm512_setzero_ps());

    for (int tok = 0; tok < T; ++tok) {
        const float* v = V + static_cast<size_t>(tok) * D;
        const __m512 p = _mm512_set1_ps(scores[tok]);
        for (j = 0; j < D16; j += 16) {
            __m512 ov = _mm512_loadu_ps(out + j);
            const __m512 vv = _mm512_loadu_ps(v + j);
            ov = _mm512_fmadd_ps(p, vv, ov);
            _mm512_storeu_ps(out + j, ov);
        }
        if (dtail) {
            __m512 ov = _mm512_maskz_loadu_ps(dmask, out + D16);
            const __m512 vv = _mm512_maskz_loadu_ps(dmask, v + D16);
            ov = _mm512_fmadd_ps(p, vv, ov);
            _mm512_mask_storeu_ps(out + D16, dmask, ov);
        }
    }
}

int main(int argc, char** argv) {
    const int T = (argc > 1) ? std::atoi(argv[1]) : 1025;
    const int D = (argc > 2) ? std::atoi(argv[2]) : 80;
    const int repeat = (argc > 3) ? std::atoi(argv[3]) : 50;

    std::vector<float> q(D), K(static_cast<size_t>(T) * D), V(static_cast<size_t>(T) * D);
    std::vector<float> out_s(D), out_v(D);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : q) x = dist(rng);
    for (float& x : K) x = dist(rng);
    for (float& x : V) x = dist(rng);

    attention_scalar_single_query(q.data(), K.data(), V.data(), out_s.data(), T, D);
    attention_avx512_single_query(q.data(), K.data(), V.data(), out_v.data(), T, D);

    float max_err = 0.0f;
    for (int i = 0; i < D; ++i) max_err = std::max(max_err, std::abs(out_s[i] - out_v[i]));

    double t0 = now_ms();
    for (int r = 0; r < repeat; ++r)
        attention_scalar_single_query(q.data(), K.data(), V.data(), out_s.data(), T, D);
    double t1 = now_ms();
    for (int r = 0; r < repeat; ++r)
        attention_avx512_single_query(q.data(), K.data(), V.data(), out_v.data(), T, D);
    double t2 = now_ms();

    std::cout << "single-query attention: T=" << T << ", D=" << D << ", repeat=" << repeat << "\n";
    std::cout << "scalar avg: " << (t1 - t0) / repeat << " ms\n";
    std::cout << "avx512 avg: " << (t2 - t1) / repeat << " ms\n";
    std::cout << "max_err: " << max_err << "\n";
    std::cout << ((max_err < 1e-4f) ? "PASS\n" : "FAIL\n");
    return (max_err < 1e-4f) ? 0 : 1;
}
