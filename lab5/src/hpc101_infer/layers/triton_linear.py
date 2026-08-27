"""融合反量化-GEMM Triton kernel（V10: coalesced dot with transposed [K_packed,N] weight）。

V10 优化（2026-08-21）：发现原 dot 路径用 [N,K_packed] 存储导致 tile [BK,BN]
加载非合并（L1 98% 饱和，V4 慢 2.8×）。将 qweight 转置为 [K_packed, N] 后，
tile [BK,BN] = BK 行 × BN 连续 N → 合并加载。配合 tl.dot（MMA, 25% occupancy）
替代 tl.sum GEMV（12.5% occupancy），decode M≤2 提速 ~3×。

三条路径（按 qweight 存储自动选择）：
- 原始 [N,K_packed]（未转置，fallback）：M≤8 GEMV（tl.sum），M>8 dot（[N,K] 存储）。
- 转置 [K_packed,N]（V10）：M≤8 MMA（tl.dot, BN=32 最优配置），M>8 dot（coalesced）。
  两条 V10 路径均用合并加载，prefill 也受益。

反量化数学不变（symmetric int4、fp32 累加），bit-close 于 dequant+F.linear。
"""

from __future__ import annotations

import os
import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from hpc101_infer.layers.linear import QuantizedLinear, QuantizedLinearFactory


# ---------------------------------------------------------------------------
# GEMV 路径（fallback, [N,K_packed] 存储）：小 M decode。tl.sum 归约。
# ---------------------------------------------------------------------------

@triton.jit
def _fused_dequant_gemv_kernel(
    x_ptr, qweight_ptr, scales_ptr, bias_ptr, y_ptr,
    M, N, K, K_padded,
    stride_xm, stride_xk, stride_qn, stride_qk, stride_sn, stride_sk,
    stride_ym, stride_yn,
    GROUP_SIZE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_BIAS: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    for k_start in range(0, K_padded, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_padded
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        k_packed = k_offs // 2
        q_ptrs = (qweight_ptr + offs_n[:, None] * stride_qn + k_packed[None, :] * stride_qk)
        q_mask = (offs_n[:, None] < N) & (k_packed[None, :] < K_padded // 2)
        packed = tl.load(q_ptrs, mask=q_mask, other=0)
        shift = (k_offs % 2) * 4
        unpacked = ((packed >> shift[None, :]) & 0x0F).to(tl.float32) - 8.0
        group_offs = k_offs // GROUP_SIZE
        s_ptrs = (scales_ptr + offs_n[:, None] * stride_sn + group_offs[None, :] * stride_sk)
        s_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K_padded)
        scales = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        w_fp32 = unpacked * scales
        acc += tl.sum(w_fp32[None, :, :] * x[:, None, :].to(tl.float32), axis=2)
    if USE_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = m_mask[:, None] & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


# ---------------------------------------------------------------------------
# MMA 路径（V10, [K_packed,N] 存储）：小 M decode。tl.dot tensor core, 合并加载。
# 权重 tile [BLOCK_K, BLOCK_N]，qweight 以 [K_packed,N] 存储（K 行、N 列连续）。
# ---------------------------------------------------------------------------

@triton.jit
def _fused_dequant_mma_kernel(
    x_ptr, qweight_ptr, scales_ptr, bias_ptr, y_ptr,
    M, N, K, K_padded,
    stride_xm, stride_xk, stride_qn, stride_qk, stride_sn, stride_sk,
    stride_ym, stride_yn,
    GROUP_SIZE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_BIAS: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Coalesced MMA: w tile [BK,BN] from [K_packed,N] storage (K rows, N cols contiguous).
    Dequant in-register, tl.dot (tensor core, 25% occupancy)."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k_start in range(0, K_padded, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_padded
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        k_packed = k_offs // 2
        # [K_packed,N] storage: stride_qk = N (k_packed dim), stride_qn = 1 (n dim)
        # tile [BK, BN]: BK k_packed rows × BN contiguous n -> coalesced
        p_ptrs = (qweight_ptr + k_packed[:, None] * stride_qk + offs_n[None, :] * stride_qn)
        p_mask = (k_packed[:, None] < K_padded // 2) & (offs_n[None, :] < N)
        packed = tl.load(p_ptrs, mask=p_mask, other=0)
        shift = (k_offs % 2) * 4
        unpacked = ((packed >> shift[:, None]) & 0x0F).to(tl.float32) - 8.0  # [BK, BN]
        group_offs = k_offs // GROUP_SIZE
        s_ptrs = (scales_ptr + group_offs[:, None] * stride_sk + offs_n[None, :] * stride_sn)
        s_mask = (k_offs[:, None] < K_padded) & (offs_n[None, :] < N)
        scales = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        w = (unpacked * scales).to(x.dtype)  # [BK, BN]
        acc += tl.dot(x, w)  # tensor core MMA, fp32 accumulate
    if USE_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = m_mask[:, None] & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


# ---------------------------------------------------------------------------
# Dot 路径：大 M（prefill）。权重 tile [BLOCK_K, BLOCK_N]，tl.dot 无转置。
# 对 [K_packed,N] 存储（V10）也合并加载；对 [N,K_packed]（fallback）原行为。
# ---------------------------------------------------------------------------

@triton.jit
def _fused_dequant_gemm_kernel_v2(
    x_ptr, qweight_ptr, scales_ptr, bias_ptr, y_ptr,
    M, N, K, K_padded,
    stride_xm, stride_xk, stride_qn, stride_qk, stride_sn, stride_sk,
    stride_ym, stride_yn,
    GROUP_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, USE_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k_start in range(0, K_padded, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        k_packed = k_offs // 2
        p_ptrs = (qweight_ptr + k_packed[:, None] * stride_qk + offs_n[None, :] * stride_qn)
        p_mask = (k_packed[:, None] < K_padded // 2) & (offs_n[None, :] < N)
        packed = tl.load(p_ptrs, mask=p_mask, other=0)
        shift = (k_offs % 2) * 4
        unpacked = ((packed >> shift[:, None]) & 0x0F).to(tl.float32) - 8.0
        group_offs = k_offs // GROUP_SIZE
        s_ptrs = (scales_ptr + group_offs[:, None] * stride_sk + offs_n[None, :] * stride_sn)
        s_mask = (k_offs[:, None] < K_padded) & (offs_n[None, :] < N)
        scales = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        w = (unpacked * scales).to(x.dtype)
        acc += tl.dot(x, w)
    if USE_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


_GEMV_MAX_M = 8


def _launch_fused_dequant_gemm(
    x, qweight, scales, bias,
    in_features, out_features, padded_in_features, group_size,
):
    original_shape = x.shape
    xr = x.reshape(-1, in_features)
    M, K = xr.shape
    N = out_features
    K_padded = padded_in_features
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)

    # 检测 qweight 存储布局：[N,K_packed]（原始）vs [K_packed,N]（V10 转置）
    transposed = (qweight.shape[0] != N)
    if transposed:
        # [K_packed, N]: stride_qk = stride(0) (k_packed dim), stride_qn = stride(1) (n dim)
        stride_qn = qweight.stride(1)
        stride_qk = qweight.stride(0)
    else:
        # [N, K_packed]: stride_qn = stride(0) (n dim), stride_qk = stride(1) (k_packed dim)
        stride_qn = qweight.stride(0)
        stride_qk = qweight.stride(1)

    if M <= _GEMV_MAX_M:
        BLOCK_M = 2 if M <= 2 else (4 if M <= 4 else 8)
        if transposed:
            # V10: coalesced MMA, BN=32 BK=128 nw=2 (sweep最优, 89.5ms/step vs 97.2 nw=4)
            BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = 32, 128, 2, 4
            grid = (triton.cdiv(N, BLOCK_N),)
            _fused_dequant_mma_kernel[grid](
                xr, qweight, scales, bias, y, M, N, K, K_padded,
                xr.stride(0), xr.stride(1), stride_qn, stride_qk,
                scales.stride(0), scales.stride(1), y.stride(0), y.stride(1),
                GROUP_SIZE=group_size, USE_BIAS=bias is not None,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=NUM_WARPS, num_stages=NUM_STAGES,
            )
        else:
            BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = 64, 256, 4, 4
            grid = (triton.cdiv(N, BLOCK_N),)
            _fused_dequant_gemv_kernel[grid](
                xr, qweight, scales, bias, y, M, N, K, K_padded,
                xr.stride(0), xr.stride(1), stride_qn, stride_qk,
                scales.stride(0), scales.stride(1), y.stride(0), y.stride(1),
                GROUP_SIZE=group_size, USE_BIAS=bias is not None,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=NUM_WARPS, num_stages=NUM_STAGES,
            )
    else:
        # Dot path (prefill M>8). Config tuned per M range from bench_prefill_sweep:
        # - M<=512: fused dot avoids materialization (1.8-3.6× faster than cuBLAS+dequant).
        #   BM=128/256 BN=128 BK=128/64 nw=8 best for gate/up/down.
        # - M>512: cuBLAS wins (handled by hybrid_linear crossover_m=512).
        BLOCK_M = 64 if M > 32 else 32
        BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = 128, 128, 8, 4
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _fused_dequant_gemm_kernel_v2[grid](
            xr, qweight, scales, bias, y, M, N, K, K_padded,
            xr.stride(0), xr.stride(1), stride_qn, stride_qk,
            scales.stride(0), scales.stride(1), y.stride(0), y.stride(1),
            GROUP_SIZE=group_size, USE_BIAS=bias is not None,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )
    return y.reshape(*original_shape[:-1], N)


class FusedQuantizedLinear(QuantizedLinear):
    def forward(self, inputs):
        return _launch_fused_dequant_gemm(
            inputs, self.qweight, self.scales, self.bias,
            self.in_features, self.out_features,
            self.padded_in_features, self.group_size)


class FusedQuantizedLinearFactory(QuantizedLinearFactory):
    def create(self, module_name, in_features, out_features, bias,
               device=None, dtype=None):
        entry = self.manifest.get(module_name)
        if entry is None:
            return self._fallback.create(module_name, in_features, out_features, bias, device, dtype)
        if entry.original_shape != (out_features, in_features):
            raise RuntimeError(f"manifest shape mismatch for {module_name}: "
                                f"manifest={entry.original_shape}, model={(out_features, in_features)}")
        scale_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else self.scale_dtype
        return FusedQuantizedLinear(in_features, out_features, entry.group_size,
                                    symmetric=entry.symmetric,
                                    padded_in_features=entry.padded_shape[1],
                                    bias=bias, device=device, scale_dtype=scale_dtype)


class FusedQuantizedLinearV2(QuantizedLinear):
    def forward(self, inputs):
        return _launch_fused_dequant_gemm(
            inputs, self.qweight, self.scales, self.bias,
            self.in_features, self.out_features,
            self.padded_in_features, self.group_size)


class FusedQuantizedLinearV2Factory(QuantizedLinearFactory):
    def create(self, module_name, in_features, out_features, bias,
               device=None, dtype=None):
        entry = self.manifest.get(module_name)
        if entry is None:
            return self._fallback.create(module_name, in_features, out_features, bias, device, dtype)
        if entry.original_shape != (out_features, in_features):
            raise RuntimeError(f"manifest shape mismatch for {module_name}: "
                                f"manifest={entry.original_shape}, model={(out_features, in_features)}")
        scale_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else self.scale_dtype
        return FusedQuantizedLinearV2(in_features, out_features, entry.group_size,
                                      symmetric=entry.symmetric,
                                      padded_in_features=entry.padded_shape[1],
                                      bias=bias, device=device, scale_dtype=scale_dtype)
