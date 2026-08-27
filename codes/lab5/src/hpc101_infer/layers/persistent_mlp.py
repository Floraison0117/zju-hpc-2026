"""Persistent fused gate+up kernel for decode (M <= 64).

Replaces: gelu(gate_proj(x), approximate="tanh") * up_proj(x)
with a single Triton kernel that:
  - loads x tile ONCE (shared by gate and up)
  - unpacks INT4 in registers (branchless)
  - dequantizes per-group (group_size=128, scale loaded once per (N,group))
  - FP32 accumulates gate and up GEMMs
  - applies tanh-approx GELU + elementwise mul in-register
  - outputs BF16 [M, F] for down_proj

Two backends:
  - "ordinary":   grid = (1, ceil(F/BLOCK_N)), one CTA per N tile
  - "persistent": grid = (NUM_CTAS,), fixed resident CTAs, tile-stride loop
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def _fused_gate_up_kernel(
    x_ptr,
    gate_qweight_ptr, gate_scales_ptr,
    up_qweight_ptr, up_scales_ptr,
    y_ptr,
    M, N, K, K_padded,
    stride_xm, stride_xk,
    stride_gqn, stride_gqk,
    stride_gsn, stride_gsk,
    stride_uqn, stride_uqk,
    stride_usn, stride_usk,
    stride_ym, stride_yn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PERSISTENT: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    if PERSISTENT:
        pid = tl.program_id(0)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
    else:
        pid = tl.program_id(1)
        num_n_tiles = 1

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M

    if PERSISTENT:
        for tile_idx in range(num_n_tiles):
            n_start = (pid + tile_idx * NUM_CTAS) * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N

            gate_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            up_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for k_start in range(0, K_padded, BLOCK_K):
                k_offs = k_start + tl.arange(0, BLOCK_K)
                k_mask = k_offs < K

                x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
                x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

                k_packed = k_offs // 2
                shift = (k_offs & 1) * 4

                g_q_ptrs = (gate_qweight_ptr + offs_n[:, None] * stride_gqn + k_packed[None, :] * stride_gqk)
                g_q_mask = n_mask[:, None] & (k_packed[None, :] < (K_padded + 1) // 2)
                g_packed = tl.load(g_q_ptrs, mask=g_q_mask, other=0)
                g_unpacked = ((g_packed >> shift[None, :]) & 0xF).to(tl.float32) - 8.0

                g_group = k_start // GROUP_SIZE
                g_s_ptrs = gate_scales_ptr + offs_n * stride_gsn + g_group * stride_gsk
                g_scales = tl.load(g_s_ptrs, mask=n_mask, other=0.0).to(tl.float32)
                g_w = (g_unpacked * g_scales[:, None]).to(x.dtype)
                gate_acc += tl.dot(x, tl.trans(g_w))

                u_q_ptrs = (up_qweight_ptr + offs_n[:, None] * stride_uqn + k_packed[None, :] * stride_uqk)
                u_packed = tl.load(u_q_ptrs, mask=g_q_mask, other=0)
                u_unpacked = ((u_packed >> shift[None, :]) & 0xF).to(tl.float32) - 8.0

                u_s_ptrs = up_scales_ptr + offs_n * stride_usn + g_group * stride_usk
                u_scales = tl.load(u_s_ptrs, mask=n_mask, other=0.0).to(tl.float32)
                u_w = (u_unpacked * u_scales[:, None]).to(x.dtype)
                up_acc += tl.dot(x, tl.trans(u_w))
            inner = 0.7978845608028654 * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc)
            tanh_inner = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
            gelu_gate = 0.5 * gate_acc * (1.0 + tanh_inner)
            result = (gelu_gate * up_acc).to(y_ptr.dtype.element_ty)

            y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
            tl.store(y_ptrs, result, mask=m_mask[:, None] & n_mask[None, :])
    else:
        n_start = pid * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        gate_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        up_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k_start in range(0, K_padded, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            k_packed = k_offs // 2
            shift = (k_offs & 1) * 4

            g_q_ptrs = (gate_qweight_ptr + offs_n[:, None] * stride_gqn + k_packed[None, :] * stride_gqk)
            g_q_mask = n_mask[:, None] & (k_packed[None, :] < (K_padded + 1) // 2)
            g_packed = tl.load(g_q_ptrs, mask=g_q_mask, other=0)
            g_unpacked = ((g_packed >> shift[None, :]) & 0xF).to(tl.float32) - 8.0

            g_group = k_start // GROUP_SIZE
            g_s_ptrs = gate_scales_ptr + offs_n * stride_gsn + g_group * stride_gsk
            g_scales = tl.load(g_s_ptrs, mask=n_mask, other=0.0).to(tl.float32)
            g_w = (g_unpacked * g_scales[:, None]).to(x.dtype)
            gate_acc += tl.dot(x, tl.trans(g_w))

            u_q_ptrs = (up_qweight_ptr + offs_n[:, None] * stride_uqn + k_packed[None, :] * stride_uqk)
            u_packed = tl.load(u_q_ptrs, mask=g_q_mask, other=0)
            u_unpacked = ((u_packed >> shift[None, :]) & 0xF).to(tl.float32) - 8.0

            u_s_ptrs = up_scales_ptr + offs_n * stride_usn + g_group * stride_usk
            u_scales = tl.load(u_s_ptrs, mask=n_mask, other=0.0).to(tl.float32)
            u_w = (u_unpacked * u_scales[:, None]).to(x.dtype)
            up_acc += tl.dot(x, tl.trans(u_w))

        inner = 0.7978845608028654 * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc)
        tanh_inner = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
        gelu_gate = 0.5 * gate_acc * (1.0 + tanh_inner)
        result = (gelu_gate * up_acc).to(y_ptr.dtype.element_ty)

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, result, mask=m_mask[:, None] & n_mask[None, :])


class PersistentGateUpMLP(torch.nn.Module):
    """Fused gate+up projection + GELU + mul, for decode M <= crossover_m."""

    def __init__(self, gate_proj, up_proj, *, backend="persistent",
                 crossover_m=64, num_ctas=None):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.backend = backend
        self.crossover_m = crossover_m
        self._sm_count = None
        self._num_ctas = num_ctas

    def _get_sm_count(self):
        if self._sm_count is None:
            self._sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        return self._sm_count

    def forward(self, hidden_states):
        if self.backend == "none":
            return (F.gelu(self.gate_proj(hidden_states), approximate="tanh")
                    * self.up_proj(hidden_states))

        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        M, K = x.shape
        N = self.gate_proj.out_features

        if M > self.crossover_m:
            return (F.gelu(self.gate_proj(hidden_states), approximate="tanh")
                    * self.up_proj(hidden_states))

        y = torch.empty(M, N, dtype=hidden_states.dtype, device=hidden_states.device)

        gate_qw = self.gate_proj.qweight
        gate_sc = self.gate_proj.scales
        up_qw = self.up_proj.qweight
        up_sc = self.up_proj.scales

        K_padded = gate_qw.shape[1] * 2

        BLOCK_M = 1 if M == 1 else (2 if M == 2 else 16)
        BLOCK_N = 256 if N >= 10240 else 128
        BLOCK_K = 128
        GROUP_SIZE = 128

        if self.backend == "ordinary":
            grid = (1, triton.cdiv(N, BLOCK_N))
            PERSISTENT = False
            NUM_CTAS = 1
        else:
            sm_count = self._get_sm_count()
            NUM_CTAS = min(sm_count * 2, triton.cdiv(N, BLOCK_N))
            grid = (NUM_CTAS,)
            PERSISTENT = True

        _fused_gate_up_kernel[grid](
            x,
            gate_qw, gate_sc,
            up_qw, up_sc,
            y,
            M, N, K, K_padded,
            x.stride(0), x.stride(1),
            gate_qw.stride(0), gate_qw.stride(1),
            gate_sc.stride(0), gate_sc.stride(1),
            up_qw.stride(0), up_qw.stride(1),
            up_sc.stride(0), up_sc.stride(1),
            y.stride(0), y.stride(1),
            GROUP_SIZE=GROUP_SIZE,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            PERSISTENT=PERSISTENT, NUM_CTAS=NUM_CTAS,
            num_warps=8, num_stages=2,
        )
        return y.reshape(*hidden_states.shape[:-1], N)
