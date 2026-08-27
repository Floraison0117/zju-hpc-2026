"""Flash Attention Triton kernel，在线 softmax 融合因果/滑动窗口掩码。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    positions_ptr,
    seq_lengths_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_pb,
    q_len,
    k_len,
    num_kv_groups,
    sliding_window,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_SLIDING: tl.constexpr,
    RING_INDEXED: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    kv_head = pid_h // num_kv_groups

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = (
        q_ptr
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q_mask = offs_m[:, None] < q_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    q_positions = tl.load(
        positions_ptr + pid_b * stride_pb + offs_m,
        mask=offs_m < q_len,
        other=0,
    )
    seq_len = tl.load(seq_lengths_ptr + pid_b)

    query_valid = q_positions < seq_len

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for n_start in range(0, k_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        if RING_INDEXED:
            phys_n = offs_n % sliding_window
        else:
            phys_n = offs_n

        k_ptrs = (
            k_ptr
            + pid_b * stride_kb
            + kv_head * stride_kh
            + phys_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        if RING_INDEXED:
            k_load_mask = offs_n[:, None] < seq_len
            k = tl.load(k_ptrs, mask=k_load_mask, other=0.0)
        else:
            k_mask = offs_n[:, None] < k_len
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32)

        causal = offs_n[None, :] <= q_positions[:, None]
        pad = offs_n[None, :] < seq_len
        mask = causal & pad
        if IS_SLIDING:
            sliding = offs_n[None, :] > (q_positions[:, None] - sliding_window)
            mask = mask & sliding

        mask = mask | (~query_valid[:, None] & (offs_n[None, :] == 0))

        scores = tl.where(mask, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)

        v_ptrs = (
            v_ptr
            + pid_b * stride_vb
            + kv_head * stride_vh
            + phys_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        if RING_INDEXED:
            v = tl.load(v_ptrs, mask=k_load_mask, other=0.0)
        else:
            v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)

        m_i = m_new
        l_i = l_new

    safe_l = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / safe_l[:, None]
    acc = tl.where(query_valid[:, None], acc, 0.0)

    o_ptrs = (
        o_ptr
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    o_mask = offs_m[:, None] < q_len
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=o_mask)


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    layer_type: str,
    sliding_window: int = -1,
    ring_indexed: bool = False,
    logical_k_len: int | None = None,
) -> torch.Tensor:
    """Flash Attention 前向。

    query:  [batch, num_q_heads, q_len, head_dim]
    key:    [batch, num_kv_heads, k_len, head_dim]
    value:  [batch, num_kv_heads, k_len, head_dim]
    返回:    [batch, q_len, num_q_heads * head_dim]
    """
    batch, num_q_heads, q_len, head_dim = query.shape
    _, num_kv_heads, k_len_physical, _ = key.shape
    if ring_indexed and logical_k_len is not None:
        k_len = logical_k_len
    else:
        k_len = k_len_physical

    output = torch.zeros_like(query)

    if head_dim >= 512:
        BLOCK_M = 32 if q_len > 16 else 16
        BLOCK_N = 16
        num_warps = 4
        num_stages = 1
    else:
        BLOCK_M = 64 if q_len > 16 else 16
        BLOCK_N = 32
        num_warps = 4
        num_stages = 2

    is_sliding = layer_type == "sliding_attention"
    if sliding_window <= 0:
        sliding_window = 0
    ring_indexed = ring_indexed and is_sliding

    grid = (batch, num_q_heads, triton.cdiv(q_len, BLOCK_M))
    _flash_attn_kernel[grid](
        query,
        key,
        value,
        output,
        positions,
        sequence_lengths,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        positions.stride(0),
        q_len,
        k_len,
        num_q_heads // num_kv_heads,
        sliding_window,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_SLIDING=is_sliding,
        RING_INDEXED=ring_indexed,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output.transpose(1, 2).reshape(batch, q_len, -1)
