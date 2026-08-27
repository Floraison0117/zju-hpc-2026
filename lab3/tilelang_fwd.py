import os

import torch
import tilelang
import tilelang.language as T

CHUNK_SIZE = 64
HEAD_DIM = 128
OUTPUT_SCALE = HEAD_DIM**-0.5
LOG2E = 1.4426950408889634
VALUE_TILE = int(os.environ.get("GDN_VALUE_TILE", "0"))
GROUP_SIZE = int(os.environ.get("GDN_GROUP_SIZE", "0"))
THREADS = int(os.environ.get("GDN_THREADS", "0"))


def _pick_value_tile(B, Hv, num_chunks):
    if VALUE_TILE in (8, 16, 32, 64, 128):
        return VALUE_TILE
    if Hv >= 64:
        return 128
    if num_chunks >= 64 and B * Hv < 14:
        return 32
    return 64


def _pick_threads(B, Hv):
    if THREADS in (64, 128, 256):
        return THREADS
    return 128


def _pick_group_size(Hv, Hq, num_chunks):
    if GROUP_SIZE in (1, 2, 4, 8):
        return GROUP_SIZE
    return 1


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_persistent_standard(
    Hq,
    Hv,
    value_tile,
    threads,
    qk_dtype,
    v_dtype,
    gate_dtype,
    accum_dtype,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = (num_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE

    qk_shape = (batch_size, num_tokens, Hq, HEAD_DIM)
    v_shape = (batch_size, num_tokens, Hv, HEAD_DIM)
    gate_shape = (batch_size, num_tokens, Hv)
    a_shape = (batch_size, num_tokens, Hv, CHUNK_SIZE)
    state_shape = (batch_size, Hv, HEAD_DIM, HEAD_DIM)
    output_shape = (batch_size, num_tokens, Hv, HEAD_DIM)
    value_tiles = HEAD_DIM // value_tile

    @T.prim_func
    def gdn_persistent_standard(
        q: T.Tensor(qk_shape, qk_dtype),
        k: T.Tensor(qk_shape, qk_dtype),
        v: T.Tensor(v_shape, v_dtype),
        g_cumsum: T.Tensor(gate_shape, gate_dtype),
        beta: T.Tensor(gate_shape, gate_dtype),
        A: T.Tensor(a_shape, qk_dtype),
        state: T.Tensor(state_shape, accum_dtype),
        output: T.Tensor(output_shape, qk_dtype),
        total_blocks: T.int32,
    ):
        with T.Kernel(total_blocks, threads=threads) as (block,):
            value_tile_index = block % value_tiles
            value_head = block // value_tiles % Hv
            batch = block // (value_tiles * Hv)
            value_start = value_tile_index * value_tile
            qk_head = value_head // (Hv // Hq)

            k_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM), dtype=qk_dtype)
            q_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM), dtype=qk_dtype)
            a_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype)
            state_bf16 = T.alloc_shared((HEAD_DIM, value_tile), dtype=qk_dtype)
            residual_bf16 = T.alloc_shared((CHUNK_SIZE, value_tile), dtype=qk_dtype)
            gate_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            gate_exp_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            gate_exp_inv_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            last_gate_exp_shared = T.alloc_shared((1,), dtype=accum_dtype)

            chunk_acc = T.alloc_fragment((CHUNK_SIZE, value_tile), dtype=accum_dtype)
            state_acc = T.alloc_fragment((HEAD_DIM, value_tile), dtype=accum_dtype)
            qk_frag = T.alloc_fragment((CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype)
            residual_bf16_frag = T.alloc_fragment((CHUNK_SIZE, value_tile), dtype=qk_dtype)
            state_bf16_frag = T.alloc_fragment((HEAD_DIM, value_tile), dtype=qk_dtype)

            for key_dim, value_offset in T.Parallel(HEAD_DIM, value_tile):
                state_bf16_frag[key_dim, value_offset] = T.cast(
                    state[batch, value_head, key_dim, value_start + value_offset], qk_dtype
                )
            T.copy(state_bf16_frag, state_bf16)
            T.sync_threads()

            for chunk in T.serial(num_chunks):
                chunk_start = chunk * CHUNK_SIZE
                chunk_length = T.min(CHUNK_SIZE, num_tokens - chunk_start)
                last_token = chunk_start + chunk_length - 1

                T.async_copy(
                    k[batch, chunk_start:chunk_start + CHUNK_SIZE, qk_head, :],
                    k_shared[0:CHUNK_SIZE, 0:HEAD_DIM]
                )
                T.async_copy(
                    q[batch, chunk_start:chunk_start + CHUNK_SIZE, qk_head, :],
                    q_shared[0:CHUNK_SIZE, 0:HEAD_DIM]
                )
                T.ptx_commit_group()

                T.async_copy(
                    A[batch, chunk_start:chunk_start + CHUNK_SIZE, value_head, 0:CHUNK_SIZE],
                    a_shared[0:CHUNK_SIZE, 0:CHUNK_SIZE]
                )

                for token in T.Parallel(CHUNK_SIZE):
                    gate_shared[token] = g_cumsum[
                        batch, chunk_start + T.min(token, chunk_length - 1), value_head
                    ]
                    beta_shared[token] = beta[
                        batch, chunk_start + T.min(token, chunk_length - 1), value_head
                    ]
                    if token >= chunk_length:
                        gate_shared[token] = 0
                        beta_shared[token] = 0

                for token in T.Parallel(CHUNK_SIZE):
                    gate_exp_shared[token] = T.exp2(gate_shared[token] * LOG2E)
                    gate_exp_inv_shared[token] = 1.0 / gate_exp_shared[token]

                for slot in T.Parallel(1):
                    last_gate_exp_shared[slot] = T.exp2(
                        g_cumsum[batch, last_token, value_head] * LOG2E
                    )

                T.ptx_wait_group(0)
                T.sync_threads()

                T.gemm(k_shared, state_bf16, chunk_acc, clear_accum=True)

                for token, value_offset in T.Parallel(CHUNK_SIZE, value_tile):
                    if token < chunk_length:
                        residual_bf16_frag[token, value_offset] = T.cast(
                            beta_shared[token]
                            * (v[batch, chunk_start + token, value_head, value_start + value_offset]
                               - gate_exp_shared[token] * chunk_acc[token, value_offset]),
                            qk_dtype,
                        )
                    else:
                        residual_bf16_frag[token, value_offset] = 0
                T.copy(residual_bf16_frag, residual_bf16)
                T.sync_threads()

                T.gemm(a_shared, residual_bf16, chunk_acc, clear_accum=True)

                for token, value_offset in T.Parallel(CHUNK_SIZE, value_tile):
                    residual_bf16_frag[token, value_offset] = T.cast(
                        chunk_acc[token, value_offset], qk_dtype
                    )
                T.copy(residual_bf16_frag, residual_bf16)

                T.gemm(q_shared, k_shared, qk_frag, transpose_B=True, clear_accum=True)

                for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                    if row < chunk_length and col <= row:
                        a_shared[row, col] = (
                            qk_frag[row, col]
                            * gate_exp_shared[row]
                            * gate_exp_inv_shared[col]
                        )
                    else:
                        a_shared[row, col] = 0

                T.sync_threads()

                T.gemm(q_shared, state_bf16, chunk_acc, clear_accum=True)

                for token, value_offset in T.Parallel(CHUNK_SIZE, value_tile):
                    if token < chunk_length:
                        chunk_acc[token, value_offset] *= gate_exp_shared[token]

                T.gemm(a_shared, residual_bf16, chunk_acc, clear_accum=False)

                for token, value_offset in T.Parallel(CHUNK_SIZE, value_tile):
                    if token < chunk_length:
                        output[batch, chunk_start + token, value_head, value_start + value_offset] = (
                            chunk_acc[token, value_offset] * OUTPUT_SCALE
                        )

                for token, value_offset in T.Parallel(CHUNK_SIZE, value_tile):
                    residual_bf16_frag[token, value_offset] = T.cast(
                        last_gate_exp_shared[0]
                        * gate_exp_inv_shared[token]
                        * T.cast(residual_bf16[token, value_offset], accum_dtype),
                        qk_dtype
                    )
                T.copy(residual_bf16_frag, residual_bf16)
                T.sync_threads()

                T.gemm(k_shared, residual_bf16, state_acc, transpose_A=True, clear_accum=True)

                T.copy(state_bf16, state_bf16_frag)
                for key_dim, value_offset in T.Parallel(HEAD_DIM, value_tile):
                    state_bf16_frag[key_dim, value_offset] = T.cast(
                        last_gate_exp_shared[0] * T.cast(state_bf16_frag[key_dim, value_offset], accum_dtype)
                        + state_acc[key_dim, value_offset],
                        qk_dtype
                    )
                T.copy(state_bf16_frag, state_bf16)
                T.sync_threads()

            for key_dim, value_offset in T.Parallel(HEAD_DIM, value_tile):
                state[batch, value_head, key_dim, value_start + value_offset] = state_bf16[key_dim, value_offset]

    return gdn_persistent_standard


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_persistent_grouped(
    Hq,
    Hv,
    group_size,
    value_tile,
    qk_dtype,
    v_dtype,
    gate_dtype,
    accum_dtype,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = (num_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE

    qk_shape = (batch_size, num_tokens, Hq, HEAD_DIM)
    v_shape = (batch_size, num_tokens, Hv, HEAD_DIM)
    gate_shape = (batch_size, num_tokens, Hv)
    a_shape = (batch_size, num_tokens, Hv, CHUNK_SIZE)
    state_shape = (batch_size, Hv, HEAD_DIM, HEAD_DIM)
    output_shape = (batch_size, num_tokens, Hv, HEAD_DIM)
    value_tiles = HEAD_DIM // value_tile
    gva_ratio = Hv // Hq
    groups_per_qk_head = tilelang.cdiv(gva_ratio, group_size)

    @T.prim_func
    def gdn_persistent_grouped(
        q: T.Tensor(qk_shape, qk_dtype),
        k: T.Tensor(qk_shape, qk_dtype),
        v: T.Tensor(v_shape, v_dtype),
        g_cumsum: T.Tensor(gate_shape, gate_dtype),
        beta: T.Tensor(gate_shape, gate_dtype),
        A: T.Tensor(a_shape, qk_dtype),
        state: T.Tensor(state_shape, accum_dtype),
        output: T.Tensor(output_shape, qk_dtype),
        total_blocks: T.int32,
    ):
        with T.Kernel(total_blocks, threads=128) as (block,):
            value_tile_index = block % value_tiles
            group_index = (block // value_tiles) % groups_per_qk_head
            qk_head = (block // (value_tiles * groups_per_qk_head)) % Hq
            batch = block // (value_tiles * groups_per_qk_head * Hq)
            value_start = value_tile_index * value_tile

            state_tile = T.alloc_fragment(
                (group_size, HEAD_DIM, value_tile), dtype=accum_dtype
            )
            k_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM), dtype=qk_dtype)
            q_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM), dtype=qk_dtype)
            a_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype)
            state_bf16 = T.alloc_shared((HEAD_DIM, value_tile), dtype=qk_dtype)
            residual_bf16 = T.alloc_shared((CHUNK_SIZE, value_tile), dtype=qk_dtype)
            gate_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            gate_exp_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            gate_exp_inv_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            last_gate_exp_shared = T.alloc_shared((1,), dtype=accum_dtype)

            chunk_acc = T.alloc_fragment((CHUNK_SIZE, value_tile), dtype=accum_dtype)
            state_acc = T.alloc_fragment((HEAD_DIM, value_tile), dtype=accum_dtype)
            qk_frag = T.alloc_fragment((CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype)
            residual_bf16_frag = T.alloc_fragment((CHUNK_SIZE, value_tile), dtype=qk_dtype)
            state_bf16_frag = T.alloc_fragment((HEAD_DIM, value_tile), dtype=qk_dtype)

            for local_head in T.serial(group_size):
                value_head = qk_head * gva_ratio + group_index * group_size + local_head
                if value_head >= Hv:
                    pass
                else:
                    for key_dim, value_offset in T.Parallel(HEAD_DIM, value_tile):
                        state_tile[local_head, key_dim, value_offset] = state[
                            batch, value_head, key_dim, value_start + value_offset
                        ]
            T.sync_threads()

            for chunk in T.serial(num_chunks):
                chunk_start = chunk * CHUNK_SIZE
                chunk_length = T.min(CHUNK_SIZE, num_tokens - chunk_start)
                last_token = chunk_start + chunk_length - 1

                T.async_copy(
                    k[batch, chunk_start:chunk_start + CHUNK_SIZE, qk_head, :],
                    k_shared[0:CHUNK_SIZE, 0:HEAD_DIM]
                )
                T.async_copy(
                    q[batch, chunk_start:chunk_start + CHUNK_SIZE, qk_head, :],
                    q_shared[0:CHUNK_SIZE, 0:HEAD_DIM]
                )
                T.ptx_commit_group()

                T.ptx_wait_group(0)
                T.sync_threads()

                T.gemm(q_shared, k_shared, qk_frag, transpose_B=True, clear_accum=True)

                for local_head in T.serial(group_size):
                    value_head = qk_head * gva_ratio + group_index * group_size + local_head
                    if value_head >= Hv:
                        continue

                    for key_dim, value_offset in T.Parallel(HEAD_DIM, value_tile):
                        state_bf16_frag[key_dim, value_offset] = T.cast(
                            state_tile[local_head, key_dim, value_offset], qk_dtype
                        )
                    T.copy(state_bf16_frag, state_bf16)
                    T.sync_threads()

                    for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        a_shared[row, col] = A[
                            batch,
                            chunk_start + T.min(row, chunk_length - 1),
                            value_head,
                            col,
                        ]
                    for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        if row >= chunk_length or col > row:
                            a_shared[row, col] = 0
                    for token in T.Parallel(CHUNK_SIZE):
                        gate_shared[token] = g_cumsum[
                            batch,
                            chunk_start + T.min(token, chunk_length - 1),
                            value_head,
                        ]
                        beta_shared[token] = beta[
                            batch,
                            chunk_start + T.min(token, chunk_length - 1),
                            value_head,
                        ]
                    for token in T.Parallel(CHUNK_SIZE):
                        if token >= chunk_length:
                            gate_shared[token] = 0
                            beta_shared[token] = 0
                    for token in T.Parallel(CHUNK_SIZE):
                        gate_exp_shared[token] = T.exp2(gate_shared[token] * LOG2E)
                        gate_exp_inv_shared[token] = 1.0 / gate_exp_shared[token]
                    for slot in T.Parallel(1):
                        last_gate_exp_shared[slot] = T.exp2(
                            g_cumsum[batch, last_token, value_head] * LOG2E
                        )
                    T.sync_threads()

                    T.gemm(k_shared, state_bf16, chunk_acc, clear_accum=True)

                    for token, value_offset in T.Parallel(CHUNK_SIZE, value_tile):
                        if token < chunk_length:
                            residual_bf16_frag[token, value_offset] = T.cast(
                                beta_shared[token]
                                * (
                                    v[
                                        batch,
                                        chunk_start + token,
                                        value_head,
                                        value_start + value_offset,
                                    ]
                                    - gate_exp_shared[token] * chunk_acc[token, value_offset]
                                ),
                                qk_dtype,
                            )
                        else:
                            residual_bf16_frag[token, value_offset] = 0
                    T.copy(residual_bf16_frag, residual_bf16)
                    T.sync_threads()

                    T.gemm(a_shared, residual_bf16, chunk_acc, clear_accum=True)

                    for token, value_offset in T.Parallel(CHUNK_SIZE, value_tile):
                        residual_bf16_frag[token, value_offset] = T.cast(
                            chunk_acc[token, value_offset], qk_dtype
                        )
                    T.copy(residual_bf16_frag, residual_bf16)
                    T.sync_threads()

                    for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        if row < chunk_length and col <= row:
                            a_shared[row, col] = (
                                qk_frag[row, col]
                                * gate_exp_shared[row]
                                * gate_exp_inv_shared[col]
                            )
                        else:
                            a_shared[row, col] = 0
                    T.sync_threads()

                    T.gemm(q_shared, state_bf16, chunk_acc, clear_accum=True)

                    for token, value_offset in T.Parallel(CHUNK_SIZE, value_tile):
                        if token < chunk_length:
                            chunk_acc[token, value_offset] *= gate_exp_shared[token]

                    T.gemm(a_shared, residual_bf16, chunk_acc, clear_accum=False)

                    for token, value_offset in T.Parallel(CHUNK_SIZE, value_tile):
                        if token < chunk_length:
                            output[
                                batch,
                                chunk_start + token,
                                value_head,
                                value_start + value_offset,
                            ] = chunk_acc[token, value_offset] * OUTPUT_SCALE
                    T.sync_threads()

                    for token, value_offset in T.Parallel(CHUNK_SIZE, value_tile):
                        residual_bf16_frag[token, value_offset] = T.cast(
                            last_gate_exp_shared[0]
                            * gate_exp_inv_shared[token]
                            * T.cast(residual_bf16[token, value_offset], accum_dtype),
                            qk_dtype
                        )
                    T.copy(residual_bf16_frag, residual_bf16)
                    T.sync_threads()

                    T.gemm(k_shared, residual_bf16, state_acc, transpose_A=True, clear_accum=True)

                    for key_dim, value_offset in T.Parallel(HEAD_DIM, value_tile):
                        state_tile[local_head, key_dim, value_offset] = (
                            last_gate_exp_shared[0]
                            * state_tile[local_head, key_dim, value_offset]
                            + state_acc[key_dim, value_offset]
                        )
                    T.sync_threads()

            for local_head in T.serial(group_size):
                value_head = qk_head * gva_ratio + group_index * group_size + local_head
                if value_head >= Hv:
                    pass
                else:
                    for key_dim, value_offset in T.Parallel(HEAD_DIM, value_tile):
                        state[batch, value_head, key_dim, value_start + value_offset] = state_tile[
                            local_head, key_dim, value_offset
                        ]

    return gdn_persistent_grouped


def _standard_kernel_forward(q, k, v, g_cumsum, beta, A, state, output, value_tile, threads):
    batch_size, _, num_heads_qk, _ = q.shape
    num_heads_v = v.shape[2]
    value_tiles = HEAD_DIM // value_tile
    total_blocks = batch_size * num_heads_v * value_tiles
    kernel = tilelang_persistent_standard(
        num_heads_qk, num_heads_v, value_tile, threads,
        qk_dtype=q.dtype, v_dtype=v.dtype,
        gate_dtype=g_cumsum.dtype, accum_dtype="float32",
    )
    kernel(q, k, v, g_cumsum, beta, A, state, output, total_blocks)


def _grouped_kernel_forward(q, k, v, g_cumsum, beta, A, state, output, value_tile, group_size):
    batch_size, _, num_heads_qk, _ = q.shape
    num_heads_v = v.shape[2]
    gva_ratio = num_heads_v // num_heads_qk
    value_tiles = HEAD_DIM // value_tile
    groups_per_qk_head = tilelang.cdiv(gva_ratio, group_size)
    total_blocks = batch_size * num_heads_qk * groups_per_qk_head * value_tiles
    kernel = tilelang_persistent_grouped(
        num_heads_qk, num_heads_v, group_size, value_tile,
        qk_dtype=q.dtype, v_dtype=v.dtype,
        gate_dtype=g_cumsum.dtype, accum_dtype="float32",
    )
    kernel(q, k, v, g_cumsum, beta, A, state, output, total_blocks)


def gdn_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_tokens, num_heads_qk, head_dim = q.shape
    if k.shape != q.shape or k.dtype != q.dtype:
        raise ValueError("q and k must have the same shape and dtype")
    if head_dim != HEAD_DIM:
        raise ValueError(f"expected head dimension {HEAD_DIM}, got {head_dim}")
    if v.shape[:2] != (batch_size, num_tokens) or v.shape[3] != HEAD_DIM:
        raise ValueError("v must have shape [B, T, Hv, 128]")
    num_heads_v = v.shape[2]
    if num_heads_v % num_heads_qk != 0:
        raise ValueError("Hv must be divisible by Hq")
    if g_cumsum.shape != (batch_size, num_tokens, num_heads_v):
        raise ValueError("g_cumsum has an invalid shape")
    if beta.shape != g_cumsum.shape:
        raise ValueError("beta must have the same shape as g_cumsum")
    if A.shape != (batch_size, num_tokens, num_heads_v, CHUNK_SIZE):
        raise ValueError("A must have shape [B, T, Hv, 64]")

    state_shape = (batch_size, num_heads_v, HEAD_DIM, HEAD_DIM)
    if initial_state is None:
        state = torch.zeros(state_shape, dtype=torch.float32, device=q.device)
    else:
        if initial_state.shape != state_shape or initial_state.dtype != torch.float32:
            raise ValueError("initial_state must be FP32 with shape [B, Hv, 128, 128]")
        state = initial_state.clone()

    output = torch.empty(
        (batch_size, num_tokens, num_heads_v, HEAD_DIM),
        dtype=q.dtype,
        device=q.device,
    )

    num_chunks = tilelang.cdiv(num_tokens, CHUNK_SIZE)
    value_tile = _pick_value_tile(batch_size, num_heads_v, num_chunks)
    group_size = _pick_group_size(num_heads_v, num_heads_qk, num_chunks)
    threads = _pick_threads(batch_size, num_heads_v)

    if group_size >= 2:
        _grouped_kernel_forward(
            q, k, v, g_cumsum, beta, A, state, output, value_tile, group_size,
        )
    else:
        _standard_kernel_forward(
            q, k, v, g_cumsum, beta, A, state, output, value_tile, threads,
        )

    return output, state
