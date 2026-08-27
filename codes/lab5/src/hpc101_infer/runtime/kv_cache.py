"""KV cache with ring-buffer support for sliding-window attention layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hpc101_infer.models.config import Gemma4TextConfig


@dataclass
class LayerKVCache:
    key: torch.Tensor
    value: torch.Tensor
    lengths: torch.Tensor
    max_batch_size: int
    max_sequence_length: int
    batch_size: int = 0
    window_size: int = 0

    def reset(self, batch_size: int) -> None:
        if not 0 < batch_size <= self.max_batch_size:
            raise ValueError(
                f"batch_size must be in [1, {self.max_batch_size}], "
                f"got {batch_size}"
            )
        self.batch_size = batch_size
        self.lengths.zero_()

    def write(
        self,
        positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_ids: torch.Tensor | None = None,
    ) -> None:
        batch_size, query_length = positions.shape
        expected_prefix = (batch_size, self.key.shape[1])
        expected_suffix = (query_length, self.key.shape[3])
        if key.shape != expected_prefix + expected_suffix:
            raise ValueError(
                f"invalid key shape {tuple(key.shape)}, expected "
                f"{expected_prefix + expected_suffix}"
            )
        phys_slots = slot_ids if slot_ids is not None and slot_ids.numel() > 0 else None
        for batch_idx in range(batch_size):
            target = positions[batch_idx]
            if self.window_size:
                target = target % self.window_size
            if phys_slots is not None:
                phys = phys_slots[batch_idx]
            else:
                phys = batch_idx
            self.key[phys].index_copy_(1, target, key[batch_idx])
            self.value[phys].index_copy_(1, target, value[batch_idx])

    def view(self, max_length: int, ring_indexed: bool = False, slot_ids: torch.Tensor | None = None) -> "LayerKVView":
        if slot_ids is not None and slot_ids.numel() > 0:
            idx = slot_ids
            bs = len(slot_ids)
        else:
            idx = slice(None, self.batch_size)
            bs = self.batch_size
        if not self.window_size or max_length <= self.window_size:
            return LayerKVView(
                key=self.key[idx, :, :max_length, :],
                value=self.value[idx, :, :max_length, :],
            )
        if ring_indexed:
            return LayerKVView(
                key=self.key[idx, :, : self.window_size, :],
                value=self.value[idx, :, : self.window_size, :],
            )
        logical_len = min(max_length, int(self.lengths[idx].max().item()))
        if logical_len <= self.window_size:
            return LayerKVView(
                key=self.key[idx, :, :logical_len, :],
                value=self.value[idx, :, :logical_len, :],
            )
        start = logical_len % self.window_size
        rolled_k = torch.roll(self.key[idx, :, :logical_len, :], shifts=-start, dims=2)
        rolled_v = torch.roll(self.value[idx, :, :logical_len, :], shifts=-start, dims=2)
        return LayerKVView(key=rolled_k, value=rolled_v)

    def commit(self, sequence_lengths: torch.Tensor, slot_ids: torch.Tensor | None = None) -> None:
        if slot_ids is not None and slot_ids.numel() > 0:
            self.lengths[slot_ids] = sequence_lengths
        else:
            self.lengths[: self.batch_size].copy_(sequence_lengths)


@dataclass(frozen=True)
class LayerKVView:
    key: torch.Tensor
    value: torch.Tensor


class KVCache:
    def __init__(
        self,
        layers: list[LayerKVCache],
        max_batch_size: int,
        max_sequence_length: int,
    ) -> None:
        if not layers:
            raise ValueError("KV cache must contain at least one layer")
        self.layers = layers
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length

    @classmethod
    def allocate(
        cls,
        config: Gemma4TextConfig,
        max_batch_size: int,
        max_sequence_length: int,
        dtype: torch.dtype,
        device: str | torch.device,
        window_aware: bool = False,
    ) -> "KVCache":
        if max_batch_size <= 0 or max_sequence_length <= 0:
            raise ValueError("cache capacities must be positive")
        device = torch.device(device)
        layers = []
        for layer_type in config.layer_types:
            if layer_type == "full_attention":
                kv_heads = config.num_global_key_value_heads
                head_dim = config.global_head_dim
            elif layer_type == "sliding_attention":
                kv_heads = config.num_key_value_heads
                head_dim = config.head_dim
            else:
                raise ValueError(f"unsupported attention type: {layer_type!r}")
            # 滑窗层的 KV 只需保留最近 sliding_window 个位置；启用 window_aware
            # 后把容量从 max_sequence_length 降到 sliding_window 并设置 window_size，
            # 使 decode 走 ring_indexed 取模寻址。full 层与未启用时保持 dense。
            if (
                window_aware
                and layer_type == "sliding_attention"
                and config.sliding_window > 0
            ):
                layer_max_seq = min(max_sequence_length, config.sliding_window)
                layer_window = config.sliding_window
            else:
                layer_max_seq = max_sequence_length
                layer_window = 0
            shape = (max_batch_size, kv_heads, layer_max_seq, head_dim)
            layers.append(
                LayerKVCache(
                    key=torch.empty(shape, dtype=dtype, device=device),
                    value=torch.empty(shape, dtype=dtype, device=device),
                    lengths=torch.zeros(max_batch_size, dtype=torch.long, device=device),
                    max_batch_size=max_batch_size,
                    max_sequence_length=layer_max_seq,
                    window_size=layer_window,
                )
            )
        return cls(layers, max_batch_size, max_sequence_length)

    @property
    def lengths(self) -> torch.Tensor:
        return self.layers[0].lengths

    @property
    def batch_size(self) -> int:
        return self.layers[0].batch_size

    @property
    def device(self) -> torch.device:
        return self.lengths.device

    @property
    def dtype(self) -> torch.dtype:
        return self.layers[0].key.dtype

    def reset(self, batch_size: int) -> None:
        for layer in self.layers:
            layer.reset(batch_size)

    def write(self, layer_id: int, positions: torch.Tensor, key: torch.Tensor, value: torch.Tensor, slot_ids: torch.Tensor | None = None) -> None:
        self.layers[layer_id].write(positions, key, value, slot_ids=slot_ids)

    def view(self, layer_id: int, max_length: int, ring_indexed: bool = False, slot_ids: torch.Tensor | None = None) -> LayerKVView:
        return self.layers[layer_id].view(max_length, ring_indexed=ring_indexed, slot_ids=slot_ids)

    def commit(self, sequence_lengths: torch.Tensor, slot_ids: torch.Tensor | None = None) -> None:
        for layer in self.layers:
            layer.commit(sequence_lengths, slot_ids=slot_ids)
