"""Per-slot KV cache with dynamic slot allocation. Single source of truth: layer.lengths."""

from __future__ import annotations

import torch
from hpc101_infer.models.config import Gemma4TextConfig
from hpc101_infer.runtime.kv_cache import KVCache, LayerKVCache, LayerKVView


class SlotKVCache(KVCache):
    def __init__(self, layers, max_batch_size, max_sequence_length):
        super().__init__(layers, max_batch_size, max_sequence_length)
        self.slot_in_use = torch.zeros(max_batch_size, dtype=torch.bool, device=self.device)

    @classmethod
    def allocate(cls, config, max_batch_size, max_sequence_length, dtype, device):
        return cls(*KVCache.allocate(config, max_batch_size, max_sequence_length, dtype, device).__dict__.values())

    def allocate_slot(self) -> int:
        free = (~self.slot_in_use).nonzero()
        if free.numel() == 0:
            raise RuntimeError("no free KV slots available")
        slot_id = int(free[0].item())
        self.slot_in_use[slot_id] = True
        for layer in self.layers:
            layer.lengths[slot_id] = 0
        return slot_id

    def release_slot(self, slot_id: int) -> None:
        if not self.slot_in_use[slot_id]:
            raise ValueError(f"slot {slot_id} is already free")
        self.slot_in_use[slot_id] = False
        for layer in self.layers:
            layer.lengths[slot_id] = 0

    def get_active_slots(self) -> torch.Tensor:
        return self.slot_in_use.nonzero().flatten()

    def get_active_mask(self) -> torch.Tensor:
        return self.slot_in_use.clone()

    def commit_slot_lengths(self, slot_ids: torch.Tensor, lengths: torch.Tensor) -> None:
        for layer in self.layers:
            layer.lengths[slot_ids] = lengths

    def view_for_slots(self, layer_id: int, slot_ids: torch.Tensor, max_length: int) -> LayerKVView:
        layer = self.layers[layer_id]
        return LayerKVView(
            key=layer.key[slot_ids, :, :max_length, :].contiguous(),
            value=layer.value[slot_ids, :, :max_length, :].contiguous(),
        )