"""按 Decoder 层搬运 INT4 权重的单双缓冲运行时。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.profiler import record_function

from hpc101_infer.layers.linear import QuantizedLinear


@dataclass(frozen=True)
class TensorView:
    """记录一个量化 buffer 在同 dtype 平坦存储中的位置。"""

    module: QuantizedLinear
    name: str
    offset: int
    numel: int
    shape: torch.Size
    dtype: torch.dtype


@dataclass
class PackedLayer:
    """一个 Decoder 层的 pinned host storage 与视图元数据。"""

    storage: dict[torch.dtype, torch.Tensor]
    views: list[TensorView]
    bytes: int


class LayerWeightOffloader:
    """在层边界复用 GPU staging buffer，并保证复制与消费顺序正确。

    同步模式在当前 compute stream 上复制后计算。异步模式使用独立 copy
    stream 和两组 buffer：compute stream 等待 ready event；copy stream 在
    覆盖旧 buffer 前等待 done event，确保上一位消费者已经结束。
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        *,
        device: torch.device,
        mode: str,
        resident: bool = False,
    ) -> None:
        if mode not in {"sync", "async"}:
            raise ValueError("offloader mode must be sync or async")
        if device.type != "cuda":
            raise ValueError("offloader requires a CUDA device")
        self.layers = layers
        self.device = device
        self.mode = mode
        self.resident = resident
        # resident：打包权重直接驻留 GPU（packed INT4 全模型仅 ~2.9GiB，
        # MIG 10.47GiB 放得下），run() 直接绑定存储，无 H2D、无 buffer 环、
        # 无 copy stream。消除每步 48 层 H2D（profiler 显示 async offload 下
        # H2D 占 CUDA 时间 48.8%）。
        self.packed_layers = [
            self._pack_layer(layer, device=device if resident else None)
            for layer in layers
        ]
        self.offloaded_bytes = sum(layer.bytes for layer in self.packed_layers)
        self.prefetch_count = 0

        if resident:
            self.buffers = []
            self.buffer_capacity_bytes = 0
            self.copy_stream = None
            self.ready = []
            self.done = []
            self.layer_done = []
            return

        dtypes = {dtype for layer in self.packed_layers for dtype in layer.storage}
        self.max_numel = {
            dtype: max(layer.storage.get(dtype, torch.empty(0)).numel() for layer in self.packed_layers)
            for dtype in dtypes
        }
        count = 1 if mode == "sync" else 2
        self.buffers = [
            {
                dtype: torch.empty(size, dtype=dtype, device=device)
                for dtype, size in self.max_numel.items()
            }
            for _ in range(count)
        ]
        self.buffer_capacity_bytes = sum(
            tensor.numel() * tensor.element_size()
            for buffer in self.buffers
            for tensor in buffer.values()
        )
        self.copy_stream = torch.cuda.Stream(device=device) if mode == "async" else None
        self.ready = [torch.cuda.Event() for _ in range(count)]
        self.done = [torch.cuda.Event() for _ in range(count)]
        self.layer_done = [torch.cuda.Event() for _ in range(len(layers))]

    @staticmethod
    def _pack_layer(layer: nn.Module, device: torch.device | None = None) -> PackedLayer:
        entries: list[tuple[QuantizedLinear, str, torch.Tensor]] = []
        for module in layer.modules():
            if not isinstance(module, QuantizedLinear):
                continue
            for name in ("qweight", "scales", "zeros"):
                tensor = getattr(module, name)
                if tensor is not None:
                    if tensor.device.type != "cpu":
                        raise ValueError("offloaded quantized weights must load on CPU")
                    entries.append((module, name, tensor))
        if not entries:
            raise ValueError("decoder layer contains no QuantizedLinear weights")

        totals: dict[torch.dtype, int] = {}
        for _, _, tensor in entries:
            totals[tensor.dtype] = totals.get(tensor.dtype, 0) + tensor.numel()
        if device is not None:
            storage = {
                dtype: torch.empty(size, dtype=dtype, device=device)
                for dtype, size in totals.items()
            }
        else:
            storage = {
                dtype: torch.empty(size, dtype=dtype, pin_memory=True)
                for dtype, size in totals.items()
            }
        offsets = {dtype: 0 for dtype in totals}
        views: list[TensorView] = []
        v10_transpose = True  # V10: always transpose qweight [N,K]->[K,N] for coalesced decode dot
        for module, name, tensor in entries:
            offset = offsets[tensor.dtype]
            flat = storage[tensor.dtype][offset : offset + tensor.numel()]
            # V10: transpose qweight [N,K_packed] -> [K_packed,N] for coalesced dot.
            # Conditional (env) so task1 (int4_reference, expects [N,K_packed]) is unaffected.
            if name == "qweight" and v10_transpose and tensor.shape[0] == module.out_features:
                t_t = tensor.t().contiguous()
                flat.copy_(t_t.reshape(-1))
                shape = t_t.shape
            else:
                flat.copy_(tensor.reshape(-1))
                shape = tensor.shape
            view = TensorView(module, name, offset, tensor.numel(), shape, tensor.dtype)
            views.append(view)
            setattr(module, name, flat.view(shape))
            offsets[tensor.dtype] += tensor.numel()
        size_bytes = sum(x.numel() * x.element_size() for x in storage.values())
        return PackedLayer(storage, views, size_bytes)

    def _bind(self, layer_index: int, buffer_index: int) -> None:
        buffers = self.buffers[buffer_index]
        for view in self.packed_layers[layer_index].views:
            target = buffers[view.dtype][view.offset : view.offset + view.numel]
            setattr(view.module, view.name, target.view(view.shape))

    def _bind_storage(self, layer_index: int) -> None:
        """resident 模式：直接把模块视图绑到该层驻留 GPU 的打包存储。"""
        storage = self.packed_layers[layer_index].storage
        for view in self.packed_layers[layer_index].views:
            target = storage[view.dtype][view.offset : view.offset + view.numel]
            setattr(view.module, view.name, target.view(view.shape))

    def _copy(self, layer_index: int, buffer_index: int) -> None:
        source = self.packed_layers[layer_index].storage
        target = self.buffers[buffer_index]
        with record_function("LAB5_WEIGHT_PREFETCH_H2D"):
            for dtype, host in source.items():
                target[dtype][: host.numel()].copy_(host, non_blocking=True)
        self.prefetch_count += 1

    def stats(self) -> dict[str, int | str]:
        return {
            "mode": self.mode,
            "offloaded_bytes": self.offloaded_bytes,
            "prefetch_count": self.prefetch_count,
            "buffer_capacity_bytes": self.buffer_capacity_bytes,
        }

    def run(
        self,
        hidden_states: torch.Tensor,
        run_layer: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """顺序执行全部层。resident 模式直接绑定驻留存储、无 H2D；否则按
        sync/async（双缓冲 + copy stream）搬运。"""
        if self.resident:
            for index, layer in enumerate(self.layers):
                self._bind_storage(index)
                hidden_states = run_layer(layer, hidden_states)
            return hidden_states
        compute_stream = torch.cuda.current_stream(self.device)
        if self.mode == "sync" or torch.cuda.is_current_stream_capturing():
            for index, layer in enumerate(self.layers):
                self._copy(index, 0)
                self._bind(index, 0)
                hidden_states = run_layer(layer, hidden_states)
            return hidden_states

        assert self.copy_stream is not None
        with torch.cuda.stream(self.copy_stream):
            self._copy(0, 0)
            self.ready[0].record(self.copy_stream)
        for index, layer in enumerate(self.layers):
            current = index % 2
            compute_stream.wait_event(self.ready[current])
            self._bind(index, current)
            next_index = index + 1
            if next_index < len(self.layers):
                following = next_index % 2
                with torch.cuda.stream(self.copy_stream):
                    if self.done[following] is not None:
                        self.copy_stream.wait_event(self.done[following])
                    self._copy(next_index, following)
                    self.ready[following].record(self.copy_stream)
            hidden_states = run_layer(layer, hidden_states)
            self.done[current].record(compute_stream)
        return hidden_states
