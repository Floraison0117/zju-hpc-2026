"""Hybrid Quantized Linear: fused Triton for small M, dequant+BF16 for large M."""

from __future__ import annotations

import torch
from hpc101_infer.layers.triton_linear import FusedQuantizedLinear, FusedQuantizedLinearFactory


class HybridQuantizedLinear(FusedQuantizedLinear):
    def __init__(self, *args, crossover_m: int = 64, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._crossover_m = crossover_m

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        M = inputs.numel() // self.in_features
        if M <= self._crossover_m:
            return super().forward(inputs)
        from hpc101_infer.quantization.packing import dequantize_weight
        weight = dequantize_weight(self.quantized_weight(), dtype=inputs.dtype)
        from torch.nn import functional as F
        return F.linear(inputs, weight, self.bias)


class HybridQuantizedLinearFactory(FusedQuantizedLinearFactory):
    def __init__(self, *args, crossover_m: int = 64, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._crossover_m = crossover_m

    def create(self, module_name, in_features, out_features, bias, device=None, dtype=None):
        entry = self.manifest.get(module_name)
        if entry is None:
            return self._fallback.create(module_name, in_features, out_features, bias, device, dtype)
        if entry.original_shape != (out_features, in_features):
            raise RuntimeError("manifest shape mismatch for " + module_name)
        scale_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else self.scale_dtype
        return HybridQuantizedLinear(
            in_features, out_features, entry.group_size,
            symmetric=entry.symmetric,
            padded_in_features=entry.padded_shape[1],
            bias=bias, device=device, scale_dtype=scale_dtype,
            crossover_m=self._crossover_m,
        )