from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from hpc101_infer.quantization.packing import pack_int4
from hpc101_infer.quantization.types import (
    LayerContext,
    LayerQuantizationResult,
    QuantizedWeight,
    SCALE_DTYPES,
)


@dataclass(frozen=True)
class GPTQOptions:
    block_size: int
    damp_percent: float


@dataclass(frozen=True)
class GPTQModuleState:
    activations: torch.Tensor
    activation_tokens: int


@dataclass(frozen=True)
class GPTQLayerState:
    modules: Mapping[str, GPTQModuleState]
    options: GPTQOptions


def _calibration_option(
    calibration: Mapping[str, Any],
    name: str,
    default: int | float,
) -> int | float:
    gptq = calibration.get("gptq", calibration)
    if not isinstance(gptq, Mapping):
        raise TypeError("config.calibration['gptq'] must be a mapping")
    return gptq.get(name, default)


def _parse_gptq_options(calibration: Mapping[str, Any]) -> GPTQOptions:
    block_size = _calibration_option(calibration, "block_size", 128)
    damp_percent = _calibration_option(calibration, "damp_percent", 0.01)

    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("GPTQ block_size must be a positive integer")
    if (
        not isinstance(damp_percent, (int, float))
        or isinstance(damp_percent, bool)
        or not 0.0 < float(damp_percent) < 1.0
    ):
        raise ValueError("GPTQ damp_percent must be in (0, 1)")

    gptq = calibration.get("gptq", calibration)
    if isinstance(gptq, Mapping) and gptq.get("desc_act", False):
        raise ValueError("GPTQ desc_act is not supported by this checkpoint format")
    return GPTQOptions(
        block_size=block_size,
        damp_percent=float(damp_percent),
    )


def quantize_weight_gptq(
    weight: torch.Tensor,
    activations: torch.Tensor,
    group_size: int,
    *,
    block_size: int = 128,
    damp_percent: float = 0.01,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[QuantizedWeight, dict[str, float | int]]:
    """
    使用 GPTQ 算法将权重量化为 INT4。

    参数：
        weight: 待量化的权重张量，形状为 (out_features, in_features)。
        activations: 校准数据集的输入激活，形状为 (calibration_tokens, in_features)。
        group_size: 量化粒度，即每个 group 中的列数。
        block_size: 分块计算时每个 block 中的列数。
        damp_percent: 阻尼比例，用于改善 Hessian 的数值稳定性。
        symmetric: 是否使用对称量化。
        scale_dtype: 缩放因子的 dtype。

    返回：
        quantized_weight: 量化后的权重对象，具体详见 QuantizedWeight 类的定义。
        metadatas: 量化过程中的统计信息，不影响评测，用于分析和调试。
    """

    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point matrix")
    if activations.ndim != 2 or not activations.is_floating_point():
        raise ValueError("activations must be a floating-point matrix")
    if activations.shape[1] != weight.shape[1]:
        raise ValueError("activation width does not match weight input size")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 < damp_percent < 1.0:
        raise ValueError("damp_percent must be in (0, 1)")
    if scale_dtype not in SCALE_DTYPES.values():
        raise ValueError("unsupported scale dtype")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must contain only finite values")

    with torch.no_grad():
        device = weight.device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda")
        W = weight.detach().float().to(device)
        X = activations.detach().float().to(device)
        out_features, in_features = W.shape
        padded_in_features = math.ceil(in_features / group_size) * group_size
        if padded_in_features != in_features:
            W = F.pad(W, (0, padded_in_features - in_features))
            X = F.pad(X, (0, padded_in_features - in_features))

        groups = W.view(out_features, -1, group_size)
        if symmetric:
            scales = groups.abs().amax(dim=-1) / 7.0
            scales = scales.clamp_min(torch.finfo(torch.float32).eps)
            zeros = None
        else:
            zero = torch.zeros_like(groups[..., 0])
            minimum = torch.minimum(groups.amin(dim=-1), zero)
            maximum = torch.maximum(groups.amax(dim=-1), zero)
            scales = ((maximum - minimum) / 15.0).clamp_min(
                torch.finfo(torch.float32).eps
            )
            zeros = torch.round(-minimum / scales).clamp(0, 15).to(torch.uint8)

        N = X.shape[0]
        H = (2.0 / N) * (X.T @ X)
        del X

        dead = torch.diag(H) == 0
        dead_count = int(dead.sum().item())
        if dead_count > 0:
            H[dead, dead] = 1.0

        damp = damp_percent * float(torch.diag(H).mean())
        H += damp * torch.eye(padded_in_features, dtype=H.dtype, device=H.device)
        H = (H + H.T) / 2

        L, h_info = torch.linalg.cholesky_ex(H)
        h_retries = 0
        while int(h_info.max().item()) != 0 and h_retries < 6:
            H.diagonal().add_(damp * (10 ** h_retries))
            L, h_info = torch.linalg.cholesky_ex(H)
            h_retries += 1
        if int(h_info.max().item()) != 0:
            raise RuntimeError(
                "damped Hessian is not positive definite after "
                f"{h_retries} stabilization retries"
            )
        del h_info, H

        G = torch.cholesky_inverse(L)
        del L
        G = (G + G.T) / 2

        U, inverse_info = torch.linalg.cholesky_ex(G, upper=True)
        inverse_hessian_retries = 0
        inverse_jitter = max(
            torch.finfo(G.dtype).eps * float(torch.diag(G).abs().mean()),
            torch.finfo(G.dtype).tiny,
        )
        while int(inverse_info.max().item()) != 0 and inverse_hessian_retries < 6:
            del U, inverse_info
            G.diagonal().add_(inverse_jitter * (10 ** inverse_hessian_retries))
            U, inverse_info = torch.linalg.cholesky_ex(G, upper=True)
            inverse_hessian_retries += 1
        if int(inverse_info.max().item()) != 0:
            raise RuntimeError(
                "inverse Hessian Cholesky failed after "
                f"{inverse_hessian_retries} stabilization retries"
            )
        del inverse_info, G

        Q = torch.zeros_like(W)
        predicted_loss = torch.zeros((), dtype=W.dtype, device=device)

        for i1 in range(0, padded_in_features, block_size):
            i2 = min(i1 + block_size, padded_in_features)
            block_width = i2 - i1
            block_errs = torch.zeros(
                out_features, block_width, dtype=W.dtype, device=device
            )

            for local_j in range(block_width):
                j = i1 + local_j
                g = j // group_size
                scale_j = scales[:, g]

                if symmetric:
                    q = torch.round(W[:, j] / scale_j).clamp(-8, 7)
                    dequant = q * scale_j
                else:
                    z_j = zeros[:, g].float()
                    q = torch.round(W[:, j] / scale_j + z_j).clamp(0, 15)
                    dequant = (q - z_j) * scale_j

                Q[:, j] = q
                err = (W[:, j] - dequant) / U[j, j]
                block_errs[:, local_j] = err
                predicted_loss.add_(0.5 * err.square().sum())

                if local_j + 1 < block_width:
                    W[:, j + 1 : i2] -= (
                        err.unsqueeze(1) * U[j, j + 1 : i2].unsqueeze(0)
                    )

            if i2 < padded_in_features:
                W[:, i2:] -= block_errs @ U[i1:i2, i2:]

        del U

        if symmetric:
            encoded = (Q.to(torch.int16) + 8).to(torch.uint8)
        else:
            encoded = Q.to(torch.uint8)

        quantized = QuantizedWeight(
            qweight=pack_int4(encoded.reshape(out_features, padded_in_features)).cpu(),
            scales=scales.to(scale_dtype).cpu(),
            zeros=(zeros.cpu() if zeros is not None else None),
            original_shape=(out_features, in_features),
            padded_shape=(out_features, padded_in_features),
            bits=4,
            group_size=group_size,
            symmetric=symmetric,
            packing="uint8_little_nibble",
        )

        if symmetric:
            W_dequant = Q * scales.repeat_interleave(group_size, dim=1)
        else:
            W_dequant = (Q - zeros.float().repeat_interleave(group_size, dim=1)) * (
                scales.repeat_interleave(group_size, dim=1)
            )

        weight_mse = float(
            ((weight.detach().float().to(device)[:, :in_features] - W_dequant[:, :in_features]) ** 2)
            .mean()
            .item()
        )
        predicted_loss_value = float(predicted_loss.item())

    metadata: dict[str, float | int] = {
        "activation_tokens": int(activations.shape[0]),
        "block_size": int(block_size),
        "damp_percent": float(damp_percent),
        "dead_columns": dead_count,
        "hessian_retries": h_retries,
        "inverse_hessian_retries": inverse_hessian_retries,
        "inverse_hessian_jitter": float(inverse_jitter),
        "predicted_loss": predicted_loss_value,
        "weight_mse": weight_mse,
    }
    return quantized, metadata


class GPTQQuantizationMethod:
    name = "gptq"
    version = "1"

    def calibrate_layer(self, context: LayerContext) -> GPTQLayerState:
        if context.activations is None:
            raise ValueError("GPTQ requires calibration activations")

        options = _parse_gptq_options(context.config.calibration or {})
        modules = dict(context.layer.named_modules())
        states: dict[str, GPTQModuleState] = {}
        for name in context.target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            if name not in context.activations:
                raise ValueError(f"missing calibration activations for {name}")

            activations = context.activations[name]
            if activations.ndim != 2 or activations.shape[1] != module.in_features:
                raise ValueError(
                    f"invalid calibration activation shape for {name}: "
                    f"expected [tokens, {module.in_features}], "
                    f"got {tuple(activations.shape)}"
                )
            if activations.shape[0] == 0:
                raise ValueError(f"calibration activations for {name} are empty")
            if not activations.is_floating_point():
                raise TypeError(f"calibration activations for {name} must be floating")
            if not torch.isfinite(activations).all():
                raise ValueError(f"calibration activations for {name} are not finite")

            states[name] = GPTQModuleState(
                activations=activations.detach(),
                activation_tokens=activations.shape[0],
            )

        return GPTQLayerState(modules=states, options=options)

    def quantize_layer(
        self, context: LayerContext, state: GPTQLayerState
    ) -> LayerQuantizationResult:
        if not isinstance(state, GPTQLayerState):
            raise TypeError("state must be a GPTQLayerState")

        scale_dtype = SCALE_DTYPES[context.config.scale_dtype]
        modules = dict(context.layer.named_modules())
        weights: dict[str, QuantizedWeight] = {}
        module_metadata: dict[str, dict[str, float | int]] = {}

        for name in context.target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            module_state = state.modules.get(name)
            if module_state is None:
                raise ValueError(f"missing GPTQ calibration state for {name}")

            try:
                weights[name], module_metadata[name] = quantize_weight_gptq(
                    module.weight,
                    module_state.activations,
                    context.config.group_size,
                    block_size=state.options.block_size,
                    damp_percent=state.options.damp_percent,
                    symmetric=context.config.symmetric,
                    scale_dtype=scale_dtype,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"GPTQ failed for layer {context.layer_index} module {name} "
                    f"with {module_state.activation_tokens} calibration tokens: {error}"
                ) from error

        return LayerQuantizationResult(
            weights=weights,
            metadata={
                "gptq": {
                    "block_size": state.options.block_size,
                    "damp_percent": state.options.damp_percent,
                    "modules": module_metadata,
                }
            },
        )
