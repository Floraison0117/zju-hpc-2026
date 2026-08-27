from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from hpc101_infer.layers.attention import AttentionLayer, SlidingAttentionLayer
from hpc101_infer.layers.linear import BF16LinearFactory, LinearFactory
from hpc101_infer.layers.norm import RMSNorm
from hpc101_infer.models.config import Gemma4TextConfig
from hpc101_infer.runtime.batch import Batch
from hpc101_infer.runtime.kv_cache import KVCache


class MLP(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        linear_factory: LinearFactory,
        module_prefix: str,
    ):
        super().__init__()
        self.gate_proj = linear_factory.create(
            f"{module_prefix}.gate_proj",
            config.hidden_size,
            config.intermediate_size,
            False,
        )
        self.up_proj = linear_factory.create(
            f"{module_prefix}.up_proj",
            config.hidden_size,
            config.intermediate_size,
            False,
        )
        self.down_proj = linear_factory.create(
            f"{module_prefix}.down_proj",
            config.intermediate_size,
            config.hidden_size,
            False,
        )
        self.persistent_gate_up = None
        if config.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(f"unsupported activation: {config.hidden_activation}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.persistent_gate_up is not None:
            activated = self.persistent_gate_up(hidden_states)
        else:
            activated = F.gelu(self.gate_proj(hidden_states), approximate="tanh") * self.up_proj(hidden_states)
        return self.down_proj(activated)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        linear_factory: LinearFactory,
        attention_backend: str = "eager",
        kv_cache_backend: str = "ring_roll",
    ):
        super().__init__()
        module_prefix = f"layers.{layer_idx}"
        self.layer_idx = layer_idx
        self.attn_type = config.layer_types[layer_idx]
        self.attention_backend = attention_backend
        self.kv_cache_backend = kv_cache_backend
        self.kv_cache_backend = kv_cache_backend
        match self.attn_type:
            case "full_attention":
                self.self_attn = AttentionLayer(
                    config.num_attention_heads,
                    config.num_global_key_value_heads,
                    config.global_head_dim,
                    config.hidden_size,
                    config.get_rope_config(self.attn_type),
                    config.rms_norm_eps,
                    config.max_position_embeddings,
                    config.attention_k_eq_v,
                    linear_factory,
                    f"{module_prefix}.self_attn",
                    attention_backend,
                    kv_cache_backend,
                )
            case "sliding_attention":
                self.self_attn = SlidingAttentionLayer(
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.hidden_size,
                    config.get_rope_config(self.attn_type),
                    config.rms_norm_eps,
                    config.max_position_embeddings,
                    config.sliding_window,
                    linear_factory,
                    f"{module_prefix}.self_attn",
                    attention_backend,
                    kv_cache_backend,
                )
            case _:
                raise ValueError(f"Unsupported attention type: {self.attn_type}")

        self.mlp = MLP(config, linear_factory, f"{module_prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
        max_seq_len: int,
        kv_cache: KVCache | None,
        slot_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
            sequence_lengths,
            max_seq_len,
            self.layer_idx,
            kv_cache,
            slot_ids=slot_ids,
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return (
            residual + self.post_feedforward_layernorm(hidden_states)
        ) * self.layer_scalar


class Gemma4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        linear_factory: LinearFactory | None = None,
        attention_backend: str = "eager",
        kv_cache_backend: str = "ring_roll",
    ):
        super().__init__()
        self.config = config
        linear_factory = linear_factory or BF16LinearFactory()
        self.attention_backend = attention_backend
        self.kv_cache_backend = kv_cache_backend
        self.kv_cache_backend = kv_cache_backend
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, idx, linear_factory, self.attention_backend, self.kv_cache_backend)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.embed_scale = math.sqrt(config.hidden_size)
        self.weight_offloader = None

    def forward(
        self,
        model_input: Batch | torch.Tensor,
        kv_cache: KVCache | None = None,
        logits_to_keep: int = 0,
        last_valid_only: bool = False,
    ) -> torch.Tensor:
        if isinstance(model_input, torch.Tensor):
            input_ids = model_input
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            sequence_lengths = torch.full(
                (input_ids.shape[0],),
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )
            max_seq_len = input_ids.shape[1]
        else:
            input_ids = model_input.input_ids
            position_ids = model_input.positions
            sequence_lengths = model_input.sequence_lengths
            max_seq_len = model_input.curr_max_seq_len
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale
        slot_ids = getattr(model_input, 'slot_ids', None)
        if slot_ids is not None and slot_ids.numel() == 0:
            slot_ids = None
        def run_layer(layer: nn.Module, states: torch.Tensor) -> torch.Tensor:
            return layer(
                states, position_ids, sequence_lengths, max_seq_len, kv_cache,
                slot_ids=slot_ids,
            )

        if self.weight_offloader is None:
            for layer in self.layers:
                hidden_states = run_layer(layer, hidden_states)
        else:
            hidden_states = self.weight_offloader.run(hidden_states, run_layer)
        if kv_cache is not None:
            kv_cache.commit(sequence_lengths, slot_ids=slot_ids)
        hidden_states = self.norm(hidden_states)
        if last_valid_only:
            assert not logits_to_keep, "last_valid_only and logits_to_keep are mutually exclusive"
            batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            last_indices = sequence_lengths - 1
            hidden_states = hidden_states[batch_indices, last_indices].unsqueeze(1)
        elif logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:]
        logits = F.linear(hidden_states, self.embed_tokens.weight)
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits
