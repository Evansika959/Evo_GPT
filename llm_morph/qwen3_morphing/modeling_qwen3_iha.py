#!/usr/bin/env python3
from __future__ import annotations

from types import MethodType

import torch
import torch.nn as nn

from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3ForCausalLM,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

try:
    from .configuration_qwen3_iha import Qwen3IHAConfig
except ImportError:
    from configuration_qwen3_iha import Qwen3IHAConfig


def _make_linear_like(module: nn.Linear, out_features: int, in_features: int) -> nn.Linear:
    new_lin = nn.Linear(in_features, out_features, bias=module.bias is not None)
    new_lin = new_lin.to(device=module.weight.device, dtype=module.weight.dtype)
    return new_lin


def _iha_attn_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None, **kwargs):
    input_shape = hidden_states.shape[:-1]

    q_shape = (*input_shape, self.iha_num_query_heads, self.iha_qk_head_dim)
    kvk_shape = (*input_shape, self.iha_num_key_value_heads, self.iha_qk_head_dim)
    kvv_shape = (*input_shape, self.iha_num_key_value_heads, self.iha_v_head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(q_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(kvk_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(kvv_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,
        eager_attention_forward,
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, self.iha_num_query_heads * self.iha_v_head_dim).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def apply_iha_overrides(model: Qwen3ForCausalLM, config: Qwen3IHAConfig) -> None:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Unexpected Qwen3 model structure; expected model.model.layers")

    layers = model.model.layers
    num_layers = len(layers)

    nq_list = config.num_query_heads_per_layer or [config.base_num_query_heads] * num_layers
    nkv_list = config.num_key_value_heads_per_layer or [config.base_num_key_value_heads] * num_layers
    d_qk_list = config.qk_head_dim_per_layer or [config.base_qk_head_dim] * num_layers
    d_v_list = config.v_head_dim_per_layer or [config.base_v_head_dim] * num_layers
    mlp_list = config.intermediate_size_per_layer or [config.base_intermediate_size] * num_layers

    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        mlp = layer.mlp

        n_q = int(nq_list[idx])
        n_kv = int(nkv_list[idx])
        d_qk = int(d_qk_list[idx])
        d_v = int(d_v_list[idx])
        new_mlp = int(mlp_list[idx])

        attn.iha_num_query_heads = n_q
        attn.iha_num_key_value_heads = n_kv
        attn.iha_qk_head_dim = d_qk
        attn.iha_v_head_dim = d_v

        attn.head_dim = d_qk
        attn.num_key_value_groups = n_q // n_kv
        attn.scaling = d_qk**-0.5

        attn.q_proj = _make_linear_like(attn.q_proj, out_features=n_q * d_qk, in_features=attn.q_proj.in_features)
        attn.k_proj = _make_linear_like(attn.k_proj, out_features=n_kv * d_qk, in_features=attn.k_proj.in_features)
        attn.v_proj = _make_linear_like(attn.v_proj, out_features=n_kv * d_v, in_features=attn.v_proj.in_features)
        attn.o_proj = _make_linear_like(attn.o_proj, out_features=attn.o_proj.out_features, in_features=n_q * d_v)

        attn.q_norm = Qwen3RMSNorm(d_qk, eps=config.rms_norm_eps).to(device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)
        attn.k_norm = Qwen3RMSNorm(d_qk, eps=config.rms_norm_eps).to(device=attn.k_proj.weight.device, dtype=attn.k_proj.weight.dtype)

        attn.forward = MethodType(_iha_attn_forward, attn)

        mlp.gate_proj = _make_linear_like(mlp.gate_proj, out_features=new_mlp, in_features=mlp.gate_proj.in_features)
        mlp.up_proj = _make_linear_like(mlp.up_proj, out_features=new_mlp, in_features=mlp.up_proj.in_features)
        mlp.down_proj = _make_linear_like(mlp.down_proj, out_features=mlp.down_proj.out_features, in_features=new_mlp)


class Qwen3IHAForCausalLM(Qwen3ForCausalLM):
    config_class = Qwen3IHAConfig

    def __init__(self, config: Qwen3IHAConfig) -> None:
        super().__init__(config)
        apply_iha_overrides(self, config)


__all__ = ["Qwen3IHAForCausalLM", "apply_iha_overrides"]
