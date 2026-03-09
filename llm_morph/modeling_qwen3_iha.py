#!/usr/bin/env python3
"""Qwen3 model variant with per-layer KV heads and MLP widths."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from configuration_qwen3_iha import Qwen3IHAConfig


def _make_linear_like(module: nn.Linear, out_features: int, in_features: int) -> nn.Linear:
    new_lin = nn.Linear(in_features, out_features, bias=module.bias is not None)
    new_lin = new_lin.to(device=module.weight.device, dtype=module.weight.dtype)
    return new_lin


def apply_iha_overrides(model: Qwen3ForCausalLM, config: Qwen3IHAConfig) -> None:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Unexpected Qwen3 model structure; expected model.model.layers")

    layers = model.model.layers
    num_layers = len(layers)

    kv_list = config.num_key_value_heads_per_layer or [config.num_key_value_heads] * num_layers
    mlp_list = config.intermediate_size_per_layer or [config.intermediate_size] * num_layers

    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        mlp = layer.mlp

        new_kv = int(kv_list[idx])
        new_mlp = int(mlp_list[idx])

        head_dim = getattr(attn, "head_dim", config.hidden_size // config.num_attention_heads)

        attn.num_key_value_heads = new_kv
        attn.num_key_value_groups = config.num_attention_heads // new_kv

        k_out = new_kv * head_dim
        v_out = new_kv * head_dim
        attn.k_proj = _make_linear_like(attn.k_proj, out_features=k_out, in_features=attn.k_proj.in_features)
        attn.v_proj = _make_linear_like(attn.v_proj, out_features=v_out, in_features=attn.v_proj.in_features)

        mlp.gate_proj = _make_linear_like(mlp.gate_proj, out_features=new_mlp, in_features=mlp.gate_proj.in_features)
        mlp.up_proj = _make_linear_like(mlp.up_proj, out_features=new_mlp, in_features=mlp.up_proj.in_features)
        mlp.down_proj = _make_linear_like(mlp.down_proj, out_features=mlp.down_proj.out_features, in_features=new_mlp)


class Qwen3IHAForCausalLM(Qwen3ForCausalLM):
    config_class = Qwen3IHAConfig

    def __init__(self, config: Qwen3IHAConfig) -> None:
        super().__init__(config)
        apply_iha_overrides(self, config)


__all__ = ["Qwen3IHAForCausalLM", "apply_iha_overrides"]
