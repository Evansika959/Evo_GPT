#!/usr/bin/env python3
"""Qwen3 config with per-layer KV and MLP schedules."""
from __future__ import annotations

from typing import List, Optional
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class Qwen3IHAConfig(Qwen3Config):
    model_type = "qwen3_iha"

    def __init__(
        self,
        num_key_value_heads_per_layer: Optional[List[int]] = None,
        intermediate_size_per_layer: Optional[List[int]] = None,
        base_num_key_value_heads: Optional[int] = None,
        base_intermediate_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_key_value_heads_per_layer = num_key_value_heads_per_layer
        self.intermediate_size_per_layer = intermediate_size_per_layer
        self.base_num_key_value_heads = (
            base_num_key_value_heads if base_num_key_value_heads is not None else self.num_key_value_heads
        )
        self.base_intermediate_size = (
            base_intermediate_size if base_intermediate_size is not None else self.intermediate_size
        )

        self.auto_map = {
            "AutoConfig": "configuration_qwen3_iha.Qwen3IHAConfig",
            "AutoModelForCausalLM": "modeling_qwen3_iha.Qwen3IHAForCausalLM",
        }

    def to_dict(self):
        output = super().to_dict()
        output["num_key_value_heads_per_layer"] = self.num_key_value_heads_per_layer
        output["intermediate_size_per_layer"] = self.intermediate_size_per_layer
        output["base_num_key_value_heads"] = self.base_num_key_value_heads
        output["base_intermediate_size"] = self.base_intermediate_size
        return output
