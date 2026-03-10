#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Optional
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class Qwen3IHAConfig(Qwen3Config):
    model_type = "qwen3_iha"

    def __init__(
        self,
        num_query_heads_per_layer: Optional[List[int]] = None,
        num_key_value_heads_per_layer: Optional[List[int]] = None,
        qk_head_dim_per_layer: Optional[List[int]] = None,
        v_head_dim_per_layer: Optional[List[int]] = None,
        intermediate_size_per_layer: Optional[List[int]] = None,
        base_num_query_heads: Optional[int] = None,
        base_num_key_value_heads: Optional[int] = None,
        base_qk_head_dim: Optional[int] = None,
        base_v_head_dim: Optional[int] = None,
        base_intermediate_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        inferred_head_dim = getattr(self, "head_dim", self.hidden_size // self.num_attention_heads)

        self.num_query_heads_per_layer = num_query_heads_per_layer
        self.num_key_value_heads_per_layer = num_key_value_heads_per_layer
        self.qk_head_dim_per_layer = qk_head_dim_per_layer
        self.v_head_dim_per_layer = v_head_dim_per_layer
        self.intermediate_size_per_layer = intermediate_size_per_layer

        self.base_num_query_heads = base_num_query_heads if base_num_query_heads is not None else self.num_attention_heads
        self.base_num_key_value_heads = (
            base_num_key_value_heads if base_num_key_value_heads is not None else self.num_key_value_heads
        )
        self.base_qk_head_dim = base_qk_head_dim if base_qk_head_dim is not None else inferred_head_dim
        self.base_v_head_dim = base_v_head_dim if base_v_head_dim is not None else inferred_head_dim
        self.base_intermediate_size = (
            base_intermediate_size if base_intermediate_size is not None else self.intermediate_size
        )

        self.auto_map = {
            "AutoConfig": "configuration_qwen3_iha.Qwen3IHAConfig",
            "AutoModelForCausalLM": "modeling_qwen3_iha.Qwen3IHAForCausalLM",
        }

    def to_dict(self):
        output = super().to_dict()
        output["num_query_heads_per_layer"] = self.num_query_heads_per_layer
        output["num_key_value_heads_per_layer"] = self.num_key_value_heads_per_layer
        output["qk_head_dim_per_layer"] = self.qk_head_dim_per_layer
        output["v_head_dim_per_layer"] = self.v_head_dim_per_layer
        output["intermediate_size_per_layer"] = self.intermediate_size_per_layer

        output["base_num_query_heads"] = self.base_num_query_heads
        output["base_num_key_value_heads"] = self.base_num_key_value_heads
        output["base_qk_head_dim"] = self.base_qk_head_dim
        output["base_v_head_dim"] = self.base_v_head_dim
        output["base_intermediate_size"] = self.base_intermediate_size
        return output
