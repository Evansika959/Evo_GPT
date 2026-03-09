#!/usr/bin/env python3
"""IHA schedule utilities for per-layer KV heads and MLP width."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class IHASchedule:
    num_key_value_heads_per_layer: List[int]
    intermediate_size_per_layer: List[int]

    def validate(self, num_hidden_layers: int, num_attention_heads: int, head_dim: int) -> None:
        if len(self.num_key_value_heads_per_layer) != num_hidden_layers:
            raise ValueError("num_key_value_heads_per_layer length must match num_hidden_layers")
        if len(self.intermediate_size_per_layer) != num_hidden_layers:
            raise ValueError("intermediate_size_per_layer length must match num_hidden_layers")

        for idx, kv in enumerate(self.num_key_value_heads_per_layer):
            if kv <= 0:
                raise ValueError(f"num_key_value_heads_per_layer[{idx}] must be positive")
            if num_attention_heads % kv != 0:
                raise ValueError(
                    f"num_attention_heads ({num_attention_heads}) must be divisible by kv heads ({kv})"
                )

        for idx, mlp in enumerate(self.intermediate_size_per_layer):
            if mlp <= 0:
                raise ValueError(f"intermediate_size_per_layer[{idx}] must be positive")

        if head_dim <= 0:
            raise ValueError("head_dim must be positive")


def build_default_schedule(
    num_hidden_layers: int,
    num_attention_heads: int,
    base_num_kv_heads: int,
    base_intermediate_size: int,
) -> IHASchedule:
    """A simple default schedule: later layers use fewer KV heads; early MLP is narrower."""
    kv_heads = []
    for i in range(num_hidden_layers):
        if i < num_hidden_layers * 2 // 3:
            kv_heads.append(base_num_kv_heads)
        elif i < num_hidden_layers * 5 // 6:
            kv_heads.append(max(1, base_num_kv_heads // 2))
        else:
            kv_heads.append(max(1, base_num_kv_heads // 4))

    mlp_sizes = []
    for i in range(num_hidden_layers):
        if i < num_hidden_layers // 3:
            mlp_sizes.append(int(base_intermediate_size * 2 / 3))
        else:
            mlp_sizes.append(base_intermediate_size)

    return IHASchedule(kv_heads, mlp_sizes)


def load_schedule(path: str) -> IHASchedule:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Schedule JSON must be an object")

    kv = raw.get("num_key_value_heads_per_layer")
    mlp = raw.get("intermediate_size_per_layer")
    if not isinstance(kv, list) or not isinstance(mlp, list):
        raise ValueError("Schedule JSON must include list fields for KV heads and MLP sizes")

    return IHASchedule([int(x) for x in kv], [int(x) for x in mlp])


def save_schedule(path: str, schedule: IHASchedule) -> None:
    payload = {
        "num_key_value_heads_per_layer": schedule.num_key_value_heads_per_layer,
        "intermediate_size_per_layer": schedule.intermediate_size_per_layer,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
