#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import json
import os
import yaml


@dataclass
class IHASchedule:
    num_query_heads_per_layer: List[int]
    num_key_value_heads_per_layer: List[int]
    qk_head_dim_per_layer: List[int]
    v_head_dim_per_layer: List[int]
    intermediate_size_per_layer: List[int]

    def validate(self, num_hidden_layers: int) -> None:
        fields = {
            "num_query_heads_per_layer": self.num_query_heads_per_layer,
            "num_key_value_heads_per_layer": self.num_key_value_heads_per_layer,
            "qk_head_dim_per_layer": self.qk_head_dim_per_layer,
            "v_head_dim_per_layer": self.v_head_dim_per_layer,
            "intermediate_size_per_layer": self.intermediate_size_per_layer,
        }
        for name, values in fields.items():
            if len(values) != num_hidden_layers:
                raise ValueError(f"{name} length must match num_hidden_layers")

        for i, (nq, nkv, d_qk, d_v, mlp) in enumerate(
            zip(
                self.num_query_heads_per_layer,
                self.num_key_value_heads_per_layer,
                self.qk_head_dim_per_layer,
                self.v_head_dim_per_layer,
                self.intermediate_size_per_layer,
            )
        ):
            if nq <= 0 or nkv <= 0:
                raise ValueError(f"Layer {i}: n_q and n_kv must be positive")
            if nq % nkv != 0:
                raise ValueError(f"Layer {i}: must satisfy n_q % n_kv == 0 (got n_q={nq}, n_kv={nkv})")
            if d_qk <= 0 or d_v <= 0:
                raise ValueError(f"Layer {i}: d_qk and d_v must be positive")
            if d_qk % 2 != 0:
                raise ValueError(f"Layer {i}: d_qk must be even for RoPE")
            if mlp <= 0:
                raise ValueError(f"Layer {i}: intermediate_size must be positive")


def build_default_schedule(
    *,
    num_hidden_layers: int,
    base_num_query_heads: int,
    base_num_kv_heads: int,
    base_qk_head_dim: int,
    base_v_head_dim: int,
    base_intermediate_size: int,
) -> IHASchedule:
    kv_heads = []
    for i in range(num_hidden_layers):
        if i < num_hidden_layers * 2 // 3:
            kv_heads.append(base_num_kv_heads)
        elif i < num_hidden_layers * 5 // 6:
            kv_heads.append(max(1, base_num_kv_heads // 2))
        else:
            kv_heads.append(max(1, base_num_kv_heads // 4))

    nq = [base_num_query_heads] * num_hidden_layers
    d_qk = [base_qk_head_dim] * num_hidden_layers
    d_v = [base_v_head_dim] * num_hidden_layers

    mlp_sizes = []
    for i in range(num_hidden_layers):
        if i < num_hidden_layers // 3:
            mlp_sizes.append(int(base_intermediate_size * 2 / 3))
        else:
            mlp_sizes.append(base_intermediate_size)

    schedule = IHASchedule(
        num_query_heads_per_layer=nq,
        num_key_value_heads_per_layer=kv_heads,
        qk_head_dim_per_layer=d_qk,
        v_head_dim_per_layer=d_v,
        intermediate_size_per_layer=mlp_sizes,
    )
    schedule.validate(num_hidden_layers)
    return schedule


def load_schedule(path: str) -> IHASchedule:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Schedule file must be a mapping/object")

    # Backward compatible: infer missing fields from legacy schedule.
    kv = raw.get("num_key_value_heads_per_layer")
    mlp = raw.get("intermediate_size_per_layer")
    nq = raw.get("num_query_heads_per_layer")
    d_qk = raw.get("qk_head_dim_per_layer")
    d_v = raw.get("v_head_dim_per_layer")

    if not isinstance(kv, list) or not isinstance(mlp, list):
        raise ValueError("Schedule file must include list fields for KV heads and MLP sizes")

    if nq is None:
        # temporary placeholder; caller should overwrite when using defaults from base config.
        nq = [0] * len(kv)
    if d_qk is None:
        d_qk = [0] * len(kv)
    if d_v is None:
        d_v = [0] * len(kv)

    return IHASchedule(
        num_query_heads_per_layer=[int(x) for x in nq],
        num_key_value_heads_per_layer=[int(x) for x in kv],
        qk_head_dim_per_layer=[int(x) for x in d_qk],
        v_head_dim_per_layer=[int(x) for x in d_v],
        intermediate_size_per_layer=[int(x) for x in mlp],
    )


def save_schedule(path: str, schedule: IHASchedule) -> None:
    payload = {
        "num_query_heads_per_layer": schedule.num_query_heads_per_layer,
        "num_key_value_heads_per_layer": schedule.num_key_value_heads_per_layer,
        "qk_head_dim_per_layer": schedule.qk_head_dim_per_layer,
        "v_head_dim_per_layer": schedule.v_head_dim_per_layer,
        "intermediate_size_per_layer": schedule.intermediate_size_per_layer,
    }
    ext = os.path.splitext(path)[1].lower()
    with open(path, "w", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            yaml.safe_dump(payload, f, sort_keys=False)
        else:
            json.dump(payload, f, indent=2)
            f.write("\n")
