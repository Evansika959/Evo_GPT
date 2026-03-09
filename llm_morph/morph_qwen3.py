#!/usr/bin/env python3
"""Morph Qwen3 checkpoints into per-layer KV/MLP schedules (IHA-style)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import List, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from configuration_qwen3_iha import Qwen3IHAConfig
from modeling_qwen3_iha import Qwen3IHAForCausalLM
from iha_config import IHASchedule, build_default_schedule, load_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Morph Qwen3 into IHA-style per-layer KV/MLP schedules")
    parser.add_argument("--model_id", type=str, required=True, help="HF model id or local path")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--schedule", type=str, default=None, help="Path to schedule JSON")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _dtype_from_arg(dtype: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]


def _kv_cache_bytes_per_token(num_kv_heads_per_layer: List[int], head_dim: int, dtype: torch.dtype) -> int:
    bytes_per = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total_kv = sum(num_kv_heads_per_layer)
    return 2 * total_kv * head_dim * bytes_per


def _mean_pool_kv(weight: torch.Tensor, old_kv: int, new_kv: int, head_dim: int) -> torch.Tensor:
    hidden = weight.shape[1]
    reshaped = weight.view(old_kv, head_dim, hidden)
    group = old_kv // new_kv
    pooled = reshaped.view(new_kv, group, head_dim, hidden).mean(dim=1)
    return pooled.view(new_kv * head_dim, hidden)


def _mean_pool_bias(bias: torch.Tensor, old_kv: int, new_kv: int, head_dim: int) -> torch.Tensor:
    reshaped = bias.view(old_kv, head_dim)
    group = old_kv // new_kv
    pooled = reshaped.view(new_kv, group, head_dim).mean(dim=1)
    return pooled.view(new_kv * head_dim)


def _select_topk_indices(weight: torch.Tensor, k: int) -> torch.Tensor:
    # weight: [out, in], select top-k rows by L2 norm
    norms = weight.norm(p=2, dim=1)
    return torch.topk(norms, k=k, largest=True).indices


def _resize_mlp(
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    new_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    old_size = gate_w.shape[0]
    if new_size == old_size:
        return gate_w, up_w, down_w

    if new_size < old_size:
        idx = _select_topk_indices(gate_w, new_size)
        gate_new = gate_w[idx]
        up_new = up_w[idx]
        down_new = down_w[:, idx]
        return gate_new, up_new, down_new

    # expand: copy old, init new rows/cols
    device = gate_w.device
    dtype = gate_w.dtype
    gate_new = torch.zeros((new_size, gate_w.shape[1]), device=device, dtype=dtype)
    up_new = torch.zeros((new_size, up_w.shape[1]), device=device, dtype=dtype)
    down_new = torch.zeros((down_w.shape[0], new_size), device=device, dtype=dtype)

    gate_new[:old_size] = gate_w
    up_new[:old_size] = up_w
    down_new[:, :old_size] = down_w

    torch.nn.init.normal_(gate_new[old_size:], mean=0.0, std=0.02)
    torch.nn.init.normal_(up_new[old_size:], mean=0.0, std=0.02)
    torch.nn.init.normal_(down_new[:, old_size:], mean=0.0, std=0.02)
    return gate_new, up_new, down_new


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=_dtype_from_arg(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)

    num_layers = base_config.num_hidden_layers
    head_dim = base_config.hidden_size // base_config.num_attention_heads

    if args.schedule:
        schedule = load_schedule(args.schedule)
    else:
        schedule = build_default_schedule(
            num_hidden_layers=num_layers,
            num_attention_heads=base_config.num_attention_heads,
            base_num_kv_heads=base_config.num_key_value_heads,
            base_intermediate_size=base_config.intermediate_size,
        )

    schedule.validate(num_layers, base_config.num_attention_heads, head_dim)

    new_config = Qwen3IHAConfig(
        **base_config.to_dict(),
        num_key_value_heads_per_layer=schedule.num_key_value_heads_per_layer,
        intermediate_size_per_layer=schedule.intermediate_size_per_layer,
        base_num_key_value_heads=base_config.num_key_value_heads,
        base_intermediate_size=base_config.intermediate_size,
    )

    new_model = Qwen3IHAForCausalLM(new_config)

    new_sd = new_model.state_dict()
    base_sd = base_model.state_dict()

    # copy matching tensors first
    for name, tensor in base_sd.items():
        if name in new_sd and new_sd[name].shape == tensor.shape:
            new_sd[name] = tensor

    # morph per-layer KV heads and MLP sizes
    for i in range(num_layers):
        base_prefix = f"model.layers.{i}."
        new_kv = schedule.num_key_value_heads_per_layer[i]
        old_kv = base_config.num_key_value_heads

        head_dim = base_config.hidden_size // base_config.num_attention_heads

        k_w = base_sd[base_prefix + "self_attn.k_proj.weight"]
        v_w = base_sd[base_prefix + "self_attn.v_proj.weight"]
        new_sd[base_prefix + "self_attn.k_proj.weight"] = _mean_pool_kv(k_w, old_kv, new_kv, head_dim)
        new_sd[base_prefix + "self_attn.v_proj.weight"] = _mean_pool_kv(v_w, old_kv, new_kv, head_dim)

        if base_prefix + "self_attn.k_proj.bias" in base_sd:
            k_b = base_sd[base_prefix + "self_attn.k_proj.bias"]
            v_b = base_sd[base_prefix + "self_attn.v_proj.bias"]
            new_sd[base_prefix + "self_attn.k_proj.bias"] = _mean_pool_bias(k_b, old_kv, new_kv, head_dim)
            new_sd[base_prefix + "self_attn.v_proj.bias"] = _mean_pool_bias(v_b, old_kv, new_kv, head_dim)

        gate_w = base_sd[base_prefix + "mlp.gate_proj.weight"]
        up_w = base_sd[base_prefix + "mlp.up_proj.weight"]
        down_w = base_sd[base_prefix + "mlp.down_proj.weight"]
        new_mlp = schedule.intermediate_size_per_layer[i]
        gate_new, up_new, down_new = _resize_mlp(gate_w, up_w, down_w, new_mlp)
        new_sd[base_prefix + "mlp.gate_proj.weight"] = gate_new
        new_sd[base_prefix + "mlp.up_proj.weight"] = up_new
        new_sd[base_prefix + "mlp.down_proj.weight"] = down_new

        if base_prefix + "mlp.gate_proj.bias" in base_sd:
            gate_b = base_sd[base_prefix + "mlp.gate_proj.bias"]
            up_b = base_sd[base_prefix + "mlp.up_proj.bias"]
            down_b = base_sd[base_prefix + "mlp.down_proj.bias"]
            if new_mlp < gate_b.numel():
                idx = _select_topk_indices(gate_w, new_mlp)
                new_sd[base_prefix + "mlp.gate_proj.bias"] = gate_b[idx]
                new_sd[base_prefix + "mlp.up_proj.bias"] = up_b[idx]
                new_sd[base_prefix + "mlp.down_proj.bias"] = down_b
            else:
                gate_b_new = torch.zeros((new_mlp,), device=gate_b.device, dtype=gate_b.dtype)
                up_b_new = torch.zeros((new_mlp,), device=up_b.device, dtype=up_b.dtype)
                gate_b_new[: gate_b.numel()] = gate_b
                up_b_new[: up_b.numel()] = up_b
                new_sd[base_prefix + "mlp.gate_proj.bias"] = gate_b_new
                new_sd[base_prefix + "mlp.up_proj.bias"] = up_b_new
                new_sd[base_prefix + "mlp.down_proj.bias"] = down_b

    new_model.load_state_dict(new_sd, strict=False)

    before_params = sum(p.numel() for p in base_model.parameters())
    after_params = sum(p.numel() for p in new_model.parameters())

    dtype = _dtype_from_arg(args.dtype)
    kv_bytes_per_token = _kv_cache_bytes_per_token(schedule.num_key_value_heads_per_layer, head_dim, dtype)

    print(f"params before: {before_params:,}")
    print(f"params after:  {after_params:,}")
    print(f"KV cache bytes/token: {kv_bytes_per_token:,}")
    print(f"KV cache @4096: {kv_bytes_per_token * 4096 / (1024**2):.2f} MiB")
    print(f"KV cache @8192: {kv_bytes_per_token * 8192 / (1024**2):.2f} MiB")

    new_model.save_pretrained(args.out_dir, safe_serialization=args.save_safetensors)
    new_config.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # Ensure custom code is bundled for trust_remote_code loading.
    for filename in ("configuration_qwen3_iha.py", "modeling_qwen3_iha.py"):
        src = os.path.join(os.path.dirname(__file__), filename)
        dst = os.path.join(args.out_dir, filename)
        shutil.copy2(src, dst)

    schedule_path = os.path.join(args.out_dir, "iha_schedule.json")
    with open(schedule_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_key_value_heads_per_layer": schedule.num_key_value_heads_per_layer,
                "intermediate_size_per_layer": schedule.intermediate_size_per_layer,
            },
            f,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
