#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .configuration_qwen3_iha import Qwen3IHAConfig
from .iha_config import IHASchedule, build_default_schedule, load_schedule
from .modeling_qwen3_iha import Qwen3IHAForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Morph Qwen3 into IHA-style per-layer attention/MLP schedules")
    parser.add_argument("--model_id", type=str, required=True, help="HF model id or local path")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--schedule", type=str, default=None, help="Path to schedule JSON")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _dtype_from_arg(dtype: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]


def _kv_cache_bytes_per_token(schedule: IHASchedule, dtype: torch.dtype) -> int:
    bytes_per = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    total = 0
    for nkv, d_qk, d_v in zip(
        schedule.num_key_value_heads_per_layer,
        schedule.qk_head_dim_per_layer,
        schedule.v_head_dim_per_layer,
    ):
        total += int(nkv) * (int(d_qk) + int(d_v))
    return total * bytes_per


def _select_topk_indices_by_norm(x: torch.Tensor, dim: int, k: int) -> torch.Tensor:
    norms = x.float().pow(2).sum(dim=tuple(i for i in range(x.ndim) if i != dim)).sqrt()
    return torch.topk(norms, k=k, largest=True).indices


def _resize_heads(x: torch.Tensor, new_heads: int, head_axis: int) -> torch.Tensor:
    old_heads = x.shape[head_axis]
    if new_heads == old_heads:
        return x

    if new_heads < old_heads:
        if old_heads % new_heads == 0:
            group = old_heads // new_heads
            new_shape = list(x.shape)
            new_shape[head_axis] = new_heads
            new_shape.insert(head_axis + 1, group)
            xg = x.reshape(new_shape)
            return xg.mean(dim=head_axis + 1)

        idx = _select_topk_indices_by_norm(x, dim=head_axis, k=new_heads).to(x.device)
        return x.index_select(head_axis, idx)

    out_shape = list(x.shape)
    out_shape[head_axis] = new_heads
    out = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
    slicer = [slice(None)] * x.ndim
    slicer[head_axis] = slice(0, old_heads)
    out[tuple(slicer)] = x
    tail_shape = list(out_shape)
    tail_shape[head_axis] = new_heads - old_heads
    tail = torch.empty(tail_shape, device=x.device, dtype=x.dtype)
    torch.nn.init.normal_(tail, mean=0.0, std=0.02)
    slicer[head_axis] = slice(old_heads, new_heads)
    out[tuple(slicer)] = tail
    return out


def _resize_dim(x: torch.Tensor, new_dim: int, dim_axis: int) -> torch.Tensor:
    old_dim = x.shape[dim_axis]
    if new_dim == old_dim:
        return x

    if new_dim < old_dim:
        slicer = [slice(None)] * x.ndim
        slicer[dim_axis] = slice(0, new_dim)
        return x[tuple(slicer)].contiguous()

    out_shape = list(x.shape)
    out_shape[dim_axis] = new_dim
    out = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
    slicer = [slice(None)] * x.ndim
    slicer[dim_axis] = slice(0, old_dim)
    out[tuple(slicer)] = x
    tail_shape = list(out_shape)
    tail_shape[dim_axis] = new_dim - old_dim
    tail = torch.empty(tail_shape, device=x.device, dtype=x.dtype)
    torch.nn.init.normal_(tail, mean=0.0, std=0.02)
    slicer[dim_axis] = slice(old_dim, new_dim)
    out[tuple(slicer)] = tail
    return out


def _resize_qkv_weight(weight: torch.Tensor, old_heads: int, old_dim: int, new_heads: int, new_dim: int) -> torch.Tensor:
    hidden = weight.shape[1]
    x = weight.view(old_heads, old_dim, hidden)
    x = _resize_heads(x, new_heads, head_axis=0)
    x = _resize_dim(x, new_dim, dim_axis=1)
    return x.reshape(new_heads * new_dim, hidden)


def _resize_qkv_bias(bias: torch.Tensor, old_heads: int, old_dim: int, new_heads: int, new_dim: int) -> torch.Tensor:
    x = bias.view(old_heads, old_dim)
    x = _resize_heads(x, new_heads, head_axis=0)
    x = _resize_dim(x, new_dim, dim_axis=1)
    return x.reshape(new_heads * new_dim)


def _resize_o_proj(weight: torch.Tensor, old_q: int, old_v: int, new_q: int, new_v: int) -> torch.Tensor:
    hidden = weight.shape[0]
    x = weight.view(hidden, old_q, old_v)
    x = _resize_heads(x, new_q, head_axis=1)
    x = _resize_dim(x, new_v, dim_axis=2)
    return x.reshape(hidden, new_q * new_v)


def _select_topk_indices(weight: torch.Tensor, k: int) -> torch.Tensor:
    norms = weight.norm(p=2, dim=1)
    return torch.topk(norms, k=k, largest=True).indices


def _resize_mlp(gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor, new_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    old_size = gate_w.shape[0]
    if new_size == old_size:
        return gate_w, up_w, down_w

    if new_size < old_size:
        idx = _select_topk_indices(gate_w, new_size)
        return gate_w[idx], up_w[idx], down_w[:, idx]

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


def _fill_legacy_schedule(schedule: IHASchedule, base_config) -> IHASchedule:
    base_q = int(base_config.num_attention_heads)
    base_dim = int(getattr(base_config, "head_dim", base_config.hidden_size // base_config.num_attention_heads))
    if any(x == 0 for x in schedule.num_query_heads_per_layer):
        schedule.num_query_heads_per_layer = [base_q] * len(schedule.num_key_value_heads_per_layer)
    if any(x == 0 for x in schedule.qk_head_dim_per_layer):
        schedule.qk_head_dim_per_layer = [base_dim] * len(schedule.num_key_value_heads_per_layer)
    if any(x == 0 for x in schedule.v_head_dim_per_layer):
        schedule.v_head_dim_per_layer = [base_dim] * len(schedule.num_key_value_heads_per_layer)
    return schedule


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=_dtype_from_arg(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    
    print(f"base model config: {base_config}")

    num_layers = base_config.num_hidden_layers
    base_nq = int(base_config.num_attention_heads)
    base_nkv = int(base_config.num_key_value_heads)
    base_qk = int(getattr(base_config, "head_dim", base_config.hidden_size // base_config.num_attention_heads))
    base_v = base_qk

    if args.schedule:
        schedule = _fill_legacy_schedule(load_schedule(args.schedule), base_config)
    else:
        schedule = build_default_schedule(
            num_hidden_layers=num_layers,
            base_num_query_heads=base_nq,
            base_num_kv_heads=base_nkv,
            base_qk_head_dim=base_qk,
            base_v_head_dim=base_v,
            base_intermediate_size=base_config.intermediate_size,
        )

    schedule.validate(num_layers)

    new_config = Qwen3IHAConfig(
        **base_config.to_dict(),
        num_query_heads_per_layer=schedule.num_query_heads_per_layer,
        num_key_value_heads_per_layer=schedule.num_key_value_heads_per_layer,
        qk_head_dim_per_layer=schedule.qk_head_dim_per_layer,
        v_head_dim_per_layer=schedule.v_head_dim_per_layer,
        intermediate_size_per_layer=schedule.intermediate_size_per_layer,
        base_num_query_heads=base_nq,
        base_num_key_value_heads=base_nkv,
        base_qk_head_dim=base_qk,
        base_v_head_dim=base_v,
        base_intermediate_size=base_config.intermediate_size,
    )

    new_model = Qwen3IHAForCausalLM(new_config)
    new_sd = new_model.state_dict()
    base_sd = base_model.state_dict()

    for name, tensor in base_sd.items():
        if name in new_sd and new_sd[name].shape == tensor.shape:
            new_sd[name] = tensor

    for i in range(num_layers):
        pfx = f"model.layers.{i}."

        new_nq = int(schedule.num_query_heads_per_layer[i])
        new_nkv = int(schedule.num_key_value_heads_per_layer[i])
        new_qk = int(schedule.qk_head_dim_per_layer[i])
        new_v = int(schedule.v_head_dim_per_layer[i])

        q_w = base_sd[pfx + "self_attn.q_proj.weight"]
        k_w = base_sd[pfx + "self_attn.k_proj.weight"]
        v_w = base_sd[pfx + "self_attn.v_proj.weight"]
        o_w = base_sd[pfx + "self_attn.o_proj.weight"]

        new_sd[pfx + "self_attn.q_proj.weight"] = _resize_qkv_weight(q_w, base_nq, base_qk, new_nq, new_qk)
        new_sd[pfx + "self_attn.k_proj.weight"] = _resize_qkv_weight(k_w, base_nkv, base_qk, new_nkv, new_qk)
        new_sd[pfx + "self_attn.v_proj.weight"] = _resize_qkv_weight(v_w, base_nkv, base_v, new_nkv, new_v)
        new_sd[pfx + "self_attn.o_proj.weight"] = _resize_o_proj(o_w, base_nq, base_v, new_nq, new_v)

        if pfx + "self_attn.q_proj.bias" in base_sd:
            new_sd[pfx + "self_attn.q_proj.bias"] = _resize_qkv_bias(base_sd[pfx + "self_attn.q_proj.bias"], base_nq, base_qk, new_nq, new_qk)
            new_sd[pfx + "self_attn.k_proj.bias"] = _resize_qkv_bias(base_sd[pfx + "self_attn.k_proj.bias"], base_nkv, base_qk, new_nkv, new_qk)
            new_sd[pfx + "self_attn.v_proj.bias"] = _resize_qkv_bias(base_sd[pfx + "self_attn.v_proj.bias"], base_nkv, base_v, new_nkv, new_v)

        gate_w = base_sd[pfx + "mlp.gate_proj.weight"]
        up_w = base_sd[pfx + "mlp.up_proj.weight"]
        down_w = base_sd[pfx + "mlp.down_proj.weight"]
        new_mlp = int(schedule.intermediate_size_per_layer[i])
        gate_new, up_new, down_new = _resize_mlp(gate_w, up_w, down_w, new_mlp)
        new_sd[pfx + "mlp.gate_proj.weight"] = gate_new
        new_sd[pfx + "mlp.up_proj.weight"] = up_new
        new_sd[pfx + "mlp.down_proj.weight"] = down_new

    new_model.load_state_dict(new_sd, strict=False)

    before_params = sum(p.numel() for p in base_model.parameters())
    after_params = sum(p.numel() for p in new_model.parameters())
    kv_bytes_per_token = _kv_cache_bytes_per_token(schedule, _dtype_from_arg(args.dtype))

    print(f"params before: {before_params:,}")
    print(f"params after:  {after_params:,}")
    print(f"KV cache bytes/token: {kv_bytes_per_token:,}")
    print(f"KV cache @4096: {kv_bytes_per_token * 4096 / (1024**2):.2f} MiB")
    print(f"KV cache @8192: {kv_bytes_per_token * 8192 / (1024**2):.2f} MiB")

    new_model.save_pretrained(args.out_dir, safe_serialization=args.save_safetensors)
    new_config.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    
    print(f"New config: {new_config}")

    src_dir = os.path.dirname(__file__)
    for filename in ("configuration_qwen3_iha.py", "modeling_qwen3_iha.py"):
        shutil.copy2(os.path.join(src_dir, filename), os.path.join(args.out_dir, filename))

    with open(os.path.join(args.out_dir, "iha_schedule.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_query_heads_per_layer": schedule.num_query_heads_per_layer,
                "num_key_value_heads_per_layer": schedule.num_key_value_heads_per_layer,
                "qk_head_dim_per_layer": schedule.qk_head_dim_per_layer,
                "v_head_dim_per_layer": schedule.v_head_dim_per_layer,
                "intermediate_size_per_layer": schedule.intermediate_size_per_layer,
            },
            f,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
