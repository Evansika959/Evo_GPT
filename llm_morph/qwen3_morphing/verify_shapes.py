#!/usr/bin/env python3
from __future__ import annotations

import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify shapes and cache for morphed Qwen3")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _dtype_from_arg(dtype: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]


def _extract_layer_kv(past_key_values, layer_idx: int):
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    try:
        layer_entry = past_key_values[layer_idx]
    except TypeError:
        layer_entry = list(past_key_values)[layer_idx]
    if isinstance(layer_entry, (tuple, list)) and len(layer_entry) >= 2:
        return layer_entry[0], layer_entry[1]
    raise ValueError(f"Unsupported past_key_values layer format at index {layer_idx}: {type(layer_entry)}")


def _materialize_known_meta_params(model) -> None:
    meta_params = [name for name, param in model.named_parameters() if param.device.type == "meta"]
    if not meta_params:
        return
    if meta_params == ["lm_head.weight"] and hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        model.lm_head.weight = model.model.embed_tokens.weight
        return
    raise ValueError(f"Model has unresolved meta parameters: {meta_params}")


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=_dtype_from_arg(args.dtype),
        low_cpu_mem_usage=False,
        trust_remote_code=args.trust_remote_code,
    )
    _materialize_known_meta_params(model)
    model = model.to(device)
    model.eval()

    vocab = config.vocab_size
    batch, seq = 2, 8
    input_ids = torch.randint(low=0, high=vocab, size=(batch, seq), device=device)

    with torch.no_grad():
        out = model(input_ids, use_cache=True)

    logits = out.logits
    if logits.shape != (batch, seq, vocab):
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    if torch.isnan(logits).any():
        raise ValueError("NaNs in logits")

    past = out.past_key_values
    if past is None:
        raise ValueError("past_key_values missing")

    nkv_list = getattr(config, "num_key_value_heads_per_layer", None)
    d_qk_list = getattr(config, "qk_head_dim_per_layer", None)
    d_v_list = getattr(config, "v_head_dim_per_layer", None)

    if d_qk_list is None:
        inferred = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        d_qk_list = [inferred] * config.num_hidden_layers
    if d_v_list is None:
        d_v_list = d_qk_list

    for i in range(len(past)):
        k, v = _extract_layer_kv(past, i)
        if k.shape[-1] != int(d_qk_list[i]):
            raise ValueError(f"Layer {i} K head_dim mismatch: {k.shape[-1]} vs {d_qk_list[i]}")
        if v.shape[-1] != int(d_v_list[i]):
            raise ValueError(f"Layer {i} V head_dim mismatch: {v.shape[-1]} vs {d_v_list[i]}")
        if nkv_list is not None and k.shape[1] != int(nkv_list[i]):
            raise ValueError(f"Layer {i} KV heads mismatch: {k.shape[1]} vs {nkv_list[i]}")

    next_ids = torch.randint(low=0, high=vocab, size=(batch, 1), device=device)
    with torch.no_grad():
        out2 = model(next_ids, use_cache=True, past_key_values=past)

    if torch.isnan(out2.logits).any():
        raise ValueError("NaNs in second-step logits")

    print("verify_shapes: OK")


if __name__ == "__main__":
    main()
