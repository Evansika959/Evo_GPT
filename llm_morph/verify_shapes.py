#!/usr/bin/env python3
"""Sanity checks for morphed Qwen3 IHA checkpoints."""
from __future__ import annotations

import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


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
    # Newer Transformers may return Cache-like objects with key_cache/value_cache lists.
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]

    try:
        layer_entry = past_key_values[layer_idx]
    except TypeError:
        # Some cache objects are iterable but not subscriptable (e.g., DynamicCache).
        layer_entry = list(past_key_values)[layer_idx]

    # Legacy format is usually a tuple/list where first two entries are (k, v).
    if isinstance(layer_entry, (tuple, list)) and len(layer_entry) >= 2:
        return layer_entry[0], layer_entry[1]

    raise ValueError(f"Unsupported past_key_values layer format at index {layer_idx}: {type(layer_entry)}")


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=_dtype_from_arg(args.dtype),
        low_cpu_mem_usage=False,
        trust_remote_code=args.trust_remote_code,
    )

    meta_params = [name for name, param in model.named_parameters() if param.device.type == "meta"]
    if meta_params:
        if meta_params == ["lm_head.weight"] and hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            model.lm_head.weight = model.model.embed_tokens.weight
        else:
            raise ValueError(f"Model has unresolved meta parameters: {meta_params}")

    model = model.to(device)

    model.eval()

    vocab = config.vocab_size
    batch = 2
    seq = 8
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

    kv_list = getattr(config, "num_key_value_heads_per_layer", None)
    head_dim = config.hidden_size // config.num_attention_heads

    for i in range(len(past)):
        k, v = _extract_layer_kv(past, i)
        if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
            raise ValueError(f"Layer {i} head_dim mismatch: {k.shape} {v.shape}")
        if kv_list is not None:
            expected_kv = kv_list[i]
            if k.shape[1] != expected_kv:
                raise ValueError(f"Layer {i} KV heads mismatch: {k.shape[1]} vs {expected_kv}")

    # Step decode with cache
    next_ids = torch.randint(low=0, high=vocab, size=(batch, 1), device=device)
    with torch.no_grad():
        out2 = model(next_ids, use_cache=True, past_key_values=past)

    if torch.isnan(out2.logits).any():
        raise ValueError("NaNs in second-step logits")

    print("verify_shapes: OK")


if __name__ == "__main__":
    main()
