#!/usr/bin/env python3
"""Perplexity comparison for original, morphed, and uptrained models."""
from __future__ import annotations

import argparse
import math
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare perplexity across original/morphed/uptrained models")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to morphed model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Uptraining output dir containing phase2")
    parser.add_argument("--original_model_id", type=str, default=None, help="HF id/path of original pre-morph model")
    parser.add_argument("--dataset_name", type=str, default="Salesforce/wikitext")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda", help="auto|cuda|cpu")
    parser.add_argument("--ppl_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--ppl_max_samples", type=int, default=256, help="Max raw text examples for PPL eval subset")
    parser.add_argument("--ppl_batch_size", type=int, default=16)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _dtype_from_arg(dtype: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _materialize_known_meta_params(model) -> None:
    meta_params = [name for name, p in model.named_parameters() if p.device.type == "meta"]
    if not meta_params:
        return

    if meta_params == ["lm_head.weight"] and hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        model.lm_head.weight = model.model.embed_tokens.weight
        return

    raise ValueError(f"Model has unresolved meta parameters: {meta_params}")


def _get_dataset(dataset_name: str, dataset_config_name: str | None, dataset_split: str, data_path: str | None):
    if dataset_name:
        config_name = dataset_config_name
        if config_name is None:
            if dataset_name == "allenai/c4":
                config_name = "en"
                print("dataset_config_name not provided for allenai/c4; defaulting to 'en'.")
            elif dataset_name == "Salesforce/wikitext":
                config_name = "wikitext-2-raw-v1"
                print("dataset_config_name not provided for Salesforce/wikitext; defaulting to 'wikitext-2-raw-v1'.")

        if config_name is not None:
            return load_dataset(dataset_name, config_name, split=dataset_split)
        return load_dataset(dataset_name, split=dataset_split)

    if data_path:
        return load_dataset("text", data_files={"train": data_path})["train"]

    raise ValueError("Provide dataset_name or data_path")


def _subset_dataset(dataset, max_samples: int):
    if max_samples is None or max_samples <= 0:
        return dataset

    try:
        n = len(dataset)
    except TypeError:
        return dataset

    if n <= max_samples:
        return dataset
    return dataset.select(range(max_samples))


def _tokenize_and_group(dataset, tokenizer, block_size: int, text_column: str):
    def tokenize_fn(batch):
        return tokenizer(batch[text_column])

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=[text_column])

    def group_texts(batch):
        concatenated = {k: sum(batch[k], []) for k in batch.keys()}
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [vals[i : i + block_size] for i in range(0, total_len, block_size)]
            for k, vals in concatenated.items()
        }
        result["labels"] = [x.copy() for x in result["input_ids"]]
        return result

    return tokenized.map(group_texts, batched=True)


def _collate_batch(examples):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in examples], dim=0),
        "labels": torch.stack([x["labels"] for x in examples], dim=0),
    }


def _compute_perplexity(model, dataset, device: str, batch_size: int, dtype: torch.dtype) -> float:
    model.eval()
    dataset_torch = dataset.with_format("torch", columns=["input_ids", "labels"])
    dataloader = DataLoader(
        dataset_torch,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
        drop_last=False,
    )

    total_nll = 0.0
    total_tokens = 0
    use_amp = device == "cuda" and dtype in (torch.float16, torch.bfloat16)

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    loss = model(**batch).loss
            else:
                loss = model(**batch).loss

            tokens = batch["labels"].numel()
            total_nll += float(loss.item()) * tokens
            total_tokens += tokens

    if total_tokens == 0:
        raise ValueError("No tokens available for perplexity computation")

    avg_nll = total_nll / total_tokens
    return float(math.exp(avg_nll))


def _load_eval_model(model_id_or_path: str, dtype: torch.dtype, device: str, trust_remote_code: bool):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        low_cpu_mem_usage=False,
    )
    _materialize_known_meta_params(model)
    return model.to(device)


def _model_size_stats(model, eval_dtype: torch.dtype) -> tuple[int, float]:
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = torch.tensor([], dtype=eval_dtype).element_size()
    total_mib = (total_params * bytes_per_param) / (1024**2)
    return total_params, total_mib


def _kv_heads_per_layer_from_config(config) -> list[int]:
    kv_per_layer = getattr(config, "num_key_value_heads_per_layer", None)
    if kv_per_layer is not None:
        return [int(x) for x in kv_per_layer]

    num_layers = int(getattr(config, "num_hidden_layers"))
    global_kv = int(getattr(config, "num_key_value_heads"))
    return [global_kv] * num_layers


def _kv_cache_usage_stats(config, cache_dtype: torch.dtype, context_len: int) -> tuple[int, float]:
    hidden_size = int(getattr(config, "hidden_size"))
    num_attention_heads = int(getattr(config, "num_attention_heads"))
    head_dim = hidden_size // num_attention_heads
    kv_heads_sum = sum(_kv_heads_per_layer_from_config(config))
    bytes_per_elem = torch.tensor([], dtype=cache_dtype).element_size()

    bytes_per_token = 2 * kv_heads_sum * head_dim * bytes_per_elem
    total_mib = (bytes_per_token * context_len) / (1024**2)
    return bytes_per_token, total_mib


def _format_params_short(param_count: int) -> str:
    if param_count >= 1_000_000_000:
        return f"{param_count / 1_000_000_000:.3f}B"
    if param_count >= 1_000_000:
        return f"{param_count / 1_000_000:.3f}M"
    return f"{param_count:,}"


def _infer_original_model_id(original_model_id: str | None, model_dir: str, trust_remote_code: bool) -> str | None:
    if original_model_id:
        return original_model_id

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    config_name_or_path = getattr(config, "_name_or_path", None)
    if config_name_or_path and config_name_or_path not in ("", model_dir):
        return config_name_or_path
    return None


def _looks_like_local_path(path: str) -> bool:
    return path.startswith(".") or path.startswith("/")


def _resolve_uptrained_model_path(output_dir: str) -> str | None:
    candidates = [os.path.join(output_dir, "phase2"), output_dir]
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return None


def run_ppl_comparison(
    *,
    model_dir: str,
    output_dir: str,
    original_model_id: str | None,
    dataset_name: str,
    dataset_config_name: str | None,
    dataset_split: str,
    data_path: str | None,
    text_column: str,
    block_size: int,
    device: str,
    ppl_dtype: str,
    ppl_max_samples: int,
    ppl_batch_size: int,
    trust_remote_code: bool,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    eval_raw = _get_dataset(dataset_name, dataset_config_name, dataset_split, data_path)
    eval_raw = _subset_dataset(eval_raw, ppl_max_samples)
    eval_dataset = _tokenize_and_group(eval_raw, tokenizer, block_size, text_column)

    resolved_device = _resolve_device(device)
    eval_dtype = _dtype_from_arg(ppl_dtype)
    phase2_dir = _resolve_uptrained_model_path(output_dir)

    resolved_original = _infer_original_model_id(original_model_id, model_dir, trust_remote_code)
    if resolved_original is None:
        print("Skipping PPL comparison: could not infer original model id. Provide --original_model_id to enable it.")
        return

    model_specs = [
        ("original", resolved_original),
        ("morphed", model_dir),
    ]
    if phase2_dir is not None:
        model_specs.append(("uptrained", phase2_dir))
    else:
        print(
            "Skipping uptrained model in comparison: no local checkpoint found at "
            f"'{os.path.join(output_dir, 'phase2')}' or '{output_dir}'."
        )

    print("\n=== Perplexity Comparison (same eval data + dtype) ===")
    print(f"dataset={dataset_name} config={dataset_config_name} split={dataset_split} samples={ppl_max_samples}")
    print(f"dtype={ppl_dtype} device={resolved_device}")

    size_stats = {}
    kv_stats = {}

    for label, model_path in model_specs:
        if _looks_like_local_path(model_path) and not os.path.isdir(model_path):
            print(f"Skipping {label}: local path not found ({model_path})")
            continue
        model = _load_eval_model(model_path, eval_dtype, resolved_device, trust_remote_code)
        params, est_mib = _model_size_stats(model, eval_dtype)
        size_stats[label] = (params, est_mib)
        kv_token_bytes, kv_4k_mib = _kv_cache_usage_stats(model.config, eval_dtype, context_len=4096)
        kv_stats[label] = (kv_token_bytes, kv_4k_mib)
        ppl = _compute_perplexity(model, eval_dataset, resolved_device, ppl_batch_size, eval_dtype)
        print(
            f"{label:9s} PPL: {ppl:.4f} | params={_format_params_short(params)} ({params:,}) "
            f"| est_weight_mem={est_mib:.2f} MiB | kv_cache={kv_token_bytes:,} B/token, {kv_4k_mib:.2f} MiB@4k ({model_path})"
        )
        del model
        if resolved_device == "cuda":
            torch.cuda.empty_cache()

    if "original" in size_stats and "morphed" in size_stats:
        orig_params, orig_mib = size_stats["original"]
        morph_params, morph_mib = size_stats["morphed"]
        morph_pct = (morph_params / orig_params * 100.0) if orig_params > 0 else float("nan")
        print("\n--- Morph Size Summary ---")
        print(
            f"before morph (original): params={_format_params_short(orig_params)} ({orig_params:,}), "
            f"est_weight_mem={orig_mib:.2f} MiB"
        )
        print(
            f"after morph  (morphed):  params={_format_params_short(morph_params)} ({morph_params:,}), "
            f"est_weight_mem={morph_mib:.2f} MiB"
        )
        print(f"morphed/original: {morph_pct:.2f}%")

    if "original" in kv_stats and "morphed" in kv_stats:
        orig_kv_bpt, orig_kv_4k_mib = kv_stats["original"]
        morph_kv_bpt, morph_kv_4k_mib = kv_stats["morphed"]
        print("\n--- KV Cache Summary ---")
        print(f"before morph (original): kv_cache={orig_kv_bpt:,} B/token, {orig_kv_4k_mib:.2f} MiB@4k")
        print(f"after morph  (morphed):  kv_cache={morph_kv_bpt:,} B/token, {morph_kv_4k_mib:.2f} MiB@4k")

    print("=== End PPL Comparison ===\n")


def main() -> None:
    args = parse_args()
    run_ppl_comparison(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        original_model_id=args.original_model_id,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
        data_path=args.data_path,
        text_column=args.text_column,
        block_size=args.block_size,
        device=args.device,
        ppl_dtype=args.ppl_dtype,
        ppl_max_samples=args.ppl_max_samples,
        ppl_batch_size=args.ppl_batch_size,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
