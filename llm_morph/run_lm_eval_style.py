#!/usr/bin/env python3
"""Run benchmark evaluation with lm-eval-harness style APIs for local/HF causal LMs."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LM-eval style benchmark evaluation")
    parser.add_argument("--model_dir", type=str, required=True, help="Local model path or HF model id")
    parser.add_argument(
        "--tasks",
        type=str,
        default="hellaswag",
        help="Comma-separated lm-eval task names (e.g., hellaswag,arc_easy,piqa)",
    )
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples per task")
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size for lm-eval (int or 'auto')")
    parser.add_argument("--max_batch_size", type=int, default=None, help="Optional cap when batch_size='auto'")
    parser.add_argument("--limit", type=float, default=None, help="Optional cap/fraction for quick eval")
    parser.add_argument("--device", type=str, default="cuda", help="cuda|cpu|auto")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model dtype passed to lm-eval hf backend",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output_yaml", type=str, default="outputs/lm_eval_results.yaml", help="Path to save full lm-eval YAML")
    parser.add_argument("--log_samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply_chat_template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--system_instruction", type=str, default=None)
    parser.add_argument("--fewshot_as_multiturn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbosity", type=str, default="INFO", help="lm-eval verbosity")
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _parse_tasks(tasks: str) -> list[str]:
    task_list = [item.strip() for item in tasks.split(",") if item.strip()]
    if not task_list:
        raise ValueError("--tasks must include at least one task")
    return task_list


def _maybe_int(value: str) -> int | str:
    if value == "auto":
        return value
    return int(value)


def _build_model_args(args: argparse.Namespace, resolved_device: str) -> str:
    parts = [
        f"pretrained={args.model_dir}",
        f"dtype={args.dtype}",
        f"trust_remote_code={str(args.trust_remote_code).lower()}",
    ]
    if resolved_device != "auto":
        parts.append(f"device={resolved_device}")
    return ",".join(parts)


def _filter_kwargs_for_signature(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted and v is not None}


def _make_yaml_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _make_yaml_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_yaml_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_make_yaml_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    args = parse_args()
    resolved_device = _resolve_device(args.device)
    task_list = _parse_tasks(args.tasks)

    try:
        from lm_eval import evaluator
    except Exception as exc:
        raise RuntimeError(
            "lm-eval-harness is not installed. Install it with: pip install lm-eval"
        ) from exc

    model_args = _build_model_args(args, resolved_device)

    eval_kwargs: dict[str, Any] = {
        "model": "hf",
        "model_args": model_args,
        "tasks": task_list,
        "num_fewshot": args.num_fewshot,
        "batch_size": _maybe_int(args.batch_size),
        "max_batch_size": args.max_batch_size,
        "limit": args.limit,
        "seed": args.seed,
        "log_samples": args.log_samples,
        "apply_chat_template": args.apply_chat_template,
        "system_instruction": args.system_instruction,
        "fewshot_as_multiturn": args.fewshot_as_multiturn,
        "verbosity": args.verbosity,
    }

    eval_kwargs = _filter_kwargs_for_signature(evaluator.simple_evaluate, eval_kwargs)
    results = evaluator.simple_evaluate(**eval_kwargs)

    summary = results.get("results", {})
    if not summary:
        print("No task results found.")
    else:
        print("\n=== LM-eval style summary ===")
        for task_name, metrics in summary.items():
            printable = {
                key: value
                for key, value in metrics.items()
                if key.endswith(",none") or key in {"acc", "acc_norm", "exact_match", "f1"}
            }
            if not printable:
                printable = metrics
            print(f"{task_name}: {yaml.safe_dump(_make_yaml_safe(printable), sort_keys=True, default_flow_style=True).strip()}")

    safe_results = _make_yaml_safe(results)

    output_path = Path(args.output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(safe_results, f, sort_keys=False, allow_unicode=True)
    print(f"\nSaved full output to: {output_path}")


if __name__ == "__main__":
    main()
