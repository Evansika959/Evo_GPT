#!/usr/bin/env python3
"""Run multiple-choice benchmark evaluation for morphed/local Hugging Face causal LMs."""
from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate morphed/local causal LM on benchmark datasets")
	parser.add_argument("--model_dir", type=str, required=True, help="Model path or HF id (e.g., ./morphed_output)")
	parser.add_argument(
		"--benchmark",
		type=str,
		default="hellaswag",
		choices=["hellaswag", "arc-easy", "arc-challenge", "sciq", "piqa", "winogrande", "boolq"],
		help="Dataset to evaluate (single)",
	)
	parser.add_argument(
		"--benchmarks",
		type=str,
		default=None,
		help="Comma-separated list of benchmarks to evaluate in one run",
	)
	parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
	parser.add_argument(
		"--dtype",
		type=str,
		default="bfloat16",
		choices=["bfloat16", "float16", "float32"],
		help="Evaluation dtype (and model load dtype)",
	)
	parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
	parser.add_argument("--max_examples", type=int, default=None, help="Optional cap on number of examples")
	parser.add_argument("--seed", type=int, default=1337, help="Random seed for shuffling")
	parser.add_argument("--block_size", type=int, default=None, help="Override model block size")
	parser.add_argument("--length_norm", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--output_json", type=str, default=None, help="Optional path to write metrics JSON")
	parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


def _parse_benchmarks(args: argparse.Namespace) -> List[str]:
	if args.benchmarks:
		items = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
		if not items:
			raise ValueError("--benchmarks must include at least one name")
		return items
	return [args.benchmark]


def _resolve_device(device_arg: str) -> str:
	if device_arg == "auto":
		return "cuda" if torch.cuda.is_available() else "cpu"
	return device_arg


def _dtype_from_arg(dtype: str):
	return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]


def _materialize_known_meta_params(model) -> None:
	meta_params = [name for name, p in model.named_parameters() if p.device.type == "meta"]
	if not meta_params:
		return

	if meta_params == ["lm_head.weight"] and hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
		model.lm_head.weight = model.model.embed_tokens.weight
		return

	raise ValueError(f"Model has unresolved meta parameters: {meta_params}")


def _build_hellaswag_context(example: dict) -> str:
	ctx = example.get("ctx")
	if ctx:
		return ctx.strip()
	ctx_a = example.get("ctx_a", "").strip()
	ctx_b = example.get("ctx_b", "").strip()
	return (ctx_a + " " + ctx_b).strip()


def _get_benchmark_dataset(benchmark: str) -> tuple[str, str | None]:
	if benchmark == "arc-easy":
		return "ai2_arc", "ARC-Easy"
	if benchmark == "arc-challenge":
		return "ai2_arc", "ARC-Challenge"
	if benchmark == "sciq":
		return "sciq", None
	if benchmark == "piqa":
		return "piqa", None
	if benchmark == "winogrande":
		return "winogrande", "winogrande_xl"
	if benchmark == "boolq":
		return "boolq", None
	return "hellaswag", None


def _extract_example(benchmark: str, example: dict) -> tuple[str, List[str], int | None]:
	if benchmark == "hellaswag":
		ctx_text = _build_hellaswag_context(example)
		endings = example["endings"]
		label = example.get("label")
		return ctx_text, endings, label

	if benchmark in ("arc-easy", "arc-challenge"):
		ctx_text = example.get("question", "").strip()
		choices = example.get("choices", {})
		endings = choices.get("text", [])
		labels = choices.get("label", [])
		answer_key = example.get("answerKey")
		label = labels.index(answer_key) if answer_key in labels else None
		return ctx_text, endings, label

	if benchmark == "sciq":
		ctx_text = example.get("question", "").strip()
		endings = [
			example.get("correct_answer", ""),
			example.get("distractor1", ""),
			example.get("distractor2", ""),
			example.get("distractor3", ""),
		]
		return ctx_text, endings, 0

	if benchmark == "piqa":
		ctx_text = example.get("goal", "").strip()
		endings = [example.get("sol1", ""), example.get("sol2", "")]
		label = example.get("label")
		return ctx_text, endings, label

	if benchmark == "winogrande":
		sentence = example.get("sentence", "").strip()
		option1 = example.get("option1", "")
		option2 = example.get("option2", "")
		if "_" in sentence:
			before, after = sentence.split("_", 1)
			ctx_text = before
			endings = [option1 + after, option2 + after]
		else:
			ctx_text = sentence
			endings = [" " + option1, " " + option2]
		label_raw = example.get("answer")
		label = int(label_raw) - 1 if label_raw is not None else None
		return ctx_text, endings, label

	if benchmark == "boolq":
		passage = example.get("passage", "").strip()
		question = example.get("question", "").strip()
		ctx_text = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
		endings = [" yes", " no"]
		answer = example.get("answer")
		label = 0 if answer is True else 1 if answer is False else None
		return ctx_text, endings, label

	raise ValueError(f"Unsupported benchmark: {benchmark}")


def _get_block_size(model, override: Optional[int]) -> int:
	if override is not None:
		return override
	if hasattr(model.config, "n_positions") and model.config.n_positions:
		return int(model.config.n_positions)
	if hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings:
		return int(model.config.max_position_embeddings)
	return 1024


def _score_example(
	model,
	encode,
	ctx_text: str,
	endings: List[str],
	block_size: int,
	length_norm: bool,
	device: str,
	ctx_autocast,
) -> List[float]:
	ctx_tokens = encode(ctx_text)
	scores: List[float] = []

	for ending in endings:
		end_tokens = encode(ending)
		if len(end_tokens) == 0:
			scores.append(-math.inf)
			continue

		max_ctx_len = max(0, block_size - len(end_tokens))
		ctx_trim = ctx_tokens[-max_ctx_len:] if len(ctx_tokens) > max_ctx_len else ctx_tokens

		full = ctx_trim + end_tokens
		if len(full) < 2:
			scores.append(-math.inf)
			continue

		input_ids = torch.tensor(full[:-1], device=device).unsqueeze(0)
		target_ids = torch.tensor(full[1:], device=device).unsqueeze(0)
		ending_start = max(len(ctx_trim) - 1, 0)

		with ctx_autocast:
			logits = model(input_ids).logits
		logprobs = torch.log_softmax(logits, dim=-1)
		target_slice = target_ids[:, ending_start:]
		lp = logprobs[:, ending_start:, :].gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)

		scores.append(lp.mean().item() if length_norm else lp.sum().item())

	return scores


def main() -> None:
	args = parse_args()

	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True

	device = _resolve_device(args.device)
	eval_dtype = _dtype_from_arg(args.dtype)

	tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=args.trust_remote_code)
	if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
		tokenizer.pad_token = tokenizer.eos_token

	model = AutoModelForCausalLM.from_pretrained(
		args.model_dir,
		dtype=eval_dtype,
		low_cpu_mem_usage=False,
		trust_remote_code=args.trust_remote_code,
	)
	_materialize_known_meta_params(model)
	model = model.to(device)
	model.eval()

	block_size = _get_block_size(model, args.block_size)
	device_type = "cuda" if "cuda" in device else "cpu"
	ctx_autocast = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=eval_dtype)

	encode = lambda s: tokenizer.encode(s, add_special_tokens=False)

	all_metrics = []
	benchmark_list = _parse_benchmarks(args)
	for benchmark_idx, benchmark in enumerate(benchmark_list, start=1):
		print(f"[{benchmark_idx}/{len(benchmark_list)}] Running benchmark: {benchmark}")
		dataset_name, dataset_config = _get_benchmark_dataset(benchmark)
		if dataset_config is None:
			dataset = load_dataset(dataset_name, split=args.split)
		else:
			dataset = load_dataset(dataset_name, dataset_config, split=args.split)

		if args.max_examples is not None:
			dataset = dataset.shuffle(seed=args.seed).select(range(args.max_examples))

		correct = 0
		total = 0
		skipped = 0
		n_examples = len(dataset)

		with torch.inference_mode():
			for example in tqdm(dataset, total=n_examples, desc=f"{benchmark} eval", unit="ex"):
				ctx_text, endings, label = _extract_example(benchmark, example)
				if label is None:
					skipped += 1
					continue

				scores = _score_example(
					model=model,
					encode=encode,
					ctx_text=ctx_text,
					endings=endings,
					block_size=block_size,
					length_norm=args.length_norm,
					device=device,
					ctx_autocast=ctx_autocast,
				)

				pred = int(np.argmax(scores))
				if pred == int(label):
					correct += 1
				total += 1

		accuracy = (correct / total) if total else float("nan")
		metrics = {
			"benchmark": benchmark,
			"split": args.split,
			"total": total,
			"correct": correct,
			"accuracy": accuracy,
			"skipped": skipped,
			"block_size": block_size,
			"length_norm": bool(args.length_norm),
			"model_dir": args.model_dir,
		}
		all_metrics.append(metrics)
		print(json.dumps(metrics, indent=2))

	if args.output_json:
		with open(args.output_json, "w", encoding="utf-8") as f:
			json.dump(all_metrics, f, indent=2)
			f.write("\n")


if __name__ == "__main__":
	main()
