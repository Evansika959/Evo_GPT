#!/usr/bin/env python3
"""Two-phase uptraining for morphed Qwen3 IHA checkpoints."""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import itertools
import json
import math
import os
import shlex
import sys
from typing import Iterable, Dict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        if not self.streams:
            return False
        return bool(getattr(self.streams[0], "isatty", lambda: False)())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uptrain morphed Qwen3 checkpoint")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="Salesforce/wikitext")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate_phase1", type=float, default=1e-5)
    parser.add_argument("--learning_rate_phase2", type=float, default=5e-6)
    parser.add_argument(
        "--max_steps_phase1",
        type=int,
        default=None,
        help="Phase 1 max steps. If unset, runs one full pass (1 epoch) over the training dataset.",
    )
    parser.add_argument(
        "--max_steps_phase2",
        type=int,
        default=None,
        help="Phase 2 max steps. If unset, runs one full pass (1 epoch) over the training dataset.",
    )
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cuda", help="auto|cuda|cpu")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of processes for dataset.map(). Defaults to max(1, nproc-2).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path to execution log file. If unset, writes to <output_dir>/uptrain_<timestamp>.log",
    )
    return parser.parse_args()


def _setup_execution_logging(args: argparse.Namespace):
    if args.log_file:
        log_file = args.log_file
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(args.output_dir, f"uptrain_{timestamp}.log")

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    log_handle = open(log_file, "a", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_handle)
    sys.stderr = _TeeStream(original_stderr, log_handle)

    started_at = dt.datetime.now()
    print(f"[uptrain] log_file: {os.path.abspath(log_file)}")
    print(f"[uptrain] start_time: {started_at.isoformat(timespec='seconds')}")
    print(f"[uptrain] argv: {' '.join(shlex.quote(a) for a in sys.argv)}")
    print(f"[uptrain] args: {json.dumps(vars(args), sort_keys=True)}")

    return log_handle, original_stdout, original_stderr, started_at, log_file


def _finalize_execution_logging(
    log_handle,
    original_stdout,
    original_stderr,
    started_at: dt.datetime,
    status: str,
) -> None:
    ended_at = dt.datetime.now()
    elapsed = ended_at - started_at
    print(f"[uptrain] end_time: {ended_at.isoformat(timespec='seconds')}")
    print(f"[uptrain] elapsed: {str(elapsed).split('.')[0]}")
    print(f"[uptrain] status: {status}")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()


def _get_dataset(args: argparse.Namespace):
    if args.dataset_name:
        dataset_config_name = args.dataset_config_name
        if dataset_config_name is None:
            if args.dataset_name == "allenai/c4":
                dataset_config_name = "en"
                print("dataset_config_name not provided for allenai/c4; defaulting to 'en'.")
            elif args.dataset_name == "Salesforce/wikitext":
                dataset_config_name = "wikitext-2-raw-v1"
                print("dataset_config_name not provided for Salesforce/wikitext; defaulting to 'wikitext-2-raw-v1'.")
            elif args.dataset_name == "openwebtext":
                dataset_config_name = None

        if dataset_config_name is not None:
            return load_dataset(args.dataset_name, dataset_config_name, split=args.dataset_split)
        return load_dataset(args.dataset_name, split=args.dataset_split)
    if args.data_path:
        return load_dataset("text", data_files={"train": args.data_path})["train"]
    raise ValueError("Provide --dataset_name or --data_path")


def _tokenize_and_group(dataset, tokenizer, block_size: int, text_column: str, num_proc: int | None = None):

    if num_proc is None:
        import multiprocessing
        num_proc = max(1, multiprocessing.cpu_count() - 2)
    print(f"[uptrain] dataset.map num_proc={num_proc}")

    def tokenize_fn(batch):
        return tokenizer(batch[text_column])

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[text_column],
        num_proc=num_proc,
        desc="Tokenizing",
    )

    def group_texts(batch):
        # Use itertools.chain instead of sum(list, []) to avoid O(n²) concat
        concatenated = {
            k: list(itertools.chain.from_iterable(batch[k])) for k in batch.keys()
        }
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [vals[i : i + block_size] for i in range(0, total_len, block_size)]
            for k, vals in concatenated.items()
        }
        result["labels"] = [x.copy() for x in result["input_ids"]]
        return result

    return tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Grouping into blocks",
    )


def _freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_param(module):
    for p in module.parameters():
        p.requires_grad = True


def _get_changed_layers(config) -> Dict[str, Iterable[int]]:
    kv_list = getattr(config, "num_key_value_heads_per_layer", None)
    mlp_list = getattr(config, "intermediate_size_per_layer", None)
    base_kv = getattr(config, "base_num_key_value_heads", config.num_key_value_heads)
    base_mlp = getattr(config, "base_intermediate_size", config.intermediate_size)

    changed_kv = []
    changed_mlp = []
    if kv_list:
        changed_kv = [i for i, kv in enumerate(kv_list) if kv != base_kv]
    if mlp_list:
        changed_mlp = [i for i, m in enumerate(mlp_list) if m != base_mlp]

    return {"kv": changed_kv, "mlp": changed_mlp}


def _set_phase1_requires_grad(model, config):
    _freeze_all(model)
    changed = _get_changed_layers(config)

    for idx in changed["kv"]:
        layer = model.model.layers[idx]
        _unfreeze_param(layer.self_attn.k_proj)
        _unfreeze_param(layer.self_attn.v_proj)

    for idx in changed["mlp"]:
        layer = model.model.layers[idx]
        _unfreeze_param(layer.mlp.gate_proj)
        _unfreeze_param(layer.mlp.up_proj)
        _unfreeze_param(layer.mlp.down_proj)


def _set_phase2_requires_grad(model):
    for p in model.parameters():
        p.requires_grad = True


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


def _prepare_train_dataset_for_torch(train_dataset):
    return train_dataset.with_format("torch", columns=["input_ids", "labels"])


def _collate_batch(examples):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in examples], dim=0),
        "labels": torch.stack([x["labels"] for x in examples], dim=0),
    }


def _run_phase_torch_loop(
    model,
    tokenizer,
    train_dataset,
    output_dir,
    args,
    learning_rate,
    max_steps: int | None,
    phase_name: str,
):
    os.makedirs(output_dir, exist_ok=True)
    train_dataset = _prepare_train_dataset_for_torch(train_dataset)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
        drop_last=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for this phase")

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    total_steps = int(max_steps) if max_steps is not None else len(dataloader)
    if total_steps <= 0:
        raise ValueError("No training steps available for this phase")

    warmup_steps = max(0, min(args.warmup_steps, total_steps))

    def lr_lambda(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        progress_denom = max(1, total_steps - warmup_steps)
        progress_ratio = min(1.0, max(0.0, (step_idx - warmup_steps) / progress_denom))
        return 0.5 * (1.0 + math.cos(math.pi * progress_ratio))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    use_cuda = next(model.parameters()).device.type == "cuda"
    use_amp = use_cuda and (args.bf16 or args.fp16)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_cuda and args.fp16)

    model.train()
    step = 0
    data_iter = iter(dataloader)
    model_device = next(model.parameters()).device
    progress = tqdm(total=total_steps, desc=phase_name, dynamic_ncols=True)
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(model_device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = model(**batch).loss
        else:
            loss = model(**batch).loss

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()

        scheduler.step()
        step += 1
        progress.update(1)

        if step % 10 == 0 or step == 1 or step == total_steps:
            lr = scheduler.get_last_lr()[0]
            progress.set_postfix(lr=f"{lr:.2e}", step=f"{step}/{total_steps}")

    progress.close()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _run_phase(
    model,
    tokenizer,
    train_dataset,
    output_dir,
    args,
    learning_rate,
    max_steps: int | None,
    phase_name: str,
):
    print(f"\n===== {phase_name} =====")
    try:
        from transformers import Trainer, TrainingArguments
        from transformers.trainer_callback import PrinterCallback, ProgressCallback

        training_args_kwargs = dict(
            output_dir=output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=learning_rate,
            warmup_steps=args.warmup_steps,
            lr_scheduler_type="cosine",
            max_grad_norm=args.max_grad_norm,
            bf16=args.bf16,
            fp16=args.fp16,
            logging_steps=10,
            save_steps=40000,
            save_total_limit=2,
            report_to=[],
            disable_tqdm=False,
            run_name=phase_name,
        )
        if max_steps is None:
            training_args_kwargs["num_train_epochs"] = 1.0
        else:
            training_args_kwargs["max_steps"] = max_steps

        training_args = TrainingArguments(**training_args_kwargs)

        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
        }
        trainer_signature = inspect.signature(Trainer.__init__).parameters
        if "tokenizer" in trainer_signature:
            trainer_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in trainer_signature:
            trainer_kwargs["processing_class"] = tokenizer

        trainer = Trainer(**trainer_kwargs)
        trainer.remove_callback(PrinterCallback)
        trainer.add_callback(ProgressCallback)
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    except Exception as exc:
        print(f"Trainer unavailable ({type(exc).__name__}: {exc}); falling back to native torch loop.")
        _run_phase_torch_loop(model, tokenizer, train_dataset, output_dir, args, learning_rate, max_steps, phase_name)


def main() -> None:
    args = parse_args()
    log_handle, original_stdout, original_stderr, started_at, _ = _setup_execution_logging(args)
    status = "success"

    try:
        device = _resolve_device(args.device)

        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=False,
        )
        _materialize_known_meta_params(model)
        model = model.to(device)

        dataset = _get_dataset(args)
        train_dataset = _tokenize_and_group(dataset, tokenizer, args.block_size, args.text_column, num_proc=args.num_workers)

        _set_phase1_requires_grad(model, model.config)
        _run_phase(
            model,
            tokenizer,
            train_dataset,
            args.output_dir + "/phase1",
            args,
            args.learning_rate_phase1,
            args.max_steps_phase1,
            "Phase 1",
        )

        _set_phase2_requires_grad(model)
        _run_phase(
            model,
            tokenizer,
            train_dataset,
            args.output_dir + "/phase2",
            args,
            args.learning_rate_phase2,
            args.max_steps_phase2,
            "Phase 2",
        )
    except Exception:
        status = "failed"
        raise
    finally:
        _finalize_execution_logging(log_handle, original_stdout, original_stderr, started_at, status)

if __name__ == "__main__":
    main()
