"""NSGA-II architecture search using surrogate predictor with optional periodic real-training verification."""

from nsga2 import Population
from typing import List, Dict, Any, Tuple
from search_space import Individual, HeteroSearchSpace
from surrogate import (
    surrogate_eval,
    load_surrogate,
    RealDataBuffer,
    finetune_surrogate,
    compute_accuracy_metrics,
    select_for_real_eval,
)
import yaml
import csv
import logging
import time
import os
import argparse
import random
import json
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
for name in ("paramiko", "paramiko.transport", "fabric", "invoke"):
    logging.getLogger(name).disabled = True


# ---------------------------------------------------------------------------
# Reusable helpers (same as run_exp.py)
# ---------------------------------------------------------------------------

def load_hosts_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hosts file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError("Hosts YAML must be a top-level list of IPs")
    hosts = [str(x).strip() for x in data if isinstance(x, (str, int, float)) and str(x).strip()]
    if not hosts:
        raise ValueError(f"No hosts parsed from file: {path}")
    return hosts


def load_search_space_from_yaml(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Search space file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Search space YAML must define a mapping with 'global_spec' and 'layer_spec'.")
    global_spec = data.get("global_spec")
    layer_spec = data.get("layer_spec")
    if not isinstance(global_spec, dict) or not isinstance(layer_spec, dict):
        raise ValueError("Search space YAML missing 'global_spec' or 'layer_spec' dictionaries.")
    return global_spec, layer_spec


def load_initial_individuals(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Initial population file not found: {path}")
    _, ext = os.path.splitext(path)
    with open(path, "r", encoding="utf-8") as f:
        if ext.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif ext.lower() == ".json":
            data = json.load(f)
        else:
            raise ValueError("Initial population file must be .json, .yaml, or .yml")
    if isinstance(data, dict) and "individuals" in data and isinstance(data["individuals"], list):
        data = data["individuals"]
    elif isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Initial population must be a list of individual dicts or a single dict")
    individuals = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Initial individual at index {idx} is not a dict")
        individuals.append(entry)
    if not individuals:
        raise ValueError("No individuals found in the initial population file")
    return individuals


def parse_constraint_arg(entry: str) -> Tuple[str, float]:
    if "=" not in entry:
        raise argparse.ArgumentTypeError("Constraints must be formatted as key=value")
    key, value = entry.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Constraint key cannot be empty")
    try:
        return key, float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Constraint value for '{key}' must be numeric")


# ---------------------------------------------------------------------------
# Real-training on a subset of individuals
# ---------------------------------------------------------------------------

def real_train_subset(
    individuals: List[Individual],
    hosts: List[str],
    user: str,
    key_filename: str,
    run_dir_name: str,
    max_iters: int = 10000,
    conda_env: str = "reallmforge",
    dataset: str = "minipile",
    timeout: int = 10000,
) -> List[float]:
    """Run real training on a subset of individuals and return their val_loss values.

    Creates a temporary Population to leverage existing to_yaml/sw_eval infrastructure,
    then extracts the val_loss results.
    """
    # Create a temporary population with just the selected individuals
    temp_pop = Population(individuals, search_space=None)
    temp_pop.gen = 0  # Treat as initial population for yaml generation
    train_yaml_path = temp_pop.to_yaml(save_path="real_eval_train")

    from remote_trainer import RemoteTrainer
    trainer = RemoteTrainer(hosts=hosts, user=user, key_filename=key_filename)
    trainer.submit_job(
        path_to_yaml=train_yaml_path,
        remote_work_dir=f"/home/{user}/Evo_GPT",
        dir_name=run_dir_name + "_realeval",
        max_iters=max_iters,
        conda_env=conda_env,
        dataset=dataset,
    )
    time.sleep(5)
    trainer.poll_jobs()
    trainer.wait_for_all(poll_interval=600, timeout=timeout, verbose=True)

    from nsga2 import load_csv_with_idx_lookup
    data_csv = trainer.fetch_results(local_dir="real_eval_train", gen=0)
    sw_data = load_csv_with_idx_lookup(data_csv)

    # Extract val_loss in order (idx is 1-based)
    real_losses = []
    for i in range(len(individuals)):
        real_losses.append(sw_data.get(i + 1, float("inf")))

    return real_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run NSGA-II search with surrogate predictor and optional real-training verification"
    )

    # Surrogate model args
    parser.add_argument("--checkpoint", type=str, default="surrogate/ckpts/model_flex40_optimized.pt", help="Path to trained surrogate checkpoint (.pt). Model config is auto-detected from the .json sidecar file.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for surrogate inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # NSGA-II args
    parser.add_argument("--pop_size", type=int, default=128, help="Population size")
    parser.add_argument("--max_layers", type=int, default=10, help="Max number of layers (L_max)")
    parser.add_argument("--min_layers", type=int, default=1, help="Min number of layers (L_min)")
    parser.add_argument("--offspring", type=int, default=64, help="Number of offspring per generation")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations to run")
    parser.add_argument("--crossover_rate", type=float, default=0.9, help="Crossover rate")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate")
    parser.add_argument(
        "--freeze_layer_mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable layer_mask mutation and keep all layers active",
    )
    parser.add_argument("--exp_name", type=str, default="surrogate_search", help="Experiment name")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to population checkpoint to resume from")
    parser.add_argument(
        "--search_space_config",
        type=str,
        default="search_space_def/search_space_200M.yaml",
        help="Path to YAML search space definition",
    )
    parser.add_argument("--objectives", type=str, nargs="+", default=["val_loss", "params"], help="Objectives to minimize")
    parser.add_argument("--max_params", type=float, default=800_000_000, help="Constraint: max parameter count")
    parser.add_argument("--max_val_loss", type=float, default=3.6, help="Constraint: max validation loss")
    parser.add_argument(
        "--constraint",
        action="append",
        type=parse_constraint_arg,
        metavar="KEY=VALUE",
        help="Custom constraint thresholds (e.g., --constraint params=5e8)",
    )
    parser.add_argument("--init_individuals", type=str, default=None, help="Path to predefined initial individuals")

    # Real-training verification args
    parser.add_argument("--real_eval_freq", type=int, default=0, help="Real-train every N generations (0=disabled)")
    parser.add_argument("--real_eval_count", type=int, default=8, help="Number of individuals to real-train per cycle")
    parser.add_argument(
        "--real_eval_strategy",
        type=str,
        default="full_population",
        choices=["full_population", "pareto_front", "pareto_and_random", "random", "top_k"],
        help="Strategy for selecting individuals for real training",
    )
    parser.add_argument("--finetune_epochs", type=int, default=10, help="Finetuning epochs per real-eval cycle")
    parser.add_argument("--finetune_lr", type=float, default=1e-4, help="Finetuning learning rate")

    # Remote training args (only needed when real_eval_freq > 0)
    parser.add_argument("--hosts-file", type=str, default="../host_configs/hosts.yaml", help="Hosts file for remote training")
    parser.add_argument("--user", type=str, default="xinting", help="SSH username")
    parser.add_argument("--key", type=str, default="/home/xinting/.ssh/id_rsa", help="SSH private key path")
    parser.add_argument("--conda_env", type=str, default="reallmforge", help="Conda env on remote hosts")
    parser.add_argument("--max_iters", type=int, default=10000, help="Max training iterations for real eval")
    parser.add_argument("--timeout", type=int, default=10000, help="Timeout for remote training (seconds)")
    parser.add_argument("--dataset", type=str, default="minipile", help="Dataset for real training")

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for log files")
    parser.add_argument("--verbose_log", action="store_true", default=False, help="Log per-offspring architecture configs in gen_results CSV")

    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Set up file logging (tee-like: console + file) ────────────────
    run_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = args.log_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.exp_name}_{run_time}.log")

    import sys as _sys

    class _Tee:
        """Write to both a file and the original stream."""
        def __init__(self, stream, filepath):
            self._stream = stream
            self._file = open(filepath, "a")
        def write(self, data):
            self._stream.write(data)
            self._file.write(data)
            self._file.flush()
        def flush(self):
            self._stream.flush()
            self._file.flush()

    _sys.stdout = _Tee(_sys.stdout, log_file)
    _sys.stderr = _Tee(_sys.stderr, log_file)
    # Also route the logging module to the same file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    logging.getLogger().addHandler(file_handler)

    print(f"Logging to {log_file}")
    print(f"Args: {vars(args)}")

    # ── Load search space ─────────────────────────────────────────────
    config_path = args.search_space_config
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)

    global_spec, layer_spec = load_search_space_from_yaml(config_path)
    search_space = HeteroSearchSpace.from_dicts(
        global_spec, layer_spec,
        L_max=args.max_layers, L_min=args.min_layers,
        freeze_layer_mask=args.freeze_layer_mask,
    )
    print("Using search space:")
    print(search_space.print_search_space())

    # ── Load surrogate ────────────────────────────────────────────────
    # Model config (d_model, nhead, etc.) is auto-detected from the .json sidecar
    surrogate_ckpt_path = args.checkpoint
    model, norm_stats, surrogate_max_layers = load_surrogate(surrogate_ckpt_path, device)
    print(f"Loaded surrogate from {surrogate_ckpt_path} (max_layers={surrogate_max_layers})")

    # ── Objectives and constraints ────────────────────────────────────
    objs = args.objectives
    if not args.constraint:
        cons = {"params": args.max_params, "val_loss": args.max_val_loss}
    else:
        cons = {k: v for k, v in args.constraint}

    # ── Initialize population ─────────────────────────────────────────
    exp_name = args.exp_name

    if args.resume_ckpt is not None:
        if not os.path.exists(args.resume_ckpt):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_ckpt}")
        logging.info(f"Resuming from checkpoint: {args.resume_ckpt}")
        population = Population.load_checkpoint(args.resume_ckpt, from_pkl=args.resume_ckpt.endswith('.pkl'))
        population.search_space = search_space
        population.objs_settings = objs
        population.cons_settings = cons
        population.print_summary()
    else:
        if args.init_individuals:
            init_path = args.init_individuals
            if not os.path.isabs(init_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                init_path = os.path.join(script_dir, init_path)
            logging.info(f"Initializing population from: {init_path}")
            individuals = load_initial_individuals(init_path)
        else:
            individuals = [search_space.sample() for _ in range(args.pop_size)]

        population = Population(individuals, search_space=search_space, objs_settings=objs, cons_settings=cons)
        population.delete_duplicates()

        # Initial surrogate evaluation
        pred_list = surrogate_eval(
            individuals=population.individuals,
            model=model, norm=norm_stats, device=device,
            max_layers=surrogate_max_layers, batch_size=args.batch_size,
        )
        population.apply_pred_loss(pred_list)
        population.print_summary()
        population.save_checkpoint(f"ckpts/{exp_name}/{run_time}_ckpt_gen0.json")

    # ── NSGA-II parameters ────────────────────────────────────────────
    population.n_population = args.pop_size
    population.n_offspring = args.offspring
    population.crossover_rate = args.crossover_rate
    population.mutation_rate = args.mutation_rate

    # ── Real-eval setup ───────────────────────────────────────────────
    buffer = RealDataBuffer()
    ckpt_dir = f"ckpts/{exp_name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Per-generation results CSV (only when --verbose_log)
    gen_results_path = os.path.join(ckpt_dir, f"{run_time}_gen_results.csv")
    GEN_CSV_FIELDS = ["gen", "offspring_idx", "eval_source", "pred_val_loss", "real_val_loss",
                      "params_M", "flops_K", "mem_bytes", "kv_cache_M"]

    # Accuracy summary CSV: one row per real-eval cycle (aggregate metrics)
    accuracy_log_path = os.path.join(ckpt_dir, f"{run_time}_accuracy_log.csv")
    ACC_CSV_FIELDS = ["gen", "n_samples", "buffer_size", "l1", "spearman_r", "pairwise_acc"]

    # Per-individual comparison CSV: one row per real-trained individual
    comparison_log_path = os.path.join(ckpt_dir, f"{run_time}_pred_vs_real.csv")
    CMP_CSV_FIELDS = ["gen", "source", "idx", "pred_val_loss", "real_val_loss", "error", "params_M"]

    # Real training results JSON: full individual configs + real losses (appended each cycle)
    real_results_path = os.path.join(ckpt_dir, f"{run_time}_real_training_results.json")

    # Write CSV headers
    csv_files = [(accuracy_log_path, ACC_CSV_FIELDS),
                 (comparison_log_path, CMP_CSV_FIELDS)]
    if args.verbose_log:
        csv_files.append((gen_results_path, GEN_CSV_FIELDS))
    for path, fields in csv_files:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    hosts = None
    if args.real_eval_freq > 0:
        hosts = load_hosts_from_file(args.hosts_file)
        logging.info(f"Real eval enabled: every {args.real_eval_freq} gens, {args.real_eval_count} individuals")
        logging.info(f"Loaded {len(hosts)} hosts for real training")

    # ── Main loop ─────────────────────────────────────────────────────
    n_gen = args.generations
    for _ in range(n_gen):
        population.generate_offspring()
        gen = population.gen
        print(f"\n================ Generation {gen} ================\n")

        # Surrogate eval (always, for all offspring)
        pred_list = surrogate_eval(
            individuals=population.offspring,
            model=model, norm=norm_stats, device=device,
            max_layers=surrogate_max_layers, batch_size=args.batch_size,
        )
        population.apply_pred_loss(pred_list)

        # Track which offspring got real-trained this generation
        real_trained_map: Dict[int, float] = {}  # offspring_idx -> real_val_loss

        # Real-training verification cycle
        if args.real_eval_freq > 0 and gen % args.real_eval_freq == 0:
            print(f"\n--- Real-training verification (gen {gen}) ---")

            selects_from_population = args.real_eval_strategy in ("full_population", "pareto_front")

            # 1. Select individuals for real training
            selected_indices = select_for_real_eval(
                offspring_evaluations=population.offspring_evaluations,
                pred_list=pred_list,
                K=args.real_eval_count,
                strategy=args.real_eval_strategy,
                population=population if selects_from_population else None,
            )

            # full_population/pareto_front select from population.individuals; others from offspring
            if selects_from_population:
                source_pool = population.individuals
                source_label = "population"
            else:
                source_pool = population.offspring
                source_label = "offspring"

            selected_individuals = [source_pool[idx] for idx in selected_indices]
            # For population-based strategies: re-run surrogate to get comparable predictions
            if selects_from_population:
                selected_preds = surrogate_eval(
                    individuals=selected_individuals,
                    model=model, norm=norm_stats, device=device,
                    max_layers=surrogate_max_layers, batch_size=args.batch_size,
                )
            else:
                selected_preds = [pred_list[idx] for idx in selected_indices]

            print(f"Selected {len(selected_indices)} {source_label} individuals "
                  f"for real training (strategy={args.real_eval_strategy}): indices {selected_indices}")

            # 2. Run real training
            real_losses = real_train_subset(
                individuals=selected_individuals,
                hosts=hosts,
                user=args.user,
                key_filename=args.key,
                run_dir_name=exp_name,
                max_iters=args.max_iters,
                conda_env=args.conda_env,
                dataset=args.dataset,
                timeout=args.timeout,
            )

            # Build map for offspring-based strategies (for gen_results CSV)
            if not selects_from_population:
                for sel_i, off_idx in enumerate(selected_indices):
                    real_trained_map[off_idx] = real_losses[sel_i]

            # 3. Compare & log accuracy (summary)
            metrics = compute_accuracy_metrics(selected_preds, real_losses)
            print(f"Surrogate accuracy (n={len(selected_indices)}): "
                  f"L1={metrics['l1']:.4f}, Spearman={metrics['spearman_r']:.4f}, "
                  f"PairAcc={metrics['pairwise_acc']:.2%}")

            with open(accuracy_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    gen, len(selected_indices), buffer.size + len(selected_indices),
                    f"{metrics['l1']:.6f}", f"{metrics['spearman_r']:.4f}", f"{metrics['pairwise_acc']:.4f}",
                ])

            # 4. Log per-individual pred vs real comparison
            with open(comparison_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                for sel_i, src_idx in enumerate(selected_indices):
                    ind = source_pool[src_idx]
                    writer.writerow([
                        gen, source_label, src_idx,
                        f"{selected_preds[sel_i]:.6f}",
                        f"{real_losses[sel_i]:.6f}",
                        f"{metrics['per_error'][sel_i]:.6f}",
                        f"{ind.estimate_params() / 1e6:.3f}",
                    ])

            # 5. Save real training results: full individual configs + losses
            real_results_entry = {
                "gen": gen,
                "strategy": args.real_eval_strategy,
                "source": source_label,
                "n_individuals": len(selected_indices),
                "results": [],
            }
            for sel_i, src_idx in enumerate(selected_indices):
                ind = source_pool[src_idx]
                real_results_entry["results"].append({
                    "idx": src_idx,
                    "pred_val_loss": float(selected_preds[sel_i]),
                    "real_val_loss": float(real_losses[sel_i]),
                    "error": float(metrics["per_error"][sel_i]),
                    "individual": dict(ind),  # full architecture config
                })
            # Append to JSON (load existing, append, rewrite)
            if os.path.exists(real_results_path):
                with open(real_results_path, "r") as f:
                    all_real_results = json.load(f)
            else:
                all_real_results = []
            all_real_results.append(real_results_entry)
            with open(real_results_path, "w") as f:
                json.dump(all_real_results, f, indent=2, default=str)

            # 6. Accumulate real data
            buffer.add(selected_individuals, real_losses, max_layers=surrogate_max_layers)
            print(f"Buffer size: {buffer.size} total real data points")

            # 7. Finetune surrogate — overwrite checkpoint in-place so resume picks it up
            model, norm_stats = finetune_surrogate(
                model=model,
                buffer=buffer,
                norm_stats=norm_stats,
                device=device,
                epochs=args.finetune_epochs,
                lr=args.finetune_lr,
                save_path=surrogate_ckpt_path,
            )
            print(f"--- End real-training verification ---\n")

        # Log per-generation results for ALL offspring (only when --verbose_log)
        if args.verbose_log:
            with open(gen_results_path, "a", newline="") as f:
                writer = csv.writer(f)
                for i, ind in enumerate(population.offspring):
                    params = ind.estimate_params() / 1e6
                    flops = ind.estimate_flops() / 1e3
                    mem_bytes = ind.estimate_mem_access()
                    kv_cache = ind.estimate_kv_cache_size() / 1e6
                    real_loss = real_trained_map.get(i)

                    if real_loss is not None:
                        eval_source = "real"
                    else:
                        eval_source = "surrogate"

                    writer.writerow([
                        gen, i, eval_source,
                        f"{pred_list[i]:.6f}",
                        f"{real_loss:.6f}" if real_loss is not None else "",
                        f"{params:.3f}", f"{flops:.3f}", mem_bytes, f"{kv_cache:.3f}",
                    ])

        # Save and advance
        population.save_checkpoint(os.path.join(ckpt_dir, f"{run_time}_ckpt_offspring_gen{gen}.json"))
        population.update_elimination()
        population.save_checkpoint(os.path.join(ckpt_dir, f"{run_time}_ckpt_gen{gen}.json"))

        if gen % 5 == 0 or gen == n_gen - 1:
            population.print_summary()

    print(f"\nSearch complete. {n_gen} generations, buffer={buffer.size} real data points.")
    print(f"Results:    {gen_results_path}")
    print(f"Accuracy:   {accuracy_log_path}")
    print(f"Comparison: {comparison_log_path}")
    population.print_summary()


if __name__ == "__main__":
    main()
