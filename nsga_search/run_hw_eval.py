"""Run hardware cost analysis on a population checkpoint file.

Evaluates all individuals through Timeloop (fused and unfused) and outputs
a CSV with per-individual energy, cycles, and architecture summary.

Usage:
    python run_hw_eval.py --ckpt ckpts/infi_search_200M/ckpt_gen50.json
    python run_hw_eval.py --ckpt ckpts/exp/pop.pkl --from_pkl
    python run_hw_eval.py --ckpt ckpts/exp/ckpt.json --no-fused  # unfused only
"""

import argparse
import csv
import json
import os
import pickle
import time
import sys

from hw_exp import eval_individual


def load_population(ckpt_path: str, from_pkl: bool = False):
    """Load individuals and evaluations from a checkpoint file."""
    if from_pkl:
        with open(ckpt_path, "rb") as f:
            pop = pickle.load(f)
        individuals = pop.individuals
        evaluations = pop.evaluations
    else:
        with open(ckpt_path, "r") as f:
            data = json.load(f)
        individuals = data.get("individuals", [])
        evaluations = data.get("evaluations", [])
    return individuals, evaluations


def main():
    parser = argparse.ArgumentParser(description="Run Timeloop hardware cost analysis on a population checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to population checkpoint (.json or .pkl)")
    parser.add_argument("--from_pkl", action="store_true", default=False, help="Load from pickle instead of JSON")
    parser.add_argument("--work_dir", type=str, default="./hw_eval/runs", help="Timeloop working directory (caches results)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: <ckpt>_hw_eval.csv)")
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True, help="Enable fused operation chain model (default: True)")
    args = parser.parse_args()

    if args.output is None:
        stem = os.path.splitext(args.ckpt)[0]
        suffix = "_hw_fused" if args.fused else "_hw_unfused"
        args.output = f"{stem}{suffix}.csv"

    individuals, evaluations = load_population(args.ckpt, args.from_pkl)
    print(f"Loaded {len(individuals)} individuals from {args.ckpt}")
    print(f"Mode: {'fused' if args.fused else 'unfused'}")
    print(f"Output: {args.output}")
    print()

    # CSV fields
    fields = [
        "idx", "n_layers", "params_M", "val_loss",
        "energy_uJ", "energy_per_token_uJ", "cycles", "cycles_per_token",
        "token_delay_s", "edp", "utilization_pct",
        "fusion_saved_energy_uJ", "fusion_saved_cycles",
        "total_ops", "total_memory_accesses",
    ]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()

        for i, ind in enumerate(individuals):
            # Get val_loss and params from stored evaluations if available
            val_loss = None
            params_M = None
            if evaluations and i < len(evaluations):
                ev = evaluations[i]
                aux = ev.get("aux", ev) if isinstance(ev, dict) else getattr(ev, "aux", {})
                val_loss = aux.get("val_loss")
                params_M = aux.get("params")

            # Count active layers
            mask = ind["globals"].get("layer_mask", [True] * len(ind["layers"]))
            n_layers = sum(1 for m in mask if m)

            t0 = time.time()
            try:
                stats = eval_individual(ind, work_dir=args.work_dir, fused=args.fused)
            except Exception as e:
                print(f"  [{i+1}/{len(individuals)}] FAILED: {e}")
                continue
            elapsed = time.time() - t0

            row = {
                "idx": i,
                "n_layers": n_layers,
                "params_M": f"{params_M:.3f}" if params_M is not None else "",
                "val_loss": f"{val_loss:.6f}" if val_loss is not None else "",
                "energy_uJ": f"{stats['energy_uJ']:.2f}" if stats.get("energy_uJ") is not None else "",
                "energy_per_token_uJ": f"{stats['energy_per_token_uJ']:.4f}" if stats.get("energy_per_token_uJ") is not None else "",
                "cycles": f"{stats['cycles']:.0f}" if stats.get("cycles") is not None else "",
                "cycles_per_token": f"{stats['cycles_per_token']:.2f}" if stats.get("cycles_per_token") is not None else "",
                "token_delay_s": f"{stats['token_delay']:.12f}" if stats.get("token_delay") is not None else "",
                "edp": f"{stats['edp']:.4f}" if stats.get("edp") is not None else "",
                "utilization_pct": f"{stats['utilization_pct']:.2f}" if stats.get("utilization_pct") is not None else "",
                "fusion_saved_energy_uJ": f"{stats.get('fusion_saved_energy_uJ', 0):.2f}",
                "fusion_saved_cycles": f"{stats.get('fusion_saved_cycles', 0):.0f}",
                "total_ops": f"{stats['total_ops']:.0f}" if stats.get("total_ops") is not None else "",
                "total_memory_accesses": f"{stats['total_memory_accesses']:.0f}" if stats.get("total_memory_accesses") is not None else "",
            }
            writer.writerow(row)
            f.flush()

            e_tok = stats.get("energy_per_token_uJ", 0) or 0
            saved = stats.get("fusion_saved_energy_uJ", 0) or 0
            print(f"  [{i+1}/{len(individuals)}] layers={n_layers:2d}  params={params_M or 0:.1f}M  "
                  f"E/tok={e_tok:.1f} uJ  saved={saved:.1f} uJ  ({elapsed:.1f}s)")

    print(f"\nDone. Results written to {args.output}")


if __name__ == "__main__":
    main()
