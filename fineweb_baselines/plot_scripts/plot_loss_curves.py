#!/usr/bin/env python3
"""Plot training and validation loss curves for baseline experiments."""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path):
    """Parse a training log file for iter/loss data."""
    train_iters, train_losses = [], []
    val_iters, val_losses = [], []

    with open(log_path, "r") as f:
        for line in f:
            # Training loss: "iter N: loss X.XXXX, ..."
            m = re.search(r"iter\s+(\d+):\s+loss\s+([\d.]+)", line)
            if m:
                train_iters.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))

            # Val loss: "val loss X.XXXX" or "best val loss X.XXXX"
            m = re.search(r"step\s+(\d+):\s+.*val loss\s+([\d.]+)", line)
            if m:
                val_iters.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))

    return {
        "train_iters": np.array(train_iters),
        "train_losses": np.array(train_losses),
        "val_iters": np.array(val_iters),
        "val_losses": np.array(val_losses),
    }


def find_experiment_logs(results_dir):
    """Find all experiment log files and label them."""
    experiments = {}
    for log_file in sorted(glob.glob(os.path.join(results_dir, "**", "*.log"), recursive=True)):
        name = os.path.basename(os.path.dirname(log_file))
        if not name:
            name = os.path.splitext(os.path.basename(log_file))[0]
        experiments[name] = log_file

    # Also check for CSV logs
    for csv_file in sorted(glob.glob(os.path.join(results_dir, "**", "*.csv"), recursive=True)):
        name = os.path.basename(os.path.dirname(csv_file))
        if name not in experiments:
            experiments[name] = csv_file

    return experiments


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves for baseline experiments")
    parser.add_argument(
        "--results_dir",
        default="fineweb_baselines/results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output_dir",
        default="fineweb_baselines/plots",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    experiments = find_experiment_logs(args.results_dir)
    if not experiments:
        print(f"No log files found in {args.results_dir}")
        return

    # --- Validation loss comparison ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for name, log_path in experiments.items():
        data = parse_log_file(log_path)
        if len(data["val_iters"]) > 0:
            ax.plot(data["val_iters"], data["val_losses"], label=name, linewidth=1.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss - FineWeb-Edu Baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "val_loss_comparison.png"), dpi=150)
    print(f"Saved: {os.path.join(args.output_dir, 'val_loss_comparison.png')}")

    # --- Training loss comparison ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for name, log_path in experiments.items():
        data = parse_log_file(log_path)
        if len(data["train_iters"]) > 0:
            # Smooth training loss with moving average
            window = min(50, len(data["train_losses"]) // 10 + 1)
            if window > 1:
                smoothed = np.convolve(data["train_losses"], np.ones(window) / window, mode="valid")
                iters = data["train_iters"][window - 1:]
            else:
                smoothed = data["train_losses"]
                iters = data["train_iters"]
            ax.plot(iters, smoothed, label=name, linewidth=1.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Loss (smoothed)")
    ax.set_title("Training Loss - FineWeb-Edu Baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "train_loss_comparison.png"), dpi=150)
    print(f"Saved: {os.path.join(args.output_dir, 'train_loss_comparison.png')}")

    plt.close("all")


if __name__ == "__main__":
    main()
