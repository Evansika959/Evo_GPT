#!/bin/bash
# Train all NSGA-evolved models on FineWeb-Edu-10BT
# Usage: bash fineweb_baselines/scripts/run_nsga.sh [output_suffix]
# Optional arg: suffix to append to results dir (default: "nsga")

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

SUFFIX="${1:-nsga}"
OUTPUT_DIR="fineweb_baselines/results_${SUFFIX}"
mkdir -p "$OUTPUT_DIR"

# Use reallmforge conda env (has plotly, torch, etc.)
export PATH="$HOME/miniconda3/envs/reallmforge/bin:$PATH"

python optimization_and_search/run_from_yaml.py \
    --yaml fineweb_baselines/config/nsga_models.yaml \
    --output_dir "$OUTPUT_DIR" \
    --prefix fineweb_nsga \
    --dataset fineweb-edu-sample-10BT \
    --override_args \
        max_iters=100000 \
        batch_size=64 \
        eval_interval=2500 \
        eval_iters=200 \
        log_interval=100 \
        learning_rate=3e-4 \
        min_lr=3e-5 \
        decay_lr=true \
        warmup_iters=2000 \
        grad_clip=1.0 \
        dropout=0.0 \
        always_save_checkpoint=true
