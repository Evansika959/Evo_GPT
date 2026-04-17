#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"
export PATH="$HOME/miniconda3/envs/reallmforge/bin:$PATH"
OUTPUT_DIR="fineweb_baselines/results_nsga_best3_best4"
mkdir -p "$OUTPUT_DIR"
python optimization_and_search/run_from_yaml.py \
    --yaml fineweb_baselines/config/host48_nsga_best3_best4.yaml \
    --output_dir "$OUTPUT_DIR" \
    --prefix fineweb_nsga_b34 \
    --dataset fineweb-edu-sample-10BT \
    --override_args \
        max_iters=100000 batch_size=64 eval_interval=2500 eval_iters=200 log_interval=100 \
        learning_rate=3e-4 min_lr=3e-5 decay_lr=true warmup_iters=2000 \
        grad_clip=1.0 dropout=0.0 always_save_checkpoint=true
