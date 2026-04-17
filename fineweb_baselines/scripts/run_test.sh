#!/bin/bash
# Quick 1000-iter test sweep to verify all 5 models start, train, save, and log correctly
# Usage: bash fineweb_baselines/scripts/run_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

python optimization_and_search/run_from_yaml.py \
    --yaml fineweb_baselines/config/baselines.yaml \
    --output_dir fineweb_baselines/results_test \
    --prefix test_baseline \
    --dataset fineweb-edu-sample-10BT \
    --override_args \
        max_iters=1000 \
        batch_size=64 \
        eval_interval=500 \
        eval_iters=20 \
        log_interval=100 \
        learning_rate=3e-4 \
        min_lr=3e-5 \
        decay_lr=true \
        warmup_iters=200 \
        grad_clip=1.0 \
        dropout=0.0 \
        always_save_checkpoint=true
