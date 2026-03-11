#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

EXP_NAME="kvgroup_only"

MODEL_DIR="${MODEL_DIR:-./morphed_trial/${EXP_NAME}/}"
OUTPUT_DIR="${OUTPUT_DIR:-./qwen3_morph_uptrain/${EXP_NAME}/}"
DATASET_NAME="${DATASET_NAME:-JeanKaddour/minipile}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"

python3 uptrain.py \
  --model_dir "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --dataset_split "$DATASET_SPLIT" \
  --max_steps_phase1 10000 \
  --max_steps_phase2 10000

