#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${MODEL_DIR:-./morphed_output}"
OUTPUT_DIR="${OUTPUT_DIR:-./qwen3_iha_uptrain}"
DATASET_NAME="${DATASET_NAME:-Skylion007/openwebtext}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"

python3 uptrain.py \
  --model_dir "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --dataset_split "$DATASET_SPLIT"\
  --max_steps_phase1 10000 \
  --max_steps_phase2 10000 \
  
