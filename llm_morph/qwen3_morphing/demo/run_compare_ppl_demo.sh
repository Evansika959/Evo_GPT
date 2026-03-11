#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

EXP_NAME="${EXP_NAME:-kvgroup_only}"
MODEL_DIR="${MODEL_DIR:-./morphed_trial/${EXP_NAME}/}"
OUTPUT_DIR="${OUTPUT_DIR:-./qwen3_morph_uptrain/${EXP_NAME}/}"
ORIGINAL_MODEL_ID="${ORIGINAL_MODEL_ID:-Qwen/Qwen3-1.7B}"
PPL_DTYPE="${PPL_DTYPE:-bfloat16}"

echo "Experiment name: $EXP_NAME"
echo "Model dir: $MODEL_DIR"
echo "Output dir: $OUTPUT_DIR"

python3 compare_ppl.py \
  --model_dir "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --original_model_id "$ORIGINAL_MODEL_ID" \
  --ppl_dtype "$PPL_DTYPE"
