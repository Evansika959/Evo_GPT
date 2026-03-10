#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${MODEL_DIR:-./morphed_output}"
OUTPUT_DIR="${OUTPUT_DIR:-./qwen3_iha_uptrain}"
ORIGINAL_MODEL_ID="${ORIGINAL_MODEL_ID:-Qwen/Qwen3-1.7B}"
PPL_DTYPE="${PPL_DTYPE:-bfloat16}"

python3 compare_ppl.py \
  --model_dir "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --original_model_id "$ORIGINAL_MODEL_ID" \
  --ppl_dtype "$PPL_DTYPE"
