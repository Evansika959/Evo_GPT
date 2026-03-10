#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
OUT_DIR="${OUT_DIR:-./morphed_output}"
SCHEDULE="${SCHEDULE:-./examples/schedule.yaml}"
DTYPE="${DTYPE:-bfloat16}"

python3 -m qwen3_morphing.morph_qwen3 \
  --model_id "$MODEL_ID" \
  --out_dir "$OUT_DIR" \
  --schedule "$SCHEDULE" \
  --dtype "$DTYPE"

python3 -m qwen3_morphing.verify_shapes --model_dir "$OUT_DIR" --device auto --dtype "$DTYPE"
