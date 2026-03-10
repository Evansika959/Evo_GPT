#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

RESUME_CHECKPOINT_DIR="${RESUME_CHECKPOINT_DIR:-}"
if [[ -z "$RESUME_CHECKPOINT_DIR" ]]; then
  echo "Please set RESUME_CHECKPOINT_DIR to a checkpoint/model directory to continue from."
  echo "Example: RESUME_CHECKPOINT_DIR=./qwen3_iha_uptrain/phase2 bash qwen3_morphing/demo/run_uptrain_resume_demo.sh"
  exit 1
fi

if [[ ! -f "$RESUME_CHECKPOINT_DIR/config.json" ]]; then
  echo "Checkpoint directory '$RESUME_CHECKPOINT_DIR' does not look like a HF model dir (missing config.json)."
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-./qwen3_iha_uptrain_resume}"
DATASET_NAME="${DATASET_NAME:-Skylion007/openwebtext}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_STEPS_PHASE1="${MAX_STEPS_PHASE1:-10000}"
MAX_STEPS_PHASE2="${MAX_STEPS_PHASE2:-10000}"
DEVICE="${DEVICE:-cuda}"
BF16_FLAG="${BF16_FLAG:---bf16}"
FP16_FLAG="${FP16_FLAG:---no-fp16}"

if [[ "$BF16_FLAG" != "--bf16" && "$BF16_FLAG" != "--no-bf16" ]]; then
  echo "BF16_FLAG must be --bf16 or --no-bf16 (got '$BF16_FLAG')."
  exit 1
fi

if [[ "$FP16_FLAG" != "--fp16" && "$FP16_FLAG" != "--no-fp16" ]]; then
  echo "FP16_FLAG must be --fp16 or --no-fp16 (got '$FP16_FLAG')."
  exit 1
fi

echo "Resuming uptraining from: $RESUME_CHECKPOINT_DIR"
echo "Output directory: $OUTPUT_DIR"

"$PYTHON_BIN" uptrain.py \
  --model_dir "$RESUME_CHECKPOINT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --dataset_split "$DATASET_SPLIT" \
  --max_steps_phase1 "$MAX_STEPS_PHASE1" \
  --max_steps_phase2 "$MAX_STEPS_PHASE2" \
  --device "$DEVICE" \
  "$BF16_FLAG" \
  "$FP16_FLAG" \
  "$@"
