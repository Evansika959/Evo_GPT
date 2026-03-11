#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="$(command -v python || true)"
if [[ -z "$DEFAULT_PYTHON" ]]; then
	DEFAULT_PYTHON="$(command -v python3)"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
DEFAULT_TASKS=(
	hellaswag
	arc_easy
	arc_challenge
	sciq
	winogrande
	boolq
)

if [[ -n "${TASKS:-}" ]]; then
	TASKS="$TASKS"
else
	TASKS="$(IFS=,; echo "${DEFAULT_TASKS[*]}")"
fi
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_YAML="${OUTPUT_YAML:-$SCRIPT_DIR/outputs/qwen3_1p7b_0shot_lm_eval.yaml}"

mkdir -p "$(dirname "$OUTPUT_YAML")"

echo "Running LM-eval style benchmark"
echo "  model:  $MODEL_ID"
echo "  tasks:  $TASKS"
echo "  output: $OUTPUT_YAML"

"$PYTHON_BIN" "$SCRIPT_DIR/run_lm_eval_style.py" \
	--model_dir "$MODEL_ID" \
	--tasks "$TASKS" \
	--num_fewshot "$NUM_FEWSHOT" \
	--batch_size "$BATCH_SIZE" \
	--device "$DEVICE" \
	--dtype "$DTYPE" \
	--output_yaml "$OUTPUT_YAML" \
	"$@"

