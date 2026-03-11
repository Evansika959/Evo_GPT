#!/usr/bin/env bash
# Separate script for generation benchmarks (GSM8K, MBPP, etc.)
# These tasks are much slower than log-likelihood tasks and benefit from
# independent tuning of batch_size, max_gen_toks, and other gen_kwargs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="$(command -v python || true)"
if [[ -z "$DEFAULT_PYTHON" ]]; then
	DEFAULT_PYTHON="$(command -v python3)"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
DEFAULT_TASKS=(
	gsm8k
	# mbpp
)

if [[ -n "${TASKS:-}" ]]; then
	TASKS="$TASKS"
else
	TASKS="$(IFS=,; echo "${DEFAULT_TASKS[*]}")"
fi
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-1024}"
OUTPUT_YAML="${OUTPUT_YAML:-$SCRIPT_DIR/outputs/qwen3_1p7b_0shot_gen_lm_eval.yaml}"

mkdir -p "$(dirname "$OUTPUT_YAML")"

echo "Running LM-eval generation benchmarks"
echo "  model:        $MODEL_ID"
echo "  tasks:        $TASKS"
echo "  batch_size:   $BATCH_SIZE"
echo "  max_gen_toks: $MAX_GEN_TOKS"
echo "  output:       $OUTPUT_YAML"

"$PYTHON_BIN" "$SCRIPT_DIR/run_lm_eval_style.py" \
	--model_dir "$MODEL_ID" \
	--tasks "$TASKS" \
	--num_fewshot "$NUM_FEWSHOT" \
	--batch_size "$BATCH_SIZE" \
	--device "$DEVICE" \
	--dtype "$DTYPE" \
	--output_yaml "$OUTPUT_YAML" \
	--attn_implementation sdpa \
	--gen_kwargs "max_gen_toks=${MAX_GEN_TOKS},do_sample=false" \
	"$@"
