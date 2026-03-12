#!/usr/bin/env bash
# Separate script for generation benchmarks (GSM8K, MBPP, etc.) on uptrained models.
# These tasks are much slower than log-likelihood tasks and benefit from
# independent tuning of batch_size, max_gen_toks, and other gen_kwargs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

DEFAULT_BENCHMARKS=(
	gsm8k
	# mbpp
)

if [[ -n "${BENCHMARKS:-}" ]]; then
	BENCHMARKS_RAW="$BENCHMARKS"
else
	BENCHMARKS_RAW="$(IFS=,; echo "${DEFAULT_BENCHMARKS[*]}")"
fi

map_task_name() {
	local task="$1"
	case "$task" in
		arc-easy) echo "arc_easy" ;;
		arc-challenge) echo "arc_challenge" ;;
		truthfulqa-mc2) echo "truthfulqa_mc2" ;;
		*) echo "$task" ;;
	esac
}

TASKS=""
IFS=',' read -r -a BENCH_ARR <<< "$BENCHMARKS_RAW"
for item in "${BENCH_ARR[@]}"; do
	trimmed="$(echo "$item" | xargs)"
	[[ -z "$trimmed" ]] && continue
	mapped="$(map_task_name "$trimmed")"
	if [[ -z "$TASKS" ]]; then
		TASKS="$mapped"
	else
		TASKS="$TASKS,$mapped"
	fi
done

if [[ -z "$TASKS" ]]; then
	echo "No valid benchmarks/tasks resolved from BENCHMARKS='$BENCHMARKS_RAW'."
	exit 1
fi

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-1024}"
EXP_NAME="${EXP_NAME:-kvgroup_only}"

UPTRAIN_OUTPUT_DIR="${UPTRAIN_OUTPUT_DIR:-./qwen3_morph_uptrain/${EXP_NAME}}"
UPTRAIN_OUTPUT_DIR="${UPTRAIN_OUTPUT_DIR:-./qwen3_iha_uptrain}"

if [[ -d "$UPTRAIN_OUTPUT_DIR/phase2" && -f "$UPTRAIN_OUTPUT_DIR/phase2/config.json" ]]; then
	UPTRAINED_MODEL_DIR="$UPTRAIN_OUTPUT_DIR/phase2"
elif [[ -d "$UPTRAIN_OUTPUT_DIR" && -f "$UPTRAIN_OUTPUT_DIR/config.json" ]]; then
	UPTRAINED_MODEL_DIR="$UPTRAIN_OUTPUT_DIR"
else
	echo "Could not locate uptrained model checkpoint in '$UPTRAIN_OUTPUT_DIR' (expected phase2/ or direct model dir)."
	exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-./report/benchmark_eval}"
mkdir -p "$RESULTS_DIR"
OUTPUT_YAML="${OUTPUT_YAML:-$RESULTS_DIR/${EXP_NAME}_uptrained_gen_lm_eval_metrics.yaml}"

LIMIT="${LIMIT:-}"
if [[ -z "$LIMIT" && -n "${MAX_EXAMPLES:-}" ]]; then
	LIMIT="$MAX_EXAMPLES"
fi

COMMON_ARGS=(
	--tasks "$TASKS"
	--device "$DEVICE"
	--dtype "$DTYPE"
	--num_fewshot "$NUM_FEWSHOT"
	--batch_size "$BATCH_SIZE"
	--trust_remote_code
	--output_yaml "$OUTPUT_YAML"
	--attn_implementation sdpa
	--gen_kwargs "max_gen_toks=${MAX_GEN_TOKS},do_sample=false"
)

if [[ -n "$LIMIT" ]]; then
	COMMON_ARGS+=(--limit "$LIMIT")
fi

echo "Running LM-eval generation benchmarks for uptrained model: $UPTRAINED_MODEL_DIR"
echo "  experiment:   $EXP_NAME"
echo "  tasks:        $TASKS"
echo "  batch_size:   $BATCH_SIZE"
echo "  max_gen_toks: $MAX_GEN_TOKS"

"$PYTHON_BIN" run_lm_eval_style.py \
	--model_dir "$UPTRAINED_MODEL_DIR" \
	"${COMMON_ARGS[@]}"

echo "Done. Results written to: $OUTPUT_YAML"
