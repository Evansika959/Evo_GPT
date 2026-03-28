#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_BENCHMARKS=(
	hellaswag
	arc-easy
	arc-challenge
	sciq
	winogrande
	boolq
)

if [[ -n "${BENCHMARKS:-}" ]]; then
	BENCHMARKS="$BENCHMARKS"
else
	BENCHMARKS="$(IFS=,; echo "${DEFAULT_BENCHMARKS[*]}")"
fi
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
SPLIT="${SPLIT:-validation}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
LENGTH_NORM="${LENGTH_NORM:---length_norm}"

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

COMMON_ARGS=(
	--benchmarks "$BENCHMARKS"
	--device "$DEVICE"
	--dtype "$DTYPE"
	--split "$SPLIT"
	--trust_remote_code
)

if [[ -n "$MAX_EXAMPLES" ]]; then
	COMMON_ARGS+=(--max_examples "$MAX_EXAMPLES")
fi

if [[ "$LENGTH_NORM" == "--length_norm" || "$LENGTH_NORM" == "--no-length_norm" ]]; then
	COMMON_ARGS+=("$LENGTH_NORM")
else
	echo "LENGTH_NORM must be --length_norm or --no-length_norm (got '$LENGTH_NORM')."
	exit 1
fi

echo "Running benchmark eval for uptrained model: $UPTRAINED_MODEL_DIR"
"$PYTHON_BIN" run_benchmark_eval.py \
	--model_dir "$UPTRAINED_MODEL_DIR" \
	--output_json "$RESULTS_DIR/uptrained_metrics.json" \
	"${COMMON_ARGS[@]}"

echo "Done. Results written to: $RESULTS_DIR"
