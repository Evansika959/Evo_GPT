#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 compare_ppl.py \
  --model_dir ./morphed_output \
  --output_dir ./qwen3_iha_uptrain \
  --original_model_id Qwen/Qwen3-1.7B \
  --ppl_dtype bfloat16
