#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 uptrain.py \
  --model_dir ./morphed_output \
  --output_dir ./qwen3_iha_uptrain
