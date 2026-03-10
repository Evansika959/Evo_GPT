#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${MODEL_DIR:-./morphed_output}"
OUTPUT_DIR="${OUTPUT_DIR:-./qwen3_iha_uptrain}"
DATASET_NAME="${DATASET_NAME:-Salesforce/wikitext}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-wikitext-2-raw-v1}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"

python3 uptrain.py \
  --model_dir "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --dataset_config_name "$DATASET_CONFIG_NAME" \
  --dataset_split "$DATASET_SPLIT"
