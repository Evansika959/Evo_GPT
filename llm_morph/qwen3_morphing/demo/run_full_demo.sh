#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

./run_morph_demo.sh
./run_uptrain_demo.sh
./run_compare_ppl_demo.sh
