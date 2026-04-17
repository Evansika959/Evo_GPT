#!/bin/bash
# Run both NSGA and standard baseline models sequentially on one host
# For distributed multi-host runs, use run_nsga.sh and run_baselines.sh on separate hosts
# Usage: bash fineweb_baselines/scripts/run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 1: NSGA-evolved models ==="
bash "$SCRIPT_DIR/run_nsga.sh"

echo "=== Phase 2: Standard baselines ==="
bash "$SCRIPT_DIR/run_baselines.sh"

echo "=== All models complete ==="
