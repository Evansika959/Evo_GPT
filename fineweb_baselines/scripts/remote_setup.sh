#!/bin/bash
# Stage code and dataset to a remote host in preparation for distributed training
# Usage: bash fineweb_baselines/scripts/remote_setup.sh <user@host> [ssh_key]
#
# Assumes:
#   - Remote host has git, rsync, and miniconda with 'reallmforge' env installed
#   - SSH access with key-based auth
#
# What it does:
#   1. git clone/pull Evo_GPT on the remote host
#   2. rsync the tokenized fineweb-edu-sample-10BT data/ (~20GB one-time)
#   3. Verify reallmforge env + plotly+torch import

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <user@host> [ssh_key_path]"
    exit 1
fi

REMOTE="$1"
SSH_KEY="${2:-}"
SSH_OPTS=""
RSYNC_SSH_OPTS="ssh"
if [ -n "$SSH_KEY" ]; then
    SSH_OPTS="-i $SSH_KEY"
    RSYNC_SSH_OPTS="ssh -i $SSH_KEY"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REMOTE_REPO="~/Evo_GPT"
REMOTE_DATA_DIR="$REMOTE_REPO/data/fineweb-edu-sample-10BT"

echo "=== Checking connectivity to $REMOTE ==="
ssh $SSH_OPTS -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" "echo connected"

echo "=== Syncing code (git clone or pull) ==="
ssh $SSH_OPTS "$REMOTE" "
    if [ ! -d $REMOTE_REPO ]; then
        git clone https://github.com/klei22/Evo_GPT.git $REMOTE_REPO
    else
        cd $REMOTE_REPO && git fetch && git reset --hard origin/master
    fi
"

echo "=== Staging FineWeb-Edu data (~20GB, one-time) ==="
ssh $SSH_OPTS "$REMOTE" "mkdir -p $REMOTE_DATA_DIR"
rsync -avz --progress -e "$RSYNC_SSH_OPTS" \
    "$REPO_DIR/data/fineweb-edu-sample-10BT/train.bin" \
    "$REPO_DIR/data/fineweb-edu-sample-10BT/val.bin" \
    "$REPO_DIR/data/fineweb-edu-sample-10BT/meta.pkl" \
    "$REMOTE:$REMOTE_DATA_DIR/"

echo "=== Verifying reallmforge env ==="
ssh $SSH_OPTS "$REMOTE" "
    export PATH=\$HOME/miniconda3/envs/reallmforge/bin:\$PATH
    python -c 'import torch, plotly; print(f\"torch {torch.__version__}, plotly {plotly.__version__}\")' || echo 'ENV CHECK FAILED'
"

echo "=== Setup complete for $REMOTE ==="
