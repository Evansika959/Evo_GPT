#!/bin/bash
# Pull back training results (checkpoints + logs + tensorboard) from a remote host
# Usage: bash fineweb_baselines/scripts/remote_collect.sh \
#            <user@host> <config: nsga|baselines> [ssh_key]

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <user@host> <config: nsga|baselines> [ssh_key]"
    exit 1
fi

REMOTE="$1"
CONFIG="$2"
SSH_KEY="${3:-}"
RSYNC_SSH_OPTS="ssh"
[ -n "$SSH_KEY" ] && RSYNC_SSH_OPTS="ssh -i $SSH_KEY"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

REMOTE_RESULTS="Evo_GPT/fineweb_baselines/results_${CONFIG}/"
LOCAL_DEST="$REPO_DIR/fineweb_baselines/results_${CONFIG}_from_$(echo $REMOTE | tr '@' '_')/"

mkdir -p "$LOCAL_DEST"

echo "=== Pulling results from $REMOTE ($CONFIG) → $LOCAL_DEST ==="
rsync -avz --progress -e "$RSYNC_SSH_OPTS" \
    "$REMOTE:$REMOTE_RESULTS" "$LOCAL_DEST"

echo ""
echo "=== Pulling tensorboard logs ==="
REMOTE_TB="Evo_GPT/logs/"
LOCAL_TB="$REPO_DIR/logs_from_$(echo $REMOTE | tr '@' '_')/"
mkdir -p "$LOCAL_TB"
rsync -avz --progress -e "$RSYNC_SSH_OPTS" \
    "$REMOTE:$REMOTE_TB" "$LOCAL_TB"

echo ""
echo "=== Collection complete ==="
