#!/bin/bash
# Launch a training config on a remote host (detached via nohup, survives SSH disconnect)
# Usage:
#   bash fineweb_baselines/scripts/remote_launch.sh \
#       <user@host> <config_name: nsga|baselines> [ssh_key]
#
# After launch, training runs under nohup on the remote host.
# Monitor via:
#   ssh <user@host> "tail -f ~/Evo_GPT/fineweb_baselines/results_<config>/training.log"

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <user@host> <config: nsga|baselines> [ssh_key]"
    exit 1
fi

REMOTE="$1"
CONFIG="$2"   # 'nsga' or 'baselines'
SSH_KEY="${3:-}"
SSH_OPTS=""
[ -n "$SSH_KEY" ] && SSH_OPTS="-i $SSH_KEY"

if [ "$CONFIG" != "nsga" ] && [ "$CONFIG" != "baselines" ]; then
    echo "Error: config must be 'nsga' or 'baselines'"
    exit 1
fi

REMOTE_REPO="~/Evo_GPT"
REMOTE_SCRIPT="fineweb_baselines/scripts/run_${CONFIG}.sh"
REMOTE_LOG="fineweb_baselines/results_${CONFIG}/training.log"

echo "=== Launching $CONFIG on $REMOTE ==="
ssh $SSH_OPTS "$REMOTE" "
    cd $REMOTE_REPO
    mkdir -p fineweb_baselines/results_${CONFIG}
    nohup bash $REMOTE_SCRIPT > $REMOTE_LOG 2>&1 &
    echo \"Launched PID \$!\"
"

echo ""
echo "=== Monitoring command (run from anywhere with SSH access): ==="
echo "  ssh $SSH_OPTS $REMOTE \"tail -f $REMOTE_REPO/$REMOTE_LOG\""
