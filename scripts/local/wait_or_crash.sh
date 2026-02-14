#!/usr/bin/env bash
set -euo pipefail

# Minimal watcher:
# - Blocks in the foreground
# - Returns when timeout is reached, run is complete, or run is crashed

VAST_HOST="${VAST_HOST:-108.55.118.247}"
VAST_PORT="${VAST_PORT:-53346}"
VAST_USER="${VAST_USER:-root}"
REMOTE_REPO="${REMOTE_REPO:-/root/ministral-prefill-mask-ablation}"
REMOTE_SESSION="${REMOTE_SESSION:-prefill-s12}"

TIMEOUT_SECS="${TIMEOUT_SECS:-21600}"   # default 6h
CHECK_EVERY="${CHECK_EVERY:-300}"       # default 5min

start_ts="$(date +%s)"

echo "watching $VAST_USER@$VAST_HOST:$VAST_PORT session=$REMOTE_SESSION timeout=${TIMEOUT_SECS}s"

while true; do
  now_ts="$(date +%s)"
  elapsed="$((now_ts - start_ts))"

  if [ "$elapsed" -ge "$TIMEOUT_SECS" ]; then
    echo "timeout reached (${elapsed}s); exiting without error"
    exit 0
  fi

  # Query remote state. If SSH fails, treat as crash/unreachable.
  if ! state="$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$VAST_PORT" "$VAST_USER@$VAST_HOST" "
    set -e
    cd '$REMOTE_REPO'
    alive=no
    tmux has-session -t '$REMOTE_SESSION' 2>/dev/null && alive=yes
    s2=0
    [ -d runs/stage2_ablation_eval ] && s2=\$(find runs/stage2_ablation_eval -name metrics.json 2>/dev/null | wc -l)
    echo \"\$alive \$s2\"
  " 2>/dev/null)"; then
    echo "crash: cannot reach remote instance/session"
    exit 2
  fi

  alive="$(awk '{print $1}' <<<"$state")"
  s2="$(awk '{print $2}' <<<"$state")"

  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$ts alive=$alive stage2_metrics=$s2 elapsed=${elapsed}s"

  if [ "$s2" -ge 1 ]; then
    echo "complete: stage2 metrics found"
    exit 0
  fi

  if [ "$alive" != "yes" ]; then
    echo "crash: remote tmux session '$REMOTE_SESSION' is down before completion"
    exit 2
  fi

  sleep "$CHECK_EVERY"
done
