#!/usr/bin/env bash
set -euo pipefail

# Minimal watcher:
# - Blocks in the foreground
# - Returns when timeout is reached, run is complete, or run is crashed

VAST_HOST="${VAST_HOST:-}"
VAST_PORT="${VAST_PORT:-22}"
VAST_USER="${VAST_USER:-root}"
REMOTE_REPO="${REMOTE_REPO:-/root/ministral-prefill-mask-ablation}"
REMOTE_SESSION="${REMOTE_SESSION:-}"
COMPLETE_FILE="${COMPLETE_FILE:-}"

TIMEOUT_SECS="${TIMEOUT_SECS:-21600}"   # default 6h
CHECK_EVERY="${CHECK_EVERY:-300}"       # default 5min

if [ -z "$VAST_HOST" ]; then
  echo "error: set VAST_HOST"
  exit 1
fi

if [ -z "$REMOTE_SESSION" ] && [ -z "$COMPLETE_FILE" ]; then
  echo "error: set REMOTE_SESSION and/or COMPLETE_FILE"
  exit 1
fi

start_ts="$(date +%s)"

echo "watching $VAST_USER@$VAST_HOST:$VAST_PORT repo=$REMOTE_REPO session=${REMOTE_SESSION:-none} complete_file=${COMPLETE_FILE:-none} timeout=${TIMEOUT_SECS}s"

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
    alive=na
    done=na
    if [ -n '$REMOTE_SESSION' ]; then
      alive=no
      tmux has-session -t '$REMOTE_SESSION' 2>/dev/null && alive=yes
    fi
    if [ -n '$COMPLETE_FILE' ]; then
      done=no
      [ -f '$COMPLETE_FILE' ] && done=yes
    fi
    echo \"\$alive \$done\"
  " 2>/dev/null)"; then
    echo "crash: cannot reach remote instance/session"
    exit 2
  fi

  alive="$(awk '{print $1}' <<<"$state")"
  done="$(awk '{print $2}' <<<"$state")"

  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$ts alive=$alive done=$done elapsed=${elapsed}s"

  if [ "$done" = "yes" ]; then
    echo "complete: completion file found"
    exit 0
  fi

  if [ "$alive" = "no" ]; then
    echo "crash: remote tmux session '$REMOTE_SESSION' is down before completion"
    exit 2
  fi

  sleep "$CHECK_EVERY"
done
