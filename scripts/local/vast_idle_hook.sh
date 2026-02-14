#!/usr/bin/env bash
set -euo pipefail

VAST_HOST="${VAST_HOST:-108.55.118.247}"
VAST_PORT="${VAST_PORT:-53346}"
VAST_USER="${VAST_USER:-root}"
REMOTE_REPO="${REMOTE_REPO:-/root/ministral-prefill-mask-ablation}"
REMOTE_SESSION="${REMOTE_SESSION:-prefill-s12}"

CHECK_INTERVAL="${CHECK_INTERVAL:-120}"
LOW_GPU_THRESH="${LOW_GPU_THRESH:-8}"
LOW_CPU_THRESH="${LOW_CPU_THRESH:-15}"
LOW_STREAK_LIMIT="${LOW_STREAK_LIMIT:-3}"

OUT_DIR="${OUT_DIR:-/home/name/Dropbox/ministral-prefill-mask-ablation/runs}"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/local_idle_hook.log"

low_streak=0
last_mtime=0

log_line() {
  local msg="$1"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$ts $msg" | tee -a "$LOG_FILE"
}

query_remote() {
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$VAST_PORT" "$VAST_USER@$VAST_HOST" "
set -e
cd '$REMOTE_REPO'

alive=no
if tmux has-session -t '$REMOTE_SESSION' 2>/dev/null; then alive=yes; fi

s1=0
s2=0
if [ -d runs/stage1_baseline_eval ]; then s1=\$(find runs/stage1_baseline_eval -name metrics.json 2>/dev/null | wc -l); fi
if [ -d runs/stage2_ablation_eval ]; then s2=\$(find runs/stage2_ablation_eval -name metrics.json 2>/dev/null | wc -l); fi

latest=''
if [ -d runs/stage2_ablation_eval ]; then
  latest=\$(ls -1dt runs/stage2_ablation_eval/* 2>/dev/null | head -n1 || true)
fi
if [ -z \"\$latest\" ] && [ -d runs/stage1_baseline_eval ]; then
  latest=\$(ls -1dt runs/stage1_baseline_eval/* 2>/dev/null | head -n1 || true)
fi

mtime=0
if [ -n \"\$latest\" ] && [ -f \"\$latest/log.txt\" ]; then
  mtime=\$(stat -c %Y \"\$latest/log.txt\")
fi

gpu=0
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{s+=\$1; n+=1} END{if(n>0) print int(s/n); else print 0}')
fi

read -r _ u1 n1 s1cpu i1 w1 irq1 si1 st1 _ < /proc/stat
sum1=\$((u1+n1+s1cpu+i1+w1+irq1+si1+st1))
sleep 1
read -r _ u2 n2 s2cpu i2 w2 irq2 si2 st2 _ < /proc/stat
sum2=\$((u2+n2+s2cpu+i2+w2+irq2+si2+st2))
dsum=\$((sum2-sum1))
didle=\$((i2-i1))
if [ \"\$dsum\" -gt 0 ]; then
  cpu=\$((100*(dsum-didle)/dsum))
else
  cpu=0
fi

printf '%s %s %s %s %s %s\n' \"\$alive\" \"\$s1\" \"\$s2\" \"\$gpu\" \"\$cpu\" \"\$mtime\"
"
}

log_line "hook_start host=$VAST_HOST port=$VAST_PORT interval=${CHECK_INTERVAL}s"

while true; do
  read -r alive s1 s2 gpu cpu mtime < <(query_remote)

  stalled=0
  if [[ "$mtime" != "0" && "$last_mtime" != "0" && "$mtime" == "$last_mtime" ]]; then
    stalled=1
  fi

  if [[ "$alive" == "yes" && "$s2" -lt 1 && "$gpu" -le "$LOW_GPU_THRESH" && "$cpu" -le "$LOW_CPU_THRESH" && "$stalled" -eq 1 ]]; then
    low_streak=$((low_streak + 1))
  else
    low_streak=0
  fi

  log_line "alive=$alive stage1_metrics=$s1 stage2_metrics=$s2 gpu=$gpu cpu=$cpu stalled=$stalled low_streak=$low_streak"

  if [[ "$s2" -ge 1 ]]; then
    log_line "run_complete stage2_metrics=$s2"
    printf '\a'
    break
  fi

  if [[ "$alive" != "yes" && "$s2" -lt 1 ]]; then
    log_line "ALERT session_down stage2_incomplete"
    printf '\a'
  fi

  if [[ "$low_streak" -ge "$LOW_STREAK_LIMIT" ]]; then
    log_line "ALERT low_utilization_persistent"
    printf '\a'
    low_streak=0
  fi

  last_mtime="$mtime"
  sleep "$CHECK_INTERVAL"
done
