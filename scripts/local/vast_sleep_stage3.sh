#!/usr/bin/env bash
set -euo pipefail

# Sleep-safe Stage3 runner:
# - launches a remote supervisor in tmux
# - monitors with adaptive cadence
# - auto-recovers simple stalls by killing stuck trainer process once detected
# - optionally tears down the Vast instance on done/fail

VAST_HOST="${VAST_HOST:-}"
VAST_PORT="${VAST_PORT:-22}"
VAST_USER="${VAST_USER:-root}"
INSTANCE_ID="${INSTANCE_ID:-}"

REMOTE_REPO="${REMOTE_REPO:-/root/ministral-prefill-mask-ablation}"
REMOTE_SESSION="${REMOTE_SESSION:-stage3_autopilot}"
MODE="${MODE:-prefill_bidir}"
REUSE_REMOTE_SESSION="${REUSE_REMOTE_SESSION:-0}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
ATTEMPT_PAUSE_SECS="${ATTEMPT_PAUSE_SECS:-30}"

AUTO_RECOVER_STALL="${AUTO_RECOVER_STALL:-1}"
STALL_CHECKS_THRESHOLD="${STALL_CHECKS_THRESHOLD:-2}"

AUTO_DESTROY_ON_DONE="${AUTO_DESTROY_ON_DONE:-0}"
AUTO_DESTROY_ON_FAIL="${AUTO_DESTROY_ON_FAIL:-1}"

# Adaptive cadence
EARLY_DELAYS_SECS="${EARLY_DELAYS_SECS:-60,120,240,480}"
STABLE_INTERVAL_SECS="${STABLE_INTERVAL_SECS:-1800}"
ALERT_INTERVAL_SECS="${ALERT_INTERVAL_SECS:-180}"

if [ -z "$VAST_HOST" ]; then
  echo "error: set VAST_HOST"
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "error: ssh is required"
  exit 1
fi

if ! command -v vastai >/dev/null 2>&1; then
  echo "error: vastai CLI is required for auto-destroy"
  exit 1
fi

q() { printf '%q' "$1"; }

build_launch_script() {
  local out="$1"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd $(q "$REMOTE_REPO")"

    # Forward commonly tuned training vars if set locally.
    for var in MODEL_ID DATASET_ID MAX_STEPS TRAIN_SAMPLES EVAL_SAMPLES EVAL_STEPS SAVE_STEPS GRAD_ACCUM LR LR_SCHEDULER WARMUP_RATIO AUTO_STOP_PATIENCE_EVALS AUTO_STOP_MIN_DELTA AUTO_STOP_MIN_STEPS; do
      val="${!var:-}"
      if [ -n "$val" ]; then
        echo "export $var=$(q "$val")"
      fi
    done

    echo "export REPO_DIR=$(q "$REMOTE_REPO")"
    echo "export MODE=$(q "$MODE")"
    echo "export MAX_ATTEMPTS=$(q "$MAX_ATTEMPTS")"
    echo "export ATTEMPT_PAUSE_SECS=$(q "$ATTEMPT_PAUSE_SECS")"
    echo "bash scripts/vast/remote_stage3_supervisor.sh"
  } > "$out"
}

push_and_launch_supervisor() {
  local tmp_launch
  tmp_launch="$(mktemp)"
  build_launch_script "$tmp_launch"

  local remote_launch
  remote_launch="$REMOTE_REPO/artifacts/autopilot/launch_supervisor.sh"

  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$VAST_PORT" "$VAST_USER@$VAST_HOST" "mkdir -p $(q "$REMOTE_REPO/artifacts/autopilot")"

  {
    echo "cat > $(q "$remote_launch") <<'EOF_REMOTE_LAUNCH'"
    cat "$tmp_launch"
    echo "EOF_REMOTE_LAUNCH"
    echo "chmod +x $(q "$remote_launch")"
    echo "tmux kill-session -t $(q "$REMOTE_SESSION") >/dev/null 2>&1 || true"
    echo "tmux new-session -d -s $(q "$REMOTE_SESSION") \"bash -lc $(q "$remote_launch")\""
    echo "tmux ls"
  } | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$VAST_PORT" "$VAST_USER@$VAST_HOST" 'bash -s'

  rm -f "$tmp_launch"
}

remote_status_line() {
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$VAST_PORT" "$VAST_USER@$VAST_HOST" "python3 - <<'PY'
from pathlib import Path
import re, subprocess

repo = Path('$REMOTE_REPO')
state_file = repo / 'artifacts' / 'autopilot' / 'state.env'

state = {
    'STATUS': 'booting',
    'ATTEMPT': '0',
    'MAX_ATTEMPTS': '0',
    'CURRENT_RUN': '',
    'LAST_FIX': 'none',
}
if state_file.exists():
    for line in state_file.read_text(errors='ignore').splitlines():
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        state[k] = v

run_rel = state.get('CURRENT_RUN', '')
# Prefer the actively running output-dir when available.
try:
    procs = subprocess.check_output(['pgrep', '-af', 'prefill-finetune'], text=True, timeout=5)
    for line in procs.splitlines():
        m = re.search(r'--output-dir\\s+(\\S+)', line)
        if m:
            run_rel = m.group(1)
            break
except Exception:
    pass

log = None
if run_rel:
    p = Path(run_rel)
    log = p / 'log.txt' if p.is_absolute() else repo / p / 'log.txt'

step = '0'
total = '0'
mtime = '0'
size = '0'
traceback = '0'
done = '0'
if log and log.exists():
    st = log.stat()
    mtime = str(int(st.st_mtime))
    size = str(st.st_size)
    with log.open('rb') as f:
        f.seek(max(0, st.st_size - 300000))
        chunk = f.read().decode(errors='ignore')
    # progress bars include patterns like "755/6000"
    ms = list(re.finditer(r'(\d+)/(\d+)', chunk))
    if ms:
        step = ms[-1].group(1)
        total = ms[-1].group(2)
    traceback = '1' if ('Traceback' in chunk or 'RuntimeError:' in chunk or 'ValueError:' in chunk) else '0'
    done = '1' if '[done]' in chunk else '0'

# GPU snapshot (best effort)
gpu_util = 'na'
gpu_mem = 'na'
try:
    out = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits'
    ], text=True, timeout=5).strip().splitlines()[0]
    u, used, total_mem = [x.strip() for x in out.split(',')]
    gpu_util = u
    if total_mem and total_mem != '0':
        gpu_mem = str(round(float(used) * 100.0 / float(total_mem), 1))
except Exception:
    pass

fields = {
    'STATUS': state.get('STATUS', 'booting'),
    'ATTEMPT': state.get('ATTEMPT', '0'),
    'MAX_ATTEMPTS': state.get('MAX_ATTEMPTS', '0'),
    'RUN': run_rel,
    'STEP': step,
    'TOTAL': total,
    'LOG_MTIME': mtime,
    'LOG_SIZE': size,
    'TRACEBACK': traceback,
    'DONE': done,
    'GPU_UTIL': gpu_util,
    'GPU_MEM_PCT': gpu_mem,
    'LAST_FIX': state.get('LAST_FIX', 'none'),
    'UPDATED_AT': state.get('UPDATED_AT', 'na'),
}
print(' '.join(f'{k}={v}' for k, v in fields.items()))
PY" 2>/dev/null
}

field() {
  local key="$1"
  awk -v k="$key" '{for (i=1;i<=NF;i++) { if ($i ~ ("^"k"=")) { sub("^"k"=", "", $i); print $i; exit } }}'
}

kill_stuck_remote_process() {
  local run_rel="$1"
  [ -z "$run_rel" ] && return 0
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$VAST_PORT" "$VAST_USER@$VAST_HOST" "pkill -f $(q "prefill-finetune.*$run_rel") || true" >/dev/null 2>&1 || true
}

destroy_instance_if_configured() {
  local reason="$1"
  local should_destroy=0
  if [ "$reason" = "done" ] && [ "$AUTO_DESTROY_ON_DONE" = "1" ]; then
    should_destroy=1
  fi
  if [ "$reason" = "failed" ] && [ "$AUTO_DESTROY_ON_FAIL" = "1" ]; then
    should_destroy=1
  fi
  if [ "$should_destroy" -eq 1 ] && [ -n "$INSTANCE_ID" ]; then
    echo "[autopilot] destroying instance $INSTANCE_ID (reason=$reason)"
    vastai destroy instance "$INSTANCE_ID" >/dev/null
  fi
}

IFS=',' read -r -a EARLY_DELAYS <<< "$EARLY_DELAYS_SECS"
early_idx=0

last_step="-1"
last_mtime="0"
stale_checks=0
alert_mode=0

echo "[autopilot] launched remote supervisor session=$REMOTE_SESSION host=$VAST_HOST:$VAST_PORT"

if [ "$REUSE_REMOTE_SESSION" = "1" ]; then
  echo "[autopilot] reusing existing remote session $REMOTE_SESSION"
else
  push_and_launch_supervisor
fi

while true; do
  line="$(remote_status_line || true)"
  if [ -z "$line" ]; then
    line="STATUS=unreachable ATTEMPT=0 MAX_ATTEMPTS=0 RUN= STEP=0 TOTAL=0 LOG_MTIME=0 LOG_SIZE=0 TRACEBACK=1 DONE=0 GPU_UTIL=na GPU_MEM_PCT=na LAST_FIX=none UPDATED_AT=na"
  fi

  status="$(printf '%s\n' "$line" | field STATUS)"
  attempt="$(printf '%s\n' "$line" | field ATTEMPT)"
  max_attempts="$(printf '%s\n' "$line" | field MAX_ATTEMPTS)"
  run_rel="$(printf '%s\n' "$line" | field RUN)"
  step="$(printf '%s\n' "$line" | field STEP)"
  total="$(printf '%s\n' "$line" | field TOTAL)"
  mtime="$(printf '%s\n' "$line" | field LOG_MTIME)"
  traceback_flag="$(printf '%s\n' "$line" | field TRACEBACK)"
  done_flag="$(printf '%s\n' "$line" | field DONE)"
  gpu_util="$(printf '%s\n' "$line" | field GPU_UTIL)"
  gpu_mem="$(printf '%s\n' "$line" | field GPU_MEM_PCT)"
  last_fix="$(printf '%s\n' "$line" | field LAST_FIX)"

  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$ts status=$status attempt=${attempt}/${max_attempts} step=${step}/${total} gpu=${gpu_util}% mem=${gpu_mem}% fix=$last_fix run=${run_rel##*/}"

  if [ "$status" = "done" ] || [ "$done_flag" = "1" ]; then
    destroy_instance_if_configured "done"
    exit 0
  fi

  if [ "$status" = "failed" ] || [ "$status" = "unreachable" ]; then
    destroy_instance_if_configured "failed"
    exit 2
  fi

  # Stall/alert detection while running or retrying.
  alert_mode=0
  if [ "$traceback_flag" = "1" ]; then
    alert_mode=1
  fi

  if [ "$step" = "$last_step" ] || [ "$mtime" = "$last_mtime" ]; then
    stale_checks=$((stale_checks + 1))
  else
    stale_checks=0
  fi

  gpu_low=0
  if [[ "$gpu_util" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    # Only treat log staleness as a real stall when GPU is also mostly idle.
    awk -v u="$gpu_util" 'BEGIN { exit (u <= 10.0 ? 0 : 1) }' && gpu_low=1 || gpu_low=0
  fi

  if [ "$stale_checks" -ge "$STALL_CHECKS_THRESHOLD" ] && [ "$gpu_low" -eq 1 ]; then
    alert_mode=1
    if [ "$AUTO_RECOVER_STALL" = "1" ]; then
      echo "[autopilot] stall detected (checks=$stale_checks); killing stuck trainer process to trigger supervisor retry"
      kill_stuck_remote_process "$run_rel"
      stale_checks=0
    fi
  fi

  last_step="$step"
  last_mtime="$mtime"

  sleep_secs="$STABLE_INTERVAL_SECS"
  if [ "$alert_mode" -eq 1 ]; then
    sleep_secs="$ALERT_INTERVAL_SECS"
  else
    if [ "$early_idx" -lt "${#EARLY_DELAYS[@]}" ]; then
      sleep_secs="${EARLY_DELAYS[$early_idx]}"
      early_idx=$((early_idx + 1))
    fi
  fi

  sleep "$sleep_secs"
done
