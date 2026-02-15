#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/ministral-prefill-mask-ablation}"
MODE="${MODE:-prefill_bidir}"   # causal | prefill_bidir
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
ATTEMPT_PAUSE_SECS="${ATTEMPT_PAUSE_SECS:-30}"
STATE_DIR="${STATE_DIR:-$REPO_DIR/artifacts/autopilot}"
STATE_FILE="$STATE_DIR/state.env"

mkdir -p "$STATE_DIR"

case "$MODE" in
  causal)
    STAGE_SUBDIR="stage3_finetune_causal"
    ;;
  prefill_bidir)
    STAGE_SUBDIR="stage3_finetune_prefill_bidir"
    ;;
  *)
    echo "[autopilot] unknown MODE=$MODE"
    exit 2
    ;;
esac

latest_run() {
  local p
  p="$(ls -1dt "$REPO_DIR/runs/$STAGE_SUBDIR"/* 2>/dev/null | head -n 1 || true)"
  echo "$p"
}

write_state() {
  local status="$1"
  local attempt="$2"
  local current_run="$3"
  local last_exit="$4"
  local last_fix="$5"
  local last_error="$6"
  {
    echo "STATUS=$status"
    echo "ATTEMPT=$attempt"
    echo "MAX_ATTEMPTS=$MAX_ATTEMPTS"
    echo "MODE=$MODE"
    echo "STAGE_SUBDIR=$STAGE_SUBDIR"
    echo "CURRENT_RUN=$current_run"
    echo "LAST_EXIT=$last_exit"
    echo "LAST_FIX=$last_fix"
    echo "LAST_ERROR_B64=$(printf '%s' "$last_error" | base64 -w0)"
    echo "UPDATED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$STATE_FILE"
}

install_uv_if_missing() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  echo "[autopilot] uv missing; installing"
  apt-get update
  apt-get install -y curl
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

install_build_tools_if_needed() {
  if command -v gcc >/dev/null 2>&1; then
    return 0
  fi
  echo "[autopilot] gcc missing; installing build-essential"
  apt-get update
  apt-get install -y build-essential
}

trim_runs_if_disk_low() {
  local avail_kb
  avail_kb="$(df -Pk "$REPO_DIR" | awk 'NR==2{print $4}')"
  # Keep >=20GB free as a simple safety threshold.
  if [ "${avail_kb:-0}" -lt 20971520 ]; then
    echo "[autopilot] low disk; trimming oldest run dirs in $REPO_DIR/runs/$STAGE_SUBDIR"
    ls -1dt "$REPO_DIR/runs/$STAGE_SUBDIR"/* 2>/dev/null | tail -n +6 | xargs -r rm -rf
  fi
}

recover_from_log() {
  local attempt_log="$1"
  local fix="none"

  if rg -q "uv: command not found" "$attempt_log"; then
    install_uv_if_missing
    fix="install_uv"
  fi

  if rg -q "Failed to find C compiler|build-essential|gcc: command not found" "$attempt_log"; then
    install_build_tools_if_needed
    fix="install_build_essential"
  fi

  if rg -q "No space left on device" "$attempt_log"; then
    trim_runs_if_disk_low
    fix="trim_old_runs"
  fi

  if rg -q "CUDA out of memory" "$attempt_log"; then
    if [ -n "${EVAL_SAMPLES:-}" ] && [ "${EVAL_SAMPLES:-0}" -gt 128 ]; then
      export EVAL_SAMPLES="$((EVAL_SAMPLES / 2))"
      fix="halve_eval_samples"
      echo "[autopilot] OOM detected; retrying with EVAL_SAMPLES=$EVAL_SAMPLES"
    else
      fix="oom_no_safe_auto_fix"
    fi
  fi

  echo "$fix"
}

cd "$REPO_DIR"
install_uv_if_missing || true

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  before_run="$(latest_run)"
  attempt_log="$STATE_DIR/attempt_${attempt}.log"
  write_state "running" "$attempt" "$before_run" "" "none" ""

  echo "[autopilot] attempt $attempt/$MAX_ATTEMPTS mode=$MODE"

  set +e
  bash scripts/vast/run_stage3_finetune.sh "$MODE" 2>&1 | tee "$attempt_log"
  cmd_exit="${PIPESTATUS[0]}"
  set -e

  after_run="$(latest_run)"
  current_run="$after_run"
  if [ -z "$current_run" ]; then
    current_run="$before_run"
  fi

  if [ "$cmd_exit" -eq 0 ]; then
    summary_path="$current_run/summary.json"
    if [ -n "$current_run" ] && [ -f "$summary_path" ]; then
      write_state "done" "$attempt" "$current_run" "$cmd_exit" "none" ""
      echo "[autopilot] done run=$current_run"
      exit 0
    fi
    # Successful exit but no summary should still be treated as failure.
    err_msg="command exited 0 but summary.json missing"
    write_state "failed" "$attempt" "$current_run" "$cmd_exit" "none" "$err_msg"
    echo "[autopilot] $err_msg"
    exit 1
  fi

  # Attempt failed: classify and try cheap automatic fixes.
  fix_applied="$(recover_from_log "$attempt_log")"
  err_excerpt="$(tail -n 80 "$attempt_log" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | cut -c1-600)"
  write_state "retrying" "$attempt" "$current_run" "$cmd_exit" "$fix_applied" "$err_excerpt"

  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    break
  fi

  echo "[autopilot] retrying after ${ATTEMPT_PAUSE_SECS}s (fix=$fix_applied)"
  sleep "$ATTEMPT_PAUSE_SECS"
done

last_run="$(latest_run)"
write_state "failed" "$MAX_ATTEMPTS" "$last_run" "1" "none" "max attempts exhausted"
echo "[autopilot] failed after $MAX_ATTEMPTS attempts"
exit 1
