#!/usr/bin/env bash
set -euo pipefail

LIMIT="${LIMIT:-500}"
TASKS="${TASKS:-hellaswag,piqa,arc_easy,arc_challenge,winogrande}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <checkpoint_path_or_model_id> [more_checkpoints...]"
  exit 1
fi

for MODEL_PATH in "$@"; do
  TS="$(date +%Y%m%d-%H%M%S)"
  SAFE_NAME="$(basename "$MODEL_PATH" | tr '/:' '__')"
  RUN_DIR="runs/stage4_postft_eval/${TS}_${SAFE_NAME}"
  mkdir -p "$RUN_DIR"

  echo "[stage4] evaluating $MODEL_PATH"
  uv run prefill-eval \
    --model-id "$MODEL_PATH" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --split validation \
    --length-normalize \
    --prefill-bidirectional \
    --output-json "$RUN_DIR/metrics.json" \
    2>&1 | tee "$RUN_DIR/log.txt"

done
