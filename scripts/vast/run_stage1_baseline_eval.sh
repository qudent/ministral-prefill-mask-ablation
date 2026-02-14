#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-mistralai/Ministral-3-3B-Instruct-2512}"
LIMIT="${LIMIT:-500}"
TASKS="${TASKS:-hellaswag,piqa,arc_easy,arc_challenge,winogrande}"

TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/stage1_baseline_eval/${TS}"
mkdir -p "$RUN_DIR"

echo "[stage1] model=$MODEL_ID limit=$LIMIT tasks=$TASKS"
uv run prefill-eval \
  --model-id "$MODEL_ID" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  --split validation \
  --length-normalize \
  --output-json "$RUN_DIR/metrics.json" \
  2>&1 | tee "$RUN_DIR/log.txt"

echo "[stage1] metrics: $RUN_DIR/metrics.json"
