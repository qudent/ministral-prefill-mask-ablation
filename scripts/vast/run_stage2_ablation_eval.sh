#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-mistralai/Ministral-3b-instruct}"
LIMIT="${LIMIT:-500}"
TASKS="${TASKS:-hellaswag,piqa,arc_easy,arc_challenge,winogrande}"

TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/stage2_ablation_eval/${TS}"
mkdir -p "$RUN_DIR"

echo "[stage2] model=$MODEL_ID limit=$LIMIT tasks=$TASKS"
uv run prefill-eval \
  --model-id "$MODEL_ID" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  --split validation \
  --length-normalize \
  --prefill-bidirectional \
  --output-json "$RUN_DIR/metrics.json" \
  2>&1 | tee "$RUN_DIR/log.txt"

echo "[stage2] metrics: $RUN_DIR/metrics.json"
