#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-causal}"  # causal | prefill_bidir
MODEL_ID="${MODEL_ID:-mistralai/Ministral-3b-instruct}"
DATASET_ID="${DATASET_ID:-yahma/alpaca-cleaned}"
MAX_STEPS="${MAX_STEPS:-1200}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-50000}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-2e-5}"

case "$MODE" in
  causal)
    STAGE_NAME="stage3_finetune_causal"
    EXTRA_FLAGS=()
    ;;
  prefill_bidir)
    STAGE_NAME="stage3_finetune_prefill_bidir"
    EXTRA_FLAGS=(--prefill-bidirectional-train)
    ;;
  *)
    echo "Unknown mode: $MODE. Use causal|prefill_bidir"
    exit 1
    ;;
esac

TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/${STAGE_NAME}/${TS}"
mkdir -p "$RUN_DIR"

echo "[$STAGE_NAME] model=$MODEL_ID dataset=$DATASET_ID max_steps=$MAX_STEPS"
uv run prefill-finetune \
  --model-id "$MODEL_ID" \
  --dataset-id "$DATASET_ID" \
  --output-dir "$RUN_DIR" \
  --max-steps "$MAX_STEPS" \
  --train-samples "$TRAIN_SAMPLES" \
  --eval-samples "$EVAL_SAMPLES" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LR" \
  --gradient-checkpointing \
  "${EXTRA_FLAGS[@]}" \
  2>&1 | tee "$RUN_DIR/log.txt"

echo "[$STAGE_NAME] summary: $RUN_DIR/summary.json"
