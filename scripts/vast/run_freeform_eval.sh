#!/usr/bin/env bash
# Run free-form evaluation on Vast instance.
# Downloads the finetuned checkpoint from HuggingFace, then generates
# responses for all 4 configs (vanilla, vanilla-ablated, finetuned, finetuned-ablated).
#
# Usage: bash scripts/vast/run_freeform_eval.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_DIR"

HF_CHECKPOINT="${HF_CHECKPOINT:-di2ox3/ministral-prefill-mask-ablation}"
CHECKPOINT_SUBDIR="${CHECKPOINT_SUBDIR:-stage3b-20260214-233534}"
LOCAL_CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR:-checkpoints/${CHECKPOINT_SUBDIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/freeform}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

echo "=== Free-form evaluation ==="
echo "HF checkpoint: ${HF_CHECKPOINT}/${CHECKPOINT_SUBDIR}"
echo "Local checkpoint dir: ${LOCAL_CHECKPOINT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

# Download checkpoint from HuggingFace if not already present
if [ ! -f "${LOCAL_CHECKPOINT_DIR}/pytorch_model.bin" ]; then
    echo "Downloading checkpoint from HuggingFace..."
    mkdir -p "${LOCAL_CHECKPOINT_DIR}"
    # Use huggingface-hub CLI to download the subfolder
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_CHECKPOINT}',
    allow_patterns='${CHECKPOINT_SUBDIR}/*',
    local_dir='checkpoints/_tmp',
)
"
    # Move files from subfolder to expected location
    mv checkpoints/_tmp/${CHECKPOINT_SUBDIR}/* "${LOCAL_CHECKPOINT_DIR}/"
    rm -rf checkpoints/_tmp
    echo "Checkpoint downloaded to ${LOCAL_CHECKPOINT_DIR}"
    ls -la "${LOCAL_CHECKPOINT_DIR}/"
else
    echo "Checkpoint already present at ${LOCAL_CHECKPOINT_DIR}"
fi

# Run evaluation
echo ""
echo "Starting generation for all 4 configs..."
uv run prefill-freeform-eval \
    --checkpoint "${LOCAL_CHECKPOINT_DIR}" \
    --dataset data/eval_freeform.json \
    --output-dir "${OUTPUT_DIR}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device cuda \
    2>&1 | tee "${OUTPUT_DIR}/generation_log.txt"

echo ""
echo "=== Generation complete ==="
echo "Results in ${OUTPUT_DIR}/"
ls -la "${OUTPUT_DIR}/"
