# Ministral 3B Prefill Mask Ablation - Status

## Current State
Execution is active on Vast.ai using a fresh instance. Initial Stage 1 attempt failed fast due invalid/undownloadable model ID (`mistralai/Ministral-3b-instruct`). Default model has been corrected to `mistralai/Ministral-3-3B-Instruct-2512`, and Stage 1/2 are being restarted with that model.

## Active Goals
- [x] Create repo, scripts, and VAST-first workflow
- [x] Provision fresh Vast instance for execution
- [x] Identify accessible Ministral 3B model ID
- [ ] Execute staged runs on Vast.ai and collect first metrics
  - [ ] Stage 1 baseline metrics
  - [ ] Stage 2 ablated baseline metrics
  - [ ] Stage 3A/3B finetune metrics
  - [ ] Stage 4 recovery comparison

## Runtime Details
- Vast instance ID: `31425306`
- GPU: `1x RTX 3090`
- Label: `prefill-mask-s12`
- SSH endpoint: `ssh://root@108.55.118.247:53346`
- Remote repo path: `/root/ministral-prefill-mask-ablation`

## Blockers
- No technical blocker after model-id correction.

## Recent Results
- Destroyed all existing Vast instances and started from clean state.
- Created new instance and completed environment bootstrap.
- Confirmed failure mode quickly from Stage 1 logs (HF 401 repo-not-found for old model id).
- Updated defaults/configs/docs to `mistralai/Ministral-3-3B-Instruct-2512`.

## Next Steps
1. Pull latest repo on the active instance.
2. Rerun Stage 1 baseline and Stage 2 ablation sequentially.
3. Capture both `metrics.json` files and compute macro-accuracy drop.
4. Rewrite this file with concrete metric values and Stage 3 launch decision.
