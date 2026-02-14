# Ministral 3B Prefill Mask Ablation - Status

## Current State
Execution is active on Vast.ai and model loading now works. Stage 1 ran through HellaSwag successfully (500/500, ~0.622 interim accuracy) but then failed on PIQA due `datasets==4.5.0` dropping dataset-script support. Dependency is patched to use `datasets<3` and runs are being restarted.

## Active Goals
- [x] Create repo, scripts, and VAST-first workflow
- [x] Provision fresh Vast instance for execution
- [x] Identify accessible Ministral 3B model ID
- [x] Patch model loader for Mistral3 architecture
- [x] Identify dataset-library incompatibility in stage eval
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
- No blocker after dependency pin; pending rerun.

## Recent Results
- Old model id failed (HF 401), corrected to `mistralai/Ministral-3-3B-Instruct-2512`.
- Mistral3 architecture mismatch fixed via loader fallback to `AutoModelForImageTextToText`.
- Stage 1 now executes model scoring and reached end of HellaSwag.
- Stage 1 failed at PIQA due incompatible `datasets` major version.

## Next Steps
1. Push dependency pin (`datasets>=2.21,<3`) and pull on active instance.
2. `uv sync` on instance to downgrade datasets.
3. Rerun Stage 1 baseline and Stage 2 ablation sequentially.
4. Capture both `metrics.json` files and compute macro-accuracy drop.
