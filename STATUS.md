# Ministral 3B Prefill Mask Ablation - Status

## Current State
Execution is active on Vast.ai but Stage 1 failed again due dataset trust policy (`piqa` requires `trust_remote_code=True` under current datasets stack). Eval loader has now been patched to pass trust for benchmark datasets and run is being restarted.

## Active Goals
- [x] Create repo, scripts, and VAST-first workflow
- [x] Provision fresh Vast instance for execution
- [x] Identify accessible Ministral 3B model ID
- [x] Patch model loader for Mistral3 architecture
- [x] Fix datasets version incompatibility
- [x] Clear incompatible HF dataset cache on runner
- [x] Patch benchmark dataset loading with trust flag
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
- Remote tmux session: `prefill-s12`

## Blockers
- No known blocker after trust-flag patch; pending rerun.

## Recent Results
- Corrected model id to `mistralai/Ministral-3-3B-Instruct-2512`.
- Added loader fallback for `Mistral3ForConditionalGeneration`.
- Pinned `datasets` to `<3` for benchmark loaders.
- Cleared stale HF dataset cache.
- Patched dataset loading calls to `trust_remote_code=True`.

## Next Steps
1. Pull latest code on active instance.
2. Relaunch Stage 1 baseline and Stage 2 ablation.
3. Monitor with periodic sleep-based check-ins and auto-restart on failure.
4. Capture both metrics files and compute macro drop.
