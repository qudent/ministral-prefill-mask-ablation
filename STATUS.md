# Ministral 3B Prefill Mask Ablation - Status

## Current State
Stage 1 is currently running on Vast and progressing through benchmark tasks (latest observed: ARC-Easy). A local idle-monitor hook is running on the local machine (not on Vast) to alert if the remote session dies or utilization stays low while metrics are still incomplete.

## Active Goals
- [x] Create repo, scripts, and VAST-first workflow
- [x] Provision fresh Vast instance for execution
- [x] Identify accessible Ministral 3B model ID
- [x] Patch model loader for Mistral3 architecture
- [x] Fix datasets/version/cache incompatibilities
- [x] Start local idle-monitor hook (no remote secrets)
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
- Local monitor session: `tmux: vast-idle-hook`
- Local monitor log: `runs/local_idle_hook.log`

## Blockers
- No known blocker at the moment.

## Recent Results
- Corrected model id to `mistralai/Ministral-3-3B-Instruct-2512`.
- Added loader fallback for `Mistral3ForConditionalGeneration`.
- Pinned `datasets` to `<3` and cleared stale HF dataset cache.
- Patched benchmark dataset loading with `trust_remote_code=True`.

## Next Steps
1. Continue periodic health checks while Stage 1 runs.
2. Confirm Stage 1 writes `metrics.json` and Stage 2 starts automatically.
3. Confirm Stage 2 writes `metrics.json`.
4. Compute baseline-vs-ablation macro drop and record it here.
