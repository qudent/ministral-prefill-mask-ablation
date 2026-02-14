# Ministral 3B Prefill Mask Ablation - Status

## Current State
Stage 1 baseline evaluation has completed successfully and Stage 2 ablation evaluation is currently running on Vast. A local event-style idle hook is active and checks remote health every 2 minutes without storing any secrets on the Vast instance.

## Active Goals
- [x] Create repo, scripts, and VAST-first workflow
- [x] Provision fresh Vast instance for execution
- [x] Identify accessible Ministral 3B model ID
- [x] Patch model loader for Mistral3 architecture
- [x] Fix datasets/version/cache incompatibilities
- [x] Start local idle-monitor hook (no remote secrets)
- [x] Stage 1 baseline metrics
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
- No known blocker currently.

## Recent Results
- Stage 1 run completed and produced baseline metrics artifact.
- Stage 2 started automatically after Stage 1 completion.
- Local idle hook observed transition `stage1_metrics: 0 -> 1` and continues monitoring.

## Next Steps
1. Wait for Stage 2 to write `metrics.json`.
2. Compute baseline-vs-ablation macro accuracy drop.
3. Decide whether to launch Stage 3A/3B immediately or adjust setup first.
