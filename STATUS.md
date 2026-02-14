# Ministral 3B Prefill Mask Ablation - Status

## Current State
Execution has started on Vast.ai. All previous instances were destroyed per request, a fresh instance was created, and Stage 1 baseline evaluation is currently running in a remote tmux session. Stage 2 is queued to run immediately after Stage 1 in the same session.

## Active Goals
- [x] Create repo, scripts, and VAST-first workflow
- [x] Confirm model id (`mistralai/Ministral-3b-instruct`)
- [x] Provision fresh Vast instance for execution
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
- Remote tmux session: `prefill-s12`
- Remote repo path: `/root/ministral-prefill-mask-ablation`
- Current command chain:
  1. `bash scripts/vast/bootstrap.sh`
  2. `bash scripts/vast/run_stage1_baseline_eval.sh`
  3. `bash scripts/vast/run_stage2_ablation_eval.sh`

## Blockers
- No current blocker. Waiting for Stage 1/2 completion and metric files.

## Recent Results
- Destroyed existing Vast instances and verified empty slate.
- Created fresh instance from offer and confirmed GPU visibility via `nvidia-smi`.
- Started bootstrap + Stage 1/2 pipeline in persistent remote tmux.

## Next Steps
1. Monitor Stage 1 until `metrics.json` is written.
2. Let Stage 2 run automatically and capture its `metrics.json`.
3. Compute baseline vs ablated macro drop and update this file with concrete numbers.
4. Launch Stage 3A and Stage 3B in parallel after Stage 1/2 summary.
