# Ministral 3B Prefill Mask Ablation - Status

## Current State
Stage 1/2 are complete and documented. Stage 3B is being rerun with leakage-safe objective
(prompt bidirectional, response causal). First rerun attempt crashed at step-save; relaunch is in progress
with intermediate checkpoint saves disabled by default. Current rerun is healthy and past the prior failure point.

## Active Goals
- [x] Stage 1 baseline eval
- [x] Stage 2 ablated eval
- [x] Stage 1/2 results documentation
- [x] Diagnose prior 3B failure mode (label leakage)
- [ ] Stage 3B rerun with leakage-safe mask
- [ ] Stage 4 post-FT ablated eval
- [ ] Recovery ratio comparison vs Stage 1 baseline

## Confirmed Results
- Stage 1 vs Stage 2 macro accuracy: `0.695057 -> 0.355898` (delta `-0.339159`, retention `0.512x`)
- Prior Stage 3A causal run: `runs/stage3_finetune_causal/20260214-201019`
  - `eval_loss=1.3752`, `train_loss=1.3377`, `train_steps_per_second=2.425`
- Prior Stage 3B full-bidir run (invalid objective): `runs/stage3_finetune_prefill_bidir/20260214-201019`
  - `eval_loss=7.0321`, `train_loss=5.4243`

## 3B Diagnosis
- Full-bidir SFT leaks labels (`loss_prefill_bidir ~= 0.070` vs `loss_causal ~= 1.023` on same sample/weights).
- Prefix-LM objective is now implemented via `--prompt-bidir-response-causal-train`.
- First prefix-LM rerun (`runs/stage3_finetune_prefill_bidir/20260214-204823`) crashed at step `400/1200`
  during Trainer step-checkpoint save (`NotImplementedError` in `save_pretrained` reverse conversion).

## Current Infra State
- Active Vast instance: `ssh7.vast.ai:33854` (`31433854`)
- Launcher fix: `scripts/vast/run_stage3_finetune.sh` now defaults `SAVE_STEPS` to `MAX_STEPS+1`
  to avoid mid-run Trainer step-saves (final save uses project fallback logic).
- Active rerun: `runs/stage3_finetune_prefill_bidir/20260214-213740` (training steps started; no traceback).
- Milestone check at `2026-02-14T22:09:50Z`: reached `step 427` (past prior crash at `400`) with tmux alive and no traceback.
- Latest check (`2026-02-14`): `step 718`, `traceback=no`, remote tmux alive.
- Completion marker path: `runs/stage3_finetune_prefill_bidir/.run_complete`
- Monitoring mode: no local watcher tmux; run `scripts/local/wait_or_crash.sh` directly from this instance when a blocking wait is needed.

## Artifacts
- Stage 1 metrics: `runs/stage1_baseline_eval/20260214-145148/metrics.json`
- Stage 2 metrics: `runs/stage2_ablation_eval/20260214-145456/metrics.json`
- Stage 1/2 writeup: `docs/RESULTS_STAGE1_STAGE2.md`
- Failed rerun log: `runs/stage3_finetune_prefill_bidir/20260214-204823/log.txt`

## Next Steps
1. Let Stage 3B rerun finish and verify `summary.json` + `final/`.
2. Run Stage 4 ablated eval on Stage 3A and new Stage 3B checkpoints.
3. Compute recovery ratios vs Stage 1 baseline and update docs.
