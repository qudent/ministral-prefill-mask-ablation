# Ministral 3B Prefill Mask Ablation - Status

## Current State
Stage 1/2 are complete and documented. Stage 3B is now being rerun with the corrected objective:
prompt tokens bidirectional, response tokens causal (prefix-LM style). Run is active on Vast as of
2026-02-14 (`runs/stage3_finetune_prefill_bidir/20260214-204823`).

## Active Goals
- [x] Stage 1 baseline eval
- [x] Stage 2 ablated eval
- [x] Stage 1/2 results documentation
- [x] Diagnose prior 3B failure mode (label leakage)
- [ ] Stage 3B rerun with leakage-safe mask (in progress)
- [ ] Stage 4 post-FT ablated eval
- [ ] Recovery ratio comparison vs Stage 1 baseline

## Confirmed Results So Far
- Stage 1 vs Stage 2 macro accuracy: `0.695057 -> 0.355898` (delta `-0.339159`, retention `0.512x`)
- Prior Stage 3A causal run: `runs/stage3_finetune_causal/20260214-201019`
  - `eval_loss=1.3752`, `train_loss=1.3377`, `train_steps_per_second=2.425`
- Prior Stage 3B full-bidir run: `runs/stage3_finetune_prefill_bidir/20260214-201019`
  - `eval_loss=7.0321`, `train_loss=5.4243`, `train_steps_per_second=2.070`
  - Invalid objective due label leakage.

## 3B Leakage Diagnosis
- Direct probe on same weights/sample:
  - `loss_causal ~= 1.023`
  - `loss_prefill_bidir ~= 0.070`
- Interpretation: full bidirectional prefill mask lets tokens directly see future target tokens during SFT,
  creating a degenerate training signal.

## Current 3B Rerun
- Commit: `e98bea1` (adds `--prompt-bidir-response-causal-train` and wires Stage 3B launcher to it)
- Vast instance: `ssh7.vast.ai:33854`
- Remote tmux session: `prefill-s3b`
- Local watcher session: `s3b-watch`
  - Script: `scripts/local/wait_or_crash.sh`
  - Poll cadence: `CHECK_EVERY=3600`, timeout `TIMEOUT_SECS=32400`
- Completion marker: `runs/stage3_finetune_prefill_bidir/.run_complete`

## Artifacts
- Stage 1 metrics: `runs/stage1_baseline_eval/20260214-145148/metrics.json`
- Stage 2 metrics: `runs/stage2_ablation_eval/20260214-145456/metrics.json`
- Stage 1/2 writeup: `docs/RESULTS_STAGE1_STAGE2.md`

## Blockers
- None currently; rerun is executing.

## Next Steps
1. Let Stage 3B rerun finish and record `summary.json`.
2. Run Stage 4 ablated eval on updated Stage 3B checkpoint (and compare to Stage 3A).
3. Compute recovery ratios vs Stage 1 baseline and update docs.
