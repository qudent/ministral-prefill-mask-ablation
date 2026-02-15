# Ministral 3B Prefill Mask Ablation - Status

## Current State
Stages 1-4 are complete for Stage 3B (`20260214-233534`).
Stage 5A run (`20260215-104518`) stopped at step 1300 due early-stop policy.
Autopilot scripts were removed; workflow is now agent-driven (Codex babysitting), not script orchestration.

## Active Goals
- [x] Stage 1 baseline eval
- [x] Stage 2 ablated eval
- [x] Stage 3B leakage-safe finetune checkpoint
- [x] Stage 4 post-FT ablated eval
- [x] Non-prefill comparison (Stage 3B vs vanilla)
- [ ] Stage 5 long-run rerun under agent-driven monitoring
- [ ] Stage 5 final eval/report vs estimate

## Confirmed Stage 1-4 Metrics
- Stage 1 macro: `0.6950568562`
- Stage 2 macro (ablated): `0.3558976000`
- Stage 4 macro (ablated, post-FT): `0.4086983278`
- Stage 2 damage vs Stage 1: `-0.3391592562` (retention `0.512x`)
- Stage 4 gain vs Stage 2: `+0.0528007278`
- Stage 4 retention vs Stage 1: `0.5880x`

## Non-Prefill Side Effect
- Vanilla non-prefill macro: `0.6950568562`
- Stage 3B non-prefill macro: `0.6199344482`
- Delta (FT - vanilla): `-0.0751224080`

## Stage 5A (Completed, Early Stop)
- Run dir: `runs/stage3_finetune_prefill_bidir/20260215-104518`
- Config gate:
  - `AUTO_STOP_PATIENCE_EVALS=4`
  - `AUTO_STOP_MIN_DELTA=0.003`
  - `AUTO_STOP_MIN_STEPS=600`
- Outcome:
  - Stopped at step `1300` by design
  - Train runtime: `3731.7643s`
  - Final eval loss: `1.44849`

## Infra State
- Vast instance `31457957` is still running and reachable.
- No active remote tmux training sessions.

## Key Artifacts
- Consolidated report: `docs/RESULTS_STAGE1_TO_STAGE4.md`
- Stage 3B checkpoint: `runs/stage3_finetune_prefill_bidir/20260214-233534/final`
- Stage 5A summary: `runs/stage3_finetune_prefill_bidir/20260215-104518/summary.json`
