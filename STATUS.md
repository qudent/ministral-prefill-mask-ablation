# Ministral 3B Prefill Mask Ablation - Status

## Current State
Stages 1-4 are complete for Stage 3B (`20260214-233534`).
Stage 5A run (`20260215-104518`) ended early at step 1300 due configured auto-stop thresholds.
Stage 5B is now running under unattended autopilot supervision with adaptive monitoring and bounded retries.

## Active Goals
- [x] Stage 1 baseline eval
- [x] Stage 2 ablated eval
- [x] Stage 3B leakage-safe finetune checkpoint
- [x] Stage 4 post-FT ablated eval
- [x] Non-prefill comparison (Stage 3B vs vanilla)
- [ ] Stage 5B long-run finetune completion (autopilot)
- [ ] Stage 5B final eval/report vs estimate

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
  - Stopped at step `1300` by design (not crash)
  - Train runtime: `3731.7643s`
  - Final eval loss: `1.44849`

## Stage 5B (Active, Autopilot)
- Vast instance: `31457957` (`A100-SXM4-40GB`, ssh `ssh6.vast.ai:17956`)
- Remote supervisor session: `stage3_autopilot`
- Local adaptive watcher session: `sleep_autopilot`
- Active run dir: `runs/stage3_finetune_prefill_bidir/20260215-121858`
- Config highlights:
  - `MAX_STEPS=6000`
  - `EVAL_SAMPLES=500`, `EVAL_STEPS=100`
  - `AUTO_STOP_PATIENCE_EVALS=8`
  - `AUTO_STOP_MIN_DELTA=0.001`
  - `AUTO_STOP_MIN_STEPS=3000`
- Autopilot teardown policy:
  - `AUTO_DESTROY_ON_DONE=1`
  - `AUTO_DESTROY_ON_FAIL=1`

## New Unattended Execution Framework
Implemented:
- `scripts/vast/remote_stage3_supervisor.sh`
  - bounded retries + cheap automatic fixes
  - state at `artifacts/autopilot/state.env`
- `scripts/local/vast_sleep_stage3.sh`
  - adaptive check cadence (dense early, sparse stable, alert escalation)
  - optional stall recovery + teardown
- `docs/VAST_AUTOPILOT_ALGORITHM.md`
- `docs/VAST_RUNBOOK.md` updated with sleep mode

## Infra State
- Active Vast instance: `31457957` only.
- Prior idle H100 terminated: `31457612`.

## Key Artifacts
- Consolidated report: `docs/RESULTS_STAGE1_TO_STAGE4.md`
- Stage 3B checkpoint: `runs/stage3_finetune_prefill_bidir/20260214-233534/final`
- Stage 5A summary: `runs/stage3_finetune_prefill_bidir/20260215-104518/summary.json`
- Stage 5B log:
  - `/root/ministral-prefill-mask-ablation/runs/stage3_finetune_prefill_bidir/20260215-121858/log.txt`
