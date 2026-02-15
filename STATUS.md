# Ministral 3B Prefill Mask Ablation - Status

## Current State
Stages 1-4 are complete for Stage 3B (`20260214-233534`).
Stage 5 long-run finetuning with auto-stop is active on Vast A100 and has passed first evaluations.

## Active Goals
- [x] Stage 1 baseline eval
- [x] Stage 2 ablated eval
- [x] Stage 3B leakage-safe finetune checkpoint
- [x] Stage 4 post-FT ablated eval
- [x] Non-prefill comparison (Stage 3B vs vanilla)
- [ ] Stage 5 long-run finetune with automatic stop on eval-loss plateau
- [ ] Stage 5 result summary (loss trajectory, stop step, runtime, downstream deltas)

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

## Stage 5 Live Run
- Vast instance: `31457957` (`A100-SXM4-40GB`, ssh `ssh6.vast.ai:17956`)
- Remote tmux session: `stage5_long`
- Run dir: `runs/stage3_finetune_prefill_bidir/20260215-104518`
- Config:
  - `MAX_STEPS=6000`
  - `TRAIN_SAMPLES=50000`, `EVAL_SAMPLES=250`, `EVAL_STEPS=100`
  - `LR=2e-5`, `LR_SCHEDULER=constant_with_warmup`, `WARMUP_RATIO=0.03`
  - `AUTO_STOP_PATIENCE_EVALS=4`, `AUTO_STOP_MIN_DELTA=0.003`, `AUTO_STOP_MIN_STEPS=600`

Latest observed run metrics:
- Progress: `457/6000` at `21:47` elapsed (`~2.82 s/step`)
- Eval @200: `eval_loss=1.118`, `eval_runtime=7.737s`
- Eval @300: `eval_loss=1.372`, `eval_runtime=7.699s`
- Eval @400: `eval_loss=1.400`, `eval_runtime=7.736s`
- Auto-stop not yet active by design before step 600.

## Runtime/$ Pre-Estimate (Live-Calibrated)
Using observed effective throughput on current run (`~0.349 step/s`).
Target horizon shown: 6000 steps.

- A100 current (`$0.7411/h`): wall `4.58 / 5.38 / 8.56 h` (opt/base/pess), cost `$3.39 / $3.98 / $6.34`
- A100 cheaper offer (`$0.6676/h`): wall `4.96 / 5.80 / 9.17 h`, cost `$3.31 / $3.87 / $6.12` (migration risk)
- H100 PCIe (`$1.1347/h`): wall `4.48 / 5.18 / 7.96 h`, cost `$5.08 / $5.87 / $9.03` (higher setup risk)

## Checkpoints on HuggingFace Hub
- Repo: [`di2ox3/ministral-prefill-mask-ablation`](https://huggingface.co/di2ox3/ministral-prefill-mask-ablation) (public)
- Stage 3B checkpoint: `stage3b-20260214-233534/` on HF
- Local checkpoint copies deleted to save ~14GB.
- Training script auto-uploads via `--hf-repo-id` after each run.
- Vast instances need `HF_TOKEN` env var for uploads; downloads are unauthenticated (public repo).

## Infra State
- Active Vast instance: `31457957` only.
- Idle H100 instance was terminated: `31457612`.

## Key Artifacts
- Consolidated report: `docs/RESULTS_STAGE1_TO_STAGE4.md`
- Stage 3B checkpoint: HF `di2ox3/ministral-prefill-mask-ablation/stage3b-20260214-233534`
- Stage 4 metrics: `runs/stage4_postft_eval/20260215-093750_final/metrics.json`
- Stage 5 live log:
  - `/root/ministral-prefill-mask-ablation/runs/stage3_finetune_prefill_bidir/20260215-104518/log.txt`
