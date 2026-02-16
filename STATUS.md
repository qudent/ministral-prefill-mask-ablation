# Ministral 3B Prefill Mask Ablation - Status

## Current State
**Experiment complete.** Stages 1-4 plus intensive fine-tuning (Stage 5) done.
More training does NOT improve prefill mask ablation recovery beyond ~0.41 macro.

## All Results

| Model | Steps | Epochs | Eval Loss | Non-ablated | Ablated | Notes |
|-------|-------|--------|-----------|-------------|---------|-------|
| Vanilla | 0 | 0 | — | 0.695 | 0.356 | baseline |
| Stage 3B | 1200 | 0.38 | ~1.16 | 0.620 | 0.409 | original FT |
| Stage 5C | 6500 | 2.08 | 1.283 | 0.591 | 0.407 | optimal long-run |
| Stage 5B | 10000 | 3.20 | 1.638 | 0.548 | 0.383 | overfitted |

## Key Findings
1. Ablated recovery plateaus at ~0.41 macro regardless of training length
2. More training (beyond ~1200 steps) degrades non-ablated performance
3. Eval loss optimum at ~6500 steps (2 epochs) but doesn't translate to better MCQ
4. Overfitting (10000 steps) hurts both ablated and non-ablated performance
5. The prefill mask FT approach has a ceiling for this model/dataset combo

## Stage 5 Eval Loss Curve (5C run)
- Step 1000: 1.437 | Step 2000: 1.415 | Step 3000: 1.334
- Step 3500: **1.289** (best mid-run) | Step 4000: 1.341
- Step 5000: 1.302 | Step 6000: 1.294 | Step 6500: **1.283** (final)

## Infra
- Vast instance `31462675` destroyed. No active instances.

## Key Artifacts
- Report: `docs/RESULTS_STAGE1_TO_STAGE4.md`
- Stage 3B: `runs/stage3_finetune_prefill_bidir/20260214-233534/final`
- Stage 5B: `runs/stage3_finetune_prefill_bidir/20260215-130710/final` (on Vast, destroyed)
- Stage 5C: `runs/stage3_finetune_prefill_bidir/20260215-201204/final` (on Vast, destroyed)
