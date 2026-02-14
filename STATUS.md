# Ministral 3B Prefill Mask Ablation - Status

## Current State
Stage 1/2 are complete and documented. Stage 3A (`causal`) and Stage 3B (`prefill_bidir`) were relaunched on 2026-02-14 after fixing a Transformers save-time crash.

## Active Goals
- [x] Stage 1 baseline eval
- [x] Stage 2 ablated eval
- [x] Stage 1/2 results documentation
- [x] Stage 3 launch in parallel on separate Vast instances
- [x] Stage 3 save-time crash fix (`NotImplementedError` in HF weight conversion)
- [ ] Stage 3A/3B final metrics
- [ ] Stage 4 post-FT ablated eval
- [ ] Recovery ratio comparison vs Stage 1 baseline

## Key Results (Stage 1 vs Stage 2)
- Macro accuracy: `0.695057 -> 0.355898` (delta `-0.339159`, retention `0.512x`)
- HellaSwag: `0.622 -> 0.242` (delta `-0.380`)
- PIQA: `0.788 -> 0.534` (delta `-0.254`)
- ARC-Easy: `0.814 -> 0.238` (delta `-0.576`)
- ARC-Challenge: `0.585284 -> 0.247492` (delta `-0.337793`)
- WinoGrande: `0.666 -> 0.518` (delta `-0.148`)

## Live Runs
- Stage 3A (`causal`): Vast instance `31425306`, tmux `prefill-s3a`, run dir `runs/stage3_finetune_causal/20260214-201019`
- Stage 3B (`prefill_bidir`): Vast instance `31433854`, tmux `prefill-s3b`, run dir `runs/stage3_finetune_prefill_bidir/20260214-201019`

## Runtime/Infra Notes
- Latest fix commit: `ebcb783` (GitHub main)
- Fix: fallback checkpoint save path in Stage 3
  - If `trainer.save_model` fails, save raw `state_dict` + `checkpoint_meta.json`
  - Loader supports rehydrating raw checkpoints via `base_model_id` for Stage 4 eval
- Current Stage 3 defaults: `eval_samples=250`, `eval_steps=200`, `save_steps=400`
- Planning utility: `scripts/local/estimate_vast_plan.py`

## Artifacts
- Stage 1 metrics: `runs/stage1_baseline_eval/20260214-145148/metrics.json`
- Stage 2 metrics: `runs/stage2_ablation_eval/20260214-145456/metrics.json`
- Stage 1/2 summary: `docs/RESULTS_STAGE1_STAGE2.md`

## Blockers
- None active. Previous save-time blocker resolved in `ebcb783`.

## Next Steps
1. Let both Stage 3 runs finish and collect `summary.json`.
2. Run Stage 4 post-finetune ablated eval on the resulting checkpoints.
3. Compute recovery ratio for `causal` vs `prefill_bidir` against Stage 1 baseline.
