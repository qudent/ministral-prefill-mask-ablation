# Ministral 3B Prefill Mask Ablation - Status

## Current State
Stage 1/2 are complete and documented. Stage 3A (`causal`) and Stage 3B (`prefill_bidir`) are currently running in parallel on two Vast instances (`31425306`, `31433854`) with `mistralai/Ministral-3-3B-Instruct-2512`.

## Active Goals
- [x] Stage 1 baseline eval
- [x] Stage 2 ablated eval
- [x] Stage 1/2 results documentation
- [x] Stage 3 launch in parallel on separate Vast instances
- [x] Add robust Transformer v5 compatibility in finetune script
- [x] Add runtime/cost planning estimator for setup decisions
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

## Runtime/Infra Notes
- Current measured early-step throughput:
  - Stage 3A: `~3.318 s/step`
  - Stage 3B: `~3.908 s/step`
- Stage 3 script defaults tuned for lower overhead on single-GPU runs:
  - `eval_samples=250`, `eval_steps=200`, `save_steps=400`
- `scripts/local/estimate_vast_plan.py` estimates wall-clock + cost across setups (including `8x H100`) using measured step times and live Vast offers.

## Artifacts
- Stage 1 metrics: `runs/stage1_baseline_eval/20260214-145148/metrics.json`
- Stage 2 metrics: `runs/stage2_ablation_eval/20260214-145456/metrics.json`
- Stage 1/2 summary: `docs/RESULTS_STAGE1_STAGE2.md`
- Planning utility: `scripts/local/estimate_vast_plan.py`

## Blockers
- No blocker. Remaining risk is runtime cost/performance tradeoff for larger GPU configurations.

## Next Steps
1. Let Stage 3A/3B complete and capture `summary.json` for both.
2. Run Stage 4 post-finetune ablated eval on both checkpoints.
3. Compute recovery ratio and decide whether to rerun Stage 3 at larger scale (e.g., H100 variants).
