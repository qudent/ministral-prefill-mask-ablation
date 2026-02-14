# Ministral 3B Prefill Mask Ablation - Status

## Current State
Stage 1/2 and Stage 3 are complete. Stage 3 runs finished on 2026-02-14 with fallback raw checkpoint saves (due HF save conversion error). Stage 4 post-finetune ablated eval is next.

## Active Goals
- [x] Stage 1 baseline eval
- [x] Stage 2 ablated eval
- [x] Stage 1/2 results documentation
- [x] Stage 3A/3B finetune runs in parallel
- [x] Robust Stage 3 checkpoint save/load fallback
- [ ] Stage 4 post-FT ablated eval
- [ ] Recovery ratio comparison vs Stage 1 baseline

## Key Results (Stage 1 vs Stage 2)
- Macro accuracy: `0.695057 -> 0.355898` (delta `-0.339159`, retention `0.512x`)
- HellaSwag: `0.622 -> 0.242` (delta `-0.380`)
- PIQA: `0.788 -> 0.534` (delta `-0.254`)
- ARC-Easy: `0.814 -> 0.238` (delta `-0.576`)
- ARC-Challenge: `0.585284 -> 0.247492` (delta `-0.337793`)
- WinoGrande: `0.666 -> 0.518` (delta `-0.148`)

## Stage 3 Outcomes
- 3A causal run: `runs/stage3_finetune_causal/20260214-201019`
  - `eval_loss=1.3752`, `train_loss=1.3377`, `train_steps_per_second=2.425`
- 3B prefill_bidir run: `runs/stage3_finetune_prefill_bidir/20260214-201019`
  - `eval_loss=7.0321`, `train_loss=5.4243`, `train_steps_per_second=2.070`
- Both wrote `summary.json` and `final/` with `checkpoint_meta.json` format `raw_state_dict`.
- Diagnosis from direct probe (2026-02-14): removing causal mask in SFT causes severe label leakage.
  - Same sample, same weights: `loss_causal ~= 1.023`, `loss_prefill_bidir ~= 0.070`.
  - This explains implausibly low early 3B loss and supports that current 3B objective is degenerate.
  - 3B then destabilizes (grad norms spike to `~102` by step 50), likely due high-LR full-weight updates on leaked objective.

## Runtime/Infra Notes
- Latest fix commits:
  - `ebcb783`: save/load fallback for HF save crash
  - `c2eb28d`: status update after relaunch
- Local watcher script is now run-configurable:
  - `scripts/local/wait_or_crash.sh` with `VAST_HOST/VAST_PORT/REMOTE_SESSION/COMPLETE_FILE`
  - Supports sparse polling for long trusted runs.

## Artifacts
- Stage 1 metrics: `runs/stage1_baseline_eval/20260214-145148/metrics.json`
- Stage 2 metrics: `runs/stage2_ablation_eval/20260214-145456/metrics.json`
- Stage 1/2 summary: `docs/RESULTS_STAGE1_STAGE2.md`

## Blockers
- None active.

## Next Steps
1. Run Stage 4 ablated prefill eval on both Stage 3 checkpoints.
2. Compute recovery ratio versus Stage 1 baseline macro/task accuracies.
3. Replace 3B training objective with leakage-safe masking (e.g., bidir on prompt tokens only, causal on response tokens), then rerun 3B.
