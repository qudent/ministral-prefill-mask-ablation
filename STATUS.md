# Ministral 3B Prefill Mask Ablation - Status

## Current State
Stage 1 (baseline causal prefill) and Stage 2 (prefill bidirectional ablation) are complete for `mistralai/Ministral-3-3B-Instruct-2512` on Vast instance `31425306`. Results show a large capability drop under prefill bidirectional attention.

## Active Goals
- [x] Create repo, scripts, and VAST-first workflow
- [x] Provision fresh Vast instance for execution
- [x] Identify accessible Ministral 3B model ID
- [x] Patch model loader for Mistral3 architecture
- [x] Fix datasets/version/cache incompatibilities
- [x] Stage 1 baseline metrics
- [x] Stage 2 ablated baseline metrics
- [ ] Stage 3A/3B finetune metrics
- [ ] Stage 4 recovery comparison

## Key Results (Stage 1 vs Stage 2)
- Macro accuracy: `0.695057 -> 0.355898` (delta `-0.339159`, retention `0.512x`)
- HellaSwag: `0.622 -> 0.242` (delta `-0.380`)
- PIQA: `0.788 -> 0.534` (delta `-0.254`)
- ARC-Easy: `0.814 -> 0.238` (delta `-0.576`)
- ARC-Challenge: `0.585284 -> 0.247492` (delta `-0.337793`)
- WinoGrande: `0.666 -> 0.518` (delta `-0.148`)

## Artifacts
- Stage 1 metrics: `runs/stage1_baseline_eval/20260214-145148/metrics.json`
- Stage 2 metrics: `runs/stage2_ablation_eval/20260214-145456/metrics.json`

## Runtime Details
- Vast instance ID: `31425306`
- GPU: `1x RTX 3090`
- SSH endpoint: `ssh://root@108.55.118.247:53346`
- Remote repo path: `/root/ministral-prefill-mask-ablation`

## Blockers
- No blocker for proceeding to Stage 3. Main decision is training budget and whether to run 3A/3B in parallel.

## Next Steps
1. Launch Stage 3A (`causal`) and Stage 3B (`prefill_bidir`) finetunes.
2. Run Stage 4 post-finetune ablated evaluation on both checkpoints.
3. Compute recovery ratio vs Stage 1 baseline and pick best path.
