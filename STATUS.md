# Ministral 3B Prefill Mask Ablation - Status

## Current State
Project scaffold is complete and pushed from local workspace design assumptions. All heavy execution is intentionally VAST-only. Repo includes prefill bidirectional mask ablation patch, zero-shot eval harness, full-weight SFT scripts, stage configs, and a VAST runbook. No model weights or datasets are stored locally in this repo.

## Active Goals
- [x] Create a clean Dropbox project repo with VAST-first workflow
  - [x] Add coordination `STATUS.md`
  - [x] Add staged experiment design and kill criteria
- [x] Implement ablation + eval + training scripts
  - [x] Prefill bidirectional patch utility
  - [x] Zero-shot multi-task MCQ evaluation
  - [x] Full-weight SFT runner with early-kill callback
- [ ] Execute staged runs on Vast.ai and collect first metrics
  - [ ] Stage 1 baseline metrics
  - [ ] Stage 2 ablated baseline metrics
  - [ ] Stage 3A/3B finetune metrics
  - [ ] Stage 4 recovery comparison

## Blockers
- Exact HF model ID for the user's "without thinking" Ministral 3B checkpoint is not yet pinned. Scripts use `mistralai/Ministral-3b-instruct` placeholder and support override via `MODEL_ID`.

## Recent Results
- Added patching logic that removes causal masking only when attention query length is greater than 1.
- Added VAST bootstrap and stage run scripts for baseline, ablation, finetune, and post-finetune eval.
- Added docs for experiment design, success criteria, and runtime management.

## Next Steps
1. Confirm exact model ID for the target "without thinking" checkpoint.
2. Launch Stage 1 and Stage 2 on Vast and compute the baseline drop from prefill ablation.
3. Run Stage 3A and Stage 3B in parallel and compare Stage 4 recovery ratio.
4. Rewrite this file with concrete metric values and best-next action after first full pass.
