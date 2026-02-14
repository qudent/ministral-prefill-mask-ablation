# Ministral 3B Prefill Mask Ablation - Status

## Current State
Project scaffold is complete and published at `https://github.com/qudent/ministral-prefill-mask-ablation`. The repo is VAST-first: local machine only edits/pushes code, while all model downloads, eval, and training execute on Vast.ai instances.

## Active Goals
- [x] Create a clean Dropbox project repo with VAST-first workflow
  - [x] Add coordination `STATUS.md`
  - [x] Add staged experiment design and kill criteria
- [x] Implement ablation + eval + training scripts
  - [x] Prefill bidirectional patch utility
  - [x] Zero-shot multi-task MCQ evaluation
  - [x] Full-weight SFT runner with early-kill callback
- [x] Publish public GitHub repo and push initial code
- [ ] Execute staged runs on Vast.ai and collect first metrics
  - [ ] Stage 1 baseline metrics
  - [ ] Stage 2 ablated baseline metrics
  - [ ] Stage 3A/3B finetune metrics
  - [ ] Stage 4 recovery comparison

## Blockers
- Exact HF model ID for the user's "without thinking" Ministral 3B checkpoint is not yet pinned. Scripts use `mistralai/Ministral-3b-instruct` placeholder and support override via `MODEL_ID`.

## Recent Results
- Added prefill-only mask ablation patch (`q_len > 1 => is_causal=False` for attention modules with `is_causal`).
- Added stage scripts for baseline eval, ablated eval, two finetune variants, and post-finetune evaluation.
- Published repo and synced with Dropbox path `~/Dropbox/ministral-prefill-mask-ablation`.

## Next Steps
1. Confirm the exact target "without thinking" model ID and set `MODEL_ID`.
2. Launch Stage 1 and Stage 2 on Vast and compare macro accuracy drop.
3. Run Stage 3A and Stage 3B in parallel and compute recovery ratio from Stage 4.
4. Rewrite this file with concrete metrics and the best next experiment.
