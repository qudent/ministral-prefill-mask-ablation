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
- [x] Confirm baseline model id for runs
- [ ] Execute staged runs on Vast.ai and collect first metrics
  - [ ] Stage 1 baseline metrics
  - [ ] Stage 2 ablated baseline metrics
  - [ ] Stage 3A/3B finetune metrics
  - [ ] Stage 4 recovery comparison

## Blockers
- No technical blocker for launching Stage 1/2. Remaining work is execution on Vast instances and collecting metrics.

## Recent Results
- Added prefill-only mask ablation patch (`q_len > 1 => is_causal=False` for attention modules with `is_causal`).
- Added stage scripts for baseline eval, ablated eval, two finetune variants, and post-finetune evaluation.
- Published repo and synced with Dropbox path `~/Dropbox/ministral-prefill-mask-ablation`.
- Confirmed `mistralai/Ministral-3b-instruct` as default model id; scripts still support alternative Ministral 3B variants via `MODEL_ID`.

## Next Steps
1. Launch Stage 1 and Stage 2 on Vast and compare macro accuracy drop.
2. Run Stage 3A and Stage 3B in parallel and compute recovery ratio from Stage 4.
3. Rewrite this file with concrete metrics and the best next experiment.
