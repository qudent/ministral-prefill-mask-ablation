# Ministral 3B Prefill Mask Ablation - Status

## Current State
Execution is active on Vast.ai with Stage 1 running again. Model loading and datasets package compatibility have been addressed. Latest blocker was a stale Hugging Face dataset cache generated under a different `datasets` major version; cache was cleared and the run restarted.

## Active Goals
- [x] Create repo, scripts, and VAST-first workflow
- [x] Provision fresh Vast instance for execution
- [x] Identify accessible Ministral 3B model ID
- [x] Patch model loader for Mistral3 architecture
- [x] Fix datasets version incompatibility
- [x] Clear incompatible HF dataset cache on runner
- [ ] Execute staged runs on Vast.ai and collect first metrics
  - [ ] Stage 1 baseline metrics
  - [ ] Stage 2 ablated baseline metrics
  - [ ] Stage 3A/3B finetune metrics
  - [ ] Stage 4 recovery comparison

## Runtime Details
- Vast instance ID: `31425306`
- GPU: `1x RTX 3090`
- Label: `prefill-mask-s12`
- SSH endpoint: `ssh://root@108.55.118.247:53346`
- Remote repo path: `/root/ministral-prefill-mask-ablation`
- Remote tmux session: `prefill-s12`

## Blockers
- No known blocker; waiting for Stage 1/2 completion.

## Recent Results
- Corrected model id to `mistralai/Ministral-3-3B-Instruct-2512`.
- Added loader fallback for `Mistral3ForConditionalGeneration`.
- Pinned `datasets` to `<3` for benchmark loaders.
- Cleared `/root/.cache/huggingface/datasets` to remove stale metadata mismatch.

## Next Steps
1. Let Stage 1 finish and write `runs/stage1_baseline_eval/<ts>/metrics.json`.
2. Let Stage 2 run automatically and write `runs/stage2_ablation_eval/<ts>/metrics.json`.
3. Compare macro accuracy baseline vs ablated and record in this file.
4. Launch Stage 3A and Stage 3B in parallel if Stage 1/2 complete cleanly.
