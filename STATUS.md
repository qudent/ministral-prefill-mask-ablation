# Ministral 3B Prefill Mask Ablation - Status

## Current State
Execution is active on Vast.ai. Stage 1 has been retried after model-id correction, but failed again because `Ministral-3-3B-Instruct-2512` loads as `Mistral3ForConditionalGeneration` (not plain `AutoModelForCausalLM`). Loader code is now patched to fallback to `AutoModelForImageTextToText`, and the run is being restarted.

## Active Goals
- [x] Create repo, scripts, and VAST-first workflow
- [x] Provision fresh Vast instance for execution
- [x] Identify accessible Ministral 3B model ID
- [x] Patch model loader for Mistral3 architecture
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

## Blockers
- No blocker after loader patch; pending rerun validation.

## Recent Results
- Fast-failed Stage 1 with old model id (HF 401 repo-not-found) and corrected defaults.
- Fast-failed Stage 1 again with `Mistral3Config` vs `AutoModelForCausalLM` mismatch.
- Implemented fallback loader path to `AutoModelForImageTextToText` for eval and finetune scripts.

## Next Steps
1. Push loader patch and pull on active instance.
2. Rerun Stage 1 and Stage 2 sequentially.
3. Capture both `metrics.json` files and compute macro-accuracy drop.
4. Rewrite this file with concrete metric values and Stage 3 launch decision.
