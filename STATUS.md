# Ministral 3B Prefill Mask Ablation - Status

## Current State
Stage 1/2 are complete and documented. Stage 3B is now leakage-safe (prompt bidirectional, response causal),
but the latest long rerun reached `1200/1200` and then crashed during model save with a Transformers
`NotImplementedError` in weight-conversion reverse ops. Training progress exists; completion marker was not written.
Local code now forces Trainer `save_strategy=no` to skip failing internal checkpoint saves; rerun is active.
Prior-art context for PrefixLM/prefill-bidirectional masking is documented in `docs/PREFILL_LM_PRIOR_ART.md`.

## Active Goals
- [x] Stage 1 baseline eval
- [x] Stage 2 ablated eval
- [x] Stage 1/2 results documentation
- [x] Diagnose prior 3B failure mode (label leakage)
- [x] Implement leakage-safe 3B objective (`--prompt-bidir-response-causal-train`)
- [ ] Produce successful Stage 3B saved artifact (`summary.json` + `final/`)
- [ ] Stage 4 post-FT ablated eval
- [ ] Recovery ratio comparison vs Stage 1 baseline

## Confirmed Results
- Stage 1 vs Stage 2 macro accuracy: `0.695057 -> 0.355898` (delta `-0.339159`, retention `0.512x`)
- Stage 3A causal run: `runs/stage3_finetune_causal/20260214-201019`
  - `eval_loss=1.3752`, `train_loss=1.3377`, `train_steps_per_second=2.425`
- Stage 3B full-bidir run (invalid objective): `runs/stage3_finetune_prefill_bidir/20260214-201019`
  - `eval_loss=7.0321`, `train_loss=5.4243`
- Stage 3B prefix-LM rerun reached final step:
  - `runs/stage3_finetune_prefill_bidir/20260214-213740/log.txt` shows `100%|...| 1200/1200`
  - latest logged eval before failure: `eval_loss ~ 1.157`

## 3B Diagnosis
- Full-bidir SFT leaks labels (`loss_prefill_bidir ~= 0.070` vs `loss_causal ~= 1.023` on same sample/weights).
- Prefix-LM objective is implemented via `--prompt-bidir-response-causal-train`.
- Save-path failure persists even with sparse save schedule:
  - crash site: `transformers/core_model_loading.py`, `reverse_op -> NotImplementedError`
  - trigger point: Trainer checkpoint/model save after training completes

## Current Infra State
- Active Vast instance: `ssh7.vast.ai:33854` (`31433854`)
- New Stage 3B rerun is active: `runs/stage3_finetune_prefill_bidir/20260214-233534`
  - remote tmux session: `prefill-s3b`
  - active training process observed: `uv run prefill-finetune ... --prompt-bidir-response-causal-train`
- Prior failed launch `20260214-233230` ended immediately due non-login PATH (`uv: command not found`).
- Local babysit tmux: `prefill-babysit` is in blocking Step3 watcher mode (`wait_or_crash.sh`) and will continue to:
  sync artifacts/checkpoints, then destroy instance on successful completion.

## Artifacts
- Stage 1 metrics: `runs/stage1_baseline_eval/20260214-145148/metrics.json`
- Stage 2 metrics: `runs/stage2_ablation_eval/20260214-145456/metrics.json`
- Stage 1/2 writeup: `docs/RESULTS_STAGE1_STAGE2.md`
- PrefixLM prior-art writeup: `docs/PREFILL_LM_PRIOR_ART.md`
- Failed 3B rerun logs:
  - `runs/stage3_finetune_prefill_bidir/20260214-204823/log.txt`
  - `runs/stage3_finetune_prefill_bidir/20260214-213740/log.txt`

## Next Steps
1. Let active Stage 3B run complete under babysitter workflow.
2. Verify synced local artifact contains `summary.json` and `final/checkpoint_meta.json`.
3. Confirm Vast instance `31433854` is destroyed by workflow.
4. Run Stage 4 ablated eval on Stage 3A and valid Stage 3B artifact.
