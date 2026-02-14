# Experiment Design

## Hypothesis
Removing the prefill causal mask will degrade zero-shot capability because prompt token representations become bidirectional and no longer match decoder-only training dynamics. Some capability may be recoverable with full-weight finetuning.

## Metrics
- Primary: macro accuracy across `hellaswag, piqa, arc_easy, arc_challenge, winogrande`
- Secondary: per-task accuracy and delta to chance baseline
- Recovery metric:
  - `recovery_ratio = post_ft_macro / baseline_macro`

## Stages

### Stage 1: Baseline Zero-Shot
- Config: `configs/stage1_baseline_eval.yaml`
- Condition: standard causal prefill
- Output: `runs/stage1_baseline_eval/<ts>/metrics.json`

### Stage 2: Ablated Zero-Shot
- Config: `configs/stage2_ablation_eval.yaml`
- Condition: prefill bidirectional (`--prefill-bidirectional`)
- Output: `runs/stage2_ablation_eval/<ts>/metrics.json`

### Stage 3A: Full-Weight FT (Causal Train)
- Config: `configs/stage3_finetune_causal.yaml`
- Condition: train without ablation, evaluate with ablation
- Goal: test if regular SFT can adapt parameters enough for ablated inference.

### Stage 3B: Full-Weight FT (Ablated Train)
- Config: `configs/stage3_finetune_prefill_bidir.yaml`
- Condition: train with prefill bidirectional patch active
- Risk: label leakage may inflate train speed. Must watch eval metrics, not train loss only.

### Stage 4: Post-FT Evaluation
- Config: `configs/stage4_postft_eval.yaml`
- Condition: evaluate both Stage 3A and 3B checkpoints under ablated prefill
- Success criteria:
  - Post-FT macro >= 80% of Stage 1 baseline
  - Post-FT macro >= Stage 2 ablated by at least +5 points

## Kill Criteria
- If no learning signal in first 150 train steps, abort and retune LR/batch.
- If eval loss worsens for 3 consecutive checkpoints, abort and reduce LR.
- If ablated train run shows suspiciously fast train loss collapse, treat as leakage risk and prioritize eval.

## Parallelization Plan
Run Stage 3A and Stage 3B in separate Vast instances or tmux sessions with independent run directories.

## Artifacts
- `runs/*/log.txt` for streaming monitoring
- `runs/*/metrics.json` for eval
- `runs/*/summary.json` for finetune summary
