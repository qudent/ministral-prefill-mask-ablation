# Ministral 3B Prefill Mask Ablation

Experiment repo for testing what happens when the triangular causal mask is removed during **prefill** so each prompt token can attend to both predecessor and successor tokens.

## Questions
1. Does zero-shot capability survive this prefill ablation?
2. Can full-weight finetuning recover capability after the ablation?

## Core Ablation
- Default decoder behavior: causal attention during prefill and decode.
- Ablated behavior in this repo: for attention calls with `q_len > 1`, temporarily set `is_causal=False`.
- Decode (`q_len == 1`) remains unchanged.

Code: `src/prefill_ablation/attention_ablation.py`

## Staged Experimental Design
Detailed plan: `docs/EXPERIMENT_DESIGN.md`
Results snapshot (Stage 1/2): `docs/RESULTS_STAGE1_STAGE2.md`

- Stage 1: Baseline zero-shot (no ablation)
- Stage 2: Ablated zero-shot (prefill bidirectional)
- Stage 3A: Full-weight finetune (causal train)
- Stage 3B: Full-weight finetune (prefill-bidirectional train)
- Stage 4: Post-finetune eval under ablated prefill

## Vast.ai-First Workflow
All heavyweight work runs on Vast instances. Local machine only authors and pushes code.

Runbook: `docs/VAST_RUNBOOK.md`

### Quick start on a VAST instance
```bash
git clone https://github.com/qudent/ministral-prefill-mask-ablation.git
cd ministral-prefill-mask-ablation
bash scripts/vast/bootstrap.sh
```

### Stage commands
```bash
# Baseline zero-shot
bash scripts/vast/run_stage1_baseline_eval.sh

# Ablated zero-shot
bash scripts/vast/run_stage2_ablation_eval.sh

# Finetune variants
bash scripts/vast/run_stage3_finetune.sh causal
bash scripts/vast/run_stage3_finetune.sh prefill_bidir

# Post-FT ablated eval
bash scripts/vast/run_stage4_postft_eval.sh \
  runs/stage3_finetune_causal/<timestamp>/final \
  runs/stage3_finetune_prefill_bidir/<timestamp>/final
```

## Model ID
Default scripts use `mistralai/Ministral-3-3B-Instruct-2512` (confirmed accessible). If you want to run the same ablation on another Ministral 3B variant, set:
```bash
export MODEL_ID=<your_model_id_or_local_path>
```

## Notes
- This repo intentionally contains no model weights and no dataset dumps.
- Results are written under `runs/` and `artifacts/` (gitignored).
- `STATUS.md` is the active coordination file.
