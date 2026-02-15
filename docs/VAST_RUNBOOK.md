# Vast.ai Runbook

## 1. Create Instance
Use a GPU with enough VRAM for full-weight 3B training. Recommended:
- 1x A100 80GB (preferred)
- or equivalent VRAM + enough system RAM (>=64GB recommended)

## 2. Bootstrap
```bash
git clone https://github.com/qudent/ministral-prefill-mask-ablation.git
cd ministral-prefill-mask-ablation
bash scripts/vast/bootstrap.sh
```

## 3. Optional: set model and dataset
```bash
export MODEL_ID=mistralai/Ministral-3-3B-Instruct-2512
export DATASET_ID=yahma/alpaca-cleaned
```

## 4. Run stages
```bash
bash scripts/vast/run_stage1_baseline_eval.sh
bash scripts/vast/run_stage2_ablation_eval.sh
bash scripts/vast/run_stage3_finetune.sh causal
bash scripts/vast/run_stage3_finetune.sh prefill_bidir
bash scripts/vast/run_stage4_postft_eval.sh \
  runs/stage3_finetune_causal/<ts>/final \
  runs/stage3_finetune_prefill_bidir/<ts>/final
```

## 5. Monitor
```bash
bash scripts/vast/tail_logs.sh 8
```

## 5b. Agent-Driven Babysitting (No Scripted Autopilot)
Use a dedicated Codex tmux session to monitor, debug, and relaunch as needed:
```bash
tmux new-session -d -s vast-babysit
tmux send-keys -t vast-babysit 'codex --dangerously-bypass-approvals-and-sandbox' Enter Enter
```

Hand the babysitting agent:
- Vast SSH target and project path
- run id / log path
- alert policy (step stall, traceback, GPU idle)
- teardown policy (destroy instance on terminal done/fail)

This keeps recovery logic in the agent loop instead of complex shell automation.

## 6. Cost discipline
- Stop idle instances immediately.
- Keep checkpoints every ~200 steps to avoid losing >30 minutes of work.

## 7. Minimal troubleshooting
- Slow internet: re-provision; download speed should be checked in bootstrap output.
- OOM: reduce sequence length, increase gradient accumulation, lower train samples per run, or move to larger VRAM.
- No early learning signal: abort early and rerun with adjusted LR.
