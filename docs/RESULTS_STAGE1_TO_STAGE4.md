# Stage 1-4 Results Summary (2026-02-15)

## Scope
- Base model: `mistralai/Ministral-3-3B-Instruct-2512`
- Eval tasks: `hellaswag, piqa, arc_easy, arc_challenge, winogrande`
- Split: `validation`
- Limit: `500` per task (ARC-Challenge has 299 valid examples)

## Stage Definitions
- Stage 1: Vanilla baseline (standard causal prefill)
- Stage 2: Ablated baseline (prefill bidirectional, no finetuning)
- Stage 3B: Leakage-safe finetune with prefix-LM objective (`--prompt-bidir-response-causal-train`)
- Stage 4: Evaluate Stage 3B checkpoint under ablated prefill (`--prefill-bidirectional`)

## Macro Accuracy by Stage
| Stage | Condition | Macro Accuracy |
|---|---|---:|
| Stage 1 | vanilla causal prefill | 0.6950568562 |
| Stage 2 | prefill bidirectional ablation | 0.3558976000 |
| Stage 4 (from Stage 3B ckpt) | ablated prefill after finetune | 0.4086983278 |

Derived:
- Stage 2 damage vs Stage 1: `-0.3391592562` (retention `0.512x`)
- Stage 4 gain vs Stage 2: `+0.0528007278` (about `+5.28` macro points)
- Stage 4 retention vs Stage 1: `0.5880x`

## Per-Task: Stage 1 vs Stage 2 vs Stage 4
| Task | Stage 1 | Stage 2 | Stage 4 | Delta S4-S2 | Delta S4-S1 |
|---|---:|---:|---:|---:|---:|
| hellaswag | 0.622000 | 0.242000 | 0.324000 | +0.082000 | -0.298000 |
| piqa | 0.788000 | 0.534000 | 0.616000 | +0.082000 | -0.172000 |
| arc_easy | 0.814000 | 0.238000 | 0.352000 | +0.114000 | -0.462000 |
| arc_challenge | 0.585284 | 0.247492 | 0.247492 | +0.000000 | -0.337793 |
| winogrande | 0.666000 | 0.518000 | 0.504000 | -0.014000 | -0.162000 |

## Direct Answer to the Main Questions
### 1) How much does prefill bidirectional attention hurt performance?
- A lot without adaptation: macro drops from `0.6951` to `0.3559` (`-33.9` points).
- This is a near-halving of baseline capability (`0.512x` retention).

### 2) Can this be fixed easily with a bit of finetuning?
- Partially, but not fully: Stage 4 improves over Stage 2 by `+5.28` macro points.
- It does **not** recover close to baseline: Stage 4 is still `-28.64` points below Stage 1.
- So this objective change is not trivially fixed by a short finetune if your target is full baseline recovery.

## Additional Check: Non-Prefill Behavior After Stage 3B Finetune
To test whether finetune just helped the ablated condition while keeping normal behavior intact, we also evaluated under standard non-prefill:
- Vanilla non-prefill macro: `0.6950568562`
- Stage 3B checkpoint non-prefill macro: `0.6199344482`
- Delta: `-0.0751224080` (retention `0.8919x`)

Interpretation:
- Finetuning for the ablated prefill condition improved that condition, but introduced a notable regression relative to vanilla under standard non-prefill evaluation.

## Artifacts
- Stage 1/2 reference summary: `docs/RESULTS_STAGE1_STAGE2.md`
- Stage 3B checkpoint: `runs/stage3_finetune_prefill_bidir/20260214-233534/final`
- Stage 4 metrics: `runs/stage4_postft_eval/20260215-093750_final/metrics.json`
- Stage 4 log: `runs/stage4_postft_eval/20260215-093750_final/log.txt`
- Non-prefill vanilla: `runs/nonprefill_compare/20260215-085905_vanilla/metrics.json`
- Non-prefill Stage 3B ckpt: `runs/nonprefill_compare/20260215-091629_prefill_ft_nonprefill_eval/metrics.json`
