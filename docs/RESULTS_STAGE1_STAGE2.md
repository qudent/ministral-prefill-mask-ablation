# Stage 1/2 Results (2026-02-14)

## Scope
- Model: `mistralai/Ministral-3-3B-Instruct-2512`
- Eval set: `hellaswag, piqa, arc_easy, arc_challenge, winogrande`
- Split: `validation`
- Limit: `500` examples per task (ARC-Challenge had 299 valid examples after filtering)

## Commit References
- Run code commit (used on Vast for these metrics): `ac2f660`
- Project state with this documented summary: `f0af92d`

## Metrics
| Task | Stage 1 Baseline | Stage 2 Ablated | Delta (S2-S1) |
|---|---:|---:|---:|
| hellaswag | 0.622000 | 0.242000 | -0.380000 |
| piqa | 0.788000 | 0.534000 | -0.254000 |
| arc_easy | 0.814000 | 0.238000 | -0.576000 |
| arc_challenge | 0.585284 | 0.247492 | -0.337793 |
| winogrande | 0.666000 | 0.518000 | -0.148000 |
| **macro** | **0.695057** | **0.355898** | **-0.339159** |

Derived:
- Macro retention ratio: `0.355898 / 0.695057 = 0.512042`

## Raw Artifacts
- Stage 1: `runs/stage1_baseline_eval/20260214-145148/metrics.json`
- Stage 2: `runs/stage2_ablation_eval/20260214-145456/metrics.json`

## Reproduction (Concise)
On a fresh Vast instance:

```bash
git clone https://github.com/qudent/ministral-prefill-mask-ablation.git
cd ministral-prefill-mask-ablation
git checkout ac2f660
bash scripts/vast/bootstrap.sh
bash scripts/vast/run_stage1_baseline_eval.sh
bash scripts/vast/run_stage2_ablation_eval.sh
```

Optional quick compare:

```bash
python - <<'PY'
import json
from pathlib import Path
s1 = json.loads(Path("runs/stage1_baseline_eval/20260214-145148/metrics.json").read_text())
s2 = json.loads(Path("runs/stage2_ablation_eval/20260214-145456/metrics.json").read_text())
print("macro", s1["macro_accuracy"], s2["macro_accuracy"], s2["macro_accuracy"] - s1["macro_accuracy"])
PY
```
