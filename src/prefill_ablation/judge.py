"""LLM-as-a-judge scoring via OpenRouter.

Reads generation results from eval_freeform.py and scores them using a strong model.

Usage:
  uv run prefill-judge \
    --results-dir results/freeform \
    --output results/freeform/scores.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import urllib.request
import urllib.error


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "anthropic/claude-sonnet-4"

TRANSLATION_JUDGE_PROMPT = """You are evaluating a machine translation from English to German.

Source (English): {source}
Reference translation: {reference}
Model output: {response}

Rate the model output on a scale of 1-5:
1 = Completely wrong, gibberish, or not German at all
2 = Partially understandable but major errors (wrong meaning, missing important parts)
3 = Understandable with some errors (grammar mistakes, awkward phrasing, minor meaning shifts)
4 = Good translation with minor issues (slightly unnatural word choice, small grammar issues)
5 = Excellent, natural-sounding German that accurately conveys the meaning

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<brief explanation>"}}"""

CODE_JUDGE_PROMPT = """You are evaluating a model's response to a programming question.

Question: {instruction}
Reference answer: {reference}
Model output: {response}

Rate the model output on a scale of 1-5:
1 = Completely wrong, gibberish, or irrelevant
2 = Shows some understanding but has major errors or missing key points
3 = Partially correct, addresses the question but has notable errors or omissions
4 = Mostly correct with minor issues or could be more precise
5 = Excellent, correct and well-explained answer

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<brief explanation>"}}"""


def _call_openrouter(prompt: str, api_key: str, max_retries: int = 3) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 200,
    }).encode()

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(OPENROUTER_URL, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            content = data["choices"][0]["message"]["content"].strip()
            # Parse JSON from response (handle markdown wrapping)
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(content)
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [retry] attempt {attempt+1} failed: {e}, waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  [error] all retries failed: {e}")
                return {"score": 0, "reason": f"judge error: {e}"}


def _judge_item(item: dict, api_key: str) -> dict:
    if item["task_type"] == "translation_en_de":
        prompt = TRANSLATION_JUDGE_PROMPT.format(
            source=item["source"],
            reference=item["reference"],
            response=item["response"],
        )
    else:
        prompt = CODE_JUDGE_PROMPT.format(
            instruction=item["instruction"],
            reference=item["reference"],
            response=item["response"],
        )
    result = _call_openrouter(prompt, api_key)
    return {
        "id": item["id"],
        "task_type": item["task_type"],
        "score": result.get("score", 0),
        "reason": result.get("reason", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-judge scoring")
    parser.add_argument("--results-dir", required=True, help="Directory with generation results")
    parser.add_argument("--output", default=None, help="Output scores JSON path")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--configs", nargs="+",
                        default=["vanilla", "vanilla-ablated", "finetuned", "finetuned-ablated"])
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY or pass --api-key")

    results_dir = Path(args.results_dir)
    all_scores = {}

    for config_name in args.configs:
        result_path = results_dir / f"{config_name}.json"
        if not result_path.exists():
            print(f"Skipping {config_name}: {result_path} not found")
            continue

        print(f"\nJudging: {config_name}")
        with open(result_path) as f:
            items = json.load(f)

        scores = []
        for i, item in enumerate(items):
            score = _judge_item(item, api_key)
            scores.append(score)
            print(f"  [{item['id']}] score={score['score']} {score['reason'][:60]}")
            # Rate limit
            if i < len(items) - 1:
                time.sleep(0.5)

        all_scores[config_name] = scores

    # Compute summary
    summary = {}
    for config_name, scores in all_scores.items():
        by_task = {}
        for s in scores:
            task = s["task_type"]
            if task not in by_task:
                by_task[task] = []
            by_task[task].append(s["score"])

        config_summary = {}
        all_task_scores = []
        for task, task_scores in by_task.items():
            valid = [s for s in task_scores if s > 0]
            avg = sum(valid) / len(valid) if valid else 0
            config_summary[task] = {"mean": round(avg, 2), "n": len(valid)}
            all_task_scores.extend(valid)

        overall = sum(all_task_scores) / len(all_task_scores) if all_task_scores else 0
        config_summary["overall"] = {"mean": round(overall, 2), "n": len(all_task_scores)}
        summary[config_name] = config_summary

    output = {
        "judge_model": JUDGE_MODEL,
        "summary": summary,
        "detailed": all_scores,
    }

    out_path = Path(args.output) if args.output else results_dir / "scores.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nScores saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY (mean score 1-5)")
    print("=" * 70)
    print(f"{'Config':<25} {'Translation':<15} {'Code':<15} {'Overall':<15}")
    print("-" * 70)
    for config_name in args.configs:
        if config_name not in summary:
            continue
        s = summary[config_name]
        tr = s.get("translation_en_de", {}).get("mean", "—")
        co = s.get("code_understanding", {}).get("mean", "—")
        ov = s.get("overall", {}).get("mean", "—")
        print(f"{config_name:<25} {tr:<15} {co:<15} {ov:<15}")
    print("=" * 70)


if __name__ == "__main__":
    main()
