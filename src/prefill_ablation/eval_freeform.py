"""Free-form evaluation: generate answers for translation and code tasks.

Runs the model in 4 configurations:
  1. vanilla (base model, normal causal attention)
  2. vanilla-ablated (base model, bidirectional prefill)
  3. finetuned (Stage 3B checkpoint, normal causal attention)
  4. finetuned-ablated (Stage 3B checkpoint, bidirectional prefill)

Usage:
  uv run prefill-freeform-eval \
    --checkpoint <path-or-hf-id> \
    --output-dir results/freeform \
    --dataset data/eval_freeform.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prefill_ablation.attention_ablation import apply_prefill_bidirectional_patch
from prefill_ablation.utils import load_model_and_tokenizer


BASE_MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"

TRANSLATION_SYSTEM = "You are a professional translator. Translate the following English text to German. Output only the German translation, nothing else."

CODE_SYSTEM = "You are an expert Python programmer. Answer the following question concisely and accurately."


def _load_dataset(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_messages(task_type: str, item: dict) -> list[dict]:
    if task_type == "translation_en_de":
        return [
            {"role": "system", "content": TRANSLATION_SYSTEM},
            {"role": "user", "content": item["source"]},
        ]
    else:
        return [
            {"role": "system", "content": CODE_SYSTEM},
            {"role": "user", "content": item["instruction"]},
        ]


def _generate_responses(
    model,
    tokenizer,
    dataset: dict,
    *,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> list[dict]:
    results = []
    for task_type, items in dataset.items():
        for item in items:
            messages = _build_messages(task_type, item)
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(device)

            t0 = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            elapsed = time.time() - t0

            # Decode only new tokens
            new_tokens = output_ids[0, input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            result = {
                "id": item["id"],
                "task_type": task_type,
                "response": response,
                "tokens_generated": len(new_tokens),
                "time_seconds": round(elapsed, 2),
            }
            if "source" in item:
                result["source"] = item["source"]
                result["reference"] = item["reference"]
            else:
                result["instruction"] = item["instruction"]
                result["reference"] = item["reference"]

            results.append(result)
            print(f"  [{item['id']}] {len(new_tokens)} tokens in {elapsed:.1f}s: {response[:80]}...")
    return results


def _run_config(
    config_name: str,
    model_path: str,
    dataset: dict,
    *,
    ablated: bool,
    output_dir: Path,
    device: str,
    max_new_tokens: int,
):
    print(f"\n{'='*60}")
    print(f"Config: {config_name} (ablated={ablated})")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(
        model_path, dtype="bfloat16", device_map=device,
    )
    model.eval()

    patch = None
    if ablated:
        patch = apply_prefill_bidirectional_patch(model)

    results = _generate_responses(
        model, tokenizer, dataset,
        max_new_tokens=max_new_tokens, device=device,
    )

    if patch:
        patch.remove()

    out_path = output_dir / f"{config_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to {out_path}")

    # Free memory
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Free-form evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to finetuned checkpoint (local or HF)")
    parser.add_argument("--base-model", default=BASE_MODEL_ID, help="Base model ID")
    parser.add_argument("--dataset", default="data/eval_freeform.json", help="Dataset JSON path")
    parser.add_argument("--output-dir", default="results/freeform", help="Output directory")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--configs", nargs="+",
                        default=["vanilla", "vanilla-ablated", "finetuned", "finetuned-ablated"],
                        help="Which configs to run")
    args = parser.parse_args()

    dataset = _load_dataset(args.dataset)
    total = sum(len(v) for v in dataset.values())
    print(f"Loaded {total} examples from {args.dataset}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_map = {
        "vanilla":           (args.base_model, False),
        "vanilla-ablated":   (args.base_model, True),
        "finetuned":         (args.checkpoint, False),
        "finetuned-ablated": (args.checkpoint, True),
    }

    for config_name in args.configs:
        if config_name not in config_map:
            print(f"Unknown config: {config_name}, skipping")
            continue
        model_path, ablated = config_map[config_name]
        _run_config(
            config_name, model_path, dataset,
            ablated=ablated, output_dir=output_dir,
            device=args.device, max_new_tokens=args.max_new_tokens,
        )

    # Save dataset alongside results for reference
    import shutil
    shutil.copy2(args.dataset, output_dir / "dataset.json")
    print(f"\nAll configs complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
