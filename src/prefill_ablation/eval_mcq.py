from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from datasets import load_dataset
from tqdm import tqdm

from prefill_ablation.attention_ablation import apply_prefill_bidirectional_patch
from prefill_ablation.utils import load_model_and_tokenizer, set_seed


@dataclass
class Example:
    prompt: str
    choices: list[str]
    label: int


@dataclass
class TaskSpec:
    name: str
    loader: Callable[[str], Iterable[Example]]


def _load_hf_dataset(name: str, *args, split: str):
    # Some benchmark datasets ship dataset scripts and require explicit trust.
    return load_dataset(name, *args, split=split, trust_remote_code=True)


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def load_hellaswag(split: str) -> Iterable[Example]:
    ds = _load_hf_dataset("hellaswag", split=split)
    for row in ds:
        prompt = _normalize_space(f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b']}")
        choices = [" " + _normalize_space(choice) for choice in row["endings"]]
        label = int(row["label"])
        yield Example(prompt=prompt, choices=choices, label=label)


def load_piqa(split: str) -> Iterable[Example]:
    ds = _load_hf_dataset("piqa", split=split)
    for row in ds:
        prompt = _normalize_space(row["goal"]) + "\nAnswer:"
        choices = [" " + _normalize_space(row["sol1"]), " " + _normalize_space(row["sol2"])]
        label = int(row["label"])
        yield Example(prompt=prompt, choices=choices, label=label)


def _load_arc(config_name: str, split: str) -> Iterable[Example]:
    ds = _load_hf_dataset("ai2_arc", config_name, split=split)
    for row in ds:
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        answer = row.get("answerKey", "")
        if answer not in labels:
            continue
        prompt = _normalize_space(row["question"]) + "\nAnswer:"
        choices = [" " + _normalize_space(choice) for choice in texts]
        label = labels.index(answer)
        yield Example(prompt=prompt, choices=choices, label=label)


def load_arc_easy(split: str) -> Iterable[Example]:
    return _load_arc("ARC-Easy", split)


def load_arc_challenge(split: str) -> Iterable[Example]:
    return _load_arc("ARC-Challenge", split)


def load_winogrande(split: str) -> Iterable[Example]:
    ds = _load_hf_dataset("winogrande", "winogrande_xl", split=split)
    for row in ds:
        sentence = row["sentence"]
        if "_" not in sentence:
            continue
        prefix, suffix = sentence.split("_", 1)
        prompt = prefix
        choices = [row["option1"] + suffix, row["option2"] + suffix]
        label = int(row["answer"]) - 1
        yield Example(prompt=prompt, choices=choices, label=label)


TASKS: dict[str, TaskSpec] = {
    "hellaswag": TaskSpec(name="hellaswag", loader=load_hellaswag),
    "piqa": TaskSpec(name="piqa", loader=load_piqa),
    "arc_easy": TaskSpec(name="arc_easy", loader=load_arc_easy),
    "arc_challenge": TaskSpec(name="arc_challenge", loader=load_arc_challenge),
    "winogrande": TaskSpec(name="winogrande", loader=load_winogrande),
}


def sequence_logprob(model, tokenizer, prompt: str, continuation: str, length_normalize: bool) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False).input_ids

    if len(full_ids) <= len(prompt_ids):
        return float("-inf")

    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model_device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits

    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)

    start = max(len(prompt_ids) - 1, 0)
    token_log_probs = log_probs[:, start:, :].gather(-1, targets[:, start:].unsqueeze(-1)).squeeze(-1)

    total = float(token_log_probs.sum().item())
    if length_normalize:
        total /= max(int(token_log_probs.numel()), 1)
    return total


def evaluate_task(
    model,
    tokenizer,
    task: TaskSpec,
    *,
    split: str,
    limit: int,
    length_normalize: bool,
    log_every: int,
):
    examples = list(task.loader(split))
    if limit > 0:
        examples = examples[:limit]

    correct = 0
    total = 0
    mean_choice_count = 0.0

    for idx, ex in enumerate(tqdm(examples, desc=task.name), start=1):
        scores = [
            sequence_logprob(model, tokenizer, ex.prompt, choice, length_normalize=length_normalize)
            for choice in ex.choices
        ]
        pred = int(torch.tensor(scores).argmax().item())
        correct += int(pred == ex.label)
        total += 1
        mean_choice_count += len(ex.choices)

        if log_every > 0 and idx % log_every == 0:
            print(
                f"[eval] task={task.name} step={idx}/{len(examples)} "
                f"acc_so_far={correct / max(total, 1):.4f}"
            )

    accuracy = correct / max(total, 1)
    chance = 1.0 / max(mean_choice_count / max(total, 1), 1.0)
    return {
        "task": task.name,
        "total": total,
        "accuracy": accuracy,
        "chance": chance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot MCQ evaluation for prefill-mask ablations")
    parser.add_argument("--model-id", required=True, help="HF model ID or local model path")
    parser.add_argument(
        "--tasks",
        default="hellaswag,piqa,arc_easy,arc_challenge,winogrande",
        help="Comma-separated task list",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=500, help="Per-task example limit. <=0 means full split")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prefill-bidirectional", action="store_true")
    parser.add_argument("--length-normalize", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-json", default="artifacts/eval/latest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
    )
    model.eval()

    patch = None
    if args.prefill_bidirectional:
        patch = apply_prefill_bidirectional_patch(model)

    selected_tasks = []
    for name in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}. Available: {sorted(TASKS)}")
        selected_tasks.append(TASKS[name])

    results = []
    for task in selected_tasks:
        metrics = evaluate_task(
            model,
            tokenizer,
            task,
            split=args.split,
            limit=args.limit,
            length_normalize=args.length_normalize,
            log_every=args.log_every,
        )
        results.append(metrics)

    macro = sum(item["accuracy"] for item in results) / max(len(results), 1)
    summary = {
        "model_id": args.model_id,
        "prefill_bidirectional": args.prefill_bidirectional,
        "tasks": results,
        "macro_accuracy": macro,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote metrics to {out_path}")
    print(json.dumps(summary, indent=2))

    if patch is not None:
        patch.remove()


if __name__ == "__main__":
    main()
