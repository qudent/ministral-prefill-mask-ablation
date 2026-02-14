from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from prefill_ablation.attention_ablation import apply_prefill_bidirectional_patch
from prefill_ablation.utils import parse_dtype, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-weight SFT for prefill-mask ablation experiments")
    parser.add_argument("--model-id", required=True, help="Base model id/path")
    parser.add_argument("--dataset-id", default="yahma/alpaca-cleaned")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--eval-samples", type=int, default=1000)

    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)

    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--optim", default="adafactor")
    parser.add_argument("--lr-scheduler-type", default="cosine")

    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prefill-bidirectional-train", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=200)

    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--kill-after-steps", type=int, default=150)
    parser.add_argument("--min-loss-improvement", type=float, default=0.08)

    return parser.parse_args()


def _format_alpaca_record(row: dict) -> tuple[str, str]:
    instruction = str(row.get("instruction", "")).strip()
    context = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()

    if not instruction or not output:
        raise ValueError("Missing instruction/output")

    if context:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{context}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )

    return prompt, output


def _encode_record(row: dict, tokenizer, max_seq_len: int):
    prompt, response = _format_alpaca_record(row)

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response + tokenizer.eos_token, add_special_tokens=False).input_ids

    input_ids = (prompt_ids + response_ids)[:max_seq_len]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_seq_len]

    if all(x == -100 for x in labels):
        return {"input_ids": [], "labels": [], "attention_mask": []}

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def _load_splits(dataset_id: str, dataset_config: str | None, seed: int) -> tuple[Dataset, Dataset]:
    ds = load_dataset(dataset_id, dataset_config)

    if isinstance(ds, DatasetDict):
        if "train" in ds:
            train = ds["train"]
        else:
            first_key = next(iter(ds.keys()))
            train = ds[first_key]

        if "validation" in ds:
            eval_ds = ds["validation"]
        elif "test" in ds:
            eval_ds = ds["test"]
        else:
            split = train.train_test_split(test_size=0.02, seed=seed)
            train, eval_ds = split["train"], split["test"]
    else:
        split = ds.train_test_split(test_size=0.02, seed=seed)
        train, eval_ds = split["train"], split["test"]

    return train, eval_ds


class SupervisedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]):
        input_features = [
            {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]}
            for x in features
        ]
        batch = self.tokenizer.pad(input_features, return_tensors="pt")

        max_len = batch["input_ids"].shape[1]
        labels = torch.full((len(features), max_len), fill_value=-100, dtype=torch.long)
        for i, feat in enumerate(features):
            seq = torch.tensor(feat["labels"], dtype=torch.long)
            labels[i, : seq.shape[0]] = seq

        batch["labels"] = labels
        return batch


@dataclass
class KillState:
    initial_loss: float | None = None
    best_loss: float | None = None


class KillIfNoLearningCallback(TrainerCallback):
    def __init__(self, kill_after_steps: int, min_loss_improvement: float):
        self.kill_after_steps = kill_after_steps
        self.min_loss_improvement = min_loss_improvement
        self.state = KillState()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs:
            return

        loss = float(logs["loss"])
        if self.state.initial_loss is None:
            self.state.initial_loss = loss
            self.state.best_loss = loss
            return

        self.state.best_loss = min(float(self.state.best_loss), loss)

        if state.global_step >= self.kill_after_steps:
            gained = float(self.state.initial_loss - float(self.state.best_loss))
            if gained < self.min_loss_improvement:
                print(
                    "[kill] stopping early: "
                    f"step={state.global_step} gain={gained:.4f} "
                    f"required={self.min_loss_improvement:.4f}"
                )
                control.should_training_stop = True


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_ds, eval_ds = _load_splits(args.dataset_id, args.dataset_config, args.seed)

    if args.train_samples > 0:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.train_samples, len(train_ds))))
    if args.eval_samples > 0:
        eval_ds = eval_ds.shuffle(seed=args.seed).select(range(min(args.eval_samples, len(eval_ds))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = train_ds.map(
        lambda row: _encode_record(row, tokenizer, args.max_seq_len),
        remove_columns=train_ds.column_names,
        desc="Tokenizing train split",
    )
    eval_ds = eval_ds.map(
        lambda row: _encode_record(row, tokenizer, args.max_seq_len),
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval split",
    )

    train_ds = train_ds.filter(lambda row: len(row["input_ids"]) > 0)
    eval_ds = eval_ds.filter(lambda row: len(row["input_ids"]) > 0)

    model = None
    load_errors: list[Exception] = []

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=parse_dtype(args.dtype),
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
        print("[model] loaded with AutoModelForCausalLM")
    except Exception as exc:
        load_errors.append(exc)

    if model is None:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_id,
                torch_dtype=parse_dtype(args.dtype),
                trust_remote_code=args.trust_remote_code,
                attn_implementation=args.attn_implementation,
            )
            print("[model] loaded with AutoModelForImageTextToText")
        except Exception as exc:
            load_errors.append(exc)

    if model is None:
        details = " | ".join(str(e) for e in load_errors)
        raise RuntimeError(f"Failed to load model {args.model_id}: {details}")

    patch = None
    if args.prefill_bidirectional_train:
        patch = apply_prefill_bidirectional_patch(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args_kwargs = {
        "output_dir": str(output_dir),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "optim": args.optim,
        "lr_scheduler_type": args.lr_scheduler_type,
        "bf16": args.dtype.lower() in {"bf16", "bfloat16"},
        "fp16": args.dtype.lower() in {"fp16", "float16"},
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": [],
        "remove_unused_columns": False,
    }
    # transformers API moved evaluation_strategy -> eval_strategy in newer releases.
    if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        training_args_kwargs["evaluation_strategy"] = "steps"
    else:
        training_args_kwargs["eval_strategy"] = "steps"

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=SupervisedDataCollator(tokenizer),
        callbacks=[KillIfNoLearningCallback(args.kill_after_steps, args.min_loss_improvement)],
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    summary = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "prefill_bidirectional_train": args.prefill_bidirectional_train,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote training summary to {output_dir / 'summary.json'}")

    if patch is not None:
        patch.remove()


if __name__ == "__main__":
    main()
