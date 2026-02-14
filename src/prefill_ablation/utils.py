from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def parse_dtype(dtype: str) -> torch.dtype:
    key = dtype.lower().strip()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return _DTYPE_MAP[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = False,
    device_map: Optional[str] = "auto",
):
    torch_dtype = parse_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    errors: list[Exception] = []

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
        print("[model] loaded with AutoModelForCausalLM")
    except Exception as exc:  # fallback for Ministral-3 models
        errors.append(exc)

    if model is None:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
                device_map=device_map,
            )
            print("[model] loaded with AutoModelForImageTextToText")
        except Exception as exc:
            errors.append(exc)

    if model is None:
        error_summary = " | ".join(str(e) for e in errors)
        raise RuntimeError(f"Failed to load model {model_name_or_path}: {error_summary}")

    return model, tokenizer
