from __future__ import annotations

import json
import random
from pathlib import Path
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


def _load_model_only(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype,
    attn_implementation: str,
    trust_remote_code: bool,
    device_map: Optional[str],
):
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
    except Exception as exc:
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

    return model


def _load_checkpoint_meta(model_name_or_path: str) -> dict | None:
    meta_path = Path(model_name_or_path) / "checkpoint_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to parse checkpoint metadata at {meta_path}: {exc}") from exc


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = False,
    device_map: Optional[str] = "auto",
):
    torch_dtype = parse_dtype(dtype)
    checkpoint_meta = _load_checkpoint_meta(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if checkpoint_meta and checkpoint_meta.get("format") == "raw_state_dict":
        base_model_id = str(checkpoint_meta.get("base_model_id", "")).strip()
        if not base_model_id:
            raise RuntimeError(
                f"Checkpoint at {model_name_or_path} is raw_state_dict but missing base_model_id"
            )

        print(f"[model] loading base model for raw_state_dict checkpoint: {base_model_id}")
        model = _load_model_only(
            base_model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )

        state_path = Path(model_name_or_path) / "pytorch_model.bin"
        if not state_path.exists():
            raise RuntimeError(f"Missing raw checkpoint weights at {state_path}")

        state_dict = torch.load(state_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            "[model] loaded raw_state_dict checkpoint "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        model = _load_model_only(
            model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        )

    return model, tokenizer
