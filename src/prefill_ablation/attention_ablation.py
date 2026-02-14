from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import types

from torch import nn


@dataclass
class _PatchRecord:
    module: nn.Module
    original_forward: Callable


class PrefillBidirectionalPatch:
    """Handle for a temporary patch that removes causal masking during prefill."""

    def __init__(self, records: list[_PatchRecord]):
        self._records = records

    @property
    def patched_module_count(self) -> int:
        return len(self._records)

    def remove(self) -> None:
        """Restore original attention forward methods."""
        for record in self._records:
            record.module.forward = record.original_forward
            if hasattr(record.module, "_prefill_bidirectional_patch"):
                delattr(record.module, "_prefill_bidirectional_patch")


def _is_prefill_call(hidden_states) -> bool:
    # In generation, prefill has q_len > 1. Decode steps are usually q_len == 1.
    if hidden_states is None or hidden_states.ndim < 2:
        return False
    return int(hidden_states.shape[-2]) > 1


def _build_wrapped_forward(original_forward: Callable) -> Callable:
    def wrapped_forward(self, hidden_states, *args, **kwargs):
        should_remove_causal = _is_prefill_call(hidden_states)
        prior_is_causal = getattr(self, "is_causal", None)

        if should_remove_causal and isinstance(prior_is_causal, bool):
            self.is_causal = False

        try:
            return original_forward(hidden_states, *args, **kwargs)
        finally:
            if should_remove_causal and isinstance(prior_is_causal, bool):
                self.is_causal = prior_is_causal

    return wrapped_forward


def apply_prefill_bidirectional_patch(model: nn.Module, *, verbose: bool = True) -> PrefillBidirectionalPatch:
    """Patch attention modules so prefill uses bidirectional attention.

    This patch targets modules with both:
    - class name containing "attention"
    - boolean attribute `is_causal`

    During forward, if q_len > 1 the module's `is_causal` is set to False for that call.
    """

    records: list[_PatchRecord] = []

    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        if "attention" not in class_name:
            continue
        if not hasattr(module, "is_causal") or not isinstance(getattr(module, "is_causal"), bool):
            continue
        if getattr(module, "_prefill_bidirectional_patch", False):
            continue

        original_forward = module.forward
        wrapped_forward = _build_wrapped_forward(original_forward)
        module.forward = types.MethodType(wrapped_forward, module)
        setattr(module, "_prefill_bidirectional_patch", True)
        records.append(_PatchRecord(module=module, original_forward=original_forward))

    patch = PrefillBidirectionalPatch(records)
    if verbose:
        print(f"[patch] applied prefill bidirectional mask ablation to {patch.patched_module_count} modules")
    return patch
