#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Setup:
    name: str
    gpu_name: str
    num_gpus: int
    single_gpu_speed_factor_vs_3090: float
    min_disk_gb: int


@dataclass
class Offer:
    offer_id: int | None
    dph_total: float | None
    location: str | None


PRESETS: list[Setup] = [
    Setup("1x RTX_3090", "RTX_3090", 1, 1.00, 120),
    Setup("1x RTX_4090", "RTX_4090", 1, 1.60, 120),
    Setup("1x A100_SXM4", "A100_SXM4", 1, 2.00, 120),
    Setup("8x H100_PCIE", "H100_PCIE", 8, 3.20, 500),
    Setup("8x H100_SXM", "H100_SXM", 8, 4.00, 500),
    Setup("8x H100_NVL", "H100_NVL", 8, 3.80, 500),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate Stage 3 wall-clock and cost across Vast setups."
    )
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--causal-sec-per-step", type=float, default=3.318)
    p.add_argument("--bidir-sec-per-step", type=float, default=3.908)
    p.add_argument(
        "--startup-minutes",
        type=float,
        default=10.0,
        help="Per-run fixed overhead: data prep + model load + warmup.",
    )
    p.add_argument(
        "--reliability-min",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--inet-down-min",
        type=float,
        default=100.0,
    )
    p.add_argument(
        "--no-live-offers",
        action="store_true",
        help="Skip Vast query and print time-only estimates.",
    )
    return p.parse_args()


def parallel_efficiency(num_gpus: int) -> float:
    # Constant-effective-batch assumption:
    # reduce grad accumulation as GPUs increase.
    if num_gpus <= 1:
        return 1.0
    if num_gpus <= 2:
        return 0.90
    if num_gpus <= 4:
        return 0.85
    if num_gpus <= 8:
        return 0.78
    return 0.72


def speedup_vs_3090(setup: Setup) -> float:
    return setup.single_gpu_speed_factor_vs_3090 * setup.num_gpus * parallel_efficiency(setup.num_gpus)


def find_cheapest_offer(setup: Setup, reliability_min: float, inet_down_min: float) -> Offer:
    if shutil.which("vastai") is None:
        return Offer(None, None, None)

    query = (
        f"gpu_name={setup.gpu_name} num_gpus={setup.num_gpus} "
        f"reliability>{reliability_min} rentable=True "
        f"inet_down>{inet_down_min} disk_space>={setup.min_disk_gb}"
    )
    cmd = ["vastai", "search", "offers", query, "-o", "dph", "--limit", "1", "--raw"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return Offer(None, None, None)
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return Offer(None, None, None)
    if not rows:
        return Offer(None, None, None)

    row = rows[0]
    return Offer(
        offer_id=int(row.get("id")) if row.get("id") is not None else None,
        dph_total=float(row.get("dph_total")) if row.get("dph_total") is not None else None,
        location=row.get("geolocation"),
    )


def fmt(x: float | None, nd: int = 2) -> str:
    if x is None or math.isnan(x):
        return "-"
    return f"{x:.{nd}f}"


def main() -> None:
    args = parse_args()

    print("Assumption: constant effective batch across GPU counts (adjust grad accumulation with N GPUs).")
    print("Modeling: runtime ~= startup + (measured_train_time / speedup).\n")

    headers = [
        "setup",
        "speedup",
        "causal_h",
        "bidir_h",
        "pair_parallel_h",
        "offer_id",
        "$/h",
        "pair_parallel_$",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))

    for setup in PRESETS:
        sp = speedup_vs_3090(setup)
        causal_h = args.startup_minutes / 60.0 + (args.steps * args.causal_sec_per_step) / (3600.0 * sp)
        bidir_h = args.startup_minutes / 60.0 + (args.steps * args.bidir_sec_per_step) / (3600.0 * sp)
        pair_parallel_h = max(causal_h, bidir_h)

        offer = Offer(None, None, None)
        if not args.no_live_offers:
            offer = find_cheapest_offer(setup, args.reliability_min, args.inet_down_min)

        pair_parallel_cost = None
        if offer.dph_total is not None:
            # Two simultaneous runs (3A + 3B), same setup type.
            pair_parallel_cost = offer.dph_total * (causal_h + bidir_h)

        row = [
            setup.name,
            fmt(sp, 2),
            fmt(causal_h, 2),
            fmt(bidir_h, 2),
            fmt(pair_parallel_h, 2),
            str(offer.offer_id) if offer.offer_id is not None else "-",
            fmt(offer.dph_total, 2),
            fmt(pair_parallel_cost, 2),
        ]
        print(" | ".join(row))

    print("\nNotes:")
    print("- Pair metrics assume 3A and 3B run on separate instances in parallel.")
    print("- Costs use cheapest currently visible offer; availability can change quickly.")
    print("- For fixed hyperparameters across GPU counts, throughput scaling can be worse than shown.")


if __name__ == "__main__":
    main()
