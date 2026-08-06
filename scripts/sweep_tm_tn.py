#!/usr/bin/env python3
"""Interleaved A/B sweep of TM/TN register-tile variants vs the current
default kernel for the cuBLAS-losing target shapes.

For every (shape, variant) the runner alternates rounds A vs B (default vs
variant), takes ``rounds`` measurements per side, and reports the median
TFLOPS. A single-shot win is rejected.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "target" / "release" / "ml_bench"

# (label, b, m, n, k, default_kernel_name)
TARGET_SHAPES = [
    ("square_512", 1, 512, 512, 512, "m128n64k64_bda_v4"),
    ("medium_768", 1, 768, 768, 768, "m128n64k64_bda_v4"),
    ("square_1024", 1, 1024, 1024, 1024, "m128n64k64_bda_v4"),
    ("non_pow2_1023x1025x1027", 1, 1023, 1025, 1027, "m128n64k64_bda_v4"),
    ("medium_384", 1, 384, 384, 384, "k64_bda_v4"),
]

# default → list of variant names to A/B against
VARIANTS = {
    "m128n64k64_bda_v4": [
        "m128n64k64_bda_v4_tm8_tn8",
        "m128n64k64_bda_v4_tm16_tn4",
    ],
    "k64_bda_v4": [
        "k64_bda_v4_tm8_tn4",
        "k64_bda_v4_tm4_tn8",
        "k64_bda_v4_tm8_tn8",
    ],
}

# Guard set (small/skinny/batched shapes where a regression would hurt
# geomean). Used after a variant is shortlisted, with its targeted
# default.
GUARD_SHAPES = [
    ("square_128", 1, 128, 128, 128),
    ("square_256", 1, 256, 256, 256),
    ("batched_4x256", 4, 256, 256, 256),
    ("batched_8x256", 8, 256, 256, 256),
    ("batched_32x64", 32, 64, 64, 64),
    ("attn_proj_2048x512x512", 1, 2048, 512, 512),
    ("attn_qkv_1024x3072x512", 1, 1024, 3072, 512),
    ("batched_64x128", 64, 128, 128, 128),
]

# Shapes currently routed to k64_bda_v4 by the selector — used to verify
# that swapping in a TM/TN variant does not regress those cases.
K64_ROUTED_SHAPES = [
    ("medium_384", 1, 384, 384, 384),
    ("skinny_1024x128x512", 1, 1024, 128, 512),
    ("wide_128x1024x512", 1, 128, 1024, 512),
    # any other shape with min_mn<=128 and max_mn>=512 and k>=256
]


def run_once(kernel: str, b: int, m: int, n: int, k: int, iters: int, warmup: int) -> float:
    env = os.environ.copy()
    env.update(
        ML_KERNEL=kernel,
        ML_B=str(b),
        ML_M=str(m),
        ML_N=str(n),
        ML_K=str(k),
        ML_ITERS=str(iters),
        ML_WARMUP=str(warmup),
        ML_OUTPUT="csv",
    )
    proc = subprocess.run(
        [str(BIN), "single"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{kernel} {b}x{m}x{n}x{k} failed:\n{proc.stdout}")
    lines = proc.stdout.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if line.startswith("device,kind,label")),
        None,
    )
    if header_idx is None:
        raise RuntimeError(f"{kernel}: no CSV in output:\n{proc.stdout}")
    row = next(csv.DictReader(lines[header_idx:]))
    return float(row["tflops"])


def ab_compare(
    default: str,
    variant: str,
    label: str,
    b: int,
    m: int,
    n: int,
    k: int,
    rounds: int,
    iters: int,
    warmup: int,
) -> tuple[float, float, float]:
    """Interleaved A/B: A=default, B=variant, alternating. Return
    (median_default, median_variant, delta_pct)."""
    a_samples: list[float] = []
    b_samples: list[float] = []
    # Pre-warm both kernels (cold-start variance kills first round).
    run_once(default, b, m, n, k, max(5, warmup), warmup)
    run_once(variant, b, m, n, k, max(5, warmup), warmup)
    for _ in range(rounds):
        a_samples.append(run_once(default, b, m, n, k, iters, warmup))
        b_samples.append(run_once(variant, b, m, n, k, iters, warmup))
    # Second pass with B-first ordering to cancel any drift.
    for _ in range(rounds):
        b_samples.append(run_once(variant, b, m, n, k, iters, warmup))
        a_samples.append(run_once(default, b, m, n, k, iters, warmup))
    med_a = statistics.median(a_samples)
    med_b = statistics.median(b_samples)
    delta = (med_b / med_a - 1.0) * 100.0
    return med_a, med_b, delta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5,
                        help="Rounds per side per ordering (total samples = 2*rounds).")
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--shapes", default="all", help="Comma-separated labels or 'all'.")
    parser.add_argument("--mode", choices=("targets", "guard", "k64_routed"), default="targets")
    parser.add_argument("--default-kernel", default=None,
                        help="Override default kernel (guard mode).")
    parser.add_argument("--variant", default=None,
                        help="Single variant to test (guard mode).")
    args = parser.parse_args()

    if not BIN.is_file():
        print(f"binary not found: {BIN}", file=sys.stderr)
        return 1

    if args.mode == "targets":
        shapes = TARGET_SHAPES
        if args.shapes != "all":
            wanted = set(args.shapes.split(","))
            shapes = [s for s in shapes if s[0] in wanted]
        print(f"{'shape':<28s} {'default':<28s} {'variant':<32s} {'TF/s def':>9s} {'TF/s var':>9s} {'delta%':>7s}")
        for label, b, m, n, k, default in shapes:
            for variant in VARIANTS.get(default, []):
                med_a, med_b, delta = ab_compare(
                    default, variant, label, b, m, n, k,
                    args.rounds, args.iters, args.warmup,
                )
                marker = " *" if delta >= 2.0 else (" -" if delta <= -2.0 else "")
                print(f"{label:<28s} {default:<28s} {variant:<32s} {med_a:>9.3f} {med_b:>9.3f} {delta:>+7.2f}{marker}")
    else:
        assert args.default_kernel and args.variant, "guard/k64_routed mode requires --default-kernel and --variant"
        shapes = GUARD_SHAPES if args.mode == "guard" else K64_ROUTED_SHAPES
        print(f"{args.mode} sweep: default={args.default_kernel} variant={args.variant}")
        print(f"{'shape':<28s} {'TF/s def':>9s} {'TF/s var':>9s} {'delta%':>7s}")
        for label, b, m, n, k in shapes:
            med_a, med_b, delta = ab_compare(
                args.default_kernel, args.variant, label, b, m, n, k,
                args.rounds, args.iters, args.warmup,
            )
            marker = " REGRESS" if delta <= -1.0 else ""
            print(f"{label:<28s} {med_a:>9.3f} {med_b:>9.3f} {delta:>+7.2f}{marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
