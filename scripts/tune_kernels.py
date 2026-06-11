#!/usr/bin/env python3
"""Run tensor-ash GEMM cases across every manual kernel override.

This is an implementation/tuning helper.  It prints a compact table to stdout
and deliberately does not update benchmark reports.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bench_compare import CASE_SETS, ensure_ml_bench  # noqa: E402

DEFAULT_KERNELS = [
    "auto",
    "small",
    "large",
    "m64n128",
    "m128n64",
    "m128n64k64",
    "m64n32",
    "k64",
    "bk16",
    "v2",
    "m64n128k64",
    "m128n128_t4",
    "m256n64",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-set", choices=sorted(CASE_SETS), default="extended")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--ml-bench", default="target/release/ml_bench")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--kernels", nargs="+", default=DEFAULT_KERNELS)
    args = parser.parse_args()
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def run_case(path: str, kernel: str, case: tuple[str, int, int, int, int], iters: int, warmup: int) -> dict[str, float | str]:
    label, b, m, n, k = case
    env = os.environ.copy()
    env.update(
        {
            "ML_KERNEL": kernel,
            "ML_B": str(b),
            "ML_M": str(m),
            "ML_N": str(n),
            "ML_K": str(k),
            "ML_ITERS": str(iters),
            "ML_WARMUP": str(warmup),
            "ML_OUTPUT": "csv",
        }
    )
    proc = subprocess.run(
        [path, "single"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{label} {kernel} failed:\n{proc.stdout}")
    lines = proc.stdout.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("device,kind,label"))
    row = next(csv.DictReader(lines[header_idx:]))
    return {
        "case": label,
        "kernel": kernel,
        "gpu_ms": float(row["gpu_ms"]),
        "wall_ms": float(row["wall_ms"]),
        "tflops": float(row["tflops"]),
    }


def main() -> int:
    args = parse_args()
    ensure_ml_bench(args.ml_bench, args.skip_build)
    for case in CASE_SETS[args.case_set]:
        rows = [
            run_case(args.ml_bench, kernel, case, args.iters, args.warmup)
            for kernel in args.kernels
        ]
        best = max(rows, key=lambda row: float(row["tflops"]))
        print(f"{case[0]:<24s} best={best['kernel']:<8s} {best['tflops']:.3f} TF/s")
        for row in rows:
            print(
                f"  {row['kernel']:<8s} gpu={row['gpu_ms']:.6f} "
                f"wall={row['wall_ms']:.6f} tf={row['tflops']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
