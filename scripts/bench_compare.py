#!/usr/bin/env python3
"""Compare tensor-ash GEMM throughput with local array/AI frameworks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Re-export the established helper API from this entry module for compatibility.
from bench_compare_backends import (
    NVIDIA_SMI_FIELDS,
    bench_cublas_pure,
    bench_cupy_cuda,
    bench_jax,
    bench_numpy,
    bench_tensor_ash,
    bench_tensorflow,
    bench_torch_cpu,
    bench_torch_cuda,
    bench_transfer,
    configure_cpu_threads,
    ensure_ml_bench,
    format_nvidia_smi_summary,
    nvidia_smi_summary,
    run_cmd,
    tensor_ash_self_check,
)
from bench_compare_models import (
    BASE_CASES,
    CASE_SETS,
    EXTENDED_CASES,
    REGRESSION_CASES,
    SHOWCASE_CASES,
    STREAMK_CASES,
    BenchResult,
    Case,
    TransferResult,
    flops_for,
    matrix_shapes,
    skipped_results,
)
from bench_compare_report import (
    PEAK_TFLOPS,
    RATIO_LIBRARY_ORDER,
    best_by_case,
    build_markdown,
    build_payload,
    result_uses_gpu,
    successful_by_library,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-json", default="benchmarks/latest.json")
    parser.add_argument("--output-md", default="benchmarks/latest.md")
    parser.add_argument("--ml-bench", default="target/release/ml_bench")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--transfer-mb", type=int, default=64)
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-cpu-frameworks", action="store_true")
    parser.add_argument("--skip-gpu-frameworks", action="store_true")
    parser.add_argument(
        "--cublas-bench",
        default="benchmarks/cublas_bench/cublas_bench",
        help=(
            "Path to the pure cuBLAS benchmark binary "
            "(built via `cd benchmarks/cublas_bench && make`)."
        ),
    )
    parser.add_argument(
        "--skip-cublas-pure",
        action="store_true",
        help="Skip the pure-cuBLAS C++ baseline row.",
    )
    parser.add_argument("--case-set", choices=sorted(CASE_SETS), default="base")
    args = parser.parse_args()
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be at least 1")
    if args.transfer_mb < 1:
        parser.error("--transfer-mb must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    cases = CASE_SETS[args.case_set]
    configure_cpu_threads(args.torch_threads)
    ensure_ml_bench(args.ml_bench, args.skip_build)
    self_check = tensor_ash_self_check(args.ml_bench)
    nvidia_smi = nvidia_smi_summary()

    results = bench_tensor_ash(args.ml_bench, cases, args.iters, args.warmup)
    transfer = (
        None
        if args.skip_transfer
        else bench_transfer(args.ml_bench, args.iters, args.transfer_mb)
    )
    if not args.skip_cpu_frameworks:
        results.extend(bench_numpy(cases, args.iters, args.warmup))
        results.extend(bench_torch_cpu(cases, args.iters, args.warmup, args.torch_threads))
    if not args.skip_cublas_pure:
        results.extend(bench_cublas_pure(args.cublas_bench, cases, args.iters, args.warmup))
    if not args.skip_gpu_frameworks:
        for runner in (bench_torch_cuda, bench_cupy_cuda, bench_jax, bench_tensorflow):
            results.extend(runner(cases, args.iters, args.warmup))

    write_outputs(
        args.output_json,
        args.output_md,
        self_check,
        nvidia_smi,
        results,
        transfer,
        args,
    )
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
