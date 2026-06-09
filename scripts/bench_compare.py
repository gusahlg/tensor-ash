#!/usr/bin/env python3
"""Compare tensor-ash GEMM throughput with major local array/AI frameworks.

The script intentionally keeps dependencies light: NumPy and PyTorch are
optional, and missing libraries are reported as skipped instead of failing the
whole benchmark. It writes both raw JSON and a compact Markdown analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BASE_CASES = [
    ("square_128", 1, 128, 128, 128),
    ("square_256", 1, 256, 256, 256),
    ("square_512", 1, 512, 512, 512),
    ("batched_4x256", 4, 256, 256, 256),
    ("tall_512x256x256", 1, 512, 256, 256),
    ("odd_255x257x263", 1, 255, 257, 263),
]

EXTENDED_CASES = [
    *BASE_CASES,
    ("square_1024", 1, 1024, 1024, 1024),
    ("batched_2x512", 2, 512, 512, 512),
    ("skinny_1024x128x512", 1, 1024, 128, 512),
    ("wide_128x1024x512", 1, 128, 1024, 512),
    ("small_k_1024x1024x64", 1, 1024, 1024, 64),
]

CASE_SETS = {
    "base": BASE_CASES,
    "extended": EXTENDED_CASES,
}


@dataclass
class BenchResult:
    library: str
    case: str
    b: int
    m: int
    n: int
    k: int
    status: str
    tflops: float | None = None
    best_ms: float | None = None
    flops: float | None = None
    details: str = ""


@dataclass
class TransferResult:
    status: str
    bytes: int | None = None
    iters: int | None = None
    upload_gibs: float | None = None
    download_gibs: float | None = None
    details: str = ""


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


def flops_for(b: int, m: int, n: int, k: int) -> float:
    return 2.0 * b * m * n * k


def skipped_results(
    library: str,
    cases: list[tuple[str, int, int, int, int]],
    details: str,
) -> list[BenchResult]:
    return [
        BenchResult(
            library,
            label,
            b,
            m,
            n,
            k,
            "skipped",
            flops=flops_for(b, m, n, k),
            details=details,
        )
        for label, b, m, n, k in cases
    ]


def run_cmd(args: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    return proc.returncode, proc.stdout


def ensure_ml_bench(path: str, skip_build: bool) -> None:
    if skip_build and Path(path).exists():
        return
    code, out = run_cmd(["cargo", "build", "--release", "--bin", "ml_bench"])
    if code != 0:
        raise RuntimeError(out)


def configure_cpu_threads(threads: int) -> None:
    value = str(max(1, threads))
    for name in [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[name] = value


def tensor_ash_self_check(path: str) -> str:
    code, out = run_cmd([path, "self-check"])
    if code != 0:
        return f"FAILED:\n{out.strip()}"
    return out.strip()


def bench_tensor_ash(path: str, cases: list[tuple[str, int, int, int, int]], iters: int, warmup: int) -> list[BenchResult]:
    results: list[BenchResult] = []
    for label, b, m, n, k in cases:
        env = os.environ.copy()
        env.update(
            {
                "ML_B": str(b),
                "ML_M": str(m),
                "ML_N": str(n),
                "ML_K": str(k),
                "ML_ITERS": str(iters),
                "ML_WARMUP": str(warmup),
                "ML_OUTPUT": "csv",
            }
        )
        code, out = run_cmd([path, "single"], env=env)
        flops = flops_for(b, m, n, k)
        if code != 0:
            results.append(BenchResult("tensor-ash", label, b, m, n, k, "failed", flops=flops, details=out.strip()))
            continue

        lines = out.splitlines()
        try:
            header_idx = next(i for i, line in enumerate(lines) if line.startswith("device,kind,label"))
            reader = csv.DictReader(lines[header_idx:])
            row = next(reader)
            results.append(
                BenchResult(
                    "tensor-ash",
                    label,
                    b,
                    m,
                    n,
                    k,
                    "ok",
                    tflops=float(row["tflops"]),
                    best_ms=float(row["gpu_ms"]),
                    flops=flops,
                    details=f"{row['device']} ({row['kind']})",
                )
            )
        except Exception as exc:  # noqa: BLE001 - report parser failures in benchmark output
            results.append(BenchResult("tensor-ash", label, b, m, n, k, "failed", flops=flops, details=f"{exc}\n{out}"))
    return results


def bench_transfer(path: str, iters: int, mb: int) -> TransferResult:
    env = os.environ.copy()
    env.update(
        {
            "ML_OUTPUT": "csv",
            "ML_ITERS": str(max(1, iters)),
            "ML_TRANSFER_MB": str(max(1, mb)),
        }
    )
    code, out = run_cmd([path, "transfer"], env=env)
    if code != 0:
        return TransferResult("failed", details=out.strip())

    lines = out.splitlines()
    try:
        header_idx = next(i for i, line in enumerate(lines) if line.startswith("device,kind,bytes"))
        reader = csv.DictReader(lines[header_idx:])
        row = next(reader)
        return TransferResult(
            "ok",
            bytes=int(row["bytes"]),
            iters=int(row["iters"]),
            upload_gibs=float(row["upload_gibs"]),
            download_gibs=float(row["download_gibs"]),
            details=f"{row['device']} ({row['kind']})",
        )
    except Exception as exc:  # noqa: BLE001 - keep benchmark failures in the report
        return TransferResult("failed", details=f"{exc}\n{out}")


def bench_numpy(cases: list[tuple[str, int, int, int, int]], iters: int, warmup: int) -> list[BenchResult]:
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return skipped_results("numpy", cases, str(exc))

    rng = np.random.default_rng(1234)
    results: list[BenchResult] = []
    for label, b, m, n, k in cases:
        shape_a = (m, k) if b == 1 else (b, m, k)
        shape_b = (k, n) if b == 1 else (b, k, n)
        a = rng.standard_normal(shape_a, dtype=np.float32)
        bb = rng.standard_normal(shape_b, dtype=np.float32)
        for _ in range(warmup):
            c = np.matmul(a, bb)
            float(c.reshape(-1)[0])

        best = float("inf")
        for _ in range(iters):
            t0 = time.perf_counter()
            c = np.matmul(a, bb)
            float(c.reshape(-1)[0])
            best = min(best, time.perf_counter() - t0)

        flops = flops_for(b, m, n, k)
        results.append(
            BenchResult(
                "numpy",
                label,
                b,
                m,
                n,
                k,
                "ok",
                tflops=flops / best * 1e-12,
                best_ms=best * 1000.0,
                flops=flops,
                details=f"numpy {np.__version__}, threads={os.environ.get('OPENBLAS_NUM_THREADS', 'env')}",
            )
        )
    return results


def bench_torch_cpu(
    cases: list[tuple[str, int, int, int, int]],
    iters: int,
    warmup: int,
    threads: int,
) -> list[BenchResult]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return skipped_results("torch_cpu", cases, str(exc))

    torch.set_num_threads(max(1, threads))
    gen = torch.Generator(device="cpu").manual_seed(1234)
    results: list[BenchResult] = []
    for label, b, m, n, k in cases:
        shape_a = (m, k) if b == 1 else (b, m, k)
        shape_b = (k, n) if b == 1 else (b, k, n)
        a = torch.randn(shape_a, dtype=torch.float32, generator=gen)
        bb = torch.randn(shape_b, dtype=torch.float32, generator=gen)
        for _ in range(warmup):
            c = torch.matmul(a, bb)
            float(c.reshape(-1)[0])

        best = float("inf")
        for _ in range(iters):
            t0 = time.perf_counter()
            c = torch.matmul(a, bb)
            float(c.reshape(-1)[0])
            best = min(best, time.perf_counter() - t0)

        flops = flops_for(b, m, n, k)
        results.append(
            BenchResult(
                "torch_cpu",
                label,
                b,
                m,
                n,
                k,
                "ok",
                tflops=flops / best * 1e-12,
                best_ms=best * 1000.0,
                flops=flops,
                details=f"torch {torch.__version__}, threads={torch.get_num_threads()}",
            )
        )
    return results


def bench_torch_cuda(
    cases: list[tuple[str, int, int, int, int]],
    iters: int,
    warmup: int,
) -> list[BenchResult]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return skipped_results("torch_cuda", cases, str(exc))
    if not torch.cuda.is_available():
        return skipped_results("torch_cuda", cases, "CUDA unavailable in this Python environment")

    torch.manual_seed(1234)
    results: list[BenchResult] = []
    for label, b, m, n, k in cases:
        shape_a = (m, k) if b == 1 else (b, m, k)
        shape_b = (k, n) if b == 1 else (b, k, n)
        a = torch.randn(shape_a, dtype=torch.float32, device="cuda")
        bb = torch.randn(shape_b, dtype=torch.float32, device="cuda")
        for _ in range(warmup):
            c = torch.matmul(a, bb)
            float(c.reshape(-1)[0].cpu())
        torch.cuda.synchronize()

        best_ms = float("inf")
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            c = torch.matmul(a, bb)
            end.record()
            torch.cuda.synchronize()
            float(c.reshape(-1)[0].cpu())
            best_ms = min(best_ms, start.elapsed_time(end))

        flops = flops_for(b, m, n, k)
        results.append(
            BenchResult(
                "torch_cuda",
                label,
                b,
                m,
                n,
                k,
                "ok",
                tflops=flops / (best_ms / 1000.0) * 1e-12,
                best_ms=best_ms,
                flops=flops,
                details=f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}",
            )
        )
    return results


def bench_jax(cases: list[tuple[str, int, int, int, int]], iters: int, warmup: int) -> list[BenchResult]:
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # noqa: BLE001
        return skipped_results("jax", cases, str(exc))

    key = jax.random.PRNGKey(1234)
    results: list[BenchResult] = []
    for label, b, m, n, k in cases:
        shape_a = (m, k) if b == 1 else (b, m, k)
        shape_b = (k, n) if b == 1 else (b, k, n)
        key, key_a, key_b = jax.random.split(key, 3)
        a = jax.random.normal(key_a, shape_a, dtype=jnp.float32)
        bb = jax.random.normal(key_b, shape_b, dtype=jnp.float32)
        for _ in range(warmup):
            jnp.matmul(a, bb).block_until_ready()

        best = float("inf")
        for _ in range(iters):
            t0 = time.perf_counter()
            jnp.matmul(a, bb).block_until_ready()
            best = min(best, time.perf_counter() - t0)

        flops = flops_for(b, m, n, k)
        results.append(
            BenchResult(
                "jax",
                label,
                b,
                m,
                n,
                k,
                "ok",
                tflops=flops / best * 1e-12,
                best_ms=best * 1000.0,
                flops=flops,
                details=f"jax {jax.__version__}, backend={jax.default_backend()}",
            )
        )
    return results


def bench_tensorflow(cases: list[tuple[str, int, int, int, int]], iters: int, warmup: int) -> list[BenchResult]:
    try:
        import tensorflow as tf
    except Exception as exc:  # noqa: BLE001
        return skipped_results("tensorflow", cases, str(exc))

    rng = tf.random.Generator.from_seed(1234)
    devices = ",".join(device.device_type for device in tf.config.list_logical_devices()) or "unknown"
    results: list[BenchResult] = []
    for label, b, m, n, k in cases:
        shape_a = (m, k) if b == 1 else (b, m, k)
        shape_b = (k, n) if b == 1 else (b, k, n)
        a = rng.normal(shape_a, dtype=tf.float32)
        bb = rng.normal(shape_b, dtype=tf.float32)
        for _ in range(warmup):
            c = tf.linalg.matmul(a, bb)
            float(tf.reshape(c, [-1])[0].numpy())

        best = float("inf")
        for _ in range(iters):
            t0 = time.perf_counter()
            c = tf.linalg.matmul(a, bb)
            float(tf.reshape(c, [-1])[0].numpy())
            best = min(best, time.perf_counter() - t0)

        flops = flops_for(b, m, n, k)
        results.append(
            BenchResult(
                "tensorflow",
                label,
                b,
                m,
                n,
                k,
                "ok",
                tflops=flops / best * 1e-12,
                best_ms=best * 1000.0,
                flops=flops,
                details=f"tensorflow {tf.__version__}, devices={devices}",
            )
        )
    return results


def best_by_case(results: list[BenchResult]) -> dict[str, BenchResult]:
    best: dict[str, BenchResult] = {}
    for result in results:
        if result.status != "ok" or result.tflops is None:
            continue
        prior = best.get(result.case)
        if prior is None or (prior.tflops or 0.0) < result.tflops:
            best[result.case] = result
    return best


def successful_by_library(results: list[BenchResult], library: str) -> dict[str, BenchResult]:
    return {
        result.case: result
        for result in results
        if result.library == library and result.status == "ok" and result.tflops
    }


def write_outputs(
    path_json: str,
    path_md: str,
    self_check: str,
    results: list[BenchResult],
    transfer: TransferResult | None,
    args: argparse.Namespace,
) -> None:
    Path(path_json).parent.mkdir(parents=True, exist_ok=True)
    Path(path_md).parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "iters": args.iters,
            "warmup": args.warmup,
            "case_set": args.case_set,
            "cpu_threads": args.torch_threads,
            "torch_threads": args.torch_threads,
            "self_check": self_check,
        },
        "transfer": None if transfer is None else asdict(transfer),
        "results": [asdict(result) for result in results],
    }
    Path(path_json).write_text(json.dumps(payload, indent=2) + "\n")

    best = best_by_case(results)
    self_check_lc = self_check.lower()
    software_vulkan = "llvmpipe" in self_check_lc or "(cpu" in self_check_lc
    ml_ok = [r for r in results if r.library == "tensor-ash" and r.status == "ok" and r.tflops]
    ml_details = next((r.details for r in ml_ok if r.details), "")
    lines: list[str] = [
        "# Benchmark Report",
        "",
        "This report compares FP32 GEMM throughput for `tensor-ash` against the local framework backends available on this machine.",
        "",
        "## Environment",
        "",
        "```text",
        self_check,
        "```",
        "",
        f"- Iterations: {args.iters}",
        f"- Warmup iterations: {args.warmup}",
        f"- Case set: {args.case_set}",
        f"- CPU library threads: {args.torch_threads}",
        "",
        "## Results",
        "",
        "| case | library | status | best ms | TFLOPS | details |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        best_ms = "" if result.best_ms is None else f"{result.best_ms:.3f}"
        tflops = "" if result.tflops is None else f"{result.tflops:.6f}"
        lines.append(f"| {result.case} | {result.library} | {result.status} | {best_ms} | {tflops} | {result.details.replace('|', '/')} |")

    if transfer is not None:
        lines.extend(
            [
                "",
                "## Transfer",
                "",
                "| status | bytes | iters | upload GiB/s | download GiB/s | details |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        transfer_bytes = "" if transfer.bytes is None else str(transfer.bytes)
        transfer_iters = "" if transfer.iters is None else str(transfer.iters)
        upload = "" if transfer.upload_gibs is None else f"{transfer.upload_gibs:.3f}"
        download = "" if transfer.download_gibs is None else f"{transfer.download_gibs:.3f}"
        lines.append(
            f"| {transfer.status} | {transfer_bytes} | {transfer_iters} | "
            f"{upload} | {download} | {transfer.details.replace('|', '/')} |"
        )

    lines.extend(["", "## Analysis", ""])
    if software_vulkan:
        lines.append("- `tensor-ash` selected CPU/software Vulkan (`llvmpipe`), so these are correctness and overhead measurements, not real GPU performance numbers.")
    elif ml_details:
        lines.append(f"- `tensor-ash` used `{ml_details}`, so the Vulkan measurements reflect real GPU kernel timings on this host.")
    if ml_ok:
        ratios = []
        wins = 0
        for result in ml_ok:
            winner = best.get(result.case)
            if winner and winner.library != result.library and result.tflops:
                ratios.append((result.case, winner.library, (winner.tflops or 0.0) / result.tflops))
            elif winner and winner.library == result.library:
                wins += 1
        if ratios:
            worst = max(ratios, key=lambda item: item[2])
            lines.append(f"- Largest gap: `{worst[0]}` is {worst[2]:.1f}x faster in `{worst[1]}` than `tensor-ash` in this environment.")
        lines.append(f"- `tensor-ash` is the fastest measured backend on {wins}/{len(ml_ok)} benchmark cases.")
        ml_by_case = {result.case: result for result in ml_ok}
        for library in ["numpy", "torch_cpu", "torch_cuda", "jax", "tensorflow"]:
            other_by_case = successful_by_library(results, library)
            shared = sorted(set(ml_by_case) & set(other_by_case))
            if not shared:
                continue
            speedups = [
                (ml_by_case[case].tflops or 0.0) / (other_by_case[case].tflops or 1.0)
                for case in shared
            ]
            geomean = math.prod(speedups) ** (1.0 / len(speedups))
            lines.append(
                f"- Throughput ratio versus `{library}` across {len(shared)} shared cases: "
                f"{min(speedups):.1f}x to {max(speedups):.1f}x, geometric mean {geomean:.1f}x."
            )
    skipped = [r for r in results if r.status == "skipped"]
    if skipped:
        lines.append("- Some libraries were skipped because their Python modules were unavailable; see details in the table.")
    if any(r.library == "torch_cuda" and r.status == "skipped" for r in results):
        lines.append("- PyTorch CUDA/cuBLAS was not available in this Python environment, so CUDA remains a separate benchmark to run when available.")
    if any(r.library in {"jax", "tensorflow"} and r.status == "skipped" for r in results):
        lines.append("- JAX and TensorFlow rows are included when those modules are installed; skipped rows mean the local Python environment does not provide them.")
    if transfer and transfer.status == "ok":
        lines.append(
            f"- Transfer staging bandwidth measured {transfer.upload_gibs:.2f} GiB/s upload "
            f"and {transfer.download_gibs:.2f} GiB/s download for {transfer.bytes} bytes."
        )
    else:
        lines.append("- Transfer overhead is separately measurable with `ml_bench transfer`; use it to distinguish copy overhead from GEMM kernel time.")
    lines.extend(["", "## Optimization Gameplan", ""])
    if software_vulkan:
        lines.append("1. Fix runtime device visibility so `ML_DEVICE=discrete ml_bench self-check` selects the actual GPU.")
        lines.append("2. Re-run this benchmark on the discrete GPU and compare against PyTorch CUDA when available.")
        lines.append("3. Tune shader variants only after measuring real GPU behavior.")
    else:
        lines.append("1. Keep benchmarking on this discrete-GPU baseline and tune the shape-based shader selector with more matrix sizes.")
        lines.append("2. Use `ML_KERNEL=large|small` A/B runs to tune the automatic shader-selector thresholds.")
        lines.append("3. Add more specialized kernels for skinny, wide, and small-K GEMMs.")
    lines.extend(
        [
            "4. Run the optional PyTorch CUDA/cuBLAS comparison when the local Python environment provides CUDA.",
            "5. Add CI checks for `cargo fmt`, `cargo clippy`, CPU tests, and an optional GPU correctness job.",
            "",
        ]
    )
    Path(path_md).write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    cases = CASE_SETS[args.case_set]
    configure_cpu_threads(args.torch_threads)
    ensure_ml_bench(args.ml_bench, args.skip_build)
    self_check = tensor_ash_self_check(args.ml_bench)
    results = []
    results.extend(bench_tensor_ash(args.ml_bench, cases, args.iters, args.warmup))
    transfer = (
        None
        if args.skip_transfer
        else bench_transfer(args.ml_bench, args.iters, args.transfer_mb)
    )
    results.extend(bench_numpy(cases, args.iters, args.warmup))
    results.extend(bench_torch_cpu(cases, args.iters, args.warmup, args.torch_threads))
    results.extend(bench_torch_cuda(cases, args.iters, args.warmup))
    results.extend(bench_jax(cases, args.iters, args.warmup))
    results.extend(bench_tensorflow(cases, args.iters, args.warmup))
    write_outputs(args.output_json, args.output_md, self_check, results, transfer, args)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
