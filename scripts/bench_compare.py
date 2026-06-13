#!/usr/bin/env python3
"""Compare tensor-ash GEMM throughput with major local array/AI frameworks.

The script intentionally keeps dependencies light: NumPy and PyTorch are
optional, and missing libraries are reported as skipped instead of failing the
whole benchmark. CUDA-capable Python frameworks are timed with GPU events when
they are available. It writes both raw JSON and a compact Markdown analysis.
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

# "Showcase" set adds scenarios where the kernel-selection-heavy
# design tends to beat cuBLAS more decisively:
#   * non-power-of-2 sizes (cuBLAS heuristic mispicks)
#   * many small batched matmuls (low launch overhead dominates)
#   * mid-range sizes between common power-of-2 lookups (where cuBLAS
#     falls back to a less-specialized kernel)
#   * wide attention-style shapes
SHOWCASE_CASES = [
    *EXTENDED_CASES,
    # Non-power-of-2 medium and large
    ("non_pow2_513x515x517", 1, 513, 515, 517),
    ("non_pow2_1023x1025x1027", 1, 1023, 1025, 1027),
    # Mid-range squares between common power-of-2 sizes
    ("medium_384", 1, 384, 384, 384),
    ("medium_768", 1, 768, 768, 768),
    # Lots of small batched matmuls — tensor-ash's
    # per-call overhead dominance shines as the batch count grows
    ("batched_8x256", 8, 256, 256, 256),
    ("batched_16x128", 16, 128, 128, 128),
    ("batched_32x64", 32, 64, 64, 64),
    ("batched_64x128", 64, 128, 128, 128),
    ("batched_128x64", 128, 64, 64, 64),
    # Attention-style projection shapes
    ("attn_proj_2048x512x512", 1, 2048, 512, 512),
    ("attn_proj_512x2048x512", 1, 512, 2048, 512),
    # Asymmetric attention QKV combined projections
    ("attn_qkv_1024x3072x512", 1, 1024, 3072, 512),
    # Heavy-batch small matmuls — tensor-ash's lower per-call
    # synchronous overhead amortizes much better when the per-matmul
    # work is tiny.  cuBLAS pays a relatively fixed launch cost per
    # batched call.
    ("tiny_b32_128", 32, 128, 128, 128),
    ("tiny_b16_192", 16, 192, 192, 192),
    ("tiny_b8_192", 8, 192, 192, 192),
]

# Stream-K focused shapes: BM=BN=128 tile, 46 SMs on RTX 3070.
# Shapes are picked so (M/128) * (N/128) modulo 92 (=46*2 waves) is
# small but non-zero — i.e. classic wave-quantization tails where
# Stream-K's even work redistribution should beat a fixed data-parallel
# tile mapping. Also includes a deep-K shape where Split-K-style
# accumulation is the natural win.
STREAMK_CASES = [
    ("sq_4096", 1, 4096, 4096, 4096),        # 1024 tiles, mod 92 = 12. Big expected win.
    ("sq_4096_k1024", 1, 4096, 4096, 1024),  # 1024 tiles, K=1024. Same wave quantization.
    ("sq_2048", 1, 2048, 2048, 2048),        # 256 tiles, mod 92 = 72. Modest expected win.
    ("attn_2048x4096x512", 1, 2048, 4096, 512),
    ("attn_4096x2048x512", 1, 4096, 2048, 512),
    ("attn_qkv_2048x6144x1024", 1, 2048, 6144, 1024),
    ("ffn_proj_2048x8192x2048", 1, 2048, 8192, 2048),
    ("sq_3584", 1, 3584, 3584, 3584),        # 784 tiles, mod 92 = 48.
    ("sq_6144", 1, 6144, 6144, 4096),        # 2304 tiles, mod 92 = 8. Big expected win.
    ("deep_k_512_8192", 1, 512, 512, 8192),  # 16 tiles, deep K (Split-K-like win).
]

CASE_SETS = {
    "base": BASE_CASES,
    "extended": EXTENDED_CASES,
    "showcase": SHOWCASE_CASES,
    "streamk": STREAMK_CASES,
}

RATIO_LIBRARY_ORDER = [
    "cublas_pure",
    "torch_cuda",
    "cupy_cuda",
    "jax",
    "tensorflow",
    "numpy",
    "torch_cpu",
]

# Peak FP32 throughput of the device used to author this benchmark
# (RTX 3070: 5888 CUDA cores * 2 FLOPs/cycle * 1.725 GHz ≈ 20.3 TFLOPS).
# Override via env var `ML_PEAK_TFLOPS` if benchmarking on a different
# GPU.  The reporter exposes this as a percent-of-peak column so it's
# obvious how much headroom each row has versus the silicon limit.
PEAK_TFLOPS = float(os.environ.get("ML_PEAK_TFLOPS", "20.32"))

NVIDIA_SMI_FIELDS = [
    "name",
    "driver",
    "memory_total_mib",
    "temperature_c",
    "utilization_pct",
    "power_draw_w",
    "power_limit_w",
]


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
    wall_ms: float | None = None
    host_overhead_ms: float | None = None
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


def format_nvidia_smi_summary(output: str) -> str:
    rows = []
    for idx, line in enumerate(output.splitlines()):
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(NVIDIA_SMI_FIELDS):
            rows.append(
                f"gpu{idx}: "
                + ", ".join(
                    f"{field}={value}"
                    for field, value in zip(NVIDIA_SMI_FIELDS, values, strict=True)
                )
            )
        else:
            rows.append(f"gpu{idx}: {line.strip()}")
    return "\n".join(rows)


def nvidia_smi_summary() -> str:
    try:
        code, out = run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,temperature.gpu,utilization.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ]
        )
    except FileNotFoundError:
        return ""
    if code == 0:
        return format_nvidia_smi_summary(out)
    return ""


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
                    wall_ms=float(row["wall_ms"]),
                    host_overhead_ms=max(0.0, float(row["wall_ms"]) - float(row["gpu_ms"])),
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

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(1234)
    precision_details = (
        f"allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
        f"precision={torch.get_float32_matmul_precision()}"
    )
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
                details=(
                    f"torch {torch.__version__}, CUDA/cuBLAS, "
                    f"{torch.cuda.get_device_name(0)}, {precision_details}"
                ),
            )
        )
    return results


def bench_cupy_cuda(cases: list[tuple[str, int, int, int, int]], iters: int, warmup: int) -> list[BenchResult]:
    try:
        import cupy as cp
    except Exception as exc:  # noqa: BLE001
        return skipped_results("cupy_cuda", cases, str(exc))

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # noqa: BLE001
        return skipped_results("cupy_cuda", cases, f"CUDA device query failed: {exc}")
    if device_count < 1:
        return skipped_results("cupy_cuda", cases, "CUDA unavailable in this Python environment")

    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = props.get("name", "unknown")
        if isinstance(device_name, bytes):
            device_name = device_name.decode(errors="replace")
    except Exception:  # noqa: BLE001
        device_name = "unknown CUDA device"

    cp.random.seed(1234)
    results: list[BenchResult] = []
    for label, b, m, n, k in cases:
        shape_a = (m, k) if b == 1 else (b, m, k)
        shape_b = (k, n) if b == 1 else (b, k, n)
        a = cp.random.standard_normal(shape_a, dtype=cp.float32)
        bb = cp.random.standard_normal(shape_b, dtype=cp.float32)
        for _ in range(warmup):
            c = cp.matmul(a, bb)
            cp.cuda.Stream.null.synchronize()
            float(c.ravel()[0].get())

        best_ms = float("inf")
        for _ in range(iters):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            c = cp.matmul(a, bb)
            end.record()
            end.synchronize()
            float(c.ravel()[0].get())
            best_ms = min(best_ms, cp.cuda.get_elapsed_time(start, end))

        flops = flops_for(b, m, n, k)
        results.append(
            BenchResult(
                "cupy_cuda",
                label,
                b,
                m,
                n,
                k,
                "ok",
                tflops=flops / (best_ms / 1000.0) * 1e-12,
                best_ms=best_ms,
                flops=flops,
                details=f"cupy {cp.__version__}, CUDA/cuBLAS, {device_name}",
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


def bench_cublas_pure(
    binary_path: str,
    cases: list[tuple[str, int, int, int, int]],
    iters: int,
    warmup: int,
) -> list[BenchResult]:
    """Run the pure-cuBLAS C++ benchmark binary on every case.

    The binary timings are pure cuBLAS SGEMM with FP32 forced (no TF32)
    and CUDA event timing — i.e. the same ground rules as ml_bench's
    Vulkan timestamp readings.  No Python wrapper / framework overhead
    is included.
    """
    if not Path(binary_path).is_file():
        return skipped_results(
            "cublas_pure",
            cases,
            (
                f"binary {binary_path} not found "
                "(build with `cd benchmarks/cublas_bench && make`)"
            ),
        )

    stdin_lines = ["label,b,m,n,k"]
    stdin_lines.extend(
        f"{label},{b},{m},{n},{k}" for label, b, m, n, k in cases
    )
    stdin_payload = "\n".join(stdin_lines) + "\n"

    proc = subprocess.run(
        [binary_path, "--iters", str(iters), "--warmup", str(warmup)],
        input=stdin_payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return skipped_results(
            "cublas_pure",
            cases,
            f"binary failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}",
        )

    rows_by_label: dict[str, dict[str, str]] = {}
    reader_lines = proc.stdout.splitlines()
    header_idx = next(
        (
            i
            for i, line in enumerate(reader_lines)
            if line.startswith("label,b,m,n,k,best_ms,mean_ms,tflops")
        ),
        None,
    )
    if header_idx is not None:
        for row in csv.DictReader(reader_lines[header_idx:]):
            rows_by_label[row["label"]] = row

    details = "pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events"
    results: list[BenchResult] = []
    for label, b, m, n, k in cases:
        row = rows_by_label.get(label)
        if row is None:
            results.append(
                BenchResult(
                    "cublas_pure",
                    label,
                    b,
                    m,
                    n,
                    k,
                    "failed",
                    details="binary did not emit a row",
                )
            )
            continue
        best_ms = float(row["best_ms"])
        flops = flops_for(b, m, n, k)
        results.append(
            BenchResult(
                "cublas_pure",
                label,
                b,
                m,
                n,
                k,
                "ok",
                tflops=float(row["tflops"]),
                best_ms=best_ms,
                flops=flops,
                details=details,
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


def result_uses_gpu(result: BenchResult) -> bool:
    if result.library in {"tensor-ash", "torch_cuda", "cupy_cuda", "cublas_pure"}:
        return result.status == "ok"
    details = result.details.lower()
    if result.library == "jax":
        return "backend=gpu" in details or "backend=cuda" in details
    if result.library == "tensorflow":
        return "gpu" in details or "cuda" in details
    return False


def write_outputs(
    path_json: str,
    path_md: str,
    self_check: str,
    nvidia_smi: str,
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
            "nvidia_smi": nvidia_smi,
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
    ]
    if nvidia_smi:
        lines.extend(
            [
                "NVIDIA-SMI GPU summary:",
                "",
                "```text",
                nvidia_smi,
                "```",
                "",
            ]
        )
    lines.extend(
        [
        f"- Iterations: {args.iters}",
        f"- Warmup iterations: {args.warmup}",
        f"- Case set: {args.case_set}",
        f"- CPU library threads: {args.torch_threads}",
        f"- CPU framework rows: {'skipped' if args.skip_cpu_frameworks else 'enabled'}",
        f"- Python GPU framework rows: {'skipped' if args.skip_gpu_frameworks else 'enabled'}",
        "",
        f"- Peak FP32 throughput used for `% peak`: {PEAK_TFLOPS:.2f} TFLOPS",
        "",
        "## Results",
        "",
        "| case | library | status | gpu ms | wall ms | host overhead ms | TFLOPS | % peak | details |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in results:
        best_ms = "" if result.best_ms is None else f"{result.best_ms:.3f}"
        wall_ms = "" if result.wall_ms is None else f"{result.wall_ms:.3f}"
        host_overhead = "" if result.host_overhead_ms is None else f"{result.host_overhead_ms:.3f}"
        tflops = "" if result.tflops is None else f"{result.tflops:.6f}"
        if result.tflops is None or PEAK_TFLOPS <= 0:
            peak_pct = ""
        else:
            peak_pct = f"{(result.tflops / PEAK_TFLOPS) * 100:.1f}%"
        lines.append(
            f"| {result.case} | {result.library} | {result.status} | {best_ms} | "
            f"{wall_ms} | {host_overhead} | {tflops} | {peak_pct} | "
            f"{result.details.replace('|', '/')} |"
        )

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
    gpu_frameworks = sorted(
        {
            result.library
            for result in results
            if result.library != "tensor-ash" and result.status == "ok" and result_uses_gpu(result)
        }
    )
    if gpu_frameworks:
        lines.append(
            "- Actual GPU framework comparisons succeeded for: "
            + ", ".join(f"`{library}`" for library in gpu_frameworks)
            + "."
        )
    else:
        lines.append(
            "- No CUDA/GPU Python framework rows completed successfully; only the Vulkan `tensor-ash` rows used GPU compute in this report."
        )
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
        for library in RATIO_LIBRARY_ORDER:
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
                f"{min(speedups):.2f}x to {max(speedups):.2f}x, geometric mean {geomean:.2f}x."
            )
        overheads = [
            result.host_overhead_ms
            for result in ml_ok
            if result.host_overhead_ms is not None and result.wall_ms is not None
        ]
        if overheads:
            overheads = sorted(overheads)
            median = overheads[len(overheads) // 2]
            lines.append(
                f"- Median `tensor-ash` host/submission overhead was {median:.3f} ms per synchronous call; "
                "GPU timestamp TFLOPS excludes that overhead."
            )
    skipped = [r for r in results if r.status == "skipped"]
    if skipped:
        skipped_libraries = sorted({r.library for r in skipped})
        lines.append(
            "- Some libraries were skipped because their Python modules or device backends were unavailable: "
            + ", ".join(f"`{library}`" for library in skipped_libraries)
            + "."
        )
    if any(r.library == "torch_cuda" and r.status == "skipped" for r in results):
        lines.append("- PyTorch CUDA/cuBLAS was not available in this Python environment.")
    if any(r.library == "cupy_cuda" and r.status == "skipped" for r in results):
        lines.append("- CuPy CUDA/cuBLAS was not available in this Python environment.")
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
        lines.append("1. Keep benchmarking on this discrete-GPU baseline and tune the shape-based shader selector with larger production-like matrix sizes.")
        lines.append("2. Use `scripts/tune_kernels.py` before accepting changes to the automatic selector.")
        lines.append("3. Focus the next shader pass on large square GEMMs, where PyTorch CUDA still has the largest lead.")
    lines.extend(
        [
            "4. Keep PyTorch CUDA and CuPy CUDA rows in regular benchmark runs so Vulkan changes are compared against cuBLAS-backed GPU compute.",
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
    nvidia_smi = nvidia_smi_summary()
    results = []
    results.extend(bench_tensor_ash(args.ml_bench, cases, args.iters, args.warmup))
    transfer = (
        None
        if args.skip_transfer
        else bench_transfer(args.ml_bench, args.iters, args.transfer_mb)
    )
    if not args.skip_cpu_frameworks:
        results.extend(bench_numpy(cases, args.iters, args.warmup))
        results.extend(bench_torch_cpu(cases, args.iters, args.warmup, args.torch_threads))
    if not args.skip_cublas_pure:
        results.extend(
            bench_cublas_pure(
                args.cublas_bench, cases, args.iters, args.warmup
            )
        )
    if not args.skip_gpu_frameworks:
        results.extend(bench_torch_cuda(cases, args.iters, args.warmup))
        results.extend(bench_cupy_cuda(cases, args.iters, args.warmup))
        results.extend(bench_jax(cases, args.iters, args.warmup))
        results.extend(bench_tensorflow(cases, args.iters, args.warmup))
    write_outputs(args.output_json, args.output_md, self_check, nvidia_smi, results, transfer, args)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
