"""Benchmark runners for tensor-ash and optional local frameworks."""

from __future__ import annotations

import csv
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, TypeAlias

from bench_compare_models import (
    BenchResult,
    Case,
    TransferResult,
    flops_for,
    matrix_shapes,
    skipped_results,
)


NVIDIA_SMI_FIELDS = [
    "name",
    "driver",
    "memory_total_mib",
    "temperature_c",
    "utilization_pct",
    "power_draw_w",
    "power_limit_w",
]

PreparedCase: TypeAlias = tuple[Callable[[], None], Callable[[], float]]


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
    for idx, line in enumerate(filter(str.strip, output.splitlines())):
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(NVIDIA_SMI_FIELDS):
            fields = zip(NVIDIA_SMI_FIELDS, values, strict=True)
            rows.append(f"gpu{idx}: " + ", ".join(f"{key}={value}" for key, value in fields))
        else:
            rows.append(f"gpu{idx}: {line.strip()}")
    return "\n".join(rows)


def nvidia_smi_summary() -> str:
    try:
        code, output = run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,temperature.gpu,utilization.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ]
        )
    except FileNotFoundError:
        return ""
    return format_nvidia_smi_summary(output) if code == 0 else ""


def ensure_ml_bench(path: str, skip_build: bool) -> None:
    if skip_build and Path(path).exists():
        return
    code, output = run_cmd(["cargo", "build", "--release", "-p", "ml-bench"])
    if code != 0:
        raise RuntimeError(output)


def configure_cpu_threads(threads: int) -> None:
    value = str(max(1, threads))
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value


def tensor_ash_self_check(path: str) -> str:
    code, output = run_cmd([path, "self-check"])
    return output.strip() if code == 0 else f"FAILED:\n{output.strip()}"


def _failed_result(library: str, case: Case, details: str) -> BenchResult:
    label, b, m, n, k = case
    return BenchResult(
        library,
        label,
        b,
        m,
        n,
        k,
        "failed",
        flops=flops_for(b, m, n, k),
        details=details,
    )


def _timed_result(library: str, case: Case, best_ms: float, details: str) -> BenchResult:
    if not math.isfinite(best_ms) or best_ms <= 0.0:
        return _failed_result(library, case, f"invalid elapsed time: {best_ms!r} ms")
    label, b, m, n, k = case
    flops = flops_for(b, m, n, k)
    return BenchResult(
        library,
        label,
        b,
        m,
        n,
        k,
        "ok",
        tflops=flops / best_ms * 1e-9,
        best_ms=best_ms,
        flops=flops,
        details=details,
    )


def _wall_clock_case(
    operation: Callable[[], Any], finish: Callable[[Any], Any]
) -> PreparedCase:
    def run_once() -> None:
        finish(operation())

    def measure_ms() -> float:
        started = time.perf_counter()
        run_once()
        return (time.perf_counter() - started) * 1000.0

    return run_once, measure_ms


def _benchmark_cases(
    library: str,
    cases: list[Case],
    iters: int,
    warmup: int,
    prepare: Callable[[Case], PreparedCase],
    details: str,
) -> list[BenchResult]:
    """Run a framework adapter while containing failures to one shape."""
    results = []
    for case in cases:
        try:
            run_once, measure_ms = prepare(case)
            for _ in range(warmup):
                run_once()
            best_ms = min(measure_ms() for _ in range(iters))
            results.append(_timed_result(library, case, best_ms, details))
        except Exception as exc:  # noqa: BLE001 - keep other cases benchmarkable
            results.append(_failed_result(library, case, f"{details}; benchmark failed: {exc}"))
    return results


def _csv_row(output: str, header_prefix: str) -> dict[str, str]:
    lines = output.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith(header_prefix))
    return next(csv.DictReader(lines[header_idx:]))


def bench_tensor_ash(path: str, cases: list[Case], iters: int, warmup: int) -> list[BenchResult]:
    results = []
    for case in cases:
        label, b, m, n, k = case
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
        code, output = run_cmd([path, "single"], env=env)
        if code != 0:
            results.append(_failed_result("tensor-ash", case, output.strip()))
            continue
        try:
            row = _csv_row(output, "device,kind,label")
            gpu_ms = float(row["gpu_ms"])
            wall_ms = float(row["wall_ms"])
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
                    best_ms=gpu_ms,
                    wall_ms=wall_ms,
                    host_overhead_ms=max(0.0, wall_ms - gpu_ms),
                    flops=flops_for(b, m, n, k),
                    details=f"{row['device']} ({row['kind']})",
                )
            )
        except Exception as exc:  # noqa: BLE001 - include malformed tool output
            results.append(_failed_result("tensor-ash", case, f"{exc}\n{output}"))
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
    code, output = run_cmd([path, "transfer"], env=env)
    if code != 0:
        return TransferResult("failed", details=output.strip())
    try:
        row = _csv_row(output, "device,kind,bytes")
        return TransferResult(
            "ok",
            bytes=int(row["bytes"]),
            iters=int(row["iters"]),
            upload_gibs=float(row["upload_gibs"]),
            download_gibs=float(row["download_gibs"]),
            details=f"{row['device']} ({row['kind']})",
        )
    except Exception as exc:  # noqa: BLE001 - include malformed tool output
        return TransferResult("failed", details=f"{exc}\n{output}")


def bench_numpy(cases: list[Case], iters: int, warmup: int) -> list[BenchResult]:
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return skipped_results("numpy", cases, str(exc))

    rng = np.random.default_rng(1234)

    def prepare(case: Case) -> PreparedCase:
        shape_a, shape_b = matrix_shapes(case)
        a = rng.standard_normal(shape_a, dtype=np.float32)
        b = rng.standard_normal(shape_b, dtype=np.float32)
        return _wall_clock_case(
            lambda: np.matmul(a, b), lambda result: float(result.reshape(-1)[0])
        )

    details = f"numpy {np.__version__}, threads={os.environ.get('OPENBLAS_NUM_THREADS', 'env')}"
    return _benchmark_cases("numpy", cases, iters, warmup, prepare, details)


def bench_torch_cpu(
    cases: list[Case], iters: int, warmup: int, threads: int
) -> list[BenchResult]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return skipped_results("torch_cpu", cases, str(exc))

    torch.set_num_threads(max(1, threads))
    generator = torch.Generator(device="cpu").manual_seed(1234)

    def prepare(case: Case) -> PreparedCase:
        shape_a, shape_b = matrix_shapes(case)
        a = torch.randn(shape_a, dtype=torch.float32, generator=generator)
        b = torch.randn(shape_b, dtype=torch.float32, generator=generator)
        return _wall_clock_case(
            lambda: torch.matmul(a, b), lambda result: float(result.reshape(-1)[0])
        )

    details = f"torch {torch.__version__}, threads={torch.get_num_threads()}"
    return _benchmark_cases("torch_cpu", cases, iters, warmup, prepare, details)


def bench_torch_cuda(cases: list[Case], iters: int, warmup: int) -> list[BenchResult]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return skipped_results("torch_cuda", cases, str(exc))
    if not torch.cuda.is_available():
        return skipped_results("torch_cuda", cases, "CUDA unavailable in this Python environment")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(1234)

    def prepare(case: Case) -> PreparedCase:
        shape_a, shape_b = matrix_shapes(case)
        a = torch.randn(shape_a, dtype=torch.float32, device="cuda")
        b = torch.randn(shape_b, dtype=torch.float32, device="cuda")

        def run_once() -> None:
            result = torch.matmul(a, b)
            float(result.reshape(-1)[0].cpu())

        def measure_ms() -> float:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = torch.matmul(a, b)
            end.record()
            torch.cuda.synchronize()
            float(result.reshape(-1)[0].cpu())
            return start.elapsed_time(end)

        return run_once, measure_ms

    details = (
        f"torch {torch.__version__}, CUDA/cuBLAS, {torch.cuda.get_device_name(0)}, "
        f"allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
        f"precision={torch.get_float32_matmul_precision()}"
    )
    return _benchmark_cases("torch_cuda", cases, iters, warmup, prepare, details)


def bench_cupy_cuda(cases: list[Case], iters: int, warmup: int) -> list[BenchResult]:
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
        device_name = cp.cuda.runtime.getDeviceProperties(0).get("name", "unknown")
        if isinstance(device_name, bytes):
            device_name = device_name.decode(errors="replace")
    except Exception:  # noqa: BLE001
        device_name = "unknown CUDA device"
    cp.random.seed(1234)

    def prepare(case: Case) -> PreparedCase:
        shape_a, shape_b = matrix_shapes(case)
        a = cp.random.standard_normal(shape_a, dtype=cp.float32)
        b = cp.random.standard_normal(shape_b, dtype=cp.float32)

        def run_once() -> None:
            result = cp.matmul(a, b)
            cp.cuda.Stream.null.synchronize()
            float(result.ravel()[0].get())

        def measure_ms() -> float:
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            result = cp.matmul(a, b)
            end.record()
            end.synchronize()
            float(result.ravel()[0].get())
            return cp.cuda.get_elapsed_time(start, end)

        return run_once, measure_ms

    details = f"cupy {cp.__version__}, CUDA/cuBLAS, {device_name}"
    return _benchmark_cases("cupy_cuda", cases, iters, warmup, prepare, details)


def bench_jax(cases: list[Case], iters: int, warmup: int) -> list[BenchResult]:
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # noqa: BLE001
        return skipped_results("jax", cases, str(exc))

    key = jax.random.PRNGKey(1234)

    def prepare(case: Case) -> PreparedCase:
        nonlocal key
        shape_a, shape_b = matrix_shapes(case)
        key, key_a, key_b = jax.random.split(key, 3)
        a = jax.random.normal(key_a, shape_a, dtype=jnp.float32)
        b = jax.random.normal(key_b, shape_b, dtype=jnp.float32)
        return _wall_clock_case(lambda: jnp.matmul(a, b), lambda result: result.block_until_ready())

    details = f"jax {jax.__version__}, backend={jax.default_backend()}"
    return _benchmark_cases("jax", cases, iters, warmup, prepare, details)


def bench_tensorflow(cases: list[Case], iters: int, warmup: int) -> list[BenchResult]:
    try:
        import tensorflow as tf
    except Exception as exc:  # noqa: BLE001
        return skipped_results("tensorflow", cases, str(exc))

    rng = tf.random.Generator.from_seed(1234)
    devices = ",".join(device.device_type for device in tf.config.list_logical_devices()) or "unknown"

    def prepare(case: Case) -> PreparedCase:
        shape_a, shape_b = matrix_shapes(case)
        a = rng.normal(shape_a, dtype=tf.float32)
        b = rng.normal(shape_b, dtype=tf.float32)
        return _wall_clock_case(
            lambda: tf.linalg.matmul(a, b),
            lambda result: float(tf.reshape(result, [-1])[0].numpy()),
        )

    details = f"tensorflow {tf.__version__}, devices={devices}"
    return _benchmark_cases("tensorflow", cases, iters, warmup, prepare, details)


def bench_cublas_pure(
    binary_path: str, cases: list[Case], iters: int, warmup: int
) -> list[BenchResult]:
    """Run the standalone FP32 cuBLAS benchmark for all cases in one process."""
    if not Path(binary_path).is_file():
        return skipped_results(
            "cublas_pure",
            cases,
            f"binary {binary_path} not found (build with `cd benchmarks/cublas_bench && make`)",
        )

    stdin_payload = "label,b,m,n,k\n" + "\n".join(
        f"{label},{b},{m},{n},{k}" for label, b, m, n, k in cases
    ) + "\n"
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

    lines = proc.stdout.splitlines()
    header_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if line.startswith("label,b,m,n,k,best_ms,mean_ms,tflops")
        ),
        None,
    )
    rows = {} if header_idx is None else {row["label"]: row for row in csv.DictReader(lines[header_idx:])}
    details = "pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events"
    results = []
    for case in cases:
        row = rows.get(case[0])
        if row is None:
            results.append(_failed_result("cublas_pure", case, "binary did not emit a row"))
            continue
        try:
            result = _timed_result("cublas_pure", case, float(row["best_ms"]), details)
            if result.status == "ok":
                result.tflops = float(row["tflops"])
            results.append(result)
        except (KeyError, ValueError) as exc:
            results.append(_failed_result("cublas_pure", case, f"invalid binary row: {exc}"))
    return results
