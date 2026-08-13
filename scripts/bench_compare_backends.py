"""Benchmark runners for tensor-ash and optional local frameworks."""

from __future__ import annotations

import csv
import math
import os
import statistics
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


def _sample_stats(samples: list[float]) -> tuple[int, float, float, float]:
    """Return count/min/median/nearest-rank-p95 for valid timings."""
    valid = sorted(sample for sample in samples if math.isfinite(sample) and sample >= 0.0)
    if not valid:
        raise ValueError("no finite, non-negative timing samples")
    p95_index = max(0, math.ceil(len(valid) * 0.95) - 1)
    return len(valid), valid[0], statistics.median(valid), valid[p95_index]


def _timed_result(
    library: str, case: Case, samples_ms: list[float], details: str
) -> BenchResult:
    try:
        sample_count, min_ms, median_ms, p95_ms = _sample_stats(samples_ms)
    except ValueError as exc:
        return _failed_result(library, case, str(exc))
    if median_ms <= 0.0:
        return _failed_result(library, case, f"invalid median elapsed time: {median_ms!r} ms")
    label, b, m, n, k = case
    flops = flops_for(b, m, n, k)
    timing_scope = "gpu" if library in {"torch_cuda", "cupy_cuda"} else "wall"
    wall_fields = (
        {
            "wall_ms": median_ms,
            "wall_min_ms": min_ms,
            "wall_median_ms": median_ms,
            "wall_p95_ms": p95_ms,
            "wall_tflops": flops / median_ms * 1e-9,
        }
        if timing_scope == "wall"
        else {}
    )
    return BenchResult(
        library,
        label,
        b,
        m,
        n,
        k,
        "ok",
        tflops=flops / median_ms * 1e-9,
        best_ms=min_ms,
        flops=flops,
        details=details,
        sample_count=sample_count,
        min_ms=min_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        timing_scope=timing_scope,
        **wall_fields,
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
            samples_ms = [measure_ms() for _ in range(iters)]
            results.append(_timed_result(library, case, samples_ms, details))
        except Exception as exc:  # noqa: BLE001 - keep other cases benchmarkable
            results.append(_failed_result(library, case, f"{details}; benchmark failed: {exc}"))
    return results


def _csv_rows(output: str, header_prefix: str) -> list[dict[str, str]]:
    """Parse complete CSV rows while ignoring log lines around/between them."""
    lines = output.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith(header_prefix))
    fieldnames = next(csv.reader([lines[header_idx]]))
    rows = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            continue
        try:
            values = next(csv.reader([line]))
        except csv.Error:
            continue
        if len(values) != len(fieldnames) or values == fieldnames:
            continue
        row = dict(zip(fieldnames, values, strict=True))
        try:
            for key in ("b", "m", "n", "k"):
                if key in row:
                    int(row[key])
            for key in ("bytes", "iters"):
                if key in row:
                    int(row[key])
        except ValueError:
            continue
        rows.append(row)
    return rows


def _csv_row(output: str, header_prefix: str) -> dict[str, str]:
    rows = _csv_rows(output, header_prefix)
    if not rows:
        raise ValueError(f"no CSV data row found after header {header_prefix!r}")
    return rows[0]


def _unique_rows_by_label(
    rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], set[str]]:
    """Index emitted cases while making duplicate output an explicit error."""
    indexed: dict[str, dict[str, str]] = {}
    duplicates = set()
    for row in rows:
        label = row.get("label", "")
        if not label:
            continue
        if label in indexed:
            duplicates.add(label)
        else:
            indexed[label] = row
    return indexed, duplicates


def _optional_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    if raw is None or not raw.strip():
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {raw!r}")
    return value


def _optional_int(row: dict[str, str], key: str) -> int | None:
    raw = row.get(key)
    return None if raw is None or not raw.strip() else int(raw)


def _tensor_ash_result(case: Case, row: dict[str, str], iters: int) -> BenchResult:
    label, b, m, n, k = case
    emitted_shape = tuple(int(row[key]) for key in ("b", "m", "n", "k"))
    if emitted_shape != (b, m, n, k):
        raise ValueError(f"shape mismatch: expected {(b, m, n, k)}, got {emitted_shape}")
    gpu_median_ms = _optional_float(row, "gpu_median_ms")
    if gpu_median_ms is None:
        gpu_median_ms = _optional_float(row, "gpu_ms")
    wall_median_ms = _optional_float(row, "wall_median_ms")
    if wall_median_ms is None:
        wall_median_ms = _optional_float(row, "wall_ms")
    if gpu_median_ms is None or wall_median_ms is None:
        raise ValueError("missing median GPU or wall timing")

    gpu_min_ms = _optional_float(row, "gpu_min_ms")
    gpu_p95_ms = _optional_float(row, "gpu_p95_ms")
    wall_min_ms = _optional_float(row, "wall_min_ms")
    wall_p95_ms = _optional_float(row, "wall_p95_ms")
    gpu_min_ms = gpu_median_ms if gpu_min_ms is None else gpu_min_ms
    gpu_p95_ms = gpu_median_ms if gpu_p95_ms is None else gpu_p95_ms
    wall_min_ms = wall_median_ms if wall_min_ms is None else wall_min_ms
    wall_p95_ms = wall_median_ms if wall_p95_ms is None else wall_p95_ms
    host_median_ms = _optional_float(row, "host_overhead_median_ms")
    if host_median_ms is None:
        host_median_ms = max(0.0, wall_median_ms - gpu_median_ms)
    host_min_ms = _optional_float(row, "host_overhead_min_ms")
    host_p95_ms = _optional_float(row, "host_overhead_p95_ms")
    host_min_ms = host_median_ms if host_min_ms is None else host_min_ms
    host_p95_ms = host_median_ms if host_p95_ms is None else host_p95_ms
    sample_count = _optional_int(row, "sample_count")
    gpu_sample_count = _optional_int(row, "gpu_sample_count")
    sample_count = iters if sample_count is None else sample_count
    gpu_sample_count = sample_count if gpu_sample_count is None else gpu_sample_count
    if sample_count < 1 or gpu_sample_count < 1:
        raise ValueError("sample counts must be positive")
    if not 0.0 <= gpu_min_ms <= gpu_median_ms <= gpu_p95_ms:
        raise ValueError("GPU timing summary is not ordered")
    if not 0.0 <= wall_min_ms <= wall_median_ms <= wall_p95_ms:
        raise ValueError("wall timing summary is not ordered")
    if not 0.0 <= host_min_ms <= host_median_ms <= host_p95_ms:
        raise ValueError("host-overhead summary is not ordered")
    flops = flops_for(b, m, n, k)
    wall_tflops = _optional_float(row, "wall_tflops")
    if wall_tflops is None:
        wall_tflops = flops / wall_median_ms * 1e-9
    return BenchResult(
        "tensor-ash",
        label,
        b,
        m,
        n,
        k,
        "ok",
        tflops=flops / gpu_median_ms * 1e-9,
        best_ms=gpu_min_ms,
        wall_ms=wall_median_ms,
        host_overhead_ms=host_median_ms,
        flops=flops,
        details=f"{row['device']} ({row['kind']})",
        sample_count=sample_count,
        min_ms=gpu_min_ms,
        median_ms=gpu_median_ms,
        p95_ms=gpu_p95_ms,
        gpu_sample_count=gpu_sample_count,
        wall_min_ms=wall_min_ms,
        wall_median_ms=wall_median_ms,
        wall_p95_ms=wall_p95_ms,
        host_overhead_min_ms=host_min_ms,
        host_overhead_median_ms=host_median_ms,
        host_overhead_p95_ms=host_p95_ms,
        wall_tflops=wall_tflops,
        kernel=row.get("kernel", ""),
        tile_m=_optional_int(row, "tile_m"),
        tile_n=_optional_int(row, "tile_n"),
        tile_k=_optional_int(row, "tile_k"),
        strategy=row.get("strategy", ""),
        split_k2_splits=_optional_int(row, "split_k2_splits"),
        timing_scope="gpu",
    )


def bench_tensor_ash(
    path: str, cases: list[Case], iters: int, warmup: int
) -> list[BenchResult]:
    env = os.environ.copy()
    env.update(
        {
            "ML_ITERS": str(iters),
            "ML_WARMUP": str(warmup),
            "ML_OUTPUT": "csv",
        }
    )
    specs = [f"{label},{b},{m},{n},{k}" for label, b, m, n, k in cases]
    code, output = run_cmd([path, "cases", *specs], env=env)
    try:
        rows = _csv_rows(output, "device,kind,label")
    except (StopIteration, csv.Error):
        rows = []
    if not rows:
        details = output.strip() or f"multi-case benchmark failed with exit code {code}"
        return [_failed_result("tensor-ash", case, details) for case in cases]

    by_label, duplicates = _unique_rows_by_label(rows)
    results = []
    for case in cases:
        if case[0] in duplicates:
            results.append(
                _failed_result(
                    "tensor-ash", case, "multi-case benchmark emitted duplicate rows"
                )
            )
            continue
        row = by_label.get(case[0])
        if row is None:
            details = f"multi-case benchmark did not emit row {case[0]!r} (exit code {code})"
            results.append(_failed_result("tensor-ash", case, details))
            continue
        try:
            results.append(_tensor_ash_result(case, row, iters))
        except Exception as exc:  # noqa: BLE001 - contain malformed rows by shape
            results.append(_failed_result("tensor-ash", case, f"invalid CSV row: {exc}"))
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
        upload_gibs = _optional_float(row, "upload_gibs")
        download_gibs = _optional_float(row, "download_gibs")
        if upload_gibs is None or upload_gibs <= 0.0:
            raise ValueError("missing positive upload bandwidth")
        if download_gibs is None or download_gibs <= 0.0:
            raise ValueError("missing positive download bandwidth")
        return TransferResult(
            "ok",
            bytes=int(row["bytes"]),
            iters=int(row["iters"]),
            upload_gibs=upload_gibs,
            download_gibs=download_gibs,
            details=f"{row['device']} ({row['kind']})",
            sample_count=_optional_int(row, "sample_count"),
            upload_min_ms=_optional_float(row, "upload_min_ms"),
            upload_median_ms=_optional_float(row, "upload_median_ms"),
            upload_p95_ms=_optional_float(row, "upload_p95_ms"),
            download_min_ms=_optional_float(row, "download_min_ms"),
            download_median_ms=_optional_float(row, "download_median_ms"),
            download_p95_ms=_optional_float(row, "download_p95_ms"),
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
        details = f"binary failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}"
        return [_failed_result("cublas_pure", case, details) for case in cases]

    try:
        rows, duplicates = _unique_rows_by_label(
            _csv_rows(proc.stdout, "label,b,m,n,k,")
        )
    except (KeyError, StopIteration, csv.Error):
        rows, duplicates = {}, set()
    details = "pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events"
    results = []
    for case in cases:
        if case[0] in duplicates:
            results.append(
                _failed_result("cublas_pure", case, "binary emitted duplicate rows")
            )
            continue
        row = rows.get(case[0])
        if row is None:
            results.append(_failed_result("cublas_pure", case, "binary did not emit a row"))
            continue
        try:
            emitted_shape = tuple(int(row[key]) for key in ("b", "m", "n", "k"))
            if emitted_shape != case[1:]:
                raise ValueError(
                    f"shape mismatch: expected {case[1:]}, got {emitted_shape}"
                )
            min_ms = _optional_float(row, "min_ms")
            if min_ms is None:
                min_ms = _optional_float(row, "best_ms")
            if min_ms is None or min_ms <= 0.0:
                raise ValueError("missing positive minimum time")
            median_ms = _optional_float(row, "median_ms")
            p95_ms = _optional_float(row, "p95_ms")
            if median_ms is None or p95_ms is None:
                raise ValueError(
                    "legacy best-only row has no median/p95; rebuild cublas_bench"
                )
            sample_count = _optional_int(row, "sample_count") or iters
            if sample_count < 1:
                raise ValueError("sample count must be positive")
            if not 0.0 < min_ms <= median_ms <= p95_ms:
                raise ValueError("timing summary is not ordered")
            flops = flops_for(case[1], case[2], case[3], case[4])
            tflops = flops / median_ms * 1e-9
            result = BenchResult(
                "cublas_pure",
                *case,
                "ok",
                tflops=tflops,
                best_ms=min_ms,
                flops=flops,
                details=details,
                sample_count=sample_count,
                min_ms=min_ms,
                median_ms=median_ms,
                p95_ms=p95_ms,
                timing_scope="gpu",
            )
            results.append(result)
        except (KeyError, TypeError, ValueError) as exc:
            results.append(_failed_result("cublas_pure", case, f"invalid binary row: {exc}"))
    return results
