"""JSON serialization and Markdown analysis for benchmark results."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from bench_compare_models import BenchResult, TransferResult


RATIO_LIBRARY_ORDER = [
    "cublas_pure",
    "torch_cuda",
    "cupy_cuda",
    "jax",
    "tensorflow",
    "numpy",
    "torch_cpu",
]

# RTX 3070 FP32 peak; override for other devices.
PEAK_TFLOPS = float(os.environ.get("ML_PEAK_TFLOPS", "20.32"))


class ReportArgs(Protocol):
    iters: int
    warmup: int
    case_set: str
    torch_threads: int
    skip_cpu_frameworks: bool
    skip_gpu_frameworks: bool


def _git_metadata() -> tuple[str, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "", None


def _has_throughput(result: BenchResult) -> bool:
    return (
        result.status == "ok"
        and result.tflops is not None
        and math.isfinite(result.tflops)
        and result.tflops > 0.0
    )


def best_by_case(results: list[BenchResult]) -> dict[str, BenchResult]:
    best: dict[str, BenchResult] = {}
    for result in results:
        if not _has_throughput(result):
            continue
        prior = best.get(result.case)
        if prior is None or (prior.tflops or 0.0) < result.tflops:
            best[result.case] = result
    return best


def successful_by_library(
    results: list[BenchResult], library: str
) -> dict[str, BenchResult]:
    return {
        result.case: result
        for result in results
        if result.library == library and _has_throughput(result)
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


def build_payload(
    self_check: str,
    nvidia_smi: str,
    results: list[BenchResult],
    transfer: TransferResult | None,
    args: ReportArgs,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    revision, dirty = _git_metadata()
    timing_scopes = {
        result.library: result.timing_scope
        for result in results
        if result.status == "ok" and result.timing_scope
    }
    timing_statistics = {
        result.library: "median"
        for result in results
        if result.status == "ok" and result.median_ms is not None
    }
    return {
        "metadata": {
            "schema_version": 2,
            "generated_at": generated_at or time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "iters": args.iters,
            "warmup": args.warmup,
            "case_set": args.case_set,
            "cpu_threads": args.torch_threads,
            "torch_threads": args.torch_threads,
            "self_check": self_check,
            "nvidia_smi": nvidia_smi,
            "timing_statistic": "median",
            "timing_statistic_by_backend": timing_statistics,
            "timing_scope_by_backend": timing_scopes,
            "git_revision": revision,
            "git_dirty": dirty,
        },
        "transfer": None if transfer is None else asdict(transfer),
        "results": [asdict(result) for result in results],
    }


def _markdown_cell(value: str) -> str:
    return value.replace("|", "/").replace("\r\n", "<br>").replace("\n", "<br>")


def _timing_triplet(result: BenchResult) -> str:
    minimum = result.min_ms if result.min_ms is not None else result.best_ms
    values = (minimum, result.median_ms, result.p95_ms)
    return "/".join("-" if value is None else f"{value:.3f}" for value in values)


def _wall_pair(result: BenchResult) -> str:
    median = result.wall_median_ms if result.wall_median_ms is not None else result.wall_ms
    return "/".join(
        "-" if value is None else f"{value:.3f}"
        for value in (median, result.wall_p95_ms)
    )


def _host_pair(result: BenchResult) -> str:
    median = (
        result.host_overhead_median_ms
        if result.host_overhead_median_ms is not None
        else result.host_overhead_ms
    )
    return "/".join(
        "-" if value is None else f"{value:.3f}"
        for value in (median, result.host_overhead_p95_ms)
    )


def _host_share(result: BenchResult) -> float | None:
    wall = result.wall_median_ms if result.wall_median_ms is not None else result.wall_ms
    host = (
        result.host_overhead_median_ms
        if result.host_overhead_median_ms is not None
        else result.host_overhead_ms
    )
    if wall is None or host is None or wall <= 0.0:
        return None
    return host / wall * 100.0


def _route(result: BenchResult) -> str:
    if not result.kernel:
        return ""
    tile = ""
    if None not in (result.tile_m, result.tile_n, result.tile_k):
        tile = f" {result.tile_m}x{result.tile_n}x{result.tile_k}"
    strategy = result.strategy
    if result.split_k2_splits is not None:
        strategy = f"{strategy or 'split_k2'}({result.split_k2_splits})"
    return f"{result.kernel}{tile}" + (f" / {strategy}" if strategy else "")


def _ratio_note(ratio: float) -> str:
    if ratio < 1.0:
        return f"cuBLAS {1.0 / ratio:.2f}x faster"
    if ratio > 1.0:
        return f"tensor-ash {ratio:.2f}x faster"
    return "parity"


def _tensor_cublas_ratios(
    results: list[BenchResult],
) -> list[tuple[float, str, BenchResult, BenchResult]]:
    def identity(result: BenchResult) -> tuple[str, int, int, int, int]:
        return (result.case, result.b, result.m, result.n, result.k)

    tensor = {
        identity(result): result
        for result in results
        if result.library == "tensor-ash" and _has_throughput(result)
    }
    cublas = {
        identity(result): result
        for result in results
        if result.library == "cublas_pure" and _has_throughput(result)
    }
    comparisons = [
        (
            (tensor[key].tflops or 0.0) / (cublas[key].tflops or 1.0),
            key[0],
            tensor[key],
            cublas[key],
        )
        for key in tensor.keys() & cublas.keys()
        if tensor[key].timing_scope == "gpu"
        and cublas[key].timing_scope == "gpu"
        and tensor[key].median_ms is not None
        and cublas[key].median_ms is not None
    ]
    return sorted(comparisons, key=lambda item: item[0])


def _analysis_lines(
    self_check: str,
    results: list[BenchResult],
    transfer: TransferResult | None,
) -> list[str]:
    best = best_by_case(results)
    software_vulkan = "llvmpipe" in self_check.lower() or "(cpu" in self_check.lower()
    ml_ok = [
        result
        for result in results
        if result.library == "tensor-ash" and _has_throughput(result)
    ]
    ml_details = next((result.details for result in ml_ok if result.details), "")
    lines = []
    if software_vulkan:
        lines.append(
            "- `tensor-ash` selected CPU/software Vulkan (`llvmpipe`), so these are correctness and overhead measurements, not real GPU performance numbers."
        )
    elif ml_details:
        lines.append(
            f"- `tensor-ash` used `{ml_details}`, so the Vulkan measurements reflect real GPU kernel timings on this host."
        )

    gpu_frameworks = sorted(
        {
            result.library
            for result in results
            if result.library != "tensor-ash"
            and result.status == "ok"
            and result_uses_gpu(result)
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
            if winner and winner.library != result.library:
                ratios.append(
                    (
                        result.case,
                        winner.library,
                        (winner.tflops or 0.0) / (result.tflops or 1.0),
                    )
                )
            elif winner:
                wins += 1
        if ratios:
            worst = max(ratios, key=lambda item: item[2])
            lines.append(
                f"- Largest gap: `{worst[0]}` is {worst[2]:.1f}x faster in `{worst[1]}` than `tensor-ash` in this environment."
            )
        lines.append(
            f"- `tensor-ash` is the fastest measured backend on {wins}/{len(ml_ok)} benchmark cases."
        )

        ml_by_case = {result.case: result for result in ml_ok}
        for library in RATIO_LIBRARY_ORDER:
            other_by_case = successful_by_library(results, library)
            shared = sorted(set(ml_by_case) & set(other_by_case))
            if not shared:
                continue
            speedups = [
                (ml_by_case[case].tflops or 0.0)
                / (other_by_case[case].tflops or 1.0)
                for case in shared
            ]
            geomean = math.prod(speedups) ** (1.0 / len(speedups))
            lines.append(
                f"- Throughput ratio versus `{library}` across {len(shared)} shared cases: "
                f"{min(speedups):.2f}x to {max(speedups):.2f}x, geometric mean {geomean:.2f}x."
            )

        overheads = sorted(
            (
                result.host_overhead_median_ms
                if result.host_overhead_median_ms is not None
                else result.host_overhead_ms
            )
            for result in ml_ok
            if (
                result.host_overhead_median_ms is not None
                or result.host_overhead_ms is not None
            )
        )
        if overheads:
            median = statistics.median(overheads)
            lines.append(
                f"- Median `tensor-ash` host/submission overhead was {median:.3f} ms per synchronous call; "
                "GPU timestamp TFLOPS excludes that overhead."
            )

        shares = [
            (share, result.case)
            for result in ml_ok
            if (share := _host_share(result)) is not None
        ]
        if shares:
            share, case = max(shares)
            lines.append(
                f"- Highest median host-overhead share was `{case}` at {share:.1f}% of wall time; "
                "use `wall TFLOPS` for latency-sensitive comparisons."
            )

        variable = [
            (result.p95_ms / result.median_ms - 1.0, result)
            for result in ml_ok
            if result.p95_ms is not None
            and result.median_ms is not None
            and result.median_ms > 0.0
        ]
        if variable:
            ratio, result = max(variable, key=lambda item: item[0])
            qualifier = (
                " (with fewer than 20 samples, p95 is effectively the observed maximum)"
                if (result.gpu_sample_count or result.sample_count or 0) < 20
                else ""
            )
            lines.append(
                f"- Highest `tensor-ash` GPU tail variability was `{result.case}`: "
                f"p95 was {ratio * 100.0:.1f}% above median{qualifier}."
            )

        cublas_ratios = _tensor_cublas_ratios(results)
        if cublas_ratios:
            worst = cublas_ratios[: min(3, len(cublas_ratios))]
            lines.append(
                "- Worst `tensor-ash` / pure-cuBLAS median-throughput ratios: "
                + ", ".join(
                    f"`{case}` {ratio:.2f}x ({_ratio_note(ratio)})"
                    for ratio, case, _, _ in worst
                    if ratio > 0.0
                )
                + "."
            )

    skipped = sorted({result.library for result in results if result.status == "skipped"})
    if skipped:
        lines.append(
            "- Some libraries were skipped because their Python modules or device backends were unavailable: "
            + ", ".join(f"`{library}`" for library in skipped)
            + "."
        )
    if any(result.library == "torch_cuda" and result.status == "skipped" for result in results):
        lines.append("- PyTorch CUDA/cuBLAS was not available in this Python environment.")
    if any(result.library == "cupy_cuda" and result.status == "skipped" for result in results):
        lines.append("- CuPy CUDA/cuBLAS was not available in this Python environment.")
    if any(
        result.library in {"jax", "tensorflow"} and result.status == "skipped"
        for result in results
    ):
        lines.append(
            "- JAX and TensorFlow rows are included when those modules are installed; skipped rows mean the local Python environment does not provide them."
        )
    if (
        transfer
        and transfer.status == "ok"
        and transfer.upload_gibs is not None
        and transfer.download_gibs is not None
        and transfer.bytes is not None
    ):
        lines.append(
            f"- Transfer staging bandwidth measured {transfer.upload_gibs:.2f} GiB/s upload "
            f"and {transfer.download_gibs:.2f} GiB/s download for {transfer.bytes} bytes."
        )
    else:
        lines.append(
            "- Transfer overhead is separately measurable with `ml_bench transfer`; use it to distinguish copy overhead from GEMM kernel time."
        )
    return lines


def build_markdown(
    self_check: str,
    nvidia_smi: str,
    results: list[BenchResult],
    transfer: TransferResult | None,
    args: ReportArgs,
) -> str:
    software_vulkan = "llvmpipe" in self_check.lower() or "(cpu" in self_check.lower()
    lines = [
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
            ["NVIDIA-SMI GPU summary:", "", "```text", nvidia_smi, "```", ""]
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
            "Times are `min/median/p95`; throughput uses the median of the backend's timed scope.",
            "",
            "| case | library | status | scope | samples | timed ms | wall med/p95 ms | host med/p95 ms | host % | TFLOPS | wall TFLOPS | route | details |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for result in results:
        timed_samples = (
            result.gpu_sample_count
            if result.timing_scope == "gpu" and result.gpu_sample_count is not None
            else result.sample_count
        )
        sample_count = "" if timed_samples is None else str(timed_samples)
        host_share = _host_share(result)
        host_share_text = "" if host_share is None else f"{host_share:.1f}%"
        tflops = "" if result.tflops is None else f"{result.tflops:.6f}"
        wall_tflops = "" if result.wall_tflops is None else f"{result.wall_tflops:.6f}"
        lines.append(
            f"| {result.case} | {result.library} | {result.status} | {result.timing_scope} | {sample_count} | "
            f"{_timing_triplet(result)} | {_wall_pair(result)} | {_host_pair(result)} | "
            f"{host_share_text} | {tflops} | {wall_tflops} | {_route(result)} | "
            f"{_markdown_cell(result.details)} |"
        )

    if transfer is not None:
        lines.extend(
            [
                "",
                "## Transfer",
                "",
                "| status | bytes | samples | upload GiB/s | download GiB/s | upload ms min/med/p95 | download ms min/med/p95 | details |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        transfer_bytes = "" if transfer.bytes is None else str(transfer.bytes)
        transfer_samples = transfer.sample_count or transfer.iters
        transfer_samples_text = "" if transfer_samples is None else str(transfer_samples)
        upload = "" if transfer.upload_gibs is None else f"{transfer.upload_gibs:.3f}"
        download = "" if transfer.download_gibs is None else f"{transfer.download_gibs:.3f}"
        upload_ms = "/".join(
            "-" if value is None else f"{value:.3f}"
            for value in (
                transfer.upload_min_ms,
                transfer.upload_median_ms,
                transfer.upload_p95_ms,
            )
        )
        download_ms = "/".join(
            "-" if value is None else f"{value:.3f}"
            for value in (
                transfer.download_min_ms,
                transfer.download_median_ms,
                transfer.download_p95_ms,
            )
        )
        lines.append(
            f"| {transfer.status} | {transfer_bytes} | {transfer_samples_text} | {upload} | "
            f"{download} | {upload_ms} | {download_ms} | {_markdown_cell(transfer.details)} |"
        )

    lines.extend(["", "## Analysis", "", *_analysis_lines(self_check, results, transfer)])
    lines.extend(["", "## Optimization Gameplan", ""])
    if software_vulkan:
        lines.extend(
            [
                "1. Fix runtime device visibility so `ML_DEVICE=discrete ml_bench self-check` selects the actual GPU.",
                "2. Re-run this benchmark on the discrete GPU and compare against PyTorch CUDA when available.",
                "3. Tune shader variants only after measuring real GPU behavior.",
            ]
        )
    else:
        lines.extend(
            [
                "1. Keep benchmarking on this discrete-GPU baseline and tune the shape-based shader selector with larger production-like matrix sizes.",
                "2. Use `scripts/tune_kernels.py` before accepting changes to the automatic selector.",
                "3. Focus the next shader pass on the lowest dynamically reported tensor/cuBLAS ratios and the highest-tail-variability cases.",
            ]
        )
    lines.extend(
        [
            "4. Keep PyTorch CUDA and CuPy CUDA rows in regular benchmark runs so Vulkan changes are compared against cuBLAS-backed GPU compute.",
            "5. Add CI checks for `cargo fmt`, `cargo clippy`, CPU tests, and an optional GPU correctness job.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    path_json: str,
    path_md: str,
    self_check: str,
    nvidia_smi: str,
    results: list[BenchResult],
    transfer: TransferResult | None,
    args: ReportArgs,
) -> None:
    Path(path_json).parent.mkdir(parents=True, exist_ok=True)
    Path(path_md).parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(self_check, nvidia_smi, results, transfer, args)
    Path(path_json).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    Path(path_md).write_text(build_markdown(self_check, nvidia_smi, results, transfer, args))
