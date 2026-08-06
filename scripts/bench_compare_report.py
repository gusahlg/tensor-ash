"""JSON serialization and Markdown analysis for benchmark results."""

from __future__ import annotations

import json
import math
import os
import platform
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
    return {
        "metadata": {
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
        },
        "transfer": None if transfer is None else asdict(transfer),
        "results": [asdict(result) for result in results],
    }


def _markdown_cell(value: str) -> str:
    return value.replace("|", "/").replace("\r\n", "<br>").replace("\n", "<br>")


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
            result.host_overhead_ms
            for result in ml_ok
            if result.host_overhead_ms is not None and result.wall_ms is not None
        )
        if overheads:
            median = overheads[len(overheads) // 2]
            lines.append(
                f"- Median `tensor-ash` host/submission overhead was {median:.3f} ms per synchronous call; "
                "GPU timestamp TFLOPS excludes that overhead."
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
            "| case | library | status | gpu ms | wall ms | host overhead ms | TFLOPS | % peak | details |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in results:
        best_ms = "" if result.best_ms is None else f"{result.best_ms:.3f}"
        wall_ms = "" if result.wall_ms is None else f"{result.wall_ms:.3f}"
        overhead = "" if result.host_overhead_ms is None else f"{result.host_overhead_ms:.3f}"
        tflops = "" if result.tflops is None else f"{result.tflops:.6f}"
        peak = "" if result.tflops is None or PEAK_TFLOPS <= 0 else f"{result.tflops / PEAK_TFLOPS * 100:.1f}%"
        lines.append(
            f"| {result.case} | {result.library} | {result.status} | {best_ms} | "
            f"{wall_ms} | {overhead} | {tflops} | {peak} | {_markdown_cell(result.details)} |"
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
            f"| {transfer.status} | {transfer_bytes} | {transfer_iters} | {upload} | "
            f"{download} | {_markdown_cell(transfer.details)} |"
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
                "3. Focus the next shader pass on large square GEMMs, where PyTorch CUDA still has the largest lead.",
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
    Path(path_json).write_text(json.dumps(payload, indent=2) + "\n")
    Path(path_md).write_text(build_markdown(self_check, nvidia_smi, results, transfer, args))
