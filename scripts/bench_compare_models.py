"""Dependency-free benchmark cases and result models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


Case: TypeAlias = tuple[str, int, int, int, int]

BASE_CASES: list[Case] = [
    ("square_128", 1, 128, 128, 128),
    ("square_256", 1, 256, 256, 256),
    ("square_512", 1, 512, 512, 512),
    ("batched_4x256", 4, 256, 256, 256),
    ("tall_512x256x256", 1, 512, 256, 256),
    ("odd_255x257x263", 1, 255, 257, 263),
]

EXTENDED_CASES: list[Case] = [
    *BASE_CASES,
    ("square_1024", 1, 1024, 1024, 1024),
    ("batched_2x512", 2, 512, 512, 512),
    ("skinny_1024x128x512", 1, 1024, 128, 512),
    ("wide_128x1024x512", 1, 128, 1024, 512),
    ("small_k_1024x1024x64", 1, 1024, 1024, 64),
]

SHOWCASE_CASES: list[Case] = [
    *EXTENDED_CASES,
    ("non_pow2_513x515x517", 1, 513, 515, 517),
    ("non_pow2_1023x1025x1027", 1, 1023, 1025, 1027),
    ("medium_384", 1, 384, 384, 384),
    ("medium_768", 1, 768, 768, 768),
    ("batched_8x256", 8, 256, 256, 256),
    ("batched_16x128", 16, 128, 128, 128),
    ("batched_32x64", 32, 64, 64, 64),
    ("batched_64x128", 64, 128, 128, 128),
    ("batched_128x64", 128, 64, 64, 64),
    ("attn_proj_2048x512x512", 1, 2048, 512, 512),
    ("attn_proj_512x2048x512", 1, 512, 2048, 512),
    ("attn_qkv_1024x3072x512", 1, 1024, 3072, 512),
    ("tiny_b32_128", 32, 128, 128, 128),
    ("tiny_b16_192", 16, 192, 192, 192),
    ("tiny_b8_192", 8, 192, 192, 192),
]

# Shapes targeting Stream-K wave-quantization tails on a 46-SM GPU.
STREAMK_CASES: list[Case] = [
    ("sq_4096", 1, 4096, 4096, 4096),
    ("sq_4096_k1024", 1, 4096, 4096, 1024),
    ("sq_2048", 1, 2048, 2048, 2048),
    ("attn_2048x4096x512", 1, 2048, 4096, 512),
    ("attn_4096x2048x512", 1, 4096, 2048, 512),
    ("attn_qkv_2048x6144x1024", 1, 2048, 6144, 1024),
    ("ffn_proj_2048x8192x2048", 1, 2048, 8192, 2048),
    ("sq_3584", 1, 3584, 3584, 3584),
    ("sq_6144", 1, 6144, 6144, 4096),
    ("deep_k_512_8192", 1, 512, 512, 8192),
]

# Compact guard set for the known throughput gaps and selector boundaries.
# Keep this intentionally small: it is meant for frequent regression runs,
# not as another exhaustive showcase.
REGRESSION_CASES: list[Case] = [
    ("medium_768", 1, 768, 768, 768),
    ("non_pow2_1023x1025x1027", 1, 1023, 1025, 1027),
    ("odd_255x257x263", 1, 255, 257, 263),
    ("wide_128x1024x512", 1, 128, 1024, 512),
    ("boundary_127x129x65", 1, 127, 129, 65),
    ("batched_8x256", 8, 256, 256, 256),
    ("deep_k_64x64x8192", 1, 64, 64, 8192),
    ("gemv_1x4096x4096", 1, 1, 4096, 4096),
    ("gemv_n1_4096x1x4096", 1, 4096, 1, 4096),
]

CASE_SETS: dict[str, list[Case]] = {
    "base": BASE_CASES,
    "extended": EXTENDED_CASES,
    "showcase": SHOWCASE_CASES,
    "streamk": STREAMK_CASES,
    "regression": REGRESSION_CASES,
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
    wall_ms: float | None = None
    host_overhead_ms: float | None = None
    flops: float | None = None
    details: str = ""
    # `best_ms` remains the compatibility alias for the minimum sample.
    # Throughput is based on `median_ms` whenever a sample distribution is
    # available.
    sample_count: int | None = None
    min_ms: float | None = None
    median_ms: float | None = None
    p95_ms: float | None = None
    gpu_sample_count: int | None = None
    wall_min_ms: float | None = None
    wall_median_ms: float | None = None
    wall_p95_ms: float | None = None
    host_overhead_min_ms: float | None = None
    host_overhead_median_ms: float | None = None
    host_overhead_p95_ms: float | None = None
    wall_tflops: float | None = None
    kernel: str = ""
    tile_m: int | None = None
    tile_n: int | None = None
    tile_k: int | None = None
    strategy: str = ""
    split_k2_splits: int | None = None
    timing_scope: str = ""


@dataclass
class TransferResult:
    status: str
    bytes: int | None = None
    iters: int | None = None
    upload_gibs: float | None = None
    download_gibs: float | None = None
    details: str = ""
    sample_count: int | None = None
    upload_min_ms: float | None = None
    upload_median_ms: float | None = None
    upload_p95_ms: float | None = None
    download_min_ms: float | None = None
    download_median_ms: float | None = None
    download_p95_ms: float | None = None


def flops_for(b: int, m: int, n: int, k: int) -> float:
    return 2.0 * b * m * n * k


def matrix_shapes(case: Case) -> tuple[tuple[int, ...], tuple[int, ...]]:
    _, b, m, n, k = case
    if b == 1:
        return (m, k), (k, n)
    return (b, m, k), (b, k, n)


def skipped_results(library: str, cases: list[Case], details: str) -> list[BenchResult]:
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
