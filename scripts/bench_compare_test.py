#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from bench_compare import (  # noqa: E402
    BenchResult,
    TransferResult,
    best_by_case,
    build_markdown,
    build_payload,
    flops_for,
    format_nvidia_smi_summary,
    matrix_shapes,
    result_uses_gpu,
    skipped_results,
)
from bench_compare_backends import _benchmark_cases  # noqa: E402


class BenchCompareHelpersTest(unittest.TestCase):
    def test_flops_for_counts_fma_as_two_ops(self) -> None:
        self.assertEqual(flops_for(2, 3, 5, 7), 420.0)

    def test_skipped_results_preserve_case_shapes(self) -> None:
        rows = skipped_results("torch_cuda", [("case", 2, 3, 5, 7)], "missing")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].library, "torch_cuda")
        self.assertEqual(rows[0].status, "skipped")
        self.assertEqual(rows[0].flops, 420.0)
        self.assertEqual(rows[0].details, "missing")

    def test_result_uses_gpu_identifies_cuda_and_framework_details(self) -> None:
        self.assertTrue(result_uses_gpu(BenchResult("torch_cuda", "case", 1, 1, 1, 1, "ok")))
        self.assertTrue(result_uses_gpu(BenchResult("cupy_cuda", "case", 1, 1, 1, 1, "ok")))
        self.assertTrue(result_uses_gpu(BenchResult("jax", "case", 1, 1, 1, 1, "ok", details="backend=gpu")))
        self.assertFalse(result_uses_gpu(BenchResult("jax", "case", 1, 1, 1, 1, "ok", details="backend=cpu")))
        self.assertFalse(result_uses_gpu(BenchResult("torch_cuda", "case", 1, 1, 1, 1, "skipped")))

    def test_format_nvidia_smi_summary_labels_query_fields(self) -> None:
        summary = format_nvidia_smi_summary(
            "NVIDIA GeForce RTX 3070, 595.71.05, 8192, 33, 0, 17.86, 220.00\n"
        )
        self.assertIn("gpu0: name=NVIDIA GeForce RTX 3070", summary)
        self.assertIn("driver=595.71.05", summary)
        self.assertIn("power_limit_w=220.00", summary)

    def test_matrix_shapes_drop_only_a_singleton_batch(self) -> None:
        self.assertEqual(matrix_shapes(("single", 1, 2, 3, 5)), ((2, 5), (5, 3)))
        self.assertEqual(matrix_shapes(("batch", 4, 2, 3, 5)), ((4, 2, 5), (4, 5, 3)))

    def test_best_by_case_ignores_unsuccessful_results(self) -> None:
        rows = [
            BenchResult("slow", "case", 1, 1, 1, 1, "ok", tflops=1.0),
            BenchResult("failed", "case", 1, 1, 1, 1, "failed", tflops=99.0),
            BenchResult("fast", "case", 1, 1, 1, 1, "ok", tflops=2.0),
        ]
        self.assertEqual(best_by_case(rows)["case"].library, "fast")

    def test_shared_runner_uses_best_timing_and_contains_case_failures(self) -> None:
        calls: list[str] = []

        def prepare(case):
            if case[0] == "bad":
                raise RuntimeError("shape rejected")
            timings = iter([3.0, 2.0])
            return lambda: calls.append("warmup"), lambda: next(timings)

        rows = _benchmark_cases(
            "fake",
            [("good", 1, 2, 3, 5), ("bad", 1, 2, 3, 5)],
            2,
            1,
            prepare,
            "fake backend",
        )
        self.assertEqual(calls, ["warmup"])
        self.assertEqual(rows[0].status, "ok")
        self.assertEqual(rows[0].best_ms, 2.0)
        self.assertAlmostEqual(rows[0].tflops, 30e-9)
        self.assertEqual(rows[1].status, "failed")
        self.assertEqual(rows[1].flops, 60.0)
        self.assertIn("shape rejected", rows[1].details)


class BenchCompareReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            iters=5,
            warmup=2,
            case_set="base",
            torch_threads=1,
            skip_cpu_frameworks=False,
            skip_gpu_frameworks=False,
        )
        self.rows = [
            BenchResult(
                "tensor-ash",
                "case",
                1,
                2,
                3,
                5,
                "ok",
                tflops=2.0,
                best_ms=0.5,
                wall_ms=0.7,
                host_overhead_ms=0.2,
                flops=60.0,
                details="GPU | Vulkan\nready",
            ),
            BenchResult("cupy_cuda", "case", 1, 2, 3, 5, "ok", tflops=1.0),
        ]

    def test_payload_serializes_models_without_framework_dependencies(self) -> None:
        transfer = TransferResult("ok", 1024, 2, 1.25, 1.5, "PCIe")
        payload = build_payload(
            "GPU",
            "gpu0: name=GPU",
            self.rows,
            transfer,
            self.args,
            generated_at="fixed",
        )
        self.assertEqual(payload["metadata"]["generated_at"], "fixed")
        self.assertEqual(payload["metadata"]["cpu_threads"], 1)
        self.assertEqual(payload["transfer"]["bytes"], 1024)
        self.assertEqual(payload["results"][0]["library"], "tensor-ash")

    def test_markdown_contains_analysis_and_escapes_table_details(self) -> None:
        markdown = build_markdown("GPU (discrete)", "", self.rows, None, self.args)
        self.assertIn("GPU / Vulkan<br>ready", markdown)
        self.assertIn("Actual GPU framework comparisons succeeded for: `cupy_cuda`.", markdown)
        self.assertIn("geometric mean 2.00x", markdown)
        self.assertIn("Median `tensor-ash` host/submission overhead was 0.200 ms", markdown)
        self.assertTrue(markdown.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
