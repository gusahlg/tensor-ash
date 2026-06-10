#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bench_compare import (  # noqa: E402
    BenchResult,
    flops_for,
    format_nvidia_smi_summary,
    result_uses_gpu,
    skipped_results,
)


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


if __name__ == "__main__":
    unittest.main()
