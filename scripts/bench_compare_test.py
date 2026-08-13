#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from bench_compare import (  # noqa: E402
    BenchResult,
    REGRESSION_CASES,
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
import bench_compare_backends as backends  # noqa: E402
from bench_compare_backends import (  # noqa: E402
    _benchmark_cases,
    _csv_rows,
    _optional_float,
    _sample_stats,
    bench_cublas_pure,
    bench_tensor_ash,
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

    def test_sample_stats_matches_rust_nearest_rank_definition(self) -> None:
        self.assertEqual(_sample_stats([2.5]), (1, 2.5, 2.5, 2.5))
        count, minimum, median, p95 = _sample_stats(list(range(20, 0, -1)))
        self.assertEqual((count, minimum, median, p95), (20, 1, 10.5, 19))

    def test_regression_set_covers_measured_gaps_and_boundaries(self) -> None:
        labels = {case[0] for case in REGRESSION_CASES}
        self.assertTrue(
            {
                "medium_768",
                "non_pow2_1023x1025x1027",
                "odd_255x257x263",
                "wide_128x1024x512",
                "boundary_127x129x65",
                "deep_k_64x64x8192",
                "gemv_1x4096x4096",
                "gemv_n1_4096x1x4096",
            }.issubset(labels)
        )

    def test_shared_runner_uses_median_timing_and_contains_case_failures(self) -> None:
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
        self.assertEqual(rows[0].median_ms, 2.5)
        self.assertEqual(rows[0].p95_ms, 3.0)
        self.assertAlmostEqual(rows[0].tflops, 24e-9)
        self.assertEqual(rows[1].status, "failed")
        self.assertEqual(rows[1].flops, 60.0)
        self.assertIn("shape rejected", rows[1].details)

    def test_csv_rows_ignore_interleaved_log_lines_and_future_columns(self) -> None:
        output = "\n".join(
            [
                "[INFO] benchmark starting",
                "device,kind,label,b,m,n,k,future",
                "[INFO] clocks warmed",
                'GPU,discrete,"case,quoted",1,2,3,4,new',
                "malformed,row",
            ]
        )
        self.assertEqual(
            _csv_rows(output, "device,kind,label")[0],
            {
                "device": "GPU",
                "kind": "discrete",
                "label": "case,quoted",
                "b": "1",
                "m": "2",
                "n": "3",
                "k": "4",
                "future": "new",
            },
        )

    def test_csv_numeric_parser_rejects_non_finite_json_values(self) -> None:
        for value in ("NaN", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _optional_float({"gpu_median_ms": value}, "gpu_median_ms")

    def test_tensor_ash_uses_one_multi_case_process_and_contains_missing_rows(self) -> None:
        header = (
            "device,kind,label,b,m,n,k,flops,wall_ms,gpu_ms,tflops,percent_peak,"
            "sample_count,gpu_sample_count,wall_min_ms,wall_median_ms,wall_p95_ms,"
            "gpu_min_ms,gpu_median_ms,gpu_p95_ms,host_overhead_min_ms,"
            "host_overhead_median_ms,host_overhead_p95_ms,wall_tflops,kernel,"
            "tile_m,tile_n,tile_k,strategy,split_k2_splits"
        )
        row = (
            "GPU,discrete,good,1,2,3,5,60,0.7,0.5,0.00000012,1,3,3,"
            "0.6,0.7,0.9,0.4,0.5,0.8,0.1,0.2,0.3,0.000000086,k64,64,64,64,"
            "split_k2,8"
        )
        second_row = (
            "GPU,discrete,second,1,7,11,13,2002,1.2,1.0,0.000002,1,3,3,"
            "1.1,1.2,1.4,0.9,1.0,1.3,0.1,0.2,0.3,0.0000017,k128,"
            "128,64,32,data_parallel,"
        )
        cases = [
            ("good", 1, 2, 3, 5),
            ("second", 1, 7, 11, 13),
            ("missing", 1, 2, 3, 5),
        ]
        output = f"{header}\n[INFO] warm\n{row}\n[DEBUG] next\n{second_row}\n"
        with patch.object(backends, "run_cmd", return_value=(0, output)) as run:
            results = bench_tensor_ash("ml_bench", cases, 3, 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args.args[0],
            [
                "ml_bench",
                "cases",
                "good,1,2,3,5",
                "second,1,7,11,13",
                "missing,1,2,3,5",
            ],
        )
        self.assertEqual(results[0].median_ms, 0.5)
        self.assertEqual(results[0].best_ms, 0.4)
        self.assertEqual(results[0].host_overhead_ms, 0.2)
        self.assertEqual(results[0].kernel, "k64")
        self.assertEqual(results[0].strategy, "split_k2")
        self.assertEqual(results[0].split_k2_splits, 8)
        self.assertEqual(results[1].case, "second")
        self.assertEqual(results[1].median_ms, 1.0)
        self.assertEqual(results[1].kernel, "k128")
        self.assertEqual(results[2].status, "failed")

    def test_legacy_cublas_row_is_not_treated_as_a_median(self) -> None:
        legacy = "label,b,m,n,k,best_ms,mean_ms,tflops\ncase,1,2,3,5,1,2,0.1\n"
        case = [("case", 1, 2, 3, 5)]
        with patch.object(Path, "is_file", return_value=True), patch.object(
            backends.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout=legacy, stderr=""),
        ):
            result = bench_cublas_pure("cublas", case, 3, 1)[0]
        self.assertEqual(result.status, "failed")
        self.assertIn("legacy best-only", result.details)

    def test_cublas_uses_reported_median_and_tail_statistics(self) -> None:
        output = (
            "label,b,m,n,k,best_ms,mean_ms,tflops,sample_count,min_ms,median_ms,p95_ms\n"
            "[INFO] clocks settled\n"
            "case,1,2,3,5,0.4,0.7,99,5,0.4,0.5,0.8\n"
        )
        with patch.object(Path, "is_file", return_value=True), patch.object(
            backends.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout=output, stderr=""),
        ):
            result = bench_cublas_pure("cublas", [("case", 1, 2, 3, 5)], 5, 1)[0]
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.sample_count, 5)
        self.assertEqual((result.best_ms, result.median_ms, result.p95_ms), (0.4, 0.5, 0.8))
        self.assertAlmostEqual(result.tflops, 60.0 / 0.5 * 1e-9)


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
                sample_count=5,
                min_ms=0.5,
                median_ms=0.6,
                p95_ms=0.8,
                timing_scope="gpu",
            ),
            BenchResult(
                "cupy_cuda",
                "case",
                1,
                2,
                3,
                5,
                "ok",
                tflops=1.0,
                sample_count=5,
                min_ms=1.0,
                median_ms=1.2,
                p95_ms=1.4,
                timing_scope="gpu",
            ),
        ]

    def test_payload_serializes_models_without_framework_dependencies(self) -> None:
        transfer = TransferResult("ok", 1024, 2, 1.25, 1.5, "PCIe")
        with patch(
            "bench_compare_report._git_metadata", return_value=("abc123", True)
        ):
            payload = build_payload(
                "GPU",
                "gpu0: name=GPU",
                self.rows,
                transfer,
                self.args,
                generated_at="fixed",
            )
        self.assertEqual(payload["metadata"]["generated_at"], "fixed")
        self.assertEqual(payload["metadata"]["schema_version"], 2)
        self.assertEqual(payload["metadata"]["timing_statistic"], "median")
        self.assertEqual(
            payload["metadata"]["timing_statistic_by_backend"],
            {"tensor-ash": "median", "cupy_cuda": "median"},
        )
        self.assertEqual(
            payload["metadata"]["timing_scope_by_backend"],
            {"tensor-ash": "gpu", "cupy_cuda": "gpu"},
        )
        self.assertEqual(payload["metadata"]["git_revision"], "abc123")
        self.assertTrue(payload["metadata"]["git_dirty"])
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

    def test_analysis_highlights_variability_overhead_and_worst_cublas_ratio(self) -> None:
        rows = [
            BenchResult(
                "tensor-ash",
                "slow",
                1,
                2,
                3,
                5,
                "ok",
                tflops=0.5,
                best_ms=0.8,
                wall_ms=2.0,
                host_overhead_ms=1.0,
                sample_count=5,
                min_ms=0.8,
                median_ms=1.0,
                p95_ms=1.5,
                wall_median_ms=2.0,
                wall_p95_ms=2.5,
                host_overhead_median_ms=1.0,
                host_overhead_p95_ms=1.2,
                kernel="k64",
                strategy="data_parallel",
                timing_scope="gpu",
            ),
            BenchResult(
                "cublas_pure",
                "slow",
                1,
                2,
                3,
                5,
                "ok",
                tflops=1.0,
                best_ms=0.4,
                sample_count=5,
                min_ms=0.4,
                median_ms=0.5,
                p95_ms=0.6,
                timing_scope="gpu",
            ),
        ]
        markdown = build_markdown("GPU (discrete)", "", rows, None, self.args)
        self.assertIn("p95 was 50.0% above median", markdown)
        self.assertIn("Highest median host-overhead share was `slow` at 50.0%", markdown)
        self.assertIn("`slow` 0.50x (cuBLAS 2.00x faster)", markdown)
        self.assertIn("0.800/1.000/1.500", markdown)
        self.assertIn("k64 / data_parallel", markdown)


if __name__ == "__main__":
    unittest.main()
