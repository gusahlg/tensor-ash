use tensor_ash::VulkanContext;

use super::cases::{BenchResult, SampleSummary};
use super::env::OutputMode;

pub(super) struct BenchReporter<'a> {
    mode: OutputMode,
    peak_tflops: f64,
    ctx: &'a VulkanContext,
}

impl<'a> BenchReporter<'a> {
    pub(super) fn new(mode: OutputMode, peak_tflops: f64, ctx: &'a VulkanContext) -> Self {
        Self {
            mode,
            peak_tflops,
            ctx,
        }
    }

    pub(super) fn print_header(&mut self) {
        match self.mode {
            OutputMode::Table => {
                println!();
                println!(
                    "{:<28} {:<28} {:>10} {:>9} {:>25} {:>25} {:>25} {:>8} {:>8} {:>8}",
                    "shape",
                    "kernel / route",
                    "GFLOPs",
                    "samples",
                    "wall ms (min/med/p95)",
                    "gpu ms (min/med/p95)",
                    "host ms (min/med/p95)",
                    "TF/s",
                    "wall TF",
                    "%peak",
                );
                println!("{}", "-".repeat(187));
            }
            OutputMode::Csv => {
                println!(
                    "device,kind,label,b,m,n,k,flops,wall_ms,gpu_ms,tflops,percent_peak,sample_count,gpu_sample_count,wall_min_ms,wall_median_ms,wall_p95_ms,gpu_min_ms,gpu_median_ms,gpu_p95_ms,host_overhead_min_ms,host_overhead_median_ms,host_overhead_p95_ms,wall_tflops,kernel,tile_m,tile_n,tile_k,strategy,split_k2_splits"
                );
            }
        }
    }

    pub(super) fn print_case(&mut self, result: &BenchResult) {
        let pct = result.tflops / self.peak_tflops * 100.0;
        let (gpu_count, gpu_min, gpu_median, gpu_p95) = summary_values(result.gpu_ms);
        let (_, host_min, host_median, host_p95) = summary_values(result.host_overhead_ms);
        let splits = result
            .dispatch
            .split_k2_splits
            .map(|value| value.to_string())
            .unwrap_or_default();
        let strategy = if splits.is_empty() {
            "data_parallel"
        } else {
            "split_k2"
        };
        match self.mode {
            OutputMode::Table => println!(
                "{:<28} {:<28} {:>10.3} {:>4}/{:<4} {:>25} {:>25} {:>25} {:>8.2} {:>8.2} {:>7.1}%",
                result.case.label,
                format!("{} / {strategy}", result.dispatch.kernel),
                result.flops / 1e9,
                result.wall_ms.count,
                gpu_count,
                summary_triplet(Some(result.wall_ms)),
                summary_triplet(result.gpu_ms),
                summary_triplet(result.host_overhead_ms),
                result.tflops,
                result.wall_tflops,
                pct,
            ),
            OutputMode::Csv => println!(
                "{},{},{},{},{},{},{},{:.0},{:.6},{:.6},{:.6},{:.3},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{}",
                csv_escape(self.ctx.device_name()),
                self.ctx.device_kind().as_str(),
                csv_escape(&result.case.label),
                result.case.b,
                result.case.m,
                result.case.n,
                result.case.k,
                result.flops,
                result.wall_ms.median,
                gpu_median,
                result.tflops,
                pct,
                result.wall_ms.count,
                gpu_count,
                result.wall_ms.min,
                result.wall_ms.median,
                result.wall_ms.p95,
                gpu_min,
                gpu_median,
                gpu_p95,
                host_min,
                host_median,
                host_p95,
                result.wall_tflops,
                result.dispatch.kernel,
                result.dispatch.tile[0],
                result.dispatch.tile[1],
                result.dispatch.tile[2],
                strategy,
                splits,
            ),
        }
    }
}

fn summary_values(summary: Option<SampleSummary>) -> (usize, f64, f64, f64) {
    summary.map_or((0, f64::NAN, f64::NAN, f64::NAN), |summary| {
        (summary.count, summary.min, summary.median, summary.p95)
    })
}

fn summary_triplet(summary: Option<SampleSummary>) -> String {
    summary.map_or_else(
        || "-".into(),
        |summary| {
            format!(
                "{:.3}/{:.3}/{:.3}",
                summary.min, summary.median, summary.p95
            )
        },
    )
}

pub(super) fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_escape_only_quotes_when_needed() {
        assert_eq!(csv_escape("RTX 3070"), "RTX 3070");
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
        assert_eq!(csv_escape("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn missing_summary_is_rendered_explicitly() {
        assert_eq!(summary_values(None).0, 0);
        assert!(summary_values(None).1.is_nan());
        assert_eq!(summary_triplet(None), "-");
    }
}
