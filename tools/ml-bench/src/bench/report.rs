use tensor_ash::VulkanContext;

use super::cases::BenchResult;
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
                    "{:<28} {:>14} {:>10} {:>10} {:>9} {:>9}",
                    "shape", "FLOPs", "wall(ms)", "gpu(ms)", "TF/s", "%peak",
                );
                println!("{}", "-".repeat(80));
            }
            OutputMode::Csv => {
                println!("device,kind,label,b,m,n,k,flops,wall_ms,gpu_ms,tflops,percent_peak");
            }
        }
    }

    pub(super) fn print_case(&mut self, result: &BenchResult) {
        let pct = result.tflops / self.peak_tflops * 100.0;
        match self.mode {
            OutputMode::Table => println!(
                "{:<28} {:>14.3} {:>10.3} {:>10.3} {:>9.2} {:>8.1}%",
                result.case.label,
                result.flops / 1e9,
                result.wall_ms,
                result.gpu_ms,
                result.tflops,
                pct,
            ),
            OutputMode::Csv => println!(
                "{},{},{},{},{},{},{},{:.0},{:.6},{:.6},{:.6},{:.3}",
                csv_escape(self.ctx.device_name()),
                self.ctx.device_kind().as_str(),
                csv_escape(&result.case.label),
                result.case.b,
                result.case.m,
                result.case.n,
                result.case.k,
                result.flops,
                result.wall_ms,
                result.gpu_ms,
                result.tflops,
                pct,
            ),
        }
    }
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
}
