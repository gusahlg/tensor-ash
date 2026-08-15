//! `ml_bench thesis` — the performance-thesis harness.
//!
//! The optimization campaign's implicit performance model, made
//! EXPLICIT and FALSIFIABLE: each thesis (T1-T7, see
//! `benchmarks/THESES.md`) is a prediction about where the time goes,
//! paired with an automated measurement.  One command re-validates the
//! whole model after every optimization leg and flags which
//! assumptions broke:
//!
//! ```console
//! ML_MODEL=<gguf> target/release/ml_bench thesis --all
//! target/release/ml_bench thesis t2 t3 t4 t6     # no model needed
//! ```
//!
//! Predictions are recorded per GPU in
//! `benchmarks/thesis-expectations.toml` (override the path with
//! `ML_THESIS_EXPECT`); a device without a recorded section still
//! measures everything but reports INFO instead of PASS/FAIL.  Any
//! FAIL verdict makes the process exit nonzero.

mod expect;
mod gemv;
mod micro;
mod model_theses;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use tensor_ash::{Executor, MatmulCall, Tensor, VulkanContext};
use tensor_ash_test_support::fill_det;

use expect::Section;

/// One printed measurement: a thesis id, what was measured, the
/// recorded prediction, the measured value, and the verdict.
pub(crate) struct Row {
    pub(crate) thesis: &'static str,
    pub(crate) item: String,
    pub(crate) prediction: String,
    pub(crate) measured: String,
    pub(crate) verdict: Verdict,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// Measured value satisfies the recorded expectation.
    Pass,
    /// Measured value regresses the recorded expectation.
    Fail,
    /// Measured, but no expectation is recorded for this device (or
    /// the row is purely informational).
    Info,
    /// Not measurable in this configuration (reason in `measured`).
    Skip,
}

impl Verdict {
    fn as_str(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Info => "info",
            Verdict::Skip => "SKIP",
        }
    }
}

/// `value >= min`, or INFO when no expectation is recorded.
pub(crate) fn check_min(value: f64, min: Option<f64>) -> Verdict {
    match min {
        Some(min) if value >= min => Verdict::Pass,
        Some(_) => Verdict::Fail,
        None => Verdict::Info,
    }
}

/// `|value - target| <= rel_tol * target`, or INFO without a target.
pub(crate) fn check_within(value: f64, target: Option<f64>, rel_tol: f64) -> Verdict {
    match target {
        Some(target) if (value - target).abs() <= rel_tol * target.abs() => Verdict::Pass,
        Some(_) => Verdict::Fail,
        None => Verdict::Info,
    }
}

/// Recorded-key lookup that tolerates a missing device section.
pub(crate) fn key(section: Option<&Section>, name: &str) -> Option<f64> {
    section.and_then(|s| s.get(name)).copied()
}

const USAGE: &str = "usage: ml_bench thesis [--all] [t1..t7 ...] [--list] \
                     [-m/--model <gguf>]\n  \
                     model path fallback: $ML_MODEL, then \
                     models/tinyllama-1.1b-chat-v1.0.f16.gguf\n  \
                     expectations: benchmarks/thesis-expectations.toml \
                     (override with $ML_THESIS_EXPECT)";

const DESCRIPTIONS: &[(&str, &str)] = &[
    (
        "t1",
        "decode-bandwidth-bound: weight bytes/token over decode GPU ms >= 80% of device BW",
    ),
    (
        "t2",
        "GEMV efficiency: each decode GEMV class runs >= 80% of its bandwidth floor",
    ),
    (
        "t3",
        "barrier cost: a full compute barrier drains ~7.7 us (trivial-kernel chain slopes)",
    ),
    (
        "t4",
        "submission cost: one vkQueueSubmit + fence round-trip costs ~30-60 us",
    ),
    (
        "t5",
        "prefill accounting: piecewise op-class sum within 10% of the whole pp512 graph",
    ),
    (
        "t6",
        "coopmat ceiling: f16w coopmat GEMM sustains >= 30 TF/s on real prefill shapes",
    ),
    (
        "t7",
        "token exactness: all decode modes / KV dtypes / prefill widths emit identical tokens",
    ),
];

struct Options {
    theses: Vec<u32>,
    model: Option<PathBuf>,
    list: bool,
}

impl Options {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self> {
        let mut theses = Vec::new();
        let mut model = None;
        let mut list = false;
        let mut all = false;
        let mut args = args;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--all" => all = true,
                "--list" => list = true,
                "-m" | "--model" => {
                    let value = args
                        .next()
                        .with_context(|| format!("{arg} needs a value"))?;
                    model = Some(PathBuf::from(value));
                }
                name => {
                    let id: Option<u32> = name
                        .strip_prefix('t')
                        .or_else(|| name.strip_prefix('T'))
                        .and_then(|n| n.parse().ok())
                        .filter(|n| (1..=7).contains(n));
                    match id {
                        Some(id) => theses.push(id),
                        None => bail!("unknown thesis '{name}'\n{USAGE}"),
                    }
                }
            }
        }
        if all || (theses.is_empty() && !list) {
            theses = (1..=7).collect();
        }
        theses.sort_unstable();
        theses.dedup();
        Ok(Self {
            theses,
            model,
            list,
        })
    }
}

fn resolve_model(explicit: Option<PathBuf>) -> Option<PathBuf> {
    explicit
        .or_else(|| std::env::var("ML_MODEL").ok().map(PathBuf::from))
        .or_else(|| {
            let fallback = PathBuf::from("models/tinyllama-1.1b-chat-v1.0.f16.gguf");
            fallback.exists().then_some(fallback)
        })
}

/// ~300 ms of large-GEMM work before anything is timed, so the first
/// thesis does not measure cold clocks (see the campaign's bench
/// clock-state findings).
fn burn_in(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let a = Tensor::uninit_device(ctx, &[1024, 1024])?;
    let b = Tensor::uninit_device(ctx, &[1024, 1024])?;
    let c = Tensor::uninit_device(ctx, &[1024, 1024])?;
    let mut host = vec![0.0f32; 1024 * 1024];
    fill_det(&mut host, 3);
    exec.upload(&host, &a)?;
    exec.upload(&host, &b)?;
    let deadline = Instant::now() + Duration::from_millis(300);
    while Instant::now() < deadline {
        exec.run_matmuls(&[MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])?;
    }
    Ok(())
}

pub(super) fn run(
    ctx: &Arc<VulkanContext>,
    exec: &Arc<Executor>,
    args: impl Iterator<Item = String>,
) -> Result<()> {
    let opts = Options::parse(args)?;
    if opts.list {
        println!("performance theses (benchmarks/THESES.md):");
        for (name, description) in DESCRIPTIONS {
            println!("  {name}  {description}");
        }
        return Ok(());
    }

    let expect_path = expect::default_path();
    let expectations = expect::load(&expect_path)?;
    let device = ctx.device_name().to_string();
    let section: Option<&Section> = expectations
        .as_ref()
        .and_then(|expectations| expectations.device(&device));
    match (&expectations, section) {
        (None, _) => log::warn!(
            "expectations file {} not found; all verdicts will be informational",
            expect_path.display()
        ),
        (Some(_), None) => log::warn!(
            "no recorded expectations for device '{device}' in {}; all verdicts will be \
             informational — record a section to arm the regression gate",
            expect_path.display()
        ),
        (Some(_), Some(_)) => log::info!("expectations: {} [{device}]", expect_path.display()),
    }

    let wants = |id: u32| opts.theses.contains(&id);
    let model_path = resolve_model(opts.model);

    log::info!("burn-in: ~300 ms of 1024^3 GEMMs before first measurement");
    burn_in(ctx, exec)?;

    let mut rows: Vec<Row> = Vec::new();
    // T3 runs first so its measured barrier price can feed T5's
    // barrier term (falling back to the recorded expectation).
    let mut barrier_us: Option<f64> = None;
    if wants(3) {
        let (t3_rows, measured) = micro::t3_barrier_cost(ctx, exec, section)?;
        rows.extend(t3_rows);
        barrier_us = measured;
    }
    if wants(4) {
        rows.extend(micro::t4_submission_cost(ctx, exec, section)?);
    }
    if wants(2) {
        rows.extend(gemv::t2_gemv_efficiency(ctx, exec, section)?);
    }
    if wants(6) {
        rows.extend(gemv::t6_coopmat_ceiling(ctx, exec, section)?);
    }

    let needs_model = wants(1) || wants(5) || wants(7);
    if needs_model {
        match &model_path {
            None => {
                for (id, thesis) in [(1u32, "T1"), (5, "T5"), (7, "T7")] {
                    if wants(id) {
                        rows.push(Row {
                            thesis,
                            item: "model-level thesis".into(),
                            prediction: "-".into(),
                            measured: "no GGUF model (pass -m/--model or set ML_MODEL)".into(),
                            verdict: Verdict::Skip,
                        });
                    }
                }
            }
            Some(path) => {
                if wants(1) || wants(5) {
                    let mut model = model_theses::load_default(ctx, exec, path)?;
                    if wants(1) {
                        rows.extend(model_theses::t1_decode_bandwidth(section, &mut model)?);
                    }
                    if wants(5) {
                        rows.extend(model_theses::t5_prefill_accounting(
                            ctx, exec, section, &mut model, barrier_us,
                        )?);
                    }
                }
                if wants(7) {
                    rows.extend(model_theses::t7_token_exactness(ctx, exec, path)?);
                }
            }
        }
    }

    rows.sort_by(|a, b| a.thesis.cmp(b.thesis));
    print_table(&device, &rows);

    let failures = rows
        .iter()
        .filter(|row| row.verdict == Verdict::Fail)
        .count();
    if failures > 0 {
        bail!("{failures} thesis measurement(s) FAILED against recorded expectations");
    }
    Ok(())
}

fn print_table(device: &str, rows: &[Row]) {
    let headers = ["thesis", "item", "prediction", "measured", "verdict"];
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in rows {
        let cells = [
            row.thesis,
            row.item.as_str(),
            row.prediction.as_str(),
            row.measured.as_str(),
            row.verdict.as_str(),
        ];
        for (width, cell) in widths.iter_mut().zip(cells) {
            *width = (*width).max(cell.len());
        }
    }
    println!();
    println!("performance-thesis results — {device}");
    let line = |cells: [&str; 5]| {
        let mut out = String::from("|");
        for (width, cell) in widths.iter().zip(cells) {
            out.push_str(&format!(" {cell:<width$} |"));
        }
        println!("{out}");
    };
    line(headers);
    let separators: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    line([
        &separators[0],
        &separators[1],
        &separators[2],
        &separators[3],
        &separators[4],
    ]);
    for row in rows {
        line([
            row.thesis,
            row.item.as_str(),
            row.prediction.as_str(),
            row.measured.as_str(),
            row.verdict.as_str(),
        ]);
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_parse_selects_and_validates() {
        let opts = Options::parse(["t3".into(), "T1".into(), "t3".into()].into_iter()).unwrap();
        assert_eq!(opts.theses, vec![1, 3]);
        let all = Options::parse(std::iter::empty()).unwrap();
        assert_eq!(all.theses, (1..=7).collect::<Vec<_>>());
        assert!(Options::parse(["t9".into()].into_iter()).is_err());
        assert!(Options::parse(["--model".into()].into_iter()).is_err());
        let with_model =
            Options::parse(["--all".into(), "-m".into(), "x.gguf".into()].into_iter()).unwrap();
        assert_eq!(with_model.model, Some(PathBuf::from("x.gguf")));
    }

    #[test]
    fn verdict_helpers() {
        assert_eq!(check_min(0.85, Some(0.80)), Verdict::Pass);
        assert_eq!(check_min(0.75, Some(0.80)), Verdict::Fail);
        assert_eq!(check_min(0.75, None), Verdict::Info);
        assert_eq!(check_within(8.0, Some(7.7), 0.5), Verdict::Pass);
        assert_eq!(check_within(20.0, Some(7.7), 0.5), Verdict::Fail);
        assert_eq!(check_within(20.0, None, 0.5), Verdict::Info);
    }
}
