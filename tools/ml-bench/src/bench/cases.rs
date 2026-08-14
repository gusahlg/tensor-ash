use std::borrow::Cow;

use anyhow::{Context, Result, bail};
use tensor_ash::Tensor;

use super::env::SweepMode;

#[derive(Clone)]
pub(super) struct BenchCase {
    pub(super) label: Cow<'static, str>,
    pub(super) b: u32,
    pub(super) m: u32,
    pub(super) n: u32,
    pub(super) k: u32,
    /// Store B as f16 (append `,f16w` to the case spec).
    pub(super) b_f16: bool,
}

impl BenchCase {
    pub(super) const fn fixed(label: &'static str, b: u32, m: u32, n: u32, k: u32) -> Self {
        Self {
            label: Cow::Borrowed(label),
            b,
            m,
            n,
            k,
            b_f16: false,
        }
    }

    pub(super) fn parse(spec: &str) -> Result<Self> {
        let mut fields = spec.split(',');
        let label = fields
            .next()
            .filter(|value| !value.is_empty())
            .context("benchmark case must use label,b,m,n,k with a non-empty label")?;
        let mut dimension = |name| -> Result<u32> {
            let value = fields
                .next()
                .with_context(|| format!("benchmark case '{label}' is missing {name}"))?;
            let parsed = value.parse::<u32>().with_context(|| {
                format!("benchmark case '{label}' has invalid {name}='{value}'")
            })?;
            if parsed == 0 {
                bail!("benchmark case '{label}' has zero {name}");
            }
            Ok(parsed)
        };
        let (b, m, n, k) = (
            dimension("B")?,
            dimension("M")?,
            dimension("N")?,
            dimension("K")?,
        );
        let b_f16 = match fields.next() {
            None => false,
            Some("f16w") => true,
            Some(other) => bail!(
                "benchmark case '{label}' has unknown flag '{other}'; expected label,b,m,n,k[,f16w]"
            ),
        };
        if fields.next().is_some() {
            bail!("benchmark case '{label}' has extra fields; expected label,b,m,n,k[,f16w]");
        }
        Ok(Self {
            label: Cow::Owned(label.into()),
            b,
            m,
            n,
            k,
            b_f16,
        })
    }
}

pub(super) struct BenchResult {
    pub(super) case: BenchCase,
    pub(super) dispatch: tensor_ash::DispatchInfo,
    pub(super) flops: f64,
    pub(super) wall_ms: SampleSummary,
    pub(super) gpu_ms: Option<SampleSummary>,
    pub(super) host_overhead_ms: Option<SampleSummary>,
    pub(super) tflops: f64,
    pub(super) wall_tflops: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SampleSummary {
    pub(super) count: usize,
    pub(super) min: f64,
    pub(super) median: f64,
    pub(super) p95: f64,
}

impl SampleSummary {
    /// Summarize finite, non-negative duration samples using the nearest-rank
    /// definition for p95. Invalid samples are omitted rather than poisoning
    /// otherwise useful benchmark output.
    pub(super) fn new(samples: impl IntoIterator<Item = f64>) -> Option<Self> {
        let mut sorted: Vec<_> = samples
            .into_iter()
            .filter(|sample| sample.is_finite() && *sample >= 0.0)
            .collect();
        if sorted.is_empty() {
            return None;
        }
        sorted.sort_by(f64::total_cmp);

        let count = sorted.len();
        let mid = count / 2;
        let median = if count.is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };
        let p95_index = count.saturating_mul(95).div_ceil(100).saturating_sub(1);
        Some(Self {
            count,
            min: sorted[0],
            median,
            p95: sorted[p95_index],
        })
    }
}

pub(super) fn sweep_cases(mode: SweepMode) -> Vec<BenchCase> {
    const SMOKE: &[BenchCase] = &[
        BenchCase::fixed("smoke 128^3 B=1", 1, 128, 128, 128),
        BenchCase::fixed("smoke 256^3 B=1", 1, 256, 256, 256),
    ];
    const STANDARD: &[BenchCase] = &[
        BenchCase::fixed("square 512^3   B=1", 1, 512, 512, 512),
        BenchCase::fixed("square 1024^3  B=1", 1, 1024, 1024, 1024),
        BenchCase::fixed("square 2048^3  B=1", 1, 2048, 2048, 2048),
        BenchCase::fixed("batched 8x1024^2", 8, 1024, 1024, 1024),
        BenchCase::fixed("tall   4096x1024x1024", 1, 4096, 1024, 1024),
        BenchCase::fixed("wide   1024x4096x1024", 1, 1024, 4096, 1024),
        BenchCase::fixed("odd   1023x1025x1027", 1, 1023, 1025, 1027),
    ];
    const FULL: &[BenchCase] = &[
        BenchCase::fixed("square 512^3   B=1", 1, 512, 512, 512),
        BenchCase::fixed("square 1024^3  B=1", 1, 1024, 1024, 1024),
        BenchCase::fixed("square 2048^3  B=1", 1, 2048, 2048, 2048),
        BenchCase::fixed("square 4096^3  B=1", 1, 4096, 4096, 4096),
        BenchCase::fixed("batched 32x1024^2", 32, 1024, 1024, 1024),
        BenchCase::fixed("batched 8x2048^2", 8, 2048, 2048, 2048),
        BenchCase::fixed("tall   8192x1024x1024", 1, 8192, 1024, 1024),
        BenchCase::fixed("wide   1024x8192x1024", 1, 1024, 8192, 1024),
        BenchCase::fixed("thin K 4096x4096x512", 1, 4096, 4096, 512),
        BenchCase::fixed("fat  K 1024x1024x8192", 1, 1024, 1024, 8192),
        BenchCase::fixed("deep K 64x64x8192", 1, 64, 64, 8192),
        BenchCase::fixed("row 1x4096x4096", 1, 1, 4096, 4096),
        BenchCase::fixed("column 4096x1x4096", 1, 4096, 1, 4096),
        BenchCase::fixed("edge 4095^3", 1, 4095, 4095, 4095),
        BenchCase::fixed("odd   1023x1025x1027", 1, 1023, 1025, 1027),
    ];
    let slice: &[BenchCase] = match mode {
        SweepMode::Smoke => SMOKE,
        SweepMode::Standard => STANDARD,
        SweepMode::Full => FULL,
    };
    slice.to_vec()
}

pub(super) fn host_len(shape: &[u32]) -> Result<usize> {
    let numel = Tensor::numel_checked(shape)?;
    usize::try_from(numel).context("tensor element count does not fit usize")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_modes_choose_expected_case_sets() {
        assert_eq!(sweep_cases(SweepMode::Smoke).len(), 2);
        assert_eq!(sweep_cases(SweepMode::Standard).len(), 7);
        assert_eq!(sweep_cases(SweepMode::Full).len(), 15);
    }

    #[test]
    fn host_len_rejects_bad_shapes() {
        assert!(host_len(&[]).is_err());
        assert!(host_len(&[1, 0, 4]).is_err());
    }

    #[test]
    fn bench_case_fixed_carries_borrowed_label() {
        let case = BenchCase::fixed("hello", 1, 2, 3, 4);
        assert_eq!(&*case.label, "hello");
        assert!(matches!(case.label, Cow::Borrowed(_)));
        assert_eq!(case.b, 1);
        assert_eq!(case.m, 2);
        assert_eq!(case.n, 3);
        assert_eq!(case.k, 4);
    }

    #[test]
    fn bench_case_parser_validates_compact_cli_specs() {
        let case = BenchCase::parse("odd,2,3,5,7").unwrap();
        assert_eq!(
            (&*case.label, case.b, case.m, case.n, case.k),
            ("odd", 2, 3, 5, 7)
        );
        for invalid in ["", "odd,1,2,3", "odd,1,2,0,4", "odd,1,2,3,4,5"] {
            assert!(BenchCase::parse(invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn sample_summary_rejects_an_empty_or_invalid_set() {
        assert_eq!(SampleSummary::new([]), None);
        assert_eq!(SampleSummary::new([f64::NAN, f64::INFINITY, -1.0]), None);
    }

    #[test]
    fn sample_summary_handles_one_sample() {
        assert_eq!(
            SampleSummary::new([2.5]),
            Some(SampleSummary {
                count: 1,
                min: 2.5,
                median: 2.5,
                p95: 2.5,
            })
        );
    }

    #[test]
    fn sample_summary_uses_even_median_and_nearest_rank_p95() {
        let summary = SampleSummary::new((1..=20).rev().map(f64::from)).unwrap();
        assert_eq!(summary.count, 20);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.median, 10.5);
        assert_eq!(summary.p95, 19.0);
    }

    #[test]
    fn sample_summary_omits_invalid_samples_from_count() {
        let summary = SampleSummary::new([3.0, f64::NAN, 1.0, -2.0, 2.0]).unwrap();
        assert_eq!(summary.count, 3);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.median, 2.0);
        assert_eq!(summary.p95, 3.0);
    }
}
