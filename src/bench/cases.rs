use std::borrow::Cow;

use anyhow::{Context, Result};
use tensor_ash::Tensor;

use super::env::SweepMode;

#[derive(Clone)]
pub(super) struct BenchCase {
    pub(super) label: Cow<'static, str>,
    pub(super) b: u32,
    pub(super) m: u32,
    pub(super) n: u32,
    pub(super) k: u32,
}

impl BenchCase {
    pub(super) const fn fixed(label: &'static str, b: u32, m: u32, n: u32, k: u32) -> Self {
        Self {
            label: Cow::Borrowed(label),
            b,
            m,
            n,
            k,
        }
    }
}

pub(super) struct BenchResult {
    pub(super) case: BenchCase,
    pub(super) flops: f64,
    pub(super) wall_ms: f64,
    pub(super) gpu_ms: f64,
    pub(super) tflops: f64,
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
        assert_eq!(sweep_cases(SweepMode::Full).len(), 11);
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
}
