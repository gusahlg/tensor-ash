//! Accumulator precision of the tensor-core (coopmat1) GEMM family.

use anyhow::{Result, bail};

/// Accumulator precision for the coopmat1 tensor-core GEMMs.
///
/// * `F32` (default): f16 products accumulated in f32 — HMMA.*.F32,
///   which GeForce Ampere/Ada tensor cores run at HALF their f16 rate.
/// * `F16`: f16 accumulator fragments (HMMA.*.F16).  Every MMA rounds
///   the running sum to f16 — the numerics of cuBLAS
///   `CUBLAS_COMPUTE_16F`, which llama.cpp's CUDA backend uses for f16
///   weight GEMMs with more than 16 rows at default precision.  Sums
///   above 65504 overflow to infinity.
/// * `F16Promote`: f16 partial fragments over at most 128 K, then added
///   into f32 accumulators: the error of one chunk partial's f16
///   rounding instead of the whole running sum's, at the cost of half
///   again the f32 accumulator registers.  The routed coopmat1 kernels
///   promote every 128 K (4 K-tiles); any backend serving this class
///   (e.g. a PTX `mma` route promoting every 64-K tile) must keep its
///   chunks at or below 128 K.  The `_pk8_` (256 K) chunk variant is a
///   pin-only experiment outside the class.
///
/// The class, not the backend, is the policy: one value selects the
/// accumulator for whichever f16 tensor-core GEMM implementation runs
/// the op.  Strings: `f32`, `f16`, `f16p` ([`Self::as_str`]).
///
/// Never chosen by a heuristic or the tuner: it is a numerics class
/// the caller opts into through [`crate::ExecutorConfig::coopmat_acc`]
/// (or `ML_COOPMAT_ACC` for env-configured executors), or by pinning an
/// `*_acc16` / `*_acc16p` kernel with `ML_KERNEL`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum CoopmatAcc {
    #[default]
    F32,
    F16,
    F16Promote,
}

impl CoopmatAcc {
    /// Parse `f32` / `f16` / `f16p` (= `promote`; plus a few spellings).
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "f32" | "fp32" | "exact" => Ok(Self::F32),
            "f16" | "fp16" | "fast16" | "acc16" => Ok(Self::F16),
            "promote" | "f16p" | "f16promote" | "f16_promote" | "acc16p" => Ok(Self::F16Promote),
            other => {
                bail!("invalid coopmat accumulator '{other}', expected f32, f16, or f16p (promote)")
            }
        }
    }

    /// `ML_COOPMAT_ACC` (unset = `F32`).
    pub fn from_env() -> Result<Self> {
        match std::env::var("ML_COOPMAT_ACC") {
            Ok(value) => Self::parse(&value).map_err(|err| err.context("ML_COOPMAT_ACC")),
            Err(_) => Ok(Self::F32),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            // `f16p`: the spelling `ML_PTX_GEMM` uses for the same class.
            Self::F16Promote => "f16p",
        }
    }

    /// Accumulator class of a catalog kernel, from its name suffix
    /// (`*_acc16` = F16, `*_acc16p` = F16Promote; everything else
    /// accumulates in f32).
    #[inline]
    pub fn of_kernel_name(name: &str) -> Self {
        if name.ends_with("_acc16") {
            Self::F16
        } else if name.ends_with("_acc16p") {
            Self::F16Promote
        } else {
            Self::F32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_documented_spellings() {
        assert_eq!(CoopmatAcc::parse("f32").unwrap(), CoopmatAcc::F32);
        assert_eq!(CoopmatAcc::parse("").unwrap(), CoopmatAcc::F32);
        assert_eq!(CoopmatAcc::parse(" F16 ").unwrap(), CoopmatAcc::F16);
        assert_eq!(CoopmatAcc::parse("fast16").unwrap(), CoopmatAcc::F16);
        assert_eq!(
            CoopmatAcc::parse("promote").unwrap(),
            CoopmatAcc::F16Promote
        );
        assert!(CoopmatAcc::parse("bf16").is_err());
    }

    #[test]
    fn kernel_name_suffix_decides_the_class() {
        assert_eq!(
            CoopmatAcc::of_kernel_name("f16w_a16_coopmat_m64n64_acc16"),
            CoopmatAcc::F16
        );
        assert_eq!(
            CoopmatAcc::of_kernel_name("f16w_coopmat_acc16p"),
            CoopmatAcc::F16Promote
        );
        assert_eq!(
            CoopmatAcc::of_kernel_name("f16w_coopmat_aligned"),
            CoopmatAcc::F32
        );
        assert_eq!(CoopmatAcc::of_kernel_name("large_bda_v4"), CoopmatAcc::F32);
    }
}
