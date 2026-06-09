//! Deterministic helpers shared by `tests/` and `ml_bench`.
//!
//! Public so the benchmark binary and the integration-test crate can pull
//! them in without duplicating code.  Not part of the load-bearing GEMM
//! API — if you're writing application code, you don't need this module.

/// Deterministic pseudo-random fill in `[-1, 1)`, seeded by `seed`.
///
/// Uses a 64-bit LCG.  Not cryptographically random — but stable across
/// runs and platforms, which is what we want for reproducible
/// correctness / benchmark inputs.
pub fn fill_det(out: &mut [f32], seed: u64) {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    for v in out.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (s >> 40) as u32;
        *v = (bits as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
    }
}

/// Stateful version of the same LCG as `fill_det`.  Useful when the test
/// needs to pull a stream of small random integers (e.g. picking
/// dimensions for a property test).
///
/// The integer constants match `fill_det` so changing one stays in sync
/// with the other.
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E3779B97F4A7C15) | 1,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Returns a value uniformly in `[lo, hi]` (both inclusive).
    /// Panics if `lo > hi`.
    pub fn range_u32(&mut self, lo: u32, hi: u32) -> u32 {
        assert!(lo <= hi, "LcgRng::range_u32 lo {lo} > hi {hi}");
        if lo == hi {
            return lo;
        }
        let span = (hi - lo) as u64 + 1;
        lo + (self.next_u64() % span) as u32
    }
}

/// Reference batched matmul on CPU with optional `alpha` and
/// `accumulate`.  Uses `f64` accumulation so that the output is
/// numerically tighter than a naive `f32` sum — close enough to be a
/// reference for the GPU kernel.
///
/// Shape semantics mirror the GPU path: any `batch == 1` input is
/// broadcast against a wider `batch`.  The function panics on shape
/// mismatch; this is reference code, not a parser.
#[allow(clippy::too_many_arguments)]
pub fn cpu_bmm(
    a: &[f32],
    b: &[f32],
    c_init: Option<&[f32]>,
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    accumulate: bool,
) -> Vec<f32> {
    let m_ = m as usize;
    let n_ = n as usize;
    let k_ = k as usize;
    let mk = m_ * k_;
    let kn = k_ * n_;
    let mn = m_ * n_;
    let batch_us = batch as usize;

    let a_batched = a.len() == batch_us * mk;
    let b_batched = b.len() == batch_us * kn;
    assert!(
        a_batched || a.len() == mk,
        "cpu_bmm: A length {} matches neither broadcast ({}) nor batched ({})",
        a.len(),
        mk,
        batch_us * mk
    );
    assert!(
        b_batched || b.len() == kn,
        "cpu_bmm: B length {} matches neither broadcast ({}) nor batched ({})",
        b.len(),
        kn,
        batch_us * kn
    );

    let mut out = vec![0.0f32; batch_us * mn];
    if let Some(c0) = c_init {
        assert_eq!(c0.len(), out.len(), "cpu_bmm: C init has wrong length");
        out.copy_from_slice(c0);
    }
    for bi in 0..batch_us {
        let a_slice = if a_batched {
            &a[bi * mk..(bi + 1) * mk]
        } else {
            a
        };
        let b_slice = if b_batched {
            &b[bi * kn..(bi + 1) * kn]
        } else {
            b
        };
        let c_slice = &mut out[bi * mn..(bi + 1) * mn];
        for i in 0..m_ {
            for j in 0..n_ {
                let mut acc = 0.0f64;
                for kk in 0..k_ {
                    acc += (a_slice[i * k_ + kk] as f64) * (b_slice[kk * n_ + j] as f64);
                }
                let scaled = alpha * (acc as f32);
                c_slice[i * n_ + j] = if accumulate {
                    c_slice[i * n_ + j] + scaled
                } else {
                    scaled
                };
            }
        }
    }
    out
}

/// `(max |a-b|, index)` where `a` and `b` are interpreted positionally.
/// NaN is treated as `(NaN-NaN).abs() = NaN`, which compares false
/// against everything — so a NaN at position `i` does not stop a real
/// difference at position `j` from being reported.
pub fn max_abs_err(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut m = 0.0f32;
    let mut idx = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let e = (x - y).abs();
        if e > m {
            m = e;
            idx = i;
        }
    }
    (m, idx)
}

/// Tolerance for f32 GEMM error: a coarse `8 * K * eps` upper bound
/// on the rounding error of summing `K` f32 products.  Larger than
/// strictly tight, so it doesn't fire on noise.
pub fn tolerance(k: u32) -> f32 {
    8.0 * (k as f32) * f32::EPSILON
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_det_is_deterministic() {
        let mut a = [0.0f32; 8];
        let mut b = [0.0f32; 8];
        fill_det(&mut a, 42);
        fill_det(&mut b, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn fill_det_stays_in_range() {
        let mut buf = [0.0f32; 1024];
        fill_det(&mut buf, 7);
        for v in buf {
            assert!((-1.0..1.0).contains(&v), "value {v} out of [-1, 1)");
        }
    }

    #[test]
    fn fill_det_varies_with_seed() {
        let mut a = [0.0f32; 16];
        let mut b = [0.0f32; 16];
        fill_det(&mut a, 1);
        fill_det(&mut b, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn lcg_range_respects_bounds() {
        let mut rng = LcgRng::new(0xCAFE);
        for _ in 0..1000 {
            let v = rng.range_u32(7, 13);
            assert!((7..=13).contains(&v));
        }
    }

    #[test]
    fn lcg_range_collapses_when_lo_eq_hi() {
        let mut rng = LcgRng::new(1);
        for _ in 0..16 {
            assert_eq!(rng.range_u32(42, 42), 42);
        }
    }

    #[test]
    fn cpu_bmm_matches_hand_computed_rank2() {
        // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let c = cpu_bmm(&a, &b, None, 1, 2, 2, 2, 1.0, false);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn cpu_bmm_accumulate_adds_to_c_init() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let c0 = [100.0f32, 100.0, 100.0, 100.0];
        let c = cpu_bmm(&a, &b, Some(&c0), 1, 2, 2, 2, 1.0, true);
        assert_eq!(c, vec![119.0, 122.0, 143.0, 150.0]);
    }

    #[test]
    fn cpu_bmm_alpha_scales_product() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let c = cpu_bmm(&a, &b, None, 1, 2, 2, 2, 0.5, false);
        assert_eq!(c, vec![9.5, 11.0, 21.5, 25.0]);
    }

    #[test]
    fn cpu_bmm_broadcasts_a() {
        // Batched B against single A; both batches should produce the
        // same per-batch output.
        let a = [1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
        let b = [
            // batch 0: [[1,2],[3,4]]
            1.0f32, 2.0, 3.0, 4.0, // batch 1: [[10,20],[30,40]]
            10.0, 20.0, 30.0, 40.0,
        ];
        let c = cpu_bmm(&a, &b, None, 2, 2, 2, 2, 1.0, false);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn cpu_bmm_broadcasts_b() {
        let a = [
            // batch 0: [[1,2],[3,4]]
            1.0f32, 2.0, 3.0, 4.0, // batch 1: [[10,20],[30,40]]
            10.0, 20.0, 30.0, 40.0,
        ];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let c = cpu_bmm(&a, &b, None, 2, 2, 2, 2, 1.0, false);

        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn max_abs_err_finds_largest_difference() {
        let (err, idx) = max_abs_err(&[1.0, 3.0, -5.0], &[1.5, 1.0, -4.0]);
        assert_eq!(err, 2.0);
        assert_eq!(idx, 1);
    }

    #[test]
    fn tolerance_grows_linearly_in_k() {
        assert!(tolerance(2 * 100) > tolerance(100));
        assert!(tolerance(0) == 0.0);
    }
}
