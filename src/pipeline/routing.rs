//! Shape-based kernel routing: row-GEMV pack tiles, coopmat wave-fill,
//! and BDA / f16w family promotion.

use super::catalog::KernelSelection;
use super::precision::CoopmatAcc;

/// Decode f16w row-GEMV tile: sixteen K-slices at tile_n=32 on deep-K
/// / narrow-N / square-ish shapes; VCOLS=2 (tile_n=64) on wide-N
/// moderate-K (gate/up, lm_head).
#[inline]
pub fn f16w_row_uses_k16(k: u32, n: u32) -> bool {
    k >= 4096 || n <= 512 || (k >= 2048 && n <= 2048)
}

/// `tile_n` matching [`f16w_row_packed_selection`]: 32 (k16) or 64
/// (k16_v2).  Used to pack B into `[N/tile_n][K][tile_n]`.
#[inline]
pub fn f16w_row_tile_n(k: u32, n: u32) -> u32 {
    if f16w_row_uses_k16(k, n) || !n.is_multiple_of(64) {
        32
    } else {
        64
    }
}

#[inline]
pub(crate) fn f16w_row_selection(k: u32, n: u32) -> KernelSelection {
    if f16w_row_uses_k16(k, n) {
        KernelSelection::F16wRowBdaK16
    } else {
        KernelSelection::F16wRowBdaK16V2
    }
}

#[inline]
pub(crate) fn f16w_row_packed_selection(k: u32, n: u32) -> KernelSelection {
    if f16w_row_uses_k16(k, n) || !n.is_multiple_of(64) {
        KernelSelection::F16wRowBdaK16Packed
    } else {
        KernelSelection::F16wRowBdaK16V2Packed
    }
}

/// Coopmat1 tile shapes.  `T64Mfast` is the 64x64 tile launched in
/// column-major (M-fastest) CTA order, so a wide-N grid streams B from
/// DRAM ~once instead of once per row block (`RASTER_M_FAST` in
/// `matmul_coopmat_kernel.glsl`).
///
/// Measured (`benchmarks/campaign/coopmat-sm.md`, `ml_bench ab`, GPU
/// median TF/s, f16w = f32 A, a16 = f16 A/C; "cold" = weights evicted
/// from L2 before every GEMM, `ML_AB_COLD_MB=64`, as in a forward pass):
///
/// | shape | GPU | 128x128 | 64x64 | 64x64 M-fast | 128x64 |
/// |---|---|---:|---:|---:|---:|
/// | a16 512x2560x2048 (QKV), cold | 3070 | 29.0 | 34.4 | **35.3** | 29.6 |
/// | a16 512x5632x2048 (gate/up), cold | 3070 | 32.7 | 29.9 | **36.7** | 34.7 |
/// | a16 512x2048x5632 (down), cold | 3070 | 24.9 | 32.9 | **34.4** | 33.2 |
/// | a16 512x5632x2048, cold | 4060 | 23.3 | 26.1 | **28.8** | 26.4 |
/// | a16 512x5632x2048, warm | 4060 | 24.3 | 28.0 | **28.3** | 28.0 |
/// | a16 1920x5632x2048, cold | 4060 | 26.4 | 29.0 | **29.9** | 28.8 |
/// | a16 1920x5632x2048, warm | 3070 | 34.0 | 26.9 | 30.0 | **35.6** |
/// | f16w 768x2048x2048 | 3070 | 21.1 | **31.5** | 29.6 | 26.7 |
/// | f16w 768x2048x2048 | 4060 | 24.5 | 24.2 | 23.8 | **25.6** |
/// | f16w 256x32000x2048 | 3070 | 30.6 | 24.8 | 32.2 | **34.5** |
/// | f16w 4096³ | 3070 | 34.4 | 23.3 | 14.2 | **34.4** |
///
/// 128x64 matches or beats 128x128 on every measured 3070 shape (the
/// old 4096³ veto is a tie), so 128x128 and 64x128 are no longer
/// auto-routed when the SM count is known; they stay selectable and
/// in the SM-agnostic fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CoopmatTile {
    T128,
    T64,
    T64Mfast,
    T64x128,
    T128x64,
}

/// Device inputs of the coopmat tile rule.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CoopmatDevice {
    /// Shader cores (SMs), [`crate::VulkanContext::sm_count`]; `None`
    /// selects the SM-agnostic [`coopmat_tile_fallback`].
    pub sm_count: Option<u32>,
    /// L2 lower bound, [`crate::VulkanContext::l2_bytes_estimate`];
    /// `None` is treated as a small (GA104-class, 4 MiB) L2.
    pub l2_bytes: Option<u64>,
}

const MIB: u64 = 1 << 20;
/// An L2 this large keeps a TinyLlama-class prefill activation
/// resident (AD107: 24 MiB); GA104's 4 MiB does not.
const LARGE_L2_BYTES: u64 = 16 * MIB;
/// CTAs per SM from the driver register counts (identical on sm_86
/// and sm_89; `examples/coopmat_occupancy.rs`): 64x64 is 72 regs x 4
/// warps -> 7; 128x64 is 128 regs x 4 warps -> 4.
const CTAS_PER_SM_64: u64 = 7;

/// The accumulator-independent part of the coopmat tile rules: one
/// GEMM grid on one device, as SM load balance and DRAM traffic see
/// it.  [`coopmat_tile`] (f32 accumulate) is this plus a per-tile
/// efficiency table; a rule for another accumulator class (the
/// f16-accumulate tiles prefer wider tiles) composes the same way:
/// build the grid, score its own candidate tiles with
/// [`CoopmatGrid::balance`], and reuse [`CoopmatGrid::m_fast_64`] for
/// the raster of a 64x64 grid (raster is about operand bytes, not the
/// accumulator).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CoopmatGrid {
    batch: u64,
    m: u64,
    n: u64,
    sms: u64,
    a_bytes: u64,
    b_bytes: u64,
    l2_bytes: u64,
}

impl CoopmatGrid {
    /// `None` when the device reports no SM count (callers then use
    /// the SM-agnostic [`coopmat_tile_fallback`]).
    pub(crate) fn new(
        batch: u32,
        m: u32,
        n: u32,
        k: u32,
        a16: bool,
        device: CoopmatDevice,
    ) -> Option<Self> {
        let sms = device.sm_count.filter(|&count| count > 0)?;
        let (m, n, k) = (u64::from(m), u64::from(n), u64::from(k));
        Some(Self {
            batch: u64::from(batch),
            m,
            n,
            sms: u64::from(sms),
            a_bytes: m * k * if a16 { 2 } else { 4 },
            b_bytes: k * n * 2,
            l2_bytes: device.l2_bytes.unwrap_or(0),
        })
    }

    /// The L2 is large enough to keep a prefill activation resident.
    pub(crate) fn large_l2(&self) -> bool {
        self.l2_bytes >= LARGE_L2_BYTES
    }

    /// How many times a 64x64 grid streams B in x-fastest order
    /// (`xp`) and A in M-fastest order (`mp`), with an in-flight
    /// window of `7 S` CTAs.
    fn passes_64(&self) -> (u64, u64) {
        let (rows, cols) = (self.m / 64, self.n / 64);
        let window = (self.sms * CTAS_PER_SM_64) as f64;
        let passes = |blocks: u64, other: u64| -> u64 {
            (blocks as f64 / (window / other as f64).max(1.0)).ceil() as u64
        };
        (passes(rows, cols), passes(cols, rows))
    }

    /// Launch the 64x64 grid M-fastest ([`CoopmatTile::T64Mfast`])?
    /// Fitted on cold-L2 sweeps (`ML_AB_COLD_MB=64`: weights evicted
    /// before every GEMM, as in a model forward):
    ///
    /// * M-fast must move clearly fewer bytes
    ///   (`|A| mp + |B| < 0.75 (|B| xp + |A|)`), or the grid is a
    ///   single x-fast pass at least 8 row blocks tall and 32 column
    ///   blocks wide (M >= 512, N >= 2048: +3-14% on the 3070 at
    ///   M=512, where traffic is equal; 1024³ lost 3% warm);
    /// * and its A re-reads must be cheap: `|A| (mp - 1) <= 8 MiB`, or
    ///   A fits half of a large L2 (re-reads then hit in L2).
    ///
    /// The earlier "B fits a large L2 -> x-fast" exception was a
    /// warm-L2 artifact: with cold weights M-fast wins every a16 pp512
    /// GEMM on the 4060 too.
    pub(crate) fn m_fast_64(&self) -> bool {
        let (xp, mp) = self.passes_64();
        let (a, b) = (self.a_bytes, self.b_bytes);
        let cheap_a = a * (mp - 1) <= 8 * MIB || (self.large_l2() && 2 * a <= self.l2_bytes);
        let fewer_bytes = ((a * mp + b) as f64) < 0.75 * (b * xp + a) as f64;
        let tall_single_pass = xp == 1 && self.m / 64 >= 8 && self.n / 64 >= 32;
        cheap_a && (fewer_bytes || tall_single_pass)
    }

    /// SM load balance times an occupancy floor for a grid of `tiles`
    /// CTAs: `n = ceil(tiles / S)` land on the busiest SM, and fewer
    /// than 3 per SM cannot hide the load/MMA latency of any tile.
    pub(crate) fn balance(&self, tiles: u64) -> f64 {
        let per_sm = tiles.div_ceil(self.sms);
        tiles as f64 / (self.sms * per_sm) as f64 * (per_sm as f64 / 3.0).min(1.0)
    }
}

/// SM-count-aware coopmat tile pick for the f32-accumulate kernels,
/// f16w (`a16 = false`) or f16-activation (`a16 = true`); `k % 32 == 0`
/// is the caller's precondition.  With no SM count this is exactly
/// the exp/grokmerge rule ([`coopmat_tile_fallback`]).
///
/// With `S` SMs the rule compares the two tiles that won the two-GPU
/// sweep, 128x64 and 64x64:
///
/// 1. **Raster of the 64x64 grid**: [`CoopmatGrid::m_fast_64`].
/// 2. **Wave balance** per tile: [`CoopmatGrid::balance`].
/// 3. **Tile efficiency rho** of 64x64 relative to 128x64 (fit to the
///    sweeps; unchanged by the cold-L2 refit): large L2 — 1.02 a16
///    (f16 A), 0.92 f16w (f32 A); small or unknown L2 — a16 1.02, but
///    0.8 when an x-fast grid re-streams B; f16w 0.9 on a multi-pass
///    grid (`xp > 1`), else 1.0 (`|A| <= 4 MiB`) / 0.85 (larger A).
///
/// `score = balance * rho`; ties go to 64x64 (finer balance, more
/// resident warps).  M or N not 128-/64-aligned: 64x64 only (`None`
/// below 64-alignment).
pub(crate) fn coopmat_tile(
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    a16: bool,
    device: CoopmatDevice,
) -> Option<CoopmatTile> {
    let Some(grid) = CoopmatGrid::new(batch, m, n, k, a16, device) else {
        return coopmat_tile_fallback(batch, m, n);
    };
    if !m.is_multiple_of(64) || !n.is_multiple_of(64) {
        return None;
    }
    let m_fast = grid.m_fast_64();
    let t64 = if m_fast {
        CoopmatTile::T64Mfast
    } else {
        CoopmatTile::T64
    };
    if !m.is_multiple_of(128) {
        return Some(t64);
    }
    let (xp, _) = grid.passes_64();
    let rho = match (grid.large_l2(), a16) {
        (true, true) => 1.02,
        (true, false) => 0.92,
        (false, true) if !m_fast && xp > 1 => 0.8,
        (false, true) => 1.02,
        (false, false) if xp > 1 => 0.9,
        (false, false) if grid.a_bytes <= 4 * MIB => 1.0,
        (false, false) => 0.85,
    };
    let cols = grid.n / 64;
    let s64 = grid.balance(grid.batch * (grid.m / 64) * cols) * rho;
    let s128x64 = grid.balance(grid.batch * (grid.m / 128) * cols);
    Some(if s64 >= s128x64 {
        t64
    } else {
        CoopmatTile::T128x64
    })
}

/// The SM-agnostic rule (exp/grokmerge): calibrated on the 46-SM
/// RTX 3070 for large squares and the 24-SM RTX 4060 for M=512
/// prefill.  Used when the device reports no shader-core count.
pub(crate) fn coopmat_tile_fallback(batch: u32, m: u32, n: u32) -> Option<CoopmatTile> {
    if !m.is_multiple_of(64) || !n.is_multiple_of(64) {
        return None;
    }
    let m128 = m.is_multiple_of(128);
    let n128 = n.is_multiple_of(128);
    if m <= 512 {
        // Wide prefill (gate/up, concat QKV): 128x64 won on AD107
        // at M=512.  Narrower M stays on 64x64 (unmeasured for 128x64).
        if m == 512 && m128 && n > 4 * m {
            return Some(CoopmatTile::T128x64);
        }
        if m == 512 && n128 && m > 4 * n {
            return Some(CoopmatTile::T64x128);
        }
        return Some(CoopmatTile::T64);
    }
    if !m128 || !n128 {
        return Some(if m128 {
            CoopmatTile::T128x64
        } else if n128 {
            CoopmatTile::T64x128
        } else {
            CoopmatTile::T64
        });
    }
    let tiles_128 = u64::from(batch)
        .saturating_mul(u64::from(m / 128))
        .saturating_mul(u64::from(n / 128));
    Some(if tiles_128 < 96 {
        CoopmatTile::T64
    } else {
        CoopmatTile::T128
    })
}

/// Tile pick for the f16-accumulator classes (`F16`, `F16Promote`).
/// Halving the accumulator registers (f16 fragments: 4 regs instead of
/// 8 per 16x16) moves the optimum to the WIDER tiles: RTX 4060 a16
/// interleaved A/B (`ml_bench ab`, 7 rounds, GPU median ms,
/// benchmarks/campaign/f16acc.md):
///
/// | shape (acc16) | 64x64 | 128x64 | 64x128 | 128x128 |
/// |---|---|---|---|---|
/// | 512x2048x2048 | 0.140 | 0.140 | 0.123 | **0.121** |
/// | 512x2560x2048 | 0.160 | **0.144** | 0.148 | 0.164 |
/// | 512x5632x2048 | 0.346 | 0.310 | **0.287** | 0.321 |
/// | 512x2048x5632 | 0.362 | 0.364 | **0.316** | 0.318 |
/// | 2048³ | 0.478 | 0.448 | **0.416** | 0.425 |
/// | 4096³ | 4.664 | 3.459 | 3.568 | **3.320** |
///
/// Promote ranks the same way (within ~2%).  Rule: 128x128 on grids of
/// more than 256 128-tiles, else 64x128 whenever N % 128 == 0 (never
/// more than 2.8% off the per-shape best above), else 128x64 / 64x64.
/// `None` when M or N is not 64-aligned.
///
/// This rule is SM-count-agnostic: it was measured on the 4060 and
/// confirmed end to end on the 3070 (pp512 1.28x), but no acc16
/// candidate has been scored with [`CoopmatGrid::balance`] yet, and
/// the acc16 tiles have no M-fast sibling, so [`coopmat_tile_acc`]
/// does not call [`CoopmatGrid::m_fast_64`] for these classes.
pub(crate) fn coopmat_tile_acc16(batch: u32, m: u32, n: u32) -> Option<CoopmatTile> {
    if !m.is_multiple_of(64) || !n.is_multiple_of(64) {
        return None;
    }
    let (m128, n128) = (m.is_multiple_of(128), n.is_multiple_of(128));
    let tiles_128 = u64::from(batch)
        .saturating_mul(u64::from(m / 128))
        .saturating_mul(u64::from(n / 128));
    Some(if m128 && n128 && tiles_128 > 256 {
        CoopmatTile::T128
    } else if n128 {
        CoopmatTile::T64x128
    } else if m128 {
        CoopmatTile::T128x64
    } else {
        CoopmatTile::T64
    })
}

/// Accumulator-aware tile dispatcher: `F32` takes the SM-aware
/// [`coopmat_tile`] (grid + its rho table), the f16-accumulator classes
/// take [`coopmat_tile_acc16`].  The class never changes here: the
/// caller's policy picks the rule, the rule picks the tile.
pub(crate) fn coopmat_tile_acc(
    acc: CoopmatAcc,
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    a16: bool,
    device: CoopmatDevice,
) -> Option<CoopmatTile> {
    match acc {
        CoopmatAcc::F32 => coopmat_tile(batch, m, n, k, a16, device),
        CoopmatAcc::F16 | CoopmatAcc::F16Promote => coopmat_tile_acc16(batch, m, n),
    }
}

/// Every coopmat1 kernel as (selection, tile, a16, accumulator): the
/// single table the tile/acc mappings below read.  The 64x64 M-fast
/// raster exists for the f32 accumulator only.
const COOPMAT1: &[(KernelSelection, CoopmatTile, bool, CoopmatAcc)] = {
    use CoopmatAcc::{F16, F16Promote, F32};
    use CoopmatTile::*;
    use KernelSelection as K;
    &[
        (K::F16wCoopmat, T128, false, F32),
        (K::F16wA16Coopmat, T128, true, F32),
        (K::F16wCoopmatM64, T64, false, F32),
        (K::F16wA16CoopmatM64, T64, true, F32),
        (K::F16wCoopmatM64Mfast, T64Mfast, false, F32),
        (K::F16wA16CoopmatM64Mfast, T64Mfast, true, F32),
        (K::F16wCoopmatM64N128, T64x128, false, F32),
        (K::F16wA16CoopmatM64N128, T64x128, true, F32),
        (K::F16wCoopmatM128N64, T128x64, false, F32),
        (K::F16wA16CoopmatM128N64, T128x64, true, F32),
        (K::F16wCoopmatAcc16, T128, false, F16),
        (K::F16wA16CoopmatAcc16, T128, true, F16),
        (K::F16wCoopmatM64Acc16, T64, false, F16),
        (K::F16wA16CoopmatM64Acc16, T64, true, F16),
        (K::F16wCoopmatM64N128Acc16, T64x128, false, F16),
        (K::F16wA16CoopmatM64N128Acc16, T64x128, true, F16),
        (K::F16wCoopmatM128N64Acc16, T128x64, false, F16),
        (K::F16wA16CoopmatM128N64Acc16, T128x64, true, F16),
        (K::F16wCoopmatAcc16p, T128, false, F16Promote),
        (K::F16wA16CoopmatAcc16p, T128, true, F16Promote),
        (K::F16wCoopmatM64Acc16p, T64, false, F16Promote),
        (K::F16wA16CoopmatM64Acc16p, T64, true, F16Promote),
        (K::F16wCoopmatM64N128Acc16p, T64x128, false, F16Promote),
        (K::F16wA16CoopmatM64N128Acc16p, T64x128, true, F16Promote),
        (K::F16wCoopmatM128N64Acc16p, T128x64, false, F16Promote),
        (K::F16wA16CoopmatM128N64Acc16p, T128x64, true, F16Promote),
    ]
};

impl CoopmatTile {
    /// The f16w (f32-A) or a16 (f16-A/C) f32-accumulate kernel for
    /// this tile.
    pub(crate) fn selection(self, a16: bool) -> KernelSelection {
        self.selection_acc(a16, CoopmatAcc::F32)
    }

    /// The kernel for this tile, A storage, and accumulator class.  The
    /// f16-accumulator classes have no M-fast twin: `T64Mfast` maps to
    /// the x-fast 64x64 kernel of that class (same tile, same numerics;
    /// only the CTA raster differs).
    pub(crate) fn selection_acc(self, a16: bool, acc: CoopmatAcc) -> KernelSelection {
        let tile = if self == Self::T64Mfast && acc != CoopmatAcc::F32 {
            Self::T64
        } else {
            self
        };
        COOPMAT1
            .iter()
            .find(|&&(_, t, is_a16, class)| t == tile && is_a16 == a16 && class == acc)
            .map(|&(selection, ..)| selection)
            .expect("COOPMAT1 covers every (tile, a16, acc)")
    }
}

/// (tile, a16, accumulator) of a coopmat1 kernel selection, `None` for
/// every other kernel.
pub(crate) fn coopmat1_parts(
    selection: KernelSelection,
) -> Option<(CoopmatTile, bool, CoopmatAcc)> {
    COOPMAT1
        .iter()
        .find(|&&(s, ..)| s == selection)
        .map(|&(_, tile, a16, acc)| (tile, a16, acc))
}

pub(crate) fn coopmat_selection(
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    a16: bool,
    device: CoopmatDevice,
) -> Option<KernelSelection> {
    Some(coopmat_tile(batch, m, n, k, a16, device)?.selection(a16))
}

pub(crate) fn to_f16w(tile: KernelSelection) -> KernelSelection {
    match tile {
        KernelSelection::Large | KernelSelection::M64N128 => KernelSelection::F16wLargeBdaV4,
        KernelSelection::M128N64 | KernelSelection::M128N64K64 => {
            KernelSelection::F16wM128N64K64BdaV4
        }
        KernelSelection::K64 => KernelSelection::F16wK64BdaV4,
        _ => KernelSelection::F16wSmallBdaV4,
    }
}

pub(crate) fn maybe_to_bda(tile: KernelSelection) -> KernelSelection {
    match tile {
        KernelSelection::Large => KernelSelection::LargeBdaV4,
        KernelSelection::Small => KernelSelection::SmallBdaV4,
        KernelSelection::M128N64K64 => KernelSelection::M128N64K64BdaV4,
        // K64 promotes to the TM=8 TN=4 register-tile variant: empirically
        // wins +6-7% on every K64-routed shape (medium_384,
        // skinny_1024x128x512, wide_128x1024x512) vs the TM=4 TN=4 default
        // by trading half the active threads for double the M-side
        // register strip per thread (verified with 10-round interleaved
        // A/B on RTX 3070).  The plain V4 sibling stays selectable via
        // ML_KERNEL=k64_bda_v4 for back-comparison.
        KernelSelection::K64 => KernelSelection::K64BdaV4Tm8Tn4,
        KernelSelection::M128N64 => KernelSelection::M128N64BdaV4,
        KernelSelection::M64N128 => KernelSelection::M64N128BdaV4,
        // TN=2 has no V4 path; use the plain BDA fallback.
        KernelSelection::M64N32 => KernelSelection::M64N32Bda,
        other => other,
    }
}

#[cfg(test)]
mod bda_tests {
    use super::*;

    #[test]
    fn bda_promotion_covers_every_auto_target() {
        // The auto-selector's possible returns should each promote to
        // their BDA_V4 sibling (or BDA for TN=2 / unchanged when no
        // BDA path exists).  Listing them explicitly here keeps the
        // rule from silently drifting when new kernels land.
        assert_eq!(
            maybe_to_bda(KernelSelection::Large),
            KernelSelection::LargeBdaV4
        );
        assert_eq!(
            maybe_to_bda(KernelSelection::Small),
            KernelSelection::SmallBdaV4
        );
        assert_eq!(
            maybe_to_bda(KernelSelection::M128N64K64),
            KernelSelection::M128N64K64BdaV4
        );
        assert_eq!(
            maybe_to_bda(KernelSelection::K64),
            KernelSelection::K64BdaV4Tm8Tn4
        );
        assert_eq!(
            maybe_to_bda(KernelSelection::M128N64),
            KernelSelection::M128N64BdaV4
        );
        assert_eq!(
            maybe_to_bda(KernelSelection::M64N128),
            KernelSelection::M64N128BdaV4
        );
        // TN=2 stays on the plain BDA path (no V4 sibling).
        assert_eq!(
            maybe_to_bda(KernelSelection::M64N32),
            KernelSelection::M64N32Bda
        );
        // No BDA sibling at all — pass through.
        assert_eq!(maybe_to_bda(KernelSelection::Bk16), KernelSelection::Bk16);
        assert_eq!(maybe_to_bda(KernelSelection::V2), KernelSelection::V2);
    }

    #[test]
    fn coopmat_fallback_is_the_grokmerge_rule() {
        use CoopmatTile::*;
        let tile = coopmat_tile_fallback;
        // q/o 512x2048: 4:1 is not strictly wider than 4:1 → 64x64.
        assert_eq!(tile(1, 512, 2048), Some(T64));
        assert_eq!(tile(1, 128, 2048), Some(T64));
        assert_eq!(tile(1, 256, 384), Some(T64));
        // Concat QKV 512x2560 and gate/up 512x5632: 128x64 won.
        assert_eq!(tile(1, 512, 2560), Some(T128x64));
        assert_eq!(tile(1, 512, 5632), Some(T128x64));
        // Large squares keep 128x128.
        assert_eq!(tile(1, 2048, 2048), Some(T128));
        assert_eq!(tile(1, 1024, 1024), Some(T64));
        // 64-aligned but not 128-aligned has no 128-tile route.
        assert_eq!(tile(1, 192, 320), Some(T64));
        assert_eq!(tile(1, 192, 300), None);
    }

    const RTX4060: CoopmatDevice = CoopmatDevice {
        sm_count: Some(24),
        l2_bytes: Some(24 << 20),
    };
    const RTX3070: CoopmatDevice = CoopmatDevice {
        sm_count: Some(46),
        l2_bytes: Some(1 << 20),
    };

    #[test]
    fn coopmat_tile_without_sm_count_is_the_fallback() {
        // No shader-core query (or ML_SM_COUNT=0): the exp/grokmerge
        // rule, whatever the L2 estimate says.
        for l2_bytes in [None, Some(1 << 20), Some(24 << 20)] {
            let device = CoopmatDevice {
                sm_count: None,
                l2_bytes,
            };
            for (m, n) in [
                (512, 2048),
                (512, 2560),
                (2048, 2048),
                (1024, 1024),
                (192, 320),
            ] {
                for (k, a16) in [(2048, false), (5632, true)] {
                    assert_eq!(
                        coopmat_tile(1, m, n, k, a16, device),
                        coopmat_tile_fallback(1, m, n),
                        "{m}x{n}x{k} a16={a16} l2={l2_bytes:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn coopmat_tile_rtx4060_24_sms_large_l2() {
        use CoopmatTile::*;
        let tile = |m, n, k, a16| coopmat_tile(1, m, n, k, a16, RTX4060);
        // TinyLlama pp512 (f16 activations), cold weights: 64x64
        // M-fast won all four GEMMs (the warm-L2 sweep had preferred
        // x-fast because B stayed resident between repeats).
        assert_eq!(tile(512, 2048, 2048, true), Some(T64Mfast)); // q/o
        assert_eq!(tile(512, 2560, 2048, true), Some(T64Mfast)); // concat QKV
        assert_eq!(tile(512, 5632, 2048, true), Some(T64Mfast)); // gate/up
        assert_eq!(tile(512, 2048, 5632, true), Some(T64Mfast)); // down
        // Long prompts: A (7.5 MiB) fits the 24 MiB L2, so M-fast's
        // A re-reads are cheap.
        assert_eq!(tile(1920, 5632, 2048, true), Some(T64Mfast));
        // Single-pass, 2 row blocks: x-fast.
        assert_eq!(tile(128, 2048, 2048, true), Some(T64));
        // f16w (f32 A): 128x64 fills 8 full waves at 768x2048...
        assert_eq!(tile(768, 2048, 2048, false), Some(T128x64));
        assert_eq!(tile(384, 2048, 2048, false), Some(T128x64));
        // ...64x64 balances better at 1024³ / 512x2048x5632.
        assert_eq!(tile(1024, 1024, 1024, false), Some(T64));
        assert_eq!(tile(512, 2048, 5632, false), Some(T64));
        // Wide N: 64x64 re-streams B; 128x64 wins.
        assert_eq!(tile(256, 32000, 2048, false), Some(T128x64));
        assert_eq!(tile(4096, 4096, 4096, false), Some(T128x64));
    }

    #[test]
    fn coopmat_tile_rtx3070_46_sms_small_l2() {
        use CoopmatTile::*;
        let tile = |m, n, k, a16| coopmat_tile(1, m, n, k, a16, RTX3070);
        // pp512, cold weights: 64x64 M-fast wins all four (single
        // x-fast pass at M=512 for q/o, QKV, down: +2.7-4.8%).
        assert_eq!(tile(512, 2048, 2048, true), Some(T64Mfast));
        assert_eq!(tile(512, 2560, 2048, true), Some(T64Mfast));
        assert_eq!(tile(512, 5632, 2048, true), Some(T64Mfast));
        assert_eq!(tile(512, 2048, 5632, true), Some(T64Mfast));
        // Long prompts: A and B both exceed the 4 MiB L2 → 128x64.
        assert_eq!(tile(1920, 5632, 2048, true), Some(T128x64));
        assert_eq!(tile(1024, 5632, 2048, true), Some(T128x64));
        // 46 SMs: 128x64 at 768x2048 is 192 tiles = 1.04 waves → 64x64.
        assert_eq!(tile(768, 2048, 2048, false), Some(T64));
        assert_eq!(tile(384, 2048, 2048, false), Some(T64));
        assert_eq!(tile(1024, 1024, 1024, false), Some(T64));
        // f32 A over 4 MiB on a small L2: 128x64.
        assert_eq!(tile(512, 2048, 5632, false), Some(T128x64));
        assert_eq!(tile(256, 32000, 2048, false), Some(T128x64));
        // The old 4096³ 128x128 veto: 128x64 ties it (34.38 vs 34.38).
        assert_eq!(tile(4096, 4096, 4096, false), Some(T128x64));
    }

    #[test]
    fn coopmat_tile_sm_count_alone_moves_the_boundary() {
        use CoopmatTile::*;
        // Same L2 class, only the SM count differs: 128x64 on 768x2048
        // is 8 full waves on 24 SMs but 1.04 waves on 46.
        for l2_bytes in [Some(1 << 20), Some(24 << 20)] {
            let with = |sm| CoopmatDevice {
                sm_count: Some(sm),
                l2_bytes,
            };
            assert_eq!(
                coopmat_tile(1, 768, 2048, 2048, false, with(24)),
                Some(T128x64)
            );
            assert_eq!(coopmat_tile(1, 768, 2048, 2048, false, with(46)), Some(T64));
            assert_eq!(
                coopmat_tile(1, 384, 2048, 2048, false, with(24)),
                Some(T128x64)
            );
            assert_eq!(coopmat_tile(1, 384, 2048, 2048, false, with(46)), Some(T64));
        }
    }

    #[test]
    fn coopmat_tile_alignment_with_sm_count() {
        use CoopmatTile::*;
        for device in [RTX4060, RTX3070] {
            // Not 64-aligned: no coopmat route at all.
            assert_eq!(coopmat_tile(1, 192, 300, 64, false, device), None);
            assert_eq!(coopmat_tile(1, 500, 2048, 2048, true, device), None);
            // 64- but not 128-aligned M: the 64-wide tile only.
            assert!(matches!(
                coopmat_tile(1, 192, 320, 64, false, device),
                Some(T64 | T64Mfast)
            ));
            assert!(matches!(
                coopmat_tile(1, 576, 2048, 2048, true, device),
                Some(T64 | T64Mfast)
            ));
            // Batched grids scale the tile count.
            assert!(coopmat_tile(8, 512, 512, 512, false, device).is_some());
        }
    }

    #[test]
    fn coopmat_tile_selection_round_trips() {
        use CoopmatTile::*;
        for tile in [T128, T64, T64Mfast, T64x128, T128x64] {
            for a16 in [false, true] {
                let selection = tile.selection(a16);
                assert_eq!(
                    coopmat1_parts(selection),
                    Some((tile, a16, CoopmatAcc::F32))
                );
                assert_eq!(
                    super::super::KERNEL_SPECS[selection.index().unwrap()].a_f16(),
                    a16
                );
            }
        }
    }

    #[test]
    fn acc16_tile_rule_tracks_the_4060_ab() {
        use CoopmatTile::*;
        assert_eq!(coopmat_tile_acc16(1, 512, 2048), Some(T64x128));
        assert_eq!(coopmat_tile_acc16(1, 512, 2560), Some(T64x128));
        assert_eq!(coopmat_tile_acc16(1, 512, 5632), Some(T64x128));
        assert_eq!(coopmat_tile_acc16(1, 2048, 2048), Some(T64x128));
        assert_eq!(coopmat_tile_acc16(1, 4096, 4096), Some(T128));
        assert_eq!(coopmat_tile_acc16(1, 192, 320), Some(T64));
        assert_eq!(coopmat_tile_acc16(1, 256, 320), Some(T128x64));
        assert_eq!(coopmat_tile_acc16(1, 192, 300), None);
    }

    #[test]
    fn coopmat_tile_acc_dispatches_on_the_class() {
        for device in [RTX4060, RTX3070, CoopmatDevice::default()] {
            for (m, n, k) in [(512, 2048, 2048), (512, 5632, 2048), (4096, 4096, 4096)] {
                for a16 in [false, true] {
                    assert_eq!(
                        coopmat_tile_acc(CoopmatAcc::F32, 1, m, n, k, a16, device),
                        coopmat_tile(1, m, n, k, a16, device)
                    );
                    for acc in [CoopmatAcc::F16, CoopmatAcc::F16Promote] {
                        assert_eq!(
                            coopmat_tile_acc(acc, 1, m, n, k, a16, device),
                            coopmat_tile_acc16(1, m, n)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn coopmat1_table_is_a_bijection_over_tile_a16_acc() {
        use CoopmatTile::*;
        let mut seen = std::collections::HashSet::new();
        for tile in [T128, T64, T64Mfast, T64x128, T128x64] {
            for a16 in [false, true] {
                for acc in [CoopmatAcc::F32, CoopmatAcc::F16, CoopmatAcc::F16Promote] {
                    let selection = tile.selection_acc(a16, acc);
                    let spec = &super::super::KERNEL_SPECS[selection.index().unwrap()];
                    assert_eq!(spec.accumulator(), acc, "{}", spec.name);
                    assert_eq!(spec.a_f16(), a16, "{}", spec.name);
                    assert!(spec.name.contains("coopmat"), "{}", spec.name);
                    if tile == T64Mfast && acc != CoopmatAcc::F32 {
                        // No f16-accumulator M-fast twin: the x-fast 64x64.
                        assert_eq!(selection, T64.selection_acc(a16, acc));
                        continue;
                    }
                    assert!(seen.insert(selection), "{selection:?} listed twice");
                    assert_eq!(coopmat1_parts(selection), Some((tile, a16, acc)));
                }
            }
        }
        assert_eq!(seen.len(), COOPMAT1.len());
        assert_eq!(coopmat1_parts(KernelSelection::LargeBdaV4), None);
    }

    #[test]
    fn packed_tile_n_matches_packed_selection() {
        use super::KernelSelection::*;
        let check = |k, n, want, tile| {
            assert_eq!(f16w_row_packed_selection(k, n), want, "k={k} n={n}");
            assert_eq!(f16w_row_tile_n(k, n), tile, "tile k={k} n={n}");
        };
        check(2048, 256, F16wRowBdaK16Packed, 32);
        check(2048, 2048, F16wRowBdaK16Packed, 32);
        check(2048, 5632, F16wRowBdaK16V2Packed, 64);
        check(2048, 32000, F16wRowBdaK16V2Packed, 64);
        // Wide-N but not a 64-multiple: fall back to k16 / tile 32.
        check(2048, 5200, F16wRowBdaK16Packed, 32);
    }
}
