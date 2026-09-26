//! Shape-based kernel routing: row-GEMV pack tiles, coopmat wave-fill,
//! and BDA / f16w family promotion.

use super::catalog::KernelSelection;

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
/// median TF/s, f16w = f32 A, a16 = f16 A/C):
///
/// | shape | GPU | 128x128 | 64x64 | 64x64 M-fast | 128x64 |
/// |---|---|---:|---:|---:|---:|
/// | a16 512x2560x2048 (QKV) | 3070 | 29.7 | 33.1 | **33.5** | 28.0 |
/// | a16 512x5632x2048 (gate/up) | 3070 | 32.9 | 29.1 | **35.2** | 33.8 |
/// | a16 512x2048x5632 (down) | 3070 | 24.9 | 32.1 | **33.6** | 31.9 |
/// | a16 512x5632x2048 | 4060 | 24.3 | **28.9** | 28.8 | 28.0 |
/// | a16 1920x5632x2048 | 3070 | 34.0 | 26.9 | 30.0 | **35.6** |
/// | a16 1920x5632x2048 | 4060 | 26.4 | **29.5** | 28.9 | 27.0 |
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
/// An L2 this large keeps a TinyLlama-class prefill operand resident
/// (AD107: 24 MiB); GA104's 4 MiB does not.
const LARGE_L2_BYTES: u64 = 16 * MIB;
/// CTAs per SM from the driver register counts (identical on sm_86
/// and sm_89; `examples/coopmat_occupancy.rs`): 64x64 is 72 regs x 4
/// warps -> 7; 128x64 is 128 regs x 4 warps -> 4.
const CTAS_PER_SM_64: u64 = 7;

/// SM-count-aware coopmat tile pick for an f16w (`a16 = false`) or
/// f16-activation (`a16 = true`) shape; `k % 32 == 0` is the caller's
/// precondition.  With no SM count this is exactly the exp/grokmerge
/// rule ([`coopmat_tile_fallback`]).
///
/// With `S` SMs the rule compares the two tiles that won the two-GPU
/// sweep, 128x64 and 64x64:
///
/// 1. **Raster of the 64x64 grid.**  In-flight window `W = 7 S` CTAs.
///    x-fastest order re-streams B `xp = ceil(rows / max(1, W/cols))`
///    times; M-fastest re-streams A `mp = ceil(cols / max(1, W/rows))`
///    times.  M-fast when it moves clearly fewer bytes
///    (`|A| mp + |B| < 0.75 (|B| xp + |A|)`), its A re-reads stay
///    small (`|A| (mp - 1) <= 8 MiB`), and B does not already fit a
///    large L2 (then re-streams hit in L2 and x-fast is as good).
/// 2. **Wave balance.**  Per tile, `n = ceil(tiles / S)` CTAs land on
///    the busiest SM; `score = tiles/(S n) * min(1, n/3) * rho`, i.e.
///    SM load balance times an occupancy term (fewer than 3 CTAs per SM
///    cannot hide the load/MMA latency of either tile).
/// 3. **Tile efficiency rho** of 64x64 relative to 128x64 (fit to the
///    sweep): large L2 — 1.02 a16 (f16 A), 0.92 f16w (f32 A); small or
///    unknown L2 — a16 1.02, but 0.8 when an x-fast grid re-streams B;
///    f16w 0.9, except 1.0 (single-pass x-fast, `|A| <= 4 MiB`) / 0.85
///    (single-pass, larger A).
///
/// Ties go to 64x64 (finer balance, more resident warps).  M or N not
/// 128-/64-aligned: 64x64 only (`None` below 64-alignment).
pub(crate) fn coopmat_tile(
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    a16: bool,
    device: CoopmatDevice,
) -> Option<CoopmatTile> {
    let Some(sms) = device.sm_count.filter(|&count| count > 0) else {
        return coopmat_tile_fallback(batch, m, n);
    };
    if !m.is_multiple_of(64) || !n.is_multiple_of(64) {
        return None;
    }
    let sms = u64::from(sms);
    let (batch, m, n, k) = (u64::from(batch), u64::from(m), u64::from(n), u64::from(k));
    let (rows, cols) = (m / 64, n / 64);
    let window = (sms * CTAS_PER_SM_64) as f64;
    let passes = |blocks: u64, other: u64| -> u64 {
        (blocks as f64 / (window / other as f64).max(1.0)).ceil() as u64
    };
    let a_bytes = m * k * if a16 { 2 } else { 4 };
    let b_bytes = k * n * 2;
    let xp = passes(rows, cols);
    let mp = passes(cols, rows);
    let l2 = device.l2_bytes.unwrap_or(0);
    let large_l2 = l2 >= LARGE_L2_BYTES;
    // An L2 that holds all of B makes x-fast re-streams free.
    let m_fast = !(large_l2 && b_bytes <= l2)
        && a_bytes * (mp - 1) <= 8 * MIB
        && ((a_bytes * mp + b_bytes) as f64) < 0.75 * (b_bytes * xp + a_bytes) as f64;
    let t64 = if m_fast {
        CoopmatTile::T64Mfast
    } else {
        CoopmatTile::T64
    };
    if !m.is_multiple_of(128) {
        return Some(t64);
    }
    let rho = match (large_l2, a16) {
        (true, true) => 1.02,
        (true, false) => 0.92,
        (false, true) if !m_fast && xp > 1 => 0.8,
        (false, true) => 1.02,
        (false, false) if m_fast || xp > 1 => 0.9,
        (false, false) if a_bytes <= 4 * MIB => 1.0,
        (false, false) => 0.85,
    };
    let score = |tiles: u64, rho: f64| -> f64 {
        let per_sm = tiles.div_ceil(sms);
        tiles as f64 / (sms * per_sm) as f64 * (per_sm as f64 / 3.0).min(1.0) * rho
    };
    let s64 = score(batch * rows * cols, rho);
    let s128x64 = score(batch * (m / 128) * cols, 1.0);
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

impl CoopmatTile {
    /// The f16w (f32-A) or a16 (f16-A/C) kernel for this tile.
    pub(crate) fn selection(self, a16: bool) -> KernelSelection {
        use KernelSelection as K;
        match (self, a16) {
            (Self::T128, false) => K::F16wCoopmat,
            (Self::T128, true) => K::F16wA16Coopmat,
            (Self::T64, false) => K::F16wCoopmatM64,
            (Self::T64, true) => K::F16wA16CoopmatM64,
            (Self::T64Mfast, false) => K::F16wCoopmatM64Mfast,
            (Self::T64Mfast, true) => K::F16wA16CoopmatM64Mfast,
            (Self::T64x128, false) => K::F16wCoopmatM64N128,
            (Self::T64x128, true) => K::F16wA16CoopmatM64N128,
            (Self::T128x64, false) => K::F16wCoopmatM128N64,
            (Self::T128x64, true) => K::F16wA16CoopmatM128N64,
        }
    }

    /// The tile of a coopmat1 kernel selection (either A storage).
    pub(crate) fn of_selection(selection: KernelSelection) -> Option<Self> {
        use KernelSelection as K;
        Some(match selection {
            K::F16wCoopmat | K::F16wA16Coopmat => Self::T128,
            K::F16wCoopmatM64 | K::F16wA16CoopmatM64 => Self::T64,
            K::F16wCoopmatM64Mfast | K::F16wA16CoopmatM64Mfast => Self::T64Mfast,
            K::F16wCoopmatM64N128 | K::F16wA16CoopmatM64N128 => Self::T64x128,
            K::F16wCoopmatM128N64 | K::F16wA16CoopmatM128N64 => Self::T128x64,
            _ => return None,
        })
    }
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
        // TinyLlama pp512 (f16 activations): 64x64 won all four GEMMs;
        // B fits the 24 MiB L2, so the raster stays x-fast.
        assert_eq!(tile(512, 2048, 2048, true), Some(T64)); // q/o
        assert_eq!(tile(512, 2560, 2048, true), Some(T64)); // concat QKV
        assert_eq!(tile(512, 5632, 2048, true), Some(T64)); // gate/up
        assert_eq!(tile(512, 2048, 5632, true), Some(T64)); // down
        // Long prompts: 64x64 x-fast keeps winning a16 on the big L2.
        assert_eq!(tile(1920, 5632, 2048, true), Some(T64));
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
        // pp512: QKV 64x64 (+17% over 128x64), gate/up 64x64 M-fast
        // (35.2 vs 29.1 x-fast vs 33.8 128x64), q/o and down 64x64.
        assert_eq!(tile(512, 2048, 2048, true), Some(T64));
        assert_eq!(tile(512, 2560, 2048, true), Some(T64));
        assert_eq!(tile(512, 5632, 2048, true), Some(T64Mfast));
        assert_eq!(tile(512, 2048, 5632, true), Some(T64));
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
                assert_eq!(CoopmatTile::of_selection(selection), Some(tile));
                assert_eq!(
                    super::super::KERNEL_SPECS[selection.index().unwrap()].a_f16(),
                    a16
                );
            }
        }
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
