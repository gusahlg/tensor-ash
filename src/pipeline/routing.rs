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

/// Coopmat1 tile pick for an f16w (or a16) shape that is already
/// known to be 64-aligned in M/N and 32-aligned in K.
///
/// RTX 4060 (24 SM) A/B, GPU median TF/s:
///
/// | shape | 128x128 | 64x64 | 64x128 | 128x64 |
/// | 512x2048x2048 | 18.6 | **22.3** | 20.5 | 21.0 |
/// | 512x2560x2048 | 19.7 | 21.6 | 21.9 | **22.8** |
/// | 512x5632x2048 | 21.2 | 22.7 | 23.1 | **24.1** |
/// | 512x2048x5632 | 18.7 | **22.9** | 21.2 | 21.7 |
/// | 1024³ | 16.8 | **21.4** | 19.3 | 19.8 |
/// | 2048³ | 22.7 | 23.2 | 23.6 | 24.5 |
///
/// 128x128 stays the large-square default (tiles_128 >= 96) because
/// that is the measured 3070 4096³ winner and this machine cannot
/// re-bench it.  Short-M prefill (M<=512) uses 64x64 unless N is
/// strictly wider than 4:1, where 128x64 won.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CoopmatTile {
    T128,
    T64,
    T64x128,
    T128x64,
}

/// SM-count-aware coopmat tile pick.  `sm_count` is the device's
/// shader-core count ([`crate::VulkanContext::sm_count`]); `None`
/// keeps the SM-agnostic rule ([`coopmat_tile_fallback`]).
pub(crate) fn coopmat_tile(
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    sm_count: Option<u32>,
) -> Option<CoopmatTile> {
    let _ = (k, sm_count);
    coopmat_tile_fallback(batch, m, n)
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

/// Every coopmat1 kernel as (selection, tile, a16, accumulator): the
/// single table the tile/acc mappings below read.
const COOPMAT1: &[(KernelSelection, CoopmatTile, bool, CoopmatAcc)] = {
    use CoopmatAcc::{F16, F16Promote, F32};
    use CoopmatTile::*;
    use KernelSelection as K;
    &[
        (K::F16wCoopmat, T128, false, F32),
        (K::F16wA16Coopmat, T128, true, F32),
        (K::F16wCoopmatM64, T64, false, F32),
        (K::F16wA16CoopmatM64, T64, true, F32),
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

    /// The kernel for this tile, A storage, and accumulator class.
    pub(crate) fn selection_acc(self, a16: bool, acc: CoopmatAcc) -> KernelSelection {
        COOPMAT1
            .iter()
            .find(|&&(_, tile, is_a16, class)| tile == self && is_a16 == a16 && class == acc)
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
    sm_count: Option<u32>,
    a16: bool,
) -> Option<KernelSelection> {
    Some(coopmat_tile(batch, m, n, k, sm_count)?.selection(a16))
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
    fn coopmat_tile_tracks_4060_prefill_ab() {
        use CoopmatTile::*;
        // q/o 512x2048: 4:1 is not strictly wider than 4:1 → 64x64.
        let tile = coopmat_tile_fallback;
        assert_eq!(tile(1, 512, 2048), Some(T64));
        assert_eq!(tile(1, 128, 2048), Some(T64));
        assert_eq!(tile(1, 256, 384), Some(T64));
        // Concat QKV 512x2560 and gate/up 512x5632: 128x64 won.
        assert_eq!(tile(1, 512, 2560), Some(T128x64));
        assert_eq!(tile(1, 512, 5632), Some(T128x64));
        // Large squares keep 128x128 (3070 4096³ winner, untested here).
        assert_eq!(tile(1, 2048, 2048), Some(T128));
        assert_eq!(tile(1, 1024, 1024), Some(T64));
        // 64-aligned but not 128-aligned has no 128-tile route.
        assert_eq!(tile(1, 192, 320), Some(T64));
        assert_eq!(tile(1, 192, 300), None);
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
    fn coopmat1_table_is_a_bijection_over_tile_a16_acc() {
        use CoopmatTile::*;
        let mut seen = std::collections::HashSet::new();
        for tile in [T128, T64, T64x128, T128x64] {
            for a16 in [false, true] {
                for acc in [CoopmatAcc::F32, CoopmatAcc::F16, CoopmatAcc::F16Promote] {
                    let selection = tile.selection_acc(a16, acc);
                    assert!(seen.insert(selection), "{selection:?} listed twice");
                    assert_eq!(coopmat1_parts(selection), Some((tile, a16, acc)));
                    let spec = &super::super::KERNEL_SPECS[selection.index().unwrap()];
                    assert_eq!(spec.accumulator(), acc, "{}", spec.name);
                    assert_eq!(spec.a_f16(), a16, "{}", spec.name);
                    assert!(spec.name.contains("coopmat"), "{}", spec.name);
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
