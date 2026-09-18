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

pub(crate) fn coopmat_tile(batch: u32, m: u32, n: u32) -> Option<CoopmatTile> {
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

pub(crate) fn coopmat_selection(batch: u32, m: u32, n: u32, a16: bool) -> Option<KernelSelection> {
    Some(match coopmat_tile(batch, m, n)? {
        CoopmatTile::T128 => {
            if a16 {
                KernelSelection::F16wA16Coopmat
            } else {
                KernelSelection::F16wCoopmat
            }
        }
        CoopmatTile::T64 => {
            if a16 {
                KernelSelection::F16wA16CoopmatM64
            } else {
                KernelSelection::F16wCoopmatM64
            }
        }
        CoopmatTile::T64x128 => {
            if a16 {
                KernelSelection::F16wA16CoopmatM64N128
            } else {
                KernelSelection::F16wCoopmatM64N128
            }
        }
        CoopmatTile::T128x64 => {
            if a16 {
                KernelSelection::F16wA16CoopmatM128N64
            } else {
                KernelSelection::F16wCoopmatM128N64
            }
        }
    })
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
        assert_eq!(coopmat_tile(1, 512, 2048), Some(T64));
        assert_eq!(coopmat_tile(1, 128, 2048), Some(T64));
        assert_eq!(coopmat_tile(1, 256, 384), Some(T64));
        // Concat QKV 512x2560 and gate/up 512x5632: 128x64 won.
        assert_eq!(coopmat_tile(1, 512, 2560), Some(T128x64));
        assert_eq!(coopmat_tile(1, 512, 5632), Some(T128x64));
        // Large squares keep 128x128 (3070 4096³ winner, untested here).
        assert_eq!(coopmat_tile(1, 2048, 2048), Some(T128));
        assert_eq!(coopmat_tile(1, 1024, 1024), Some(T64));
        // 64-aligned but not 128-aligned has no 128-tile route.
        assert_eq!(coopmat_tile(1, 192, 320), Some(T64));
        assert_eq!(coopmat_tile(1, 192, 300), None);
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
