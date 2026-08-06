//! Pure host-side Stream-K scheduling and routing policy.

use anyhow::{Result, bail};

/// Hybrid Stream-K schedule precomputed on host.  Stored as `u32` so
/// it goes straight into the push constant block.
///
/// The executor issues two dispatches sharing this schedule:
///
///   * DP-flat: 2D `(n_tiles, m_tiles)` covering tiles
///     `[0, dp_tiles_total)`.  Each WG plain-stores its tile.
///   * SK-tail: 1D `g_sk` persistent workgroups splitting the tail
///     iter-space `total_iters_sk = (T - dp_tiles_total) *
///     iters_per_tile` evenly.  Single-owner tiles plain-store;
///     seam tiles atomic-add.
///
/// `grid_total = dp_tiles_total + g_sk` is retained only as a
/// device-limit sanity check (no single dispatch ever reaches it).
///
/// When `T <= preferred_g`, `dp_tiles_total = 0` and the schedule
/// degenerates to pure Stream-K — the DP dispatch is skipped.  When
/// `T % preferred_g == 0`, `g_sk = 0` and the schedule degenerates
/// to pure DP — the SK-tail dispatch is skipped.
#[derive(Copy, Clone, Debug)]
pub struct StreamKSchedule {
    pub iters_per_tile: u32,
    pub iters_per_wg_sk: u32,
    pub rem_sk: u32,
    pub n_tiles: u32,
    pub dp_tiles_total: u32,
    pub g_sk: u32,
    pub total_iters_sk: u32,
    /// `dp_tiles_total + g_sk`, retained as a device-limit sanity
    /// check against `maxComputeWorkGroupCount[0]`.  No single
    /// dispatch reaches this size — the two halves dispatch
    /// independently.
    pub grid_total: u32,
}

impl StreamKSchedule {
    /// Compute the hybrid schedule for a single-batch GEMM of dims
    /// (m, n, k) against tile dims (bm, bn, bk).  Caller must ensure
    /// shape is aligned to (BM, BN, BK).  `sm_count` drives the
    /// preferred persistent-grid size `G = sm_count * 2`.
    pub fn for_shape(m: u32, n: u32, k: u32, bm: u32, bn: u32, bk: u32, sm_count: u32) -> Self {
        Self::try_for_shape(m, n, k, bm, bn, bk, sm_count)
            .expect("invalid or overflowing Stream-K shape")
    }

    /// Fallible schedule construction for untrusted dimensions.
    pub fn try_for_shape(
        m: u32,
        n: u32,
        k: u32,
        bm: u32,
        bn: u32,
        bk: u32,
        sm_count: u32,
    ) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 || bm == 0 || bn == 0 || bk == 0 {
            bail!("Stream-K dimensions and tile sizes must be non-zero");
        }
        if !m.is_multiple_of(bm) || !n.is_multiple_of(bn) || !k.is_multiple_of(bk) {
            bail!("Stream-K shape must be aligned to its tile sizes");
        }
        let m_tiles = m / bm;
        let n_tiles = n / bn;
        let iters_per_tile = (k / bk).max(1);
        let total_tiles = m_tiles
            .checked_mul(n_tiles)
            .ok_or_else(|| anyhow::anyhow!("Stream-K tile count overflows u32"))?;
        let preferred_g = sm_count.max(1).saturating_mul(2);

        let (dp_tiles_total, tail_tiles) = if total_tiles < preferred_g {
            // Pure Stream-K: the shape doesn't have enough work for
            // the persistent grid to be a clean tiling.  Run all tiles
            // through the SK path.
            (0u32, total_tiles)
        } else {
            // Hybrid: clean waves go through DP, tail through SK.
            // When T % preferred_g == 0 this degenerates to pure DP
            // (tail_tiles = 0, g_sk = 0).
            let dp_raw = (total_tiles / preferred_g) * preferred_g;
            // Round dp_tiles_total down to a multiple of n_tiles so the
            // DP dispatch is a clean 2D rectangle.  This lets the
            // dispatcher use the branchless BDA_V4 kernel — the
            // tile_id early-out, even when spec-const-elided, forces
            // glslang to wrap main() in an OpSwitch structural merge
            // that the NVIDIA driver does not fully optimise away.
            // Empirically the wrapper costs ~12% on the aligned hot
            // path.  The cost of pushing 0-31 tiles back into the SK
            // tail is small (each SK tile is ~iters_per_tile FFMAs
            // through the atomic-add path); the DP-side gain dominates.
            let dp = (dp_raw / n_tiles) * n_tiles;
            (dp, total_tiles - dp)
        };

        let total_iters_sk = tail_tiles
            .checked_mul(iters_per_tile)
            .ok_or_else(|| anyhow::anyhow!("Stream-K iteration count overflows u32"))?;
        let g_sk = if total_iters_sk == 0 {
            0
        } else {
            // Use the full preferred persistent grid on the tail, even
            // if many workgroups end up sharing each tile.  The shader's
            // atomicAdd epilogue handles arbitrary fan-in correctly;
            // the two-tile clamp from v1 was overly conservative and
            // collapsed g_sk down to `tail_tiles` for square shapes,
            // erasing the wave-quantization savings.  Only floor at 1
            // to avoid a zero-division in the iters_per_wg_sk
            // computation.
            preferred_g.min(total_iters_sk.max(1)).max(1)
        };

        let (iters_per_wg_sk, rem_sk) = match total_iters_sk.checked_div(g_sk) {
            Some(i) => (i, total_iters_sk - i * g_sk),
            None => (0, 0),
        };

        let grid_total = dp_tiles_total
            .checked_add(g_sk)
            .ok_or_else(|| anyhow::anyhow!("Stream-K grid size overflows u32"))?;

        Ok(Self {
            iters_per_tile,
            iters_per_wg_sk,
            rem_sk,
            n_tiles,
            dp_tiles_total,
            g_sk,
            total_iters_sk,
            grid_total,
        })
    }

    /// Returns true when the schedule has *no* Stream-K tail
    /// component.  In that case the dispatch is pure DP and callers
    /// can short-circuit to the regular `run_matmuls` path (or just
    /// dispatch this kernel; it's equivalent to DP for `g_sk == 0`).
    pub fn is_pure_dp(&self) -> bool {
        self.g_sk == 0
    }

    /// Returns true when the schedule has *no* DP-mode tiles — every
    /// workgroup is in the SK tail.  Used for the "tiny shape"
    /// regression where T <= preferred_g.
    pub fn is_pure_sk(&self) -> bool {
        self.dp_tiles_total == 0
    }
}

/// Heuristic gate: should the auto-selector route this shape to
/// Stream-K instead of the regular DP kernel?
///
/// Stream-K only beats DP when wave-quantization tail is the
/// dominant perf loss.  Outside that bucket the persistent-grid
/// bookkeeping and atomic-store costs swamp the savings.  Fire SK
/// only when:
///
///   1. The shape is aligned to (128, 128, 32) (the only tile we
///      ship today).
///   2. `total_tiles >= preferred_g`, so DP can fill at least one
///      full wave cleanly and the SK part only sweeps the tail.
///   3. `tail = total_tiles % preferred_g` is non-zero (otherwise
///      DP has no wave-quantization at all).
///   4. The tail fraction `tail / preferred_g` is below
///      `tail_fraction_max`, so the wave-quantization tax is small
///      enough that SK's persistent-grid cost has any chance of
///      being amortized.
///
/// The 0.05 default for `tail_fraction_max` is intentionally tight:
/// the current hybrid kernel pays ~7% overhead per dispatch versus
/// the regular aligned kernel, so SK only wins when the wave-quant
/// tax decisively exceeds that overhead.  Raise this threshold once
/// the SK overhead gap is closed.
pub fn stream_k_should_fire(m: u32, n: u32, k: u32, sm_count: u32, tail_fraction_max: f64) -> bool {
    const BM: u32 = 128;
    const BN: u32 = 128;
    const BK: u32 = 32;
    if !m.is_multiple_of(BM) || !n.is_multiple_of(BN) || !k.is_multiple_of(BK) {
        return false;
    }
    let m_tiles = m / BM;
    let n_tiles = n / BN;
    let total_tiles = m_tiles.checked_mul(n_tiles).unwrap_or(0);
    let preferred_g = sm_count.max(1).saturating_mul(2);
    if total_tiles < preferred_g {
        return false;
    }
    let tail = total_tiles % preferred_g;
    if tail == 0 {
        return false;
    }
    (tail as f64) / (preferred_g as f64) <= tail_fraction_max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_schedule_for_aligned_4096_cube() {
        // 4096^3 / 128x128/BK=32: m_tiles=32, n_tiles=32, T=1024.
        // G_pref=92.  Raw dp = 11*92 = 1012, then rounded down to a
        // multiple of n_tiles=32: 992 = 31 full rows.  Tail = 32 tiles.
        // The round-down keeps the DP dispatch a clean 2D rectangle so
        // the dispatcher can use the branchless BDA_V4 kernel — see
        // `for_shape` for the motivation.
        // iters_per_tile=128; total_iters_sk = 32*128 = 4096.
        let s = StreamKSchedule::for_shape(4096, 4096, 4096, 128, 128, 32, 46);
        assert_eq!(s.iters_per_tile, 128);
        assert_eq!(s.n_tiles, 32);
        assert_eq!(s.dp_tiles_total, 992);
        assert_eq!(s.g_sk, 92);
        assert_eq!(s.total_iters_sk, 32 * 128);
        let total_iters = 32 * 128;
        assert_eq!(s.iters_per_wg_sk, total_iters / 92);
        assert_eq!(s.rem_sk, total_iters - (total_iters / 92) * 92);
        assert_eq!(s.grid_total, 992 + 92);
        // The two-tile clamp from v1 was removed: g_sk is sized to the
        // full preferred grid, so a tail tile may be touched by many
        // more than 2 WGs.  The shader's atomicAdd path handles this
        // correctly.
        assert!(!s.is_pure_dp());
        assert!(!s.is_pure_sk());
        // The DP grid is always a clean rectangle.
        assert_eq!(s.dp_tiles_total % s.n_tiles, 0);
    }

    #[test]
    fn schedule_degenerates_to_pure_sk_for_small_shape() {
        // 512^3 / 128x128/BK=32: m_tiles=4, n_tiles=4, T=16,
        // iters_per_tile=16, total_iters=256.  T < G_pref=92 so all
        // tiles go through SK; g_sk is capped at total_iters_sk and
        // sized to spread work across the full preferred grid.
        let s = StreamKSchedule::for_shape(512, 512, 512, 128, 128, 32, 46);
        assert_eq!(s.dp_tiles_total, 0);
        assert_eq!(s.total_iters_sk, 256);
        // 92 < 256, so the preferred grid is honored.
        assert_eq!(s.g_sk, 92);
        assert!(s.is_pure_sk());
        assert!(!s.is_pure_dp());
    }

    #[test]
    fn schedule_rem_distributes_correctly_on_sk_tail() {
        let s = StreamKSchedule::for_shape(4096, 4096, 4096, 128, 128, 32, 46);
        assert_eq!(s.iters_per_wg_sk * s.g_sk + s.rem_sk, s.total_iters_sk);
        assert!(s.rem_sk < s.g_sk);
    }

    #[test]
    fn schedule_degenerates_to_pure_dp_when_tiles_clean_multiple_of_g() {
        // Construct a shape where T is exactly a multiple of G_pref=92.
        // T = 92 needs m_tiles*n_tiles = 92.  92 = 4*23, so use
        // M=128*4=512, N=128*23=2944, K=128.  iters_per_tile = 4.
        let s = StreamKSchedule::for_shape(512, 2944, 128, 128, 128, 32, 46);
        assert_eq!(s.dp_tiles_total, 92);
        assert_eq!(s.g_sk, 0);
        assert_eq!(s.total_iters_sk, 0);
        assert_eq!(s.grid_total, 92);
        assert!(s.is_pure_dp());
    }

    #[test]
    fn gate_blocks_unaligned_shapes() {
        // Any unaligned dim should disqualify SK.
        assert!(!stream_k_should_fire(127, 128, 32, 46, 0.5));
        assert!(!stream_k_should_fire(128, 127, 32, 46, 0.5));
        assert!(!stream_k_should_fire(128, 128, 31, 46, 0.5));
    }

    #[test]
    fn gate_blocks_small_shapes_with_no_full_wave() {
        // T < preferred_g => can't fill a full wave => skip SK.
        // 384x384x128: m_tiles=3, n_tiles=3, T=9.  G_pref=92.
        assert!(!stream_k_should_fire(384, 384, 128, 46, 0.5));
    }

    #[test]
    fn gate_blocks_clean_multiple_shapes_no_tail() {
        // T % G_pref == 0 => no wave-quant tail => skip SK.
        assert!(!stream_k_should_fire(512, 2944, 128, 46, 0.5));
    }

    #[test]
    fn gate_blocks_when_tail_fraction_too_large() {
        // sq_2048: T=256, G_pref=92.  tail = 72.  tail/G = 0.78 > 0.5.
        // Skip SK because the persistent-grid overhead dominates.
        assert!(!stream_k_should_fire(2048, 2048, 2048, 46, 0.5));
    }

    #[test]
    fn gate_fires_on_severe_wave_quant() {
        // Construct a shape with tail = 1 tile only: T = G+1 = 93.
        // M=128*1, N=128*93=11904 -> not square but valid.  Tail
        // fraction = 1/92 = 0.011, well below the default 0.05.
        assert!(stream_k_should_fire(128, 11904, 128, 46, 0.05));
    }

    #[test]
    fn fallible_schedule_rejects_invalid_or_overflowing_shapes() {
        assert!(StreamKSchedule::try_for_shape(128, 128, 32, 0, 128, 32, 46).is_err());
        assert!(StreamKSchedule::try_for_shape(129, 128, 32, 128, 128, 32, 46).is_err());
        assert!(StreamKSchedule::try_for_shape(u32::MAX, u32::MAX, 1, 1, 1, 1, 46).is_err());
    }
}
