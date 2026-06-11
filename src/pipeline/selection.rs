use crate::context::DeviceKind;

use super::types::{KERNEL_SPECS, KernelSelection};

/// Output-tile dimensions of the large kernel.  The auto-selector
/// thresholds are all expressed in large-kernel tiles, so we read them
/// out of the kernel registry rather than duplicating the constants.
const LARGE_TILE_M: u32 = KERNEL_SPECS[KernelSelection::Large.index().unwrap()].tile_m;
const LARGE_TILE_N: u32 = KERNEL_SPECS[KernelSelection::Large.index().unwrap()].tile_n;

/// Threshold (in 128x128 tiles) below which the auto selector prefers
/// the small kernel.  Tuned per device kind so we don't underuse a beefy
/// discrete GPU or oversubscribe a tiny integrated one.
pub(super) fn auto_min_large_tiles_for(kind: DeviceKind) -> u64 {
    use DeviceKind::*;
    match kind {
        // Measured on RTX 3070 (46 SMs).  Mid-range to high-end discrete
        // GPUs all sit in the 30-130 SM range, and 256 large tiles
        // (=~ 5-8 tiles/SM) is enough headroom to saturate memory
        // latency hiding across that whole band.
        DiscreteGpu => 256,
        // Integrated GPUs typically have a handful of compute units.
        // Use a much lower threshold so we don't fall off to the small
        // kernel on shapes that the iGPU could feed with a single full
        // large-tile wave.
        IntegratedGpu => 64,
        // Software Vulkan (llvmpipe etc.) and anything else: be cautious
        // and prefer the small kernel.
        VirtualGpu | Cpu | Other => u64::MAX,
    }
}

pub(super) fn auto_selects_small_kernel(m: u32, n: u32, k: u32, min_large_tiles: u64) -> bool {
    // Off-tile M/N shapes waste more work in the large 128x128 kernel
    // and pay its edge-path branches. Prefer the smaller tile for these
    // edge-heavy cases, while still allowing manual large-kernel runs.
    if !m.is_multiple_of(LARGE_TILE_M) || !n.is_multiple_of(LARGE_TILE_N) {
        return true;
    }
    // For tiny K the fixed per-workgroup load + barrier cost dominates,
    // so the smaller-tile kernel (4x more workgroups, 4x more chances to
    // overlap latency) wins.
    if k < 128 {
        return true;
    }
    // The large 128-tile kernel needs ~2-6 workgroups per SM to hide
    // memory latency.  Below the per-device threshold the small kernel
    // (4x more workgroups for the same output) wins on occupancy
    // despite its lower arithmetic intensity.
    //
    // On RTX 3070 (46 SMs): at 1024^3 the small kernel is faster
    // (7.8 vs 7.0 TFLOPS), at 2048^3 large wins decisively (9.8 vs 8.6).
    let large_tiles = (m / LARGE_TILE_M) as u64 * (n / LARGE_TILE_N) as u64;
    large_tiles < min_large_tiles
}

pub(super) fn auto_select_kernel(
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    min_large_tiles: u64,
) -> KernelSelection {
    let min_mn = m.min(n);
    let max_mn = m.max(n);
    let near_square = max_mn <= min_mn.saturating_mul(5) / 4;

    // The 64x32 tile reduces N-side edge waste and accumulator pressure
    // enough to beat the 64x64/k64 paths on small near-square GEMMs.  The
    // 512^3 case moves to the 128x64xK64 tile below, so keep this rule to
    // the smaller cases where it clearly wins.
    //
    // batch>=8 cases are excluded because the extra workgroups make the
    // 64x32 tile's lower arithmetic intensity per-WG dominate the edge-
    // waste savings; the 64x64 small kernel wins on batched_8x256 etc.
    if k >= 128
        && min_mn >= 128
        && max_mn <= 320
        && near_square
        && (batch == 1 || (max_mn <= 256 && batch < 8))
    {
        return KernelSelection::M64N32;
    }

    // The k64 (64x64 BK=64) kernel beats the k32 small variant on mid-
    // square shapes where K reuse can amortize the wider strip.  Found
    // empirically on medium_384 (384^3) - it's a ~15% win.
    if batch == 1 && (192..=480).contains(&min_mn) && near_square && k >= 256 {
        return KernelSelection::K64;
    }

    if batch == 1 {
        // The m128n64k64 tile is the runaway winner for medium and large
        // moderately-rectangular (up to ~5x aspect) GEMMs as long as M
        // and N are big enough to fill a few large tiles.  Bounds tuned
        // empirically:
        //   - lower bound 512 (m128n64k64 wins 768^3, non_pow2
        //     1023x1025x1027, attn_proj 2048x512x512);
        //   - upper bound max_mn / min_mn <= ~5 covers attention-style
        //     projection shapes; skinny shapes with min_mn <= 128 still
        //     go to k64 in the explicit rule below;
        //   - for min_mn>=2048 the 128x128 Large kernel takes over
        //     since its higher arithmetic intensity wins once both
        //     dimensions are big enough to saturate the SMs even at
        //     low occupancy.
        if (512..2048).contains(&min_mn) && max_mn <= min_mn.saturating_mul(5) && k >= 256 {
            return KernelSelection::M128N64K64;
        }
        if m == n && (m == 512 || m == 1024) && k >= 512 {
            return KernelSelection::M128N64K64;
        }
        if min_mn >= 2048 && (128..=1024).contains(&k) {
            return KernelSelection::M128N64K64;
        }
        // Skinny tall/wide shapes where one dimension is far bigger
        // than the other.  For near-square big shapes (e.g. 4096^3)
        // the Large 128x128 BDA_V4 tile wins by ~4-5pp peak because
        // its higher arithmetic intensity beats m128n64k64's
        // narrower N tile.  Limiting this rule to min_mn < 2048 keeps
        // the win for 1024x8192-style shapes while letting near-
        // square >= 2048 dispatches fall through to Large.
        if min_mn >= 1024 && min_mn < 2048 && max_mn >= 4096 && k >= 512 {
            return KernelSelection::M128N64K64;
        }
        if k <= 128 && m >= 2048 && n >= 2048 {
            return KernelSelection::M128N64;
        }
        if min_mn <= 128 && max_mn >= 512 && k >= 256 {
            return KernelSelection::K64;
        }
        if min_mn <= 256 && max_mn == min_mn.saturating_mul(2) && k >= 256 {
            return KernelSelection::K64;
        }
    }
    if auto_selects_small_kernel(m, n, k, min_large_tiles) {
        KernelSelection::Small
    } else {
        KernelSelection::Large
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_selector_prefers_small_for_edge_or_small_shapes() {
        let t = 256u64;
        assert!(auto_selects_small_kernel(1023, 2048, 1024, t));
        assert!(auto_selects_small_kernel(2048, 1025, 1024, t));
        assert!(auto_selects_small_kernel(2048, 2048, 64, t));
        assert!(auto_selects_small_kernel(512, 512, 512, t));
        assert!(auto_selects_small_kernel(1024, 1024, 1024, t));
        assert!(!auto_selects_small_kernel(2048, 2048, 128, t));
        assert!(!auto_selects_small_kernel(2048, 2048, 2048, t));
        assert!(!auto_selects_small_kernel(4096, 1024, 1024, t));
        assert!(!auto_selects_small_kernel(1024, 4096, 1024, t));
    }

    #[test]
    fn auto_threshold_tracks_device_kind() {
        use DeviceKind::*;
        assert_eq!(auto_min_large_tiles_for(DiscreteGpu), 256);
        assert_eq!(auto_min_large_tiles_for(IntegratedGpu), 64);
        assert_eq!(auto_min_large_tiles_for(Cpu), u64::MAX);
        assert_eq!(auto_min_large_tiles_for(VirtualGpu), u64::MAX);
        assert_eq!(auto_min_large_tiles_for(Other), u64::MAX);

        assert!(auto_selects_small_kernel(1024, 1024, 1024, 256));
        assert!(!auto_selects_small_kernel(1024, 1024, 1024, 64));
    }

    #[test]
    fn auto_selector_uses_specialized_variants_for_target_shapes() {
        assert_eq!(
            auto_select_kernel(1, 256, 256, 256, 256),
            KernelSelection::M64N32
        );
        assert_eq!(
            auto_select_kernel(1, 255, 257, 263, 256),
            KernelSelection::M64N32
        );
        assert_eq!(
            auto_select_kernel(4, 256, 256, 256, 256),
            KernelSelection::M64N32
        );
        assert_eq!(
            auto_select_kernel(1, 512, 256, 256, 256),
            KernelSelection::K64
        );
        assert_eq!(
            auto_select_kernel(1, 512, 512, 512, 256),
            KernelSelection::M128N64K64
        );
        assert_eq!(
            auto_select_kernel(1, 1024, 1024, 64, 256),
            KernelSelection::Small
        );
        assert_eq!(
            auto_select_kernel(1, 1024, 128, 512, 256),
            KernelSelection::K64
        );
        assert_eq!(
            auto_select_kernel(1, 128, 1024, 512, 256),
            KernelSelection::K64
        );
        assert_eq!(
            auto_select_kernel(1, 1024, 1024, 1024, 256),
            KernelSelection::M128N64K64
        );
        assert_eq!(
            auto_select_kernel(1, 2048, 2048, 512, 256),
            KernelSelection::M128N64K64
        );
        assert_eq!(
            auto_select_kernel(2, 512, 512, 512, 256),
            KernelSelection::Small
        );
        assert_eq!(
            auto_select_kernel(1, 2048, 2048, 2048, 256),
            KernelSelection::Large
        );
    }
}
