use crate::context::DeviceKind;

use super::types::{TILE_M, TILE_N};

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
    if !m.is_multiple_of(TILE_M) || !n.is_multiple_of(TILE_N) {
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
    let large_tiles = (m / TILE_M) as u64 * (n / TILE_N) as u64;
    large_tiles < min_large_tiles
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
}
