//! Theoretical occupancy from registers / shared memory / threads,
//! using the per-SM limits NVIDIA publishes for compute capability 8.6
//! (GA10x) and 8.9 (AD10x): CUDA C++ Programming Guide, "Technical
//! Specifications per Compute Capability", and the register model of
//! `cuda_occupancy.h` (registers are allocated per warp in 256-register
//! units inside one of the four SM sub-partitions).
//!
//! | limit                         | sm_86   | sm_89   |
//! |-------------------------------|---------|---------|
//! | resident threads / SM         | 1536    | 1536    |
//! | resident warps / SM           | 48      | 48      |
//! | resident blocks / SM          | 16      | 24      |
//! | 32-bit registers / SM         | 65536   | 65536   |
//! | shared memory / SM (max)      | 100 KiB | 100 KiB |
//! | reserved shared / block       | 1 KiB   | 1 KiB   |
//! | shared allocation granularity | 128 B   | 128 B   |
//!
//! The Vulkan driver chooses the L1/shared carveout itself; this model
//! assumes the maximum (100 KiB) carveout, so it is an upper bound.

#[derive(Clone, Copy, Debug)]
pub(crate) struct SmLimits {
    pub(crate) arch: &'static str,
    pub(crate) max_threads: u32,
    pub(crate) max_warps: u32,
    pub(crate) max_blocks: u32,
    pub(crate) regs_per_sm: u32,
    pub(crate) smem_per_sm: u32,
    pub(crate) smem_reserved_per_block: u32,
}

impl SmLimits {
    pub(crate) fn for_arch(arch: &str) -> Option<Self> {
        let base = SmLimits {
            arch: "SM86",
            max_threads: 1536,
            max_warps: 48,
            max_blocks: 16,
            regs_per_sm: 65536,
            smem_per_sm: 100 * 1024,
            smem_reserved_per_block: 1024,
        };
        match arch.to_ascii_uppercase().as_str() {
            "SM86" | "SM87" => Some(base),
            "SM89" => Some(SmLimits {
                arch: "SM89",
                max_blocks: 24,
                ..base
            }),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Occupancy {
    pub(crate) blocks_per_sm: u32,
    pub(crate) warps_per_sm: u32,
    /// Which resource bound the block count ("regs", "smem", "threads",
    /// "blocks").
    pub(crate) limiter: &'static str,
    pub(crate) by_regs: u32,
    pub(crate) by_smem: u32,
    pub(crate) by_threads: u32,
}

impl Occupancy {
    pub(crate) fn percent(&self, limits: &SmLimits) -> f64 {
        100.0 * self.warps_per_sm as f64 / limits.max_warps as f64
    }
}

pub(crate) fn occupancy(
    limits: &SmLimits,
    threads_per_block: u32,
    regs_per_thread: u32,
    smem_per_block: u32,
) -> Occupancy {
    let warps_per_block = threads_per_block.div_ceil(32).max(1);
    // Registers: per-warp allocation rounded to 256, packed into each of
    // the 4 sub-partitions independently.
    let regs_per_warp = (regs_per_thread.max(1) * 32).div_ceil(256) * 256;
    let warps_per_partition = (limits.regs_per_sm / 4) / regs_per_warp;
    let by_regs = (warps_per_partition * 4) / warps_per_block;
    let by_smem = if smem_per_block == 0 {
        u32::MAX
    } else {
        let per_block = smem_per_block.div_ceil(128) * 128 + limits.smem_reserved_per_block;
        limits.smem_per_sm / per_block
    };
    let by_threads =
        (limits.max_warps / warps_per_block).min(limits.max_threads / threads_per_block);
    let candidates = [
        (by_regs, "regs"),
        (by_smem, "smem"),
        (by_threads, "threads"),
        (limits.max_blocks, "blocks"),
    ];
    let (blocks, limiter) = candidates
        .iter()
        .copied()
        .min_by_key(|(n, _)| *n)
        .expect("non-empty");
    Occupancy {
        blocks_per_sm: blocks,
        warps_per_sm: blocks * warps_per_block,
        limiter,
        by_regs,
        by_smem,
        by_threads,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_hand_computed_cases() {
        let l86 = SmLimits::for_arch("sm86").unwrap();
        // 208 regs x 128 threads: 6656 regs/warp -> 2 warps/partition ->
        // 8 warps/SM -> 2 blocks of 4 warps.
        let o = occupancy(&l86, 128, 208, 18_944);
        assert_eq!((o.blocks_per_sm, o.warps_per_sm, o.limiter), (2, 8, "regs"));
        // 32 regs x 256 threads: threads bound 6 blocks (48 warps).
        let o = occupancy(&l86, 256, 32, 0);
        assert_eq!((o.blocks_per_sm, o.limiter), (6, "threads"));
        // 49,664 B shared: 2 blocks (2 x 50,688 <= 102,400).
        let o = occupancy(&l86, 128, 64, 49_664);
        assert_eq!((o.blocks_per_sm, o.limiter), (2, "smem"));
        // Tiny blocks hit the block cap: 16 on sm_86, 24 on sm_89.
        let l89 = SmLimits::for_arch("SM89").unwrap();
        assert_eq!(occupancy(&l86, 32, 32, 0).blocks_per_sm, 16);
        assert_eq!(occupancy(&l89, 32, 32, 0).blocks_per_sm, 24);
    }
}
