//! Atomic and two-stage split-K execution paths.

use anyhow::{Result, anyhow, bail};
use ash::vk;

use crate::buffer::{Buffer, BufferLocation};
use crate::matmul::ResolvedMatmul;

use super::recording::{record_matmul_split_k_commands, update_split_k_descriptor_set};
use super::splitk::{self, SplitKPipeline};
use super::splitk2::{self, SplitK2Pipeline};
use super::{Executor, MatmulCall, RunStats, Slot, default_num_k_splits};

impl Executor {
    /// Experimental: dispatch a single matmul using the split-K
    /// kernel.  Splits the K reduction into `num_k_splits` chunks
    /// (`0` = pick a default based on the shape), zero-fills C, then
    /// dispatches one workgroup per (block_row, block_col, batch,
    /// k_split) and atomically reduces partial sums into C.
    ///
    /// Returns the same `RunStats` shape as `run_matmuls`, including
    /// GPU time over the zero-fill **plus** the dispatch — both
    /// happen inside the timestamp window.
    ///
    /// Caveats:
    ///   * `accumulate=true` is rejected: split-K's correctness
    ///     story relies on starting from a zeroed C.
    ///   * Atomic-add ordering is non-deterministic, so results
    ///     differ across runs in the low-order bit.  Tests should
    ///     budget `tolerance(K) * num_k_splits` for the extra
    ///     rounding.
    ///
    /// This entry point is intentionally a *separate* method from
    /// `run_matmuls` so callers opt in explicitly.
    pub fn run_matmuls_split_k(&self, call: MatmulCall<'_>, num_k_splits: u32) -> Result<RunStats> {
        self.validate_call_context(&call)?;
        if call.accumulate {
            bail!("run_matmuls_split_k: accumulate=true is not supported");
        }
        let resolved = crate::matmul::ResolvedMatmul::from_call(&call)?;
        if num_k_splits > 0xFFFF {
            bail!("run_matmuls_split_k: num_k_splits={num_k_splits} out of range [1, 65535]");
        }
        if num_k_splits > resolved.k {
            bail!(
                "run_matmuls_split_k: num_k_splits={num_k_splits} exceeds K={}",
                resolved.k
            );
        }
        let (tile_m, tile_n) = if resolved.m >= 128 && resolved.n >= 128 {
            (128, 128)
        } else {
            (64, 64)
        };
        let num_k_splits = if num_k_splits == 0 {
            default_num_k_splits(
                resolved.batch,
                resolved.m,
                resolved.n,
                resolved.k,
                tile_m,
                tile_n,
            )
        } else {
            num_k_splits
        };
        if num_k_splits == 1 {
            return self.run_matmuls(&[call]);
        }
        if !self.ctx.buffer_device_address_enabled {
            bail!("run_matmuls_split_k: bufferDeviceAddress not enabled");
        }
        if !self.ctx.shader_buffer_float32_atomic_add_enabled {
            bail!(
                "run_matmuls_split_k: VK_EXT_shader_atomic_float (shaderBufferFloat32AtomicAdd) \
                 not enabled — kernel relies on the hardware atomicAdd path"
            );
        }
        // Stable equivalent of `get_or_try_init`: try-init outside, then
        // race the get_or_init.  Worst case we build the pipeline twice
        // on first concurrent access but only the first init wins.
        let split_k = if let Some(s) = self.split_k.get() {
            s
        } else {
            let built = SplitKPipeline::new(&self.ctx, self.pipeline.pipeline_layout)?;
            self.split_k.get_or_init(|| built)
        };
        let kernel = split_k.pick(resolved.m, resolved.n);

        let mut slot = self.checkout_slot();
        let result = self.record_and_run_split_k(&mut slot, &call, &resolved, kernel, num_k_splits);

        let gpu_time_ns = result?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: 1,
            total_flops: resolved.total_flops,
        })
    }

    fn record_and_run_split_k(
        &self,
        slot: &mut Slot,
        call: &MatmulCall<'_>,
        resolved: &crate::matmul::ResolvedMatmul,
        kernel: &splitk::SplitKKernel,
        num_k_splits: u32,
    ) -> Result<Option<u64>> {
        let set_count = 1usize;
        let max_groups = self
            .ctx
            .device_properties
            .limits
            .max_compute_work_group_count;
        let gx = resolved.n.div_ceil(kernel.tile_n);
        let gy = resolved.m.div_ceil(kernel.tile_m);
        let gz = resolved
            .batch
            .checked_mul(num_k_splits)
            .ok_or_else(|| anyhow!("split-K dispatch grid Z overflows u32"))?;
        if gx > max_groups[0] || gy > max_groups[1] || gz > max_groups[2] {
            bail!(
                "split-K dispatch ({gx}, {gy}, {gz}) for M={}, N={}, K={}, batch={}, splits={} \
                 exceeds device maxComputeWorkGroupCount ({}, {}, {})",
                resolved.m,
                resolved.n,
                resolved.k,
                resolved.batch,
                num_k_splits,
                max_groups[0],
                max_groups[1],
                max_groups[2],
            );
        }

        unsafe {
            // Split-K still binds a descriptor set on the matmul
            // pipeline layout that SplitKKernel was built against, even
            // though the shader addresses A/B/C through BDA pointers.
            // Use the dedicated single-set helper so we don't pull in
            // the matmul path's per-call BDA-vs-descriptor reasoning.
            debug_assert_eq!(set_count, 1);
            update_split_k_descriptor_set(&self.ctx, slot.descriptor_sets[0], call);

            self.submit_timed(
                slot,
                "get_query_pool_results (split-K)",
                |_dev, cb, slot| {
                    let descriptor_sets = &slot.descriptor_sets[..set_count];
                    record_matmul_split_k_commands(
                        &self.ctx,
                        &self.pipeline,
                        kernel,
                        cb,
                        descriptor_sets,
                        call,
                        resolved,
                        num_k_splits,
                        gx,
                        gy,
                        gz,
                    )
                },
            )
        }
    }

    /// Two-stage split-K: stage 1 writes per-split partial planes to
    /// slot-local scratch (plain stores, no atomics), stage 2 reduces
    /// them into C.  Deterministic (bit-stable across runs, unlike the
    /// atomic [`run_matmuls_split_k`]) and needs no
    /// `VK_EXT_shader_atomic_float`.
    ///
    /// `num_k_splits == 0` picks a default from the shape; a resolved
    /// split count of 1 falls back to the regular `run_matmuls` path.
    /// `accumulate=true` is rejected (the reducer overwrites C).
    ///
    /// [`run_matmuls_split_k`]: Self::run_matmuls_split_k
    pub fn run_matmuls_split_k2(
        &self,
        call: MatmulCall<'_>,
        num_k_splits: u32,
    ) -> Result<RunStats> {
        self.validate_call_context(&call)?;
        if call.accumulate {
            bail!("run_matmuls_split_k2: accumulate=true is not supported");
        }
        let resolved = ResolvedMatmul::from_call(&call)?;
        if num_k_splits > 0xFFFF {
            bail!("run_matmuls_split_k2: num_k_splits={num_k_splits} out of range [1, 65535]");
        }
        if num_k_splits > resolved.k {
            bail!(
                "run_matmuls_split_k2: num_k_splits={num_k_splits} exceeds K={}",
                resolved.k
            );
        }
        let (_, [tile_m, tile_n, _]) = splitk2::stage1_dispatch_info(resolved.m, resolved.n);
        let num_k_splits = if num_k_splits == 0 {
            default_num_k_splits(
                resolved.batch,
                resolved.m,
                resolved.n,
                resolved.k,
                tile_m,
                tile_n,
            )
        } else {
            num_k_splits
        };
        if num_k_splits <= 1 {
            return self.run_matmuls(std::slice::from_ref(&call));
        }
        if !self.ctx.buffer_device_address_enabled {
            bail!("run_matmuls_split_k2: bufferDeviceAddress not enabled");
        }

        self.split_k2_pipeline()?;
        let mut slot = self.checkout_slot();
        let result = self.record_and_run_split_k2(&mut slot, &call, &resolved, num_k_splits);

        let gpu_time_ns = result?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: 1,
            total_flops: resolved.total_flops,
        })
    }

    /// Lazily build (or fetch) the split-K2 pipeline.
    pub(super) fn split_k2_pipeline(&self) -> Result<&SplitK2Pipeline> {
        if let Some(s) = self.split_k2.get() {
            return Ok(s);
        }
        let built = SplitK2Pipeline::new(&self.ctx)?;
        Ok(self.split_k2.get_or_init(|| built))
    }

    /// Grow the slot-local split-K2 scratch to at least `bytes` and
    /// return its device address.
    pub(super) fn ensure_splitk2_scratch(&self, slot: &mut Slot, bytes: u64) -> Result<u64> {
        let needs_new = slot
            .splitk2_scratch
            .as_ref()
            .is_none_or(|buffer| buffer.size < bytes);
        if needs_new {
            slot.splitk2_scratch = Some(Buffer::new(
                &self.ctx,
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                BufferLocation::Device,
            )?);
        }
        let scratch = slot
            .splitk2_scratch
            .as_ref()
            .expect("split-K2 scratch initialized above");
        Ok(self.ctx.buffer_device_address(scratch.raw))
    }

    fn record_and_run_split_k2(
        &self,
        slot: &mut Slot,
        call: &MatmulCall<'_>,
        resolved: &ResolvedMatmul,
        num_k_splits: u32,
    ) -> Result<Option<u64>> {
        let split_k2 = self
            .split_k2
            .get()
            .expect("split_k2 pipeline built by caller");
        let plan = splitk2::SplitK2Dispatch::plan(&self.ctx, split_k2, resolved, num_k_splits)?;

        let scratch_ptr = self.ensure_splitk2_scratch(slot, plan.scratch_bytes)?;

        let a_ptr = call.a.device_address();
        let b_ptr = call.b.device_address();
        let c_ptr = call.c.device_address();

        unsafe {
            self.submit_timed(
                slot,
                "get_query_pool_results (split-K2)",
                |_dev, cb, _slot| {
                    splitk2::record_split_k2_commands(
                        &self.ctx,
                        split_k2,
                        cb,
                        call.alpha,
                        resolved,
                        &plan,
                        a_ptr,
                        b_ptr,
                        c_ptr,
                        scratch_ptr,
                    );
                    Ok(())
                },
            )
        }
    }
}
