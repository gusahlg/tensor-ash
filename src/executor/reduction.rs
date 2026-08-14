//! Two-stage split-K execution path.

use anyhow::{Result, bail};
use ash::vk;

use crate::buffer::{Buffer, BufferLocation};
use crate::matmul::ResolvedMatmul;

use super::splitk2::{self, SplitK2Pipeline, default_num_k_splits};
use super::{Executor, MatmulCall, RunStats, Slot};

impl Executor {
    /// Two-stage split-K: stage 1 writes per-split partial planes to
    /// slot-local scratch (plain stores, no atomics), stage 2 reduces
    /// them into C.  Deterministic (bit-stable across runs) and needs
    /// no `VK_EXT_shader_atomic_float`.
    ///
    /// `num_k_splits == 0` picks a default from the shape; a resolved
    /// split count of 1 falls back to the regular `run_matmuls` path.
    /// `accumulate=true` is rejected (the reducer overwrites C).
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
