//! Online measured kernel and reduction-strategy selection.

use anyhow::{Result, bail};
use ash::vk;

use crate::matmul::{MatmulOp, ResolvedMatmul};
use crate::pipeline::{TuneEntry, TuneKey};
use crate::tensor::Tensor;

use super::recording::record_one_matmul;
use super::{Executor, MatmulCall, Slot, splitk2};

impl Executor {
    /// Explicitly tune one GEMM shape against scratch tensors (both
    /// operands filled with 1.0).  Useful to pre-warm the persistent
    /// tuning store for shapes an inference workload will hit, without
    /// paying the measurement cost on the first real call.
    pub fn tune_shape(&self, batch: u32, m: u32, n: u32, k: u32) -> Result<()> {
        let key = TuneKey {
            batch,
            m,
            n,
            k,
            b_f16: false,
        };
        if self.pipeline.is_tuned(key) {
            return Ok(());
        }
        if !self.ctx.timestamps_supported {
            bail!("tune_shape: device has no timestamp support");
        }
        let shape = |rows: u32, cols: u32| -> Vec<u32> {
            if batch == 1 {
                vec![rows, cols]
            } else {
                vec![batch, rows, cols]
            }
        };
        let a = Tensor::uninit_device(&self.ctx, &shape(m, k))?;
        let b = Tensor::uninit_device(&self.ctx, &shape(k, n))?;
        let c = Tensor::uninit_device(&self.ctx, &shape(m, n))?;
        // 0x3F800000 == 1.0f32: deterministic, denormal-free inputs.
        self.fill_buffer_bits(&a, 0x3F80_0000)?;
        self.fill_buffer_bits(&b, 0x3F80_0000)?;
        let op = MatmulOp::new(MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        });
        let dims = ResolvedMatmul::from_op(&op)?;
        self.tune_op(&op, &dims)
    }

    fn fill_buffer_bits(&self, tensor: &Tensor, bits: u32) -> Result<()> {
        let mut slot = self.checkout_slot();
        unsafe {
            self.submit_recorded(&mut slot, |dev, cb, _slot| {
                dev.cmd_fill_buffer(cb, tensor.raw_buffer(), 0, vk::WHOLE_SIZE, bits);
                Ok(())
            })
        }
    }

    /// Measure every candidate kernel on `op`'s real shape and record
    /// the winner in the pipeline's tuning store.
    ///
    /// Protocol (VkSplat-style measured selection, simplified to
    /// benchmark-once-and-cache): one clock-warming dispatch, then one
    /// warmup round plus `R` measured rounds over all candidates,
    /// candidates *interleaved* with a well-spaced, alternating order so GPU clock drift
    /// biases no single kernel. Per candidate we keep the median GPU time.
    /// The heuristic's pick stays unless a challenger beats it by >2%
    /// — measured noise should not churn the store.
    pub(super) fn tune_op(&self, op: &MatmulOp<'_>, dims: &ResolvedMatmul) -> Result<()> {
        let max_groups = self
            .ctx
            .device_properties
            .limits
            .max_compute_work_group_count;
        let candidates = self
            .pipeline
            .tune_candidate_indices(dims.m, dims.n, dims.k, dims.b_f16)
            .into_iter()
            .filter(|&idx| {
                let kernel = self.pipeline.kernel_at(idx);
                dims.n.div_ceil(kernel.tile_n) <= max_groups[0]
                    && dims.m.div_ceil(kernel.tile_m) <= max_groups[1]
                    && dims.batch <= max_groups[2]
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            bail!("no tunable kernel fits this shape and device's dispatch limits");
        }
        let heuristic_idx = self
            .pipeline
            .heuristic_kernel_index(dims.batch, dims.m, dims.n, dims.k, dims.b_f16);

        // Fewer rounds for expensive shapes: at ~5 TFLOPS a round of
        // ~20 candidates on a 30 ms problem already costs ~600 ms.
        let est_ms = dims.total_flops as f64 / 5e12 * 1e3;
        let rounds = if est_ms > 8.0 { 2 } else { 4 };

        let mut slot = self.checkout_slot();
        let measure = (|| -> Result<Vec<Vec<u64>>> {
            // Spin the GPU clocks up before anything is timed.
            let warmup_idx = if candidates.contains(&heuristic_idx) {
                heuristic_idx
            } else {
                candidates[0]
            };
            let _ = self.run_forced_once(&mut slot, op, dims, warmup_idx)?;
            let mut samples = vec![Vec::with_capacity(rounds as usize); candidates.len()];
            for round in 0..=rounds {
                for offset in 0..candidates.len() {
                    let ci = candidate_at(candidates.len(), round, offset);
                    let kernel_idx = candidates[ci];
                    match self.run_forced_once(&mut slot, op, dims, kernel_idx) {
                        // Round 0 is warmup (first-touch pipeline
                        // fetch, cache state); discard its timing.
                        Ok(Some(ns)) if round > 0 => {
                            samples[ci].push(ns);
                        }
                        Ok(_) => {}
                        // Candidate can't run this shape (e.g. dispatch
                        // grid limits) — leave it unmeasured.
                        Err(err) => return Err(err),
                    }
                }
            }
            Ok(samples)
        })();
        let samples = measure?;
        // The split-K2 probe checks out its own slot. Release the DP
        // measurement lease first so single-slot executors cannot wait on
        // themselves while tuning deep-K shapes.
        drop(slot);

        let heuristic_ns = candidates
            .iter()
            .position(|&idx| idx == heuristic_idx)
            .and_then(|ci| median(&samples[ci]));
        let challenger = candidates
            .iter()
            .zip(samples.iter())
            .filter_map(|(&idx, samples)| median(samples).map(|ns| (idx, ns)))
            .min_by_key(|&(_, ns)| ns);

        let Some((mut winner_idx, mut winner_ns)) = challenger else {
            bail!("no candidate produced a measurement");
        };
        if let Some(heur_ns) = heuristic_ns
            && (winner_ns as f64) >= (heur_ns as f64) * 0.98
        {
            winner_idx = heuristic_idx;
            winner_ns = heur_ns;
        }

        // Reduction-strategy pass: on deep-K low-tile shapes, probe the
        // two-stage split-K against the DP winner.  Record its split
        // count only when it clears the same 2% margin.
        let splitk2 = self.tune_splitk2(op, dims, winner_ns, rounds)?;
        let splitk2_splits = splitk2.map(|(splits, _)| splits);
        let route_ns = splitk2.map_or(winner_ns, |(_, ns)| ns);

        let key = TuneKey {
            batch: dims.batch,
            m: dims.m,
            n: dims.n,
            k: dims.k,
            b_f16: dims.b_f16,
        };
        self.pipeline.record_tuned(
            key,
            TuneEntry {
                kernel: winner_idx,
                splitk2_splits,
            },
        );
        log::info!(
            "tensor-ash: tuned B={} {}x{}x{}: {}{} ({:.3} ms) — heuristic was {} ({})",
            dims.batch,
            dims.m,
            dims.n,
            dims.k,
            self.pipeline.kernel_at(winner_idx).name,
            splitk2_splits
                .map(|s| format!(" + splitk2={s}"))
                .unwrap_or_default(),
            route_ns as f64 / 1e6,
            self.pipeline.kernel_at(heuristic_idx).name,
            heuristic_ns
                .map(|ns| format!("{:.3} ms", ns as f64 / 1e6))
                .unwrap_or_else(|| "unmeasured".into()),
        );
        Ok(())
    }

    /// Measure the two-stage split-K against the best DP time.
    /// Returns `Some(splits)` when a split count beats `dp_best_ns` by
    /// the tuning margin.
    fn tune_splitk2(
        &self,
        op: &MatmulOp<'_>,
        dims: &ResolvedMatmul,
        dp_best_ns: u64,
        rounds: u32,
    ) -> Result<Option<(u32, u64)>> {
        // Deep-K, few-tiles gate: DP already saturates the device
        // otherwise and the probe would be wasted work.  Split-K2's
        // stage-1 kernels read f32 B only.
        let tiles = dims.m.div_ceil(128) as u64 * dims.n.div_ceil(128) as u64 * dims.batch as u64;
        if dims.b_f16 || dims.k < 1024 || tiles > 48 {
            return Ok(None);
        }
        let max_groups = self
            .ctx
            .device_properties
            .limits
            .max_compute_work_group_count;
        let mut best: Option<(u64, u32)> = None;
        for splits in [4u32, 8, 16, 32, 64] {
            // Each split needs enough K to amortize its tile loads; the
            // grid/addressing/scratch limits are the same checks the
            // automatic route applies.
            if dims.k / splits < 128
                || !splitk2::auto_route_fits(max_groups, dims.batch, dims.m, dims.n, dims.k, splits)
            {
                continue;
            }
            let mut samples = Vec::with_capacity(rounds as usize);
            for round in 0..=rounds {
                let stats = self.run_matmuls_split_k2(op.call, splits)?;
                if round > 0
                    && let Some(ns) = stats.gpu_time_ns
                {
                    samples.push(ns);
                }
            }
            if let Some(ns) = median(&samples)
                && best.is_none_or(|(b, _)| ns < b)
            {
                best = Some((ns, splits));
            }
        }
        Ok(best
            .filter(|&(ns, _)| (ns as f64) < (dp_best_ns as f64) * 0.98)
            .map(|(ns, splits)| (splits, ns)))
    }

    /// Submit a single dispatch with a forced kernel and return its
    /// GPU time.  Tuning-only path: requires a BDA kernel (no
    /// descriptor set is written).
    fn run_forced_once(
        &self,
        slot: &mut Slot,
        op: &MatmulOp<'_>,
        dims: &ResolvedMatmul,
        kernel_idx: usize,
    ) -> Result<Option<u64>> {
        let kernel = self.pipeline.kernel_at(kernel_idx);
        if kernel.uses_descriptors {
            bail!("run_forced_once: descriptor-bound kernel '{}'", kernel.name);
        }
        unsafe {
            self.submit_timed(
                slot,
                "get_query_pool_results (tuning)",
                |_dev, cb, _slot| {
                    let mut bound = vk::Pipeline::null();
                    record_one_matmul(
                        &self.ctx,
                        &self.pipeline,
                        cb,
                        vk::DescriptorSet::null(),
                        op,
                        dims,
                        kernel,
                        &mut bound,
                    )
                },
            )
        }
    }
}

fn median(samples: &[u64]) -> Option<u64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Some(((sorted[mid - 1] as u128 + sorted[mid] as u128) / 2) as u64)
    } else {
        Some(sorted[mid])
    }
}

fn candidate_at(len: usize, round: u32, offset: usize) -> usize {
    if round == 0 {
        return offset;
    }
    let pair = (round - 1) as usize / 2;
    let start = pair * len.div_ceil(2) % len;
    if !round.is_multiple_of(2) {
        (start + offset) % len
    } else {
        (start + len - 1 - offset) % len
    }
}

#[cfg(test)]
mod tests {
    use super::{candidate_at, median};

    #[test]
    fn median_handles_empty_and_unsorted_samples() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[9, 1, 5]), Some(5));
        assert_eq!(median(&[8, 2]), Some(5));
    }

    #[test]
    fn candidate_order_is_balanced_in_measured_pairs() {
        for round in 0..=4 {
            let mut order = (0..19)
                .map(|offset| candidate_at(19, round, offset))
                .collect::<Vec<_>>();
            order.sort_unstable();
            assert_eq!(order, (0..19).collect::<Vec<_>>());
        }
        for rounds in [2, 4] {
            let mut rank_sums = [0; 19];
            for round in 1..=rounds {
                for rank in 0..19 {
                    rank_sums[candidate_at(19, round, rank)] += rank;
                }
            }
            assert!(rank_sums.iter().all(|sum| *sum == rank_sums[0]));
        }
    }
}
