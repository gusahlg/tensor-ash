//! Online measured kernel and reduction-strategy selection.

use anyhow::{Result, bail};
use ash::vk;

use crate::matmul::{MatmulOp, ResolvedMatmul};
use crate::pipeline::{TuneEntry, TuneKey};
use crate::tensor::Tensor;

use super::recording::record_one_matmul;
use super::{Executor, MatmulCall, Slot};

impl Executor {
    /// Explicitly tune one GEMM shape against scratch tensors (both
    /// operands filled with 1.0).  Useful to pre-warm the persistent
    /// tuning store for shapes an inference workload will hit, without
    /// paying the measurement cost on the first real call.
    pub fn tune_shape(&self, batch: u32, m: u32, n: u32, k: u32) -> Result<()> {
        let key = TuneKey { batch, m, n, k };
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
    /// candidates *interleaved* per round so GPU clock drift biases no
    /// single kernel.  Per candidate we keep the minimum GPU time.
    /// The heuristic's pick stays unless a challenger beats it by >2%
    /// — measured noise should not churn the store.
    pub(super) fn tune_op(&self, op: &MatmulOp<'_>, dims: &ResolvedMatmul) -> Result<()> {
        let candidates = self.pipeline.tune_candidate_indices();
        if candidates.is_empty() {
            bail!("no tunable kernels on this device (bufferDeviceAddress required)");
        }
        let heuristic_idx = self
            .pipeline
            .heuristic_kernel_index(dims.batch, dims.m, dims.n, dims.k);

        // Fewer rounds for expensive shapes: at ~5 TFLOPS a round of
        // ~20 candidates on a 30 ms problem already costs ~600 ms.
        let est_ms = dims.total_flops as f64 / 5e12 * 1e3;
        let rounds = if est_ms > 30.0 {
            1
        } else if est_ms > 8.0 {
            2
        } else {
            3
        };

        let mut slot = self.checkout_slot();
        let measure = (|| -> Result<Vec<Option<u64>>> {
            // Spin the GPU clocks up before anything is timed.
            let _ = self.run_forced_once(&mut slot, op, dims, heuristic_idx)?;
            let mut best: Vec<Option<u64>> = vec![None; candidates.len()];
            for round in 0..=rounds {
                for (ci, &kernel_idx) in candidates.iter().enumerate() {
                    match self.run_forced_once(&mut slot, op, dims, kernel_idx) {
                        // Round 0 is warmup (first-touch pipeline
                        // fetch, cache state); discard its timing.
                        Ok(Some(ns)) if round > 0 => {
                            best[ci] = Some(best[ci].map_or(ns, |b: u64| b.min(ns)));
                        }
                        Ok(_) => {}
                        // Candidate can't run this shape (e.g. dispatch
                        // grid limits) — leave it unmeasured.
                        Err(_) => {}
                    }
                }
            }
            Ok(best)
        })();
        let best = measure?;

        let heuristic_ns = candidates
            .iter()
            .position(|&idx| idx == heuristic_idx)
            .and_then(|ci| best[ci]);
        let challenger = candidates
            .iter()
            .zip(best.iter())
            .filter_map(|(&idx, ns)| ns.map(|ns| (idx, ns)))
            .min_by_key(|&(_, ns)| ns);

        let Some((mut winner_idx, winner_ns)) = challenger else {
            bail!("no candidate produced a measurement");
        };
        if let Some(heur_ns) = heuristic_ns
            && (winner_ns as f64) >= (heur_ns as f64) * 0.98
        {
            winner_idx = heuristic_idx;
        }

        // Reduction-strategy pass: on deep-K low-tile shapes, probe the
        // two-stage split-K against the DP winner.  Record its split
        // count only when it clears the same 2% margin.
        let splitk2_splits = self
            .tune_splitk2(op, dims, winner_ns)
            .inspect_err(|err| log::debug!("tensor-ash: split-K2 probe skipped: {err}"))
            .unwrap_or(None);

        let key = TuneKey {
            batch: dims.batch,
            m: dims.m,
            n: dims.n,
            k: dims.k,
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
            winner_ns as f64 / 1e6,
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
    ) -> Result<Option<u32>> {
        // Deep-K, few-tiles gate: DP already saturates the device
        // otherwise and the probe would be wasted work.
        let tiles = dims.m.div_ceil(128) as u64 * dims.n.div_ceil(128) as u64 * dims.batch as u64;
        if dims.k < 1024 || tiles > 48 {
            return Ok(None);
        }
        let mn = dims.m as u64 * dims.n as u64 * dims.batch as u64;
        let mut best: Option<(u64, u32)> = None;
        for splits in [4u32, 8, 16, 32, 64] {
            // Each split needs enough K to amortize its tile loads,
            // and the scratch must stay modest.
            if dims.k / splits < 128 || mn * splits as u64 * 4 > 256 << 20 {
                continue;
            }
            let mut split_best: Option<u64> = None;
            for round in 0..3 {
                let stats = self.run_matmuls_split_k2(op.call, splits)?;
                if round > 0
                    && let Some(ns) = stats.gpu_time_ns
                {
                    split_best = Some(split_best.map_or(ns, |b| b.min(ns)));
                }
            }
            if let Some(ns) = split_best
                && best.is_none_or(|(b, _)| ns < b)
            {
                best = Some((ns, splits));
            }
        }
        Ok(best
            .filter(|&(ns, _)| (ns as f64) < (dp_best_ns as f64) * 0.98)
            .map(|(_, splits)| splits))
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
