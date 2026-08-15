//! T3 / T4: trivial-kernel microbenches for barrier drain and
//! submission cost.
//!
//! The "trivial kernel" is `op_binary_f32` over one 256-element
//! workgroup — a single-dispatch 1-add through the normal elementwise
//! pipeline machinery (a dedicated empty kernel could not force
//! barriers: the graph's hazard tracker only emits one when an op
//! WRITES something, so the minimal barrier-forcing kernel IS a 1-add).
//!
//! T3 measures two GPU-timestamped chains of the same trivial dispatch:
//! a serialized chain (every op accumulates into ONE tensor, so the
//! hazard tracker emits a barrier before each op) and an overlapped
//! chain (every op writes its OWN tensor — zero barriers).  The
//! difference of the two chains' slopes over N is the effective cost
//! of one full compute barrier: drain + lost overlap, which is exactly
//! the price the decode graph pays per hazard barrier.
//!
//! T4 compares N single-dispatch submissions against one N-dispatch
//! submission of the same op list (wall clock); the slope difference
//! is the per-`vkQueueSubmit`-plus-fence-wait cost.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, ensure};
use tensor_ash::{BinaryOp, ExecOp, Executor, Tensor, VulkanContext};
use tensor_ash_test_support::fill_det;

use super::expect::Section;
use super::{Row, Verdict, check_within, key};
use crate::bench::cases::SampleSummary;

/// Elements per trivial dispatch: one 256-thread workgroup.
const ELEMS: u32 = 256;
const N_SHORT: usize = 32;
const N_LONG: usize = 256;
const REPS: usize = 9;

struct TrivialOps {
    x: Tensor,
    y: Tensor,
    acc: Tensor,
    outs: Vec<Tensor>,
}

impl TrivialOps {
    fn new(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<Self> {
        let x = Tensor::uninit_device(ctx, &[ELEMS])?;
        let y = Tensor::uninit_device(ctx, &[ELEMS])?;
        let acc = Tensor::uninit_device(ctx, &[ELEMS])?;
        let outs = (0..N_LONG)
            .map(|_| Tensor::uninit_device(ctx, &[ELEMS]))
            .collect::<Result<Vec<_>>>()?;
        let mut host = vec![0.0f32; ELEMS as usize];
        fill_det(&mut host, 5);
        exec.upload(&host, &x)?;
        exec.upload(&host, &y)?;
        exec.upload(&host, &acc)?;
        Ok(Self { x, y, acc, outs })
    }

    /// N dependent 1-adds into ONE tensor: RAW/WAW hazard per op, so
    /// the graph records a barrier before every op after the first.
    fn serial_chain(&self, n: usize) -> Vec<ExecOp<'_>> {
        (0..n)
            .map(|_| ExecOp::Binary {
                a: &self.acc,
                b: &self.y,
                out: &self.acc,
                op: BinaryOp::AddScaled { beta: 1.0 },
            })
            .collect()
    }

    /// N independent 1-adds into N distinct tensors: no hazards, zero
    /// barriers, dispatches free to overlap.
    fn overlap_chain(&self, n: usize) -> Vec<ExecOp<'_>> {
        self.outs[..n]
            .iter()
            .map(|out| ExecOp::Binary {
                a: &self.x,
                b: &self.y,
                out,
                op: BinaryOp::AddScaled { beta: 1.0 },
            })
            .collect()
    }
}

fn median(samples: &[f64]) -> Option<f64> {
    SampleSummary::new(samples.iter().copied()).map(|summary| summary.median)
}

fn skip(thesis: &'static str, item: &str, reason: &str) -> Vec<Row> {
    vec![Row {
        thesis,
        item: item.into(),
        prediction: "-".into(),
        measured: reason.into(),
        verdict: Verdict::Skip,
    }]
}

pub(crate) fn t3_barrier_cost(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    section: Option<&Section>,
) -> Result<(Vec<Row>, Option<f64>)> {
    const ITEM: &str = "compute-barrier drain (trivial-kernel chain slope)";
    if !ctx.buffer_device_address_enabled {
        return Ok((skip("T3", ITEM, "device lacks bufferDeviceAddress"), None));
    }
    let ops = TrivialOps::new(ctx, exec)?;

    // The measurement is only meaningful if the graphs really contain
    // the barrier structure the docs claim; the plan-only census
    // asserts it without touching the queue.
    let (dispatches, barriers) = exec.exec_ops_barrier_count(&ops.serial_chain(N_LONG))?;
    ensure!(
        dispatches == N_LONG && barriers == N_LONG - 1,
        "serial chain planned {dispatches} dispatches / {barriers} barriers, \
         expected {N_LONG} / {}",
        N_LONG - 1
    );
    let (dispatches, barriers) = exec.exec_ops_barrier_count(&ops.overlap_chain(N_LONG))?;
    ensure!(
        dispatches == N_LONG && barriers == 0,
        "overlap chain planned {dispatches} dispatches / {barriers} barriers, \
         expected {N_LONG} / 0"
    );

    // Warm both chains, then interleave the four measurements so clock
    // drift hits every series equally.
    exec.run_exec_ops(&ops.serial_chain(N_LONG))?;
    exec.run_exec_ops(&ops.overlap_chain(N_LONG))?;
    let mut series: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for _ in 0..REPS {
        for (index, chain) in [
            ops.serial_chain(N_SHORT),
            ops.serial_chain(N_LONG),
            ops.overlap_chain(N_SHORT),
            ops.overlap_chain(N_LONG),
        ]
        .into_iter()
        .enumerate()
        {
            match exec.run_exec_ops(&chain)?.gpu_time_ns {
                Some(ns) => series[index].push(ns as f64 / 1e3),
                None => return Ok((skip("T3", ITEM, "device reports no GPU timestamps"), None)),
            }
        }
    }
    let [serial_short, serial_long, overlap_short, overlap_long] =
        [&series[0], &series[1], &series[2], &series[3]].map(|s| median(s).unwrap_or(f64::NAN));

    let per_extra = (N_LONG - N_SHORT) as f64;
    let barrier_us = ((serial_long - serial_short) - (overlap_long - overlap_short)) / per_extra;
    let serial_slope = (serial_long - serial_short) / per_extra;

    let expected = key(section, "t3_barrier_us");
    let rel_tol = key(section, "t3_rel_tol").unwrap_or(0.5);
    let rows = vec![
        Row {
            thesis: "T3",
            item: ITEM.into(),
            prediction: expected.map_or_else(
                || "unrecorded".into(),
                |value| format!("~{value:.1} us/barrier (+-{:.0}%)", rel_tol * 100.0),
            ),
            measured: format!(
                "{barrier_us:.2} us/barrier (serial {N_SHORT}->{N_LONG}: \
                 {serial_short:.0}->{serial_long:.0} us; overlapped: \
                 {overlap_short:.0}->{overlap_long:.0} us)"
            ),
            verdict: check_within(barrier_us, expected, rel_tol),
        },
        Row {
            thesis: "T3",
            item: "serialized trivial dispatch (kernel + barrier)".into(),
            prediction: "-".into(),
            measured: format!("{serial_slope:.2} us/dispatch"),
            verdict: Verdict::Info,
        },
    ];
    Ok((rows, Some(barrier_us)))
}

const N_SUBMITS: usize = 64;

pub(crate) fn t4_submission_cost(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    section: Option<&Section>,
) -> Result<Vec<Row>> {
    const ITEM: &str = "vkQueueSubmit + fence round-trip (N submits vs 1)";
    if !ctx.buffer_device_address_enabled {
        return Ok(skip("T4", ITEM, "device lacks bufferDeviceAddress"));
    }
    let ops = TrivialOps::new(ctx, exec)?;
    let batch = ops.overlap_chain(N_SUBMITS);

    // Warm both paths.
    exec.run_exec_ops(&batch)?;
    for out in &ops.outs[..4] {
        exec.run_exec_ops(&[ExecOp::Binary {
            a: &ops.x,
            b: &ops.y,
            out,
            op: BinaryOp::AddScaled { beta: 1.0 },
        }])?;
    }

    let mut many_ms = Vec::with_capacity(REPS);
    let mut one_ms = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        for out in &ops.outs[..N_SUBMITS] {
            exec.run_exec_ops(&[ExecOp::Binary {
                a: &ops.x,
                b: &ops.y,
                out,
                op: BinaryOp::AddScaled { beta: 1.0 },
            }])?;
        }
        many_ms.push(start.elapsed().as_secs_f64() * 1e3);

        let start = Instant::now();
        exec.run_exec_ops(&batch)?;
        one_ms.push(start.elapsed().as_secs_f64() * 1e3);
    }
    let many = median(&many_ms).unwrap_or(f64::NAN);
    let one = median(&one_ms).unwrap_or(f64::NAN);
    let submit_us = (many - one) * 1e3 / (N_SUBMITS - 1) as f64;

    let lo = key(section, "t4_submit_us_lo");
    let hi = key(section, "t4_submit_us_hi");
    let edge_tol = key(section, "t4_edge_tol").unwrap_or(0.2);
    let verdict = match (lo, hi) {
        (Some(lo), Some(hi)) => {
            if submit_us >= lo * (1.0 - edge_tol) && submit_us <= hi * (1.0 + edge_tol) {
                Verdict::Pass
            } else {
                Verdict::Fail
            }
        }
        _ => Verdict::Info,
    };
    Ok(vec![Row {
        thesis: "T4",
        item: ITEM.into(),
        prediction: match (lo, hi) {
            (Some(lo), Some(hi)) => {
                format!(
                    "{lo:.0}-{hi:.0} us/submit (+-{:.0}% at edges)",
                    edge_tol * 100.0
                )
            }
            _ => "unrecorded".into(),
        },
        measured: format!(
            "{submit_us:.1} us/submit ({N_SUBMITS} submits {many:.2} ms vs 1 submit {one:.2} ms)"
        ),
        verdict,
    }])
}
