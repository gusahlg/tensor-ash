//! GPU token-loop ops: argmax and embedding gather.
//!
//! The argmax tie-break contract is exact equivalence with the CPU
//! greedy sampler used by llama-ash — Rust's
//! `Iterator::max_by(f32::total_cmp)`, which returns the LAST maximum
//! element (largest index) on ties — so a GPU-argmaxed decode loop
//! must reproduce the CPU loop token-for-token.

use crate::common::*;

use tensor_ash::dtype::{f16_bits_to_f32, f32_to_f16_bits};
use tensor_ash::{ExecOp, MatmulCall, MatmulOp, Tensor};

/// Mirror of llama-ash's greedy sampler (last max wins on ties).
fn cpu_argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("non-empty")
}

#[test]
#[ignore]
fn argmax_matches_cpu_across_sizes() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let result = exec.create_host_u32_buffer().unwrap();
    for (n, seed) in [(1_u32, 7101_u64), (255, 7102), (257, 7103), (32000, 7104)] {
        let (input, host) = upload_det(&ctx, &exec, &[1, n], seed);
        exec.run_argmax(&input, &result).unwrap();
        let got = result.read().unwrap();
        let want = cpu_argmax(&host);
        assert_eq!(got, want, "argmax n={n}: gpu {got} vs cpu {want}");
    }
}

#[test]
#[ignore]
fn argmax_tie_break_matches_rust_max_by() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let result = exec.create_host_u32_buffer().unwrap();
    let n = 32000_u32;
    let tensor = Tensor::uninit_device(&ctx, &[1, n]).unwrap();

    // Constant input: every element ties; Rust max_by keeps the last.
    let host = vec![1.5_f32; n as usize];
    exec.upload(&host, &tensor).unwrap();
    exec.run_argmax(&tensor, &result).unwrap();
    assert_eq!(result.read().unwrap(), cpu_argmax(&host));
    assert_eq!(result.read().unwrap(), n - 1);

    // Duplicated max at two interior positions: the later one wins.
    let mut host = vec![0.0_f32; n as usize];
    fill_det(&mut host, 7201);
    host[5] = 1.0e6;
    host[20000] = 1.0e6;
    exec.upload(&host, &tensor).unwrap();
    exec.run_argmax(&tensor, &result).unwrap();
    assert_eq!(result.read().unwrap(), cpu_argmax(&host));
    assert_eq!(result.read().unwrap(), 20000);

    // Unique max at index 0 stays at index 0.
    let mut host = vec![-2.0_f32; n as usize];
    host[0] = 3.0;
    exec.upload(&host, &tensor).unwrap();
    exec.run_argmax(&tensor, &result).unwrap();
    assert_eq!(result.read().unwrap(), 0);

    // All -inf: still deterministic, last index (ties at the sentinel).
    let host = vec![f32::NEG_INFINITY; n as usize];
    exec.upload(&host, &tensor).unwrap();
    exec.run_argmax(&tensor, &result).unwrap();
    assert_eq!(result.read().unwrap(), cpu_argmax(&host));
    assert_eq!(result.read().unwrap(), n - 1);
}

#[test]
#[ignore]
fn embed_gather_matches_cpu_f32_and_f16() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let (vocab, embd) = (301_u32, 320_u32);
    let (table_f32, host_table) = upload_det(&ctx, &exec, &[vocab, embd], 7301);
    let table_f16 = Tensor::uninit_device_f16(&ctx, &[vocab, embd]).unwrap();
    exec.upload(&host_table, &table_f16).unwrap();

    let token = exec.create_host_u32_buffer().unwrap();
    let out = Tensor::uninit_device(&ctx, &[1, embd]).unwrap();
    let mut got = vec![0.0_f32; embd as usize];
    for tok in [0_u32, 7, vocab - 1] {
        token.set(tok).unwrap();
        let row = &host_table[(tok * embd) as usize..((tok + 1) * embd) as usize];

        exec.run_embed_gather(&token, &table_f32, &out).unwrap();
        exec.download(&out, &mut got).unwrap();
        assert_close_tol(&got, row, 0.0, &format!("gather f32 tok={tok}"));

        // The f16 table narrows on upload; the gather must widen the
        // stored half exactly.
        let row_f16: Vec<f32> = row
            .iter()
            .map(|&v| f16_bits_to_f32(f32_to_f16_bits(v)))
            .collect();
        exec.run_embed_gather(&token, &table_f16, &out).unwrap();
        exec.download(&out, &mut got).unwrap();
        assert_close_tol(&got, &row_f16, 0.0, &format!("gather f16 tok={tok}"));
    }

    // Out-of-range ids clamp to the last row instead of reading OOB.
    token.set(vocab + 100).unwrap();
    exec.run_embed_gather(&token, &table_f32, &out).unwrap();
    exec.download(&out, &mut got).unwrap();
    let last = &host_table[((vocab - 1) * embd) as usize..];
    assert_close_tol(&got, last, 0.0, "gather clamp");
}

/// The decode-tail chain as one graph submission — lm-head matmul,
/// argmax, embedding gather — must match the stepwise ops (and the CPU
/// sampler + gather) bitwise: same kernels, only the submission shape
/// differs.
#[test]
#[ignore]
fn exec_chain_matmul_argmax_gather_matches_stepwise() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let (embd, vocab) = (256_u32, 1000_u32);
    let (x, _) = upload_det(&ctx, &exec, &[1, embd], 7401);
    let (w, _) = upload_det(&ctx, &exec, &[embd, vocab], 7402);
    let (table, host_table) = upload_det(&ctx, &exec, &[vocab, embd], 7403);

    let logits = Tensor::uninit_device(&ctx, &[1, vocab]).unwrap();
    let next_x = Tensor::uninit_device(&ctx, &[1, embd]).unwrap();
    let token = exec.create_host_u32_buffer().unwrap();

    let mm = MatmulCall {
        a: &x,
        b: &w,
        c: &logits,
        alpha: 1.0,
        accumulate: false,
    };
    exec.run_exec_ops(&[
        ExecOp::Matmul(MatmulOp::new(mm)),
        ExecOp::Argmax {
            input: &logits,
            result: &token,
        },
        ExecOp::EmbedGather {
            token: &token,
            table: &table,
            out: &next_x,
        },
    ])
    .unwrap();
    let chained_token = token.read().unwrap();
    let mut chained_x = vec![0.0_f32; embd as usize];
    exec.download(&next_x, &mut chained_x).unwrap();

    // Stepwise: same matmul kernel, CPU argmax of the downloaded
    // logits, CPU gather.
    exec.run_matmuls(&[mm]).unwrap();
    let mut host_logits = vec![0.0_f32; vocab as usize];
    exec.download(&logits, &mut host_logits).unwrap();
    let cpu_token = cpu_argmax(&host_logits);
    assert_eq!(chained_token, cpu_token, "chained argmax vs CPU");
    let cpu_row = &host_table[(cpu_token * embd) as usize..((cpu_token + 1) * embd) as usize];
    assert_close_tol(&chained_x, cpu_row, 0.0, "chained gather vs CPU row");
}

/// Prepared replay of the argmax + gather tail: one recording, new
/// logits between replays, correct token and row every time — the
/// self-feeding decode loop's core loop invariant.
#[test]
#[ignore]
fn prepared_argmax_gather_replays_with_fresh_inputs() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let (vocab, embd) = (500_u32, 64_u32);
    let (table, host_table) = upload_det(&ctx, &exec, &[vocab, embd], 7501);
    let logits = Tensor::uninit_device(&ctx, &[1, vocab]).unwrap();
    let out = Tensor::uninit_device(&ctx, &[1, embd]).unwrap();
    let token = exec.create_host_u32_buffer().unwrap();

    let ops = [
        ExecOp::Argmax {
            input: &logits,
            result: &token,
        },
        ExecOp::EmbedGather {
            token: &token,
            table: &table,
            out: &out,
        },
    ];
    let mut prepared = exec.prepare_exec_ops(&ops, None).unwrap();

    let mut got = vec![0.0_f32; embd as usize];
    for seed in [7502_u64, 7503, 7504] {
        let mut host_logits = vec![0.0_f32; vocab as usize];
        fill_det(&mut host_logits, seed);
        exec.upload(&host_logits, &logits).unwrap();
        prepared.run().unwrap();
        let want = cpu_argmax(&host_logits);
        assert_eq!(token.read().unwrap(), want, "replay seed {seed}");
        exec.download(&out, &mut got).unwrap();
        let row = &host_table[(want * embd) as usize..((want + 1) * embd) as usize];
        assert_close_tol(&got, row, 0.0, &format!("replay gather seed {seed}"));
    }
}
