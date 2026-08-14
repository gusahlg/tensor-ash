//! Position-buffer replay correctness: a command buffer recorded ONCE
//! with pos-relative descs must, as the host bumps the 4-byte position
//! cell between replays, produce bit-identical results to fresh
//! recordings with the position baked into the push constants.

use crate::common::*;

use tensor_ash::{AttnDecodeDesc, CopyDesc, ExecOp, MatmulCall, MatmulOp, RopeDesc, Tensor};

/// Bitwise equality: replay vs direct must not differ in a single ULP
/// (identical kernels, identical arithmetic, only the position source
/// differs).
fn assert_bitwise(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.to_bits() == y.to_bits(),
            "{label}: bitwise mismatch at idx {i}: {x:?} vs {y:?}"
        );
    }
}

#[test]
#[ignore]
fn pos_buffer_rope_replay_matches_direct() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let (tokens, heads, dh) = (2_u32, 2_u32, 8_u32);
    let (input, _) = upload_det(&ctx, &exec, &[tokens, heads * dh], 9101);
    let (table, _) = upload_det(&ctx, &exec, &[64, dh / 2, 2], 9102);
    let replayed = Tensor::uninit_device(&ctx, &[tokens, heads * dh]).unwrap();
    let direct = Tensor::uninit_device(&ctx, &[tokens, heads * dh]).unwrap();

    let pos = exec.create_pos_buffer().unwrap();
    let ops = [ExecOp::Rope {
        input: &input,
        table: &table,
        output: &replayed,
        desc: RopeDesc {
            heads,
            head_dim: dh,
            rot_dim: dh,
            pos_base: 0,
            pos_addr: pos.device_address(),
        },
    }];
    let mut prepared = exec.prepare_exec_ops(&ops, Some(&pos)).unwrap();

    let mut got = vec![0.0; (tokens * heads * dh) as usize];
    let mut want = vec![0.0; got.len()];
    for p in [5_u32, 9] {
        pos.set(p).unwrap();
        prepared.run().unwrap();
        exec.download(&replayed, &mut got).unwrap();

        exec.run_rope(
            &input,
            &table,
            &direct,
            RopeDesc {
                heads,
                head_dim: dh,
                rot_dim: dh,
                pos_base: p,
                ..Default::default()
            },
        )
        .unwrap();
        exec.download(&direct, &mut want).unwrap();
        assert_bitwise(&got, &want, &format!("rope replay pos={p}"));
    }
}

#[test]
#[ignore]
fn attn_decode_fixed_grid_matches_variable_grid() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let (kv_heads, group, dh, t_max) = (2_u32, 4_u32, 64_u32, 1024_u32);
    let (q, _) = upload_det(&ctx, &exec, &[kv_heads, group, dh], 9201);
    let (kt, _) = upload_det(&ctx, &exec, &[kv_heads, dh, t_max], 9202);
    let (v, _) = upload_det(&ctx, &exec, &[kv_heads, t_max, dh], 9203);
    let scratch = Tensor::uninit_device(
        &ctx,
        &[kv_heads * tensor_ash::ATTN_DECODE_MAX_CHUNKS * group * (dh + 2)],
    )
    .unwrap();
    let replayed = Tensor::uninit_device(&ctx, &[kv_heads, group, dh]).unwrap();
    let immediate = Tensor::uninit_device(&ctx, &[kv_heads, group, dh]).unwrap();
    let variable = Tensor::uninit_device(&ctx, &[kv_heads, group, dh]).unwrap();
    let scale = 1.0 / (dh as f32).sqrt();

    let pos = exec.create_pos_buffer().unwrap();
    // Recorded ONCE with kv_len = 1 (effective length 1 + p) and the
    // fixed 32-chunk grid; chunks past the effective length write
    // neutral partials that the combine merges as exact no-ops.
    let pos_desc = AttnDecodeDesc {
        kv_len: 1,
        scale,
        pos_addr: pos.device_address(),
    };
    let ops = [ExecOp::AttnDecode {
        q: &q,
        kt: &kt,
        v: &v,
        scratch: &scratch,
        out: &replayed,
        desc: pos_desc,
    }];
    let mut prepared = exec.prepare_exec_ops(&ops, Some(&pos)).unwrap();

    let mut got = vec![0.0; (kv_heads * group * dh) as usize];
    let mut imm = vec![0.0; got.len()];
    let mut var = vec![0.0; got.len()];
    for kv_len in [1_u32, 33, 640] {
        pos.set(kv_len - 1).unwrap();
        prepared.run().unwrap();
        exec.download(&replayed, &mut got).unwrap();

        // Same pos-driven dispatch, freshly recorded: identical chunk
        // decomposition, so the replay must match bitwise.
        exec.run_attn_decode(&q, &kt, &v, &scratch, &immediate, pos_desc)
            .unwrap();
        exec.download(&immediate, &mut imm).unwrap();
        assert_bitwise(&got, &imm, &format!("attn_decode replay kv_len={kv_len}"));

        // The variable-grid path picks its own chunk count, so the
        // summation order differs; agreement is numerical, not bitwise.
        exec.run_attn_decode(
            &q,
            &kt,
            &v,
            &scratch,
            &variable,
            AttnDecodeDesc {
                kv_len,
                scale,
                ..Default::default()
            },
        )
        .unwrap();
        exec.download(&variable, &mut var).unwrap();
        assert_close_tol(
            &got,
            &var,
            1e-4,
            &format!("attn_decode fixed vs variable grid kv_len={kv_len}"),
        );
    }
}

#[test]
#[ignore]
fn prepared_exec_ops_replays_mixed_graph_at_two_positions() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    // rope -> strided append at a position-scaled offset -> GEMV: the
    // dependent chain exercises the hazard barriers, an elementwise
    // pos-driven op, and a recorded matmul in one prepared graph.
    let (heads, dh, k, n) = (2_u32, 16_u32, 32_u32, 16_u32);
    let cache_rows = 64_u32;
    let (input, _) = upload_det(&ctx, &exec, &[1, k], 9301);
    let (table, _) = upload_det(&ctx, &exec, &[cache_rows, dh / 2, 2], 9302);
    let (weight, _) = upload_det(&ctx, &exec, &[k, n], 9303);
    let roped = Tensor::uninit_device(&ctx, &[1, k]).unwrap();
    let roped_direct = Tensor::uninit_device(&ctx, &[1, k]).unwrap();
    let cache = Tensor::uninit_device(&ctx, &[cache_rows, k]).unwrap();
    let cache_direct = Tensor::uninit_device(&ctx, &[cache_rows, k]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
    let c_direct = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
    let zeros = vec![0.0_f32; (cache_rows * k) as usize];
    exec.upload(&zeros, &cache).unwrap();
    exec.upload(&zeros, &cache_direct).unwrap();

    let pos = exec.create_pos_buffer().unwrap();
    let mm = |a, b, c| MatmulCall {
        a,
        b,
        c,
        alpha: 1.0,
        accumulate: false,
    };
    let ops = [
        ExecOp::Rope {
            input: &input,
            table: &table,
            output: &roped,
            desc: RopeDesc {
                heads,
                head_dim: dh,
                rot_dim: dh,
                pos_base: 0,
                pos_addr: pos.device_address(),
            },
        },
        ExecOp::CopyStrided {
            src: &roped,
            dst: &cache,
            desc: CopyDesc {
                extent: [k, 1, 1],
                src_offset: 0,
                src_strides: [1, 0, 0],
                dst_offset: 0,
                dst_strides: [1, 0, 0],
                pos_addr: pos.device_address(),
                pos_scale: k,
            },
        },
        ExecOp::Matmul(MatmulOp::new(mm(&roped, &weight, &c))),
    ];
    let mut prepared = exec.prepare_exec_ops(&ops, Some(&pos)).unwrap();

    // A mismatched position buffer is rejected at prepare time.
    let other = exec.create_pos_buffer().unwrap();
    let err = exec
        .prepare_exec_ops(&ops, Some(&other))
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(err.contains("pos_addr"), "unexpected error: {err}");

    for p in [5_u32, 6] {
        pos.set(p).unwrap();
        prepared.run().unwrap();

        exec.run_exec_ops(&[
            ExecOp::Rope {
                input: &input,
                table: &table,
                output: &roped_direct,
                desc: RopeDesc {
                    heads,
                    head_dim: dh,
                    rot_dim: dh,
                    pos_base: p,
                    ..Default::default()
                },
            },
            ExecOp::CopyStrided {
                src: &roped_direct,
                dst: &cache_direct,
                desc: CopyDesc {
                    extent: [k, 1, 1],
                    src_offset: 0,
                    src_strides: [1, 0, 0],
                    dst_offset: p * k,
                    dst_strides: [1, 0, 0],
                    ..Default::default()
                },
            },
            ExecOp::Matmul(MatmulOp::new(mm(&roped_direct, &weight, &c_direct))),
        ])
        .unwrap();

        let mut got = vec![0.0; n as usize];
        let mut want = got.clone();
        exec.download(&c, &mut got).unwrap();
        exec.download(&c_direct, &mut want).unwrap();
        assert_bitwise(&got, &want, &format!("mixed replay GEMV pos={p}"));
    }

    // Both caches saw the appends at rows 5 and 6; compare in full.
    let mut got = vec![0.0; zeros.len()];
    let mut want = got.clone();
    exec.download(&cache, &mut got).unwrap();
    exec.download(&cache_direct, &mut want).unwrap();
    assert_bitwise(&got, &want, "mixed replay cache appends");
}
