use crate::common::*;

#[test]
#[ignore]
fn batched_no_broadcast() {
    let (ctx, exec) = make_setup(2, 16);
    let cases: &[(u32, u32, u32, u32)] = &[
        (1, 8, 8, 8),
        (4, 32, 32, 32),
        (3, 47, 53, 41),
        (8, 128, 128, 128),
        (12, 256, 192, 64),
    ];
    for &(bsz, m, n, k) in cases {
        let (gpu, cpu) = run_one(
            &ctx,
            &exec,
            &[bsz, m, k],
            &[bsz, k, n],
            &[bsz, m, n],
            1.0,
            false,
            21,
            23,
            None,
        );
        assert_close(&gpu, &cpu, k, &format!("B={bsz} M={m} N={n} K={k}"));
    }
}

#[test]
#[ignore]
fn batched_broadcast_a() {
    let (ctx, exec) = make_setup(2, 8);
    let bsz = 5u32;
    let (m, n, k) = (32u32, 48u32, 24u32);
    let (gpu, cpu) = run_one(
        &ctx,
        &exec,
        &[1, m, k],
        &[bsz, k, n],
        &[bsz, m, n],
        1.0,
        false,
        31,
        37,
        None,
    );
    assert_close(&gpu, &cpu, k, "broadcast A");
}

#[test]
#[ignore]
fn batched_broadcast_b() {
    let (ctx, exec) = make_setup(2, 8);
    let bsz = 5u32;
    let (m, n, k) = (32u32, 48u32, 24u32);
    let (gpu, cpu) = run_one(
        &ctx,
        &exec,
        &[bsz, m, k],
        &[1, k, n],
        &[bsz, m, n],
        1.0,
        false,
        41,
        43,
        None,
    );
    assert_close(&gpu, &cpu, k, "broadcast B");
}

#[test]
#[ignore]
fn batched_broadcast_both_inputs() {
    let (ctx, exec) = make_setup(2, 8);
    let bsz = 5u32;
    let (m, n, k) = (32u32, 48u32, 24u32);
    let (gpu, cpu) = run_one(
        &ctx,
        &exec,
        &[1, m, k],
        &[1, k, n],
        &[bsz, m, n],
        1.0,
        false,
        45,
        47,
        None,
    );
    assert_close(&gpu, &cpu, k, "broadcast A+B");
}
