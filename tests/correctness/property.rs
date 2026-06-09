use crate::common::*;

#[test]
#[ignore]
fn property_random_shapes() {
    let (ctx, exec) = make_setup(2, 8);
    let mut rng = LcgRng::new(0xC0FFEE);

    let trials = 32;
    let mut failures: Vec<String> = Vec::new();

    for trial in 0..trials {
        let batch = rng.range_u32(1, 4);
        let m = rng.range_u32(1, 320);
        let n = rng.range_u32(1, 320);
        let k = rng.range_u32(1, 320);

        let shape_a = if batch == 1 {
            vec![m, k]
        } else {
            vec![batch, m, k]
        };
        let shape_b = if batch == 1 {
            vec![k, n]
        } else {
            vec![batch, k, n]
        };
        let shape_c = if batch == 1 {
            vec![m, n]
        } else {
            vec![batch, m, n]
        };

        let (gpu, cpu) = run_one(
            &ctx,
            &exec,
            &shape_a,
            &shape_b,
            &shape_c,
            1.0,
            false,
            1000 + trial as u64,
            2000 + trial as u64,
            None,
        );
        let (e, idx) = max_abs_err(&gpu, &cpu);
        let tol = tolerance(k);
        if e > tol {
            failures.push(format!(
                "trial {trial}: B={batch} M={m} N={n} K={k} err={e:.3e} > tol={tol:.3e} \
                 at idx {idx}: gpu={:.6} cpu={:.6}",
                gpu[idx], cpu[idx],
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {trials} random shapes failed:\n{}",
        failures.len(),
        failures.join("\n"),
    );
}
