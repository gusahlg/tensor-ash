# ml-bench

The `tensor-ash` benchmark and correctness CLI is a non-published workspace
package. Keeping it separate means its logging, environment parsing, fixture,
and reporting concerns do not become runtime dependencies or public API of the
core library. CPU reference math and deterministic fixtures come from the
sibling `tensor-ash-test-support` package.

Run it from the workspace root with an explicit package selection:

```console
cargo run --release -p ml-bench -- self-check
cargo run --release -p ml-bench -- correctness
cargo run --release -p ml-bench -- sweep
```

Other subcommands are `single`, `cases`, `concurrent`, `transfer`, and
`prepared` (which compares synchronous dispatch, prepared replay, and
pipelined prepared submission for one repeated shape). `cases`
runs multiple `label,b,m,n,k[,f16w]` arguments in one process, avoiding repeated
pipeline startup and GPU clock perturbation in automation. The environment
knobs are documented in the workspace `README.md`; `ML_DEVICE`, `ML_KERNEL`,
`ML_TUNE`, `ML_B`/`ML_M`/`ML_N`/`ML_K`, and `ML_OUTPUT` are the common ones.

The executable remains named `ml_bench`, so release builds still produce
`target/release/ml_bench` and existing benchmark automation can keep using that
path.

The cross-library harness in `scripts/bench_compare.py` builds this package
with `cargo build --release -p ml-bench` unless `--skip-build` is passed.

Timing output summarizes all measured samples as minimum, median, and p95;
reported TFLOPS uses the median GPU timestamp. Wall and GPU samples stay paired,
so host overhead is computed per submission before it is summarized. CSV keeps
`wall_ms`, `gpu_ms`, and `tflops` as median-valued compatibility columns and
adds explicit `*_min_ms`, `*_median_ms`, `*_p95_ms`, sample counts, and
end-to-end `wall_tflops` fields.

Set `ML_SPLIT_K2=N` to benchmark an explicit two-stage split-K factor with the
same validation and sample reporting; values below 2 (including unset) use
normal dispatch and report its route.

```console
ML_OUTPUT=csv target/release/ml_bench cases \
  square_512,1,512,512,512 odd_255,1,255,257,263
```
