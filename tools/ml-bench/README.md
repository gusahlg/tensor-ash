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

Other subcommands are `single`, `concurrent`, and `transfer`. The environment
knobs are documented in the workspace `README.md`; `ML_DEVICE`, `ML_KERNEL`,
`ML_TUNE`, `ML_B`/`ML_M`/`ML_N`/`ML_K`, and `ML_OUTPUT` are the common ones.

The executable remains named `ml_bench`, so release builds still produce
`target/release/ml_bench` and existing benchmark automation can keep using that
path.

The cross-library harness in `scripts/bench_compare.py` builds this package
with `cargo build --release -p ml-bench` unless `--skip-build` is passed.
