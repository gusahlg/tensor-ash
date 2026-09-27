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

Other subcommands are `single`, `cases`, `ab`, `concurrent`, `transfer`,
`prepared` (which compares synchronous dispatch, prepared replay, and
pipelined prepared submission for one repeated shape), and `thesis` —
the performance-thesis harness: it re-measures the campaign's explicit
performance model (T1-T7, documented in `benchmarks/THESES.md`) against
per-GPU expectations checked in at `benchmarks/thesis-expectations.toml`,
prints a prediction-vs-measured table, and exits nonzero on any
regression. `ML_MODEL=<gguf> ml_bench thesis --all` runs everything (the
model-level theses T1/T5/T7 load a llama-family GGUF through the
`llama-ash` library); `ml_bench thesis t2 t3 t4 t6` runs the
kernel-level subset without a model. `cases`
runs multiple `label,b,m,n,k[,f16w][,packed]` arguments in one process, avoiding repeated
pipeline startup and GPU clock perturbation in automation. `ab` does the same
for a multi-kernel comparison: `ML_AB_KERNELS` lists catalog names or `auto`,
one context is shared, tensors are allocated once per case, and `ML_ROUNDS`
(default 5) interleaved rounds rotate which kernel runs first; `ML_AB_COLD_MB=N`
evicts L2 with an N MiB buffer pass and re-reads A before every timed GEMM
(cold weights, warm activations — what a model forward sees). The environment
knobs are documented in the workspace `README.md`; `ML_DEVICE`, `ML_KERNEL`,
`ML_TUNE`, `ML_B`/`ML_M`/`ML_N`/`ML_K`, `ML_AB_KERNELS`, `ML_ROUNDS`, and
`ML_OUTPUT` are the common ones.

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
ML_OUTPUT=csv target/release/ml_bench cases \
  lm_head,1,1,32000,2048,packed kv,1,1,256,2048,f16w,packed

ML_AB_KERNELS=k64_bda_v4_tm8_tn4,m128n64k64_bda_v4,auto \
  target/release/ml_bench ab \
  square_512,1,512,512,512 medium_768,1,768,768,768
```

## GPU assembly: `isa`, `isa-analyze`, `isa-cache`

`ml_bench isa` shows the SASS the NVIDIA driver actually generates for a
routed kernel. Each argument is a `cases` spec
(`label,b,m,n,k[,f16w][,packed]`); the shape is routed exactly like a
dispatch (tuned winners and `ML_KERNEL=` included), the bound
specialization is derived (`KernelVariant::for_dispatch`, zero epilogue),
and that pipeline is compiled through the library's own creation path
(`MatmulPipeline::create_probe_pipeline`) into a fresh, empty
`VkPipelineCache`. Pipeline creation dispatches no GPU work.

```console
ML_ISA_OUT=isa-out target/release/ml_bench isa \
  gateup,1,1,5632,2048,f16w lm_head,1,1,32000,2048,packed \
  odd,1,1023,1025,1027 sq4096,1,4096,4096,4096
```

Per case it writes `<label>.<kernel>.v<variant>.<arch>.*` into `ML_ISA_OUT`:

| file | content |
|---|---|
| `.meta` | route, variant, spec constants, workgroup size, declared shared bytes, registers (NVuc `REG` section), code hash, driver statistics (`stat.*`), `capture_flags_code` (capture flags must not change code), `in_production_cache` (the same UCODE exists in the persisted cache) |
| `.nvuc` | the driver's ISA container, decoded from the cache blob (`nvcache.rs` documents the layout) |
| `.ucode.bin` | the exact raw SASS (the container's UCODE section) |
| `.stats.txt` | section table + every `VK_KHR_pipeline_executable_properties` statistic / internal representation |
| `.sass`, `.census.txt` | only when `nvdisasm` is available: the disassembly, and the census (opcode classes, LDG/STG/LDS/STS/LDSM widths, bytes/thread, loops with math density and decoded control bits — summed stall cycles per iteration, scoreboard waits, reuse — occupancy for sm_86/sm_89, and a decode sanity check) |

A markdown summary row per case is printed at the end.

`isa` sets `ML_PIPELINE_EXEC_INFO=1`, which makes the library enable
`VK_KHR_pipeline_executable_properties` for that process only. Knobs:
`ML_ISA_ARCH` (nvdisasm `-b` value; default from the device name, RTX
30xx -> `SM86`, RTX 40xx -> `SM89`), `ML_ISA_NVDISASM` (path or `off`),
`ML_ISA_SPV` (compile this SPIR-V file instead of the embedded module;
same interface required, e.g. a `glslc -Os` build of the same shader),
`ML_ISA_TAG` (suffix for the output stem).

On a host without the CUDA toolkit (louise-pc), copy the output directory
to one that has `nvdisasm` and run the census offline (no GPU, no Vulkan):

```console
ml_bench isa-analyze isa-out/            # every .meta in the directory
ml_bench isa-cache ~/.cache/tensor-ash/pipeline_cache_v10de_2488.bin out/
```

`isa-cache` splits a persisted pipeline cache into one container per
entry and censuses each (the cache stores no kernel names, so those rows
have no occupancy).
