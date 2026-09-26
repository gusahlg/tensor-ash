# Campaign track f16acc: f16 accumulation on tensor cores (E1) and generator M1

Branch `campaign/f16acc` (worktree `~/repos/tensor-ash-wt/c-f16acc`), based on
exp/grokmerge 7cfca58 + campaign/coopmat-sm (fast-forward: `ml_bench ab`, a16
case flag, SM-aware coopmat routing) + campaign/sass (`ml_bench isa`).
Hosts: RTX 4060 (AD107, 24 SM, driver 595.80) on louise-pc first, RTX 3070
(GA104, 46 SM, driver 595.99.02) second, in short batches of validated binaries.

Raw outputs: `benchmarks/campaign/f16acc-results/` (the `results/...` paths below;
SASS census/stats under `f16acc-results/isa-3070{,-gen}/`); job scripts:
`benchmarks/campaign/f16acc-scripts/` (they reference the session scratchpad
`$SP/f16acc`, where the `.sass` listings and logits dumps stay).

## 1. What llama.cpp b9503 CUDA actually does (verified)

Source: `ggml/src/ggml-cuda/ggml-cuda.cu`, `mmf.cu`, `mmvf.cu`, `common.cuh`,
`src/llama-graph.cpp` at tag b9503 (fetched from GitHub; `ggml-cuda.cu` is
byte-identical to the judge's `$SP/ggml-cuda-b9503.cu`).

Routing of `GGML_OP_MUL_MAT` with an **f16 weight** (`src0`), f32 activations
(`src1`) and f32 output, NVIDIA cc >= Ampere (`ggml_cuda_mul_mat`, l.2540-2630):

| activation rows `ne11` | path | accumulator |
|---|---|---|
| 1 (decode) | `mul_mat_vec_f` (`should_use_mmvf`: Ampere `ne11 == 1`, Ada `ne11 <= 4`) | **half2 per-thread partial sums** at `GGML_PREC_DEFAULT` (`mmvf.cu` l.193-219, `sumh2 += tmpx * y`), reduced across the warp in f32 |
| 2..16 | `mul_mat_f` (`should_use_mmf`: `src1_ncols > 16 -> false`) | llama.cpp's own mma kernel |
| > 16 (prefill, pp512) | `ggml_cuda_op_mul_mat_cublas` | see below |

`ggml_cuda_op_mul_mat_cublas` (l.1627-1745): `use_fp16 = src0 is F16 (or quantized)
&& contiguous && not row-split && dst->op_params[0] == GGML_PREC_DEFAULT`; with
`fast_fp16_hardware_available(cc)` (NVIDIA cc >= Pascal except 610) it converts
src1 to f16 and calls

```
cublasGemmEx(..., CUDA_R_16F A, CUDA_R_16F B, CUDA_R_16F C, CUBLAS_COMPUTE_16F,
             CUBLAS_GEMM_DEFAULT_TENSOR_OP)
```

then converts the **f16 C back to f32** (`to_fp32_cuda`). `CUBLAS_COMPUTE_32F` is
used only on CDNA, RDNA4, Volta (cc == 700) or with
`GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F`; Turing/Ampere/Ada/Hopper take 16F.

Which ops: every weight GEMM of the LLaMA graph is `GGML_PREC_DEFAULT`
(`build_lora_mm` for wq/wk/wv/wo/ffn_gate/ffn_up/ffn_down/output).
`llama-graph.cpp` raises precision to F32 only for attention KQ (l.2092, and
flash attention l.2066) and for down/wo of GLM4 / GLM4_MOE / JAIS2 ("seem to
have numerical issues with half-precision accumulators", l.1374, 2325, 2412).
The LM head in `llama-bench -p` runs for the last token only (ne11 = 1 -> mmvf).

**Empirical confirmation (3070).** `CUBLAS_LOGINFO_DBG=1` around
`llama-bench -m tinyllama-f16 -p 512 -n 0 -r 1 -ngl 99` (b9503, cuBLAS 12.9.1):
all 302 `cublasGemmEx` calls are `Atype=Btype=Ctype=CUDA_R_16F,
computeType=CUBLAS_COMPUTE_16F(64), algo=CUBLAS_GEMM_DEFAULT_TENSOR_OP`, shapes
(cuBLAS m x n x k) 2048x512x2048 (q, o), 256x512x2048 (k, v), 5632x512x2048
(gate, up), 2048x512x5632 (down).  Log: `$SP/f16acc/cublas-3070.log`.

So llama.cpp CUDA's prefill is f16-in / **f16-accumulate** / f16-out on both of
our GPUs, with f16 rounding of every GEMM output.  tensor-ash's a16 route
already stores f16 C; what it did not do is accumulate in f16.

## 2. What was built

| piece | where |
|---|---|
| `ACC_F16` / `ACC_F16_PROMOTE` in the coopmat1 body (f16 accumulator fragments; f16 partials over `PROMOTE_KT`=4 K-tiles = 128 K promoted into f32 fragments).  The f32 path is byte-identical SPIR-V to 7cfca58 | `shaders/matmul_coopmat_kernel.glsl` |
| 16 wrappers: {128x128, 64x64, 64x128, 128x64} x {a32, a16} x {`_acc16`, `_acc16p`} | `shaders/matmul_f16w{,_a16}_coopmat*_acc16{,p}.comp` |
| capability query: 16x16x16 subgroup coopmat with f16 A/B/C/Result | `VulkanContext::coopmat_f16_acc_enabled` |
| policy: `CoopmatAcc {F32, F16, F16Promote}`, `ExecutorConfig::coopmat_acc` (default F32), `ML_COOPMAT_ACC` for env-built executors; explicit `ML_KERNEL` pins keep their own class; never a tuner candidate; unsupported device -> F32 + warning | `src/pipeline/precision.rs`, `src/executor/mod.rs`, `src/pipeline/mod.rs` |
| accumulator-aware tile rule (f16 classes prefer 64x128 / 128x128) | `routing::coopmat_tile_acc16` |
| acc twins build no eager fallback pipelines (lazy, zero cold-start cost) | `create::create_kernel_lazy` |
| `ml_bench ab` entries `auto@f16` / `auto@promote`; accumulator-aware sampled validation | `tools/ml-bench/src/bench/{ab,commands}.rs` |
| f64-reference numerics gate + error table | `examples/coopmat_acc_numerics.rs` |
| real-activation numerics (`llama_ash numerics`), prefill logits dump (`LLAMA_ASH_LOGITS_OUT`) | `tools/llama-ash/src/model/numerics.rs`, `main.rs` |

## 3. Numerics (measured before any timing)

### 3a. Correctness gate: every coopmat1 kernel, full output vs f64

`cargo run --release --example coopmat_acc_numerics` (4060; the 3070 run with
`ML_NUMERICS_SKIP_GEN=1` is identical to the last digit).  256x256x1024,
A, B ~ U[-1,1) rounded to f16, full 65,536-element output vs an f64 reference.
Error unit u = 2^-11 * sqrt(sum_k (a_k b_k)^2) (one f16 rounding of the
typical running-sum size):

| class | kernels | max abs err | max err (u) | rel RMS err | mean err | bound |
|---|---|---|---|---|---|---|
| f32 acc, f32 C (a32) | 4 tiles + 15 gen configs | 1.19e-4 | 0.02 | 1.73e-6 | -2.1e-7 | 1 u |
| f32 acc, f16 C (a16) | 4 tiles + 15 gen configs | 1.56e-2 | 3.13 | 2.07e-4 | -4.0e-6 | 9 u |
| f16 acc (a32 and a16) | 8 | 1.79e-1 | 33.2 | 1.66e-3 | +1.2e-4 | K/16 u |
| f16 promote (128 K chunks) | 8 | 3.7e-2 - 4.0e-2 | 7.2 - 7.8 | 6.0e-4 - 6.4e-4 | +1.5e-6 - +3.9e-6 | 8 sqrt(K/128) u |

All 54 kernels pass; every tile of one class gives bit-identical error
statistics (the per-element accumulation order does not depend on the tile).
Output: `$SP/f16acc/results/numerics-synth-4060.md`.

### 3b. Prefill shapes, K = 2048 and 5632 (64 sampled rows x all N vs f64)

Same example, part 2, auto route.  `unif` = U[-1,1); `llm` = A ~ N(0,1) with
0.5% outlier channels x20, B ~ N(0, 0.02^2).  Relative RMS error (and max abs
error / max|ref|):

| dist | shape | f32 acc, a32 (f32 C) | f32 acc, a16 (f16 C) | f16 acc | f16 promote |
|---|---|---|---|---|---|
| unif | 512x2048x2048 | 3.4e-6 (4.9e-6) | 2.08e-4 (2.4e-4) | 2.36e-3 (5.0e-3) | 6.4e-4 (7.8e-4) |
| unif | 512x5632x2048 | 3.4e-6 (4.3e-6) | 2.07e-4 (4.2e-4) | 2.35e-3 (4.7e-3) | 6.4e-4 (7.5e-4) |
| unif | 512x2048x5632 | 9.1e-6 (1.1e-5) | 2.07e-4 (2.7e-4) | **3.90e-3** (7.6e-3) | 6.4e-4 (6.7e-4) |
| llm | 512x2048x2048 | 2.5e-6 (3.8e-6) | 2.07e-4 (2.6e-4) | 1.98e-3 (4.4e-3) | 5.9e-4 (1.0e-3) |
| llm | 512x5632x2048 | 2.6e-6 (3.2e-6) | 2.07e-4 (4.5e-4) | 1.99e-3 (4.0e-3) | 5.9e-4 (1.0e-3) |
| llm | 512x2048x5632 | 9.4e-6 (1.1e-5) | 2.07e-4 (2.7e-4) | **4.00e-3** (7.4e-3) | 6.2e-4 (6.9e-4) |

* f16 accumulation error grows like sqrt(K) (2.36e-3 -> 3.90e-3 for K 2048 ->
  5632: ratio 1.65 = sqrt(2.75)); the mean error is ~2% of the RMS error, so
  the HMMA.F16 rounding is unbiased (a truncating accumulator would drift
  linearly).  It is ~10-19x the unavoidable f16-output floor of the a16 route.
* Promote is flat in K at ~3x the f16-output floor.
* No overflow on any synthetic set (sums up to 119).

### 3c. Real TinyLlama activations (last layer of a 512-token real-text prefill)

`llama_ash numerics -m tinyllama-1.1b-chat-v1.0.f16.gguf --ids <512 real-text
ids>` (4060): the prefill leaves the last layer's GEMM inputs in scratch; each
GEMM is rerun through executors at every policy and compared, over 64 sampled
rows x all N, with an f64 reference.  "floor" = the f64 reference rounded to
f16 (what any f16-C route must pay).

| GEMM (MxNxK) | max abs A | max abs ref | floor rel RMS | f32 acc | f16 acc | promote |
|---|---|---|---|---|---|---|
| qkv 512x2560x2048 | 10.2 | 21.9 | 2.02e-4 | 2.02e-4 (max 5.0e-3) | 2.02e-3 (max 7.2e-2) | 3.82e-4 (max 8.5e-3) |
| o 512x2048x2048 | 2.79 | 6.86 | 2.07e-4 | 2.08e-4 (max 2.0e-3) | 2.70e-3 (max 3.7e-2) | 4.91e-4 (max 2.9e-3) |
| up 512x5632x2048 | 13.4 | 17.3 | 2.06e-4 | 2.07e-4 (max 5.0e-3) | 2.07e-3 (max 5.2e-2) | 4.29e-4 (max 8.7e-3) |
| down 512x2048x5632 | **173** | **134** | 1.49e-4 | 1.53e-4 (max 1.0e-2) | **1.14e-2** (max **1.76**) | 3.59e-4 (max 1.0e-2) |

(relative RMS error; "max" = max abs error).  No non-finite output (the
largest down-projection output, 134, is far from f16's 65504).  The
down-projection is the stress case: its input (SiLU(gate)*up) has outliers
up to 173, so the running f16 sum carries large magnitudes and pure f16
accumulation reaches 1.1% RMS / 1.76 absolute error; promote stays within
2.4x of the f16-output floor everywhere.

## 4. Throughput (GPU timestamps, in-process interleaved A/B)

Protocol: `ml_bench ab` (one process, one context, ~300 ms 4096^3 burn-in,
per-round kernel order rotated, `gpu_ms` = median of 7 per-round GPU medians of
20 iterations after 5 warm-ups; round-0 sampled correctness check with the
accumulator-aware budget).  Speedups are ratios within the same process.

### 4a. Route level, a16 (llama prefill route): `auto` vs `auto@f16` vs `auto@promote`

Command (both GPUs):

```
ML_ROUNDS=7 ML_AB_KERNELS=auto,auto@f16,auto@promote ml_bench ab \
  qo,1,512,2048,2048,a16 qkv,1,512,2560,2048,a16 gu,1,512,5632,2048,a16 \
  dn,1,512,2048,5632,a16 s2k,1,2048,2048,2048,a16 s4k,1,4096,4096,4096,a16
```

`auto` = today's f32-accumulate route (SM-agnostic tile rule); `@f16` /
`@promote` = the same executor policy with the accumulator-aware tile rule.

**RTX 4060** (`results/ab-auto3-a16-4060.txt`):

| shape | f32 (route) | f16 acc (route) | promote (route) |
|---|---|---|---|
| 512x2048x2048 (q/o) | 0.161 ms, 26.6 TF/s (64x64) | 0.115 ms, 37.3 TF/s, **1.40x** (64x128) | 0.120 ms, 35.8 TF/s, **1.34x** |
| 512x2560x2048 (qkv) | 0.212 ms, 25.3 TF/s (128x64) | 0.150 ms, 35.8 TF/s, **1.41x** | 0.163 ms, 33.0 TF/s, **1.30x** |
| 512x5632x2048 (gate/up) | 0.434 ms, 27.2 TF/s (128x64) | 0.294 ms, 40.1 TF/s, **1.48x** | 0.325 ms, 36.3 TF/s, **1.34x** |
| 512x2048x5632 (down) | 0.437 ms, 27.0 TF/s (64x64) | 0.304 ms, 38.8 TF/s, **1.44x** | 0.327 ms, 36.1 TF/s, **1.34x** |
| 2048³ | 0.733 ms, 23.4 TF/s (128x128) | 0.434 ms, 39.6 TF/s, **1.69x** | 0.471 ms, 36.5 TF/s, **1.56x** |
| 4096³ | 5.127 ms, 26.8 TF/s (128x128) | 3.218 ms, 42.7 TF/s, **1.59x** (128x128) | 4.101 ms, 33.5 TF/s, **1.25x** |

42.7 TF/s is **above the 4060's f32-accumulate tensor peak** (24 SM x 2.46 GHz
x 256 FMA x 2 ~ 30 TF/s at boost clock): f32 accumulation *was* a hard ceiling.

**RTX 3070** (`results/ab-auto2-a16-3070.txt`; same binary as the 4060 tile
rule, 7 rounds):

| shape | f32 (route) | f16 acc | promote |
|---|---|---|---|
| 512x2048x2048 | 0.133 ms, 32.3 TF/s | 0.094 ms, 45.9 TF/s, **1.41x** | 0.086 ms, 49.8 TF/s, **1.55x** |
| 512x2560x2048 | 0.182 ms, 29.6 TF/s | 0.118 ms, 45.5 TF/s, **1.54x** | 0.152 ms, 35.4 TF/s, 1.20x |
| 512x5632x2048 | 0.350 ms, 33.8 TF/s | 0.251 ms, 47.0 TF/s, **1.39x** | 0.257 ms, 45.9 TF/s, **1.36x** |
| 512x2048x5632 | 0.370 ms, 31.9 TF/s | 0.243 ms, 48.7 TF/s, **1.52x** | 0.217 ms, 54.5 TF/s, **1.71x** |
| 2048³ | 0.563 ms, 30.5 TF/s | 0.320 ms, 53.7 TF/s, **1.76x** | 0.341 ms, 50.4 TF/s, **1.65x** |
| 4096³ | 3.779 ms, 36.4 TF/s | 2.709 ms, 50.7 TF/s, **1.39x** | 2.957 ms, 46.5 TF/s, **1.28x** |

(The 3070's f32-accumulate tensor peak is ~40.6 TF/s; f16 accumulation
reaches 54.5 TF/s.  On GA104 promote sometimes beats pure f16 at 64x128 —
lower occupancy (161 vs 80 registers) helps there; a per-GPU tile/occupancy
search is the natural follow-up, see §7.)

## 5. SASS evidence

Method: `ml_bench isa` (campaign/sass) — compile the routed pipeline into a
fresh `VkPipelineCache`, extract the NVuc ucode, `nvdisasm`, census.  Pipeline
creation only, no dispatch.  Case `q512,1,512,2048,2048,f16w` with `ML_KERNEL`
pinned per kernel (so the variant is the dispatch's interior/K-multiple one).

**RTX 3070 / SM86** (`$SP/f16acc/isa-3070/`; compile-only):

| kernel (a16 unless noted) | HMMA | regs | warps/SM (limiter) | hot-loop instrs | LDL/STL |
|---|---|---|---|---|---|
| m64n64 f32 | 128 x HMMA.16816.**F32** | 72 | 28 (regs) | 545 | 0/0 |
| m64n64 acc16 | 128 x HMMA.16816.**F16** | **48** | 36 (smem) | 502 | 0/0 |
| m64n64 acc16p | 128 x HMMA.16816.F16 | 80 | 24 (regs) | 504 | 0/0 |
| m128n64 f32 | 256 x HMMA.16816.F32 | 128 | 16 (regs) | 832 | 0/0 |
| m128n64 acc16 | 256 x HMMA.16816.F16 | **95** | 20 (regs) | 778 | 0/0 |
| m128n64 acc16p | 256 x HMMA.16816.F16 | 162 | 12 (regs) | 795 | 0/0 |
| m64n128 f32 | 256 x HMMA.16816.F32 | 155 | 12 (regs) | 777 | 0/0 |
| m64n128 acc16 | 256 x HMMA.16816.F16 | **80** | 24 (regs) | 807 | 0/0 |
| m64n128 acc16p | 256 x HMMA.16816.F16 | 161 | 12 (regs) | 824 | 0/0 |
| 128x128 f32 | 256 x HMMA.16816.F32 | 128 | 16 (regs) | 661 | 0/0 |
| 128x128 acc16 | 256 x HMMA.16816.F16 | **79** | 24 (regs) | 663 | 0/0 |
| 128x128 acc16p | 256 x HMMA.16816.F16 | 128 | 16 (regs) | 652 | 0/0 |
| a32 m64n64 f32 / acc16 / acc16p | F32 / F16 / F16 x128 | 72 / 55 / 80 | 28 / 36 / 24 | 638 / 577 / 579 | 0/0 |

* Every `_acc16` / `_acc16p` kernel emits `HMMA.16816.F16` and no `.F32`
  HMMA; the f32 kernels emit only `HMMA.16816.F32`.
* f16 fragments cut the register count 25-48% (m64n128: 155 -> 80), which
  lifts occupancy; that is why the optimum tile moves to 64x128/128x128.
* Promote keeps the f32 fragments live beside the f16 ones (+1.5x
  accumulator registers).  The 64x64 promote kernel additionally parks 13
  words/thread in shared memory (`STS [R1+...]` zero-init + LDS/STS at each
  promotion) — the driver's register-pressure relief, not an LDL/STL spill.
* The chunk loop is fully unrolled by the driver (4 K-tiles x 16 HMMA per
  promotion for 64x64).

**RTX 4060 / SM89**: `ml_bench isa` cannot extract the ucode on driver 595.80
(`zstd header: Frame header specified dictionary id 0x53FED98E that wasnt
provided` — the 4060 driver dictionary-compresses its pipeline-cache entries;
reported to the sass track), so SM89 evidence is the driver's own statistics
via `VK_KHR_pipeline_executable_properties`
(`ML_STATS_FILTER=coopmat examples/coopmat_occupancy`,
`results/occupancy-4060.txt`; compile-only on a second logical device):

| kernel (a16) | f32 regs | acc16 regs | acc16p regs | driver shared bytes (declared) f32 / acc16 / acc16p |
|---|---|---|---|---|
| 64x64 | 72 | **48** | 80 | 9,728 / 9,728 / **16,384** (9,728) |
| 128x64 | 128 | **95** | 162 | 16,384 / 14,848 / 14,848 (14,848) |
| 64x128 | 155 | **80** | 161 | 13,824 / 13,824 / 13,824 (13,824) |
| 128x128 | 128 | **79** | 128 | 18,944 / 18,944 / **44,544** (18,944) |

Register counts match SM86 exactly.  The driver-reported shared size exposes
the register parking seen in the SM86 SASS: where it exceeds the declared
size the compiler moved accumulator registers to shared memory (128x128
promote: +25,600 B = 25 words/thread, which is why promote gains least on
the 128x128 tile).

## 6. End to end (TinyLlama-1.1B-Chat f16, llama-ash)

Switch: `ML_COOPMAT_ACC=f16|promote` (read by `Executor::new`; llama-ash builds
its executor that way).  Only the prefill GEMMs change (a16 coopmat route);
decode GEMVs, attention and the LM head are untouched.

### 6a. pp512, fresh-process A/B (6 rounds, order f32,f16,promote then reversed)

Command (per run): `[ML_COOPMAT_ACC=f16|promote] llama_ash bench -m
tinyllama-1.1b-chat-v1.0.f16.gguf --pp 512 --tg 16` (fast Rust profile, glslc
-O on louise), `s4_e2e.sh`.  pp512 = 512 / wall time of one timed prefill after
a warm-up prefill + 128 decode steps; GPU = the prefill graph's timestamps.

| GPU | mode | GPU ms (median, range) | pp512 t/s (median, range) | vs f32 |
|---|---|---|---|---|
| RTX 4060 | f32 acc | 39.45 (39.23-39.53) | 12,882 (12,860-12,963) | 1.000 |
| RTX 4060 | **f16 acc** | 31.57 (31.32-31.66) | **16,066** (16,027-16,200) | **1.247** |
| RTX 4060 | f16 promote | 34.24 (33.79-34.36) | 14,836 (14,786-15,025) | 1.152 |
| RTX 3070 | f32 acc | 32.08 (31.47-32.11) | 15,897 (15,879-16,200) | 1.000 |
| RTX 3070 | **f16 acc** | 24.80 (24.52-24.80) | **20,543** (20,540-20,794) | **1.292** |
| RTX 3070 | f16 promote | 24.55 (24.54-24.57) | 20,728 (20,720-20,750) | 1.304 |

4060: `results/pp512-4060-v2.txt` (6 rounds).  An earlier 6-round run with the
f32 tile rule for all classes (`pp512-4060-v1-oldtiles.txt`) gave f16 1.216x /
promote 1.144x, so the accumulator-aware tiles add ~3%.  3070:
`results/pp512-3070-b1.txt`, **3 rounds only**: a second batch was contaminated
(the user started a game on the display GPU; runs 2x slower, file kept as
`pp512-3070-b2-CONTAMINATED.txt`) and all 3070 work stopped.

**vs llama.cpp CUDA b9503 (llama-bench -p 512 -n 0 -r 10 -ngl 99, one warm-up
invocation first, median of the 10 per-rep samples):**

| GPU | llama.cpp CUDA pp512 | tensor-ash f32 acc | tensor-ash f16 acc | tensor-ash promote |
|---|---|---|---|---|
| RTX 4060 | 13,684 (reps 13,396-14,045 after a 10,284 cold rep) | 12,882 (0.94x) | **16,066 (1.17x)** | 14,836 (1.08x) |
| RTX 3070 | 16,802 (reps 16,674-17,131 after a 12,931 cold rep) | 15,897 (0.95x) | **20,543 (1.22x)** | 20,728 (1.23x) |

(Method caveat: llama-bench times `llama_decode` of 512 tokens; llama-ash times
one prefill call; both exclude model load, both include the last-token LM head.)

### 6b. Greedy tokens and prefill logits

`llama_ash generate -m ... --ids <prompt> -n 32` per mode, prompts of 128/192
tokens (real text: history prose, Python code, a chat-template cooking
question, biology prose) and the 512-token synthetic bench prompt; every
prompt takes the a16 route (T >= 128, T % 64 == 0).  llama.cpp CUDA reference:
`llama-server` fed the SAME token ids (`/completion` with a token-array prompt,
`temperature 0, top_k 1, ignore_eos`), `prompts/llama_ref.py`.  4060:

| prompt | T | f16 vs f32 | promote vs f32 | f32 vs llama.cpp | f16 vs llama.cpp | promote vs llama.cpp |
|---|---|---|---|---|---|---|
| p1 prose | 128 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| p2 code | 192 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| p3 chat | 128 | 32/32 | 32/32 | 8/8 + EOS | 8/8 + EOS | 8/8 + EOS |
| p4 prose | 192 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |
| p5 synthetic | 512 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |

p3: every tensor-ash mode emits EOS (id 2) at step 8; llama-server with
`ignore_eos` masks EOS there, and without it (the first 3070 reference run)
it also stops with EOS at step 8 — so the streams agree up to and including
EOS.  All tensor-ash modes produce identical 32-token streams.

Prefill logits (32000, `LLAMA_ASH_LOGITS_OUT`) vs the f32-accumulate run:

| prompt | f16: max abs / RMS delta | promote: max abs / RMS delta | max abs logit | argmax | top-5 overlap (f16 / promote) |
|---|---|---|---|---|---|
| p1 | 0.063 / 0.014 | 0.016 / 0.0037 | 18.1 | same | 4/5 / 5/5 |
| p2 | 0.064 / 0.016 | 0.021 / 0.0044 | 26.0 | same | 5/5 / 5/5 |
| p3 | 0.077 / 0.015 | 0.020 / 0.0042 | 16.5 | same | 5/5 / 5/5 |
| p4 | 0.085 / 0.016 | 0.020 / 0.0041 | 20.3 | same | 5/5 / 5/5 |
| p5 | 0.091 / 0.021 | 0.015 / 0.0039 | 11.1 | same | 5/5 / 5/5 |

### 4b. Route level, a32 (f32 activations converted while staging; `f16w` cases), RTX 4060

`results/ab-auto3-f16w-4060.txt`, same protocol with `,f16w` cases:

| shape | f32 (route) | f16 acc | promote |
|---|---|---|---|
| 512x2048x2048 | 0.184 ms, 23.3 TF/s | 0.130 ms, **1.42x** | 0.127 ms, **1.45x** |
| 512x2560x2048 | 0.223 ms, 24.1 TF/s | 0.177 ms, **1.26x** | 0.182 ms, 1.23x |
| 512x5632x2048 | 0.470 ms, 25.1 TF/s | 0.358 ms, **1.31x** | 0.388 ms, 1.21x |
| 512x2048x5632 | 0.518 ms, 22.8 TF/s | 0.364 ms, **1.42x** | 0.360 ms, **1.44x** |
| 2048³ | 0.771 ms, 22.3 TF/s | 0.486 ms, **1.59x** | 0.554 ms, **1.39x** |
| 4096³ | 5.431 ms, 25.3 TF/s | 3.793 ms, **1.43x** | 4.600 ms, 1.18x |

### 4c. Per tile (why the tile rule changes), RTX 4060, a16

`results/ab-tiles-a16-4060.txt` (12 kernels, one process, 7 rounds; GPU ms):

| shape | 64x64 f32 / f16 / prom | 128x64 f32 / f16 / prom | 64x128 f32 / f16 / prom | 128x128 f32 / f16 / prom |
|---|---|---|---|---|
| 512x2048x2048 | 0.162 / 0.140 / 0.147 | 0.189 / 0.140 / 0.130 | 0.211 / 0.123 / 0.128 | 0.218 / **0.121** / 0.153 |
| 512x2560x2048 | 0.199 / 0.160 / 0.178 | 0.205 / **0.144** / 0.158 | 0.231 / 0.148 / 0.161 | 0.247 / 0.164 / 0.185 |
| 512x5632x2048 | 0.433 / 0.346 / 0.377 | 0.458 / 0.310 / 0.327 | 0.456 / **0.287** / 0.321 | 0.483 / 0.321 / 0.387 |
| 512x2048x5632 | 0.427 / 0.362 / 0.377 | 0.508 / 0.364 / 0.353 | 0.568 / **0.316** / 0.340 | 0.602 / 0.318 / 0.435 |
| 2048³ | 0.598 / 0.478 / 0.538 | 0.639 / 0.448 / 0.452 | 0.702 / **0.416** / 0.461 | 0.688 / 0.425 / 0.510 |
| 4096³ | 5.497 / 4.664 / 5.091 | 5.011 / 3.459 / 4.345 | 5.745 / 3.568 / 4.443 | 5.177 / **3.320** / 4.285 |

With f32 accumulation the small 64x64 tile wins on M=512 (register-limited
occupancy); with f16 accumulation the wide tiles win by 12-25%.  On the a32
path f16 accumulation on the *same* 64x64 tile is even slower (0.82-0.98x,
`ab-tiles-f16w-4060.txt`) while 64x128/128x128 f16 are 1.33-1.79x faster than
the f32 64x64 — the accumulator class and the tile must be chosen together.
The RTX 3070 per-tile table is `results/ab-tiles-a16-3070.txt` (there
128x64/64x128 f16 and promote are within ~7% of each other; the 4060 rule is
never >7% off the 3070 per-shape best).

## 7. Generator milestone M1 (spec-constant coopmat1 module)

Built as designed (design-gen-final §4): the coopmat body in `SPEC_TILE` mode
(spec ids 16-23), modules `matmul_f16w{,_a16}_coopmat_spec` + the accumulator
axis `matmul_f16w_a16_coopmat_spec_acc16` + a `K_UNROLL=8` body axis
`matmul_f16w_a16_coopmat_spec_ku8`; 15 configs per module (t1-t4 twins, n1-n11)
as `KernelSpec::spec` tails; `ML_GEN=1` (or an `ML_KERNEL` pin) builds them,
lazily.

**G0 (CPU): PASS.** Evaluated shared bytes equal the analytic formula for
every config and the literal twins (18,944 / 9,728 / 13,824 / 14,848)
(`generated_configs_are_legal_and_consistent`,
`generated_twins_match_the_literal_tiles_shared_bytes`); spirv-val clean;
glslc -O 0.09 s / 61 MB per module; spec-bounded `[[unroll]]` loops stay
rolled (12 `OpLoopMerge`).  C8 is fixed (the evaluator folds
`OpSpecConstantOp`).  Correctness: all 45+ generated kernels pass the f64
full-output gate (§3a).

**S1 (twin parity, +-2%, 9 interleaved rounds, 6 shapes, 4060): FAIL.**
(gen time / literal time - 1; `results/gen-a16-4060.txt`,
`results/gen-f16w-4060.txt`):

| twin | a16: q/o, qkv, gate/up, down, 1024³, 2048³ | a32: same shapes |
|---|---|---|
| t1 = 128x128 | +8.8, +14.0, +9.9, +4.6, +4.6, +8.6 % | +4.2, +14.0, +6.6, +1.0, +3.4, +5.6 % |
| t2 = 64x64 | -1.3, +0.0, +0.7, -1.4, +1.2, +2.6 % | -3.3, -5.6, -4.3, -2.3, +1.1, +2.2 % |
| t3 = 64x128 | +5.3, +1.8, +6.2, -1.1, +5.1, -2.9 % | **-13.8, -11.6, -9.4, -14.3, -12.4, -12.6 %** |
| t4 = 128x64 | +6.0, +3.4, +6.1, +2.5, +4.2, +6.2 % | -4.7, -2.3, +0.2, -5.5, -5.8, -2.0 % |

Only t2 (a16) holds parity.  The spec module is not a faithful twin in
*either* direction: a32 t3 is 9-14% FASTER than the hand-written 64x128.

SASS diagnosis (SM86, `$SP/f16acc/isa-3070-gen/`, compile-only):
* the spec-bounded fragment loops ARE fully unrolled by the driver: per
  K-tile the twin has the same 16/32 HMMA, the same LDSM.16.M88.4 /
  LDSM.16.MT88.4 mix, the same 6 LDG.E.128 + 6 STS.128 and 2 BAR — the
  per-K-tile instruction stream is structurally identical;
* the runtime K loop is not: the driver unrolls the literal wrappers' K loop
  **8x with uniform exit checks** (16 BAR per loop body) and leaves the spec
  module's loop rolled (2 BAR per body; ~7% more instructions per K-tile);
* register allocation differs: a16 t1 128 regs plus 25 words/thread parked in
  shared memory (`STS/LDS [R1+..]`) vs the literal's clean 128; a32 t3 125
  regs vs the literal's 168 (hence faster: 12 -> 16 warps/SM);
* no LDL/STL anywhere.

Per the design's S1 rule this falsifies "spec-first twins are free" for
coopmat on this driver; see the `K_UNROLL` rescue below before concluding
Tier C.

**S2 (search signal: best new config >= 3% faster than the routed tile on >= 2
of 4 prefill shapes, reproduced in a fresh process):**

| module | q/o | qkv | gate/up | down | verdict |
|---|---|---|---|---|---|
| a16, f32 acc | -4.3% (n10) | +3.0% (n1) | +4.5% (n10) | +4.1% (n1) | 3/4 in run 1 — but the best *literal* tile (64x64) matches it on gate/up/down, i.e. the win is the f32 tile rule mis-routing 128x64, not a new config |
| a32, f32 acc | +1.7% (n6) | +3.4% (n6) | +2.6% (n10) | +5.8% (n10) | 2/4 in run 1 |
| a16, f16 acc (acc axis) | +2.7% (n2/n3/n7) | +7.2% (t3) | -0% | +4.2% (n3) | 2/4 in run 1 |

(`results/gen-*.txt`; fresh-process reproduction: see §7a below.)
Observations: skew 0 (n9) is 1.7-1.8x slower (bank conflicts — the skew is
essential); BK=64 at 128x128 (n8) and the 4-warp 128x128 (n7) lose with f32
accumulators; 32x64 (n11) never wins.
