# SASS census of the routed kernels (RTX 3070 + RTX 4060)

What the NVIDIA driver actually emits for tensor-ash's shipped kernels,
extracted from the driver's own pipeline cache and counted
instruction by instruction, with the 128-bit control bits decoded.
Tool: `ml_bench isa` / `isa-analyze` / `isa-cache`
(`tools/ml-bench/src/bench/isa/`, usage in `tools/ml-bench/README.md`).
Per-kernel census files and metadata: `benchmarks/sass/`.

Branch `campaign/sass`, 2026-09-26. Hosts: desktop RTX 3070 (GA104,
46 SM, driver 595.99.02) and louise-pc RTX 4060 (AD107, 24 SM, driver
595.80). CUDA 12.9 `nvdisasm` on the desktop.

## Headline findings

1. **The driver emits byte-identical SASS on GA104 (sm_86, 595.99) and
   AD107 (sm_89, 595.80)** for all 11 probed routes (UCODE `cmp`-equal,
   same register counts). Every static number below holds for both
   GPUs; their performance differences come from SM count, clocks, L2
   (4 MB vs 24 MB) and DRAM, not from code generation.
2. **`glslc -O` vs `-Os` makes no difference after the driver**: the
   SPIR-V differs only in dead push-constant members / constant ids,
   and the UCODE is byte-identical for all 14 compared shaders,
   including the tm8_tn8 tiles whose `-O` compile needs >6 GB (see
   Q(a)). The evidence supports building everything with `-Os`.
3. **The aligned SIMT SGEMM loop is already 87-88% FFMA** (large_bda_v4:
   2048 FFMA of 2318 instructions per BK=32 iteration). Its gap to peak
   (3070: 12.1 of 20.3 TF/s) is not instruction overhead. What the
   census does show: every global load is consumed **34-86 static
   cycles** after issue (LDG.128 -> STS -> BAR -> FFMA: each tile is
   loaded, stored and only then used, no in-warp overlap of the next
   tile's loads with this tile's math), and occupancy is **16 warps/SM
   (33%)**, register-limited at exactly 128 registers.
4. **Coopmat kernels are tensor-pipe-bound by construction**: per loop
   iteration (8 K-tiles unrolled) a warp issues 737 instructions but
   needs 8192 tensor-pipe cycles (256 x HMMA.16816.F32 at 64 FMA/clk per
   SM sub-partition). Measured 34.7 TF/s (3070) and 25.1-25.7 TF/s
   (4060) at 4096^3 are ~85% of the nominal f16-in/f32-acc peaks
   (40.6 / 30.2).
   Global loads are consumed 14-64 cycles after issue (no prefetch
   distance); the loop is strictly LDG -> F2FP -> STS -> BAR -> LDSM ->
   HMMA -> BAR per K-tile.
5. **The odd shape 1023x1025x1027 pays ~31% more dynamic
   instructions** than 1024^3 for the same flops: scalar bounds-checked
   loads in *every* CTA (K%64 != 0 disables the vector path
   everywhere, N%4 != 0 disables vector B loads everywhere), a full
   64-deep K-tail pass for 3 remaining K values (2048 FFMA, 61/64
   multiplying zeros), scalar stores, and a 17th CTA column that
   computes one useful column. Measured: -15% TF/s (+18% time) on both
   GPUs. See Q(b).
6. **Decode GEMVs are not instruction-bound**: 44-54 instructions per
   512 B of weights per warp is ~12-14% of the SM issue capacity at the
   DRAM feed rate. lm_head (packed) runs at 96% (3070) / 97% (4060) of
   DRAM bandwidth.
7. **Why CM2 plain GEMM trails coopmat1 by 18-35% (a known dead end,
   now explained)**: the `f16w_cm2` kernel's tensor-addressed loads
   lower to per-element clamped scalar loads (LDG.U16 / LDG.32 with
   ISETP+SEL per element: 644 integer instructions per 64 HMMA), it
   uses 208 registers and 44 KB of driver-managed shared memory, so
   occupancy is 8 warps/SM (coopmat1: 16), and per warp its issue time
   is 48% of its tensor time (coopmat1: 9%).

## How to reproduce

The three driver scripts are checked in next to the artifacts
(`benchmarks/sass/run-isa.sh`, `run-isa2.sh`, `run-measure.sh`):

```console
# 4060 (louise): ISA probes = pipeline creation only, no dispatch
lgpu c-sass 'benchmarks/sass/run-isa.sh  ./target/release/ml_bench ~/out-4060 <spv-dir>'
lgpu c-sass 'benchmarks/sass/run-isa2.sh ./target/release/ml_bench ~/out-4060/emb'
lgpu c-sass 'benchmarks/sass/run-measure.sh ./target/release/ml_bench ~/out-4060/measure.csv'
# 3070 (desktop): same scripts under the GPU lock; census later
ML_ISA_NVDISASM=off gpu3070 benchmarks/sass/run-isa.sh  target/release/ml_bench out-3070 <spv-dir>
ML_ISA_NVDISASM=off gpu3070 benchmarks/sass/run-isa2.sh target/release/ml_bench out-3070/emb
gpu3070 benchmarks/sass/run-measure.sh target/release/ml_bench out-3070/measure.csv
# census (desktop, no GPU, CUDA 12.9 nvdisasm): artifacts from either host
target/release/ml_bench isa-analyze out-3070/main out-4060/main
# the CM2 probe
ML_KERNEL=f16w_cm2 ml_bench isa cm2al,1,512,512,512,f16w cm2odd,1,500,500,500,f16w
```

`<spv-dir>` holds `glslc --target-env=vulkan1.2 -O|-Os shaders/<s>.comp
-o <s>.O|Os.spv` for the five shaders of Q(a) (shaderc v2026.1; the
`-O` builds are byte-identical to the embedded ones). `run-isa.sh`
probes the 11 routed cases (`gateup,1,1,5632,2048,f16w
gateup_packed,1,1,5632,2048,packed lmhead,1,1,32000,2048,packed
q512,1,512,2048,2048,f16w gu512,1,512,5632,2048,f16w
sq4096h,1,4096,4096,4096,f16w sq4096,1,4096,4096,4096
odd,1,1023,1025,1027 sq1024,1,1024,1024,1024 oddk,1,1024,1024,1027
oddmn,1,1023,1025,1024`) and the `-O`/`-Os` swaps (`ML_KERNEL=<k>
ML_ISA_SPV=<spv> ML_ISA_TAG=O|Os`). `run-measure.sh` is `ML_ITERS=50
ML_WARMUP=10 ML_OUTPUT=csv ml_bench cases burn,1,4096,4096,4096 <the
same cases + gateup_b8,8,1,5632,2048,f16w>`, twice; the first case of
each repetition (4096^3 f32, 0.7-1 s of GPU work) is the clock
burn-in. Throughput numbers are GPU-timestamp medians of 50 samples.

## Method

**Extraction.** `ml_bench isa` routes a `cases` spec exactly like a
dispatch (`Executor::dispatch_info_for`, tuned winners and `ML_KERNEL`
honored), derives the bound specialization with the dispatcher's own
`KernelVariant::for_dispatch` (zero epilogue, `C = A@B`), and compiles
that pipeline through `MatmulPipeline::create_probe_pipeline` (the same
`create_specialized_pipeline_in` + `spec_data_for` the lazy dispatcher
path uses) into a fresh, empty `VkPipelineCache`. It then decodes
`vkGetPipelineCacheData`:

- cache blob = Vulkan header, `u32` entry count, entries of
  `key[16] | u32 len | "CPKV" | 0 | key[16] | u64 hash | len | 0 | value`;
  value = `u32 raw_size | zstd frame`, except one entry keyed
  `..._NVDICT_` whose value is a 2 MB **zstd dictionary** (magic
  `37 a4 30 ec`). AD107/595.80 compresses fresh entries against it;
  the 3070's long-lived production cache (254 entries, 43 MB) uses no
  dictionary. The parser walks all 254 production entries and ends
  exactly at the blob length.
- each decoded payload holds `NVDANVVM` + an **NVuc container**
  (`NVuc`, version 1, `u16` section count, `u64` size at +0x18, then
  32-byte section headers `type|size|offset|v0..v3`). Section types, as
  `nvdisasm` itself names them: 0x01 UCODE (raw SASS), 0x03 REG
  (allocated, requested registers), 0x17 STACK, 0x18 BARRIER,
  0x21 STRING_TABLE, 0x37 DEBUG_FRAME_TABLE, 0x39 SYMBOL_TABLE,
  0x3a DEBUG_FUNCTION_MAP, 0x3c SCRATCH, 0x45 PROGRAM_INFO, 0x1204
  CTAID_INFO. The UCODE section gives exact code boundaries (the old
  "offset 8" trick worked because `nvdisasm` understands this container
  natively and prints it as `cuVN`).
- The payload tail (after the container) is a driver knob/hash dump;
  the SPIR-V is not stored.

**Validation of the probe.** For 10 of 11 routes the probe's UCODE is
byte-identical to an entry of the 3070's persisted production cache
(`in_production_cache = yes`; the 11th, `oddk`, was simply never
dispatched there before), i.e. the probe compiles exactly what the
dispatcher runs. Recompiling with `CAPTURE_STATISTICS |
CAPTURE_INTERNAL_REPRESENTATIONS` does not change the code
(`capture_flags_code = identical`, all routes, both GPUs).

**Driver statistics** (`VK_KHR_pipeline_executable_properties`, opt-in
via `ML_PIPELINE_EXEC_INFO=1`): NVIDIA reports exactly five, for one
executable `CS`: Register Count (always equals the NVuc REG section),
Binary Size (= UCODE bytes), Stack Size (0), Local Memory Size
(68719476736 = 2^36 for every probed kernel, all of them spill-free:
not a usable spill signal; count LDL/STL instead), Shared Memory
Size (declared bytes, sometimes rounded up: large_bda_v4 33,280 -> 33,792,
coopmat_m128n64 14,848 -> 16,384, row GEMVs 4,096 -> 4,352; or
driver-managed: f16w_cm2 declares none and gets 44,032). **No
internal representations** are exposed.

**Control bits** (bits 105-125 of each 128-bit instruction: stall[4],
yield[1], write scoreboard[3], read scoreboard[3], wait mask[6],
reuse[4]; layout per CuAssembler `CuControlCode`/`CuSMVersion`,
maxas, Jia et al. arXiv:1804.06826). Verified on data: across all 254
production kernels (~1.5 M instructions) **0** fixed-latency ALU
instructions (FFMA/IADD3/LOP3/ISETP/...) set a write scoreboard, 94-100%
of variable-latency register writers (LDG/LDS/LDSM/S2R/...) do (the
exceptions are in-order LDS groups covered by the last load's
scoreboard), and every NOP after EXIT decodes to CuAssembler's pad
`B------:R-:W-:Y:S00`.

**Census metrics.** Main loop = innermost backward-branch range with
the most FFMA/HMMA; ties go to the leanest body (the row GEMVs compile
four equal-math K-loops: normed-A x vector/scalar B; the leanest is the
plain vector path). "Static stall" = sum of stall fields over one
iteration (issue cycles of one warp ignoring scoreboard waits);
"excess" = that sum minus the instruction count. "LDG->use" = static
cycles from a load to the first instruction that waits on its
scoreboard (in-warp latency cover). Static counts are per iteration of
the loop *body as laid out*: a loop with runtime if/else (e.g. `oddmn`)
counts both paths.

**Occupancy model.** CC 8.6 / 8.9 per-SM limits (1536 threads, 48 warps,
16 / 24 blocks, 64 K registers allocated per warp in 256-register units
per sub-partition, 100 KB shared at the maximum carveout plus 1 KB
reserved per block). The Vulkan driver picks the carveout, so this is
an upper bound. Here no kernel is block-cap-limited, so the 3070 and
4060 numbers coincide.

## Per-kernel census (identical SASS on both GPUs)

Route = kernel `v<variant>` (v14 = interior + K-multiple, v6 = interior
only, v10 = K-multiple only, v2 = fully bounds-checked; all with
alpha = 1, no accumulate). Widths are `count x bits`. Main-loop numbers
are per loop iteration of one thread/warp.

| case | route | regs | thr | smem B (drv) | occupancy warps/SM | instrs | math | LDG | STG | LDS / LDSM | STS | BAR | spill |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode gate/up 1x5632x2048 f16w | f16w_row_bda_k16_v2 v6 | 35 | 512 | 4,352 | 48 (100%, threads/regs) | 883 | 73 FFMA | 32x16 72x32 | 2x32 | 35 LDS | 11 | 12 | 0 |
| decode lm_head 1x32000x2048 packed | f16w_row_bda_k16_v2_packed v6 | 30 | 512 | 4,352 | 48 (100%, threads) | 527 | 41 FFMA | 48x32 | 2x32 | 35 LDS | 11 | 12 | 0 |
| prefill 512x2048x2048 f16w | f16w_coopmat_m64n64 v14 | 72 | 128 | 9,728 | 28 (58%, regs) | 793 | 128 HMMA | 48x128 | 32x32 | 64 LDSM.x4 | 32x128 | 16 | 0 |
| prefill 512x5632x2048 f16w | f16w_coopmat_m128n64 v14 | 128 | 128 | 16,384 | 16 (33%, regs) | 1254 | 256 HMMA | 80x128 | 64x32 | 96 LDSM.x4 | 48x128 | 16 | 0 |
| 4096^3 f16w | f16w_coopmat_aligned (128x128) v14 | 128 | 256 | 18,944 | 16 (33%, regs) | 999 | 256 HMMA | 48x128 | 64x32 | 96 LDSM.x4 | 32x128 | 16 | 0 |
| 4096^3 f32 | large_bda_v4 v14 | 128 | 256 | 33,792 | 16 (33%, regs = smem) | 2473 | 2048 FFMA | 8x128 | 16x128 | 130x128 | 16x32 4x128 | 2 | 0 |
| 1024^3 f32 | m128n64k64_bda_v4 v14 | 118 | 256 | 49,664 | 16 (33%, regs = smem) | 2491 | 2048 FFMA | 12x128 | 8x128 | 194x128 | 32x32 4x128 | 2 | 0 |
| 1023x1025x1027 f32 | m128n64k64_bda_v4 v2 | 128 | 256 | 49,664 | 16 (33%, regs = smem) | 5982 | 4096 FFMA | 96x32 | 64x32 | 388x128 | 64x32 8x128 | 4 | 0 |
| (isolate) 1024x1024x1027 | m128n64k64_bda_v4 v6 | 128 | 256 | 49,664 | 16 (33%) | 5696 | 4096 FFMA | 96x32 | 8x128 | 388x128 | 64x32 8x128 | 4 | 0 |
| (isolate) 1023x1025x1024 | m128n64k64_bda_v4 v10 | 120 | 256 | 49,664 | 16 (33%) | 3275 | 2048 FFMA | 48x32 12x128 | 64x32 | 194x128 | 64x32 8x128 | 2 | 0 |

Main loops:

| route | loop instrs / iter | K per iter | math / density | int+uniform | static stall (excess) | scoreboard waits | LDG->use median (min-max) cycles | pipe bounds per warp-iter |
|---|---|---|---|---|---|---|---|---|
| row_bda_k16_v2 (vector, plain) | 54 | 4 k-steps | 8 FFMA + 8 HADD2.F32 / 14.8% | 29 (54%) | 84 (30) | 7 | 33 (18-42) | issue 54 |
| row_bda_k16_v2_packed | 44 | 4 k-steps | 8 FFMA + 8 HADD2.F32 / 18.2% | 19 (43%) | 83 (39) | 5 | 34 (19-36) | issue 44 |
| coopmat_m64n64 | 638 | 8 K-tiles (256) | 128 HMMA / 20.1% | 240 (38%) | 2291 (1653) | 101 | 24 (14-64) | issue 638, tensor 4096 |
| coopmat_m128n64 | 976 | 8 K-tiles (256) | 256 HMMA / 26.2% | 295 (30%) | 3647 (2671) | 133 | 47 (27-66) | issue 976, tensor 8192 |
| coopmat_aligned 128x128 | 737 | 8 K-tiles (256) | 256 HMMA / 34.7% | 197 (27%) | 3221 (2484) | 90 | 22 (14-43) | issue 737, tensor 8192 |
| large_bda_v4 | 2318 | BK=32 | 2048 FFMA / 88.4% | 42 (1.8%) | 2537 (219) | 135 | 46 (34-68) | issue 2318, FP32 2048 |
| m128n64k64_bda_v4 v14 | 2358 | BK=64 | 2048 FFMA / 86.9% | 51 (2.2%) | 2563 (205) | 195 | 62 (52-86) | issue 2358, FP32 2048 |
| m128n64k64_bda_v4 v2 (odd) | 2738 | BK=64 | 2048 FFMA / 74.8% | 350 (12.8%) | 3134 (396) | 237 | 198 (42-490) | issue 2738, FP32 2048 |

Notes per family:

- **Row GEMVs.** Four K-loops are compiled (normed-A x f16vec2/scalar
  B), unrolled 4x by the driver; the plain vector loop issues per
  4 k-steps 4 broadcast A loads + 4 f16vec2 B loads (LDG.32 each), 8
  `HADD2.F32` half->float conversions, 8 FFMA and 29 integer
  instructions (address math per load). The other 12 BAR / 35 LDS are
  the 16-slice partial reduction and the (unused here) RMSNorm
  preamble. At the DRAM feed rate (3070 ~5 B/clk/SM, 4060 ~4.5
  B/clk/SM) a warp's 512 B of weights per iteration take ~100-115
  cycles to arrive while its 44-54 instructions need ~13 cycles of the
  SM's 4 issue slots: issue is ~12% utilized, so the instruction mix
  is not the limiter. Packing removes 10 instructions/iteration (no
  per-load column index math) and 356 kernel instructions, but not
  bandwidth.
- **Coopmat (KHR cooperative matrix, 16x8x16 HMMA).** The driver
  unrolls the K loop 8x (K=256 per loop iteration, 2 `BAR.SYNC` per
  32-deep K-tile). Per K-tile per warp: 6 LDG.128 -> 8 `F2FP.PACK_AB`
  (A f32 -> f16) -> 4 STS.128 -> BAR -> 12 LDSM.16.(M|MT)88.4 -> 32 HMMA
  (stall S08 between independent HMMAs) -> BAR -> exit test (numbers
  for the 128x128 tile; 64x64: 6 LDG / 8 LDSM / 16 HMMA, 128x64: 10
  LDG / 16 F2FP / 6 STS / 12 LDSM / 32 HMMA). Loads
  for tile k+1 are issued after tile k's MMAs: there is no in-warp
  prefetch, so the 4 warps/SM-sub-partition (2 CTAs/SM for 128x128) are
  what hides global latency. HMMA density looks low (20-35%) but per
  warp the tensor pipe needs 11x (128x128), 8.4x (128x64) and 6.4x
  (64x64) more cycles than issue, so these loops are tensor-bound, not
  issue-bound. The coopmat shaders ignore spec constants 2/3
  (`INTERIOR_ONLY`/`K_MULTIPLE`): v10 and v14 compile to the same
  UCODE.
- **SIMT SGEMM (BDA_V4).** 87-88% FFMA density with FFMA stall S01 and
  reuse flags on ~65% of instructions; A and B fragments come in as
  LDS.128 (4 per k-step for 128x128 TM=TN=8, 3 for 128x64 TM=8 TN=4). 62 `MOV` per iteration in large_bda_v4 are
  the register-double-buffer rotation. Occupancy is 16 warps/SM,
  register-limited at exactly 128 registers (the compiler's requested
  cap is 255: 128 is the 2-CTA/SM point for 256 threads) and, for
  large_bda_v4 / m128n64k64, shared-memory-limited at 2 CTAs too.
  Global loads for the next tile are consumed 34-86 cycles after issue
  (LDG.128 -> STS -> BAR): the tile loop is load-then-compute inside
  every warp, i.e. exactly the cp.async/`LDGSTS`-shaped hole the PTX
  track is after (no `LDGSTS`/`DEPBAR` appears in any kernel).

### CM2 GEMM (`f16w_cm2`, the fused-epilogue tensor-core route)

Probed with `ML_KERNEL=f16w_cm2 ml_bench isa cm2al,1,512,512,512,f16w`
(and an unaligned 500^3: same UCODE, the tensor-layout clamping is
unconditional). It is the 208-register kernel that shows up in the
3070's production cache (`in_production_cache = yes`).

| | f16w_cm2 | f16w_coopmat_aligned (coopmat1) |
|---|---|---|
| threads / regs / shared (driver) | 128 / 208 / 44,032 B | 256 / 128 / 18,944 B |
| occupancy | 8 warps/SM (regs and smem) | 16 warps/SM |
| main loop | 992 instrs per 64 HMMA (K 64) | 737 per 256 HMMA (K 256) |
| global loads / iter | 32 LDG.U16 + 64 LDG.32, each clamped (195 ISETP, 96 SEL) | 48 LDG.128 |
| shared traffic | 48 STS + 64 LDS.64 + 8 LDSM (driver staging) | 32 STS.128 + 96 LDSM.x4 |
| integer share of loop | 65% | 27% |
| issue / tensor cycles per warp-iter | 992 / 2048 (48%) | 737 / 8192 (9%) |

With half the warps and issue at half the tensor time, latency is no
longer hidden: the measured 18-35% deficit is the expected outcome, and
it is a codegen property of `coopMatLoadTensorNV` with clamped layouts on
this driver, not of the MMA path.

## Measured throughput (routed kernels)

GPU-timestamp medians of 50 iterations (commands above). 3070: one
run (17:58), two repetitions. 4060: two runs an hour apart (17:25,
18:24), two repetitions each. GB/s counts weight bytes only (K*N*2).

| case | route | RTX 3070 | RTX 4060 run 1 | RTX 4060 run 2 |
|---|---|---|---|---|
| gate/up 1x5632x2048 f16w (23 MB) | row_bda_k16_v2 | 55.9 / 56.1 us = 412 / 411 GB/s (92% of 448) | 24.4 / 24.5 us = 944 / 940 GB/s | 24.5 / 24.7 us = 940 / 935 GB/s |
| gate/up x8 independent weight sets (185 MB) | row_bda_k16_v2 | 434.2 / 433.4 us = 425 / 426 GB/s (95%) | - | 698.8 / 698.3 us = 264 / 264 GB/s (97% of 272) |
| gate/up packed | row_bda_k16_v2_packed | 56.9 / 56.0 us | 24.1 / 24.2 us | 24.4 / 24.7 us |
| lm_head 1x32000x2048 packed (131 MB) | row_bda_k16_v2_packed | 304.1 / 303.6 us = 431 / 432 GB/s (96%) | 495.3 / 495.4 us = 265 / 265 GB/s (97%) | 495.5 / 495.1 us = 265 / 265 GB/s |
| 512x2048x2048 f16w | coopmat_m64n64 | 29.43 / 31.21 TF/s | 24.34 / 24.28 | 22.19 / 22.23 |
| 512x5632x2048 f16w | coopmat_m128n64 | 32.44 / 34.42 TF/s | 26.44 / 26.35 | 24.14 / 24.06 |
| 4096^3 f16w | coopmat_aligned | 34.64 / 34.70 TF/s | 25.33 / 25.12 | 25.59 / 25.70 |
| 4096^3 f32 | large_bda_v4 | 12.08 / 11.99 TF/s | 8.50 / 8.50 | 8.50 / 8.50 |
| 1024^3 f32 | m128n64k64_bda_v4 v14 | 11.07 / 11.21 TF/s | 7.63 / 7.94 | 7.99 / 7.99 |
| 1023x1025x1027 f32 | m128n64k64_bda_v4 v2 | 9.51 / 9.37 TF/s | 6.50 / 6.77 | 6.80 / 6.79 |
| 1024x1024x1027 f32 | m128n64k64_bda_v4 v6 | 9.98 / 9.79 TF/s | 7.09 / 7.07 | 6.98 / 7.08 |
| 1023x1025x1024 f32 | m128n64k64_bda_v4 v10 | 10.11 / 9.93 TF/s | 7.31 / 7.30 | 7.20 / 7.26 |

The 4060's 24 MB L2 holds a single 23 MB gate/up weight matrix across
benchmark iterations, so its ~940 GB/s is an L2 number; with eight
independent weight sets (what decode actually streams: every layer from
DRAM) the same kernel runs at 264 GB/s = 97% of DRAM bandwidth, and at
95% on the 3070. The 4060's 512-row prefill shapes moved 9% between
runs (24.3 -> 22.2, 26.4 -> 24.1 TF/s) while every other case held
within ~5%: run-to-run clock state on that shape class; run 2 matches
the numbers in `NEXT.md` (22.3 / 24.2).

## Q(a): does `glslc -O` vs `-Os` change the SASS?

No. Same shader, same route and specialization, SPIR-V swapped via
`ML_ISA_SPV` (both compiled by shaderc v2026.1 on the desktop):

| shader | route | SPIR-V -O / -Os | UCODE -O = -Os | regs | GPUs |
|---|---|---|---|---|---|
| matmul_f32_large_bda_v4 | sq4096 v14 | 340,540 / 340,388 B, differ | identical (2b06434275eb5ff1, 39,808 B) | 128 = 128 | 3070 + 4060 |
| matmul_f32_k64_bda_v4_tm8_tn4 | 384^3 v14 | 332,924 / 332,772 B | identical (b8ad218c4c351725) | 121 = 121 | 3070 + 4060 |
| matmul_f16w_coopmat_aligned | 4096^3 v14 | 8,640 / 8,400 B | identical (dfd5ce7796a5ee0f) | 128 = 128 | 3070 + 4060 |
| matmul_f16w_coopmat_m128n64 | 512x5632x2048 v14 | 9,920 / 9,680 B | identical (f56e3841605cd6b2) | 128 = 128 | 3070 + 4060 |
| matmul_f32_m128n64k64_bda_v4 | 1023x1025x1027 v2 | 312,604 / 312,452 B | identical (8e65fcb143003cd5) | 128 = 128 | 3070 + 4060 |
| matmul_f32_m128n64k64_bda_v4 | 1024^3 v14 | (same pair) | identical (37be2ed94fd5cc14) | 118 = 118 | 3070 + 4060 |

`spirv-diff` shows why: `-Os` strips unused push-constant members and
unused constants, which renumbers member indices in `OpAccessChain`;
the instruction stream is otherwise the same (the kernels unroll with
`[[unroll]]` in the front end, so spirv-opt's `-O`-only passes have
nothing left to do). The driver's NVVM backend re-optimizes from
there and lands on the same binary.

**The heavy shaders too.** The desktop's LOWMEM build falls back to
`-Os` for the nine shaders whose `glslc -O` needs >2.6 GB (the
tm8_tn8 / tm16_tn4 / BK=64 tiles, `matmul_f32`, ...; up to >6 GB for
tm8_tn8), while louise's build uses `-O` for all. Because the SASS is
byte-identical across the two GPUs (11/11 routes above, and the
control below), comparing each host's embedded build answers `-O` vs
`-Os` for them without another multi-GB compile
(`ML_KERNEL=<k> ml_bench isa sq1024,1,1024,1024,1024` on both hosts,
metadata in `benchmarks/sass/opt-embedded/`):

| kernel (1024^3, v14) | desktop SPIR-V (-Os) | louise SPIR-V (-O) | UCODE | regs |
|---|---|---|---|---|
| large_bda_v4 (control, -O on both) | 0ad7f90ab399fb81 | 0ad7f90ab399fb81 | same (2b06434275eb5ff1) | 128 |
| k64_bda_v4_tm8_tn8 | 5ee677663e4d1861 | 451a3481f5fbf66b | same (cd8ac9e5b7fbc99b) | 236 |
| m128n64k64_bda_v4_tm8_tn8 | 8c8f2ba5f7742a62 | 14f1a072ad75a208 | same (30038e173f645857) | 197 |
| m128n64k64_bda_v4_tm16_tn4 | d9dbd5b19b61e213 | b5fea8924101ced3 | same (24dc224485487ca0) | 196 |
| m128n64k64_bda | 6c3e714a5db04f91 | 39f156681d7938ea | same (90858b19cc09d842) | 125 |
| large_bda | 5b66d28b02139ffc | 1a9813b5403cbf8e | same (6aa7e51347549697) | 144 |
| large_128 (`matmul_f32`, descriptors) | e39e0fb696b6d1fd | 32881e324300b06a | same (3efa65b894f41ffe) | 128 |
| m128n64k64 (descriptors) | 08a3fd6e5c4c464f | 32a287666a3d7d46 | same (7b5ceea12f53bb22) | 128 |
| m256n64 | 601cb86cbc31d652 | 10b70cabc168441e | same (82bc7a9c5a6075cd) | 165 |
| m64n128k64 | 8ed1409285c311ec | de4e8c326161d4a0 | same (ede074061c2f4305) | 128 |

Verdict: all 14 compared shaders (15 routes: 5 shaders swapped directly
on both GPUs, 9 via the cross-host builds) produce identical SASS from
`-O` and `-Os` SPIR-V.
Moving the whole build to `-Os` (~140 MB peak instead of >6 GB, no
LOWMEM fallback asymmetry between hosts) is safe on this driver
family. Keep `ml_bench isa` hash comparisons as the regression check
when the driver or glslc changes; the flash-attention shaders
(`op_flash_attention_*dh128`, also in the fallback list) were not
compared here.

## Q(b): where do the odd shape's extra instructions go?

Same kernel (m128n64k64_bda_v4, 128x64 tile, BK=64), 1023x1025x1027
(v2) vs 1024^3 (v14). Static code by region (instructions):

| region | 1024^3 v14 | odd v2 | what the odd code does |
|---|---|---|---|
| prologue | 79 | 62 | |
| K-loop body (per BK=64) | 2358 | 2738 (+380, +16%) | 48 scalar LDG.32 instead of 12 LDG.128, each with its own bounds test and 64-bit address: int/ctl 62 -> 406 |
| K-tail (after loop) | 0 | 2837 | a second copy of the whole 64-deep block (2048 FFMA, 194 LDS.128, 48 bounds-checked LDG) for K % 64 = 3 |
| epilogue | 54 (8 STG.128) | 345 (64 STG.32, two paths) | `!INTERIOR_ONLY`: even full CTAs store scalar, N % 4 != 0 |

Dynamic estimate per CTA: 1024^3 = 79 + 16 x 2358 + 54 = 37.9 k
warp-instructions; odd = 62 + 16 x 2738 + 2837 + ~170 = 46.9 k (+24%).
The grid grows from 8x16 = 128 to 8x17 = 136 CTAs (+6%; the 17th CTA
column computes one valid column), so the whole odd GEMM issues ~31%
more warp-instructions for 0.3% more useful flops; measured time is
+18% (3070 0.2265-0.2298 vs 0.1916-0.1940 ms; 4060 +18% as well), i.e.
the aligned kernel is not purely issue-bound either.

Where it comes from, in the GLSL (`matmul_bda_v4_kernel.glsl`):

1. **Loads (the dominant term).** With `K_MULTIPLE=false` the K-loop
   takes the `else` branch — `load_a_tile_scalar` + `load_b_tile_scalar`
   — in **every** CTA, not just edge CTAs. Each 4-byte load costs ~8
   integer instructions: index `IADD3`, `SHF` (row = idx / BK), `LEA`,
   `ISETP.LT.U32.OR` bounds test, predicated `IMAD` (row * K), `IADD3`,
   `LEA` + `IADD3.X` (64-bit address), `@!P MOV` (zero fill). These are
   loop-invariant per thread but re-materialized every iteration: the
   kernel sits at the 128-register cap.
2. **K tail.** `has_k_tail` runs the full BK=64 unrolled block once
   more for 3 valid k values: +6% FFMA work on a 17-iteration K loop,
   61/64 of it multiplying zeros.
3. **Stores.** `!INTERIOR_ONLY` routes even full interior CTAs to scalar
   `STG.32` (the vec4 store would need N % 4 == 0).
4. **Isolates.** K-only odd (1024x1024x1027, v6) has the same 2738
   scalar loop and K tail but STG.128 stores and no extra CTA column:
   9.98 / 9.79 TF/s (3070) vs 11.07 / 11.21 aligned, i.e. scalar loads +
   K tail cost ~11%; the fully odd shape adds scalar stores and the 17th
   CTA column for another ~5% (9.51 / 9.37). M/N-only odd
   (1023x1025x1024, v10) keeps vector A loads in full CTAs, but **B
   stays scalar everywhere** because `b_rows_packet_aligned = (N & 3) ==
   0` is false for N = 1025; no K tail, scalar stores: 10.11 / 9.93
   TF/s (-10%). The 4060 shows the same ordering (run 2: 7.99 aligned,
   7.23 M/N-odd, 7.03 K-odd, 6.80 fully odd).

What would close it (for the odd-shape / kernel-generator tracks):
peel the K remainder into a masked tail of `K % BK` k-steps (not a full
BK pass); give full-but-misaligned rows a load path whose bounds test
is per row, not per element (or stage misaligned operands through a
padded copy when the GEMM is reused, e.g. weights); keep vector stores
for full CTAs whose row starts are 16-byte aligned; hoist the
per-thread load index math (needs registers: the kernel is at 128).

## Other driver facts observed

- NVIDIA's `Registers requested` (NVuc REG word 2) is 255 for 128/256-
  thread CTAs and 128 for the 512-thread row GEMVs (the per-CTA
  hardware limit). The routed 256-thread kernels land at or just below
  128 (118-128: the 2-CTA/SM point) without spilling; the heavier
  catalog tiles probed for Q(a) do not (large_bda 144, m256n64 165,
  m128n64k64 tm16_tn4 196 / tm8_tn8 197, k64 tm8_tn8 236 registers:
  1 CTA/SM for 256 threads).
- None of the 11 kernels contains `LDGSTS`, `DEPBAR`, `LDL`/`STL` or
  `SHFL`. The only `CALL` in each GEMM loop is `@P CALL.REL.NOINC
  <after the loop>` right before the back-branch `BRA <loop start>`:
  the driver's loop-exit idiom, not a real call.
- The capture flags of `VK_KHR_pipeline_executable_properties` do not
  change code; no internal representations are exposed.

## Artifacts

`benchmarks/sass/` (from the 3070 run; the 4060's UCODE is
byte-identical, its metadata differs only in device/driver/arch):

- `<case>.<kernel>.v<variant>.sm86.census.txt` — whole-kernel and
  main-loop mix, loops, load->use distances, occupancy, control-bit
  sanity; `.meta` — route, spec constants, SPIR-V/UCODE hashes, driver
  statistics; for the 11 routes above plus `cm2al.f16w_cm2`.
- `*.loop.sass` — annotated main loops (`[wait:R:W:yield:stall]` per
  instruction) for the two decode GEMVs and the 128x128 coopmat kernel.
- `opt/` — `.meta` of the direct `-O`/`-Os` swaps on both GPUs;
  `opt-embedded/` — `.meta` of the cross-host embedded-build comparison.

Everything else (full `.sass`, `.nvuc`, raw `.vkcache` probe caches,
the SIMT loop listings of 150-180 KB each) regenerates with the
commands above.
