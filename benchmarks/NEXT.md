# Next ideas (from the measured campaign, not a new GPU sweep)

The 3070 is the gate for anything that can change large-square coopmat
or TinyLlama pp512/tg128.  This note is the campaign's dead-end list,
the remaining levers that still have a physical story, and the
architecture verdict from walking the tree without running more GPU
work.

## What the results say over time

Wins stacked in layers.  Early GEMM work paid for **load width** (BDA
then BDA_V4, +5–15% each).  Deep-K holes got a **two-stage split-K**,
not atomics.  Decode then stopped being a GEMM problem: dedicated row
/ column / outer GEMVs, fused stores (normed-A, rope-scatter), fused
split-K attention, prepared replay, and the on-GPU token loop.  The
last decode kernel that was *not* at the bandwidth floor was **lm_head
packing** (4060: 3759 → 511 µs, 94% of 272 GB/s).  Prefill is still
GEMM-bound (~96%); the 4060 hole was **occupancy on 512-row shapes**,
not tensor-core peak — 64×64 and 128×64 filled the short wave that
128×128 left idle.

Two numbers that looked like levers were not:

- T3's 7.7 µs/barrier was inferred from a 22-barrier elimination.  The
  slope-difference measurement is ~0.9 µs.  Barrier cutting is not a
  decode lever.
- CM2 plain GEMM loses 18–35% to coopmat1 on GA104.  Keep CM2 for
  fused-epilogue *library* callers; llama prefill keeps composed
  coopmat1 + Binary (the Binary pass is cheaper than the GEMM gap).

## Dead ends (do not retry without a new mechanism)

| experiment | result | why |
|---|---|---|
| Coopmat register-prefetch | −44% on 4096³ | occupancy cliff; extra regs live across MMA |
| Coopmat LDS double-buffer | 2048³ 6.4 TF/s on 4060 (single-buffer 22.7) | 37.9 KiB shared, 1 CTA/SM |
| Stream-K (FP32) | −6 to −75% | DP-flat kernel slower than BDA_V4 |
| CTA GROUP_M swizzle | −0.6% except 4096³ +1.5% | L2 already holds smaller shapes |
| Atomic split-K on K≤1024 | −17 to −53% | fill + contention, not atomic cost |
| CM2 plain GEMM vs coopmat1 | −18 to −35% | keep CM2 only for fused-epilogue callers |
| GEMV-chain as default | CM2 flash numerics scramble | `ML_DEVICE_SCOPE=1` stays opt-in |
| Packing q/o / k/v / down | 1.00–0.99× vs unpacked | already at the BW floor |
| v3 static double-buffer SMEM | lost everywhere vs BDA_V4 | driver already schedules v2 |

## What the model still says

- **Decode is weight-bandwidth-bound.** lm_head packing is the one
  GEMV that was *not* at the floor.  Everything else is launch
  overhead or already sequential.  Host work per token is one position
  u32 write and one token u32 read.
- **Prefill is GEMM-bound (~96%).** CUDA-parity needs ~33 TF/s average
  on real 512-row shapes.  4060 now: q/o 22.3, concat QKV 23.0, gate/up
  24.2, down 22.9 — occupancy was the hole, not tensor-core peak.
- **CM2 fused epilogue** is a library win vs SIMT demote, but llama
  prefill should keep composed coopmat1 + Binary.

## Next experiments (need 3070 to land)

1. **128×64 as the large-square default?** +8% on 4060 2048³
   (24.5 vs 22.7).  The 3070 4096³ 128×128 number (34.6 TF/s) is the
   veto.  One interleaved A/B on the 3070 decides it.
2. **SM-count-aware coopmat heuristic.** 96-tile threshold is 2 waves
   on 46 SM.  4060 has 24 SM; a `shaderSMCount` query would pick 64×64
   / 128×64 earlier without a 3070 regression if keyed by SM count.
3. **BK / WM retune of 128×64** on 512×5632 (the fattest prefill GEMM).
4. **ml-bench `cases` packed_b flag** so T2 can see packed lm_head.
5. **B-column offset into concat `w_qkv`** would drop packed q/k/v
   copies (~220 MB on TinyLlama).  Decode-only memory, not speed.
6. **Record a 4060 section in `thesis-expectations.toml`** so `ml_bench
   thesis` gates this GPU the way the 3070 is gated.
7. **Flash vs GEMM split of pp512** (info on 4060 is fine): confirms
   T5's "96% GEMM" after concat+pack.  If flash is now >10%, CM2 flash
   occupancy is back on the table.
8. **Odd-shape store path** (gameplan 17): the fully-odd cuBLAS gap is
   not a tile-choice problem.  Needs a different edge strategy, not
   more tiles.
9. **C ABI prepared/replay** (`ta_prepared_create` / run / destroy) if
   an Ollama-style integration is still the goal.  `OLLAMA_LLM_LIBRARY`
   is not a GEMM callback; a real backend is a ggml adapter.

## Architecture (walked, no new crates)

The workspace already splits at autonomy boundaries:

| package | why it is a crate |
|---|---|
| `tensor-ash` | Vulkan runtime, kernels, executor |
| `tensor-ash-capi` | FFI / panic barrier / C handles |
| `llama-ash` | GGUF + llama forward (one consumer of the runtime) |
| `ml-bench` | harness; must not hang deps off the library |
| `tensor-ash-test-support` | CPU fixtures, no Vulkan |

Candidates that look tempting and stay **modules**:

- **Elementwise / flash / GEMV-chain as `tensor-ash-ops`.** Same
  `Executor` slots, BDA layout, `plan_elementwise`, `VulkanContext`.
  A crate boundary is a circular dep or a public-internals dump.
- **GGUF parser.** 417 lines, llama-metadata-specific, one consumer.
- **Shader catalog.** Bound to `build.rs` `OUT_DIR` SPIR-V.

Shared machinery pulled out of the big files rather than new crates:

- `pipeline::create_pc_only_layout` — BDA matmul, split-K2, elementwise
- `pipeline::SPLITK2_SPIRV` — one include for load + tune-store hash
- `executor/cells.rs` — `PosBuffer` / `HostU32Buffer` over one `U32Cell`
- `HazardCursor` — census and recording cannot disagree on barriers
- `elementwise/pc.rs` — `pc_block!` for every GLSL push-constant mirror

Shaders are already a preprocessor family (`.comp` wrappers around
`.glsl` bodies).  Packed vs unpacked row GEMV is one `#define B_PACKED`.
That is the right grain; do not crate-split shaders.
