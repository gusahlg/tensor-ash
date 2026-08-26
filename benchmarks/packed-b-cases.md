# Packed-B cases, T2, and the C ABI

What landed, why, and how to GPU-validate later. No GPU work was run
for this change: the 3070 and louise's 4060 were busy, and this note
is the deferred gate.

## What

Packed-B decode GEMVs (`[N/tile_n][K][tile_n]`, sequential K-walk)
already existed in the library and llama-ash, but two callers could
not opt in:

1. **ml-bench `cases`** only accepted `,f16w`. T2 therefore measured
   the unpacked row-major GEMV, so lm_head (N=32000, 64 KiB stride)
   could not show the 7.4x packed win that made TinyLlama's head hit
   94% of device bandwidth.
2. **C ABI** forced `packed_b: false` on every `ta_matmul_op`, with
   no pack helper.

This change wires the existing path through, without new kernels.

- `ml-bench cases` flags: `label,b,m,n,k[,f16w][,packed]`. `packed`
  implies f16w and requires B=1, M=1, N a multiple of 32.
- T2 packs k/v, gate/up, and lm_head (llama-ash does); q/o and down
  stay unpacked (already at the unpacked bandwidth floor).
- C ABI: `ta_f16w_row_tile_n`, `ta_pack_f16w_row_tiles` (host-only),
  `ta_matmul_op.packed_b`, `ta_dispatch_info_packed`.
- Rust: `Executor::dispatch_info_packed(n, k)` for the fail-closed
  packed row-GEMV route.

## Why

T2's claim is "every decode GEMV class runs at ≥ 80% of its bandwidth
floor." Unpacked lm_head is ~13% of that floor on the 4060 (3759 µs
vs 511 µs packed). Measuring the production layout is the only way
the thesis can pass for that class. C callers otherwise cannot use
the packed kernels at all.

## CPU tests (done)

```console
cargo test -p tensor-ash-capi --lib
cargo test -p ml-bench --lib
cargo test -p tensor-ash --lib pack_tiles
```

Parser, host pack, C ABI error paths, and tile-n agreement do not
need a GPU.

## GPU validation (later)

Do **not** run this on the desktop RTX 3070 or louise's RTX 4060
while a training job owns the card. The only allowed GPU is
nixos-server's GTX 1650, which also runs the live Discord bot — tiny
shapes, no hammering, never touch `sighurt-llm.service` /
`discord-bot.service`.

```bash
ssh nixos-server
export VK_ICD_FILENAMES=$(ls -d /nix/store/*mesa-*/share/vulkan/icd.d 2>/dev/null|head -1)/nouveau_icd.x86_64.json

# Tiny packed vs unpacked pair (N=256, K=128). Expect both to complete
# and the packed kernel name to contain `packed`.
cargo run --release -p ml-bench -- cases \
  kv_unpacked,1,1,256,128,f16w \
  kv_packed,1,1,256,128,packed

# When a 3070/4060 is free, the real T2 gate (lm_head is the one
# that used to be 7.4x off the floor unpacked):
target/release/ml_bench thesis t2
```

Correctness for packed vs unpacked vs CPU is already in
`tests/correctness/f16.rs` (`packed_row_gemv_matches_unpacked_and_cpu`);
re-run the ignored GPU suite on a free card with `--test-threads=1`.
