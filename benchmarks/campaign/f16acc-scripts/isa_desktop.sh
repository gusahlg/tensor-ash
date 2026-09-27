#!/usr/bin/env bash
# Compile-only ISA probes on the desktop (no dispatch): acc variants.
set -u
BIN=$1; OUT=$2
for k in f16w_a16_coopmat_m64n64 f16w_a16_coopmat_m64n64_acc16 f16w_a16_coopmat_m64n64_acc16p \
         f16w_a16_coopmat_m128n64 f16w_a16_coopmat_m128n64_acc16 f16w_a16_coopmat_m128n64_acc16p \
         f16w_a16_coopmat_aligned f16w_a16_coopmat_acc16 f16w_a16_coopmat_acc16p \
         f16w_a16_coopmat_m64n128 f16w_a16_coopmat_m64n128_acc16 f16w_a16_coopmat_m64n128_acc16p \
         f16w_coopmat_m64n64 f16w_coopmat_m64n64_acc16 f16w_coopmat_m64n64_acc16p; do
  ML_KERNEL=$k ML_ISA_OUT=$OUT RUST_LOG=warn "$BIN" isa q512,1,512,2048,2048,f16w >> "$OUT/isa.log" 2>&1 || echo "isa $k failed"
done
