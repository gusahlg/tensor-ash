#!/usr/bin/env bash
# Static ISA probes (pipeline creation only, no dispatch): the a16 and a32
# coopmat tiles at every accumulator class.  nvdisasm runs later on the
# desktop (ml_bench isa-analyze).
set -euo pipefail
WT=${WT:-$HOME/ta-wt/f16acc}; OUT=${OUT:-$HOME/ta-wt/f16acc-results}; TAG=${TAG:-4060}
cd "$WT"; mkdir -p "$OUT/isa-$TAG"
: > "$OUT/isa-$TAG.log"
for fam in f16w_a16_coopmat f16w_coopmat; do
  for tile in "" _m64n64 _m64n128 _m128n64; do
    for acc in "" _acc16 _acc16p; do
      k=$fam$tile$acc
      [ "$k" = f16w_coopmat ] && k=f16w_coopmat_aligned
      [ "$k" = f16w_a16_coopmat ] && k=f16w_a16_coopmat_aligned
      ML_KERNEL=$k ML_ISA_OUT=$OUT/isa-$TAG ML_ISA_NVDISASM=off RUST_LOG=warn \
        ./target/release/ml_bench isa q512,1,512,2048,2048,f16w >> "$OUT/isa-$TAG.log" 2>&1 || echo "isa $k failed" | tee -a "$OUT/isa-$TAG.log"
    done
  done
done
ls "$OUT/isa-$TAG" | wc -l
