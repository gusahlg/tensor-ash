#!/usr/bin/env bash
# Usage: run-isa.sh <bench-binary> <out-root> <spv-dir>
# Static ISA probes only (pipeline creation, no dispatch).
set -euo pipefail
BIN=$1; OUT=$2; SPV=$3
mkdir -p "$OUT"
export RUST_LOG=warn
# 1. the five representative kernels as routed + (b) isolation shapes
ML_ISA_OUT=$OUT/main "$BIN" isa \
  gateup,1,1,5632,2048,f16w gateup_packed,1,1,5632,2048,packed lmhead,1,1,32000,2048,packed \
  q512,1,512,2048,2048,f16w gu512,1,512,5632,2048,f16w \
  sq4096h,1,4096,4096,4096,f16w sq4096,1,4096,4096,4096 \
  odd,1,1023,1025,1027 sq1024,1,1024,1024,1024 \
  oddk,1,1024,1024,1027 oddmn,1,1023,1025,1024 > "$OUT/main.log" 2>&1
# 2. glslc -O vs -Os of the same shader, same route/specialization
probe() { # kernel shader spec
  for o in O Os; do
    ML_KERNEL=$1 ML_ISA_SPV=$SPV/$2.$o.spv ML_ISA_TAG=$o ML_ISA_OUT=$OUT/opt "$BIN" isa $3 >> "$OUT/opt.log" 2>&1
  done
}
: > "$OUT/opt.log"
probe large_bda_v4 matmul_f32_large_bda_v4 sq4096,1,4096,4096,4096
probe k64_bda_v4_tm8_tn4 matmul_f32_k64_bda_v4_tm8_tn4 sq384,1,384,384,384
probe f16w_coopmat_aligned matmul_f16w_coopmat_aligned sq4096h,1,4096,4096,4096,f16w
probe m128n64k64_bda_v4 matmul_f32_m128n64k64_bda_v4 odd,1,1023,1025,1027
probe m128n64k64_bda_v4 matmul_f32_m128n64k64_bda_v4 sq1024,1,1024,1024,1024
probe f16w_coopmat_m128n64 matmul_f16w_coopmat_m128n64 gu512,1,512,5632,2048,f16w
echo done
