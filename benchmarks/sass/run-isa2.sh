#!/usr/bin/env bash
# Usage: run-isa2.sh <bench-binary> <out-dir>
# Question (a) at scale: the desktop LOWMEM build embeds glslc -Os SPIR-V for
# ~11 heavy shaders, louise embeds -O for all.  SASS is byte-identical across
# GA104/AD107 (control: large_bda_v4, -O on both), so comparing the embedded
# builds' UCODE per kernel answers -O vs -Os for the tm8_tn8-class shaders
# without another multi-GB glslc -O run.  Pipeline creation only.
set -uo pipefail
BIN=$1; OUT=$2
export RUST_LOG=warn
for k in large_bda_v4 k64_bda_v4_tm8_tn8 m128n64k64_bda_v4_tm8_tn8 m128n64k64_bda_v4_tm16_tn4 \
         m128n64k64_bda large_bda large_128 m128n64k64 m256n64 m64n128k64; do
  ML_KERNEL=$k ML_ISA_OUT=$OUT "$BIN" isa sq1024,1,1024,1024,1024 > "$OUT.$k.log" 2>&1 || echo "FAIL $k"
done
echo done
