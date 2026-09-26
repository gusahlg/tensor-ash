#!/usr/bin/env bash
# Usage: run-measure.sh <bench-binary> <out-file>; GPU-timestamp medians, 2 reps,
# first case of each rep is a ~1.4 s burn-in (4096^3 f32).
set -uo pipefail
BIN=$1; OUT=$2
export RUST_LOG=warn ML_ITERS=${ML_ITERS:-50} ML_WARMUP=${ML_WARMUP:-10} ML_OUTPUT=csv
: > "$OUT"
for rep in 1 2; do
  echo "# rep $rep $(date +%T)" >> "$OUT"
  "$BIN" cases burn,1,4096,4096,4096 \
    gateup,1,1,5632,2048,f16w gateup_b8,8,1,5632,2048,f16w gateup_packed,1,1,5632,2048,packed lmhead,1,1,32000,2048,packed \
    q512,1,512,2048,2048,f16w gu512,1,512,5632,2048,f16w \
    sq4096h,1,4096,4096,4096,f16w sq4096,1,4096,4096,4096 \
    odd,1,1023,1025,1027 sq1024,1,1024,1024,1024 oddk,1,1024,1024,1027 oddmn,1,1023,1025,1024 \
    >> "$OUT" 2>&1
done
echo "measure done rc=$?"
