#!/usr/bin/env bash
# Interleaved in-process A/B: f32-acc vs f16-acc vs promote, per tile and
# for the auto route, on the prefill + square shapes.  $1 = a16|f16w.
set -euo pipefail
WT=${WT:-$HOME/ta-wt/f16acc}; OUT=${OUT:-$HOME/ta-wt/f16acc-results}; TAG=${TAG:-4060}
ROUNDS=${ROUNDS:-7}; KIND=${1:-a16}
cd "$WT"; mkdir -p "$OUT"
CASES="qo,1,512,2048,2048,$KIND qkv,1,512,2560,2048,$KIND gu,1,512,5632,2048,$KIND dn,1,512,2048,5632,$KIND s2k,1,2048,2048,2048,$KIND s4k,1,4096,4096,4096,$KIND"
if [ "$KIND" = a16 ]; then f=a16_coopmat; else f=coopmat; fi
AUTO="auto,auto@f16,auto@promote"
TILES=""
for t in _m64 _m128n64 _m64n128 ""; do
  base=$f$t; [ -z "$t" ] && base=$([ "$KIND" = a16 ] && echo a16_coopmat || echo f16w_coopmat_aligned)
  acc16=$f$t; [ -z "$t" ] && acc16=$f
  TILES="$TILES,$base,${acc16}_acc16,${acc16}_acc16p"
done
TILES=${TILES#,}
echo "== auto route ($(date +%T))"
ML_ROUNDS=$ROUNDS ML_AB_KERNELS=$AUTO RUST_LOG=warn ./target/release/ml_bench ab $CASES | tee "$OUT/ab-auto-$KIND-$TAG.txt"
echo "== tiles ($(date +%T))"
ML_ROUNDS=$ROUNDS ML_AB_KERNELS=$TILES RUST_LOG=warn ./target/release/ml_bench ab $CASES | tee "$OUT/ab-tiles-$KIND-$TAG.txt"
echo "== done ($(date +%T))"
