#!/usr/bin/env bash
# Accumulator axis of the generator: the 15 configs on the acc16 a16 module
# vs the routed acc16 tile (auto@f16) and the literal 64x128/128x128 acc16.
set -euo pipefail
WT=${WT:-$HOME/ta-wt/f16acc}; OUT=${OUT:-$HOME/ta-wt/f16acc-results}; TAG=${TAG:-4060}
ROUNDS=${ROUNDS:-9}
cd "$WT"; mkdir -p "$OUT"
CASES="qo,1,512,2048,2048,a16 qkv,1,512,2560,2048,a16 gu,1,512,5632,2048,a16 dn,1,512,2048,5632,a16 s1k,1,1024,1024,1024,a16 s2k,1,2048,2048,2048,a16"
K="auto@f16,a16_coopmat_m64n128_acc16,a16_coopmat_acc16"
for c in t1 t2 t3 t4 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11; do K="$K,a16_coopmat_g_${c}_acc16"; done
echo "== gen acc16 ($(date +%T))"
ML_ROUNDS=$ROUNDS ML_ITERS=${ITERS:-30} ML_AB_KERNELS=$K RUST_LOG=warn ./target/release/ml_bench ab $CASES | tee "$OUT/gen-acc16-a16-$TAG${SUFFIX:-}.txt"
echo "== done ($(date +%T))"
