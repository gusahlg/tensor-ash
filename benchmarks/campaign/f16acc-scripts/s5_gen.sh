#!/usr/bin/env bash
# Generator M1 on the 4060: S1 parity (t1-t4 vs literal tiles) and S2
# search signal (n1-n11 vs the routed tile), 9 interleaved rounds.
# $1 = a16|f16w.
set -euo pipefail
WT=${WT:-$HOME/ta-wt/f16acc}; OUT=${OUT:-$HOME/ta-wt/f16acc-results}; TAG=${TAG:-4060}
ROUNDS=${ROUNDS:-9}; KIND=${1:-a16}
cd "$WT"; mkdir -p "$OUT"
CASES="qo,1,512,2048,2048,$KIND qkv,1,512,2560,2048,$KIND gu,1,512,5632,2048,$KIND dn,1,512,2048,5632,$KIND s1k,1,1024,1024,1024,$KIND s2k,1,2048,2048,2048,$KIND"
if [ "$KIND" = a16 ]; then p=a16_; lit1=a16_coopmat; else p=; lit1=f16w_coopmat_aligned; fi
K="auto,$lit1,${p}coopmat_g_t1,${p}coopmat_m64,${p}coopmat_g_t2,${p}coopmat_m64n128,${p}coopmat_g_t3,${p}coopmat_m128n64,${p}coopmat_g_t4"
for n in 1 2 3 4 5 6 7 8 9 10 11; do K="$K,${p}coopmat_g_n$n"; done
echo "== gen $KIND ($(date +%T))"
ML_ROUNDS=$ROUNDS ML_ITERS=${ITERS:-30} ML_AB_KERNELS=$K RUST_LOG=warn ./target/release/ml_bench ab $CASES | tee "$OUT/gen-$KIND-$TAG${SUFFIX:-}.txt"
echo "== done ($(date +%T))"
