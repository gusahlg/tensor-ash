#!/usr/bin/env bash
set -uo pipefail
S=~/ta-wt/f16acc-scripts; R=~/ta-wt/f16acc-results
$S/build.sh || { echo "BUILD FAILED"; exit 1; }
cd ~/ta-wt/f16acc
CASES_A16="qo,1,512,2048,2048,a16 qkv,1,512,2560,2048,a16 gu,1,512,5632,2048,a16 dn,1,512,2048,5632,a16 s2k,1,2048,2048,2048,a16 s4k,1,4096,4096,4096,a16"
echo "== ab auto a16 (acc-aware tiles) $(date +%T)"
ML_ROUNDS=7 ML_AB_KERNELS=auto,auto@f16,auto@promote RUST_LOG=warn ./target/release/ml_bench ab $CASES_A16 | tee $R/ab-auto3-a16-4060.txt
echo "== pp512 $(date +%T)"
SKIP_GEN=1 TAG=4060-v2 $S/s4_e2e.sh
echo "== end $(date +%T)"
