#!/usr/bin/env bash
# ku8 rescue: build, numerics gate (validates the new kernels first), then
# twins vs literal vs ku8 (9 interleaved rounds).
set -uo pipefail
S=~/ta-wt/f16acc-scripts; R=~/ta-wt/f16acc-results
$S/build.sh || { echo "BUILD FAILED"; exit 1; }
cd ~/ta-wt/f16acc
echo "== numerics gate $(date +%T)"
ML_NUMERICS_GATE_ONLY=1 RUST_LOG=warn ./target/release/examples/coopmat_acc_numerics > $R/numerics-gate3-4060.md 2>&1; rc=$?; echo "gate exit $rc"
grep -c "| ok |" $R/numerics-gate3-4060.md; grep "ku8" $R/numerics-gate3-4060.md; grep FAIL $R/numerics-gate3-4060.md
[ $rc = 0 ] || exit 1
echo "== ku8 ab $(date +%T)"
CASES="qo,1,512,2048,2048,a16 qkv,1,512,2560,2048,a16 gu,1,512,5632,2048,a16 dn,1,512,2048,5632,a16 s1k,1,1024,1024,1024,a16 s2k,1,2048,2048,2048,a16"
K=a16_coopmat,a16_coopmat_g_t1,a16_coopmat_g_t1_ku8,a16_coopmat_m64,a16_coopmat_g_t2,a16_coopmat_g_t2_ku8,a16_coopmat_m64n128,a16_coopmat_g_t3,a16_coopmat_g_t3_ku8,a16_coopmat_m128n64,a16_coopmat_g_t4,a16_coopmat_g_t4_ku8,a16_coopmat_g_n1,a16_coopmat_g_n1_ku8,a16_coopmat_g_n10,a16_coopmat_g_n10_ku8
ML_ROUNDS=9 ML_ITERS=30 ML_AB_KERNELS=$K RUST_LOG=warn ./target/release/ml_bench ab $CASES | tee $R/gen-ku8-a16-4060.txt
echo "== end $(date +%T)"
