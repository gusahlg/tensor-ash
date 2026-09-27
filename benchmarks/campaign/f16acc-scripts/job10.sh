#!/usr/bin/env bash
# 2026-09-27 continuation, GPU part (4060): numerics gate + error table with the
# promote chunk variants, promote chunk-length ab sweep, f16-accumulator
# integration tests (+ f16:: regression), auto-vs-pinned noise control.
set -uo pipefail
R=~/ta-wt/f16acc-results
cd ~/ta-wt/f16acc
nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm --format=csv,noheader
echo "== numerics $(date +%T)"
ML_NUMERICS_SKIP_GEN=1 ML_NUMERICS_KERNELS=a16_coopmat_m64n128_pk2_acc16p,a16_coopmat_m64n128_acc16p,a16_coopmat_m64n128_pk8_acc16p \
  RUST_LOG=warn ./target/release/examples/coopmat_acc_numerics > $R/numerics-pk-4060.md 2>&1; rc=$?; echo "numerics exit $rc"
grep -E "pk[28]|FAIL" $R/numerics-pk-4060.md | head -20
[ $rc = 0 ] || exit 1
echo "== promote chunk ab $(date +%T)"
CASES="qo,1,512,2048,2048,a16 qkv,1,512,2560,2048,a16 gu,1,512,5632,2048,a16 dn,1,512,2048,5632,a16 s2k,1,2048,2048,2048,a16"
ML_ROUNDS=9 ML_ITERS=30 ML_AB_KERNELS=auto,auto@f16,a16_coopmat_m64n128_pk2_acc16p,a16_coopmat_m64n128_acc16p,a16_coopmat_m64n128_pk8_acc16p \
  RUST_LOG=warn ./target/release/ml_bench ab $CASES | tee $R/ab-pk-a16-4060.txt
echo "== tests $(date +%T)"
RUST_LOG=warn cargo test --release -p tensor-ash --test correctness -- --ignored --test-threads=1 coopmat_acc 2>&1 | grep -E "^test |test result|panicked" | tee $R/tests-coopmat-acc-4060.txt
RUST_LOG=warn cargo test --release -p tensor-ash --test correctness -- --ignored --test-threads=1 f16:: 2>&1 | grep -E "test result|panicked|FAILED" | tee $R/tests-f16-4060.txt
echo "== auto vs pinned control $(date +%T)"
ML_ROUNDS=9 ML_ITERS=30 ML_AB_KERNELS=auto,a16_coopmat_m64,a16_coopmat_m128n64,auto,a16_coopmat_m64,a16_coopmat_m128n64 \
  RUST_LOG=warn ./target/release/ml_bench ab $CASES | tee $R/ab-autopin-a16-4060.txt
nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm --format=csv,noheader
echo "== end $(date +%T)"
