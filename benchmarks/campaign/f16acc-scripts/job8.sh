#!/usr/bin/env bash
# Integration tests for the f16-accumulator classes (+ the a16 suite as a
# regression check), then an auto-vs-pinned control: the same kernel via
# the auto route and via an ML_KERNEL-style pin, twice each, one process.
set -uo pipefail
S=~/ta-wt/f16acc-scripts; R=~/ta-wt/f16acc-results
$S/build.sh || { echo "BUILD FAILED"; exit 1; }
cd ~/ta-wt/f16acc
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=10
echo "== test build $(date +%T)"
nice -n 5 cargo test --release -p tensor-ash --test correctness --no-run 2>&1 | grep -E "^error|warning: unused|Finished|could not compile" | head -20
echo "== tests $(date +%T)"
RUST_LOG=warn cargo test --release -p tensor-ash --test correctness -- --ignored --test-threads=1 coopmat_acc 2>&1 | grep -E "^test |test result|panicked|error" | tee $R/tests-coopmat-acc-4060.txt
RUST_LOG=warn cargo test --release -p tensor-ash --test correctness -- --ignored --test-threads=1 f16:: 2>&1 | grep -E "^test |test result|panicked|error" | tee $R/tests-f16-4060.txt
echo "== auto vs pinned control $(date +%T)"
CASES="qo,1,512,2048,2048,a16 qkv,1,512,2560,2048,a16 gu,1,512,5632,2048,a16 dn,1,512,2048,5632,a16 s2k,1,2048,2048,2048,a16"
ML_ROUNDS=9 ML_ITERS=30 ML_AB_KERNELS=auto,a16_coopmat_m64,a16_coopmat_m128n64,a16_coopmat,auto,a16_coopmat_m64,a16_coopmat_m128n64,a16_coopmat RUST_LOG=warn ./target/release/ml_bench ab $CASES | tee $R/ab-autopin-a16-4060.txt
echo "== end $(date +%T)"
