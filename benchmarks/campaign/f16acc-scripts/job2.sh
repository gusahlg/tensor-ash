#!/usr/bin/env bash
# Build; auto-route A/B with the acc-aware tile rule (a16 + f16w); f16w
# per-tile A/B; pp512 fresh-process A/B.
set -uo pipefail
cd ~/ta-wt/f16acc
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=10
echo "== build $(date +%T)"
nice -n 5 cargo build --release -p ml-bench -p llama-ash -p tensor-ash --bins --examples 2>&1 | grep -E "^error|Finished|could not compile"
[ -x target/release/llama_ash ] || { echo "BUILD FAILED"; exit 1; }
S=~/ta-wt/f16acc-scripts; R=~/ta-wt/f16acc-results
CASES_A16="qo,1,512,2048,2048,a16 qkv,1,512,2560,2048,a16 gu,1,512,5632,2048,a16 dn,1,512,2048,5632,a16 s2k,1,2048,2048,2048,a16 s4k,1,4096,4096,4096,a16"
echo "== ab auto a16 (acc-aware tiles) $(date +%T)"
ML_ROUNDS=7 ML_AB_KERNELS=auto,auto@f16,auto@promote RUST_LOG=warn ./target/release/ml_bench ab $CASES_A16 | tee $R/ab-auto2-a16-4060.txt
echo "== ab f16w $(date +%T)"
$S/s3_ab.sh f16w
echo "== pp512 $(date +%T)"
SKIP_GEN=1 $S/s4_e2e.sh
echo "== end $(date +%T)"
