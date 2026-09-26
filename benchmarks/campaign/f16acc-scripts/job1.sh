#!/usr/bin/env bash
# One heavy-lock hold: build, then numerics first, then the a16 A/B.
set -uo pipefail
cd ~/ta-wt/f16acc
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=10
echo "== build $(date +%T)"
nice -n 5 cargo build --release -p ml-bench -p llama-ash -p tensor-ash --bins --examples 2>&1 | grep -E "^(error|warning: unused)|Finished|could not compile" 
[ -x target/release/examples/coopmat_acc_numerics ] || { echo "BUILD FAILED"; exit 1; }
echo "== numerics $(date +%T)"
~/ta-wt/f16acc-scripts/s1_numerics.sh
echo "== ab a16 $(date +%T)"
~/ta-wt/f16acc-scripts/s3_ab.sh a16
echo "== end $(date +%T)"
