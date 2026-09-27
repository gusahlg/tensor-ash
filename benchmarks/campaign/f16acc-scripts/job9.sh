#!/usr/bin/env bash
# 2026-09-27 continuation, CPU part (build lock only): release build of the
# bins/examples (incl. the promote chunk variants pk2/pk8), the correctness
# test binary, clippy -D warnings on the touched crates, CPU unit tests.
set -uo pipefail
S=~/ta-wt/f16acc-scripts
$S/build.sh || { echo "BUILD FAILED"; tail -40 /tmp/f16acc-build.log; exit 1; }
cd ~/ta-wt/f16acc
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=10
echo "== test build $(date +%T)"
nice -n 5 cargo test --release -p tensor-ash --test correctness --no-run 2>&1 | grep -E "^(error|warning)|Finished|could not compile" | head -20
echo "== unit tests $(date +%T)"
RUST_LOG=warn nice -n 5 cargo test --release -p tensor-ash --lib 2>&1 | grep -E "test result|FAILED|panicked" | head
RUST_LOG=warn nice -n 5 cargo test --release -p ml-bench -p llama-ash 2>&1 | grep -E "test result|FAILED|panicked" | head
echo "== clippy $(date +%T)"
nice -n 5 cargo clippy --release -p tensor-ash -p ml-bench -p llama-ash --all-targets -- -D warnings 2>&1 | grep -E "^(error|warning)|Finished|could not compile" | head -30
echo "== end $(date +%T)"
