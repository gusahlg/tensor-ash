#!/usr/bin/env bash
# Opt-in PTX tests + the a16 suite through the PTX route (run under lgpu).
cd ~/ta-wt/c-ptx
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=10
nice -n 5 cargo test --release --test correctness --no-run 2>&1 | grep -E "^error|Finished|Executable" || exit 1
T=$(ls -t target/release/deps/correctness-* | grep -v '\.d$' | head -1)
echo "## ptx:: (ML_PTX=1)"; ML_PTX=1 timeout 180 $T --ignored --test-threads=1 ptx:: 2>&1 | grep -E "^test |test result|panicked|skipping"
for mode in off f32 f16p f16; do
  echo "## f16:: with ML_PTX_GEMM=$mode"; ML_PTX=1 ML_PTX_GEMM=$mode timeout 180 $T --ignored --test-threads=1 f16:: 2>&1 | grep -E "^test |test result|panicked"
done
echo "## graph:: prepared:: with ML_PTX_GEMM=f32"; ML_PTX=1 ML_PTX_GEMM=f32 timeout 240 $T --ignored --test-threads=1 graph:: prepared:: replay:: 2>&1 | grep -E "test result|panicked|FAILED"
