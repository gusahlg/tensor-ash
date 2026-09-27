#!/usr/bin/env bash
# Final validation of the branch head on the 4060 (run under lgpu).
cd ~/ta-wt/c-ptx
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=10
nice -n 5 cargo build --release -p ml-bench -p llama-ash 2>&1 | grep -E "^error|Finished" || exit 1
nice -n 5 cargo test --release --test correctness --no-run 2>&1 | grep -E "^error|Finished" || exit 1
T=$(ls -t target/release/deps/correctness-* | grep -v '\.d$' | head -1)
echo "## ptx:: tests"; ML_PTX=1 timeout 120 $T --ignored --test-threads=1 ptx:: 2>&1 | grep -E "test result|panicked"
echo "## f16:: through the f32 PTX route"; ML_PTX=1 ML_PTX_GEMM=f32 timeout 120 $T --ignored --test-threads=1 f16:: 2>&1 | grep -E "test result|panicked"
echo "## route picks g8 at 4096^3 (TA side = route)"
ML_PTX=1 ML_PTX_GEMM=f16 ML_PTX_SPLITS=0 ML_PTX_KERNELS=ta_hgemm_f16_128x128_s3,ta_hgemm_f16_128x128_s3_g8 timeout 120 target/release/ml_bench ptx hgemm 4096x4096x4096 2048x2048x2048 2>&1 | grep -E "route on|x4096|x2048"
