#!/usr/bin/env bash
# Round 3 on the 4060: f16p (promoted f16 accumulate) non-split kernels.
cd ~/ta-wt/c-ptx
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=10
nice -n 5 cargo build --release -p ml-bench -p llama-ash 2>&1 | grep -E "^error|Finished" || exit 1
export ML_PTX=1
B=target/release/ml_bench
echo "### hgemm round 3 (f16p)"
ML_PTX_SPLITS=0 ML_PTX_KERNELS=ta_hgemm_f32_128x128_s3,ta_hgemm_f16_128x128_s3,ta_hgemm_f16p_128x64_s3,ta_hgemm_f16p_128x128_w64x32_s3,ta_hgemm_f16p_64x64_s4,ta_hgemm_f16p_128x64_k64_s2 \
  timeout 240 $B ptx hgemm 2>&1 | grep -v "INFO"
