#!/usr/bin/env bash
# Round 2 on the 4060: incremental rebuild + new HGEMM variants (run under lgpu).
cd ~/ta-wt/c-ptx
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=10
nice -n 5 cargo build --release -p ml-bench -p llama-ash 2>&1 | grep -E "^error|Finished" || exit 1
export ML_PTX=1
B=target/release/ml_bench
echo "### hgemm round 2"
ML_PTX_SPLITS=0 ML_PTX_KERNELS=ta_hgemm_f32_128x128_s3,ta_hgemm_f32_64x64_s4,ta_hgemm_f32_128x128_w64x32_s3,ta_hgemm_f32_256x128_s3,ta_hgemm_f16_128x128_s3,ta_hgemm_f16_128x64_s3,ta_hgemm_f16_128x128_w64x32_s3,ta_hgemm_f16_128x128_w32x64_s3,ta_hgemm_f16_256x128_s3,ta_hgemm_f16_128x128_k64_s2,ta_hgemm_f16_128x64_k64_s2,ta_hgemm_f16_128x128_w64x32_k64_s2,ta_hgemm_f16p_128x64_s3,ta_hgemm_f16p_128x128_w64x32_s3,ta_hgemm_f16p_64x64_s4,ta_hgemm_f16p_128x64_k64_s2 \
  timeout 240 $B ptx hgemm 2>&1 | grep -v "INFO"
echo "### hgemm split-K"
ML_PTX_SPLITS=2,3,4 ML_PTX_KERNELS=ta_hgemm_f16_128x128_s3_splitk,ta_hgemm_f16_128x64_s3_splitk,ta_hgemm_f32_128x128_s3_splitk,ta_hgemm_f16p_128x64_s3_splitk \
  timeout 150 $B ptx hgemm 512x2048x2048 512x5632x2048 512x2048x5632 2048x2048x2048 2>&1 | grep -v "INFO"
echo "### sync (CudaAlt)"
timeout 60 $B ptx sync 2>&1 | grep -v INFO
