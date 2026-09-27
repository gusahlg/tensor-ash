#!/usr/bin/env bash
# Round 4 on the 4060: grouped rasterization (_g8) + cuBLAS f16 numerics (run under lgpu).
cd ~/ta-wt/c-ptx
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=10
nice -n 5 cargo build --release -p ml-bench -p llama-ash 2>&1 | grep -E "^error|Finished" || exit 1
export ML_PTX=1
B=target/release/ml_bench
echo "### hgemm round 4 (grouped raster)"
ML_PTX_SPLITS=0 ML_PTX_KERNELS=ta_hgemm_f16_128x128_s3,ta_hgemm_f16_128x128_s3_g8,ta_hgemm_f32_128x128_s3,ta_hgemm_f32_128x128_s3_g8,ta_hgemm_f16p_128x64_s3,ta_hgemm_f16p_128x64_s3_g8 \
  timeout 200 $B ptx hgemm 512x5632x2048 2048x2048x2048 4096x4096x4096 2>&1 | grep -v "INFO"
echo "### cuBLAS f16 with numerics"
cd ~/ta-wt/c-ptx-art
LD=/nix/store/57iz36553175g3178pvxjij8z5rcsd4n-glibc-2.42-61/lib64/ld-linux-x86-64.so.2
LD_LIBRARY_PATH=/nix/store/n474iszgrq3ganwy2b9a35m9mv57szfd-cuda12.9-libcublas-12.9.1.4-lib/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH \
  timeout 200 $LD ./cublas_hgemm_bench2 --iters 30 --warmup 8 512x2048x2048 512x5632x2048 512x2048x5632 2048x2048x2048 4096x4096x4096
