#!/usr/bin/env bash
cd ~/ta-wt/c-ptx-art
LD=/nix/store/57iz36553175g3178pvxjij8z5rcsd4n-glibc-2.42-61/lib64/ld-linux-x86-64.so.2
export LD_LIBRARY_PATH=/nix/store/n474iszgrq3ganwy2b9a35m9mv57szfd-cuda12.9-libcublas-12.9.1.4-lib/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH
echo "### cublas_hgemm_bench"
timeout 180 $LD ./cublas_hgemm_bench --iters 30 --warmup 8 512x2048x2048 512x5632x2048 512x2048x5632 2048x2048x2048 4096x4096x4096
