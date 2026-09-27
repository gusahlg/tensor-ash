#!/usr/bin/env bash
# PTX GEMM round 1 on the 4060 (run under lgpu).
cd ~/ta-wt/c-ptx
export ML_PTX=1
B=target/release/ml_bench
mkdir -p ~/ta-wt/c-ptx-art/jit
echo "### dump"; ML_PTX_DUMP_DIR=~/ta-wt/c-ptx-art/jit timeout 120 $B ptx dump 2>&1 | grep -v INFO
echo "### sgemm"; timeout 240 $B ptx sgemm 2>&1 | grep -v "INFO  tensor_ash"
echo "### hgemm"; timeout 240 $B ptx hgemm 2>&1 | grep -v "INFO  tensor_ash"
