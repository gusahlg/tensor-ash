#!/usr/bin/env bash
# 2026-09-27 3070 confirmation (desktop, under gpu3070), only after run_v2_tests.sh passed on the 4060.
# Batch A: tests.  Batch B: pp512 + tokens (run separately: run_v2_3070.sh llama).
cd ~/repos/tensor-ash-wt/c-ptx
T=target/release/deps/correctness-5c3f13bb949fe450
L=target/release/llama_ash
M=~/models/tinyllama-1.1b-chat-v1.0.f16.gguf
if [ "$1" != llama ]; then
  echo "## ptx:: ML_PTX=1"
  ML_PTX=1 timeout 300 $T --ignored --test-threads=1 ptx:: 2>&1 | grep -E "^test |test result|panicked|rel-RMS"
  echo "## f16:: graph:: prepared:: replay:: with ML_PTX_GEMM=all"
  ML_PTX=1 ML_PTX_GEMM=all timeout 300 $T --ignored --test-threads=1 f16:: graph:: prepared:: replay:: 2>&1 | grep -E "test result|panicked|FAILED"
else
  IDS=$(cat ~/repos/tensor-ash-wt/campaign-tools/work/ptx/ids128.txt)
  for cfg in "ML_PTX=0" "ML_PTX=1 ML_COOPMAT_ACC=f16p" "ML_PTX=1 ML_COOPMAT_ACC=f16" "ML_PTX=0" "ML_PTX=1 ML_COOPMAT_ACC=f16p" "ML_PTX=1 ML_COOPMAT_ACC=f16"; do
    echo "# $cfg"; env $cfg timeout 100 $L bench -m $M --pp 512 --tg 32 2>&1 | grep -E "prefill timing|pp512"
  done
  for cfg in "ML_PTX=0" "ML_PTX=1 ML_COOPMAT_ACC=f16p" "ML_PTX=1 ML_COOPMAT_ACC=f16"; do
    echo "# $cfg"; env $cfg timeout 100 $L generate -m $M --ids "$IDS" -n 32 2>&1 | grep "generated_ids"
  done
fi
