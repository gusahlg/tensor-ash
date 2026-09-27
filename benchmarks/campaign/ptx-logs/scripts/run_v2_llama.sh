#!/usr/bin/env bash
# 2026-09-27: TinyLlama pp512 + token exactness through the new policy API (4060, under lgpu).
cd ~/ta-wt/c-ptx
L=target/release/llama_ash
M=~/models/tinyllama-1.1b-chat-v1.0.f16.gguf
IDS=$(cat ~/ta-wt/c-ptx-art/ids128.txt)
run() { # label, env...
  local label=$1; shift
  echo "# $label"; env "$@" timeout 100 $L bench -m $M --pp 512 --tg 32 2>&1 | grep -E "prefill timing|pp512|accumulator policy|route on"
}
for rep in 1 2; do
  run base          ML_PTX=0
  run ptx_f16p      ML_PTX=1 ML_COOPMAT_ACC=f16p
  run ptx_f16       ML_PTX=1 ML_COOPMAT_ACC=f16
  run ptx_on_f32    ML_PTX=1
done
echo "## generate, 128-token prompt, 32 greedy tokens"
for cfg in "ML_PTX=0" "ML_PTX=1" "ML_PTX=1 ML_PTX_GEMM=all" "ML_PTX=1 ML_COOPMAT_ACC=f16p" "ML_PTX=1 ML_COOPMAT_ACC=f16"; do
  echo "# $cfg"; env $cfg timeout 100 $L generate -m $M --ids "$IDS" -n 32 2>&1 | grep "generated_ids"
done
