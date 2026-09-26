#!/usr/bin/env bash
# llama prefill with the opt-in PTX a16 route (run under lgpu after run_gemm2.sh built it).
cd ~/ta-wt/c-ptx
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=10
nice -n 5 cargo build --release -p ml-bench -p llama-ash 2>&1 | grep -E "^error|Finished" || exit 1
L=target/release/llama_ash
M=~/models/tinyllama-1.1b-chat-v1.0.f16.gguf
run() { # label env...
  local label=$1; shift
  echo "## $label"
  env "$@" timeout 120 $L bench -m $M --pp 512 --tg 32 2>&1 | grep -E "prefill timing|pp512|prefill:|t/s|tok/s|route on|error|Error" | head -8
}
for rep in 1 2; do
  run base ML_PTX=0
  run ptx_f32 ML_PTX=1 ML_PTX_GEMM=f32
  run ptx_f16 ML_PTX=1 ML_PTX_GEMM=f16
  run ptx_f16p ML_PTX=1 ML_PTX_GEMM=f16p
done
echo "## generate token ids (24 greedy)"
for mode in off f32 f16 f16p; do
  echo "# $mode"; ML_PTX=1 ML_PTX_GEMM=$mode timeout 120 $L generate -m $M --ids "1,15043,3186,338,263,1243" -n 24 2>&1 | grep -v INFO | tail -3
done
