#!/usr/bin/env bash
# Token agreement (tensor-ash f32/f16/promote + prefill logits) and the
# llama.cpp CUDA references (greedy tokens via llama-server, pp512).
set -uo pipefail
S=~/ta-wt/f16acc-scripts; R=~/ta-wt/f16acc-results; P=~/ta-wt/f16acc-prompts
cd ~/ta-wt/f16acc
echo "== gen tokens $(date +%T)"
SKIP_PP=1 $S/s4_e2e.sh
echo "== llama.cpp reference tokens $(date +%T)"
cd $P && LD_LIBRARY_PATH=$HOME/models/llama-cpp-cuda/lib:/run/opengl-driver/lib timeout 180 python3 llama_ref.py ~/models/llama-cpp-cuda/bin/llama-server ~/models/tinyllama-1.1b-chat-v1.0.f16.gguf 32 $R/ref_4060.json p1.csv p2.csv p3.csv p4.csv p5.csv
echo "== llama-bench $(date +%T)"
LD_LIBRARY_PATH=$HOME/models/llama-cpp-cuda/lib:/run/opengl-driver/lib $S/llamabench.sh | tee $R/llamacpp-pp512-4060.txt
echo "== end $(date +%T)"
