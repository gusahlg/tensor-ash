#!/usr/bin/env bash
# llama.cpp CUDA pp512: one warm-up invocation, then per-rep samples (json).
B=${B:-$HOME/models/llama-cpp-cuda/bin/llama-bench}; M=${M:-$HOME/models/tinyllama-1.1b-chat-v1.0.f16.gguf}
PARSE=${PARSE:-$(dirname "$0")/llamabench_parse.py}
"$B" -m "$M" -p 512 -n 0 -r 5 -ngl 99 >/dev/null 2>&1
"$B" -m "$M" -p 512 -n 0 -r 10 -ngl 99 -o json 2>/dev/null | python3 "$PARSE"
