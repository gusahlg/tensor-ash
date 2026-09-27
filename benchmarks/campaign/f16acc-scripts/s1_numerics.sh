#!/usr/bin/env bash
# Numerics first: synthetic f64-reference gate + error table, then the
# real-activation (TinyLlama last-layer prefill) GEMM numerics.
set -euo pipefail
WT=${WT:-$HOME/ta-wt/f16acc}; OUT=${OUT:-$HOME/ta-wt/f16acc-results}; TAG=${TAG:-4060}
MODEL=${MODEL:-$HOME/models/tinyllama-1.1b-chat-v1.0.f16.gguf}
P=${P:-$HOME/ta-wt/f16acc-prompts}
mkdir -p "$OUT"; cd "$WT"
echo "== synthetic ($(date +%T))"
RUST_LOG=warn ./target/release/examples/coopmat_acc_numerics > "$OUT/numerics-synth-$TAG.md" 2> "$OUT/numerics-synth-$TAG.log" || echo "numerics example exit $?"
tail -3 "$OUT/numerics-synth-$TAG.md"
echo "== real activations ($(date +%T))"
RUST_LOG=warn ./target/release/llama_ash numerics -m "$MODEL" --ids "$(cat $P/real512.csv)" > "$OUT/numerics-real-$TAG.md" 2> "$OUT/numerics-real-$TAG.log" || echo "llama numerics exit $?"
cat "$OUT/numerics-real-$TAG.md"
echo "== done ($(date +%T))"
