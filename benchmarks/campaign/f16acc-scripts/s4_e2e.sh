#!/usr/bin/env bash
# End to end: pp512 fresh-process A/B (f32 vs f16 vs promote, rotated
# order, $PAIRS rounds), then greedy generation + prefill logits per mode.
set -euo pipefail
WT=${WT:-$HOME/ta-wt/f16acc}; OUT=${OUT:-$HOME/ta-wt/f16acc-results}; TAG=${TAG:-4060}
MODEL=${MODEL:-$HOME/models/tinyllama-1.1b-chat-v1.0.f16.gguf}
P=${P:-$HOME/ta-wt/f16acc-prompts}; PAIRS=${PAIRS:-6}; N=${N:-32}
cd "$WT"; mkdir -p "$OUT/logits-$TAG"
run_mode() { # mode cmd...
  local mode=$1; shift
  if [ "$mode" = f32 ]; then env -u ML_COOPMAT_ACC RUST_LOG=info "$@"; else ML_COOPMAT_ACC=$mode RUST_LOG=info "$@"; fi
}
if [ -z "${SKIP_PP:-}" ]; then
: > "$OUT/pp512-$TAG.txt"
for i in $(seq 1 "$PAIRS"); do
  if [ $((i % 2)) = 1 ]; then order="f32 f16 promote"; else order="promote f16 f32"; fi
  for mode in $order; do
    line=$(run_mode $mode ./target/release/llama_ash bench -m "$MODEL" --pp 512 --tg 16 2>&1 | grep -E "\| pp512|prefill timing" | tr '\n' ' ')
    echo "round $i mode $mode $line" | tee -a "$OUT/pp512-$TAG.txt"
  done
done
fi
if [ -z "${SKIP_GEN:-}" ]; then
: > "$OUT/gen-$TAG.txt"
for p in p1 p2 p3 p4 p5; do
  for mode in f32 f16 promote; do
    ids=$(LLAMA_ASH_LOGITS_OUT=$OUT/logits-$TAG/$p-$mode.bin run_mode $mode ./target/release/llama_ash generate -m "$MODEL" --ids "$(cat $P/$p.csv)" -n "$N" 2>/dev/null | grep generated_ids | sed 's/generated_ids: //')
    echo "$p $mode $ids" | tee -a "$OUT/gen-$TAG.txt"
  done
done
fi
echo "== done ($(date +%T))"
