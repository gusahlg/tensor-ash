#!/usr/bin/env bash
# llama_ab.sh <A-bin> <B-bin> <pairs> <out.log> [pp]   alternating A,B,A,B... pp512 prefill
A="$1"; B="$2"; pairs="$3"; out="$4"; pp=${5:-512}
M=${MODEL:-$HOME/models/tinyllama-1.1b-chat-v1.0.f16.gguf}
: > "$out"
echo "# $(hostname) $(date -Is) A=$A B=$B pairs=$pairs pp=$pp model=$M" >> "$out"
for i in $(seq 1 "$pairs"); do
  for tag in A B; do
    bin=$A; [ $tag = B ] && bin=$B
    res=$(RUST_LOG=info "$bin" bench -m "$M" --pp "$pp" --tg 32 2>&1)
    gpu=$(echo "$res" | grep -o "prefill timing: [0-9.]* ms GPU" | grep -o "[0-9.]*" | head -1)
    wall=$(echo "$res" | grep -o "vs [0-9.]* ms wall" | grep -o "[0-9.]*" | head -1)
    tps=$(echo "$res" | grep -E "^\| pp" | awk -F'|' '{gsub(/ /,"",$5); print $5}')
    arg=$(echo "$res" | grep -o "argmax [0-9]*" | head -1)
    echo "run=$i tag=$tag prefill_gpu_ms=$gpu prefill_wall_ms=$wall pp_tps=$tps $arg" >> "$out"
  done
done
cat "$out"
