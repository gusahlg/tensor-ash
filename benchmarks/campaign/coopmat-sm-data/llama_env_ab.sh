#!/usr/bin/env bash
# llama_env_ab.sh <rounds> <out.log> <label=ENV...:bin>...   rotate variants per round
rounds="$1"; out="$2"; shift 2
M=${MODEL:-$HOME/models/tinyllama-1.1b-chat-v1.0.f16.gguf}
: > "$out"; echo "# $(hostname) $(date -Is) model=$M pp=512 variants: $*" >> "$out"
n=$#
for i in $(seq 1 "$rounds"); do
  # rotate start by round
  for j in $(seq 0 $((n-1))); do
    idx=$(( (j + i) % n + 1 ))
    spec=${!idx}
    label=${spec%%=*}; rest=${spec#*=}; envs=${rest%%:*}; bin=${rest##*:}
    res=$(env $envs RUST_LOG=info "$bin" bench -m "$M" --pp 512 --tg 32 2>&1)
    gpu=$(echo "$res" | grep -o "prefill timing: [0-9.]* ms GPU" | grep -o "[0-9.]*" | head -1)
    tps=$(echo "$res" | grep -E "^\| pp" | awk -F'|' '{gsub(/ /,"",$5); print $5}')
    cores=$(echo "$res" | grep -o "shader_cores=[^,]*" | head -1); arg=$(echo "$res" | grep -o "argmax [0-9]*" | head -1); wall=$(echo "$res" | grep -o "vs [0-9.]* ms wall" | grep -o "[0-9.]*" | head -1)
    echo "round=$i variant=$label prefill_gpu_ms=$gpu prefill_wall_ms=$wall pp_tps=$tps $arg $cores" >> "$out"
  done
done
cat "$out"
