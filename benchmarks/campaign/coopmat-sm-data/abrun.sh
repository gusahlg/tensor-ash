#!/usr/bin/env bash
# Usage (inside the GPU lock, cwd = worktree root): abrun.sh <outdir> <batch> <case>...
# Env passthrough: ML_AB_KERNELS ML_ROUNDS ML_ITERS ML_WARMUP ML_SM_COUNT BIN
set -u
out="$1"; batch="$2"; shift 2
mkdir -p "$out"
bin=${BIN:-target/release/ml_bench}
: "${ML_AB_KERNELS:=f16w_coopmat_aligned,f16w_coopmat_m64n64,f16w_coopmat_m64n128,f16w_coopmat_m128n64,auto}"
: "${ML_ROUNDS:=7}"
export ML_AB_KERNELS ML_ROUNDS ML_OUTPUT=csv
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader -lms 250 > "$out/$batch.smi.csv" 2>/dev/null &
smi=$!
{
  echo "# host=$(hostname) date=$(date -Is) bin=$bin git=$(cat GIT_HEAD 2>/dev/null)"
  echo "# ML_AB_KERNELS=$ML_AB_KERNELS ML_ROUNDS=$ML_ROUNDS ML_ITERS=${ML_ITERS:-20} ML_WARMUP=${ML_WARMUP:-5} ML_SM_COUNT=${ML_SM_COUNT:-}"
  echo "# cmd: $bin ab $*"
} > "$out/$batch.csv"
start=$(date +%s.%N)
RUST_LOG=${RUST_LOG:-info} "$bin" ab "$@" >> "$out/$batch.csv" 2> "$out/$batch.log"
rc=$?
end=$(date +%s.%N)
kill $smi 2>/dev/null; wait $smi 2>/dev/null
echo "# rc=$rc elapsed_s=$(awk "BEGIN{print $end - $start}")" >> "$out/$batch.csv"
echo "batch $batch rc=$rc elapsed=$(awk "BEGIN{print $end - $start}")s"
exit $rc
