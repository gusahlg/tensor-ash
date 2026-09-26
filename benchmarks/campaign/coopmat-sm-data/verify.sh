#!/usr/bin/env bash
# verify.sh <outdir> [batch...]   (cwd = worktree; BIN overrides the binary)
out="$1"; shift
here=$(dirname "$0")
bf=${BF:-$here/batches.txt}
batches=${*:-sq m256 m384 m512 m768 m1024 m2048 t128 t1920 a16_128 a16_256 a16_512 a16_1024 a16_1920}
export ML_ROUNDS=${ML_ROUNDS:-7}
export ML_AB_KERNELS=f16w_coopmat_m128n64,f16w_coopmat_m64n64,f16w_coopmat_m64n64_mfast,f16w_coopmat_aligned,f16w_coopmat_m64n128,auto
for batch in $batches; do
  cases=$(grep "^$batch " "$bf" | cut -d' ' -f2-)
  timeout 300 $here/abrun.sh "$out" "$batch" $cases
  [ -n "$GAP" ] && sleep "$GAP"
done
echo VERIFY_DONE
