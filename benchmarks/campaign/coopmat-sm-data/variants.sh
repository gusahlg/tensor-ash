#!/usr/bin/env bash
# variants.sh <outdir> [rounds]   (cwd = worktree with the candidate kernels)
out="$1"; R=${2:-9}
bf=${BF:-$HOME/ta-wt/cm-scripts/batches.txt}
[ -f "$bf" ] || bf=$(dirname "$0")/batches.txt
cases() { grep "^$1 " "$bf" | cut -d' ' -f2-; }
AB=${AB:-$HOME/ta-wt/cm-scripts/abrun.sh}; [ -x "$AB" ] || AB=$(dirname "$0")/abrun.sh
ML_ROUNDS=$R ML_AB_KERNELS=f16w_coopmat_m128n64,f16w_coopmat_m128n64_mfast,f16w_coopmat_m64n64,f16w_coopmat_m64n64_mfast,f16w_coopmat_aligned,f16w_coopmat_m128n128_mfast \
  timeout 400 $AB "$out" vr_f16w $(cases vr_f16w)
ML_ROUNDS=$R ML_AB_KERNELS=f16w_coopmat_m128n64,f16w_coopmat_m64n64,f16w_a16_coopmat_m64n64_mfast \
  timeout 200 $AB "$out" vr_a16 $(cases vr_a16)
ML_ROUNDS=$R ML_AB_KERNELS=f16w_coopmat_m128n64,f16w_coopmat_m128n64_bk64,f16w_coopmat_m128n64_t256,f16w_coopmat_m64n128,f16w_coopmat_m64n128_w64,f16w_coopmat_m64n64,f16w_coopmat_m64n64_bk64 \
  timeout 400 $AB "$out" vt_f16w $(cases vt_f16w)
