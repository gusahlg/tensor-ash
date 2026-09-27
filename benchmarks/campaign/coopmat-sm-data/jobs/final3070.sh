#!/usr/bin/env bash
# Wait (<= 45 min) for the desktop embargo to be gone for 5 min, then run the pending 3070 items,
# one gpu3070 hold each, re-checking the embargo inside the lock.
SP=/home/gusahlg/repos/tensor-ash-wt/campaign-tools; W=$SP/work/coopmat
D=/home/gusahlg/repos/tensor-ash-wt/c-coopmat-sm/benchmarks/campaign/coopmat-sm-data
emb() { [ -e $SP/GPU3070_EMBARGO ] || [ -e /tmp/gusahlg-desktop-embargo ]; }
clear=0; for i in $(seq 1 45); do if emb; then clear=0; else clear=$((clear+1)); fi; [ $clear -ge 5 ] && break; sleep 60; done
[ $clear -ge 5 ] || { echo "STILL_EMBARGOED $(date +%T)"; exit 75; }
echo "embargo clear since 5 min, $(date +%T)"
guard='[ -e '$SP'/GPU3070_EMBARGO ] || [ -e /tmp/gusahlg-desktop-embargo ] && { echo EMBARGO-inside; exit 75; };'
cd $W
GPU_WAIT=1800 $SP/bin/gpu3070 bash -c "$guard timeout 580 $D/llama_env_ab.sh 5 $W/results/pp512_final_3070.log base=ML_X=0:$W/bin/llama_ash-7cfca58 new=ML_X=0:$W/bin/llama_ash-0fc29f9" || { echo "pp rc=$?"; exit 75; }
sleep 20
cases=$(grep "^large " $W/large.txt | cut -d' ' -f2-)
for mode in warm cold; do
  cold=0; [ $mode = cold ] && cold=64
  GPU_WAIT=1800 ML_AB_COLD_MB=$cold ML_AB_KERNELS=f16w_coopmat_aligned,f16w_coopmat_m128n64 ML_ROUNDS=15 BIN=$W/bin/ml_bench-0fc29f9 \
    $SP/bin/gpu3070 bash -c "$guard timeout 400 $D/abrun.sh $W/results/large-3070 $mode $cases" || { echo "large rc=$?"; exit 75; }
  sleep 20
done
for b in a16_128 a16_256 a16_1024 a16_1920 m256 m384 sq; do
  GPU_WAIT=1800 ML_AB_COLD_MB=64 BIN=$W/bin/ml_bench-0fc29f9 $SP/bin/gpu3070 bash -c "$guard $D/verify.sh $W/results/cold-3070b $b" || { echo "cold rc=$?"; exit 75; }
  sleep 15
done
echo FINAL3070_DONE
