#!/usr/bin/env bash
# 128x128 vs 128x64 on large 3070 shapes, 15 interleaved rounds, warm then cold (one hold each)
SP=/home/gusahlg/repos/tensor-ash-wt/campaign-tools; W=$SP/work/coopmat
D=/home/gusahlg/repos/tensor-ash-wt/c-coopmat-sm/benchmarks/campaign/coopmat-sm-data
cd $W; cases=$(grep "^large " $W/large.txt | cut -d' ' -f2-)
export ML_AB_KERNELS=f16w_coopmat_aligned,f16w_coopmat_m128n64 ML_ROUNDS=15
GPU_WAIT=3600 $SP/bin/gpu3070 timeout 400 $D/abrun.sh $W/results/large-3070 warm $cases; sleep 20
ML_AB_COLD_MB=64 GPU_WAIT=3600 $SP/bin/gpu3070 timeout 400 $D/abrun.sh $W/results/large-3070 cold $cases
echo LARGE_DONE
