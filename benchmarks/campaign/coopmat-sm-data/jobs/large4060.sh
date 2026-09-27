#!/usr/bin/env bash
# 128x128 vs 128x64 on large shapes, 15 interleaved rounds, warm then cold, RTX 4060 (0fc29f9 ml_bench)
cd ~/ta-wt/cm-results
export LD_LIBRARY_PATH=/etc/profiles/per-user/gusahlg/lib:/run/opengl-driver/lib
cases=$(grep "^large " ~/ta-wt/cm-results/large.txt | cut -d' ' -f2-)
export ML_AB_KERNELS=f16w_coopmat_aligned,f16w_coopmat_m128n64 ML_ROUNDS=15 BIN=~/ta-wt/cm-results/bin/ml_bench-0fc29f9
echo 0fc29f9 > GIT_HEAD
TA_JOB=gpu4060-coopmat-large ~/ta-heavy.sh flock -w 7200 /tmp/gusahlg-gpu-4060.lock bash -c "
  timeout 280 ~/ta-wt/cm-scripts/abrun.sh ~/ta-wt/cm-results/large-4060 warm $cases; sleep 10;
  ML_AB_COLD_MB=64 timeout 280 ~/ta-wt/cm-scripts/abrun.sh ~/ta-wt/cm-results/large-4060 cold $cases"
echo LARGE_DONE
