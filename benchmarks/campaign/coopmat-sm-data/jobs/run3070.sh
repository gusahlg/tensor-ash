#!/usr/bin/env bash
# run3070.sh <sub> <batch>...  one gpu3070 hold per batch (short batches), validated 34ad9f1 binary
SP=/home/gusahlg/repos/tensor-ash-wt/campaign-tools; W=$SP/work/coopmat
D=/home/gusahlg/repos/tensor-ash-wt/c-coopmat-sm/benchmarks/campaign/coopmat-sm-data
sub=$1; shift; cd $W
for b in "$@"; do
  GPU_WAIT=3600 BIN=$W/bin/ml_bench-34ad9f1 $SP/bin/gpu3070 $D/verify.sh $W/results/$sub $b
  rc=$?; [ $rc = 75 ] && { echo "EMBARGO rc=75, stopping"; exit 75; }
  sleep 15
done
echo "RUN3070_DONE $sub"
