#!/usr/bin/env bash
# SASS probes (compile only), generator M1 a16 gates, numerics gate incl.
# the acc16 generator axis.
set -uo pipefail
S=~/ta-wt/f16acc-scripts; R=~/ta-wt/f16acc-results
cd ~/ta-wt/f16acc
echo "== isa $(date +%T)"
$S/s2_isa.sh
echo "== numerics gate $(date +%T)"
ML_NUMERICS_GATE_ONLY=1 RUST_LOG=warn ./target/release/examples/coopmat_acc_numerics > $R/numerics-gate2-4060.md 2>&1; echo "gate exit $?"
grep -c "| ok |" $R/numerics-gate2-4060.md; grep "FAIL" $R/numerics-gate2-4060.md
echo "== ab auto f16w (acc-aware tiles) $(date +%T)"
CASES="qo,1,512,2048,2048,f16w qkv,1,512,2560,2048,f16w gu,1,512,5632,2048,f16w dn,1,512,2048,5632,f16w s2k,1,2048,2048,2048,f16w s4k,1,4096,4096,4096,f16w"
ML_ROUNDS=7 ML_AB_KERNELS=auto,auto@f16,auto@promote RUST_LOG=warn ./target/release/ml_bench ab $CASES | tee $R/ab-auto3-f16w-4060.txt
echo "== M1 a16 $(date +%T)"
$S/s5_gen.sh a16
echo "== end $(date +%T)"
