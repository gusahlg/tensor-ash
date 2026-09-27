#!/usr/bin/env bash
# Token agreement + logits dumps; SASS probes; M1 a16 gates; gate-only
# numerics (now incl. the acc16 generator axis).
set -uo pipefail
cd ~/ta-wt/f16acc
S=~/ta-wt/f16acc-scripts; R=~/ta-wt/f16acc-results
[ -x target/release/llama_ash ] || { echo "no build"; exit 1; }
echo "== gen tokens $(date +%T)"
SKIP_PP=1 $S/s4_e2e.sh
echo "== isa $(date +%T)"
$S/s2_isa.sh
echo "== M1 a16 $(date +%T)"
$S/s5_gen.sh a16
echo "== numerics gate $(date +%T)"
ML_NUMERICS_GATE_ONLY=1 RUST_LOG=warn ./target/release/examples/coopmat_acc_numerics > $R/numerics-gate2-4060.md 2>&1; echo "gate exit $?"
grep -c "| ok |" $R/numerics-gate2-4060.md; grep "FAIL" $R/numerics-gate2-4060.md
echo "== end $(date +%T)"
