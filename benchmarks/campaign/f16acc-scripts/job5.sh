#!/usr/bin/env bash
# Rebuild (occupancy example change), SM89 driver register counts for every
# coopmat kernel (compile only, second device), M1 a16 reproduction run.
set -uo pipefail
S=~/ta-wt/f16acc-scripts; R=~/ta-wt/f16acc-results
$S/build.sh || { echo "BUILD FAILED"; exit 1; }
cd ~/ta-wt/f16acc
echo "== occupancy $(date +%T)"
ML_STATS_FILTER=coopmat RUST_LOG=warn ./target/release/examples/coopmat_occupancy > $R/occupancy-4060.txt 2>&1; echo "occ exit $?"; wc -l $R/occupancy-4060.txt
echo "== M1 a16 reproduction $(date +%T)"
SUFFIX=-rep $S/s5_gen.sh a16
echo "== end $(date +%T)"
