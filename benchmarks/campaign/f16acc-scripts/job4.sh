#!/usr/bin/env bash
# Generator: acc16 module axis sweep (a16) and the a32 M1 gates.
set -uo pipefail
S=~/ta-wt/f16acc-scripts
cd ~/ta-wt/f16acc
$S/s6_gen_acc16.sh
$S/s5_gen.sh f16w
echo "== end $(date +%T)"
