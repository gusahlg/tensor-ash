#!/usr/bin/env bash
# Fresh-process reproductions of the S2 signals (a32 module and acc16 axis).
set -uo pipefail
S=~/ta-wt/f16acc-scripts
cd ~/ta-wt/f16acc
SUFFIX=-rep $S/s6_gen_acc16.sh
SUFFIX=-rep $S/s5_gen.sh f16w
echo "== end $(date +%T)"
