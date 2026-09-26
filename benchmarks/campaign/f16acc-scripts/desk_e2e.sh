#!/usr/bin/env bash
# 3070 e2e batch (validated on the 4060 first): pp512 rounds + tokens.
set -uo pipefail
SP=/tmp/claude-1000/-home-gusahlg-repos-tensor-ash/36c7b90e-5e00-4f5c-86e3-41f713a8da23/scratchpad
cd ~/repos/tensor-ash-wt/c-f16acc
$SP/f16acc/scripts/idle3070.sh || exit 3
WT=$PWD OUT=$SP/f16acc/results P=$SP/f16acc/prompts TAG=${TAG:-3070-b3} PAIRS=${PAIRS:-3} SKIP_GEN=${SKIP_GEN:-} SKIP_PP=${SKIP_PP:-} bash $SP/f16acc/scripts/s4_e2e.sh 2>&1 | sed 's/\[2026[^]]*\] //'
$SP/f16acc/scripts/idle3070.sh || echo "WARNING: 3070 became busy during the batch"
