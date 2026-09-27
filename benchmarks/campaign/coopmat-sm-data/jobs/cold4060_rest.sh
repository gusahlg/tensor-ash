#!/usr/bin/env bash
# successor #5: finish the 4060 cold sweep (cm-var binary, 100df0d+cold-ab)
cd ~/ta-wt/cm-var
export LD_LIBRARY_PATH=/etc/profiles/per-user/gusahlg/lib:/run/opengl-driver/lib
for chunk in "m1024 m2048 t128 t1920" "a16_128 a16_256 a16_512 a16_1024 a16_1920"; do
  TA_JOB=gpu4060-coopmat-cold ~/ta-heavy.sh flock -w 7200 /tmp/gusahlg-gpu-4060.lock env ML_AB_COLD_MB=64 GAP=5 ~/ta-wt/cm-scripts/verify.sh ~/ta-wt/cm-results/4060cold $chunk
  sleep 20
done
echo ALLDONE
