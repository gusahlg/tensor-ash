#!/usr/bin/env bash
# 4060: new-binary auto check (cold a16_512 + m512) then pp512 A/B base vs 100df0d vs 0fc29f9
cd ~/ta-wt/cm-var
export LD_LIBRARY_PATH=/etc/profiles/per-user/gusahlg/lib:/run/opengl-driver/lib
until grep -q BUILD_ALLDONE ~/ta-wt/cm-results/build_louise.log; do sleep 10; done
cp -p target/release/llama_ash ~/ta-wt/cm-results/bin/llama_ash-0fc29f9
cp -p target/release/ml_bench ~/ta-wt/cm-results/bin/ml_bench-0fc29f9
cp -p ~/ta-wt/cm-sm/target/release/llama_ash ~/ta-wt/cm-results/bin/llama_ash-100df0d
B=~/ta-wt/cm-results/bin
TA_JOB=gpu4060-coopmat-auto ~/ta-heavy.sh flock -w 7200 /tmp/gusahlg-gpu-4060.lock \
  env ML_AB_COLD_MB=64 BIN=$B/ml_bench-0fc29f9 GAP=3 ~/ta-wt/cm-scripts/verify.sh ~/ta-wt/cm-results/4060new-cold a16_512 m512 a16_128 a16_1920
sleep 10
TA_JOB=gpu4060-coopmat-pp512 ~/ta-heavy.sh flock -w 7200 /tmp/gusahlg-gpu-4060.lock \
  timeout 580 ~/ta-wt/cm-scripts/llama_env_ab.sh 6 ~/ta-wt/cm-results/pp512_final_4060.log \
  base=ML_X=0:$B/llama_ash-7cfca58 old=ML_X=0:$B/llama_ash-100df0d new=ML_X=0:$B/llama_ash-0fc29f9
echo PP_ALLDONE
