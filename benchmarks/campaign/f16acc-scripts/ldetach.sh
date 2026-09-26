#!/usr/bin/env bash
# ldetach <job-name> <script-on-louise>: run a job script on louise-pc
# detached (setsid nohup) under the heavy + 4060 locks; log to
# ~/ta-wt/f16acc-logs/<job>.log, final line "EXIT <rc>".
job="$1"; script="$2"
ssh -o BatchMode=yes louise-pc "mkdir -p ~/ta-wt/f16acc-logs; cd ~/ta-wt/f16acc && LD_LIBRARY_PATH=/etc/profiles/per-user/gusahlg/lib:/run/opengl-driver/lib TA_JOB='f16acc-$job' setsid nohup bash -c '~/ta-heavy.sh flock -w 14400 /tmp/gusahlg-gpu-4060.lock bash -l $script; echo EXIT \$?' > ~/ta-wt/f16acc-logs/$job.log 2>&1 < /dev/null & echo launched $job"
