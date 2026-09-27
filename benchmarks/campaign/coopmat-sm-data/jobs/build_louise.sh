#!/usr/bin/env bash
# build + test the refit on louise (cm-var tree); run detached
cd ~/ta-wt/cm-var
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=10
export LD_LIBRARY_PATH=/etc/profiles/per-user/gusahlg/lib:/run/opengl-driver/lib
echo "== $(date -Is) $(cat GIT_HEAD)"
TA_JOB=build-coopmat ~/ta-heavy.sh flock -w 10800 /tmp/ta-build-louise.lock bash -c '
  cargo test --release --lib -- pipeline::routing context::device 2>&1 | tail -8 &&
  cargo build --release -p llama-ash -p ml-bench 2>&1 | tail -3 &&
  cargo test --release --test correctness --no-run 2>&1 | tail -2'
echo "build rc=$?"
TA_JOB=gpu4060-coopmat-tests ~/ta-heavy.sh flock -w 7200 /tmp/gusahlg-gpu-4060.lock \
  cargo test --release --test correctness -- --ignored f16_routes_pick coopmat_asymmetric 2>&1 | tail -8
echo "gputest rc=$?"
echo BUILD_ALLDONE
