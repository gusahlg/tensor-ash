#!/usr/bin/env bash
# Build inside an already-held heavy lock.  rsync preserves source mtimes,
# so a shader created before the previous build-script run would not
# trigger build.rs: touch every shader newer than the last OUT_DIR spv set.
set -uo pipefail
cd ~/ta-wt/f16acc
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_BUILD_JOBS=10
for f in shaders/*.comp; do
  stem=$(basename "$f" .comp)
  ls target/release/build/tensor-ash-*/out/"$stem".spv >/dev/null 2>&1 || { echo "touch $f (no spv yet)"; touch "$f"; }
done
echo "== build $(date +%T)"
nice -n 5 cargo build --release -p ml-bench -p llama-ash -p tensor-ash --bins --examples > /tmp/f16acc-build.log 2>&1
rc=$?
grep -E "^error|Finished|could not compile" /tmp/f16acc-build.log | head -5
exit $rc
