#!/usr/bin/env bash
# Error figures printed by the mixed-graph test, and the veto/plain logs (4060, under lgpu).
cd ~/ta-wt/c-ptx
T=$(ls -t target/release/deps/correctness-* | grep -v '\.d$' | head -1)
ML_PTX=1 timeout 200 $T --ignored --test-threads=1 --nocapture ptx::ptx_route_in_mixed 2>&1 | grep -E "rel-RMS|test result"
echo "## plain / veto device creation logs"
RUST_LOG=info timeout 100 $T --ignored --test-threads=1 --nocapture ptx::ptx_policy 2>&1 | grep -E "PTX|ptx|skipping|test result"
ML_PTX=1 ML_NO_PTX=1 RUST_LOG=info timeout 100 $T --ignored --test-threads=1 --nocapture ptx::ptx_policy 2>&1 | grep -E "PTX|ptx|skipping|test result"
