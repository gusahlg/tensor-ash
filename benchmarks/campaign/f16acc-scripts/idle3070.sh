#!/usr/bin/env bash
# Exit 0 only if the 3070 looks idle (no big foreign GPU process, low util).
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
big=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1>500' | wc -l)
if [ "${util:-100}" -gt 30 ] || [ "$big" -gt 0 ]; then echo "3070 busy (util $util%, $big big apps) - skipping"; exit 1; fi
exit 0
