#!/usr/bin/env bash
# Regenerate the checked-in PTX from the CUDA sources.  Only needed when
# a .cu file changes: tensor-ash itself embeds the .ptx text and never
# needs a CUDA toolkit.  The driver JIT-compiles the PTX for the actual
# GPU (sm_86 / sm_89 / ...) at vkCreateCudaModuleNV time.
#
#   ptx/build.sh            # PTX (target sm_80) for every *.cu
#   ptx/build.sh cubin DIR  # also sm_86/sm_89 cubins + ptxas -v stats into DIR (probe only)
#
# CUDA_HOME may point at a toolkit whose include/ has cuda_runtime.h
# (NixOS: a cuda-merged store path); nvcc/ptxas come from PATH.
set -euo pipefail
cd "$(dirname "$0")"
inc=()
if [ -n "${CUDA_HOME:-}" ]; then inc=(-I"$CUDA_HOME/include"); fi
for cu in *.cu; do
  stem="${cu%.cu}"
  nvcc "${inc[@]}" -std=c++17 -O3 -ptx -arch=sm_80 -o "$stem.ptx" "$cu"
  echo "ptx/$stem.ptx: $(grep -c '^\.visible \.entry' "$stem.ptx") entries"
done
if [ "${1:-}" = cubin ]; then
  out="${2:?cubin DIR}"
  mkdir -p "$out"
  for cu in *.cu; do
    stem="${cu%.cu}"
    for sm in 86 89; do
      nvcc "${inc[@]}" -std=c++17 -O3 -cubin -arch=sm_$sm -Xptxas -v \
        -o "$out/${stem}_sm$sm.cubin" "$cu" 2> "$out/${stem}_sm$sm.ptxas.txt"
    done
    nvcc "${inc[@]}" -std=c++17 -O3 -fatbin \
      -gencode arch=compute_80,code=compute_80 -gencode arch=compute_86,code=sm_86 \
      -gencode arch=compute_89,code=sm_89 -o "$out/${stem}.fatbin" "$cu"
  done
  echo "cubins/fatbins in $out"
fi
