# Benchmark Report

This report compares FP32 GEMM throughput for `tensor-ash` against the local framework backends available on this machine.

## Environment

```text
[2026-06-11T11:58:31Z INFO  tensor_ash::context] tensor-ash: using device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329)
[2026-06-11T11:58:31Z INFO  ml_bench::bench] device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496774464, compute_family=2, timestamps=true) slots=2
tensor-ash self-check
status: OK
selected: device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496774464, compute_family=2, timestamps=true)
executor_slots: 2
glslc: /nix/store/x90sdkmir007jaxx1s893iqgw5kxxc5w-shaderc-2026.1-bin/bin/glslc
vulkaninfo: /nix/store/pq7kd4gdh4ll73d83mlwhhzbjdl6fvl9-vulkan-tools-1.4.341.0/bin/vulkaninfo
LD_LIBRARY_PATH: /nix/store/d6vnmfmcz5b299180issiaad4m96wh8k-vulkan-loader-1.4.341.0/lib:/nix/store/chqq8mpmpyfi9kgsngya71akv5xicn03-gcc-15.2.0-lib/lib:/run/opengl-driver/lib:/nix/store/3zqm4hxrxarx85cxd03gj5yvxkzialqj-alsa-lib-1.2.15.3/lib
VK_ICD_FILENAMES: <unset>
```

NVIDIA-SMI GPU summary:

```text
gpu0: name=NVIDIA GeForce RTX 3070, driver=595.71.05, memory_total_mib=8192, temperature_c=38, utilization_pct=0, power_draw_w=29.30, power_limit_w=220.00
```

- Iterations: 30
- Warmup iterations: 10
- Case set: showcase
- CPU library threads: 1
- CPU framework rows: enabled
- Python GPU framework rows: enabled

- Peak FP32 throughput used for `% peak`: 20.32 TFLOPS

## Results

| case | library | status | gpu ms | wall ms | host overhead ms | TFLOPS | % peak | details |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| square_128 | tensor-ash | ok | 0.006 | 0.027 | 0.020 | 0.668735 | 3.3% | NVIDIA GeForce RTX 3070 (discrete) |
| square_256 | tensor-ash | ok | 0.011 | 0.030 | 0.019 | 2.995931 | 14.7% | NVIDIA GeForce RTX 3070 (discrete) |
| square_512 | tensor-ash | ok | 0.042 | 0.062 | 0.021 | 6.447816 | 31.7% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_4x256 | tensor-ash | ok | 0.024 | 0.043 | 0.019 | 5.667978 | 27.9% | NVIDIA GeForce RTX 3070 (discrete) |
| tall_512x256x256 | tensor-ash | ok | 0.015 | 0.035 | 0.019 | 4.369067 | 21.5% | NVIDIA GeForce RTX 3070 (discrete) |
| odd_255x257x263 | tensor-ash | ok | 0.014 | 0.044 | 0.030 | 2.459433 | 12.1% | NVIDIA GeForce RTX 3070 (discrete) |
| square_1024 | tensor-ash | ok | 0.211 | 0.230 | 0.020 | 10.195816 | 50.2% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_2x512 | tensor-ash | ok | 0.068 | 0.088 | 0.020 | 7.898878 | 38.9% | NVIDIA GeForce RTX 3070 (discrete) |
| skinny_1024x128x512 | tensor-ash | ok | 0.027 | 0.047 | 0.019 | 4.928677 | 24.3% | NVIDIA GeForce RTX 3070 (discrete) |
| wide_128x1024x512 | tensor-ash | ok | 0.028 | 0.047 | 0.019 | 4.848906 | 23.9% | NVIDIA GeForce RTX 3070 (discrete) |
| small_k_1024x1024x64 | tensor-ash | ok | 0.020 | 0.040 | 0.019 | 6.689480 | 32.9% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_513x515x517 | tensor-ash | ok | 0.053 | 0.072 | 0.019 | 5.151962 | 25.4% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_1023x1025x1027 | tensor-ash | ok | 0.248 | 0.268 | 0.020 | 8.689053 | 42.8% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_384 | tensor-ash | ok | 0.022 | 0.041 | 0.019 | 5.173895 | 25.5% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_768 | tensor-ash | ok | 0.108 | 0.128 | 0.019 | 8.366298 | 41.2% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_8x256 | tensor-ash | ok | 0.036 | 0.056 | 0.020 | 7.397362 | 36.4% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_16x128 | tensor-ash | ok | 0.014 | 0.033 | 0.019 | 4.934475 | 24.3% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_32x64 | tensor-ash | ok | 0.006 | 0.025 | 0.019 | 2.962079 | 14.6% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_64x128 | tensor-ash | ok | 0.039 | 0.058 | 0.019 | 6.915588 | 34.0% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_128x64 | tensor-ash | ok | 0.016 | 0.036 | 0.019 | 4.096000 | 20.2% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_2048x512x512 | tensor-ash | ok | 0.109 | 0.129 | 0.019 | 9.811237 | 48.3% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_512x2048x512 | tensor-ash | ok | 0.109 | 0.129 | 0.020 | 9.819851 | 48.3% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_qkv_1024x3072x512 | tensor-ash | ok | 0.309 | 0.332 | 0.023 | 10.412008 | 51.2% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b32_128 | tensor-ash | ok | 0.021 | 0.041 | 0.020 | 6.393756 | 31.5% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b16_192 | tensor-ash | ok | 0.036 | 0.055 | 0.020 | 6.302661 | 31.0% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b8_192 | tensor-ash | ok | 0.019 | 0.039 | 0.019 | 5.820632 | 28.6% | NVIDIA GeForce RTX 3070 (discrete) |
| square_128 | numpy | ok | 0.039 |  |  | 0.107129 | 0.5% | numpy 2.4.4, threads=1 |
| square_256 | numpy | ok | 0.288 |  |  | 0.116355 | 0.6% | numpy 2.4.4, threads=1 |
| square_512 | numpy | ok | 2.247 |  |  | 0.119475 | 0.6% | numpy 2.4.4, threads=1 |
| batched_4x256 | numpy | ok | 1.154 |  |  | 0.116328 | 0.6% | numpy 2.4.4, threads=1 |
| tall_512x256x256 | numpy | ok | 0.569 |  |  | 0.117928 | 0.6% | numpy 2.4.4, threads=1 |
| odd_255x257x263 | numpy | ok | 0.309 |  |  | 0.111379 | 0.5% | numpy 2.4.4, threads=1 |
| square_1024 | numpy | ok | 17.740 |  |  | 0.121051 | 0.6% | numpy 2.4.4, threads=1 |
| batched_2x512 | numpy | ok | 4.499 |  |  | 0.119329 | 0.6% | numpy 2.4.4, threads=1 |
| skinny_1024x128x512 | numpy | ok | 1.159 |  |  | 0.115768 | 0.6% | numpy 2.4.4, threads=1 |
| wide_128x1024x512 | numpy | ok | 1.185 |  |  | 0.113241 | 0.6% | numpy 2.4.4, threads=1 |
| small_k_1024x1024x64 | numpy | ok | 1.206 |  |  | 0.111327 | 0.5% | numpy 2.4.4, threads=1 |
| non_pow2_513x515x517 | numpy | ok | 2.341 |  |  | 0.116718 | 0.6% | numpy 2.4.4, threads=1 |
| non_pow2_1023x1025x1027 | numpy | ok | 18.129 |  |  | 0.118800 | 0.6% | numpy 2.4.4, threads=1 |
| medium_384 | numpy | ok | 0.960 |  |  | 0.117918 | 0.6% | numpy 2.4.4, threads=1 |
| medium_768 | numpy | ok | 7.489 |  |  | 0.120976 | 0.6% | numpy 2.4.4, threads=1 |
| batched_8x256 | numpy | ok | 2.314 |  |  | 0.116013 | 0.6% | numpy 2.4.4, threads=1 |
| batched_16x128 | numpy | ok | 0.627 |  |  | 0.106971 | 0.5% | numpy 2.4.4, threads=1 |
| batched_32x64 | numpy | ok | 0.177 |  |  | 0.094884 | 0.5% | numpy 2.4.4, threads=1 |
| batched_64x128 | numpy | ok | 2.588 |  |  | 0.103714 | 0.5% | numpy 2.4.4, threads=1 |
| batched_128x64 | numpy | ok | 0.700 |  |  | 0.095838 | 0.5% | numpy 2.4.4, threads=1 |
| attn_proj_2048x512x512 | numpy | ok | 8.975 |  |  | 0.119634 | 0.6% | numpy 2.4.4, threads=1 |
| attn_proj_512x2048x512 | numpy | ok | 8.961 |  |  | 0.119827 | 0.6% | numpy 2.4.4, threads=1 |
| attn_qkv_1024x3072x512 | numpy | ok | 26.370 |  |  | 0.122156 | 0.6% | numpy 2.4.4, threads=1 |
| tiny_b32_128 | numpy | ok | 1.249 |  |  | 0.107487 | 0.5% | numpy 2.4.4, threads=1 |
| tiny_b16_192 | numpy | ok | 1.995 |  |  | 0.113517 | 0.6% | numpy 2.4.4, threads=1 |
| tiny_b8_192 | numpy | ok | 0.996 |  |  | 0.113750 | 0.6% | numpy 2.4.4, threads=1 |
| square_128 | torch_cpu | ok | 0.041 |  |  | 0.103497 | 0.5% | torch 2.11.0+cu128, threads=1 |
| square_256 | torch_cpu | ok | 0.287 |  |  | 0.116809 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_512 | torch_cpu | ok | 2.140 |  |  | 0.125448 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_4x256 | torch_cpu | ok | 1.150 |  |  | 0.116730 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tall_512x256x256 | torch_cpu | ok | 0.572 |  |  | 0.117265 | 0.6% | torch 2.11.0+cu128, threads=1 |
| odd_255x257x263 | torch_cpu | ok | 0.311 |  |  | 0.110773 | 0.5% | torch 2.11.0+cu128, threads=1 |
| square_1024 | torch_cpu | ok | 17.378 |  |  | 0.123577 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_2x512 | torch_cpu | ok | 4.306 |  |  | 0.124670 | 0.6% | torch 2.11.0+cu128, threads=1 |
| skinny_1024x128x512 | torch_cpu | ok | 1.175 |  |  | 0.114230 | 0.6% | torch 2.11.0+cu128, threads=1 |
| wide_128x1024x512 | torch_cpu | ok | 1.142 |  |  | 0.117512 | 0.6% | torch 2.11.0+cu128, threads=1 |
| small_k_1024x1024x64 | torch_cpu | ok | 1.095 |  |  | 0.122584 | 0.6% | torch 2.11.0+cu128, threads=1 |
| non_pow2_513x515x517 | torch_cpu | ok | 2.372 |  |  | 0.115186 | 0.6% | torch 2.11.0+cu128, threads=1 |
| non_pow2_1023x1025x1027 | torch_cpu | ok | 17.828 |  |  | 0.120811 | 0.6% | torch 2.11.0+cu128, threads=1 |
| medium_384 | torch_cpu | ok | 0.938 |  |  | 0.120683 | 0.6% | torch 2.11.0+cu128, threads=1 |
| medium_768 | torch_cpu | ok | 7.258 |  |  | 0.124825 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_8x256 | torch_cpu | ok | 2.289 |  |  | 0.117290 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_16x128 | torch_cpu | ok | 0.575 |  |  | 0.116734 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_32x64 | torch_cpu | ok | 0.141 |  |  | 0.118942 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_64x128 | torch_cpu | ok | 2.421 |  |  | 0.110895 | 0.5% | torch 2.11.0+cu128, threads=1 |
| batched_128x64 | torch_cpu | ok | 0.575 |  |  | 0.116748 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_proj_2048x512x512 | torch_cpu | ok | 8.875 |  |  | 0.120991 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_proj_512x2048x512 | torch_cpu | ok | 8.966 |  |  | 0.119755 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_qkv_1024x3072x512 | torch_cpu | ok | 27.141 |  |  | 0.118684 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b32_128 | torch_cpu | ok | 1.207 |  |  | 0.111185 | 0.5% | torch 2.11.0+cu128, threads=1 |
| tiny_b16_192 | torch_cpu | ok | 1.950 |  |  | 0.116143 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b8_192 | torch_cpu | ok | 0.978 |  |  | 0.115825 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_128 | cublas_pure | ok | 0.008 |  |  | 0.522199 | 2.6% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_256 | cublas_pure | ok | 0.011 |  |  | 2.978909 | 14.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_512 | cublas_pure | ok | 0.036 |  |  | 7.489828 | 36.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_4x256 | cublas_pure | ok | 0.037 |  |  | 3.640889 | 17.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tall_512x256x256 | cublas_pure | ok | 0.015 |  |  | 4.568959 | 22.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| odd_255x257x263 | cublas_pure | ok | 0.012 |  |  | 2.857378 | 14.1% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_1024 | cublas_pure | ok | 0.195 |  |  | 11.001453 | 54.1% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_2x512 | cublas_pure | ok | 0.069 |  |  | 7.825194 | 38.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| skinny_1024x128x512 | cublas_pure | ok | 0.025 |  |  | 5.461333 | 26.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| wide_128x1024x512 | cublas_pure | ok | 0.024 |  |  | 5.518821 | 27.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| small_k_1024x1024x64 | cublas_pure | ok | 0.023 |  |  | 5.957818 | 29.3% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_513x515x517 | cublas_pure | ok | 0.050 |  |  | 5.475818 | 26.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_1023x1025x1027 | cublas_pure | ok | 0.210 |  |  | 10.267797 | 50.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_384 | cublas_pure | ok | 0.020 |  |  | 5.529600 | 27.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_768 | cublas_pure | ok | 0.091 |  |  | 9.958337 | 49.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_8x256 | cublas_pure | ok | 0.039 |  |  | 6.898527 | 33.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_16x128 | cublas_pure | ok | 0.022 |  |  | 3.004516 | 14.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_32x64 | cublas_pure | ok | 0.014 |  |  | 1.170286 | 5.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_64x128 | cublas_pure | ok | 0.043 |  |  | 6.241524 | 30.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_128x64 | cublas_pure | ok | 0.035 |  |  | 1.927529 | 9.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_2048x512x512 | cublas_pure | ok | 0.110 |  |  | 9.748528 | 48.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_512x2048x512 | cublas_pure | ok | 0.105 |  |  | 10.180350 | 50.1% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_qkv_1024x3072x512 | cublas_pure | ok | 0.282 |  |  | 11.439011 | 56.3% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b32_128 | cublas_pure | ok | 0.027 |  |  | 5.047297 | 24.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b16_192 | cublas_pure | ok | 0.053 |  |  | 4.253539 | 20.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b8_192 | cublas_pure | ok | 0.030 |  |  | 3.813517 | 18.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_128 | torch_cuda | ok | 0.014 |  |  | 0.303407 | 1.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_256 | torch_cuda | ok | 0.015 |  |  | 2.212186 | 10.9% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_512 | torch_cuda | ok | 0.041 |  |  | 6.600006 | 32.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_4x256 | torch_cuda | ok | 0.043 |  |  | 3.137101 | 15.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tall_512x256x256 | torch_cuda | ok | 0.019 |  |  | 3.560530 | 17.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| odd_255x257x263 | torch_cuda | ok | 0.016 |  |  | 2.198432 | 10.8% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_1024 | torch_cuda | ok | 0.201 |  |  | 10.677623 | 52.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_2x512 | torch_cuda | ok | 0.075 |  |  | 7.118038 | 35.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| skinny_1024x128x512 | torch_cuda | ok | 0.028 |  |  | 4.809982 | 23.7% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| wide_128x1024x512 | torch_cuda | ok | 0.028 |  |  | 4.809982 | 23.7% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| small_k_1024x1024x64 | torch_cuda | ok | 0.027 |  |  | 4.999171 | 24.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| non_pow2_513x515x517 | torch_cuda | ok | 0.054 |  |  | 5.063346 | 24.9% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| non_pow2_1023x1025x1027 | torch_cuda | ok | 0.215 |  |  | 10.023143 | 49.3% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| medium_384 | torch_cuda | ok | 0.025 |  |  | 4.519724 | 22.2% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| medium_768 | torch_cuda | ok | 0.096 |  |  | 9.415215 | 46.3% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_8x256 | torch_cuda | ok | 0.047 |  |  | 5.761406 | 28.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_16x128 | torch_cuda | ok | 0.028 |  |  | 2.391279 | 11.8% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_32x64 | torch_cuda | ok | 0.015 |  |  | 1.087734 | 5.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_64x128 | torch_cuda | ok | 0.050 |  |  | 5.384216 | 26.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_128x64 | torch_cuda | ok | 0.041 |  |  | 1.623183 | 8.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_proj_2048x512x512 | torch_cuda | ok | 0.115 |  |  | 9.357064 | 46.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_proj_512x2048x512 | torch_cuda | ok | 0.110 |  |  | 9.765550 | 48.1% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_qkv_1024x3072x512 | torch_cuda | ok | 0.288 |  |  | 11.189784 | 55.1% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b32_128 | torch_cuda | ok | 0.034 |  |  | 3.960627 | 19.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b16_192 | torch_cuda | ok | 0.061 |  |  | 3.735033 | 18.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b8_192 | torch_cuda | ok | 0.036 |  |  | 3.148527 | 15.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_128 | cupy_cuda | ok | 0.048 |  |  | 0.087909 | 0.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_256 | cupy_cuda | ok | 0.050 |  |  | 0.671303 | 3.3% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_512 | cupy_cuda | ok | 0.076 |  |  | 3.533533 | 17.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_4x256 | cupy_cuda | ok | 0.084 |  |  | 1.602102 | 7.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tall_512x256x256 | cupy_cuda | ok | 0.056 |  |  | 1.201806 | 5.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| odd_255x257x263 | cupy_cuda | ok | 0.052 |  |  | 0.657651 | 3.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_1024 | cupy_cuda | ok | 0.242 |  |  | 8.888591 | 43.7% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_2x512 | cupy_cuda | ok | 0.118 |  |  | 4.555312 | 22.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| skinny_1024x128x512 | cupy_cuda | ok | 0.067 |  |  | 1.997288 | 9.8% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| wide_128x1024x512 | cupy_cuda | ok | 0.067 |  |  | 2.013588 | 9.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| small_k_1024x1024x64 | cupy_cuda | ok | 0.065 |  |  | 2.080508 | 10.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| non_pow2_513x515x517 | cupy_cuda | ok | 0.093 |  |  | 2.945756 | 14.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| non_pow2_1023x1025x1027 | cupy_cuda | ok | 0.256 |  |  | 8.422651 | 41.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| medium_384 | cupy_cuda | ok | 0.063 |  |  | 1.805584 | 8.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| medium_768 | cupy_cuda | ok | 0.135 |  |  | 6.694621 | 32.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_8x256 | cupy_cuda | ok | 0.086 |  |  | 3.125413 | 15.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_16x128 | cupy_cuda | ok | 0.069 |  |  | 0.972705 | 4.8% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_32x64 | cupy_cuda | ok | 0.060 |  |  | 0.277695 | 1.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_64x128 | cupy_cuda | ok | 0.090 |  |  | 2.991658 | 14.7% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_128x64 | cupy_cuda | ok | 0.080 |  |  | 0.838525 | 4.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_proj_2048x512x512 | cupy_cuda | ok | 0.152 |  |  | 7.076009 | 34.8% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_proj_512x2048x512 | cupy_cuda | ok | 0.147 |  |  | 7.281778 | 35.8% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_qkv_1024x3072x512 | cupy_cuda | ok | 0.327 |  |  | 9.863149 | 48.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b32_128 | cupy_cuda | ok | 0.072 |  |  | 1.870787 | 9.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b16_192 | cupy_cuda | ok | 0.099 |  |  | 2.292062 | 11.3% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b8_192 | cupy_cuda | ok | 0.074 |  |  | 1.535334 | 7.6% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_128 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| square_256 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| square_512 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| batched_4x256 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| tall_512x256x256 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| odd_255x257x263 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| square_1024 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| batched_2x512 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| skinny_1024x128x512 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| wide_128x1024x512 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| small_k_1024x1024x64 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| non_pow2_513x515x517 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| non_pow2_1023x1025x1027 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| medium_384 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| medium_768 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| batched_8x256 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| batched_16x128 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| batched_32x64 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| batched_64x128 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| batched_128x64 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| attn_proj_2048x512x512 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| attn_proj_512x2048x512 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| attn_qkv_1024x3072x512 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| tiny_b32_128 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| tiny_b16_192 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| tiny_b8_192 | jax | skipped |  |  |  |  |  | No module named 'jax' |
| square_128 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| square_256 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| square_512 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| batched_4x256 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| tall_512x256x256 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| odd_255x257x263 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| square_1024 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| batched_2x512 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| skinny_1024x128x512 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| wide_128x1024x512 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| small_k_1024x1024x64 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| non_pow2_513x515x517 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| non_pow2_1023x1025x1027 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| medium_384 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| medium_768 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| batched_8x256 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| batched_16x128 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| batched_32x64 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| batched_64x128 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| batched_128x64 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| attn_proj_2048x512x512 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| attn_proj_512x2048x512 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| attn_qkv_1024x3072x512 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| tiny_b32_128 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| tiny_b16_192 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |
| tiny_b8_192 | tensorflow | skipped |  |  |  |  |  | No module named 'tensorflow' |

## Transfer

| status | bytes | iters | upload GiB/s | download GiB/s | details |
| --- | ---: | ---: | ---: | ---: | --- |
| ok | 67108864 | 30 | 9.703 | 10.244 | NVIDIA GeForce RTX 3070 (discrete) |

## Analysis

- `tensor-ash` used `NVIDIA GeForce RTX 3070 (discrete)`, so the Vulkan measurements reflect real GPU kernel timings on this host.
- Actual GPU framework comparisons succeeded for: `cublas_pure`, `cupy_cuda`, `torch_cuda`.
- Largest gap: `medium_768` is 1.2x faster in `cublas_pure` than `tensor-ash` in this environment.
- `tensor-ash` is the fastest measured backend on 14/26 benchmark cases.
- Throughput ratio versus `cublas_pure` across 26 shared cases: 0.84x to 2.53x, geometric mean 1.12x.
- Throughput ratio versus `torch_cuda` across 26 shared cases: 0.87x to 2.72x, geometric mean 1.31x.
- Throughput ratio versus `cupy_cuda` across 26 shared cases: 1.03x to 10.67x, geometric mean 2.62x.
- Throughput ratio versus `numpy` across 26 shared cases: 6.24x to 85.24x, geometric mean 48.03x.
- Throughput ratio versus `torch_cpu` across 26 shared cases: 6.46x to 87.73x, geometric mean 46.40x.
- Median `tensor-ash` host/submission overhead was 0.019 ms per synchronous call; GPU timestamp TFLOPS excludes that overhead.
- Some libraries were skipped because their Python modules or device backends were unavailable: `jax`, `tensorflow`.
- JAX and TensorFlow rows are included when those modules are installed; skipped rows mean the local Python environment does not provide them.
- Transfer staging bandwidth measured 9.70 GiB/s upload and 10.24 GiB/s download for 67108864 bytes.

## Optimization Gameplan

1. Keep benchmarking on this discrete-GPU baseline and tune the shape-based shader selector with larger production-like matrix sizes.
2. Use `scripts/tune_kernels.py` before accepting changes to the automatic selector.
3. Focus the next shader pass on large square GEMMs, where PyTorch CUDA still has the largest lead.
4. Keep PyTorch CUDA and CuPy CUDA rows in regular benchmark runs so Vulkan changes are compared against cuBLAS-backed GPU compute.
5. Add CI checks for `cargo fmt`, `cargo clippy`, CPU tests, and an optional GPU correctness job.
