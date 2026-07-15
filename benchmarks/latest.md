# Benchmark Report

This report compares FP32 GEMM throughput for `tensor-ash` against the local framework backends available on this machine.

## Environment

```text
[2026-07-15T09:07:33Z INFO  tensor_ash::context] tensor-ash: using device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329)
[2026-07-15T09:07:35Z INFO  ml_bench::bench] device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496921600, compute_family=2, timestamps=true) slots=2 kernel=auto
tensor-ash self-check
status: OK
selected: device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496921600, compute_family=2, timestamps=true)
executor_slots: 2
glslc: /nix/store/x90sdkmir007jaxx1s893iqgw5kxxc5w-shaderc-2026.1-bin/bin/glslc
vulkaninfo: /nix/store/pq7kd4gdh4ll73d83mlwhhzbjdl6fvl9-vulkan-tools-1.4.341.0/bin/vulkaninfo
LD_LIBRARY_PATH: /nix/store/d6vnmfmcz5b299180issiaad4m96wh8k-vulkan-loader-1.4.341.0/lib:/nix/store/chqq8mpmpyfi9kgsngya71akv5xicn03-gcc-15.2.0-lib/lib:/run/opengl-driver/lib:/nix/store/3zqm4hxrxarx85cxd03gj5yvxkzialqj-alsa-lib-1.2.15.3/lib:/nix/store/2kdz3m7ic8w226pcvkz1dlg169v91p6a-zlib-1.3.2/lib
VK_ICD_FILENAMES: <unset>
```

NVIDIA-SMI GPU summary:

```text
gpu0: name=NVIDIA GeForce RTX 3070, driver=595.80, memory_total_mib=8192, temperature_c=37, utilization_pct=0, power_draw_w=43.12, power_limit_w=220.00
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
| square_128 | tensor-ash | ok | 0.006 | 0.039 | 0.032 | 0.648871 | 3.2% | NVIDIA GeForce RTX 3070 (discrete) |
| square_256 | tensor-ash | ok | 0.011 | 0.031 | 0.020 | 3.013149 | 14.8% | NVIDIA GeForce RTX 3070 (discrete) |
| square_512 | tensor-ash | ok | 0.039 | 0.059 | 0.020 | 6.870277 | 33.8% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_4x256 | tensor-ash | ok | 0.022 | 0.040 | 0.019 | 6.213784 | 30.6% | NVIDIA GeForce RTX 3070 (discrete) |
| tall_512x256x256 | tensor-ash | ok | 0.015 | 0.034 | 0.019 | 4.433725 | 21.8% | NVIDIA GeForce RTX 3070 (discrete) |
| odd_255x257x263 | tensor-ash | ok | 0.014 | 0.033 | 0.019 | 2.476394 | 12.2% | NVIDIA GeForce RTX 3070 (discrete) |
| square_1024 | tensor-ash | ok | 0.211 | 0.230 | 0.020 | 10.188077 | 50.1% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_2x512 | tensor-ash | ok | 0.057 | 0.077 | 0.020 | 9.425402 | 46.4% | NVIDIA GeForce RTX 3070 (discrete) |
| skinny_1024x128x512 | tensor-ash | ok | 0.026 | 0.046 | 0.020 | 5.171768 | 25.5% | NVIDIA GeForce RTX 3070 (discrete) |
| wide_128x1024x512 | tensor-ash | ok | 0.026 | 0.046 | 0.019 | 5.090175 | 25.1% | NVIDIA GeForce RTX 3070 (discrete) |
| small_k_1024x1024x64 | tensor-ash | ok | 0.018 | 0.038 | 0.020 | 7.371360 | 36.3% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_513x515x517 | tensor-ash | ok | 0.049 | 0.069 | 0.020 | 5.572324 | 27.4% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_1023x1025x1027 | tensor-ash | ok | 0.249 | 0.269 | 0.020 | 8.664445 | 42.6% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_384 | tensor-ash | ok | 0.021 | 0.041 | 0.020 | 5.436166 | 26.8% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_768 | tensor-ash | ok | 0.108 | 0.128 | 0.020 | 8.351490 | 41.1% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_8x256 | tensor-ash | ok | 0.031 | 0.051 | 0.020 | 8.612534 | 42.4% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_16x128 | tensor-ash | ok | 0.012 | 0.032 | 0.020 | 5.433036 | 26.7% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_32x64 | tensor-ash | ok | 0.006 | 0.025 | 0.019 | 2.978909 | 14.7% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_64x128 | tensor-ash | ok | 0.034 | 0.054 | 0.020 | 7.810622 | 38.4% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_128x64 | tensor-ash | ok | 0.016 | 0.036 | 0.020 | 4.152776 | 20.4% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_2048x512x512 | tensor-ash | ok | 0.107 | 0.126 | 0.020 | 10.064317 | 49.5% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_512x2048x512 | tensor-ash | ok | 0.107 | 0.127 | 0.020 | 10.028222 | 49.4% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_qkv_1024x3072x512 | tensor-ash | ok | 0.297 | 0.324 | 0.027 | 10.840329 | 53.3% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b32_128 | tensor-ash | ok | 0.020 | 0.040 | 0.020 | 6.636557 | 32.7% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b16_192 | tensor-ash | ok | 0.032 | 0.052 | 0.020 | 7.092072 | 34.9% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b8_192 | tensor-ash | ok | 0.017 | 0.037 | 0.020 | 6.481582 | 31.9% | NVIDIA GeForce RTX 3070 (discrete) |
| square_128 | numpy | ok | 0.039 |  |  | 0.106954 | 0.5% | numpy 2.4.4, threads=1 |
| square_256 | numpy | ok | 0.288 |  |  | 0.116517 | 0.6% | numpy 2.4.4, threads=1 |
| square_512 | numpy | ok | 2.099 |  |  | 0.127866 | 0.6% | numpy 2.4.4, threads=1 |
| batched_4x256 | numpy | ok | 1.058 |  |  | 0.126845 | 0.6% | numpy 2.4.4, threads=1 |
| tall_512x256x256 | numpy | ok | 0.544 |  |  | 0.123449 | 0.6% | numpy 2.4.4, threads=1 |
| odd_255x257x263 | numpy | ok | 0.310 |  |  | 0.111135 | 0.5% | numpy 2.4.4, threads=1 |
| square_1024 | numpy | ok | 16.740 |  |  | 0.128285 | 0.6% | numpy 2.4.4, threads=1 |
| batched_2x512 | numpy | ok | 4.219 |  |  | 0.127261 | 0.6% | numpy 2.4.4, threads=1 |
| skinny_1024x128x512 | numpy | ok | 1.101 |  |  | 0.121939 | 0.6% | numpy 2.4.4, threads=1 |
| wide_128x1024x512 | numpy | ok | 1.109 |  |  | 0.120989 | 0.6% | numpy 2.4.4, threads=1 |
| small_k_1024x1024x64 | numpy | ok | 1.156 |  |  | 0.116104 | 0.6% | numpy 2.4.4, threads=1 |
| non_pow2_513x515x517 | numpy | ok | 2.253 |  |  | 0.121268 | 0.6% | numpy 2.4.4, threads=1 |
| non_pow2_1023x1025x1027 | numpy | ok | 17.096 |  |  | 0.125984 | 0.6% | numpy 2.4.4, threads=1 |
| medium_384 | numpy | ok | 0.961 |  |  | 0.117903 | 0.6% | numpy 2.4.4, threads=1 |
| medium_768 | numpy | ok | 7.090 |  |  | 0.127781 | 0.6% | numpy 2.4.4, threads=1 |
| batched_8x256 | numpy | ok | 2.135 |  |  | 0.125752 | 0.6% | numpy 2.4.4, threads=1 |
| batched_16x128 | numpy | ok | 0.626 |  |  | 0.107263 | 0.5% | numpy 2.4.4, threads=1 |
| batched_32x64 | numpy | ok | 0.168 |  |  | 0.099760 | 0.5% | numpy 2.4.4, threads=1 |
| batched_64x128 | numpy | ok | 2.406 |  |  | 0.111583 | 0.5% | numpy 2.4.4, threads=1 |
| batched_128x64 | numpy | ok | 0.663 |  |  | 0.101171 | 0.5% | numpy 2.4.4, threads=1 |
| attn_proj_2048x512x512 | numpy | ok | 8.400 |  |  | 0.127821 | 0.6% | numpy 2.4.4, threads=1 |
| attn_proj_512x2048x512 | numpy | ok | 8.497 |  |  | 0.126370 | 0.6% | numpy 2.4.4, threads=1 |
| attn_qkv_1024x3072x512 | numpy | ok | 25.264 |  |  | 0.127503 | 0.6% | numpy 2.4.4, threads=1 |
| tiny_b32_128 | numpy | ok | 1.252 |  |  | 0.107235 | 0.5% | numpy 2.4.4, threads=1 |
| tiny_b16_192 | numpy | ok | 1.906 |  |  | 0.118834 | 0.6% | numpy 2.4.4, threads=1 |
| tiny_b8_192 | numpy | ok | 0.950 |  |  | 0.119248 | 0.6% | numpy 2.4.4, threads=1 |
| square_128 | torch_cpu | ok | 0.039 |  |  | 0.108495 | 0.5% | torch 2.11.0+cu128, threads=1 |
| square_256 | torch_cpu | ok | 0.273 |  |  | 0.122846 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_512 | torch_cpu | ok | 2.076 |  |  | 0.129294 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_4x256 | torch_cpu | ok | 1.089 |  |  | 0.123227 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tall_512x256x256 | torch_cpu | ok | 0.520 |  |  | 0.129121 | 0.6% | torch 2.11.0+cu128, threads=1 |
| odd_255x257x263 | torch_cpu | ok | 0.290 |  |  | 0.118960 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_1024 | torch_cpu | ok | 16.732 |  |  | 0.128345 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_2x512 | torch_cpu | ok | 4.265 |  |  | 0.125870 | 0.6% | torch 2.11.0+cu128, threads=1 |
| skinny_1024x128x512 | torch_cpu | ok | 1.115 |  |  | 0.120411 | 0.6% | torch 2.11.0+cu128, threads=1 |
| wide_128x1024x512 | torch_cpu | ok | 1.099 |  |  | 0.122167 | 0.6% | torch 2.11.0+cu128, threads=1 |
| small_k_1024x1024x64 | torch_cpu | ok | 1.041 |  |  | 0.128947 | 0.6% | torch 2.11.0+cu128, threads=1 |
| non_pow2_513x515x517 | torch_cpu | ok | 2.211 |  |  | 0.123543 | 0.6% | torch 2.11.0+cu128, threads=1 |
| non_pow2_1023x1025x1027 | torch_cpu | ok | 17.131 |  |  | 0.125726 | 0.6% | torch 2.11.0+cu128, threads=1 |
| medium_384 | torch_cpu | ok | 0.892 |  |  | 0.126903 | 0.6% | torch 2.11.0+cu128, threads=1 |
| medium_768 | torch_cpu | ok | 6.975 |  |  | 0.129885 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_8x256 | torch_cpu | ok | 2.170 |  |  | 0.123711 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_16x128 | torch_cpu | ok | 0.575 |  |  | 0.116752 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_32x64 | torch_cpu | ok | 0.148 |  |  | 0.113432 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_64x128 | torch_cpu | ok | 2.296 |  |  | 0.116928 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_128x64 | torch_cpu | ok | 0.571 |  |  | 0.117454 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_proj_2048x512x512 | torch_cpu | ok | 8.417 |  |  | 0.127562 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_proj_512x2048x512 | torch_cpu | ok | 8.345 |  |  | 0.128668 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_qkv_1024x3072x512 | torch_cpu | ok | 25.979 |  |  | 0.123992 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b32_128 | torch_cpu | ok | 1.146 |  |  | 0.117068 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b16_192 | torch_cpu | ok | 1.848 |  |  | 0.122529 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b8_192 | torch_cpu | ok | 0.928 |  |  | 0.122074 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_128 | cublas_pure | ok | 0.008 |  |  | 0.524288 | 2.6% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_256 | cublas_pure | ok | 0.011 |  |  | 2.978909 | 14.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_512 | cublas_pure | ok | 0.036 |  |  | 7.489828 | 36.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_4x256 | cublas_pure | ok | 0.037 |  |  | 3.640889 | 17.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tall_512x256x256 | cublas_pure | ok | 0.015 |  |  | 4.369067 | 21.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| odd_255x257x263 | cublas_pure | ok | 0.012 |  |  | 2.872617 | 14.1% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_1024 | cublas_pure | ok | 0.196 |  |  | 10.979853 | 54.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_2x512 | cublas_pure | ok | 0.069 |  |  | 7.825194 | 38.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| skinny_1024x128x512 | cublas_pure | ok | 0.025 |  |  | 5.461333 | 26.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| wide_128x1024x512 | cublas_pure | ok | 0.024 |  |  | 5.698782 | 28.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| small_k_1024x1024x64 | cublas_pure | ok | 0.022 |  |  | 5.966293 | 29.4% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_513x515x517 | cublas_pure | ok | 0.050 |  |  | 5.451341 | 26.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_1023x1025x1027 | cublas_pure | ok | 0.209 |  |  | 10.321332 | 50.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_384 | cublas_pure | ok | 0.020 |  |  | 5.529600 | 27.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_768 | cublas_pure | ok | 0.090 |  |  | 10.053818 | 49.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_8x256 | cublas_pure | ok | 0.040 |  |  | 6.721641 | 33.1% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_16x128 | cublas_pure | ok | 0.022 |  |  | 3.013149 | 14.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_32x64 | cublas_pure | ok | 0.014 |  |  | 1.170286 | 5.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_64x128 | cublas_pure | ok | 0.042 |  |  | 6.340596 | 31.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_128x64 | cublas_pure | ok | 0.035 |  |  | 1.932859 | 9.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_2048x512x512 | cublas_pure | ok | 0.109 |  |  | 9.857353 | 48.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_512x2048x512 | cublas_pure | ok | 0.104 |  |  | 10.280157 | 50.6% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_qkv_1024x3072x512 | cublas_pure | ok | 0.282 |  |  | 11.439011 | 56.3% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b32_128 | cublas_pure | ok | 0.027 |  |  | 5.041231 | 24.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b16_192 | cublas_pure | ok | 0.053 |  |  | 4.253539 | 20.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b8_192 | cublas_pure | ok | 0.030 |  |  | 3.813517 | 18.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_128 | torch_cuda | ok | 0.013 |  |  | 0.318136 | 1.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_256 | torch_cuda | ok | 0.015 |  |  | 2.274568 | 11.2% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_512 | torch_cuda | ok | 0.041 |  |  | 6.584464 | 32.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_4x256 | torch_cuda | ok | 0.044 |  |  | 3.021833 | 14.9% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tall_512x256x256 | torch_cuda | ok | 0.019 |  |  | 3.506943 | 17.3% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| odd_255x257x263 | torch_cuda | ok | 0.016 |  |  | 2.124717 | 10.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_1024 | torch_cuda | ok | 0.201 |  |  | 10.667440 | 52.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_2x512 | torch_cuda | ok | 0.075 |  |  | 7.121059 | 35.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| skinny_1024x128x512 | torch_cuda | ok | 0.028 |  |  | 4.733977 | 23.3% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| wide_128x1024x512 | torch_cuda | ok | 0.028 |  |  | 4.755447 | 23.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| small_k_1024x1024x64 | torch_cuda | ok | 0.027 |  |  | 4.946113 | 24.3% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| non_pow2_513x515x517 | torch_cuda | ok | 0.054 |  |  | 5.036461 | 24.8% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| non_pow2_1023x1025x1027 | torch_cuda | ok | 0.215 |  |  | 10.012706 | 49.3% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| medium_384 | torch_cuda | ok | 0.025 |  |  | 4.462729 | 22.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| medium_768 | torch_cuda | ok | 0.096 |  |  | 9.427756 | 46.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_8x256 | torch_cuda | ok | 0.047 |  |  | 5.694914 | 28.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_16x128 | torch_cuda | ok | 0.028 |  |  | 2.366989 | 11.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_32x64 | torch_cuda | ok | 0.020 |  |  | 0.819200 | 4.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_64x128 | torch_cuda | ok | 0.050 |  |  | 5.349878 | 26.3% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_128x64 | torch_cuda | ok | 0.041 |  |  | 1.629489 | 8.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_proj_2048x512x512 | torch_cuda | ok | 0.115 |  |  | 9.336236 | 45.9% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_proj_512x2048x512 | torch_cuda | ok | 0.110 |  |  | 9.754196 | 48.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_qkv_1024x3072x512 | torch_cuda | ok | 0.288 |  |  | 11.179842 | 55.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b32_128 | torch_cuda | ok | 0.033 |  |  | 4.040755 | 19.9% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b16_192 | torch_cuda | ok | 0.060 |  |  | 3.750868 | 18.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b8_192 | torch_cuda | ok | 0.036 |  |  | 3.156953 | 15.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_128 | cupy_cuda | ok | 0.051 |  |  | 0.082177 | 0.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_256 | cupy_cuda | ok | 0.053 |  |  | 0.638597 | 3.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_512 | cupy_cuda | ok | 0.080 |  |  | 3.372983 | 16.6% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_4x256 | cupy_cuda | ok | 0.084 |  |  | 1.607013 | 7.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tall_512x256x256 | cupy_cuda | ok | 0.056 |  |  | 1.200430 | 5.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| odd_255x257x263 | cupy_cuda | ok | 0.053 |  |  | 0.648154 | 3.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_1024 | cupy_cuda | ok | 0.240 |  |  | 8.958599 | 44.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_2x512 | cupy_cuda | ok | 0.117 |  |  | 4.607859 | 22.7% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| skinny_1024x128x512 | cupy_cuda | ok | 0.066 |  |  | 2.046002 | 10.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| wide_128x1024x512 | cupy_cuda | ok | 0.066 |  |  | 2.032124 | 10.0% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| small_k_1024x1024x64 | cupy_cuda | ok | 0.064 |  |  | 2.113000 | 10.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| non_pow2_513x515x517 | cupy_cuda | ok | 0.093 |  |  | 2.934617 | 14.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| non_pow2_1023x1025x1027 | cupy_cuda | ok | 0.255 |  |  | 8.461832 | 41.6% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| medium_384 | cupy_cuda | ok | 0.063 |  |  | 1.805584 | 8.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| medium_768 | cupy_cuda | ok | 0.132 |  |  | 6.861743 | 33.8% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_8x256 | cupy_cuda | ok | 0.086 |  |  | 3.138274 | 15.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_16x128 | cupy_cuda | ok | 0.066 |  |  | 1.012628 | 5.0% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_32x64 | cupy_cuda | ok | 0.058 |  |  | 0.289183 | 1.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_64x128 | cupy_cuda | ok | 0.087 |  |  | 3.076130 | 15.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_128x64 | cupy_cuda | ok | 0.079 |  |  | 0.848362 | 4.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_proj_2048x512x512 | cupy_cuda | ok | 0.151 |  |  | 7.125596 | 35.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_proj_512x2048x512 | cupy_cuda | ok | 0.145 |  |  | 7.392473 | 36.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_qkv_1024x3072x512 | cupy_cuda | ok | 0.323 |  |  | 9.985448 | 49.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b32_128 | cupy_cuda | ok | 0.068 |  |  | 1.968233 | 9.7% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b16_192 | cupy_cuda | ok | 0.096 |  |  | 2.353021 | 11.6% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b8_192 | cupy_cuda | ok | 0.072 |  |  | 1.569377 | 7.7% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
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
| ok | 67108864 | 30 | 9.691 | 10.315 | NVIDIA GeForce RTX 3070 (discrete) |

## Analysis

- `tensor-ash` used `NVIDIA GeForce RTX 3070 (discrete)`, so the Vulkan measurements reflect real GPU kernel timings on this host.
- Actual GPU framework comparisons succeeded for: `cublas_pure`, `cupy_cuda`, `torch_cuda`.
- Largest gap: `medium_768` is 1.2x faster in `cublas_pure` than `tensor-ash` in this environment.
- `tensor-ash` is the fastest measured backend on 16/26 benchmark cases.
- Throughput ratio versus `cublas_pure` across 26 shared cases: 0.83x to 2.55x, geometric mean 1.18x.
- Throughput ratio versus `torch_cuda` across 26 shared cases: 0.87x to 3.64x, geometric mean 1.40x.
- Throughput ratio versus `cupy_cuda` across 26 shared cases: 1.02x to 10.30x, geometric mean 2.74x.
- Throughput ratio versus `numpy` across 26 shared cases: 6.07x to 85.02x, geometric mean 48.47x.
- Throughput ratio versus `torch_cpu` across 26 shared cases: 5.98x to 87.43x, geometric mean 46.86x.
- Median `tensor-ash` host/submission overhead was 0.020 ms per synchronous call; GPU timestamp TFLOPS excludes that overhead.
- Some libraries were skipped because their Python modules or device backends were unavailable: `jax`, `tensorflow`.
- JAX and TensorFlow rows are included when those modules are installed; skipped rows mean the local Python environment does not provide them.
- Transfer staging bandwidth measured 9.69 GiB/s upload and 10.31 GiB/s download for 67108864 bytes.

## Optimization Gameplan

1. Keep benchmarking on this discrete-GPU baseline and tune the shape-based shader selector with larger production-like matrix sizes.
2. Use `scripts/tune_kernels.py` before accepting changes to the automatic selector.
3. Focus the next shader pass on large square GEMMs, where PyTorch CUDA still has the largest lead.
4. Keep PyTorch CUDA and CuPy CUDA rows in regular benchmark runs so Vulkan changes are compared against cuBLAS-backed GPU compute.
5. Add CI checks for `cargo fmt`, `cargo clippy`, CPU tests, and an optional GPU correctness job.
