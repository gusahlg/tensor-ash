# Benchmark Report

This report compares FP32 GEMM throughput for `tensor-ash` against the local framework backends available on this machine.

## Environment

```text
[2026-06-11T07:06:35Z INFO  tensor_ash::context] tensor-ash: using device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329)
[2026-06-11T07:06:35Z INFO  ml_bench::bench] device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496774464, compute_family=2, timestamps=true) slots=2
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
gpu0: name=NVIDIA GeForce RTX 3070, driver=595.71.05, memory_total_mib=8192, temperature_c=35, utilization_pct=3, power_draw_w=21.46, power_limit_w=220.00
```

- Iterations: 25
- Warmup iterations: 8
- Case set: showcase
- CPU library threads: 1
- CPU framework rows: enabled
- Python GPU framework rows: enabled

- Peak FP32 throughput used for `% peak`: 20.32 TFLOPS

## Results

| case | library | status | gpu ms | wall ms | host overhead ms | TFLOPS | % peak | details |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| square_128 | tensor-ash | ok | 0.007 | 0.025 | 0.018 | 0.587767 | 2.9% | NVIDIA GeForce RTX 3070 (discrete) |
| square_256 | tensor-ash | ok | 0.012 | 0.030 | 0.018 | 2.723574 | 13.4% | NVIDIA GeForce RTX 3070 (discrete) |
| square_512 | tensor-ash | ok | 0.046 | 0.064 | 0.018 | 5.886742 | 29.0% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_4x256 | tensor-ash | ok | 0.026 | 0.044 | 0.018 | 5.178153 | 25.5% | NVIDIA GeForce RTX 3070 (discrete) |
| tall_512x256x256 | tensor-ash | ok | 0.017 | 0.034 | 0.018 | 4.064248 | 20.0% | NVIDIA GeForce RTX 3070 (discrete) |
| odd_255x257x263 | tensor-ash | ok | 0.015 | 0.033 | 0.018 | 2.326634 | 11.4% | NVIDIA GeForce RTX 3070 (discrete) |
| square_1024 | tensor-ash | ok | 0.238 | 0.264 | 0.026 | 9.017585 | 44.4% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_2x512 | tensor-ash | ok | 0.073 | 0.091 | 0.018 | 7.358428 | 36.2% | NVIDIA GeForce RTX 3070 (discrete) |
| skinny_1024x128x512 | tensor-ash | ok | 0.029 | 0.047 | 0.018 | 4.644855 | 22.9% | NVIDIA GeForce RTX 3070 (discrete) |
| wide_128x1024x512 | tensor-ash | ok | 0.029 | 0.054 | 0.026 | 4.634590 | 22.8% | NVIDIA GeForce RTX 3070 (discrete) |
| small_k_1024x1024x64 | tensor-ash | ok | 0.026 | 0.044 | 0.018 | 5.140078 | 25.3% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_513x515x517 | tensor-ash | ok | 0.055 | 0.080 | 0.025 | 4.966144 | 24.4% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_1023x1025x1027 | tensor-ash | ok | 0.265 | 0.285 | 0.020 | 8.128672 | 40.0% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_384 | tensor-ash | ok | 0.023 | 0.041 | 0.018 | 4.894805 | 24.1% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_768 | tensor-ash | ok | 0.122 | 0.140 | 0.018 | 7.413342 | 36.5% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_8x256 | tensor-ash | ok | 0.039 | 0.058 | 0.019 | 6.831114 | 33.6% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_16x128 | tensor-ash | ok | 0.016 | 0.034 | 0.018 | 4.245247 | 20.9% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_32x64 | tensor-ash | ok | 0.007 | 0.025 | 0.018 | 2.259862 | 11.1% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_64x128 | tensor-ash | ok | 0.044 | 0.070 | 0.026 | 6.123072 | 30.1% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_128x64 | tensor-ash | ok | 0.018 | 0.036 | 0.018 | 3.628291 | 17.9% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_2048x512x512 | tensor-ash | ok | 0.124 | 0.143 | 0.018 | 8.648049 | 42.6% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_512x2048x512 | tensor-ash | ok | 0.126 | 0.144 | 0.018 | 8.529342 | 42.0% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_qkv_1024x3072x512 | tensor-ash | ok | 0.360 | 0.396 | 0.036 | 8.950235 | 44.0% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b32_128 | tensor-ash | ok | 0.025 | 0.050 | 0.026 | 5.418997 | 26.7% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b16_192 | tensor-ash | ok | 0.042 | 0.060 | 0.018 | 5.423669 | 26.7% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b8_192 | tensor-ash | ok | 0.022 | 0.040 | 0.018 | 5.041231 | 24.8% | NVIDIA GeForce RTX 3070 (discrete) |
| square_128 | numpy | ok | 0.039 |  |  | 0.107041 | 0.5% | numpy 2.4.4, threads=1 |
| square_256 | numpy | ok | 0.288 |  |  | 0.116387 | 0.6% | numpy 2.4.4, threads=1 |
| square_512 | numpy | ok | 2.068 |  |  | 0.129789 | 0.6% | numpy 2.4.4, threads=1 |
| batched_4x256 | numpy | ok | 1.050 |  |  | 0.127845 | 0.6% | numpy 2.4.4, threads=1 |
| tall_512x256x256 | numpy | ok | 0.521 |  |  | 0.128755 | 0.6% | numpy 2.4.4, threads=1 |
| odd_255x257x263 | numpy | ok | 0.309 |  |  | 0.111384 | 0.5% | numpy 2.4.4, threads=1 |
| square_1024 | numpy | ok | 16.183 |  |  | 0.132699 | 0.7% | numpy 2.4.4, threads=1 |
| batched_2x512 | numpy | ok | 4.092 |  |  | 0.131197 | 0.6% | numpy 2.4.4, threads=1 |
| skinny_1024x128x512 | numpy | ok | 1.055 |  |  | 0.127195 | 0.6% | numpy 2.4.4, threads=1 |
| wide_128x1024x512 | numpy | ok | 1.105 |  |  | 0.121481 | 0.6% | numpy 2.4.4, threads=1 |
| small_k_1024x1024x64 | numpy | ok | 1.128 |  |  | 0.118993 | 0.6% | numpy 2.4.4, threads=1 |
| non_pow2_513x515x517 | numpy | ok | 2.186 |  |  | 0.124983 | 0.6% | numpy 2.4.4, threads=1 |
| non_pow2_1023x1025x1027 | numpy | ok | 16.600 |  |  | 0.129744 | 0.6% | numpy 2.4.4, threads=1 |
| medium_384 | numpy | ok | 0.963 |  |  | 0.117637 | 0.6% | numpy 2.4.4, threads=1 |
| medium_768 | numpy | ok | 6.784 |  |  | 0.133537 | 0.7% | numpy 2.4.4, threads=1 |
| batched_8x256 | numpy | ok | 2.098 |  |  | 0.127918 | 0.6% | numpy 2.4.4, threads=1 |
| batched_16x128 | numpy | ok | 0.575 |  |  | 0.116669 | 0.6% | numpy 2.4.4, threads=1 |
| batched_32x64 | numpy | ok | 0.165 |  |  | 0.101657 | 0.5% | numpy 2.4.4, threads=1 |
| batched_64x128 | numpy | ok | 2.333 |  |  | 0.115040 | 0.6% | numpy 2.4.4, threads=1 |
| batched_128x64 | numpy | ok | 0.669 |  |  | 0.100372 | 0.5% | numpy 2.4.4, threads=1 |
| attn_proj_2048x512x512 | numpy | ok | 8.242 |  |  | 0.130280 | 0.6% | numpy 2.4.4, threads=1 |
| attn_proj_512x2048x512 | numpy | ok | 8.224 |  |  | 0.130561 | 0.6% | numpy 2.4.4, threads=1 |
| attn_qkv_1024x3072x512 | numpy | ok | 24.080 |  |  | 0.133771 | 0.7% | numpy 2.4.4, threads=1 |
| tiny_b32_128 | numpy | ok | 1.165 |  |  | 0.115169 | 0.6% | numpy 2.4.4, threads=1 |
| tiny_b16_192 | numpy | ok | 1.812 |  |  | 0.125024 | 0.6% | numpy 2.4.4, threads=1 |
| tiny_b8_192 | numpy | ok | 0.906 |  |  | 0.125059 | 0.6% | numpy 2.4.4, threads=1 |
| square_128 | torch_cpu | ok | 0.038 |  |  | 0.111113 | 0.5% | torch 2.11.0+cu128, threads=1 |
| square_256 | torch_cpu | ok | 0.261 |  |  | 0.128675 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_512 | torch_cpu | ok | 2.072 |  |  | 0.129569 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_4x256 | torch_cpu | ok | 1.044 |  |  | 0.128575 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tall_512x256x256 | torch_cpu | ok | 0.519 |  |  | 0.129225 | 0.6% | torch 2.11.0+cu128, threads=1 |
| odd_255x257x263 | torch_cpu | ok | 0.289 |  |  | 0.119154 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_1024 | torch_cpu | ok | 16.409 |  |  | 0.130872 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_2x512 | torch_cpu | ok | 4.160 |  |  | 0.129044 | 0.6% | torch 2.11.0+cu128, threads=1 |
| skinny_1024x128x512 | torch_cpu | ok | 1.090 |  |  | 0.123158 | 0.6% | torch 2.11.0+cu128, threads=1 |
| wide_128x1024x512 | torch_cpu | ok | 1.053 |  |  | 0.127436 | 0.6% | torch 2.11.0+cu128, threads=1 |
| small_k_1024x1024x64 | torch_cpu | ok | 1.015 |  |  | 0.132239 | 0.7% | torch 2.11.0+cu128, threads=1 |
| non_pow2_513x515x517 | torch_cpu | ok | 2.204 |  |  | 0.123936 | 0.6% | torch 2.11.0+cu128, threads=1 |
| non_pow2_1023x1025x1027 | torch_cpu | ok | 16.770 |  |  | 0.128432 | 0.6% | torch 2.11.0+cu128, threads=1 |
| medium_384 | torch_cpu | ok | 0.853 |  |  | 0.132785 | 0.7% | torch 2.11.0+cu128, threads=1 |
| medium_768 | torch_cpu | ok | 6.805 |  |  | 0.133139 | 0.7% | torch 2.11.0+cu128, threads=1 |
| batched_8x256 | torch_cpu | ok | 2.173 |  |  | 0.123514 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_16x128 | torch_cpu | ok | 0.574 |  |  | 0.116943 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_32x64 | torch_cpu | ok | 0.140 |  |  | 0.119531 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_64x128 | torch_cpu | ok | 2.233 |  |  | 0.120229 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_128x64 | torch_cpu | ok | 0.531 |  |  | 0.126298 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_proj_2048x512x512 | torch_cpu | ok | 8.192 |  |  | 0.131071 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_proj_512x2048x512 | torch_cpu | ok | 8.300 |  |  | 0.129361 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_qkv_1024x3072x512 | torch_cpu | ok | 25.217 |  |  | 0.127741 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b32_128 | torch_cpu | ok | 1.120 |  |  | 0.119845 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b16_192 | torch_cpu | ok | 1.762 |  |  | 0.128532 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b8_192 | torch_cpu | ok | 0.884 |  |  | 0.128162 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_128 | cublas_pure | ok | 0.008 |  |  | 0.526394 | 2.6% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_256 | cublas_pure | ok | 0.011 |  |  | 2.978909 | 14.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_512 | cublas_pure | ok | 0.036 |  |  | 7.489828 | 36.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_4x256 | cublas_pure | ok | 0.037 |  |  | 3.640889 | 17.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tall_512x256x256 | cublas_pure | ok | 0.015 |  |  | 4.369067 | 21.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| odd_255x257x263 | cublas_pure | ok | 0.011 |  |  | 3.060317 | 15.1% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_1024 | cublas_pure | ok | 0.196 |  |  | 10.979853 | 54.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_2x512 | cublas_pure | ok | 0.069 |  |  | 7.825194 | 38.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| skinny_1024x128x512 | cublas_pure | ok | 0.025 |  |  | 5.461333 | 26.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| wide_128x1024x512 | cublas_pure | ok | 0.025 |  |  | 5.468454 | 26.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| small_k_1024x1024x64 | cublas_pure | ok | 0.022 |  |  | 6.009031 | 29.6% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_513x515x517 | cublas_pure | ok | 0.049 |  |  | 5.557813 | 27.4% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_1023x1025x1027 | cublas_pure | ok | 0.209 |  |  | 10.310265 | 50.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_384 | cublas_pure | ok | 0.021 |  |  | 5.337774 | 26.3% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_768 | cublas_pure | ok | 0.090 |  |  | 10.053818 | 49.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_8x256 | cublas_pure | ok | 0.040 |  |  | 6.668210 | 32.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_16x128 | cublas_pure | ok | 0.022 |  |  | 3.120762 | 15.4% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_32x64 | cublas_pure | ok | 0.014 |  |  | 1.170286 | 5.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_64x128 | cublas_pure | ok | 0.043 |  |  | 6.241524 | 30.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_128x64 | cublas_pure | ok | 0.035 |  |  | 1.927529 | 9.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_2048x512x512 | cublas_pure | ok | 0.110 |  |  | 9.799776 | 48.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_512x2048x512 | cublas_pure | ok | 0.105 |  |  | 10.211330 | 50.3% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_qkv_1024x3072x512 | cublas_pure | ok | 0.282 |  |  | 11.439011 | 56.3% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b32_128 | cublas_pure | ok | 0.027 |  |  | 5.041231 | 24.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b16_192 | cublas_pure | ok | 0.053 |  |  | 4.253539 | 20.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b8_192 | cublas_pure | ok | 0.030 |  |  | 3.813517 | 18.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_128 | torch_cuda | ok | 0.013 |  |  | 0.332670 | 1.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_256 | torch_cuda | ok | 0.014 |  |  | 2.461446 | 12.1% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_512 | torch_cuda | ok | 0.037 |  |  | 7.200522 | 35.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_4x256 | torch_cuda | ok | 0.040 |  |  | 3.352761 | 16.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tall_512x256x256 | torch_cuda | ok | 0.018 |  |  | 3.813004 | 18.8% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| odd_255x257x263 | torch_cuda | ok | 0.015 |  |  | 2.367542 | 11.7% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_1024 | torch_cuda | ok | 0.187 |  |  | 11.512929 | 56.7% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_2x512 | torch_cuda | ok | 0.070 |  |  | 7.671338 | 37.8% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| skinny_1024x128x512 | torch_cuda | ok | 0.026 |  |  | 5.127511 | 25.2% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| wide_128x1024x512 | torch_cuda | ok | 0.026 |  |  | 5.178153 | 25.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| small_k_1024x1024x64 | torch_cuda | ok | 0.024 |  |  | 5.489927 | 27.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| non_pow2_513x515x517 | torch_cuda | ok | 0.050 |  |  | 5.454825 | 26.8% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| non_pow2_1023x1025x1027 | torch_cuda | ok | 0.199 |  |  | 10.815588 | 53.2% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| medium_384 | torch_cuda | ok | 0.023 |  |  | 4.956504 | 24.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| medium_768 | torch_cuda | ok | 0.089 |  |  | 10.184012 | 50.1% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_8x256 | torch_cuda | ok | 0.043 |  |  | 6.195427 | 30.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_16x128 | torch_cuda | ok | 0.026 |  |  | 2.589077 | 12.7% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_32x64 | torch_cuda | ok | 0.019 |  |  | 0.903945 | 4.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_64x128 | torch_cuda | ok | 0.047 |  |  | 5.769331 | 28.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_128x64 | torch_cuda | ok | 0.038 |  |  | 1.752007 | 8.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_proj_2048x512x512 | torch_cuda | ok | 0.106 |  |  | 10.122000 | 49.8% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_proj_512x2048x512 | torch_cuda | ok | 0.103 |  |  | 10.469402 | 51.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_qkv_1024x3072x512 | torch_cuda | ok | 0.266 |  |  | 12.096047 | 59.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b32_128 | torch_cuda | ok | 0.031 |  |  | 4.359983 | 21.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b16_192 | torch_cuda | ok | 0.056 |  |  | 4.042198 | 19.9% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b8_192 | torch_cuda | ok | 0.033 |  |  | 3.415969 | 16.8% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_128 | cupy_cuda | ok | 0.045 |  |  | 0.092630 | 0.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_256 | cupy_cuda | ok | 0.047 |  |  | 0.718695 | 3.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_512 | cupy_cuda | ok | 0.071 |  |  | 3.797468 | 18.7% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_4x256 | cupy_cuda | ok | 0.076 |  |  | 1.777247 | 8.7% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tall_512x256x256 | cupy_cuda | ok | 0.050 |  |  | 1.328994 | 6.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| odd_255x257x263 | cupy_cuda | ok | 0.048 |  |  | 0.723460 | 3.6% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_1024 | cupy_cuda | ok | 0.222 |  |  | 9.686614 | 47.7% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_2x512 | cupy_cuda | ok | 0.107 |  |  | 5.017110 | 24.7% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| skinny_1024x128x512 | cupy_cuda | ok | 0.059 |  |  | 2.268418 | 11.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| wide_128x1024x512 | cupy_cuda | ok | 0.059 |  |  | 2.258645 | 11.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| small_k_1024x1024x64 | cupy_cuda | ok | 0.057 |  |  | 2.340571 | 11.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| non_pow2_513x515x517 | cupy_cuda | ok | 0.084 |  |  | 3.259565 | 16.0% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| non_pow2_1023x1025x1027 | cupy_cuda | ok | 0.233 |  |  | 9.240171 | 45.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| medium_384 | cupy_cuda | ok | 0.056 |  |  | 2.015344 | 9.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| medium_768 | cupy_cuda | ok | 0.123 |  |  | 7.372800 | 36.3% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_8x256 | cupy_cuda | ok | 0.080 |  |  | 3.359475 | 16.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_16x128 | cupy_cuda | ok | 0.062 |  |  | 1.079337 | 5.3% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_32x64 | cupy_cuda | ok | 0.054 |  |  | 0.310597 | 1.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_64x128 | cupy_cuda | ok | 0.082 |  |  | 3.265320 | 16.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_128x64 | cupy_cuda | ok | 0.074 |  |  | 0.904335 | 4.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_proj_2048x512x512 | cupy_cuda | ok | 0.143 |  |  | 7.521728 | 37.0% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_proj_512x2048x512 | cupy_cuda | ok | 0.138 |  |  | 7.788865 | 38.3% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_qkv_1024x3072x512 | cupy_cuda | ok | 0.301 |  |  | 10.700893 | 52.7% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b32_128 | cupy_cuda | ok | 0.069 |  |  | 1.934642 | 9.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b16_192 | cupy_cuda | ok | 0.094 |  |  | 2.409084 | 11.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b8_192 | cupy_cuda | ok | 0.071 |  |  | 1.594119 | 7.8% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
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
| ok | 67108864 | 25 | 9.791 | 10.315 | NVIDIA GeForce RTX 3070 (discrete) |

## Analysis

- `tensor-ash` used `NVIDIA GeForce RTX 3070 (discrete)`, so the Vulkan measurements reflect real GPU kernel timings on this host.
- Actual GPU framework comparisons succeeded for: `cublas_pure`, `cupy_cuda`, `torch_cuda`.
- Largest gap: `medium_768` is 1.4x faster in `torch_cuda` than `tensor-ash` in this environment.
- `tensor-ash` is the fastest measured backend on 9/26 benchmark cases.
- Throughput ratio versus `cublas_pure` across 26 shared cases: 0.74x to 1.93x, geometric mean 1.00x.
- Throughput ratio versus `torch_cuda` across 26 shared cases: 0.73x to 2.50x, geometric mean 1.09x.
- Throughput ratio versus `cupy_cuda` across 26 shared cases: 0.84x to 7.28x, geometric mean 2.14x.
- Throughput ratio versus `numpy` across 26 shared cases: 5.49x to 67.96x, geometric mean 39.85x.
- Throughput ratio versus `torch_cpu` across 26 shared cases: 5.29x to 70.07x, geometric mean 38.57x.
- Median `tensor-ash` host/submission overhead was 0.018 ms per synchronous call; GPU timestamp TFLOPS excludes that overhead.
- Some libraries were skipped because their Python modules or device backends were unavailable: `jax`, `tensorflow`.
- JAX and TensorFlow rows are included when those modules are installed; skipped rows mean the local Python environment does not provide them.
- Transfer staging bandwidth measured 9.79 GiB/s upload and 10.31 GiB/s download for 67108864 bytes.

## Optimization Gameplan

1. Keep benchmarking on this discrete-GPU baseline and tune the shape-based shader selector with larger production-like matrix sizes.
2. Use `scripts/tune_kernels.py` before accepting changes to the automatic selector.
3. Focus the next shader pass on large square GEMMs, where PyTorch CUDA still has the largest lead.
4. Keep PyTorch CUDA and CuPy CUDA rows in regular benchmark runs so Vulkan changes are compared against cuBLAS-backed GPU compute.
5. Add CI checks for `cargo fmt`, `cargo clippy`, CPU tests, and an optional GPU correctness job.
