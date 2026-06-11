# Benchmark Report

This report compares FP32 GEMM throughput for `tensor-ash` against the local framework backends available on this machine.

## Environment

```text
[2026-06-11T11:33:44Z INFO  tensor_ash::context] tensor-ash: using device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329)
[2026-06-11T11:33:45Z INFO  ml_bench::bench] device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496774464, compute_family=2, timestamps=true) slots=2
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
gpu0: name=NVIDIA GeForce RTX 3070, driver=595.71.05, memory_total_mib=8192, temperature_c=38, utilization_pct=1, power_draw_w=42.61, power_limit_w=220.00
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
| square_128 | tensor-ash | ok | 0.006 | 0.025 | 0.019 | 0.668735 | 3.3% | NVIDIA GeForce RTX 3070 (discrete) |
| square_256 | tensor-ash | ok | 0.011 | 0.030 | 0.019 | 2.987396 | 14.7% | NVIDIA GeForce RTX 3070 (discrete) |
| square_512 | tensor-ash | ok | 0.042 | 0.060 | 0.019 | 6.467701 | 31.8% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_4x256 | tensor-ash | ok | 0.024 | 0.043 | 0.019 | 5.637505 | 27.7% | NVIDIA GeForce RTX 3070 (discrete) |
| tall_512x256x256 | tensor-ash | ok | 0.015 | 0.036 | 0.020 | 4.415057 | 21.7% | NVIDIA GeForce RTX 3070 (discrete) |
| odd_255x257x263 | tensor-ash | ok | 0.014 | 0.033 | 0.019 | 2.487833 | 12.2% | NVIDIA GeForce RTX 3070 (discrete) |
| square_1024 | tensor-ash | ok | 0.210 | 0.230 | 0.020 | 10.209777 | 50.2% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_2x512 | tensor-ash | ok | 0.070 | 0.100 | 0.030 | 7.703038 | 37.9% | NVIDIA GeForce RTX 3070 (discrete) |
| skinny_1024x128x512 | tensor-ash | ok | 0.027 | 0.057 | 0.030 | 4.922892 | 24.2% | NVIDIA GeForce RTX 3070 (discrete) |
| wide_128x1024x512 | tensor-ash | ok | 0.028 | 0.047 | 0.019 | 4.871433 | 24.0% | NVIDIA GeForce RTX 3070 (discrete) |
| small_k_1024x1024x64 | tensor-ash | ok | 0.021 | 0.040 | 0.019 | 6.502797 | 32.0% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_513x515x517 | tensor-ash | ok | 0.053 | 0.073 | 0.020 | 5.158188 | 25.4% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_1023x1025x1027 | tensor-ash | ok | 0.248 | 0.269 | 0.021 | 8.677850 | 42.7% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_384 | tensor-ash | ok | 0.022 | 0.053 | 0.030 | 5.099343 | 25.1% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_768 | tensor-ash | ok | 0.108 | 0.128 | 0.019 | 8.363826 | 41.2% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_8x256 | tensor-ash | ok | 0.037 | 0.068 | 0.030 | 7.194346 | 35.4% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_16x128 | tensor-ash | ok | 0.014 | 0.033 | 0.019 | 4.712701 | 23.2% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_32x64 | tensor-ash | ok | 0.006 | 0.025 | 0.019 | 2.945438 | 14.5% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_64x128 | tensor-ash | ok | 0.039 | 0.059 | 0.020 | 6.898526 | 33.9% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_128x64 | tensor-ash | ok | 0.016 | 0.036 | 0.019 | 4.120141 | 20.3% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_2048x512x512 | tensor-ash | ok | 0.109 | 0.129 | 0.020 | 9.811237 | 48.3% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_512x2048x512 | tensor-ash | ok | 0.109 | 0.129 | 0.020 | 9.822726 | 48.3% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_qkv_1024x3072x512 | tensor-ash | ok | 0.310 | 0.338 | 0.029 | 10.405551 | 51.2% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b32_128 | tensor-ash | ok | 0.022 | 0.041 | 0.019 | 6.213784 | 30.6% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b16_192 | tensor-ash | ok | 0.037 | 0.057 | 0.020 | 6.149338 | 30.3% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b8_192 | tensor-ash | ok | 0.020 | 0.040 | 0.020 | 5.590749 | 27.5% | NVIDIA GeForce RTX 3070 (discrete) |
| square_128 | numpy | ok | 0.039 |  |  | 0.106552 | 0.5% | numpy 2.4.4, threads=1 |
| square_256 | numpy | ok | 0.289 |  |  | 0.115965 | 0.6% | numpy 2.4.4, threads=1 |
| square_512 | numpy | ok | 2.249 |  |  | 0.119351 | 0.6% | numpy 2.4.4, threads=1 |
| batched_4x256 | numpy | ok | 1.166 |  |  | 0.115151 | 0.6% | numpy 2.4.4, threads=1 |
| tall_512x256x256 | numpy | ok | 0.544 |  |  | 0.123259 | 0.6% | numpy 2.4.4, threads=1 |
| odd_255x257x263 | numpy | ok | 0.297 |  |  | 0.115915 | 0.6% | numpy 2.4.4, threads=1 |
| square_1024 | numpy | ok | 17.175 |  |  | 0.125035 | 0.6% | numpy 2.4.4, threads=1 |
| batched_2x512 | numpy | ok | 4.373 |  |  | 0.122767 | 0.6% | numpy 2.4.4, threads=1 |
| skinny_1024x128x512 | numpy | ok | 1.106 |  |  | 0.121374 | 0.6% | numpy 2.4.4, threads=1 |
| wide_128x1024x512 | numpy | ok | 1.135 |  |  | 0.118279 | 0.6% | numpy 2.4.4, threads=1 |
| small_k_1024x1024x64 | numpy | ok | 1.157 |  |  | 0.116025 | 0.6% | numpy 2.4.4, threads=1 |
| non_pow2_513x515x517 | numpy | ok | 2.345 |  |  | 0.116484 | 0.6% | numpy 2.4.4, threads=1 |
| non_pow2_1023x1025x1027 | numpy | ok | 17.545 |  |  | 0.122757 | 0.6% | numpy 2.4.4, threads=1 |
| medium_384 | numpy | ok | 0.922 |  |  | 0.122825 | 0.6% | numpy 2.4.4, threads=1 |
| medium_768 | numpy | ok | 7.327 |  |  | 0.123648 | 0.6% | numpy 2.4.4, threads=1 |
| batched_8x256 | numpy | ok | 2.365 |  |  | 0.113519 | 0.6% | numpy 2.4.4, threads=1 |
| batched_16x128 | numpy | ok | 0.633 |  |  | 0.106085 | 0.5% | numpy 2.4.4, threads=1 |
| batched_32x64 | numpy | ok | 0.176 |  |  | 0.095172 | 0.5% | numpy 2.4.4, threads=1 |
| batched_64x128 | numpy | ok | 2.554 |  |  | 0.105111 | 0.5% | numpy 2.4.4, threads=1 |
| batched_128x64 | numpy | ok | 0.681 |  |  | 0.098576 | 0.5% | numpy 2.4.4, threads=1 |
| attn_proj_2048x512x512 | numpy | ok | 8.665 |  |  | 0.123918 | 0.6% | numpy 2.4.4, threads=1 |
| attn_proj_512x2048x512 | numpy | ok | 8.723 |  |  | 0.123086 | 0.6% | numpy 2.4.4, threads=1 |
| attn_qkv_1024x3072x512 | numpy | ok | 25.671 |  |  | 0.125480 | 0.6% | numpy 2.4.4, threads=1 |
| tiny_b32_128 | numpy | ok | 1.305 |  |  | 0.102884 | 0.5% | numpy 2.4.4, threads=1 |
| tiny_b16_192 | numpy | ok | 1.914 |  |  | 0.118343 | 0.6% | numpy 2.4.4, threads=1 |
| tiny_b8_192 | numpy | ok | 0.962 |  |  | 0.117761 | 0.6% | numpy 2.4.4, threads=1 |
| square_128 | torch_cpu | ok | 0.041 |  |  | 0.103461 | 0.5% | torch 2.11.0+cu128, threads=1 |
| square_256 | torch_cpu | ok | 0.287 |  |  | 0.117080 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_512 | torch_cpu | ok | 2.124 |  |  | 0.126356 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_4x256 | torch_cpu | ok | 1.091 |  |  | 0.122979 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tall_512x256x256 | torch_cpu | ok | 0.571 |  |  | 0.117524 | 0.6% | torch 2.11.0+cu128, threads=1 |
| odd_255x257x263 | torch_cpu | ok | 0.296 |  |  | 0.116460 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_1024 | torch_cpu | ok | 17.099 |  |  | 0.125588 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_2x512 | torch_cpu | ok | 4.303 |  |  | 0.124766 | 0.6% | torch 2.11.0+cu128, threads=1 |
| skinny_1024x128x512 | torch_cpu | ok | 1.120 |  |  | 0.119855 | 0.6% | torch 2.11.0+cu128, threads=1 |
| wide_128x1024x512 | torch_cpu | ok | 1.148 |  |  | 0.116928 | 0.6% | torch 2.11.0+cu128, threads=1 |
| small_k_1024x1024x64 | torch_cpu | ok | 1.041 |  |  | 0.128973 | 0.6% | torch 2.11.0+cu128, threads=1 |
| non_pow2_513x515x517 | torch_cpu | ok | 2.256 |  |  | 0.121074 | 0.6% | torch 2.11.0+cu128, threads=1 |
| non_pow2_1023x1025x1027 | torch_cpu | ok | 17.508 |  |  | 0.123016 | 0.6% | torch 2.11.0+cu128, threads=1 |
| medium_384 | torch_cpu | ok | 0.938 |  |  | 0.120718 | 0.6% | torch 2.11.0+cu128, threads=1 |
| medium_768 | torch_cpu | ok | 7.167 |  |  | 0.126404 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_8x256 | torch_cpu | ok | 2.182 |  |  | 0.123031 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_16x128 | torch_cpu | ok | 0.574 |  |  | 0.116888 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_32x64 | torch_cpu | ok | 0.140 |  |  | 0.120049 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_64x128 | torch_cpu | ok | 2.430 |  |  | 0.110461 | 0.5% | torch 2.11.0+cu128, threads=1 |
| batched_128x64 | torch_cpu | ok | 0.575 |  |  | 0.116648 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_proj_2048x512x512 | torch_cpu | ok | 8.697 |  |  | 0.123459 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_proj_512x2048x512 | torch_cpu | ok | 8.601 |  |  | 0.124841 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_qkv_1024x3072x512 | torch_cpu | ok | 26.332 |  |  | 0.122331 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b32_128 | torch_cpu | ok | 1.147 |  |  | 0.117052 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b16_192 | torch_cpu | ok | 1.856 |  |  | 0.122065 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b8_192 | torch_cpu | ok | 0.929 |  |  | 0.121942 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_128 | cublas_pure | ok | 0.008 |  |  | 0.524288 | 2.6% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_256 | cublas_pure | ok | 0.011 |  |  | 2.978909 | 14.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_512 | cublas_pure | ok | 0.036 |  |  | 7.489828 | 36.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_4x256 | cublas_pure | ok | 0.037 |  |  | 3.640889 | 17.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tall_512x256x256 | cublas_pure | ok | 0.014 |  |  | 4.681143 | 23.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| odd_255x257x263 | cublas_pure | ok | 0.011 |  |  | 3.060317 | 15.1% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_1024 | cublas_pure | ok | 0.193 |  |  | 11.099713 | 54.6% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_2x512 | cublas_pure | ok | 0.069 |  |  | 7.825194 | 38.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| skinny_1024x128x512 | cublas_pure | ok | 0.025 |  |  | 5.461333 | 26.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| wide_128x1024x512 | cublas_pure | ok | 0.024 |  |  | 5.698782 | 28.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| small_k_1024x1024x64 | cublas_pure | ok | 0.022 |  |  | 6.043666 | 29.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_513x515x517 | cublas_pure | ok | 0.049 |  |  | 5.557813 | 27.4% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_1023x1025x1027 | cublas_pure | ok | 0.209 |  |  | 10.310265 | 50.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_384 | cublas_pure | ok | 0.020 |  |  | 5.529600 | 27.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_768 | cublas_pure | ok | 0.090 |  |  | 10.053818 | 49.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_8x256 | cublas_pure | ok | 0.040 |  |  | 6.721641 | 33.1% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_16x128 | cublas_pure | ok | 0.022 |  |  | 3.120762 | 15.4% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_32x64 | cublas_pure | ok | 0.014 |  |  | 1.170286 | 5.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_64x128 | cublas_pure | ok | 0.043 |  |  | 6.250826 | 30.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_128x64 | cublas_pure | ok | 0.035 |  |  | 1.929303 | 9.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_2048x512x512 | cublas_pure | ok | 0.110 |  |  | 9.799776 | 48.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_512x2048x512 | cublas_pure | ok | 0.104 |  |  | 10.280157 | 50.6% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_qkv_1024x3072x512 | cublas_pure | ok | 0.282 |  |  | 11.439011 | 56.3% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b32_128 | cublas_pure | ok | 0.027 |  |  | 5.041231 | 24.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b16_192 | cublas_pure | ok | 0.053 |  |  | 4.253539 | 20.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b8_192 | cublas_pure | ok | 0.030 |  |  | 3.813517 | 18.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_128 | torch_cuda | ok | 0.013 |  |  | 0.321255 | 1.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_256 | torch_cuda | ok | 0.015 |  |  | 2.284479 | 11.2% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_512 | torch_cuda | ok | 0.041 |  |  | 6.615621 | 32.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_4x256 | torch_cuda | ok | 0.043 |  |  | 3.086316 | 15.2% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tall_512x256x256 | torch_cuda | ok | 0.019 |  |  | 3.454945 | 17.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| odd_255x257x263 | torch_cuda | ok | 0.016 |  |  | 2.176225 | 10.7% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_1024 | torch_cuda | ok | 0.201 |  |  | 10.667440 | 52.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_2x512 | torch_cuda | ok | 0.076 |  |  | 7.031524 | 34.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| skinny_1024x128x512 | torch_cuda | ok | 0.031 |  |  | 4.306267 | 21.2% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| wide_128x1024x512 | torch_cuda | ok | 0.028 |  |  | 4.771677 | 23.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| small_k_1024x1024x64 | torch_cuda | ok | 0.027 |  |  | 4.981359 | 24.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| non_pow2_513x515x517 | torch_cuda | ok | 0.055 |  |  | 4.992281 | 24.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| non_pow2_1023x1025x1027 | torch_cuda | ok | 0.215 |  |  | 10.002290 | 49.2% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| medium_384 | torch_cuda | ok | 0.025 |  |  | 4.451502 | 21.9% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| medium_768 | torch_cuda | ok | 0.096 |  |  | 9.443480 | 46.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_8x256 | torch_cuda | ok | 0.047 |  |  | 5.722106 | 28.2% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_16x128 | torch_cuda | ok | 0.029 |  |  | 2.353706 | 11.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_32x64 | torch_cuda | ok | 0.020 |  |  | 0.826953 | 4.1% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_64x128 | torch_cuda | ok | 0.050 |  |  | 5.349878 | 26.3% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_128x64 | torch_cuda | ok | 0.042 |  |  | 1.597222 | 7.9% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_proj_2048x512x512 | torch_cuda | ok | 0.115 |  |  | 9.310331 | 45.8% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_proj_512x2048x512 | torch_cuda | ok | 0.110 |  |  | 9.737212 | 47.9% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_qkv_1024x3072x512 | torch_cuda | ok | 0.288 |  |  | 11.167439 | 55.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b32_128 | torch_cuda | ok | 0.033 |  |  | 4.060314 | 20.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b16_192 | torch_cuda | ok | 0.061 |  |  | 3.738979 | 18.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b8_192 | torch_cuda | ok | 0.036 |  |  | 3.112528 | 15.3% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_128 | cupy_cuda | ok | 0.049 |  |  | 0.084781 | 0.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_256 | cupy_cuda | ok | 0.051 |  |  | 0.659897 | 3.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_512 | cupy_cuda | ok | 0.078 |  |  | 3.437954 | 16.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_4x256 | cupy_cuda | ok | 0.083 |  |  | 1.618797 | 8.0% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tall_512x256x256 | cupy_cuda | ok | 0.055 |  |  | 1.217858 | 6.0% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| odd_255x257x263 | cupy_cuda | ok | 0.052 |  |  | 0.667016 | 3.3% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_1024 | cupy_cuda | ok | 0.239 |  |  | 8.966978 | 44.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_2x512 | cupy_cuda | ok | 0.117 |  |  | 4.607859 | 22.7% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| skinny_1024x128x512 | cupy_cuda | ok | 0.064 |  |  | 2.101355 | 10.3% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| wide_128x1024x512 | cupy_cuda | ok | 0.064 |  |  | 2.111936 | 10.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| small_k_1024x1024x64 | cupy_cuda | ok | 0.061 |  |  | 2.195971 | 10.8% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| non_pow2_513x515x517 | cupy_cuda | ok | 0.089 |  |  | 3.067481 | 15.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| non_pow2_1023x1025x1027 | cupy_cuda | ok | 0.251 |  |  | 8.579401 | 42.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| medium_384 | cupy_cuda | ok | 0.059 |  |  | 1.911909 | 9.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| medium_768 | cupy_cuda | ok | 0.134 |  |  | 6.760160 | 33.3% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_8x256 | cupy_cuda | ok | 0.084 |  |  | 3.200537 | 15.8% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_16x128 | cupy_cuda | ok | 0.066 |  |  | 1.015570 | 5.0% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_32x64 | cupy_cuda | ok | 0.058 |  |  | 0.290303 | 1.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_64x128 | cupy_cuda | ok | 0.087 |  |  | 3.069377 | 15.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_128x64 | cupy_cuda | ok | 0.079 |  |  | 0.848706 | 4.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_proj_2048x512x512 | cupy_cuda | ok | 0.151 |  |  | 7.130139 | 35.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_proj_512x2048x512 | cupy_cuda | ok | 0.146 |  |  | 7.355202 | 36.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_qkv_1024x3072x512 | cupy_cuda | ok | 0.324 |  |  | 9.945983 | 48.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b32_128 | cupy_cuda | ok | 0.070 |  |  | 1.909105 | 9.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b16_192 | cupy_cuda | ok | 0.098 |  |  | 2.301752 | 11.3% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b8_192 | cupy_cuda | ok | 0.073 |  |  | 1.554213 | 7.6% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
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
| ok | 67108864 | 25 | 8.840 | 9.446 | NVIDIA GeForce RTX 3070 (discrete) |

## Analysis

- `tensor-ash` used `NVIDIA GeForce RTX 3070 (discrete)`, so the Vulkan measurements reflect real GPU kernel timings on this host.
- Actual GPU framework comparisons succeeded for: `cublas_pure`, `cupy_cuda`, `torch_cuda`.
- Largest gap: `odd_255x257x263` is 1.2x faster in `cublas_pure` than `tensor-ash` in this environment.
- `tensor-ash` is the fastest measured backend on 13/26 benchmark cases.
- Throughput ratio versus `cublas_pure` across 26 shared cases: 0.81x to 2.52x, geometric mean 1.10x.
- Throughput ratio versus `torch_cuda` across 26 shared cases: 0.87x to 3.56x, geometric mean 1.32x.
- Throughput ratio versus `cupy_cuda` across 26 shared cases: 1.01x to 10.15x, geometric mean 2.55x.
- Throughput ratio versus `numpy` across 26 shared cases: 6.28x to 82.93x, geometric mean 46.75x.
- Throughput ratio versus `torch_cpu` across 26 shared cases: 6.46x to 85.06x, geometric mean 44.96x.
- Median `tensor-ash` host/submission overhead was 0.020 ms per synchronous call; GPU timestamp TFLOPS excludes that overhead.
- Some libraries were skipped because their Python modules or device backends were unavailable: `jax`, `tensorflow`.
- JAX and TensorFlow rows are included when those modules are installed; skipped rows mean the local Python environment does not provide them.
- Transfer staging bandwidth measured 8.84 GiB/s upload and 9.45 GiB/s download for 67108864 bytes.

## Optimization Gameplan

1. Keep benchmarking on this discrete-GPU baseline and tune the shape-based shader selector with larger production-like matrix sizes.
2. Use `scripts/tune_kernels.py` before accepting changes to the automatic selector.
3. Focus the next shader pass on large square GEMMs, where PyTorch CUDA still has the largest lead.
4. Keep PyTorch CUDA and CuPy CUDA rows in regular benchmark runs so Vulkan changes are compared against cuBLAS-backed GPU compute.
5. Add CI checks for `cargo fmt`, `cargo clippy`, CPU tests, and an optional GPU correctness job.
