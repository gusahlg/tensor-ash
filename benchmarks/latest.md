# Benchmark Report

This report compares FP32 GEMM throughput for `tensor-ash` against the local framework backends available on this machine.

## Environment

```text
[2026-06-11T07:32:14Z INFO  tensor_ash::context] tensor-ash: using device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329)
[2026-06-11T07:32:14Z INFO  ml_bench::bench] device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496774464, compute_family=2, timestamps=true) slots=2
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
gpu0: name=NVIDIA GeForce RTX 3070, driver=595.71.05, memory_total_mib=8192, temperature_c=36, utilization_pct=0, power_draw_w=19.87, power_limit_w=220.00
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
| square_128 | tensor-ash | ok | 0.006 | 0.024 | 0.018 | 0.675629 | 3.3% | NVIDIA GeForce RTX 3070 (discrete) |
| square_256 | tensor-ash | ok | 0.011 | 0.036 | 0.025 | 2.945438 | 14.5% | NVIDIA GeForce RTX 3070 (discrete) |
| square_512 | tensor-ash | ok | 0.045 | 0.062 | 0.018 | 6.000435 | 29.5% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_4x256 | tensor-ash | ok | 0.024 | 0.042 | 0.018 | 5.637505 | 27.7% | NVIDIA GeForce RTX 3070 (discrete) |
| tall_512x256x256 | tensor-ash | ok | 0.016 | 0.034 | 0.018 | 4.194304 | 20.6% | NVIDIA GeForce RTX 3070 (discrete) |
| odd_255x257x263 | tensor-ash | ok | 0.014 | 0.031 | 0.017 | 2.453830 | 12.1% | NVIDIA GeForce RTX 3070 (discrete) |
| square_1024 | tensor-ash | ok | 0.233 | 0.252 | 0.019 | 9.216984 | 45.4% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_2x512 | tensor-ash | ok | 0.071 | 0.089 | 0.018 | 7.570946 | 37.3% | NVIDIA GeForce RTX 3070 (discrete) |
| skinny_1024x128x512 | tensor-ash | ok | 0.028 | 0.054 | 0.026 | 4.782559 | 23.5% | NVIDIA GeForce RTX 3070 (discrete) |
| wide_128x1024x512 | tensor-ash | ok | 0.029 | 0.046 | 0.017 | 4.707412 | 23.2% | NVIDIA GeForce RTX 3070 (discrete) |
| small_k_1024x1024x64 | tensor-ash | ok | 0.024 | 0.042 | 0.018 | 5.511569 | 27.1% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_513x515x517 | tensor-ash | ok | 0.053 | 0.070 | 0.017 | 5.189545 | 25.5% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_1023x1025x1027 | tensor-ash | ok | 0.262 | 0.280 | 0.019 | 8.230057 | 40.5% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_384 | tensor-ash | ok | 0.023 | 0.040 | 0.017 | 4.942659 | 24.3% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_768 | tensor-ash | ok | 0.120 | 0.138 | 0.018 | 7.523665 | 37.0% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_8x256 | tensor-ash | ok | 0.038 | 0.056 | 0.018 | 7.067067 | 34.8% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_16x128 | tensor-ash | ok | 0.015 | 0.033 | 0.017 | 4.415057 | 21.7% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_32x64 | tensor-ash | ok | 0.006 | 0.024 | 0.017 | 2.608398 | 12.8% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_64x128 | tensor-ash | ok | 0.043 | 0.060 | 0.017 | 6.264830 | 30.8% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_128x64 | tensor-ash | ok | 0.018 | 0.035 | 0.017 | 3.738239 | 18.4% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_2048x512x512 | tensor-ash | ok | 0.121 | 0.139 | 0.018 | 8.874486 | 43.7% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_512x2048x512 | tensor-ash | ok | 0.122 | 0.139 | 0.018 | 8.834764 | 43.5% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_qkv_1024x3072x512 | tensor-ash | ok | 0.352 | 0.380 | 0.028 | 9.160369 | 45.1% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b32_128 | tensor-ash | ok | 0.024 | 0.041 | 0.018 | 5.667978 | 27.9% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b16_192 | tensor-ash | ok | 0.039 | 0.057 | 0.018 | 5.796796 | 28.5% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b8_192 | tensor-ash | ok | 0.022 | 0.039 | 0.018 | 5.266286 | 25.9% | NVIDIA GeForce RTX 3070 (discrete) |
| square_128 | numpy | ok | 0.039 |  |  | 0.107126 | 0.5% | numpy 2.4.4, threads=1 |
| square_256 | numpy | ok | 0.263 |  |  | 0.127818 | 0.6% | numpy 2.4.4, threads=1 |
| square_512 | numpy | ok | 2.045 |  |  | 0.131272 | 0.6% | numpy 2.4.4, threads=1 |
| batched_4x256 | numpy | ok | 1.053 |  |  | 0.127507 | 0.6% | numpy 2.4.4, threads=1 |
| tall_512x256x256 | numpy | ok | 0.537 |  |  | 0.124924 | 0.6% | numpy 2.4.4, threads=1 |
| odd_255x257x263 | numpy | ok | 0.295 |  |  | 0.116798 | 0.6% | numpy 2.4.4, threads=1 |
| square_1024 | numpy | ok | 16.072 |  |  | 0.133620 | 0.7% | numpy 2.4.4, threads=1 |
| batched_2x512 | numpy | ok | 4.123 |  |  | 0.130208 | 0.6% | numpy 2.4.4, threads=1 |
| skinny_1024x128x512 | numpy | ok | 1.052 |  |  | 0.127574 | 0.6% | numpy 2.4.4, threads=1 |
| wide_128x1024x512 | numpy | ok | 1.132 |  |  | 0.118609 | 0.6% | numpy 2.4.4, threads=1 |
| small_k_1024x1024x64 | numpy | ok | 1.124 |  |  | 0.119458 | 0.6% | numpy 2.4.4, threads=1 |
| non_pow2_513x515x517 | numpy | ok | 2.178 |  |  | 0.125438 | 0.6% | numpy 2.4.4, threads=1 |
| non_pow2_1023x1025x1027 | numpy | ok | 16.574 |  |  | 0.129946 | 0.6% | numpy 2.4.4, threads=1 |
| medium_384 | numpy | ok | 0.874 |  |  | 0.129593 | 0.6% | numpy 2.4.4, threads=1 |
| medium_768 | numpy | ok | 6.780 |  |  | 0.133618 | 0.7% | numpy 2.4.4, threads=1 |
| batched_8x256 | numpy | ok | 2.100 |  |  | 0.127849 | 0.6% | numpy 2.4.4, threads=1 |
| batched_16x128 | numpy | ok | 0.572 |  |  | 0.117280 | 0.6% | numpy 2.4.4, threads=1 |
| batched_32x64 | numpy | ok | 0.164 |  |  | 0.102493 | 0.5% | numpy 2.4.4, threads=1 |
| batched_64x128 | numpy | ok | 2.349 |  |  | 0.114276 | 0.6% | numpy 2.4.4, threads=1 |
| batched_128x64 | numpy | ok | 0.650 |  |  | 0.103223 | 0.5% | numpy 2.4.4, threads=1 |
| attn_proj_2048x512x512 | numpy | ok | 8.152 |  |  | 0.131715 | 0.6% | numpy 2.4.4, threads=1 |
| attn_proj_512x2048x512 | numpy | ok | 8.180 |  |  | 0.131262 | 0.6% | numpy 2.4.4, threads=1 |
| attn_qkv_1024x3072x512 | numpy | ok | 24.008 |  |  | 0.134172 | 0.7% | numpy 2.4.4, threads=1 |
| tiny_b32_128 | numpy | ok | 1.145 |  |  | 0.117240 | 0.6% | numpy 2.4.4, threads=1 |
| tiny_b16_192 | numpy | ok | 1.808 |  |  | 0.125278 | 0.6% | numpy 2.4.4, threads=1 |
| tiny_b8_192 | numpy | ok | 0.906 |  |  | 0.124973 | 0.6% | numpy 2.4.4, threads=1 |
| square_128 | torch_cpu | ok | 0.038 |  |  | 0.111084 | 0.5% | torch 2.11.0+cu128, threads=1 |
| square_256 | torch_cpu | ok | 0.260 |  |  | 0.129154 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_512 | torch_cpu | ok | 2.069 |  |  | 0.129751 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_4x256 | torch_cpu | ok | 1.044 |  |  | 0.128608 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tall_512x256x256 | torch_cpu | ok | 0.518 |  |  | 0.129450 | 0.6% | torch 2.11.0+cu128, threads=1 |
| odd_255x257x263 | torch_cpu | ok | 0.289 |  |  | 0.119206 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_1024 | torch_cpu | ok | 16.376 |  |  | 0.131134 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_2x512 | torch_cpu | ok | 4.156 |  |  | 0.129175 | 0.6% | torch 2.11.0+cu128, threads=1 |
| skinny_1024x128x512 | torch_cpu | ok | 1.091 |  |  | 0.123025 | 0.6% | torch 2.11.0+cu128, threads=1 |
| wide_128x1024x512 | torch_cpu | ok | 1.059 |  |  | 0.126701 | 0.6% | torch 2.11.0+cu128, threads=1 |
| small_k_1024x1024x64 | torch_cpu | ok | 1.015 |  |  | 0.132173 | 0.7% | torch 2.11.0+cu128, threads=1 |
| non_pow2_513x515x517 | torch_cpu | ok | 2.203 |  |  | 0.124030 | 0.6% | torch 2.11.0+cu128, threads=1 |
| non_pow2_1023x1025x1027 | torch_cpu | ok | 16.757 |  |  | 0.128534 | 0.6% | torch 2.11.0+cu128, threads=1 |
| medium_384 | torch_cpu | ok | 0.856 |  |  | 0.132354 | 0.7% | torch 2.11.0+cu128, threads=1 |
| medium_768 | torch_cpu | ok | 6.842 |  |  | 0.132419 | 0.7% | torch 2.11.0+cu128, threads=1 |
| batched_8x256 | torch_cpu | ok | 2.123 |  |  | 0.126461 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_16x128 | torch_cpu | ok | 0.562 |  |  | 0.119509 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_32x64 | torch_cpu | ok | 0.137 |  |  | 0.122125 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_64x128 | torch_cpu | ok | 2.226 |  |  | 0.120586 | 0.6% | torch 2.11.0+cu128, threads=1 |
| batched_128x64 | torch_cpu | ok | 0.532 |  |  | 0.126149 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_proj_2048x512x512 | torch_cpu | ok | 8.185 |  |  | 0.131190 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_proj_512x2048x512 | torch_cpu | ok | 8.266 |  |  | 0.129896 | 0.6% | torch 2.11.0+cu128, threads=1 |
| attn_qkv_1024x3072x512 | torch_cpu | ok | 25.330 |  |  | 0.127173 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b32_128 | torch_cpu | ok | 1.118 |  |  | 0.120083 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b16_192 | torch_cpu | ok | 1.764 |  |  | 0.128430 | 0.6% | torch 2.11.0+cu128, threads=1 |
| tiny_b8_192 | torch_cpu | ok | 0.884 |  |  | 0.128141 | 0.6% | torch 2.11.0+cu128, threads=1 |
| square_128 | cublas_pure | ok | 0.007 |  |  | 0.585143 | 2.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_256 | cublas_pure | ok | 0.011 |  |  | 3.048186 | 15.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_512 | cublas_pure | ok | 0.036 |  |  | 7.489828 | 36.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_4x256 | cublas_pure | ok | 0.037 |  |  | 3.640889 | 17.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tall_512x256x256 | cublas_pure | ok | 0.014 |  |  | 4.681143 | 23.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| odd_255x257x263 | cublas_pure | ok | 0.012 |  |  | 2.888020 | 14.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_1024 | cublas_pure | ok | 0.196 |  |  | 10.945827 | 53.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_2x512 | cublas_pure | ok | 0.069 |  |  | 7.825194 | 38.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| skinny_1024x128x512 | cublas_pure | ok | 0.025 |  |  | 5.461333 | 26.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| wide_128x1024x512 | cublas_pure | ok | 0.025 |  |  | 5.468454 | 26.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| small_k_1024x1024x64 | cublas_pure | ok | 0.022 |  |  | 5.983315 | 29.4% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_513x515x517 | cublas_pure | ok | 0.050 |  |  | 5.451341 | 26.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_1023x1025x1027 | cublas_pure | ok | 0.210 |  |  | 10.259971 | 50.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_384 | cublas_pure | ok | 0.020 |  |  | 5.529600 | 27.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_768 | cublas_pure | ok | 0.091 |  |  | 9.940854 | 48.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_8x256 | cublas_pure | ok | 0.040 |  |  | 6.662913 | 32.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_16x128 | cublas_pure | ok | 0.022 |  |  | 3.013149 | 14.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_32x64 | cublas_pure | ok | 0.014 |  |  | 1.170286 | 5.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_64x128 | cublas_pure | ok | 0.043 |  |  | 6.241524 | 30.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_128x64 | cublas_pure | ok | 0.035 |  |  | 1.927529 | 9.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_2048x512x512 | cublas_pure | ok | 0.110 |  |  | 9.728742 | 47.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_512x2048x512 | cublas_pure | ok | 0.105 |  |  | 10.180350 | 50.1% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_qkv_1024x3072x512 | cublas_pure | ok | 0.282 |  |  | 11.439011 | 56.3% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b32_128 | cublas_pure | ok | 0.027 |  |  | 5.041231 | 24.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b16_192 | cublas_pure | ok | 0.053 |  |  | 4.253539 | 20.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b8_192 | cublas_pure | ok | 0.030 |  |  | 3.813517 | 18.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_128 | torch_cuda | ok | 0.013 |  |  | 0.334367 | 1.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_256 | torch_cuda | ok | 0.014 |  |  | 2.314737 | 11.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_512 | torch_cuda | ok | 0.040 |  |  | 6.668210 | 32.8% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_4x256 | torch_cuda | ok | 0.042 |  |  | 3.158361 | 15.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tall_512x256x256 | torch_cuda | ok | 0.019 |  |  | 3.597173 | 17.7% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| odd_255x257x263 | torch_cuda | ok | 0.016 |  |  | 2.221096 | 10.9% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_1024 | torch_cuda | ok | 0.200 |  |  | 10.727120 | 52.8% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_2x512 | torch_cuda | ok | 0.075 |  |  | 7.200522 | 35.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| skinny_1024x128x512 | torch_cuda | ok | 0.028 |  |  | 4.821039 | 23.7% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| wide_128x1024x512 | torch_cuda | ok | 0.027 |  |  | 4.888466 | 24.1% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| small_k_1024x1024x64 | torch_cuda | ok | 0.026 |  |  | 5.096360 | 25.1% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| non_pow2_513x515x517 | torch_cuda | ok | 0.053 |  |  | 5.124130 | 25.2% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| non_pow2_1023x1025x1027 | torch_cuda | ok | 0.214 |  |  | 10.047083 | 49.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| medium_384 | torch_cuda | ok | 0.025 |  |  | 4.572279 | 22.5% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| medium_768 | torch_cuda | ok | 0.096 |  |  | 9.471915 | 46.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_8x256 | torch_cuda | ok | 0.046 |  |  | 5.886742 | 29.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_16x128 | torch_cuda | ok | 0.028 |  |  | 2.430072 | 12.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_32x64 | torch_cuda | ok | 0.020 |  |  | 0.859489 | 4.2% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_64x128 | torch_cuda | ok | 0.049 |  |  | 5.482750 | 27.0% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_128x64 | torch_cuda | ok | 0.041 |  |  | 1.648704 | 8.1% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_proj_2048x512x512 | torch_cuda | ok | 0.114 |  |  | 9.404269 | 46.3% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_proj_512x2048x512 | torch_cuda | ok | 0.110 |  |  | 9.791197 | 48.2% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| attn_qkv_1024x3072x512 | torch_cuda | ok | 0.287 |  |  | 11.227225 | 55.3% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b32_128 | torch_cuda | ok | 0.032 |  |  | 4.140478 | 20.4% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b16_192 | torch_cuda | ok | 0.060 |  |  | 3.786992 | 18.6% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tiny_b8_192 | torch_cuda | ok | 0.035 |  |  | 3.208471 | 15.8% | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_128 | cupy_cuda | ok | 0.048 |  |  | 0.086573 | 0.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_256 | cupy_cuda | ok | 0.049 |  |  | 0.679130 | 3.3% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_512 | cupy_cuda | ok | 0.076 |  |  | 3.548480 | 17.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_4x256 | cupy_cuda | ok | 0.081 |  |  | 1.657173 | 8.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tall_512x256x256 | cupy_cuda | ok | 0.054 |  |  | 1.254277 | 6.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| odd_255x257x263 | cupy_cuda | ok | 0.048 |  |  | 0.714345 | 3.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_1024 | cupy_cuda | ok | 0.236 |  |  | 9.109388 | 44.8% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_2x512 | cupy_cuda | ok | 0.113 |  |  | 4.733977 | 23.3% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| skinny_1024x128x512 | cupy_cuda | ok | 0.063 |  |  | 2.145424 | 10.6% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| wide_128x1024x512 | cupy_cuda | ok | 0.063 |  |  | 2.145424 | 10.6% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| small_k_1024x1024x64 | cupy_cuda | ok | 0.060 |  |  | 2.255002 | 11.1% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| non_pow2_513x515x517 | cupy_cuda | ok | 0.088 |  |  | 3.093044 | 15.2% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| non_pow2_1023x1025x1027 | cupy_cuda | ok | 0.236 |  |  | 9.121210 | 44.9% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| medium_384 | cupy_cuda | ok | 0.057 |  |  | 1.994895 | 9.8% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| medium_768 | cupy_cuda | ok | 0.124 |  |  | 7.308093 | 36.0% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_8x256 | cupy_cuda | ok | 0.080 |  |  | 3.350083 | 16.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_16x128 | cupy_cuda | ok | 0.062 |  |  | 1.084360 | 5.3% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_32x64 | cupy_cuda | ok | 0.054 |  |  | 0.309132 | 1.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_64x128 | cupy_cuda | ok | 0.083 |  |  | 3.245110 | 16.0% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_128x64 | cupy_cuda | ok | 0.075 |  |  | 0.893927 | 4.4% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_proj_2048x512x512 | cupy_cuda | ok | 0.141 |  |  | 7.627741 | 37.5% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_proj_512x2048x512 | cupy_cuda | ok | 0.137 |  |  | 7.865549 | 38.7% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| attn_qkv_1024x3072x512 | cupy_cuda | ok | 0.302 |  |  | 10.679323 | 52.6% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b32_128 | cupy_cuda | ok | 0.067 |  |  | 1.990652 | 9.8% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b16_192 | cupy_cuda | ok | 0.093 |  |  | 2.443179 | 12.0% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tiny_b8_192 | cupy_cuda | ok | 0.070 |  |  | 1.624859 | 8.0% | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
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
| ok | 67108864 | 30 | 9.732 | 10.259 | NVIDIA GeForce RTX 3070 (discrete) |

## Analysis

- `tensor-ash` used `NVIDIA GeForce RTX 3070 (discrete)`, so the Vulkan measurements reflect real GPU kernel timings on this host.
- Actual GPU framework comparisons succeeded for: `cublas_pure`, `cupy_cuda`, `torch_cuda`.
- Largest gap: `medium_768` is 1.3x faster in `cublas_pure` than `tensor-ash` in this environment.
- `tensor-ash` is the fastest measured backend on 10/26 benchmark cases.
- Throughput ratio versus `cublas_pure` across 26 shared cases: 0.76x to 2.23x, geometric mean 1.04x.
- Throughput ratio versus `torch_cuda` across 26 shared cases: 0.79x to 3.03x, geometric mean 1.21x.
- Throughput ratio versus `cupy_cuda` across 26 shared cases: 0.86x to 8.44x, geometric mean 2.29x.
- Throughput ratio versus `numpy` across 26 shared cases: 6.31x to 68.98x, geometric mean 41.22x.
- Throughput ratio versus `torch_cpu` across 26 shared cases: 6.08x to 72.03x, geometric mean 40.22x.
- Median `tensor-ash` host/submission overhead was 0.018 ms per synchronous call; GPU timestamp TFLOPS excludes that overhead.
- Some libraries were skipped because their Python modules or device backends were unavailable: `jax`, `tensorflow`.
- JAX and TensorFlow rows are included when those modules are installed; skipped rows mean the local Python environment does not provide them.
- Transfer staging bandwidth measured 9.73 GiB/s upload and 10.26 GiB/s download for 67108864 bytes.

## Optimization Gameplan

1. Keep benchmarking on this discrete-GPU baseline and tune the shape-based shader selector with larger production-like matrix sizes.
2. Use `scripts/tune_kernels.py` before accepting changes to the automatic selector.
3. Focus the next shader pass on large square GEMMs, where PyTorch CUDA still has the largest lead.
4. Keep PyTorch CUDA and CuPy CUDA rows in regular benchmark runs so Vulkan changes are compared against cuBLAS-backed GPU compute.
5. Add CI checks for `cargo fmt`, `cargo clippy`, CPU tests, and an optional GPU correctness job.
