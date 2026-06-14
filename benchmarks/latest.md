# Benchmark Report

This report compares FP32 GEMM throughput for `tensor-ash` against the local framework backends available on this machine.

## Environment

```text
[2026-06-14T09:02:15Z INFO  tensor_ash::context] tensor-ash: using device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329)
[2026-06-14T09:02:15Z INFO  ml_bench::bench] device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496774464, compute_family=2, timestamps=true) slots=2 kernel=auto
tensor-ash self-check
status: OK
selected: device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496774464, compute_family=2, timestamps=true)
executor_slots: 2
glslc: /run/current-system/sw/bin/glslc
vulkaninfo: /run/current-system/sw/bin/vulkaninfo
LD_LIBRARY_PATH: /nix/store/00qqlzn7kc79zlc351lcjdsk4q97y8a4-system-path/lib
VK_ICD_FILENAMES: <unset>
```

NVIDIA-SMI GPU summary:

```text
gpu0: name=NVIDIA GeForce RTX 3070, driver=595.71.05, memory_total_mib=8192, temperature_c=36, utilization_pct=0, power_draw_w=21.65, power_limit_w=220.00
```

- Iterations: 5
- Warmup iterations: 2
- Case set: showcase
- CPU library threads: 1
- CPU framework rows: skipped
- Python GPU framework rows: enabled

- Peak FP32 throughput used for `% peak`: 20.32 TFLOPS

## Results

| case | library | status | gpu ms | wall ms | host overhead ms | TFLOPS | % peak | details |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| square_128 | tensor-ash | ok | 0.006 | 0.026 | 0.020 | 0.661980 | 3.3% | NVIDIA GeForce RTX 3070 (discrete) |
| square_256 | tensor-ash | ok | 0.012 | 0.044 | 0.032 | 2.912711 | 14.3% | NVIDIA GeForce RTX 3070 (discrete) |
| square_512 | tensor-ash | ok | 0.042 | 0.062 | 0.020 | 6.457743 | 31.8% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_4x256 | tensor-ash | ok | 0.024 | 0.045 | 0.020 | 5.562737 | 27.4% | NVIDIA GeForce RTX 3070 (discrete) |
| tall_512x256x256 | tensor-ash | ok | 0.015 | 0.048 | 0.033 | 4.359983 | 21.5% | NVIDIA GeForce RTX 3070 (discrete) |
| odd_255x257x263 | tensor-ash | ok | 0.014 | 0.034 | 0.019 | 2.431674 | 12.0% | NVIDIA GeForce RTX 3070 (discrete) |
| square_1024 | tensor-ash | ok | 0.211 | 0.245 | 0.034 | 10.171092 | 50.1% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_2x512 | tensor-ash | ok | 0.068 | 0.088 | 0.020 | 7.858181 | 38.7% | NVIDIA GeForce RTX 3070 (discrete) |
| skinny_1024x128x512 | tensor-ash | ok | 0.026 | 0.047 | 0.021 | 5.152708 | 25.4% | NVIDIA GeForce RTX 3070 (discrete) |
| wide_128x1024x512 | tensor-ash | ok | 0.026 | 0.047 | 0.020 | 5.108775 | 25.1% | NVIDIA GeForce RTX 3070 (discrete) |
| small_k_1024x1024x64 | tensor-ash | ok | 0.020 | 0.041 | 0.020 | 6.689480 | 32.9% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_513x515x517 | tensor-ash | ok | 0.053 | 0.073 | 0.020 | 5.114920 | 25.2% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_1023x1025x1027 | tensor-ash | ok | 0.249 | 0.270 | 0.021 | 8.639975 | 42.5% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_384 | tensor-ash | ok | 0.021 | 0.041 | 0.020 | 5.345837 | 26.3% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_768 | tensor-ash | ok | 0.109 | 0.132 | 0.023 | 8.339191 | 41.0% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_8x256 | tensor-ash | ok | 0.037 | 0.057 | 0.020 | 7.351979 | 36.2% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_16x128 | tensor-ash | ok | 0.014 | 0.033 | 0.020 | 4.922892 | 24.2% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_32x64 | tensor-ash | ok | 0.006 | 0.025 | 0.019 | 2.702515 | 13.3% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_64x128 | tensor-ash | ok | 0.039 | 0.060 | 0.021 | 6.932734 | 34.1% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_128x64 | tensor-ash | ok | 0.017 | 0.038 | 0.021 | 4.025244 | 19.8% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_2048x512x512 | tensor-ash | ok | 0.111 | 0.144 | 0.033 | 9.661512 | 47.5% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_512x2048x512 | tensor-ash | ok | 0.110 | 0.132 | 0.022 | 9.728742 | 47.9% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_qkv_1024x3072x512 | tensor-ash | ok | 0.310 | 0.377 | 0.067 | 10.375520 | 51.1% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b32_128 | tensor-ash | ok | 0.021 | 0.053 | 0.032 | 6.384024 | 31.4% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b16_192 | tensor-ash | ok | 0.037 | 0.061 | 0.024 | 6.160042 | 30.3% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b8_192 | tensor-ash | ok | 0.020 | 0.041 | 0.021 | 5.698783 | 28.0% | NVIDIA GeForce RTX 3070 (discrete) |
| square_128 | cublas_pure | ok | 0.009 |  |  | 0.456697 | 2.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_256 | cublas_pure | ok | 0.012 |  |  | 2.730667 | 13.4% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_512 | cublas_pure | ok | 0.038 |  |  | 7.157516 | 35.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_4x256 | cublas_pure | ok | 0.038 |  |  | 3.557510 | 17.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tall_512x256x256 | cublas_pure | ok | 0.015 |  |  | 4.369067 | 21.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| odd_255x257x263 | cublas_pure | ok | 0.013 |  |  | 2.633818 | 13.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_1024 | cublas_pure | ok | 0.196 |  |  | 10.979853 | 54.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_2x512 | cublas_pure | ok | 0.069 |  |  | 7.825194 | 38.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| skinny_1024x128x512 | cublas_pure | ok | 0.026 |  |  | 5.256020 | 25.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| wide_128x1024x512 | cublas_pure | ok | 0.025 |  |  | 5.461333 | 26.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| small_k_1024x1024x64 | cublas_pure | ok | 0.024 |  |  | 5.698782 | 28.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_513x515x517 | cublas_pure | ok | 0.050 |  |  | 5.451341 | 26.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_1023x1025x1027 | cublas_pure | ok | 0.210 |  |  | 10.280344 | 50.6% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_384 | cublas_pure | ok | 0.022 |  |  | 5.114081 | 25.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_768 | cublas_pure | ok | 0.090 |  |  | 10.053818 | 49.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_8x256 | cublas_pure | ok | 0.041 |  |  | 6.558724 | 32.3% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_16x128 | cublas_pure | ok | 0.023 |  |  | 2.978909 | 14.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_32x64 | cublas_pure | ok | 0.014 |  |  | 1.170286 | 5.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_64x128 | cublas_pure | ok | 0.044 |  |  | 6.100806 | 30.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_128x64 | cublas_pure | ok | 0.035 |  |  | 1.908237 | 9.4% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_2048x512x512 | cublas_pure | ok | 0.110 |  |  | 9.791197 | 48.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_512x2048x512 | cublas_pure | ok | 0.105 |  |  | 10.198916 | 50.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_qkv_1024x3072x512 | cublas_pure | ok | 0.282 |  |  | 11.409192 | 56.1% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b32_128 | cublas_pure | ok | 0.027 |  |  | 4.882775 | 24.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b16_192 | cublas_pure | ok | 0.054 |  |  | 4.198036 | 20.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b8_192 | cublas_pure | ok | 0.030 |  |  | 3.813517 | 18.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_128 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| square_256 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| square_512 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| batched_4x256 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| tall_512x256x256 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| odd_255x257x263 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| square_1024 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| batched_2x512 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| skinny_1024x128x512 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| wide_128x1024x512 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| small_k_1024x1024x64 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| non_pow2_513x515x517 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| non_pow2_1023x1025x1027 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| medium_384 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| medium_768 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| batched_8x256 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| batched_16x128 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| batched_32x64 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| batched_64x128 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| batched_128x64 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| attn_proj_2048x512x512 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| attn_proj_512x2048x512 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| attn_qkv_1024x3072x512 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| tiny_b32_128 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| tiny_b16_192 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| tiny_b8_192 | torch_cuda | skipped |  |  |  |  |  | No module named 'torch' |
| square_128 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| square_256 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| square_512 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| batched_4x256 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| tall_512x256x256 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| odd_255x257x263 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| square_1024 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| batched_2x512 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| skinny_1024x128x512 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| wide_128x1024x512 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| small_k_1024x1024x64 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| non_pow2_513x515x517 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| non_pow2_1023x1025x1027 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| medium_384 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| medium_768 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| batched_8x256 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| batched_16x128 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| batched_32x64 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| batched_64x128 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| batched_128x64 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| attn_proj_2048x512x512 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| attn_proj_512x2048x512 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| attn_qkv_1024x3072x512 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| tiny_b32_128 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| tiny_b16_192 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
| tiny_b8_192 | cupy_cuda | skipped |  |  |  |  |  | No module named 'cupy' |
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

## Analysis

- `tensor-ash` used `NVIDIA GeForce RTX 3070 (discrete)`, so the Vulkan measurements reflect real GPU kernel timings on this host.
- Actual GPU framework comparisons succeeded for: `cublas_pure`.
- Largest gap: `medium_768` is 1.2x faster in `cublas_pure` than `tensor-ash` in this environment.
- `tensor-ash` is the fastest measured backend on 14/26 benchmark cases.
- Throughput ratio versus `cublas_pure` across 26 shared cases: 0.83x to 2.31x, geometric mean 1.15x.
- Median `tensor-ash` host/submission overhead was 0.021 ms per synchronous call; GPU timestamp TFLOPS excludes that overhead.
- Some libraries were skipped because their Python modules or device backends were unavailable: `cupy_cuda`, `jax`, `tensorflow`, `torch_cuda`.
- PyTorch CUDA/cuBLAS was not available in this Python environment.
- CuPy CUDA/cuBLAS was not available in this Python environment.
- JAX and TensorFlow rows are included when those modules are installed; skipped rows mean the local Python environment does not provide them.
- Transfer overhead is separately measurable with `ml_bench transfer`; use it to distinguish copy overhead from GEMM kernel time.

## Optimization Gameplan

1. Keep benchmarking on this discrete-GPU baseline and tune the shape-based shader selector with larger production-like matrix sizes.
2. Use `scripts/tune_kernels.py` before accepting changes to the automatic selector.
3. Focus the next shader pass on large square GEMMs, where PyTorch CUDA still has the largest lead.
4. Keep PyTorch CUDA and CuPy CUDA rows in regular benchmark runs so Vulkan changes are compared against cuBLAS-backed GPU compute.
5. Add CI checks for `cargo fmt`, `cargo clippy`, CPU tests, and an optional GPU correctness job.
