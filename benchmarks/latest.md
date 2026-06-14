# Benchmark Report

This report compares FP32 GEMM throughput for `tensor-ash` against the local framework backends available on this machine.

## Environment

```text
[2026-06-13T19:18:56Z INFO  tensor_ash::context] tensor-ash: using device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329)
[2026-06-13T19:18:57Z INFO  ml_bench::bench] device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496774464, compute_family=2, timestamps=true) slots=2 kernel=auto
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
gpu0: name=NVIDIA GeForce RTX 3070, driver=595.71.05, memory_total_mib=8192, temperature_c=38, utilization_pct=0, power_draw_w=23.44, power_limit_w=220.00
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
| square_128 | tensor-ash | ok | 0.006 | 0.025 | 0.018 | 0.661980 | 3.3% | NVIDIA GeForce RTX 3070 (discrete) |
| square_256 | tensor-ash | ok | 0.011 | 0.031 | 0.020 | 2.970470 | 14.6% | NVIDIA GeForce RTX 3070 (discrete) |
| square_512 | tensor-ash | ok | 0.042 | 0.061 | 0.019 | 6.423130 | 31.6% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_4x256 | tensor-ash | ok | 0.024 | 0.051 | 0.027 | 5.562737 | 27.4% | NVIDIA GeForce RTX 3070 (discrete) |
| tall_512x256x256 | tensor-ash | ok | 0.016 | 0.084 | 0.069 | 4.271185 | 21.0% | NVIDIA GeForce RTX 3070 (discrete) |
| odd_255x257x263 | tensor-ash | ok | 0.014 | 0.034 | 0.020 | 2.453830 | 12.1% | NVIDIA GeForce RTX 3070 (discrete) |
| square_1024 | tensor-ash | ok | 0.211 | 0.235 | 0.024 | 10.191171 | 50.2% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_2x512 | tensor-ash | ok | 0.068 | 0.087 | 0.020 | 7.902598 | 38.9% | NVIDIA GeForce RTX 3070 (discrete) |
| skinny_1024x128x512 | tensor-ash | ok | 0.028 | 0.047 | 0.019 | 4.826587 | 23.8% | NVIDIA GeForce RTX 3070 (discrete) |
| wide_128x1024x512 | tensor-ash | ok | 0.028 | 0.047 | 0.019 | 4.821039 | 23.7% | NVIDIA GeForce RTX 3070 (discrete) |
| small_k_1024x1024x64 | tensor-ash | ok | 0.020 | 0.088 | 0.068 | 6.710886 | 33.0% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_513x515x517 | tensor-ash | ok | 0.053 | 0.071 | 0.018 | 5.164429 | 25.4% | NVIDIA GeForce RTX 3070 (discrete) |
| non_pow2_1023x1025x1027 | tensor-ash | ok | 0.250 | 0.349 | 0.099 | 8.632219 | 42.5% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_384 | tensor-ash | ok | 0.022 | 0.044 | 0.022 | 5.114081 | 25.2% | NVIDIA GeForce RTX 3070 (discrete) |
| medium_768 | tensor-ash | ok | 0.108 | 0.169 | 0.061 | 8.366298 | 41.2% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_8x256 | tensor-ash | ok | 0.037 | 0.065 | 0.028 | 7.332699 | 36.1% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_16x128 | tensor-ash | ok | 0.014 | 0.035 | 0.021 | 4.854519 | 23.9% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_32x64 | tensor-ash | ok | 0.006 | 0.084 | 0.079 | 2.995931 | 14.7% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_64x128 | tensor-ash | ok | 0.039 | 0.103 | 0.064 | 6.892858 | 33.9% | NVIDIA GeForce RTX 3070 (discrete) |
| batched_128x64 | tensor-ash | ok | 0.016 | 0.035 | 0.019 | 4.080062 | 20.1% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_2048x512x512 | tensor-ash | ok | 0.111 | 0.181 | 0.070 | 9.655952 | 47.5% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_proj_512x2048x512 | tensor-ash | ok | 0.109 | 0.138 | 0.029 | 9.834242 | 48.4% | NVIDIA GeForce RTX 3070 (discrete) |
| attn_qkv_1024x3072x512 | tensor-ash | ok | 0.309 | 0.348 | 0.039 | 10.412008 | 51.2% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b32_128 | tensor-ash | ok | 0.022 | 0.042 | 0.020 | 6.159037 | 30.3% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b16_192 | tensor-ash | ok | 0.036 | 0.056 | 0.019 | 6.208674 | 30.6% | NVIDIA GeForce RTX 3070 (discrete) |
| tiny_b8_192 | tensor-ash | ok | 0.019 | 0.040 | 0.020 | 5.811074 | 28.6% | NVIDIA GeForce RTX 3070 (discrete) |
| square_128 | cublas_pure | ok | 0.008 |  |  | 0.532813 | 2.6% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_256 | cublas_pure | ok | 0.012 |  |  | 2.730667 | 13.4% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_512 | cublas_pure | ok | 0.036 |  |  | 7.489828 | 36.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_4x256 | cublas_pure | ok | 0.037 |  |  | 3.640889 | 17.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tall_512x256x256 | cublas_pure | ok | 0.015 |  |  | 4.405782 | 21.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| odd_255x257x263 | cublas_pure | ok | 0.012 |  |  | 2.872617 | 14.1% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| square_1024 | cublas_pure | ok | 0.195 |  |  | 11.037642 | 54.3% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_2x512 | cublas_pure | ok | 0.069 |  |  | 7.825194 | 38.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| skinny_1024x128x512 | cublas_pure | ok | 0.025 |  |  | 5.461333 | 26.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| wide_128x1024x512 | cublas_pure | ok | 0.025 |  |  | 5.468454 | 26.9% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| small_k_1024x1024x64 | cublas_pure | ok | 0.022 |  |  | 6.000435 | 29.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_513x515x517 | cublas_pure | ok | 0.049 |  |  | 5.557813 | 27.4% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| non_pow2_1023x1025x1027 | cublas_pure | ok | 0.209 |  |  | 10.310265 | 50.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_384 | cublas_pure | ok | 0.021 |  |  | 5.289902 | 26.0% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| medium_768 | cublas_pure | ok | 0.090 |  |  | 10.053818 | 49.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_8x256 | cublas_pure | ok | 0.041 |  |  | 6.610408 | 32.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_16x128 | cublas_pure | ok | 0.022 |  |  | 3.013149 | 14.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_32x64 | cublas_pure | ok | 0.015 |  |  | 1.103764 | 5.4% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_64x128 | cublas_pure | ok | 0.043 |  |  | 6.241524 | 30.7% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| batched_128x64 | cublas_pure | ok | 0.035 |  |  | 1.927529 | 9.5% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_2048x512x512 | cublas_pure | ok | 0.110 |  |  | 9.805503 | 48.3% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_proj_512x2048x512 | cublas_pure | ok | 0.105 |  |  | 10.202017 | 50.2% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| attn_qkv_1024x3072x512 | cublas_pure | ok | 0.282 |  |  | 11.439011 | 56.3% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b32_128 | cublas_pure | ok | 0.027 |  |  | 5.041231 | 24.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
| tiny_b16_192 | cublas_pure | ok | 0.054 |  |  | 4.230656 | 20.8% | pure cuBLAS, FP32 forced (CUBLAS_PEDANTIC_MATH), CUDA events |
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
- `tensor-ash` is the fastest measured backend on 13/26 benchmark cases.
- Throughput ratio versus `cublas_pure` across 26 shared cases: 0.83x to 2.71x, geometric mean 1.12x.
- Median `tensor-ash` host/submission overhead was 0.022 ms per synchronous call; GPU timestamp TFLOPS excludes that overhead.
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
