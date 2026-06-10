# Benchmark Report

This report compares FP32 GEMM throughput for `tensor-ash` against the local framework backends available on this machine.

## Environment

```text
[2026-06-10T15:37:28Z INFO  tensor_ash::context] tensor-ash: using device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329)
[2026-06-10T15:37:29Z INFO  ml_bench::bench] device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496774464, compute_family=2, timestamps=true) slots=2
tensor-ash self-check
status: OK
selected: device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496774464, compute_family=2, timestamps=true)
executor_slots: 2
glslc: /nix/store/x90sdkmir007jaxx1s893iqgw5kxxc5w-shaderc-2026.1-bin/bin/glslc
vulkaninfo: /nix/store/pq7kd4gdh4ll73d83mlwhhzbjdl6fvl9-vulkan-tools-1.4.341.0/bin/vulkaninfo
LD_LIBRARY_PATH: /nix/store/d6vnmfmcz5b299180issiaad4m96wh8k-vulkan-loader-1.4.341.0/lib:/nix/store/chqq8mpmpyfi9kgsngya71akv5xicn03-gcc-15.2.0-lib/lib:/run/opengl-driver/lib:
VK_ICD_FILENAMES: <unset>
```

NVIDIA-SMI GPU summary:

```text
gpu0: name=NVIDIA GeForce RTX 3070, driver=595.71.05, memory_total_mib=8192, temperature_c=39, utilization_pct=6, power_draw_w=19.62, power_limit_w=220.00
```

- Iterations: 40
- Warmup iterations: 10
- Case set: extended
- CPU library threads: 1
- CPU framework rows: enabled
- Python GPU framework rows: enabled

## Results

| case | library | status | gpu ms | wall ms | host overhead ms | TFLOPS | details |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| square_128 | tensor-ash | ok | 0.007 | 0.026 | 0.019 | 0.590414 | NVIDIA GeForce RTX 3070 (discrete) |
| square_256 | tensor-ash | ok | 0.012 | 0.031 | 0.019 | 2.730667 | NVIDIA GeForce RTX 3070 (discrete) |
| square_512 | tensor-ash | ok | 0.046 | 0.064 | 0.019 | 5.890876 | NVIDIA GeForce RTX 3070 (discrete) |
| batched_4x256 | tensor-ash | ok | 0.026 | 0.045 | 0.019 | 5.184554 | NVIDIA GeForce RTX 3070 (discrete) |
| tall_512x256x256 | tensor-ash | ok | 0.016 | 0.036 | 0.020 | 4.088016 | NVIDIA GeForce RTX 3070 (discrete) |
| odd_255x257x263 | tensor-ash | ok | 0.015 | 0.034 | 0.019 | 2.352034 | NVIDIA GeForce RTX 3070 (discrete) |
| square_1024 | tensor-ash | ok | 0.239 | 0.268 | 0.029 | 8.987393 | NVIDIA GeForce RTX 3070 (discrete) |
| batched_2x512 | tensor-ash | ok | 0.073 | 0.092 | 0.019 | 7.364888 | NVIDIA GeForce RTX 3070 (discrete) |
| skinny_1024x128x512 | tensor-ash | ok | 0.029 | 0.048 | 0.019 | 4.660338 | NVIDIA GeForce RTX 3070 (discrete) |
| wide_128x1024x512 | tensor-ash | ok | 0.029 | 0.048 | 0.019 | 4.696869 | NVIDIA GeForce RTX 3070 (discrete) |
| small_k_1024x1024x64 | tensor-ash | ok | 0.026 | 0.045 | 0.019 | 5.190970 | NVIDIA GeForce RTX 3070 (discrete) |
| square_128 | numpy | ok | 0.037 |  |  | 0.112415 | numpy 2.4.4, threads=1 |
| square_256 | numpy | ok | 0.274 |  |  | 0.122367 | numpy 2.4.4, threads=1 |
| square_512 | numpy | ok | 2.075 |  |  | 0.129381 | numpy 2.4.4, threads=1 |
| batched_4x256 | numpy | ok | 1.050 |  |  | 0.127801 | numpy 2.4.4, threads=1 |
| tall_512x256x256 | numpy | ok | 0.523 |  |  | 0.128235 | numpy 2.4.4, threads=1 |
| odd_255x257x263 | numpy | ok | 0.282 |  |  | 0.122354 | numpy 2.4.4, threads=1 |
| square_1024 | numpy | ok | 16.276 |  |  | 0.131945 | numpy 2.4.4, threads=1 |
| batched_2x512 | numpy | ok | 4.143 |  |  | 0.129592 | numpy 2.4.4, threads=1 |
| skinny_1024x128x512 | numpy | ok | 1.054 |  |  | 0.127373 | numpy 2.4.4, threads=1 |
| wide_128x1024x512 | numpy | ok | 1.102 |  |  | 0.121809 | numpy 2.4.4, threads=1 |
| small_k_1024x1024x64 | numpy | ok | 1.123 |  |  | 0.119570 | numpy 2.4.4, threads=1 |
| square_128 | torch_cpu | ok | 0.039 |  |  | 0.108596 | torch 2.11.0+cu128, threads=1 |
| square_256 | torch_cpu | ok | 0.260 |  |  | 0.128954 | torch 2.11.0+cu128, threads=1 |
| square_512 | torch_cpu | ok | 2.070 |  |  | 0.129676 | torch 2.11.0+cu128, threads=1 |
| batched_4x256 | torch_cpu | ok | 1.043 |  |  | 0.128732 | torch 2.11.0+cu128, threads=1 |
| tall_512x256x256 | torch_cpu | ok | 0.520 |  |  | 0.129174 | torch 2.11.0+cu128, threads=1 |
| odd_255x257x263 | torch_cpu | ok | 0.289 |  |  | 0.119259 | torch 2.11.0+cu128, threads=1 |
| square_1024 | torch_cpu | ok | 16.853 |  |  | 0.127424 | torch 2.11.0+cu128, threads=1 |
| batched_2x512 | torch_cpu | ok | 4.183 |  |  | 0.128361 | torch 2.11.0+cu128, threads=1 |
| skinny_1024x128x512 | torch_cpu | ok | 1.090 |  |  | 0.123180 | torch 2.11.0+cu128, threads=1 |
| wide_128x1024x512 | torch_cpu | ok | 1.066 |  |  | 0.125914 | torch 2.11.0+cu128, threads=1 |
| small_k_1024x1024x64 | torch_cpu | ok | 1.202 |  |  | 0.111691 | torch 2.11.0+cu128, threads=1 |
| square_128 | torch_cuda | ok | 0.012 |  |  | 0.339565 | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_256 | torch_cuda | ok | 0.014 |  |  | 2.319858 | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_512 | torch_cuda | ok | 0.040 |  |  | 6.732430 | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_4x256 | torch_cuda | ok | 0.043 |  |  | 3.153612 | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| tall_512x256x256 | torch_cuda | ok | 0.018 |  |  | 3.647221 | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| odd_255x257x263 | torch_cuda | ok | 0.015 |  |  | 2.263092 | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_1024 | torch_cuda | ok | 0.200 |  |  | 10.759798 | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| batched_2x512 | torch_cuda | ok | 0.075 |  |  | 7.166688 | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| skinny_1024x128x512 | torch_cuda | ok | 0.028 |  |  | 4.815504 | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| wide_128x1024x512 | torch_cuda | ok | 0.028 |  |  | 4.860144 | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| small_k_1024x1024x64 | torch_cuda | ok | 0.027 |  |  | 5.029141 | torch 2.11.0+cu128, CUDA/cuBLAS, NVIDIA GeForce RTX 3070, allow_tf32=False, precision=highest |
| square_128 | cupy_cuda | ok | 0.047 |  |  | 0.088622 | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_256 | cupy_cuda | ok | 0.050 |  |  | 0.669589 | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_512 | cupy_cuda | ok | 0.076 |  |  | 3.551485 | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_4x256 | cupy_cuda | ok | 0.080 |  |  | 1.669043 | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| tall_512x256x256 | cupy_cuda | ok | 0.054 |  |  | 1.240918 | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| odd_255x257x263 | cupy_cuda | ok | 0.049 |  |  | 0.696786 | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_1024 | cupy_cuda | ok | 0.235 |  |  | 9.151624 | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| batched_2x512 | cupy_cuda | ok | 0.113 |  |  | 4.771677 | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| skinny_1024x128x512 | cupy_cuda | ok | 0.063 |  |  | 2.143231 | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| wide_128x1024x512 | cupy_cuda | ok | 0.062 |  |  | 2.174341 | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| small_k_1024x1024x64 | cupy_cuda | ok | 0.059 |  |  | 2.259862 | cupy 13.6.0, CUDA/cuBLAS, NVIDIA GeForce RTX 3070 |
| square_128 | jax | skipped |  |  |  |  | No module named 'jax' |
| square_256 | jax | skipped |  |  |  |  | No module named 'jax' |
| square_512 | jax | skipped |  |  |  |  | No module named 'jax' |
| batched_4x256 | jax | skipped |  |  |  |  | No module named 'jax' |
| tall_512x256x256 | jax | skipped |  |  |  |  | No module named 'jax' |
| odd_255x257x263 | jax | skipped |  |  |  |  | No module named 'jax' |
| square_1024 | jax | skipped |  |  |  |  | No module named 'jax' |
| batched_2x512 | jax | skipped |  |  |  |  | No module named 'jax' |
| skinny_1024x128x512 | jax | skipped |  |  |  |  | No module named 'jax' |
| wide_128x1024x512 | jax | skipped |  |  |  |  | No module named 'jax' |
| small_k_1024x1024x64 | jax | skipped |  |  |  |  | No module named 'jax' |
| square_128 | tensorflow | skipped |  |  |  |  | No module named 'tensorflow' |
| square_256 | tensorflow | skipped |  |  |  |  | No module named 'tensorflow' |
| square_512 | tensorflow | skipped |  |  |  |  | No module named 'tensorflow' |
| batched_4x256 | tensorflow | skipped |  |  |  |  | No module named 'tensorflow' |
| tall_512x256x256 | tensorflow | skipped |  |  |  |  | No module named 'tensorflow' |
| odd_255x257x263 | tensorflow | skipped |  |  |  |  | No module named 'tensorflow' |
| square_1024 | tensorflow | skipped |  |  |  |  | No module named 'tensorflow' |
| batched_2x512 | tensorflow | skipped |  |  |  |  | No module named 'tensorflow' |
| skinny_1024x128x512 | tensorflow | skipped |  |  |  |  | No module named 'tensorflow' |
| wide_128x1024x512 | tensorflow | skipped |  |  |  |  | No module named 'tensorflow' |
| small_k_1024x1024x64 | tensorflow | skipped |  |  |  |  | No module named 'tensorflow' |

## Transfer

| status | bytes | iters | upload GiB/s | download GiB/s | details |
| --- | ---: | ---: | ---: | ---: | --- |
| ok | 67108864 | 40 | 9.771 | 10.335 | NVIDIA GeForce RTX 3070 (discrete) |

## Analysis

- `tensor-ash` used `NVIDIA GeForce RTX 3070 (discrete)`, so the Vulkan measurements reflect real GPU kernel timings on this host.
- Actual GPU framework comparisons succeeded for: `cupy_cuda`, `torch_cuda`.
- Largest gap: `square_1024` is 1.2x faster in `torch_cuda` than `tensor-ash` in this environment.
- `tensor-ash` is the fastest measured backend on 7/11 benchmark cases.
- Throughput ratio versus `torch_cuda` across 11 shared cases: 0.84x to 1.74x, geometric mean 1.10x.
- Throughput ratio versus `cupy_cuda` across 11 shared cases: 0.98x to 6.66x, geometric mean 2.51x.
- Throughput ratio versus `numpy` across 11 shared cases: 5.25x to 68.11x, geometric mean 31.61x.
- Throughput ratio versus `torch_cpu` across 11 shared cases: 5.44x to 70.53x, geometric mean 31.91x.
- Median `tensor-ash` host/submission overhead was 0.019 ms per synchronous call; GPU timestamp TFLOPS excludes that overhead.
- Some libraries were skipped because their Python modules or device backends were unavailable: `jax`, `tensorflow`.
- JAX and TensorFlow rows are included when those modules are installed; skipped rows mean the local Python environment does not provide them.
- Transfer staging bandwidth measured 9.77 GiB/s upload and 10.33 GiB/s download for 67108864 bytes.

## Optimization Gameplan

1. Keep benchmarking on this discrete-GPU baseline and tune the shape-based shader selector with larger production-like matrix sizes.
2. Use `scripts/tune_kernels.py` before accepting changes to the automatic selector.
3. Focus the next shader pass on large square GEMMs, where PyTorch CUDA still has the largest lead.
4. Keep PyTorch CUDA and CuPy CUDA rows in regular benchmark runs so Vulkan changes are compared against cuBLAS-backed GPU compute.
5. Add CI checks for `cargo fmt`, `cargo clippy`, CPU tests, and an optional GPU correctness job.
