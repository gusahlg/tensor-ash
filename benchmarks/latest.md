# Benchmark Report

This report compares FP32 GEMM throughput for `tensor-ash` against the local framework backends available on this machine.

## Environment

```text
[2026-06-09T13:04:28Z INFO  tensor_ash::context] tensor-ash: using device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329)
[2026-06-09T13:04:28Z INFO  ml_bench::bench] device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496774464, compute_family=2, timestamps=true) slots=2
tensor-ash self-check
status: OK
selected: device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496774464, compute_family=2, timestamps=true)
executor_slots: 2
glslc: /nix/store/x90sdkmir007jaxx1s893iqgw5kxxc5w-shaderc-2026.1-bin/bin/glslc
vulkaninfo: /nix/store/pq7kd4gdh4ll73d83mlwhhzbjdl6fvl9-vulkan-tools-1.4.341.0/bin/vulkaninfo
LD_LIBRARY_PATH: /nix/store/d6vnmfmcz5b299180issiaad4m96wh8k-vulkan-loader-1.4.341.0/lib:/run/opengl-driver/lib:
VK_ICD_FILENAMES: <unset>
```

- Iterations: 5
- Warmup iterations: 2
- Case set: extended
- CPU library threads: 1

## Results

| case | library | status | best ms | TFLOPS | details |
| --- | --- | ---: | ---: | ---: | --- |
| square_128 | tensor-ash | ok | 0.011 | 0.382134 | NVIDIA GeForce RTX 3070 (discrete) |
| square_256 | tensor-ash | ok | 0.017 | 1.920469 | NVIDIA GeForce RTX 3070 (discrete) |
| square_512 | tensor-ash | ok | 0.052 | 5.171768 | NVIDIA GeForce RTX 3070 (discrete) |
| batched_4x256 | tensor-ash | ok | 0.028 | 4.877098 | NVIDIA GeForce RTX 3070 (discrete) |
| tall_512x256x256 | tensor-ash | ok | 0.018 | 3.731587 | NVIDIA GeForce RTX 3070 (discrete) |
| odd_255x257x263 | tensor-ash | ok | 0.019 | 1.780548 | NVIDIA GeForce RTX 3070 (discrete) |
| square_1024 | tensor-ash | ok | 0.273 | 7.858181 | NVIDIA GeForce RTX 3070 (discrete) |
| batched_2x512 | tensor-ash | ok | 0.074 | 7.206708 | NVIDIA GeForce RTX 3070 (discrete) |
| skinny_1024x128x512 | tensor-ash | ok | 0.032 | 4.165148 | NVIDIA GeForce RTX 3070 (discrete) |
| wide_128x1024x512 | tensor-ash | ok | 0.031 | 4.382763 | NVIDIA GeForce RTX 3070 (discrete) |
| small_k_1024x1024x64 | tensor-ash | ok | 0.025 | 5.295838 | NVIDIA GeForce RTX 3070 (discrete) |
| square_128 | numpy | ok | 0.040 | 0.105893 | numpy 2.4.4, threads=1 |
| square_256 | numpy | ok | 0.289 | 0.116061 | numpy 2.4.4, threads=1 |
| square_512 | numpy | ok | 2.248 | 0.119422 | numpy 2.4.4, threads=1 |
| batched_4x256 | numpy | ok | 1.156 | 0.116089 | numpy 2.4.4, threads=1 |
| tall_512x256x256 | numpy | ok | 0.577 | 0.116212 | numpy 2.4.4, threads=1 |
| odd_255x257x263 | numpy | ok | 0.311 | 0.110877 | numpy 2.4.4, threads=1 |
| square_1024 | numpy | ok | 17.953 | 0.119620 | numpy 2.4.4, threads=1 |
| batched_2x512 | numpy | ok | 4.544 | 0.118146 | numpy 2.4.4, threads=1 |
| skinny_1024x128x512 | numpy | ok | 1.172 | 0.114475 | numpy 2.4.4, threads=1 |
| wide_128x1024x512 | numpy | ok | 1.191 | 0.112677 | numpy 2.4.4, threads=1 |
| small_k_1024x1024x64 | numpy | ok | 1.212 | 0.110762 | numpy 2.4.4, threads=1 |
| square_128 | torch_cpu | ok | 0.043 | 0.098398 | torch 2.11.0, threads=1 |
| square_256 | torch_cpu | ok | 0.295 | 0.113932 | torch 2.11.0, threads=1 |
| square_512 | torch_cpu | ok | 2.269 | 0.118331 | torch 2.11.0, threads=1 |
| batched_4x256 | torch_cpu | ok | 1.169 | 0.114769 | torch 2.11.0, threads=1 |
| tall_512x256x256 | torch_cpu | ok | 0.573 | 0.117051 | torch 2.11.0, threads=1 |
| odd_255x257x263 | torch_cpu | ok | 0.320 | 0.107665 | torch 2.11.0, threads=1 |
| square_1024 | torch_cpu | ok | 18.263 | 0.117584 | torch 2.11.0, threads=1 |
| batched_2x512 | torch_cpu | ok | 4.585 | 0.117093 | torch 2.11.0, threads=1 |
| skinny_1024x128x512 | torch_cpu | ok | 1.171 | 0.114603 | torch 2.11.0, threads=1 |
| wide_128x1024x512 | torch_cpu | ok | 1.193 | 0.112514 | torch 2.11.0, threads=1 |
| small_k_1024x1024x64 | torch_cpu | ok | 1.282 | 0.104693 | torch 2.11.0, threads=1 |
| square_128 | torch_cuda | skipped |  |  | CUDA unavailable in this Python environment |
| square_256 | torch_cuda | skipped |  |  | CUDA unavailable in this Python environment |
| square_512 | torch_cuda | skipped |  |  | CUDA unavailable in this Python environment |
| batched_4x256 | torch_cuda | skipped |  |  | CUDA unavailable in this Python environment |
| tall_512x256x256 | torch_cuda | skipped |  |  | CUDA unavailable in this Python environment |
| odd_255x257x263 | torch_cuda | skipped |  |  | CUDA unavailable in this Python environment |
| square_1024 | torch_cuda | skipped |  |  | CUDA unavailable in this Python environment |
| batched_2x512 | torch_cuda | skipped |  |  | CUDA unavailable in this Python environment |
| skinny_1024x128x512 | torch_cuda | skipped |  |  | CUDA unavailable in this Python environment |
| wide_128x1024x512 | torch_cuda | skipped |  |  | CUDA unavailable in this Python environment |
| small_k_1024x1024x64 | torch_cuda | skipped |  |  | CUDA unavailable in this Python environment |
| square_128 | jax | skipped |  |  | No module named 'jax' |
| square_256 | jax | skipped |  |  | No module named 'jax' |
| square_512 | jax | skipped |  |  | No module named 'jax' |
| batched_4x256 | jax | skipped |  |  | No module named 'jax' |
| tall_512x256x256 | jax | skipped |  |  | No module named 'jax' |
| odd_255x257x263 | jax | skipped |  |  | No module named 'jax' |
| square_1024 | jax | skipped |  |  | No module named 'jax' |
| batched_2x512 | jax | skipped |  |  | No module named 'jax' |
| skinny_1024x128x512 | jax | skipped |  |  | No module named 'jax' |
| wide_128x1024x512 | jax | skipped |  |  | No module named 'jax' |
| small_k_1024x1024x64 | jax | skipped |  |  | No module named 'jax' |
| square_128 | tensorflow | skipped |  |  | No module named 'tensorflow' |
| square_256 | tensorflow | skipped |  |  | No module named 'tensorflow' |
| square_512 | tensorflow | skipped |  |  | No module named 'tensorflow' |
| batched_4x256 | tensorflow | skipped |  |  | No module named 'tensorflow' |
| tall_512x256x256 | tensorflow | skipped |  |  | No module named 'tensorflow' |
| odd_255x257x263 | tensorflow | skipped |  |  | No module named 'tensorflow' |
| square_1024 | tensorflow | skipped |  |  | No module named 'tensorflow' |
| batched_2x512 | tensorflow | skipped |  |  | No module named 'tensorflow' |
| skinny_1024x128x512 | tensorflow | skipped |  |  | No module named 'tensorflow' |
| wide_128x1024x512 | tensorflow | skipped |  |  | No module named 'tensorflow' |
| small_k_1024x1024x64 | tensorflow | skipped |  |  | No module named 'tensorflow' |

## Transfer

| status | bytes | iters | upload GiB/s | download GiB/s | details |
| --- | ---: | ---: | ---: | ---: | --- |
| ok | 67108864 | 5 | 9.608 | 10.180 | NVIDIA GeForce RTX 3070 (discrete) |

## Analysis

- `tensor-ash` used `NVIDIA GeForce RTX 3070 (discrete)`, so the Vulkan measurements reflect real GPU kernel timings on this host.
- `tensor-ash` is the fastest measured backend on 11/11 benchmark cases.
- Throughput ratio versus `numpy` across 11 shared cases: 3.6x to 65.7x, geometric mean 29.6x.
- Throughput ratio versus `torch_cpu` across 11 shared cases: 3.9x to 66.8x, geometric mean 30.2x.
- Some libraries were skipped because their Python modules were unavailable; see details in the table.
- PyTorch CUDA/cuBLAS was not available in this Python environment, so CUDA remains a separate benchmark to run when available.
- JAX and TensorFlow rows are included when those modules are installed; skipped rows mean the local Python environment does not provide them.
- Transfer staging bandwidth measured 9.61 GiB/s upload and 10.18 GiB/s download for 67108864 bytes.

## Optimization Gameplan

1. Keep benchmarking on this discrete-GPU baseline and tune the shape-based shader selector with more matrix sizes.
2. Use `ML_KERNEL=large|small` A/B runs to tune the automatic shader-selector thresholds.
3. Add more specialized kernels for skinny, wide, and small-K GEMMs.
4. Run the optional PyTorch CUDA/cuBLAS comparison when the local Python environment provides CUDA.
5. Add CI checks for `cargo fmt`, `cargo clippy`, CPU tests, and an optional GPU correctness job.
