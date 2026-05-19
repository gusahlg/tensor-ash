# Benchmark Report

This report compares FP32 GEMM throughput for `tensor-ash`, NumPy, and PyTorch on the current machine.

## Environment

```text
[2026-05-15T17:09:13Z INFO  tensor_ash::context] tensor-ash: using device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329)
[2026-05-15T17:09:13Z INFO  ml_bench] device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496561344, compute_family=2, timestamps=true) slots=2
tensor-ash self-check
status: OK
selected: device #0: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2488, driver=2496561344, compute_family=2, timestamps=true)
executor_slots: 2
glslc: /nix/store/gsgx9c110n8kq3q3ryhsq97mmxsrax5i-shaderc-2026.1-bin/bin/glslc
vulkaninfo: /nix/store/61857bc2nkcv1is2qwwfplfbgba0fkmz-vulkan-tools-1.4.341.0/bin/vulkaninfo
LD_LIBRARY_PATH: /nix/store/zs7y2aadk71bawprdcn000az9y05s8nf-vulkan-loader-1.4.341.0/lib:/run/opengl-driver/lib:
VK_ICD_FILENAMES: <unset>
```

- Iterations: 10
- Warmup iterations: 3
- Case set: extended
- CPU library threads: 1

## Results

| case | library | status | best ms | TFLOPS | details |
| --- | --- | ---: | ---: | ---: | --- |
| square_128 | tensor-ash | ok | 0.011 | 0.377729 | NVIDIA GeForce RTX 3070 (discrete) |
| square_256 | tensor-ash | ok | 0.019 | 1.768256 | NVIDIA GeForce RTX 3070 (discrete) |
| square_512 | tensor-ash | ok | 0.043 | 6.236883 | NVIDIA GeForce RTX 3070 (discrete) |
| batched_4x256 | tensor-ash | ok | 0.024 | 5.629938 | NVIDIA GeForce RTX 3070 (discrete) |
| tall_512x256x256 | tensor-ash | ok | 0.018 | 3.705216 | NVIDIA GeForce RTX 3070 (discrete) |
| odd_255x257x263 | tensor-ash | ok | 0.019 | 1.841421 | NVIDIA GeForce RTX 3070 (discrete) |
| square_1024 | tensor-ash | ok | 0.234 | 9.190477 | NVIDIA GeForce RTX 3070 (discrete) |
| batched_2x512 | tensor-ash | ok | 0.064 | 8.367689 | NVIDIA GeForce RTX 3070 (discrete) |
| skinny_1024x128x512 | tensor-ash | ok | 0.031 | 4.288654 | NVIDIA GeForce RTX 3070 (discrete) |
| wide_128x1024x512 | tensor-ash | ok | 0.031 | 4.310693 | NVIDIA GeForce RTX 3070 (discrete) |
| small_k_1024x1024x64 | tensor-ash | ok | 0.026 | 5.184554 | NVIDIA GeForce RTX 3070 (discrete) |
| square_128 | numpy | ok | 0.039 | 0.106293 | numpy 2.4.4, threads=1 |
| square_256 | numpy | ok | 0.289 | 0.115979 | numpy 2.4.4, threads=1 |
| square_512 | numpy | ok | 2.256 | 0.119008 | numpy 2.4.4, threads=1 |
| batched_4x256 | numpy | ok | 1.159 | 0.115845 | numpy 2.4.4, threads=1 |
| tall_512x256x256 | numpy | ok | 0.543 | 0.123659 | numpy 2.4.4, threads=1 |
| odd_255x257x263 | numpy | ok | 0.296 | 0.116411 | numpy 2.4.4, threads=1 |
| square_1024 | numpy | ok | 16.836 | 0.127555 | numpy 2.4.4, threads=1 |
| batched_2x512 | numpy | ok | 4.314 | 0.124458 | numpy 2.4.4, threads=1 |
| skinny_1024x128x512 | numpy | ok | 1.113 | 0.120586 | numpy 2.4.4, threads=1 |
| wide_128x1024x512 | numpy | ok | 1.145 | 0.117260 | numpy 2.4.4, threads=1 |
| small_k_1024x1024x64 | numpy | ok | 1.193 | 0.112478 | numpy 2.4.4, threads=1 |
| square_128 | torch_cpu | ok | 0.040 | 0.104965 | torch 2.11.0, threads=1 |
| square_256 | torch_cpu | ok | 0.279 | 0.120319 | torch 2.11.0, threads=1 |
| square_512 | torch_cpu | ok | 2.145 | 0.125151 | torch 2.11.0, threads=1 |
| batched_4x256 | torch_cpu | ok | 1.114 | 0.120497 | torch 2.11.0, threads=1 |
| tall_512x256x256 | torch_cpu | ok | 0.573 | 0.117029 | torch 2.11.0, threads=1 |
| odd_255x257x263 | torch_cpu | ok | 0.313 | 0.109975 | torch 2.11.0, threads=1 |
| square_1024 | torch_cpu | ok | 17.380 | 0.123562 | torch 2.11.0, threads=1 |
| batched_2x512 | torch_cpu | ok | 4.385 | 0.122426 | torch 2.11.0, threads=1 |
| skinny_1024x128x512 | torch_cpu | ok | 1.110 | 0.120886 | torch 2.11.0, threads=1 |
| wide_128x1024x512 | torch_cpu | ok | 1.132 | 0.118612 | torch 2.11.0, threads=1 |
| small_k_1024x1024x64 | torch_cpu | ok | 1.165 | 0.115195 | torch 2.11.0, threads=1 |
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

## Transfer

| status | bytes | iters | upload GiB/s | download GiB/s | details |
| --- | ---: | ---: | ---: | ---: | --- |
| ok | 67108864 | 10 | 9.686 | 10.231 | NVIDIA GeForce RTX 3070 (discrete) |

## Analysis

- `tensor-ash` used `NVIDIA GeForce RTX 3070 (discrete)`, so the Vulkan measurements reflect real GPU kernel timings on this host.
- `tensor-ash` is the fastest measured backend on 11/11 benchmark cases.
- Throughput ratio versus `numpy` across 11 shared cases: 3.6x to 72.1x, geometric mean 30.2x.
- Throughput ratio versus `torch_cpu` across 11 shared cases: 3.6x to 74.4x, geometric mean 30.2x.
- Some libraries were skipped because their Python modules were unavailable; see details in the table.
- PyTorch CUDA/cuBLAS was not available in this Python environment, so CUDA remains a separate benchmark to run when available.
- Transfer staging bandwidth measured 9.69 GiB/s upload and 10.23 GiB/s download for 67108864 bytes.

## Optimization Gameplan

1. Keep benchmarking on this discrete-GPU baseline and tune the shape-based shader selector with more matrix sizes.
2. Use `ML_KERNEL=large|small` A/B runs to tune the automatic shader-selector thresholds.
3. Add more specialized kernels for skinny, wide, and small-K GEMMs.
4. Run the optional PyTorch CUDA/cuBLAS comparison when the local Python environment provides CUDA.
5. Add CI checks for `cargo fmt`, `cargo clippy`, CPU tests, and an optional GPU correctness job.
