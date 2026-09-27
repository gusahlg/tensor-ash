device #0: NVIDIA GeForce RTX 4060 (discrete, Vulkan 1.4.329, vendor=0x10de, device=0x2882, driver=2496921600, compute_family=2, timestamps=true (64 bits), shader_cores=24 (VK_NV_shader_sm_builtins))
## part 1: every coopmat1 kernel, 256x256x1024 unif, full output vs f64
| kernel | acc | max abs err | max err / (2^-11 l2) | rel rms | mean err | verdict |
|---|---|---|---|---|---|---|
| f16w_coopmat_aligned | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_a16_coopmat_aligned | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_coopmat_m64n64 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_a16_coopmat_m64n64 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_coopmat_m64n128 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_a16_coopmat_m64n128 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_coopmat_m128n64 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_a16_coopmat_m128n64 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_coopmat_acc16 | f16 | 1.786e-1 | 33.19 (limit 64.0) | 1.664e-3 | +1.17e-4 | ok |
| f16w_coopmat_acc16p | f16-promote | 4.039e-2 | 7.83 (limit 22.6) | 6.036e-4 | +1.46e-6 | ok |
| f16w_coopmat_m64n64_acc16 | f16 | 1.786e-1 | 33.19 (limit 64.0) | 1.664e-3 | +1.17e-4 | ok |
| f16w_coopmat_m64n64_acc16p | f16-promote | 4.039e-2 | 7.83 (limit 22.6) | 6.036e-4 | +1.46e-6 | ok |
| f16w_coopmat_m64n128_acc16 | f16 | 1.786e-1 | 33.19 (limit 64.0) | 1.664e-3 | +1.17e-4 | ok |
| f16w_coopmat_m64n128_acc16p | f16-promote | 4.039e-2 | 7.83 (limit 22.6) | 6.036e-4 | +1.46e-6 | ok |
| f16w_coopmat_m128n64_acc16 | f16 | 1.786e-1 | 33.19 (limit 64.0) | 1.664e-3 | +1.17e-4 | ok |
| f16w_coopmat_m128n64_acc16p | f16-promote | 4.039e-2 | 7.83 (limit 22.6) | 6.036e-4 | +1.46e-6 | ok |
| f16w_a16_coopmat_acc16 | f16 | 1.786e-1 | 33.19 (limit 72.0) | 1.664e-3 | +1.17e-4 | ok |
| f16w_a16_coopmat_acc16p | f16-promote | 3.721e-2 | 7.21 (limit 30.6) | 6.374e-4 | +3.93e-6 | ok |
| f16w_a16_coopmat_m64n64_acc16 | f16 | 1.786e-1 | 33.19 (limit 72.0) | 1.664e-3 | +1.17e-4 | ok |
| f16w_a16_coopmat_m64n64_acc16p | f16-promote | 3.721e-2 | 7.21 (limit 30.6) | 6.374e-4 | +3.93e-6 | ok |
| f16w_a16_coopmat_m64n128_acc16 | f16 | 1.786e-1 | 33.19 (limit 72.0) | 1.664e-3 | +1.17e-4 | ok |
| f16w_a16_coopmat_m64n128_acc16p | f16-promote | 3.721e-2 | 7.21 (limit 30.6) | 6.374e-4 | +3.93e-6 | ok |
| f16w_a16_coopmat_m128n64_acc16 | f16 | 1.786e-1 | 33.19 (limit 72.0) | 1.664e-3 | +1.17e-4 | ok |
| f16w_a16_coopmat_m128n64_acc16p | f16-promote | 3.721e-2 | 7.21 (limit 30.6) | 6.374e-4 | +3.93e-6 | ok |
| f16w_coopmat_g_t1 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_t2 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_t3 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_t4 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_n1 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_n2 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_n3 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_n4 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_n5 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_n6 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_n7 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_n8 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_n9 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_n10 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_coopmat_g_n11 | f32 | 1.188e-4 | 0.02 (limit 1.0) | 1.732e-6 | -2.09e-7 | ok |
| f16w_a16_coopmat_g_t1 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_t2 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_t3 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_t4 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_n1 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_n2 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_n3 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_n4 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_n5 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_n6 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_n7 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_n8 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_n9 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_n10 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |
| f16w_a16_coopmat_g_n11 | f32 | 1.560e-2 | 3.13 (limit 9.0) | 2.070e-4 | -3.96e-6 | ok |

## part 2: prefill shapes, auto route, 64 sampled rows x all N vs f64
| dist | A | MxNxK | max|ref| | rms(ref) | acc | kernel | max abs err | max err / max|ref| | rel rms err | mean err | max err / (2^-11 l2) | nonfinite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unif | a16 | 512x2048x2048 | 65.862 | 15.0515 | f32 | f16w_a16_coopmat_m64n64 | 1.578e-2 | 2.396e-4 | 2.076e-4 | +2.96e-7 | 2.20 | 0 |
| unif | a16 | 512x2048x2048 | 65.862 | 15.0515 | f16 | f16w_a16_coopmat_m64n64_acc16 | 3.307e-1 | 5.020e-3 | 2.359e-3 | +4.67e-5 | 44.93 | 0 |
| unif | a16 | 512x2048x2048 | 65.862 | 15.0515 | f16-promote | f16w_a16_coopmat_m64n64_acc16p | 5.137e-2 | 7.799e-4 | 6.388e-4 | +2.31e-5 | 7.02 | 0 |
| unif | a32 | 512x2048x2048 | 65.862 | 15.0515 | f32 | f16w_coopmat_m64n64 | 3.218e-4 | 4.887e-6 | 3.388e-6 | +2.11e-7 | 0.04 | 0 |
| unif | a32 | 512x2048x2048 | 65.862 | 15.0515 | f16 | f16w_coopmat_m64n64_acc16 | 3.307e-1 | 5.020e-3 | 2.359e-3 | +4.67e-5 | 44.93 | 0 |
| unif | a32 | 512x2048x2048 | 65.862 | 15.0515 | f16-promote | f16w_coopmat_m64n64_acc16p | 5.625e-2 | 8.540e-4 | 6.039e-4 | +3.85e-5 | 7.69 | 0 |
| unif | a16 | 512x5632x2048 | 71.217 | 15.0711 | f32 | f16w_a16_coopmat_m128n64 | 2.961e-2 | 4.157e-4 | 2.072e-4 | +4.03e-7 | 4.03 | 0 |
| unif | a16 | 512x5632x2048 | 71.217 | 15.0711 | f16 | f16w_a16_coopmat_m128n64_acc16 | 3.368e-1 | 4.729e-3 | 2.346e-3 | +1.16e-4 | 45.68 | 0 |
| unif | a16 | 512x5632x2048 | 71.217 | 15.0711 | f16-promote | f16w_a16_coopmat_m128n64_acc16p | 5.316e-2 | 7.464e-4 | 6.387e-4 | +5.76e-6 | 7.24 | 0 |
| unif | a32 | 512x5632x2048 | 71.217 | 15.0711 | f32 | f16w_coopmat_m128n64 | 3.055e-4 | 4.290e-6 | 3.382e-6 | -9.66e-8 | 0.04 | 0 |
| unif | a32 | 512x5632x2048 | 71.217 | 15.0711 | f16 | f16w_coopmat_m128n64_acc16 | 3.368e-1 | 4.729e-3 | 2.346e-3 | +1.16e-4 | 45.68 | 0 |
| unif | a32 | 512x5632x2048 | 71.217 | 15.0711 | f16-promote | f16w_coopmat_m128n64_acc16p | 4.781e-2 | 6.714e-4 | 6.047e-4 | -2.67e-6 | 6.58 | 0 |
| unif | a16 | 512x2048x5632 | 118.865 | 25.0521 | f32 | f16w_a16_coopmat_m64n64 | 3.209e-2 | 2.700e-4 | 2.073e-4 | -2.14e-5 | 2.64 | 0 |
| unif | a16 | 512x2048x5632 | 118.865 | 25.0521 | f16 | f16w_a16_coopmat_m64n64_acc16 | 9.071e-1 | 7.632e-3 | 3.901e-3 | +2.95e-4 | 74.35 | 0 |
| unif | a16 | 512x2048x5632 | 118.865 | 25.0521 | f16-promote | f16w_a16_coopmat_m64n64_acc16p | 7.966e-2 | 6.702e-4 | 6.394e-4 | -3.91e-5 | 6.51 | 0 |
| unif | a32 | 512x2048x5632 | 118.865 | 25.0521 | f32 | f16w_coopmat_m64n64 | 1.362e-3 | 1.146e-5 | 9.087e-6 | -1.15e-7 | 0.11 | 0 |
| unif | a32 | 512x2048x5632 | 118.865 | 25.0521 | f16 | f16w_coopmat_m64n64_acc16 | 9.071e-1 | 7.632e-3 | 3.901e-3 | +2.95e-4 | 74.35 | 0 |
| unif | a32 | 512x2048x5632 | 118.865 | 25.0521 | f16-promote | f16w_coopmat_m64n64_acc16p | 7.385e-2 | 6.213e-4 | 6.054e-4 | -3.53e-7 | 5.84 | 0 |
| llm | a16 | 512x2048x2048 | 7.408 | 1.5097 | f32 | f16w_a16_coopmat_m64n64 | 1.963e-3 | 2.649e-4 | 2.068e-4 | +8.30e-7 | 3.37 | 0 |
| llm | a16 | 512x2048x2048 | 7.408 | 1.5097 | f16 | f16w_a16_coopmat_m64n64_acc16 | 3.238e-2 | 4.371e-3 | 1.978e-3 | +4.33e-6 | 36.72 | 0 |
| llm | a16 | 512x2048x2048 | 7.408 | 1.5097 | f16-promote | f16w_a16_coopmat_m64n64_acc16p | 7.759e-3 | 1.047e-3 | 5.894e-4 | -2.54e-6 | 6.18 | 0 |
| llm | a32 | 512x2048x2048 | 7.408 | 1.5097 | f32 | f16w_coopmat_m64n64 | 2.832e-5 | 3.823e-6 | 2.537e-6 | +1.86e-9 | 0.04 | 0 |
| llm | a32 | 512x2048x2048 | 7.408 | 1.5097 | f16 | f16w_coopmat_m64n64_acc16 | 3.238e-2 | 4.371e-3 | 1.978e-3 | +4.33e-6 | 36.72 | 0 |
| llm | a32 | 512x2048x2048 | 7.408 | 1.5097 | f16-promote | f16w_coopmat_m64n64_acc16p | 7.503e-3 | 1.013e-3 | 5.509e-4 | -2.87e-6 | 5.89 | 0 |
| llm | a16 | 512x5632x2048 | 8.543 | 1.5118 | f32 | f16w_a16_coopmat_m128n64 | 3.866e-3 | 4.525e-4 | 2.072e-4 | +2.85e-7 | 3.43 | 0 |
| llm | a16 | 512x5632x2048 | 8.543 | 1.5118 | f16 | f16w_a16_coopmat_m128n64_acc16 | 3.456e-2 | 4.046e-3 | 1.986e-3 | +7.71e-7 | 43.63 | 0 |
| llm | a16 | 512x5632x2048 | 8.543 | 1.5118 | f16-promote | f16w_a16_coopmat_m128n64_acc16p | 8.746e-3 | 1.024e-3 | 5.897e-4 | -1.62e-6 | 8.35 | 0 |
| llm | a32 | 512x5632x2048 | 8.543 | 1.5118 | f32 | f16w_coopmat_m128n64 | 2.711e-5 | 3.173e-6 | 2.550e-6 | -2.30e-9 | 0.04 | 0 |
| llm | a32 | 512x5632x2048 | 8.543 | 1.5118 | f16 | f16w_coopmat_m128n64_acc16 | 3.456e-2 | 4.046e-3 | 1.986e-3 | +7.71e-7 | 43.63 | 0 |
| llm | a32 | 512x5632x2048 | 8.543 | 1.5118 | f16-promote | f16w_coopmat_m128n64_acc16p | 7.958e-3 | 9.315e-4 | 5.522e-4 | -1.78e-6 | 7.60 | 0 |
| llm | a16 | 512x2048x5632 | 14.424 | 2.6226 | f32 | f16w_a16_coopmat_m64n64 | 3.949e-3 | 2.738e-4 | 2.072e-4 | +1.14e-6 | 3.30 | 0 |
| llm | a16 | 512x2048x5632 | 14.424 | 2.6226 | f16 | f16w_a16_coopmat_m64n64_acc16 | 1.063e-1 | 7.372e-3 | 3.996e-3 | -3.07e-5 | 74.74 | 0 |
| llm | a16 | 512x2048x5632 | 14.424 | 2.6226 | f16-promote | f16w_a16_coopmat_m64n64_acc16p | 9.919e-3 | 6.877e-4 | 6.168e-4 | +7.26e-6 | 6.61 | 0 |
| llm | a32 | 512x2048x5632 | 14.424 | 2.6226 | f32 | f16w_coopmat_m64n64 | 1.649e-4 | 1.143e-5 | 9.430e-6 | -3.48e-8 | 0.10 | 0 |
| llm | a32 | 512x2048x5632 | 14.424 | 2.6226 | f16 | f16w_coopmat_m64n64_acc16 | 1.063e-1 | 7.372e-3 | 3.996e-3 | -3.07e-5 | 74.74 | 0 |
| llm | a32 | 512x2048x5632 | 14.424 | 2.6226 | f16-promote | f16w_coopmat_m64n64_acc16p | 9.706e-3 | 6.729e-4 | 5.806e-4 | +6.59e-6 | 6.47 | 0 |
