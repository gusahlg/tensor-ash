// cublas_hgemm_bench.cu — cuBLAS f16 GEMM reference for the PTX track.
//
// Mirrors what llama.cpp b9503's CUDA backend does for f16-weight
// GEMMs with more than 16 tokens (ggml_cuda_op_mul_mat_cublas on
// Ampere/Ada): cublasGemmEx with f16 A/B and CUBLAS_COMPUTE_16F (f16
// accumulate, f16 C, then a separate f16->f32 convert).  Also measures
// CUBLAS_COMPUTE_32F with f16 C and with f32 C for comparison.
//
// Row-major C[M,N] = A[M,K] * B[K,N]:
//   NN: B stored [K][N] (tensor-ash's layout)
//       -> column-major C^T = B^T * A^T: GemmEx(N, N, n, m, k, B, ldb=n, A, lda=k)
//   TN: weights stored [N][K] (ggml's layout: src0 = W with ne00 = K)
//       -> GemmEx(T, N, n, m, k, W, lda=k, A, ldb=k)   (exactly llama.cpp's call)
//
// Usage: cublas_hgemm_bench [--iters N] [--warmup N] MxNxK ...
// Output CSV: layout,compute,c_type,m,n,k,median_ms,min_ms,tflops
//
// Build (desktop, CUDA 12.9):
//   nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//        -gencode arch=compute_89,code=sm_89 -lcublas -o cublas_hgemm_bench cublas_hgemm_bench.cu
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                        \
    do {                                                                             \
        cudaError_t e = (x);                                                         \
        if (e != cudaSuccess) {                                                      \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)
#define CB(x)                                                               \
    do {                                                                    \
        cublasStatus_t s = (x);                                             \
        if (s != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "%s:%d cublas status %d\n", __FILE__, __LINE__, (int)s); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

__global__ void fill_half(__half* p, size_t n, uint32_t seed, float scale) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = __float2half(((x & 0xffffff) / 16777216.0f * 2.f - 1.f) * scale);
    }
}

int main(int argc, char** argv) {
    int iters = 20, warmup = 5;
    std::vector<std::string> shapes;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
        else shapes.push_back(argv[i]);
    }
    if (shapes.empty()) shapes = {"512x2048x2048", "512x5632x2048", "512x2048x5632", "2048x2048x2048", "4096x4096x4096"};

    cublasHandle_t h;
    CB(cublasCreate(&h));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    printf("layout,compute,c_type,m,n,k,median_ms,min_ms,tflops\n");

    // ~300 ms burn-in on the largest shape's GEMM happens implicitly via
    // warmup below; add an explicit clock burn first.
    bool burned = false;
    for (auto& s : shapes) {
        int m, n, k;
        if (sscanf(s.c_str(), "%dx%dx%d", &m, &n, &k) != 3) { fprintf(stderr, "bad shape %s\n", s.c_str()); return 1; }
        __half *A, *B, *C16;
        float* C32;
        CK(cudaMalloc(&A, (size_t)m * k * 2));
        CK(cudaMalloc(&B, (size_t)k * n * 2));
        CK(cudaMalloc(&C16, (size_t)m * n * 2));
        CK(cudaMalloc(&C32, (size_t)m * n * 4));
        fill_half<<<(unsigned)(((size_t)m * k + 255) / 256), 256>>>(A, (size_t)m * k, 1, 1.0f);
        fill_half<<<(unsigned)(((size_t)k * n + 255) / 256), 256>>>(B, (size_t)k * n, 2, 0.035f);
        CK(cudaDeviceSynchronize());

        const __half ha = __float2half(1.f), hb = __float2half(0.f);
        const float fa = 1.f, fb = 0.f;
        struct Mode { const char* layout; const char* compute; const char* ctype; };
        const Mode modes[] = {
            {"TN", "16F", "f16"}, {"TN", "32F", "f16"}, {"TN", "32F", "f32"},
            {"NN", "16F", "f16"}, {"NN", "32F", "f16"}, {"NN", "32F", "f32"},
        };
        for (const Mode& md : modes) {
            const bool tn = md.layout[0] == 'T';
            const bool c16f = !strcmp(md.compute, "16F");
            const bool out16 = !strcmp(md.ctype, "f16");
            auto run = [&]() {
                CB(cublasGemmEx(h, tn ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                c16f ? (const void*)&ha : (const void*)&fa,
                                B, CUDA_R_16F, tn ? k : n, A, CUDA_R_16F, k,
                                c16f ? (const void*)&hb : (const void*)&fb,
                                out16 ? (void*)C16 : (void*)C32, out16 ? CUDA_R_16F : CUDA_R_32F, n,
                                c16f ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            };
            if (!burned) {
                auto t0 = std::chrono::steady_clock::now();
                while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(300)) {
                    run();
                    CK(cudaDeviceSynchronize());
                }
                burned = true;
            }
            for (int i = 0; i < warmup; ++i) run();
            CK(cudaDeviceSynchronize());
            std::vector<float> ts;
            for (int i = 0; i < iters; ++i) {
                CK(cudaEventRecord(e0));
                run();
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                ts.push_back(ms);
            }
            std::sort(ts.begin(), ts.end());
            const double med = ts[ts.size() / 2];
            printf("%s,%s,%s,%d,%d,%d,%.4f,%.4f,%.2f\n", md.layout, md.compute, md.ctype, m, n, k, med,
                   ts[0], 2.0 * m * n * (double)k / med * 1e-9);
            fflush(stdout);
        }
        cudaFree(A); cudaFree(B); cudaFree(C16); cudaFree(C32);
    }
    cublasDestroy(h);
    return 0;
}
