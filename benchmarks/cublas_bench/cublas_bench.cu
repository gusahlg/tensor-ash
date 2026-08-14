// cublas_bench.cu  -  pure cuBLAS FP32 SGEMM benchmark.
//
// Goal: a comparison baseline for `tensor-ash` that calls cuBLAS via
// its C API with as little wrapper overhead as possible — no PyTorch,
// no CuPy, no Python event loop.  Timing is via CUDA events
// (GPU-only, like the Vulkan timestamp queries in `ml_bench`), so the
// two numbers are directly comparable.
//
// FP32 is forced by `CUBLAS_PEDANTIC_MATH` so cuBLAS does NOT silently
// fall back to TF32 (the Ampere default for `CUBLAS_DEFAULT_MATH`),
// which would slash precision and inflate throughput by ~8x.
//
// Row-major convention
// --------------------
// `ml_bench` and the rest of `tensor-ash` use row-major tensors:
//
//     C[M, N]  =  alpha * A[M, K] @ B[K, N]
//
// cuBLAS is column-major, so we call SGEMM with arguments swapped:
//
//     C^T[N, M]  =  alpha * B^T[N, K] @ A^T[K, M]
//
// which in cuBLAS-speak is:
//
//     cublasSgemm(N, N,  N, M, K,  alpha,  B, ldb=N,  A, lda=K,  beta,  C, ldc=N)
//
// The same swap is used for `cublasSgemmStridedBatched`.
//
// Input
// -----
// Reads benchmark cases from stdin as CSV:
//
//     label,b,m,n,k
//     square_1024,1,1024,1024,1024
//     ...
//
// Output (stdout, CSV with header)
// --------------------------------
//
//     label,b,m,n,k,best_ms,mean_ms,tflops,sample_count,min_ms,median_ms,p95_ms
//
// Usage
// -----
//
//     ./cublas_bench --iters 30 --warmup 8 < cases.csv

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t _err = (call);                                         \
        if (_err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(_err));                             \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

#define CHECK_CUBLAS(call)                                                 \
    do {                                                                   \
        cublasStatus_t _status = (call);                                   \
        if (_status != CUBLAS_STATUS_SUCCESS) {                            \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__,          \
                    __LINE__, (int)_status);                               \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

struct BenchCase {
    std::string label;
    int b{};
    int m{};
    int n{};
    int k{};
};

struct BenchResult {
    double best_ms;
    double mean_ms;
    int sample_count;
    double median_ms;
    double p95_ms;
    double tflops;
};

static BenchResult bench_one(cublasHandle_t handle, const BenchCase& c,
                             int iters, int warmup) {
    const size_t a_count = static_cast<size_t>(c.b) * c.m * c.k;
    const size_t b_count = static_cast<size_t>(c.b) * c.k * c.n;
    const size_t c_count = static_cast<size_t>(c.b) * c.m * c.n;
    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, a_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, b_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, c_count * sizeof(float)));

    // Zero out so kernels run on deterministic data (the actual
    // values don't matter for GPU-time benchmarking).
    CHECK_CUDA(cudaMemset(dA, 0, a_count * sizeof(float)));
    CHECK_CUDA(cudaMemset(dB, 0, b_count * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC, 0, c_count * sizeof(float)));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major -> column-major swap.  See header comment.
    auto do_gemm = [&]() {
        if (c.b == 1) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     c.n, c.m, c.k,
                                     &alpha,
                                     dB, c.n,
                                     dA, c.k,
                                     &beta,
                                     dC, c.n));
        } else {
            CHECK_CUBLAS(cublasSgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                c.n, c.m, c.k,
                &alpha,
                dB, c.n, static_cast<long long>(c.k) * c.n,
                dA, c.k, static_cast<long long>(c.m) * c.k,
                &beta,
                dC, c.n, static_cast<long long>(c.m) * c.n,
                c.b));
        }
    };

    for (int i = 0; i < warmup; ++i) {
        do_gemm();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double best_ms = 1e18;
    double sum_ms = 0.0;
    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        do_gemm();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        const double ms_d = static_cast<double>(ms);
        if (ms_d < best_ms) best_ms = ms_d;
        sum_ms += ms_d;
        samples_ms.push_back(ms_d);
    }
    const double mean_ms = sum_ms / iters;
    std::sort(samples_ms.begin(), samples_ms.end());
    const size_t count = samples_ms.size();
    const double median_ms = count % 2 == 0
        ? (samples_ms[count / 2 - 1] + samples_ms[count / 2]) / 2.0
        : samples_ms[count / 2];
    const size_t p95_index = (count * 95 + 99) / 100 - 1;
    const double p95_ms = samples_ms[p95_index];

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    const double flops =
        2.0 * static_cast<double>(c.b) * c.m * c.n * c.k;
    const double tflops = flops / (median_ms * 1e-3) * 1e-12;
    return {best_ms, mean_ms, static_cast<int>(count), median_ms, p95_ms, tflops};
}

static bool parse_case(const std::string& line, BenchCase& out) {
    std::stringstream ss(line);
    std::string field;
    if (!std::getline(ss, out.label, ',')) return false;
    if (!std::getline(ss, field, ',')) return false;
    out.b = std::atoi(field.c_str());
    if (!std::getline(ss, field, ',')) return false;
    out.m = std::atoi(field.c_str());
    if (!std::getline(ss, field, ',')) return false;
    out.n = std::atoi(field.c_str());
    if (!std::getline(ss, field, ',')) return false;
    out.k = std::atoi(field.c_str());
    return out.b > 0 && out.m > 0 && out.n > 0 && out.k > 0;
}

int main(int argc, char** argv) {
    int iters = 30;
    int warmup = 8;
    bool allow_tf32 = false;
    bool device_info_only = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--allow-tf32") == 0) {
            allow_tf32 = true;
        } else if (std::strcmp(argv[i], "--device-info") == 0) {
            device_info_only = true;
        } else if (std::strcmp(argv[i], "--help") == 0 ||
                   std::strcmp(argv[i], "-h") == 0) {
            std::fprintf(stderr,
                "Usage: %s [--iters N] [--warmup N] [--allow-tf32] [--device-info]\n"
                "Reads cases on stdin (CSV: label,b,m,n,k) and writes "
                "results on stdout.\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            return 1;
        }
    }
    if (iters < 1 || warmup < 0) {
        std::fprintf(stderr, "--iters must be >= 1 and --warmup must be >= 0\n");
        return 1;
    }

    int device_id = 0;
    CHECK_CUDA(cudaSetDevice(device_id));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    if (device_info_only) {
        std::printf("device,sm_count,clock_mhz,major,minor,memory_mib\n");
        std::printf("%s,%d,%d,%d,%d,%zu\n",
                    prop.name, prop.multiProcessorCount,
                    prop.clockRate / 1000, prop.major, prop.minor,
                    prop.totalGlobalMem / (1024 * 1024));
        return 0;
    }

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle,
                                   allow_tf32 ? CUBLAS_DEFAULT_MATH
                                              : CUBLAS_PEDANTIC_MATH));

    std::printf(
        "label,b,m,n,k,best_ms,mean_ms,tflops,sample_count,min_ms,median_ms,p95_ms\n");
    std::fflush(stdout);

    std::string line;
    bool header_skipped = false;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (!header_skipped) {
            // Allow caller to either include or omit the header.  If the
            // first token doesn't parse as an integer for the b column,
            // assume it's a header and skip.
            std::stringstream ss(line);
            std::string label;
            std::string field;
            std::getline(ss, label, ',');
            std::getline(ss, field, ',');
            char* endp = nullptr;
            (void)std::strtol(field.c_str(), &endp, 10);
            if (endp == field.c_str()) {
                header_skipped = true;
                continue;
            }
            header_skipped = true;  // either way, consider header handled
        }
        BenchCase c{};
        if (!parse_case(line, c)) {
            std::fprintf(stderr, "skipping malformed line: %s\n",
                         line.c_str());
            continue;
        }
        const BenchResult r = bench_one(handle, c, iters, warmup);
        std::printf("%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f\n",
                    c.label.c_str(), c.b, c.m, c.n, c.k,
                    r.best_ms, r.mean_ms, r.tflops, r.sample_count,
                    r.best_ms, r.median_ms, r.p95_ms);
        std::fflush(stdout);
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}
