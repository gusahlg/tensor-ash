// f32 SIMT SGEMM with a multi-stage cp.async shared-memory pipeline —
// the primitive GLSL lacks (every source-level smem pipeline in GLSL
// lost; see benchmarks/NEXT.md dead ends).
//
//   C[M,N] = alpha * A[M,K] * B[K,N]   (+ beta * C when beta != 0)
//   row-major, batch = gridDim.z with dense per-batch strides.
//
// Tiling: CTA 128x128, BK=8, 256 threads (8 warps as 4(M) x 2(N)), warp
// tile 32x64 as 4x8 threads, thread tile 8x8.  A is kept [m][k] in
// shared memory (cp.async cannot transpose) with a 12-float row stride:
// a thread's 8 rows are interleaved (row = i*4 + ty), so one LDS.128
// of A (4 consecutive k of one row) touches 4 distinct rows whose
// 16-byte groups land in distinct banks (offsets 0/12/24/36 floats).
// B is [k][n], LDS.128 along n.  Per 4 k-steps a thread issues 8 A +
// 8 B LDS.128 for 256 FFMA.
//
// Variants: `_v4` needs K%4==0, N%4==0 and 16-byte aligned bases (16-
// byte cp.async + vector stores); `_s1` is fully general (4-byte
// cp.async, scalar stores).  Edges zero-fill through the cp.async
// src-size operand.
#include "common.cuh"

namespace {

constexpr int BM = 128, BN = 128, BK = 8;
constexpr int THREADS = 256;
constexpr int AS_STRIDE = BK + 4;  // floats per A row in smem (48 B)
constexpr int BS_STRIDE = BN;      // floats per B row in smem
constexpr int A_STAGE = BM * AS_STRIDE;
constexpr int B_STAGE = BK * BS_STRIDE;

template <int STAGES, bool VEC>
TA_DEVICE void load_tile(float* As, float* Bs, const float* __restrict__ A,
                         const float* __restrict__ B, int M, int N, int K, int m0, int n0, int k0,
                         int tid) {
    if (VEC) {
        {  // A: 128 rows x 2 chunks of 4 floats = 256 chunks, one per thread
            const int row = tid >> 1, ch = tid & 1;
            const int gm = m0 + row, gk = k0 + ch * 4;
            const bool ok = gm < M && gk < K;
            const float* src = ok ? A + (size_t)gm * K + gk : A;
            cp_async16(smem_u32(As + row * AS_STRIDE + ch * 4), src, ok ? 16u : 0u);
        }
        {  // B: 8 rows x 32 chunks = 256 chunks
            const int row = tid >> 5, ch = tid & 31;
            const int gk = k0 + row, gn = n0 + ch * 4;
            const bool ok = gk < K && gn < N;
            const float* src = ok ? B + (size_t)gk * N + gn : B;
            cp_async16(smem_u32(Bs + row * BS_STRIDE + ch * 4), src, ok ? 16u : 0u);
        }
    } else {
#pragma unroll
        for (int r = 0; r < 4; ++r) {  // A: 1024 floats, 4 per thread
            const int e = tid + r * THREADS;
            const int row = e >> 3, col = e & 7;
            const int gm = m0 + row, gk = k0 + col;
            const bool ok = gm < M && gk < K;
            const float* src = ok ? A + (size_t)gm * K + gk : A;
            cp_async4(smem_u32(As + row * AS_STRIDE + col), src, ok ? 4u : 0u);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {  // B: 1024 floats, 4 per thread
            const int e = tid + r * THREADS;
            const int row = e >> 7, col = e & 127;
            const int gk = k0 + row, gn = n0 + col;
            const bool ok = gk < K && gn < N;
            const float* src = ok ? B + (size_t)gk * N + gn : B;
            cp_async4(smem_u32(Bs + row * BS_STRIDE + col), src, ok ? 4u : 0u);
        }
    }
}

template <int STAGES, bool VEC>
TA_DEVICE void sgemm_body(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K, float alpha, float beta) {
    __shared__ __align__(16) float As[STAGES][A_STAGE];
    __shared__ __align__(16) float Bs[STAGES][B_STAGE];

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;  // 4 x 2 warps
    const int ty = lane >> 3, tx = lane & 7;  // 4 x 8 threads

    const size_t bz = blockIdx.z;
    A += bz * (size_t)M * K;
    B += bz * (size_t)K * N;
    C += bz * (size_t)M * N;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int KT = (K + BK - 1) / BK;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load_tile<STAGES, VEC>(As[s], Bs[s], A, B, M, N, K, m0, n0, s * BK, tid);
        cp_async_commit();
    }

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;

    const int a_row = wm * 32 + ty;       // + i*4
    const int b_col = wn * 64 + tx * 4;   // + j*32

    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nk = kt + STAGES - 1;
            if (nk < KT) {
                const int s = nk % STAGES;
                load_tile<STAGES, VEC>(As[s], Bs[s], A, B, M, N, K, m0, n0, nk * BK, tid);
            }
            cp_async_commit();
        }
        const float* as = As[kt % STAGES];
        const float* bs = Bs[kt % STAGES];
#pragma unroll
        for (int kk = 0; kk < BK; kk += 4) {
            float4 a[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                a[i] = *reinterpret_cast<const float4*>(as + (a_row + i * 4) * AS_STRIDE + kk);
#pragma unroll
            for (int k4 = 0; k4 < 4; ++k4) {
                const float4 b0 = *reinterpret_cast<const float4*>(bs + (kk + k4) * BS_STRIDE + b_col);
                const float4 b1 = *reinterpret_cast<const float4*>(bs + (kk + k4) * BS_STRIDE + b_col + 32);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float av = k4 == 0 ? a[i].x : k4 == 1 ? a[i].y : k4 == 2 ? a[i].z : a[i].w;
                    acc[i][0] = fmaf(av, b0.x, acc[i][0]);
                    acc[i][1] = fmaf(av, b0.y, acc[i][1]);
                    acc[i][2] = fmaf(av, b0.z, acc[i][2]);
                    acc[i][3] = fmaf(av, b0.w, acc[i][3]);
                    acc[i][4] = fmaf(av, b1.x, acc[i][4]);
                    acc[i][5] = fmaf(av, b1.y, acc[i][5]);
                    acc[i][6] = fmaf(av, b1.z, acc[i][6]);
                    acc[i][7] = fmaf(av, b1.w, acc[i][7]);
                }
            }
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int gm = m0 + a_row + i * 4;
        if (gm >= M) continue;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int gn = n0 + b_col + j * 32;
            float* dst = C + (size_t)gm * N + gn;
            float v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) v[q] = alpha * acc[i][j * 4 + q];
            if (VEC) {
                if (gn < N) {
                    if (beta != 0.f) {
                        const float4 old = *reinterpret_cast<const float4*>(dst);
                        v[0] += beta * old.x; v[1] += beta * old.y;
                        v[2] += beta * old.z; v[3] += beta * old.w;
                    }
                    *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
                }
            } else {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    if (gn + q < N) dst[q] = beta != 0.f ? v[q] + beta * dst[q] : v[q];
                }
            }
        }
    }
}

}  // namespace

extern "C" {

__global__ void __launch_bounds__(THREADS, 2)
ta_sgemm_128x128x8_s4_v4(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K, float alpha, float beta) {
    sgemm_body<4, true>(A, B, C, M, N, K, alpha, beta);
}

__global__ void __launch_bounds__(THREADS, 2)
ta_sgemm_128x128x8_s4_s1(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K, float alpha, float beta) {
    sgemm_body<4, false>(A, B, C, M, N, K, alpha, beta);
}

__global__ void __launch_bounds__(THREADS, 2)
ta_sgemm_128x128x8_s3_v4(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K, float alpha, float beta) {
    sgemm_body<3, true>(A, B, C, M, N, K, alpha, beta);
}

}  // extern "C"
