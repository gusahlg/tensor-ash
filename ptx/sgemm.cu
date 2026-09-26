// f32 SIMT SGEMM with a multi-stage cp.async shared-memory pipeline —
// the primitive GLSL lacks (every source-level smem pipeline in GLSL
// lost; see benchmarks/NEXT.md dead ends).
//
//   C[M,N] = alpha * A[M,K] * B[K,N]   (+ beta * C when beta != 0)
//   row-major, batch = gridDim.z with dense per-batch strides.
//
// Tiling: CTA 128x128, BK=8 or 16, 256 threads (8 warps as 4(M) x 2(N)), warp
// tile 32x64 as 4x8 threads, thread tile 8x8.  A is kept [m][k] in
// shared memory (cp.async cannot transpose) with a (BK+4)-float row stride:
// a thread's 8 rows are interleaved (row = i*4 + ty), so one LDS.128
// of A (4 consecutive k of one row) touches 4 distinct rows whose
// 16-byte groups land in distinct banks (BK=8: offsets 0/12/24/36 floats;
// BK=16: 0/20/40/60).
// B is [k][n], LDS.128 along n.  Per 4 k-steps a thread issues 8 A +
// 8 B LDS.128 for 256 FFMA.
//
// Variants: `_v4` needs K%4==0, N%4==0 and 16-byte aligned bases (16-
// byte cp.async + vector stores); `_s1` is fully general (4-byte
// cp.async, scalar stores).  Edges zero-fill through the cp.async
// src-size operand.
#include "common.cuh"

namespace {

constexpr int BM = 128, BN = 128;
constexpr int THREADS = 256;

template <int BK>
struct SCfg {
    static constexpr int AS_STRIDE = BK + 4;  // floats per A row in smem
    static constexpr int BS_STRIDE = BN;      // floats per B row in smem
    static constexpr int A_STAGE = BM * AS_STRIDE;
    static constexpr int B_STAGE = BK * BS_STRIDE;
};

template <int BK, bool VEC>
TA_DEVICE void load_tile(float* As, float* Bs, const float* __restrict__ A,
                         const float* __restrict__ B, int M, int N, int K, int m0, int n0, int k0,
                         int tid) {
    using G = SCfg<BK>;
    if (VEC) {
        constexpr int ACH = BK / 4;  // 16-byte chunks per A row
#pragma unroll
        for (int r = 0; r < BM * ACH / THREADS; ++r) {
            const int c = tid + r * THREADS;
            const int row = c / ACH, ch = c % ACH;
            const int gm = m0 + row, gk = k0 + ch * 4;
            const bool ok = gm < M && gk < K;
            const float* src = ok ? A + (size_t)gm * K + gk : A;
            cp_async16(smem_u32(As + row * G::AS_STRIDE + ch * 4), src, ok ? 16u : 0u);
        }
#pragma unroll
        for (int r = 0; r < BK * 32 / THREADS; ++r) {  // B: BK rows x 32 chunks
            const int c = tid + r * THREADS;
            const int row = c >> 5, ch = c & 31;
            const int gk = k0 + row, gn = n0 + ch * 4;
            const bool ok = gk < K && gn < N;
            const float* src = ok ? B + (size_t)gk * N + gn : B;
            cp_async16(smem_u32(Bs + row * G::BS_STRIDE + ch * 4), src, ok ? 16u : 0u);
        }
    } else {
#pragma unroll
        for (int r = 0; r < BM * BK / THREADS; ++r) {
            const int e = tid + r * THREADS;
            const int row = e / BK, col = e % BK;
            const int gm = m0 + row, gk = k0 + col;
            const bool ok = gm < M && gk < K;
            const float* src = ok ? A + (size_t)gm * K + gk : A;
            cp_async4(smem_u32(As + row * G::AS_STRIDE + col), src, ok ? 4u : 0u);
        }
#pragma unroll
        for (int r = 0; r < BK * BN / THREADS; ++r) {
            const int e = tid + r * THREADS;
            const int row = e >> 7, col = e & 127;
            const int gk = k0 + row, gn = n0 + col;
            const bool ok = gk < K && gn < N;
            const float* src = ok ? B + (size_t)gk * N + gn : B;
            cp_async4(smem_u32(Bs + row * G::BS_STRIDE + col), src, ok ? 4u : 0u);
        }
    }
}

// Shared memory: STAGES * (A_STAGE + B_STAGE) floats — a static array
// when it fits the 48 KiB static limit (lets ptxas fold stage offsets:
// the dynamic form spilled 4 B at BK=8/4 stages), dynamic otherwise.
template <int BK, int STAGES>
struct SmemBytes {
    static constexpr int value = STAGES * (SCfg<BK>::A_STAGE + SCfg<BK>::B_STAGE) * 4;
};

template <int BK, int STAGES, bool VEC>
TA_DEVICE void sgemm_body(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K, float alpha, float beta) {
    using G = SCfg<BK>;
    constexpr bool kStatic = SmemBytes<BK, STAGES>::value <= 48 * 1024;
    __shared__ __align__(16) float smem_s[kStatic ? SmemBytes<BK, STAGES>::value / 4 : 4];
    extern __shared__ __align__(16) float smem_f[];
    float* As = kStatic ? smem_s : smem_f;
    float* Bs = As + STAGES * G::A_STAGE;

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
        if (s < KT)
            load_tile<BK, VEC>(As + s * G::A_STAGE, Bs + s * G::B_STAGE, A, B, M, N, K, m0, n0,
                               s * BK, tid);
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
                load_tile<BK, VEC>(As + s * G::A_STAGE, Bs + s * G::B_STAGE, A, B, M, N, K, m0,
                                   n0, nk * BK, tid);
            }
            cp_async_commit();
        }
        const float* as = As + (kt % STAGES) * G::A_STAGE;
        const float* bs = Bs + (kt % STAGES) * G::B_STAGE;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 4) {
            float4 a[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                a[i] = *reinterpret_cast<const float4*>(as + (a_row + i * 4) * G::AS_STRIDE + kk);
#pragma unroll
            for (int k4 = 0; k4 < 4; ++k4) {
                const float4 b0 =
                    *reinterpret_cast<const float4*>(bs + (kk + k4) * G::BS_STRIDE + b_col);
                const float4 b1 =
                    *reinterpret_cast<const float4*>(bs + (kk + k4) * G::BS_STRIDE + b_col + 32);
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

#define TA_SGEMM(NAME, BK, ST, VEC)                                                          \
    extern "C" __global__ void __launch_bounds__(THREADS, 2)                                    \
        NAME(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,   \
             int M, int N, int K, float alpha, float beta) {                                    \
        sgemm_body<BK, ST, VEC>(A, B, C, M, N, K, alpha, beta);                                \
    }

// name: ta_sgemm_128x128x<BK>_s<stages>_<v4|s1>; shared memory =
// stages * (128*(BK+4) + BK*128) * 4 bytes (dynamic above 48 KiB).
TA_SGEMM(ta_sgemm_128x128x8_s4_v4, 8, 4, true)
TA_SGEMM(ta_sgemm_128x128x8_s3_v4, 8, 3, true)
TA_SGEMM(ta_sgemm_128x128x8_s4_s1, 8, 4, false)
TA_SGEMM(ta_sgemm_128x128x16_s3_v4, 16, 3, true)
TA_SGEMM(ta_sgemm_128x128x16_s2_v4, 16, 2, true)
