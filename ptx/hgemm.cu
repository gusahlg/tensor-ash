// f16 tensor-core GEMM: mma.sync.m16n8k16 + ldmatrix + multi-stage
// cp.async.  The Vulkan coopmat1 path cannot express the cp.async
// pipeline or ldmatrix.trans directly (coopmat register prefetch and LDS
// double-buffering were measured dead ends, NEXT.md).
//
//   C[M,N] = alpha * A[M,K] (f16) * B[K,N] (f16)   row-major
//   C is f16 (OUT=1, tensor-ash's a16 route / llama prefill contract)
//   or f32 (OUT=0).  batch = gridDim.z with dense per-batch strides.
//
// Requirements: K % 8 == 0 and N % 8 == 0 (16-byte rows for cp.async),
// 16-byte aligned bases.  M, N, K edges zero-fill via cp.async src-size.
//
// Shared layout (per stage), 16-byte chunks XOR-swizzled so every
// ldmatrix phase (8 rows x 16 B) hits 8 distinct bank groups:
//   A: BM rows x BK halves: BK=32 (4 chunks/row): chunk' = c ^ ((row>>1)&3)
//                           BK=64 (8 chunks/row): chunk' = c ^ (row & 7)
//   B: BK rows x BN halves    (BN/8 chunks):  chunk' = c ^ (row & 7)
// B fragments come from ldmatrix.x4.trans on the [k][n] tile (B is not
// pre-transposed; f16 weights stay in their [K,N] layout).
//
// ACC = 0: f32 accumulate (mma ...f32.f16.f16.f32)
// ACC = 1: f16 accumulate (mma ...f16.f16.f16.f16) — 2x the tensor
//          throughput on GeForce Ampere/Ada; what cuBLAS does under
//          CUBLAS_COMPUTE_16F.  Converted to f32 at the store.
// ACC = 2: "f16p" — f16-accumulate mma within one BK k-tile, promoted
//          into f32 registers after every k-tile: the f16-rate tensor
//          pipe with each f16 partial sum spanning only BK products.
#include <cuda_fp16.h>
#include "common.cuh"

namespace {

template <int BM, int BN, int WM, int WN, int STAGES, int ACC, int BK_ = 32>
struct Cfg {
    static constexpr int BK = BK_;
    static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int MI = WM / 16, NI = WN / 8;
    static constexpr int A_STAGE = BM * BK;  // halves
    static constexpr int B_STAGE = BK * BN;  // halves
    static constexpr int A_CHUNKS = BM * BK / 8, B_CHUNKS = BK * BN / 8;
    static constexpr int ACH = BK / 8;  // chunks per A row
    static constexpr int BCH = BN / 8;  // chunks per B row
    static_assert(BK == 32 || BK == 64, "A swizzle handles 4 or 8 chunks per row");
    static_assert(BN >= 64, "B swizzle needs >= 8 chunks per row");
    static_assert(A_CHUNKS % THREADS == 0 && B_CHUNKS % THREADS == 0, "chunk split");
    static_assert(NI % 2 == 0, "B ldmatrix.x4 covers two n8 tiles");
};

// A rows are BK halves = 4 (BK=32) or 8 (BK=64) chunks; the XOR keeps
// the 8 rows of one ldmatrix phase on 8 distinct 16-byte bank groups.
template <int BK>
TA_DEVICE int a_off(int row, int ch) {
    const int sw = BK == 32 ? ((row >> 1) & 3) : (row & 7);
    return row * BK + ((ch ^ sw) << 3);
}
template <int BCH>
TA_DEVICE int b_off(int row, int ch) {
    return row * (BCH * 8) + ((ch ^ (row & 7)) << 3);
}

template <class G>
TA_DEVICE void load_stage(__half* As, __half* Bs, const __half* __restrict__ A,
                          const __half* __restrict__ B, int M, int N, int K, int m0, int n0,
                          int k0, int tid) {
#pragma unroll
    for (int r = 0; r < G::A_CHUNKS / G::THREADS; ++r) {
        const int c = tid + r * G::THREADS;
        const int row = c / G::ACH, ch = c % G::ACH;
        const int gm = m0 + row, gk = k0 + ch * 8;
        const bool ok = gm < M && gk < K;
        const __half* src = ok ? A + (size_t)gm * K + gk : A;
        cp_async16(smem_u32(As + a_off<G::BK>(row, ch)), src, ok ? 16u : 0u);
    }
#pragma unroll
    for (int r = 0; r < G::B_CHUNKS / G::THREADS; ++r) {
        const int c = tid + r * G::THREADS;
        const int row = c / G::BCH, ch = c % G::BCH;
        const int gk = k0 + row, gn = n0 + ch * 8;
        const bool ok = gk < K && gn < N;
        const __half* src = ok ? B + (size_t)gk * N + gn : B;
        cp_async16(smem_u32(Bs + b_off<G::BCH>(row, ch)), src, ok ? 16u : 0u);
    }
}

template <int BM, int BN, int WM, int WN, int STAGES, int ACC, int OUT, int BK_>
TA_DEVICE void hgemm_body(const __half* __restrict__ A, const __half* __restrict__ B,
                          void* __restrict__ Cv, int M, int N, int K, float alpha) {
    using G = Cfg<BM, BN, WM, WN, STAGES, ACC, BK_>;
    constexpr int BK = G::BK;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* As = reinterpret_cast<__half*>(smem_raw);
    __half* Bs = As + STAGES * G::A_STAGE;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % G::WARPS_M, wn = warp / G::WARPS_M;

    const size_t bz = blockIdx.z;
    A += bz * (size_t)M * K;
    B += bz * (size_t)K * N;
    const size_t c_base = bz * (size_t)M * N;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int KT = (K + BK - 1) / BK;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT)
            load_stage<G>(As + s * G::A_STAGE, Bs + s * G::B_STAGE, A, B, M, N, K, m0, n0, s * BK,
                          tid);
        cp_async_commit();
    }

    // FA: f32 accumulators live (ACC 0, 2); HA: f16 accumulators live (ACC 1, 2).
    constexpr bool FA = ACC != 1, HA = ACC != 0;
    float accf[FA ? G::MI : 1][FA ? G::NI : 1][4];
    uint32_t acch[HA ? G::MI : 1][HA ? G::NI : 1][2];
    if (FA) {
#pragma unroll
        for (int i = 0; i < G::MI; ++i)
#pragma unroll
            for (int j = 0; j < G::NI; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) accf[FA ? i : 0][FA ? j : 0][q] = 0.f;
    }
    if (HA) {
#pragma unroll
        for (int i = 0; i < G::MI; ++i)
#pragma unroll
            for (int j = 0; j < G::NI; ++j) acch[HA ? i : 0][HA ? j : 0][0] = acch[HA ? i : 0][HA ? j : 0][1] = 0u;
    }

    // Per-lane ldmatrix row/chunk selectors.
    const int a_lrow = wm * WM + (lane & 15);          // + mi*16
    const int a_lch = lane >> 4;                        // + ks*2
    const int b_lrow = (lane & 7) + ((lane >> 3) & 1) * 8;  // + ks*16
    const int b_lch = wn * (WN / 8) + (lane >> 4);      // + nj*2

    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nk = kt + STAGES - 1;
            if (nk < KT) {
                const int s = nk % STAGES;
                load_stage<G>(As + s * G::A_STAGE, Bs + s * G::B_STAGE, A, B, M, N, K, m0, n0,
                              nk * BK, tid);
            }
            cp_async_commit();
        }
        const __half* as = As + (kt % STAGES) * G::A_STAGE;
        const __half* bs = Bs + (kt % STAGES) * G::B_STAGE;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            uint32_t af[G::MI][4];
            uint32_t bf[G::NI][2];
#pragma unroll
            for (int mi = 0; mi < G::MI; ++mi)
                ldmatrix_x4(af[mi], smem_u32(as + a_off<BK>(a_lrow + mi * 16, ks * 2 + a_lch)));
#pragma unroll
            for (int nj = 0; nj < G::NI / 2; ++nj)
                ldmatrix_x4_trans(bf[2 * nj][0], bf[2 * nj][1], bf[2 * nj + 1][0],
                                  bf[2 * nj + 1][1],
                                  smem_u32(bs + b_off<G::BCH>(ks * 16 + b_lrow, b_lch + nj * 2)));
#pragma unroll
            for (int mi = 0; mi < G::MI; ++mi)
#pragma unroll
                for (int ni = 0; ni < G::NI; ++ni) {
                    if (ACC == 0)
                        mma_f32(accf[FA ? mi : 0][FA ? ni : 0], af[mi], bf[ni]);
                    else
                        mma_f16(acch[HA ? mi : 0][HA ? ni : 0], af[mi], bf[ni]);
                }
        }
        if (ACC == 2) {
            // Promote: fold this k-tile's f16 partial sums (<= BK terms)
            // into the f32 accumulators and restart them from zero.
#pragma unroll
            for (int mi = 0; mi < G::MI; ++mi)
#pragma unroll
                for (int ni = 0; ni < G::NI; ++ni) {
                    uint32_t* d = acch[HA ? mi : 0][HA ? ni : 0];
                    float* f = accf[FA ? mi : 0][FA ? ni : 0];
                    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&d[0]));
                    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&d[1]));
                    f[0] += lo.x; f[1] += lo.y; f[2] += hi.x; f[3] += hi.y;
                    d[0] = d[1] = 0u;
                }
        }
    }
    cp_async_wait<0>();

    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mi = 0; mi < G::MI; ++mi) {
#pragma unroll
        for (int ni = 0; ni < G::NI; ++ni) {
            const int gn = n0 + wn * WN + ni * 8 + t * 2;
            float v[4];
            if (FA) {
#pragma unroll
                for (int q = 0; q < 4; ++q) v[q] = accf[FA ? mi : 0][FA ? ni : 0][q];
            } else {
                const uint32_t* d = acch[HA ? mi : 0][HA ? ni : 0];
                const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&d[0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&d[1]));
                v[0] = lo.x; v[1] = lo.y; v[2] = hi.x; v[3] = hi.y;
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int gm = m0 + wm * WM + mi * 16 + g + h * 8;
                if (gm < M && gn < N) {
                    const size_t idx = c_base + (size_t)gm * N + gn;
                    if (OUT == 0) {
                        *reinterpret_cast<float2*>(static_cast<float*>(Cv) + idx) =
                            make_float2(alpha * v[2 * h], alpha * v[2 * h + 1]);
                    } else {
                        *reinterpret_cast<__half2*>(static_cast<__half*>(Cv) + idx) =
                            __floats2half2_rn(alpha * v[2 * h], alpha * v[2 * h + 1]);
                    }
                }
            }
        }
    }
}

}  // namespace

#define TA_HGEMM_K(NAME, BM, BN, WM, WN, ST, ACC, OUT, MINB, BK)                              \
    extern "C" __global__ void __launch_bounds__(Cfg<BM, BN, WM, WN, ST, ACC, BK>::THREADS, MINB) \
        NAME(const __half* __restrict__ A, const __half* __restrict__ B, void* __restrict__ C,  \
             int M, int N, int K, float alpha) {                                               \
        hgemm_body<BM, BN, WM, WN, ST, ACC, OUT, BK>(A, B, C, M, N, K, alpha);                \
    }
#define TA_HGEMM(NAME, BM, BN, WM, WN, ST, ACC, OUT, MINB) \
    TA_HGEMM_K(NAME, BM, BN, WM, WN, ST, ACC, OUT, MINB, 32)

// name: ta_hgemm_<acc>_<BM>x<BN>[_w<WM>x<WN>][_k64]_s<stages>[_o32]; the
// default warp tile is 64x64 for BN >= 128, 64x32 for 128x64, 32x32
// for 64x64; BK = 32 unless _k64.  C is f16 unless _o32.
TA_HGEMM(ta_hgemm_f32_128x128_s3, 128, 128, 64, 64, 3, 0, 1, 2)
TA_HGEMM(ta_hgemm_f32_128x128_s4, 128, 128, 64, 64, 4, 0, 1, 1)
TA_HGEMM(ta_hgemm_f32_128x256_s3, 128, 256, 64, 64, 3, 0, 1, 1)
TA_HGEMM(ta_hgemm_f32_128x64_s3, 128, 64, 64, 32, 3, 0, 1, 2)
TA_HGEMM(ta_hgemm_f32_128x64_s4, 128, 64, 64, 32, 4, 0, 1, 2)
TA_HGEMM(ta_hgemm_f32_64x64_s4, 64, 64, 32, 32, 4, 0, 1, 3)
TA_HGEMM(ta_hgemm_f32_128x128_s3_o32, 128, 128, 64, 64, 3, 0, 0, 2)
TA_HGEMM(ta_hgemm_f32_128x128_w64x32_s3, 128, 128, 64, 32, 3, 0, 1, 2)
TA_HGEMM(ta_hgemm_f32_256x128_s3, 256, 128, 64, 64, 3, 0, 1, 1)
TA_HGEMM(ta_hgemm_f16_128x128_s3, 128, 128, 64, 64, 3, 1, 1, 2)
TA_HGEMM(ta_hgemm_f16_128x128_s4, 128, 128, 64, 64, 4, 1, 1, 1)
TA_HGEMM(ta_hgemm_f16_128x256_s3, 128, 256, 64, 64, 3, 1, 1, 1)
TA_HGEMM(ta_hgemm_f16_128x64_s3, 128, 64, 64, 32, 3, 1, 1, 2)
TA_HGEMM(ta_hgemm_f16_128x64_s4, 128, 64, 64, 32, 4, 1, 1, 2)
TA_HGEMM(ta_hgemm_f16_64x64_s4, 64, 64, 32, 32, 4, 1, 1, 3)
TA_HGEMM(ta_hgemm_f16_128x128_w64x32_s3, 128, 128, 64, 32, 3, 1, 1, 2)
TA_HGEMM(ta_hgemm_f16_128x128_w32x64_s3, 128, 128, 32, 64, 3, 1, 1, 2)
TA_HGEMM(ta_hgemm_f16_256x128_s3, 256, 128, 64, 64, 3, 1, 1, 1)
TA_HGEMM_K(ta_hgemm_f16_128x128_k64_s2, 128, 128, 64, 64, 2, 1, 1, 1, 64)
TA_HGEMM_K(ta_hgemm_f16_128x64_k64_s2, 128, 64, 64, 32, 2, 1, 1, 2, 64)
TA_HGEMM_K(ta_hgemm_f16_128x128_w64x32_k64_s2, 128, 128, 64, 32, 2, 1, 1, 1, 64)
TA_HGEMM(ta_hgemm_f16p_128x64_s3, 128, 64, 64, 32, 3, 2, 1, 2)
TA_HGEMM(ta_hgemm_f16p_128x128_w64x32_s3, 128, 128, 64, 32, 3, 2, 1, 1)
TA_HGEMM(ta_hgemm_f16p_64x64_s4, 64, 64, 32, 32, 4, 2, 1, 3)
TA_HGEMM_K(ta_hgemm_f16p_128x64_k64_s2, 128, 64, 64, 32, 2, 2, 1, 2, 64)
