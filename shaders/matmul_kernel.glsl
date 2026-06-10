// =====================================================================
//  Shared GEMM (f32) kernel body, parameterized by tile constants.
//
//  The two wrapper shaders (matmul_f32.comp, matmul_f32_small.comp) set
//  BM / BN / BK / TM / TN via preprocessor #defines and then #include
//  this file.  Keeping the kernel logic in a single place stops the two
//  variants from drifting apart.
//
//      C[b]  =  alpha * A[b] @ B[b]   (+ C[b]   if accumulate)
//
//  Tile layout
//  -----------
//      BM x BN  output tile per workgroup
//      BK       K-strip width
//      TM x TN  register tile per thread
//      WG       (BN/TN) x (BM/TM) threads
//
//  Optimization: workgroup-uniform predicates `m_full`, `n_full` and a
//  separate K-tail loop hoist all bounds checks out of the inner loads
//  for interior tiles (i.e. the overwhelming majority of work on
//  AI-sized problems).  An edge tile pays one extra branch per
//  cooperative load.
//
//  Shared-memory layout
//  --------------------
//      As[row][k]   +1 col pad to break bank conflicts on the column-
//                   strided inner read `As[a_row0+i][k]`.
//      Bs[k][col]   row-major; inner read `Bs[k][b_col0+j]` is contiguous.
// =====================================================================

#extension GL_EXT_control_flow_attributes : require
#extension GL_GOOGLE_include_directive   : require

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// ------------------------------------------------------------------
// Specialization constants.  These are set at pipeline-create time so
// the SPIR-V compiler folds the dependent branches before generating
// ISA — eliminating the per-store ternary in the epilogue and (when
// INTERIOR_ONLY is set) the entire bounds-checked slow path.
//
//   ACCUMULATE     true  -> C = alpha*A@B + C   (read-modify-write)
//                  false -> C = alpha*A@B       (pure store)
//   ALPHA_IS_ONE   true  -> skip the `alpha * x` multiply entirely
//   INTERIOR_ONLY  true  -> host guarantees M and N are tile-aligned;
//                           drop all m_full/n_full predicates
//   K_MULTIPLE     true  -> host guarantees K is a multiple of BK;
//                           drop the K-tail branch and modulo
// ------------------------------------------------------------------
layout(constant_id = 0) const bool ACCUMULATE    = false;
layout(constant_id = 1) const bool ALPHA_IS_ONE  = true;
layout(constant_id = 2) const bool INTERIOR_ONLY = false;
layout(constant_id = 3) const bool K_MULTIPLE    = false;

const uint WG_X    = BN / TN;
const uint WG_Y    = BM / TM;
const uint THREADS = WG_X * WG_Y;
const uint A_TILE  = BM * BK;
const uint B_TILE  = BK * BN;
const uint A_PER_T = A_TILE / THREADS;
const uint B_PER_T = B_TILE / THREADS;

layout(set = 0, binding = 0, std430) readonly  buffer A_BUF { float a[]; };
layout(set = 0, binding = 1, std430) readonly  buffer B_BUF { float b[]; };
layout(set = 0, binding = 2, std430)           buffer C_BUF { float c[]; };

layout(push_constant) uniform PC {
    uint  M;
    uint  N;
    uint  K;
    uint  batch_stride_a;
    uint  batch_stride_b;
    uint  batch_stride_c;
    uint  flags;       // bit 0: accumulate (C = alpha*A@B + C)
    float alpha;
} pc;

shared float As[BM][BK + 1u];
shared float Bs[BK][BN];

// ------------------------------------------------------------------
//  Cooperative loaders.  `K_FULL` lets the caller signal that this
//  K-strip has no partial column on the right edge, so we can skip
//  the `g_col < pc.K` (for A) / `g_row < pc.K` (for B) test.
//  `M_FULL` / `N_FULL` similarly hoist the row/col-edge tests.
// ------------------------------------------------------------------
void load_a_tile(uint a_base, uint block_row, uint k_base,
                 uint tid, bool m_full, bool k_full) {
    [[unroll]] for (uint i = 0u; i < A_PER_T; ++i) {
        const uint idx   = tid + i * THREADS;
        const uint row   = idx / BK;
        const uint col   = idx % BK;
        const uint g_row = block_row * BM + row;
        const uint g_col = k_base + col;
        // When INTERIOR_ONLY, the host promises this tile is fully inside
        // M and K, so the read is unconditional.  The compiler folds both
        // predicates away.
        float v = 0.0;
        if ((INTERIOR_ONLY || m_full || g_row < pc.M)
            && (INTERIOR_ONLY || k_full || g_col < pc.K)) {
            v = a[a_base + g_row * pc.K + g_col];
        }
        As[row][col] = v;
    }
}

void load_b_tile(uint b_base, uint block_col, uint k_base,
                 uint tid, bool n_full, bool k_full) {
    [[unroll]] for (uint i = 0u; i < B_PER_T; ++i) {
        const uint idx   = tid + i * THREADS;
        const uint row   = idx / BN;
        const uint col   = idx % BN;
        const uint g_row = k_base + row;
        const uint g_col = block_col * BN + col;
        float v = 0.0;
        if ((INTERIOR_ONLY || k_full || g_row < pc.K)
            && (INTERIOR_ONLY || n_full || g_col < pc.N)) {
            v = b[b_base + g_row * pc.N + g_col];
        }
        Bs[row][col] = v;
    }
}

void main() {
    const uint batch     = gl_WorkGroupID.z;
    const uint block_row = gl_WorkGroupID.y;
    const uint block_col = gl_WorkGroupID.x;

    const uint tid = gl_LocalInvocationIndex;
    const uint tx  = tid % WG_X;
    const uint ty  = tid / WG_X;

    const uint a_base = batch * pc.batch_stride_a;
    const uint b_base = batch * pc.batch_stride_b;
    const uint c_base = batch * pc.batch_stride_c;

    // Workgroup-uniform predicates.  GPUs handle these branches "for free":
    // every thread in the WG takes the same path.
    const bool m_full = ((block_row + 1u) * BM) <= pc.M;
    const bool n_full = ((block_col + 1u) * BN) <= pc.N;

    const uint num_full_k = pc.K / BK;
    const bool has_k_tail = !K_MULTIPLE && ((pc.K % BK) != 0u);

    const uint a_row0 = ty * TM;
    const uint b_col0 = tx * TN;

    float acc[TM][TN];
    [[unroll]] for (uint i = 0u; i < TM; ++i)
        [[unroll]] for (uint j = 0u; j < TN; ++j)
            acc[i][j] = 0.0;

    // ---- Main K loop: full BK strips, no K bounds check ----
    for (uint kt = 0u; kt < num_full_k; ++kt) {
        const uint k_base = kt * BK;
        load_a_tile(a_base, block_row, k_base, tid, m_full, true);
        load_b_tile(b_base, block_col, k_base, tid, n_full, true);

        barrier();

        [[unroll]] for (uint k = 0u; k < BK; ++k) {
            float a_reg[TM];
            float b_reg[TN];
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                a_reg[i] = As[a_row0 + i][k];
            [[unroll]] for (uint j = 0u; j < TN; ++j)
                b_reg[j] = Bs[k][b_col0 + j];
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j = 0u; j < TN; ++j)
                    acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
        }

        barrier();
    }

    // ---- K-tail loop (at most one extra strip, partial K columns) ----
    if (has_k_tail) {
        const uint k_base = num_full_k * BK;
        load_a_tile(a_base, block_row, k_base, tid, m_full, false);
        load_b_tile(b_base, block_col, k_base, tid, n_full, false);

        barrier();

        [[unroll]] for (uint k = 0u; k < BK; ++k) {
            float a_reg[TM];
            float b_reg[TN];
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                a_reg[i] = As[a_row0 + i][k];
            [[unroll]] for (uint j = 0u; j < TN; ++j)
                b_reg[j] = Bs[k][b_col0 + j];
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j = 0u; j < TN; ++j)
                    acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
        }

        barrier();
    }

    // ---- Epilogue: alpha-scale (+ accumulate) + store. ----
    // ACCUMULATE and ALPHA_IS_ONE are specialization constants, so each of
    // these branches collapses to a pure store / pure FMA at SPIR-V compile
    // time.  INTERIOR_ONLY collapses the entire bounds-check slow path.
    const float alpha = ALPHA_IS_ONE ? 1.0 : pc.alpha;

    const uint row_base = block_row * BM + a_row0;
    const uint col_base = block_col * BN + b_col0;

    if (INTERIOR_ONLY || (m_full && n_full)) {
        [[unroll]] for (uint i = 0u; i < TM; ++i) {
            const uint row_off = c_base + (row_base + i) * pc.N + col_base;
            [[unroll]] for (uint j = 0u; j < TN; ++j) {
                const float v = ALPHA_IS_ONE ? acc[i][j] : alpha * acc[i][j];
                if (ACCUMULATE) {
                    c[row_off + j] = c[row_off + j] + v;
                } else {
                    c[row_off + j] = v;
                }
            }
        }
    } else {
        [[unroll]] for (uint i = 0u; i < TM; ++i) {
            const uint g_row = row_base + i;
            if (g_row >= pc.M) continue;
            const uint row_off = c_base + g_row * pc.N;
            [[unroll]] for (uint j = 0u; j < TN; ++j) {
                const uint g_col = col_base + j;
                if (g_col >= pc.N) continue;
                const float v = ALPHA_IS_ONE ? acc[i][j] : alpha * acc[i][j];
                if (ACCUMULATE) {
                    c[row_off + g_col] = c[row_off + g_col] + v;
                } else {
                    c[row_off + g_col] = v;
                }
            }
        }
    }
}
