// Persistent GEMV chain: one dispatch runs a sequence of M=1 f16-weight
// row GEMVs with in-kernel device-scope quorum barriers between
// dependent groups. Independent neighbours (gate || up) share one
// flattened tile space so they stay concurrent — the same overlap a
// barrier-free pair of vkCmdDispatch calls would get, without the
// ~7.7 µs compute-pipeline drain between groups.
//
// Workgroups are persistent (fixed grid, loop over tiles). Grid-sync
// is a sense-reversed pair of arrival counters plus a generation
// phase; `expected` is the in-shader group index so a fast workgroup
// cannot collide with a slow neighbour's still-open barrier.
//
// Inner-product order matches matmul_row_bda_kernel.glsl (KSLICES=16,
// fixed-order slice reduce) so chained results stay bit-exact with
// the standalone row kernels.

#pragma use_vulkan_memory_model

#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#ifndef KSLICES
#define KSLICES 16
#endif
const uint KSLICES_U = uint(KSLICES);

#define GEMV_CHAIN_MAX_JOBS 8u

layout(local_size_x = 32, local_size_y = KSLICES, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F32ReadOnly {
    float v[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) restrict buffer F32ReadWrite {
    float v[];
};
layout(buffer_reference, std430, buffer_reference_align = 2) restrict readonly buffer F16ReadOnly {
    float16_t v[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer F16Vec2ReadOnly {
    f16vec2 v[];
};

// 80-byte job: 16-byte header + 16-byte scalars + 5*8 pointers + 8 pad.
struct GemvJob {
    uint n;
    uint k;
    uint flags;
    uint vcols;
    float alpha;
    float beta;
    uint pad0;
    uint pad1;
    uint64_t a_ptr;
    uint64_t b_ptr;
    uint64_t c_ptr;
    uint64_t d_ptr;
    uint64_t bias_ptr;
    uint64_t pad2;
};

layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer JobTable {
    GemvJob jobs[];
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer SyncBuf {
    uint arrived[2];
    uint phase;
};

layout(push_constant) uniform PC {
    uint64_t jobs_ptr;
    uint64_t sync_ptr;
    uint n_jobs;
    uint n_wg;
} pc;

// flags: bit0 NORM_A, bits 8-9 EPI_BINARY (0/1 add / 2 mul),
// bits 16-17 EPI_ACT (0 none / 2 silu), bit 24 SYNC_AFTER.
const uint FLAG_NORM_A = 1u;
const uint FLAG_SYNC_AFTER = 1u << 24u;

shared float partial[KSLICES][32][2];

float silu(float x) {
    return x / (1.0 + exp(-x));
}

void grid_sync(uint expected) {
    barrier();
    if (gl_LocalInvocationIndex == 0u) {
        SyncBuf sync = SyncBuf(pc.sync_ptr);
        const uint slot = expected & 1u;
        const uint old = atomicAdd(
            sync.arrived[slot],
            1u,
            gl_ScopeDevice,
            gl_StorageSemanticsBuffer,
            gl_SemanticsAcquireRelease
        );
        if (old == pc.n_wg - 1u) {
            atomicExchange(
                sync.arrived[slot],
                0u,
                gl_ScopeDevice,
                gl_StorageSemanticsBuffer,
                gl_SemanticsRelease
            );
            atomicAdd(
                sync.phase,
                1u,
                gl_ScopeDevice,
                gl_StorageSemanticsBuffer,
                gl_SemanticsRelease
            );
        } else {
            while (atomicAdd(
                       sync.phase,
                       0u,
                       gl_ScopeDevice,
                       gl_StorageSemanticsBuffer,
                       gl_SemanticsAcquire
                   )
                   == expected) {}
        }
    }
    barrier();
}

float rms_scale(GemvJob job) {
    const uint lane = gl_LocalInvocationID.x;
    const uint slice = gl_LocalInvocationID.y;
    const uint tid = slice * 32u + lane;
    const uint wg = KSLICES_U * 32u;
    F32ReadOnly a = F32ReadOnly(job.a_ptr);
    float sumsq = 0.0;
    for (uint k = tid; k < job.k; k += wg) {
        const float av = a.v[k];
        sumsq = fma(av, av, sumsq);
    }
    partial[slice][lane][0] = sumsq;
    barrier();
    [[unroll]] for (uint step = wg >> 1u; step > 0u; step >>= 1u) {
        if (tid < step) {
            const uint other = tid + step;
            partial[slice][lane][0] += partial[other >> 5u][other & 31u][0];
        }
        barrier();
    }
    const float scale = inversesqrt(partial[0][0][0] / float(job.k) + job.beta);
    barrier();
    return scale;
}

void gemv_tile(GemvJob job, uint tile, float a_scale) {
    const uint vcols = job.vcols == 2u ? 2u : 1u;
    const uint lane = gl_LocalInvocationID.x;
    const uint slice = gl_LocalInvocationID.y;
    const uint col = tile * (32u * vcols) + lane * vcols;
    const bool live = col < job.n;
    const bool norm_a = (job.flags & FLAG_NORM_A) != 0u;
    const uint epi_bin = (job.flags >> 8u) & 3u;
    const uint epi_act = (job.flags >> 16u) & 3u;

    F32ReadOnly a = F32ReadOnly(job.a_ptr);
    F16ReadOnly b = F16ReadOnly(job.b_ptr);
    F32ReadWrite cout = F32ReadWrite(job.c_ptr);
    F32ReadOnly d = F32ReadOnly(job.d_ptr);
    F32ReadOnly bias = F32ReadOnly(job.bias_ptr);

    float acc0 = 0.0;
    float acc1 = 0.0;
    const bool vec_ok = vcols == 2u
        && (job.n % 2u) == 0u
        && (job.b_ptr & 3ul) == 0ul;
    if (vec_ok) {
        if (live) {
            F16Vec2ReadOnly bv = F16Vec2ReadOnly(job.b_ptr + uint64_t(col) * 2ul);
            const uint n_vec = job.n / 2u;
            if (norm_a) {
                for (uint inner = slice; inner < job.k; inner += KSLICES_U) {
                    const float av = a.v[inner] * a_scale * bias.v[inner];
                    const f16vec2 w = bv.v[inner * n_vec];
                    acc0 = fma(av, float(w.x), acc0);
                    acc1 = fma(av, float(w.y), acc1);
                }
            } else {
                for (uint inner = slice; inner < job.k; inner += KSLICES_U) {
                    const float av = a.v[inner];
                    const f16vec2 w = bv.v[inner * n_vec];
                    acc0 = fma(av, float(w.x), acc0);
                    acc1 = fma(av, float(w.y), acc1);
                }
            }
        }
    } else if (live) {
        if (norm_a) {
            for (uint inner = slice; inner < job.k; inner += KSLICES_U) {
                const float av = a.v[inner] * a_scale * bias.v[inner];
                acc0 = fma(av, float(b.v[inner * job.n + col]), acc0);
                if (vcols == 2u && col + 1u < job.n) {
                    acc1 = fma(av, float(b.v[inner * job.n + col + 1u]), acc1);
                }
            }
        } else {
            for (uint inner = slice; inner < job.k; inner += KSLICES_U) {
                const float av = a.v[inner];
                acc0 = fma(av, float(b.v[inner * job.n + col]), acc0);
                if (vcols == 2u && col + 1u < job.n) {
                    acc1 = fma(av, float(b.v[inner * job.n + col + 1u]), acc1);
                }
            }
        }
    }
    partial[slice][lane][0] = acc0;
    partial[slice][lane][1] = acc1;
    barrier();

    if (slice == 0u && live) {
        float sum0 = partial[0][lane][0];
        float sum1 = partial[0][lane][1];
        [[unroll]] for (uint s = 1u; s < KSLICES_U; ++s) {
            sum0 += partial[s][lane][0];
            sum1 += partial[s][lane][1];
        }
        const uint n_store = (vcols == 2u && col + 1u < job.n) ? 2u : 1u;
        [[unroll]] for (uint v = 0u; v < 2u; ++v) {
            if (v >= n_store) {
                break;
            }
            const uint gcol = col + v;
            float value = (v == 0u) ? sum0 : sum1;
            if (job.alpha != 1.0) {
                value *= job.alpha;
            }
            if (epi_act == 2u) {
                value = silu(value);
            }
            if (epi_bin == 1u) {
                value = fma(job.beta, d.v[gcol], value);
            } else if (epi_bin == 2u) {
                value *= d.v[gcol];
            }
            cout.v[gcol] = value;
        }
    }
    // The next tile (or the group-level grid_sync) needs a clean
    // shared-memory view; every path above hits this barrier.
    barrier();
}

uint job_tiles(GemvJob job) {
    const uint vcols = job.vcols == 2u ? 2u : 1u;
    return (job.n + 32u * vcols - 1u) / (32u * vcols);
}

void main() {
    if (gl_WorkGroupID.x >= pc.n_wg || pc.n_jobs == 0u || pc.n_jobs > GEMV_CHAIN_MAX_JOBS) {
        return;
    }
    JobTable table = JobTable(pc.jobs_ptr);
    const uint wg = gl_WorkGroupID.x;

    uint job = 0u;
    uint sync_id = 0u;
    while (job < pc.n_jobs) {
        uint group_end = job + 1u;
        while (group_end < pc.n_jobs && (table.jobs[group_end - 1u].flags & FLAG_SYNC_AFTER) == 0u) {
            group_end++;
        }

        uint total_tiles = 0u;
        for (uint j = job; j < group_end; ++j) {
            total_tiles += job_tiles(table.jobs[j]);
        }
        float scales[8];
        bool have_scale[8];
        [[unroll]] for (uint i = 0u; i < 8u; ++i) {
            scales[i] = 1.0;
            have_scale[i] = false;
        }
        for (uint t = wg; t < total_tiles; t += pc.n_wg) {
            uint acc = 0u;
            for (uint j = job; j < group_end; ++j) {
                const uint nt = job_tiles(table.jobs[j]);
                if (t < acc + nt) {
                    const uint ji = j;
                    if (!have_scale[ji] && (table.jobs[j].flags & FLAG_NORM_A) != 0u) {
                        scales[ji] = rms_scale(table.jobs[j]);
                    }
                    have_scale[ji] = true;
                    gemv_tile(table.jobs[j], t - acc, scales[ji]);
                    break;
                }
                acc += nt;
            }
        }

        if (group_end < pc.n_jobs) {
            grid_sync(sync_id);
            sync_id++;
        }
        job = group_end;
    }
}
