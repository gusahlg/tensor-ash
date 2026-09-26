// Shared PTX helpers for the tensor-ash PTX kernels (sm_80+).
#pragma once
#include <stdint.h>

#define TA_DEVICE __device__ __forceinline__

TA_DEVICE uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte async global->shared copy, L1-bypassing (.cg).  src_bytes in
// {0, 16}: 0 zero-fills the destination without touching global memory.
TA_DEVICE void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
                 "r"(src_bytes));
}

// 4-byte async copy (.ca is the only legal cache op for 4 bytes).
TA_DEVICE void cp_async4(uint32_t dst, const void* src, uint32_t src_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src),
                 "r"(src_bytes));
}

TA_DEVICE void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
TA_DEVICE void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

TA_DEVICE void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

TA_DEVICE void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                 uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// D = A*B + D, f16 x f16 -> f32, m16n8k16.
TA_DEVICE void mma_f32(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// D = A*B + D, f16 x f16 -> f16 (f16 accumulate), m16n8k16.
TA_DEVICE void mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
