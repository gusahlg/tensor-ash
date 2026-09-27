// Feasibility-probe kernels for VK_NV_cuda_kernel_launch.
//
// Compiled to PTX by ptx/build.sh (CUDA toolkit needed only there); the
// generated probe.ptx is checked in so tensor-ash builds without CUDA.
// All kernels take plain device pointers: the host passes Vulkan
// buffer-device-addresses (VkDeviceAddress) as the pointer values.
#include <stdint.h>

extern "C" {

// dst[i] = value
__global__ void probe_fill(uint32_t* __restrict__ dst, uint32_t n, uint32_t value) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = value;
}

// Forward-order affine map (block 0 handles the FIRST elements):
// dst[i] = src[i] * mul + add, wrapping u32 arithmetic.
// The Vulkan twin (shaders/ptx_probe_affine.comp) walks the buffer in
// REVERSE order, so a producer/consumer pair without a working barrier
// races on exactly the elements the producer writes last.
__global__ void probe_affine(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
                             uint32_t n, uint32_t mul, uint32_t add) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        dst[i] = src[i] * mul + add;
    }
}

// Reverse-order twin of probe_affine (block 0 handles the LAST
// elements), so a CUDA->CUDA chain can alternate traversal directions
// and still detect a missing dependency.
__global__ void probe_affine_rev(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
                                 uint32_t n, uint32_t mul, uint32_t add) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const uint32_t j = n - 1u - i;
        dst[j] = src[j] * mul + add;
    }
}

// Launch-overhead probe: a single thread bumps one counter.
__global__ void probe_bump(uint32_t* __restrict__ counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) counter[0] += 1;
}

// Dynamic shared-memory probe: fills `smem_bytes` of dynamic shared
// memory with a pattern, syncs, and writes a per-block checksum, so a
// launch with sharedMemBytes > 48 KiB either computes the right sum or
// visibly fails.
__global__ void probe_smem(uint32_t* __restrict__ out, uint32_t smem_words) {
    extern __shared__ uint32_t smem[];
    for (uint32_t i = threadIdx.x; i < smem_words; i += blockDim.x) smem[i] = i * 2654435761u + blockIdx.x;
    __syncthreads();
    uint32_t acc = 0;
    for (uint32_t i = threadIdx.x; i < smem_words; i += blockDim.x) acc += smem[smem_words - 1 - i];
    // reduce across the block through atomics (probe, not speed)
    atomicAdd(&out[blockIdx.x], acc);
}

// Clock probe: records %globaltimer (ns) at entry and exit of block 0 so
// the host can see whether consecutive launches overlap.
__global__ void probe_stamp(unsigned long long* __restrict__ out, uint32_t spin_iters) {
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    uint32_t x = threadIdx.x;
    for (uint32_t i = 0; i < spin_iters; ++i) x = x * 1664525u + 1013904223u;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = t0;
        out[1] = t1;
        out[2] = x;  // keep the spin alive
    }
}

}  // extern "C"
