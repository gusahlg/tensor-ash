#ifndef TENSOR_ASH_H
#define TENSOR_ASH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ta_context ta_context;
typedef struct ta_executor ta_executor;
typedef struct ta_tensor ta_tensor;

typedef struct ta_run_stats {
  uint32_t has_gpu_time;
  uint64_t gpu_time_ns;
  size_t n_calls;
  uint64_t total_flops;
  double tflops;
} ta_run_stats;

typedef struct ta_matmul_call {
  const ta_tensor *a;
  const ta_tensor *b;
  const ta_tensor *c;
  float alpha;
  uint32_t accumulate;
} ta_matmul_call;

const char *ta_version(void);
const char *ta_last_error(void);

ta_context *ta_context_create(uint32_t enable_validation,
                              const char *device_preference);
void ta_context_destroy(ta_context *ctx);

ta_executor *ta_executor_create(const ta_context *ctx, size_t n_slots,
                                uint32_t max_calls_per_submit,
                                const char *kernel_selection);
void ta_executor_destroy(ta_executor *exec);

ta_tensor *ta_tensor_create(const ta_context *ctx, const uint32_t *shape,
                            size_t rank);
ta_tensor *ta_tensor_create_on_executor(const ta_executor *exec,
                                        const uint32_t *shape, size_t rank);
void ta_tensor_destroy(ta_tensor *tensor);
uint64_t ta_tensor_len(const ta_tensor *tensor);
uint64_t ta_tensor_size_bytes(const ta_tensor *tensor);

int ta_upload(const ta_executor *exec, const ta_tensor *dst, const float *src,
              size_t len);
int ta_download(const ta_executor *exec, const ta_tensor *src, float *dst,
                size_t len);

int ta_matmul(const ta_executor *exec, const ta_tensor *a, const ta_tensor *b,
              const ta_tensor *c, float alpha, uint32_t accumulate,
              ta_run_stats *stats);
int ta_matmul_batch(const ta_executor *exec, const ta_matmul_call *calls,
                    size_t n_calls, ta_run_stats *stats);

#ifdef __cplusplus
}
#endif

#endif
