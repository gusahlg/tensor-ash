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
typedef struct ta_prepared ta_prepared;

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

/* Values for ta_epilogue.activation. */
enum {
  TA_ACTIVATION_NONE = 0,
  TA_ACTIVATION_RELU = 1,
  TA_ACTIVATION_SILU = 2, /* x * sigmoid(x) */
  TA_ACTIVATION_GELU = 3  /* tanh-approximated GELU */
};

/* Values for ta_epilogue.binary_kind. */
enum {
  TA_BINARY_NONE = 0,
  TA_BINARY_ADD_SCALED = 1, /* out = act_result + beta * D */
  TA_BINARY_MUL = 2         /* out = act_result * D */
};

/* Fused epilogue applied while the output tile is still in registers,
 * in order: +bias -> activation -> binary op with D. An all-zero value
 * means "no epilogue". bias may be null (absent) and may be broadcast
 * [N] or batched [B, N]; d must be non-null iff binary_kind != 0 and
 * must have exactly C's shape. Epilogues require a bufferDeviceAddress
 * device (see ta_context_supports_bda). */
typedef struct ta_epilogue {
  const ta_tensor *bias;
  uint32_t activation;  /* TA_ACTIVATION_* */
  uint32_t binary_kind; /* TA_BINARY_* */
  const ta_tensor *d;
  float beta; /* used by TA_BINARY_ADD_SCALED */
} ta_epilogue;

/* One GEMM plus its fused epilogue. */
typedef struct ta_matmul_op {
  ta_matmul_call call;
  ta_epilogue epilogue;
} ta_matmul_op;

/* Route description for one matmul shape. kernel points to an interned
 * NUL-terminated name valid for the process lifetime. */
typedef struct ta_dispatch_info {
  const char *kernel;
  uint32_t tile[3];
  uint32_t has_split_k2;
  uint32_t split_k2_splits; /* meaningful when has_split_k2 != 0 */
} ta_dispatch_info;

const char *ta_version(void);
const char *ta_last_error(void);

ta_context *ta_context_create(uint32_t enable_validation,
                              const char *device_preference);
void ta_context_destroy(ta_context *ctx);

/* Capability queries; return 0 for null contexts. */
uint32_t ta_context_supports_f16(const ta_context *ctx);
uint32_t ta_context_supports_bda(const ta_context *ctx);

ta_executor *ta_executor_create(const ta_context *ctx, size_t n_slots,
                                uint32_t max_calls_per_submit,
                                const char *kernel_selection);
/* ta_executor_create with an explicit tuning policy instead of the
 * process-wide ML_TUNE environment variable. tune != 0 measures every
 * eligible kernel the first time a new shape is submitted and persists
 * the winner. */
ta_executor *ta_executor_create_v2(const ta_context *ctx, size_t n_slots,
                                   uint32_t max_calls_per_submit,
                                   const char *kernel_selection,
                                   uint32_t tune);
void ta_executor_destroy(ta_executor *exec);

ta_tensor *ta_tensor_create(const ta_context *ctx, const uint32_t *shape,
                            size_t rank);
ta_tensor *ta_tensor_create_on_executor(const ta_executor *exec,
                                        const uint32_t *shape, size_t rank);
/* f16-storage tensors: kernels read half precision and accumulate in
 * f32. Fails (returns null) when ta_context_supports_f16 is 0.
 * ta_upload / ta_download keep the float host API and convert
 * automatically (round-to-nearest half on upload, widen on download). */
ta_tensor *ta_tensor_create_f16(const ta_context *ctx, const uint32_t *shape,
                                size_t rank);
ta_tensor *ta_tensor_create_f16_on_executor(const ta_executor *exec,
                                            const uint32_t *shape, size_t rank);
void ta_tensor_destroy(ta_tensor *tensor);
uint64_t ta_tensor_len(const ta_tensor *tensor);
uint64_t ta_tensor_size_bytes(const ta_tensor *tensor);
uint32_t ta_tensor_is_f16(const ta_tensor *tensor);

int ta_upload(const ta_executor *exec, const ta_tensor *dst, const float *src,
              size_t len);
int ta_download(const ta_executor *exec, const ta_tensor *src, float *dst,
                size_t len);

int ta_matmul(const ta_executor *exec, const ta_tensor *a, const ta_tensor *b,
              const ta_tensor *c, float alpha, uint32_t accumulate,
              ta_run_stats *stats);
int ta_matmul_batch(const ta_executor *exec, const ta_matmul_call *calls,
                    size_t n_calls, ta_run_stats *stats);

/* Run n_ops independent GEMM+epilogue ops in one submit (no inter-op
 * barriers). */
int ta_run_ops(const ta_executor *exec, const ta_matmul_op *ops, size_t n_ops,
               ta_run_stats *stats);
/* ta_run_ops for dependent chains: hazards between ops (including
 * epilogue bias / D operands) are resolved with automatic barriers, so
 * op N may read outputs of earlier ops in the same submission. */
int ta_run_op_graph(const ta_executor *exec, const ta_matmul_op *ops,
                    size_t n_ops, ta_run_stats *stats);

/* Prepared replay: validate, route, and record a fixed op batch once,
 * then replay it with one queue submit per call. as_graph != 0 records
 * dependency barriers (like ta_run_op_graph); 0 requires independent
 * ops (like ta_run_ops). Requires ta_context_supports_bda.
 *
 * LIFETIME CONTRACT (undefined behavior if violated): the executor and
 * every tensor referenced by the ops must outlive the ta_prepared —
 * the recorded command buffer bakes their device addresses in.
 * Destroy the ta_prepared (which waits for in-flight work) before
 * destroying any of them. */
ta_prepared *ta_prepared_create(const ta_executor *exec,
                                const ta_matmul_op *ops, size_t n_ops,
                                uint32_t as_graph);
/* Submit and block until GPU completion. */
int ta_prepared_run(ta_prepared *prepared, ta_run_stats *stats);
/* Pipelined path: submit without waiting. Call ta_prepared_wait before
 * submitting this handle again; overlap by ping-ponging two handles.
 * While work is in flight the handle must reach ta_prepared_wait or
 * ta_prepared_destroy before its executor/tensors are destroyed or the
 * process exits. */
int ta_prepared_submit(ta_prepared *prepared);
int ta_prepared_wait(ta_prepared *prepared, ta_run_stats *stats);
void ta_prepared_destroy(ta_prepared *prepared);

/* Report the kernel and optional two-stage split-K route a plain,
 * non-accumulating matmul of this shape would use (b_f16 != 0: route
 * for f16 weights). out->kernel stays valid for the process lifetime. */
int ta_dispatch_info_for(const ta_executor *exec, uint32_t batch, uint32_t m,
                         uint32_t n, uint32_t k, uint32_t b_f16,
                         ta_dispatch_info *out);
/* Measure every eligible kernel for one shape and persist the winner
 * in the tuning store; no-op if already tuned. */
int ta_tune_shape(const ta_executor *exec, uint32_t batch, uint32_t m,
                  uint32_t n, uint32_t k);

#ifdef __cplusplus
}
#endif

#endif
