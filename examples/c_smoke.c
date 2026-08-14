#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor_ash.h"

static void die(const char *label) {
  fprintf(stderr, "%s: %s\n", label, ta_last_error());
  exit(1);
}

static void check_close(const char *label, float got, float expected,
                        float tol) {
  if (fabsf(got - expected) > tol) {
    fprintf(stderr, "%s: got %.6f expected %.6f (tol %.4f)\n", label, got,
            expected, tol);
    exit(2);
  }
}

int main(void) {
  ta_context *ctx = ta_context_create(0, "auto");
  if (!ctx)
    die("ta_context_create");

  ta_executor *exec = ta_executor_create(ctx, 2, 16, "auto");
  if (!exec)
    die("ta_executor_create");

  const uint32_t supports_f16 = ta_context_supports_f16(ctx);
  const uint32_t supports_bda = ta_context_supports_bda(ctx);
  printf("tensor-ash %s: f16=%u bda=%u\n", ta_version(), supports_f16,
         supports_bda);

  const uint32_t a_shape[2] = {2, 3};
  const uint32_t b_shape[2] = {3, 2};
  const uint32_t c_shape[2] = {2, 2};
  ta_tensor *a = ta_tensor_create(ctx, a_shape, 2);
  ta_tensor *b = ta_tensor_create(ctx, b_shape, 2);
  ta_tensor *c = ta_tensor_create(ctx, c_shape, 2);
  if (!a || !b || !c)
    die("ta_tensor_create");

  const float host_a[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const float host_b[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  float host_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if (ta_upload(exec, a, host_a, 6) != 0)
    die("ta_upload A");
  if (ta_upload(exec, b, host_b, 6) != 0)
    die("ta_upload B");

  /* 1. Plain synchronous GEMM. */
  ta_run_stats stats = {0};
  if (ta_matmul(exec, a, b, c, 1.0f, 0, &stats) != 0)
    die("ta_matmul");
  if (ta_download(exec, c, host_c, 4) != 0)
    die("ta_download");

  const float expected[4] = {58.0f, 64.0f, 139.0f, 154.0f};
  for (size_t i = 0; i < 4; ++i)
    check_close("ta_matmul result", host_c[i], expected[i], 1e-4f);

  printf("plain matmul OK: %.6f %.6f %.6f %.6f", host_c[0], host_c[1],
         host_c[2], host_c[3]);
  if (stats.has_gpu_time)
    printf(" gpu_ns=%llu tflops=%.6f", (unsigned long long)stats.gpu_time_ns,
           stats.tflops);
  printf("\n");

  ta_dispatch_info info = {0};
  if (ta_dispatch_info_for(exec, 1, 2, 2, 3, 0, &info) != 0)
    die("ta_dispatch_info_for");
  printf("dispatch 1x2x2x3 -> kernel=%s tile=%ux%ux%u split_k2=%u\n",
         info.kernel, info.tile[0], info.tile[1], info.tile[2],
         info.has_split_k2 ? info.split_k2_splits : 0);

  if (supports_bda) {
    /* 2. Fused bias + SiLU epilogue via ta_run_ops.
     * c00 = 58 + bias0 = -0.5; silu(-0.5) = -0.5*sigmoid(-0.5). */
    const uint32_t bias_shape[1] = {2};
    ta_tensor *bias = ta_tensor_create(ctx, bias_shape, 1);
    if (!bias)
      die("ta_tensor_create bias");
    const float host_bias[2] = {-58.5f, 0.0f};
    if (ta_upload(exec, bias, host_bias, 2) != 0)
      die("ta_upload bias");

    ta_matmul_op op = {0};
    op.call.a = a;
    op.call.b = b;
    op.call.c = c;
    op.call.alpha = 1.0f;
    op.epilogue.bias = bias;
    op.epilogue.activation = TA_ACTIVATION_SILU;
    if (ta_run_ops(exec, &op, 1, &stats) != 0)
      die("ta_run_ops");
    if (ta_download(exec, c, host_c, 4) != 0)
      die("ta_download epilogue");
    const float x = 58.0f - 58.5f; /* -0.5 */
    const float silu = x / (1.0f + expf(-x));
    check_close("bias+silu c[0]", host_c[0], silu, 1e-3f);
    check_close("bias+silu c[1]", host_c[1], 64.0f, 1e-2f);
    printf("bias+silu op OK: c[0]=%.6f (expected %.6f)\n", host_c[0], silu);
    ta_tensor_destroy(bias);

    /* 3. Prepared create/run/run/destroy round trip. */
    ta_matmul_op plain = {0};
    plain.call.a = a;
    plain.call.b = b;
    plain.call.c = c;
    plain.call.alpha = 1.0f;
    ta_prepared *prepared = ta_prepared_create(exec, &plain, 1, 0);
    if (!prepared)
      die("ta_prepared_create");
    if (ta_prepared_run(prepared, &stats) != 0)
      die("ta_prepared_run #1");
    if (ta_prepared_run(prepared, &stats) != 0)
      die("ta_prepared_run #2");
    if (ta_prepared_submit(prepared) != 0)
      die("ta_prepared_submit");
    if (ta_prepared_wait(prepared, &stats) != 0)
      die("ta_prepared_wait");
    if (ta_download(exec, c, host_c, 4) != 0)
      die("ta_download prepared");
    for (size_t i = 0; i < 4; ++i)
      check_close("prepared result", host_c[i], expected[i], 1e-4f);
    ta_prepared_destroy(prepared);
    printf("prepared replay OK: %.6f %.6f %.6f %.6f\n", host_c[0], host_c[1],
           host_c[2], host_c[3]);
  } else {
    printf("skipping epilogue + prepared checks (no bufferDeviceAddress)\n");
  }

  if (supports_f16) {
    /* 4. f16 weight storage: upload converts to half, kernels widen. */
    ta_tensor *b16 = ta_tensor_create_f16(ctx, b_shape, 2);
    if (!b16)
      die("ta_tensor_create_f16");
    if (!ta_tensor_is_f16(b16)) {
      fprintf(stderr, "ta_tensor_is_f16 returned 0 for an f16 tensor\n");
      return 2;
    }
    if (ta_upload(exec, b16, host_b, 6) != 0)
      die("ta_upload B f16");
    if (ta_matmul(exec, a, b16, c, 1.0f, 0, &stats) != 0)
      die("ta_matmul f16");
    if (ta_download(exec, c, host_c, 4) != 0)
      die("ta_download f16");
    check_close("f16 matmul c[0]", host_c[0], 58.0f, 0.5f);
    ta_tensor_destroy(b16);
    printf("f16 matmul OK: c[0]=%.6f\n", host_c[0]);
  } else {
    printf("skipping f16 check (no f16 storage support)\n");
  }

  if (supports_bda) {
    /* 5. Model ops: rms_norm, prefix-masked softmax, strided copy.
     * A still holds {1,2,3,4,5,6} as a 2x3 matrix. */
    ta_tensor *w = ta_tensor_create(ctx, a_shape + 1, 1); /* [3] */
    ta_tensor *out23 = ta_tensor_create(ctx, a_shape, 2); /* [2,3] */
    ta_tensor *out32 = ta_tensor_create(ctx, b_shape, 2); /* [3,2] */
    if (!w || !out23 || !out32)
      die("ta_tensor_create model-op operands");
    const float host_w[3] = {1.0f, 0.5f, 2.0f};
    if (ta_upload(exec, w, host_w, 3) != 0)
      die("ta_upload weight");

    /* rms_norm: out = x * w / sqrt(mean(x^2) + eps), per row. */
    const float eps = 1e-5f;
    if (ta_rms_norm(exec, a, w, out23, eps, &stats) != 0)
      die("ta_rms_norm");
    float host_out[6];
    if (ta_download(exec, out23, host_out, 6) != 0)
      die("ta_download rms_norm");
    for (size_t r = 0; r < 2; ++r) {
      float ss = 0.0f;
      for (size_t j = 0; j < 3; ++j)
        ss += host_a[3 * r + j] * host_a[3 * r + j];
      const float inv_rms = 1.0f / sqrtf(ss / 3.0f + eps);
      for (size_t j = 0; j < 3; ++j)
        check_close("rms_norm", host_out[3 * r + j],
                    host_a[3 * r + j] * host_w[j] * inv_rms, 1e-4f);
    }
    printf("rms_norm OK: row0 = %.6f %.6f %.6f\n", host_out[0], host_out[1],
           host_out[2]);

    /* softmax with a prefix mask: columns >= 2 store exactly 0.0. */
    if (ta_softmax_rows(exec, a, out23, 1.0f, TA_SOFTMAX_MASK_PREFIX, 2, 0,
                        &stats) != 0)
      die("ta_softmax_rows");
    if (ta_download(exec, out23, host_out, 6) != 0)
      die("ta_download softmax");
    for (size_t r = 0; r < 2; ++r) {
      const float m = fmaxf(host_a[3 * r], host_a[3 * r + 1]);
      const float e0 = expf(host_a[3 * r] - m);
      const float e1 = expf(host_a[3 * r + 1] - m);
      check_close("softmax col0", host_out[3 * r], e0 / (e0 + e1), 1e-5f);
      check_close("softmax col1", host_out[3 * r + 1], e1 / (e0 + e1), 1e-5f);
      check_close("softmax masked col2", host_out[3 * r + 2], 0.0f, 0.0f);
    }
    printf("softmax(prefix=2) OK: row0 = %.6f %.6f %.6f\n", host_out[0],
           host_out[1], host_out[2]);

    /* copy_strided: transpose the 2x3 A into a 3x2 tensor. */
    ta_copy_desc transpose = {0};
    transpose.extent[0] = 3;
    transpose.extent[1] = 2;
    transpose.extent[2] = 1;
    transpose.src_strides[0] = 1; /* walk columns of A */
    transpose.src_strides[1] = 3; /* then rows */
    transpose.dst_strides[0] = 2; /* which are rows of A^T */
    transpose.dst_strides[1] = 1; /* and columns */
    if (ta_copy_strided(exec, a, out32, &transpose, &stats) != 0)
      die("ta_copy_strided");
    if (ta_download(exec, out32, host_out, 6) != 0)
      die("ta_download copy_strided");
    const float expected_t[6] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    for (size_t i = 0; i < 6; ++i)
      check_close("copy_strided transpose", host_out[i], expected_t[i], 0.0f);
    /* In-place copy must be rejected (unordered invocations race). */
    if (ta_copy_strided(exec, a, a, &transpose, NULL) == 0) {
      fprintf(stderr, "ta_copy_strided accepted src == dst\n");
      return 2;
    }
    printf("copy_strided transpose OK (in-place rejected: %s)\n",
           ta_last_error());

    ta_tensor_destroy(out32);
    ta_tensor_destroy(out23);
    ta_tensor_destroy(w);
  } else {
    printf("skipping model-op checks (no bufferDeviceAddress)\n");
  }

  printf("tensor-ash C smoke OK\n");

  ta_tensor_destroy(c);
  ta_tensor_destroy(b);
  ta_tensor_destroy(a);
  ta_executor_destroy(exec);
  ta_context_destroy(ctx);
  return 0;
}
