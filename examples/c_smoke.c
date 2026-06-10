#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor_ash.h"

static void die(const char *label) {
  fprintf(stderr, "%s: %s\n", label, ta_last_error());
  exit(1);
}

int main(void) {
  ta_context *ctx = ta_context_create(0, "auto");
  if (!ctx)
    die("ta_context_create");

  ta_executor *exec = ta_executor_create(ctx, 2, 16, "auto");
  if (!exec)
    die("ta_executor_create");

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

  ta_run_stats stats = {0};
  if (ta_matmul(exec, a, b, c, 1.0f, 0, &stats) != 0)
    die("ta_matmul");
  if (ta_download(exec, c, host_c, 4) != 0)
    die("ta_download");

  const float expected[4] = {58.0f, 64.0f, 139.0f, 154.0f};
  for (size_t i = 0; i < 4; ++i) {
    if (fabsf(host_c[i] - expected[i]) > 1e-4f) {
      fprintf(stderr, "bad result at %zu: got %.6f expected %.6f\n", i,
              host_c[i], expected[i]);
      return 2;
    }
  }

  printf("tensor-ash C smoke OK: %.6f %.6f %.6f %.6f",
         host_c[0], host_c[1], host_c[2], host_c[3]);
  if (stats.has_gpu_time)
    printf(" gpu_ns=%llu tflops=%.6f", (unsigned long long)stats.gpu_time_ns,
           stats.tflops);
  printf("\n");

  ta_tensor_destroy(c);
  ta_tensor_destroy(b);
  ta_tensor_destroy(a);
  ta_executor_destroy(exec);
  ta_context_destroy(ctx);
  return 0;
}
