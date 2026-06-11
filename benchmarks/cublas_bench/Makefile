# Build the pure cuBLAS FP32 SGEMM benchmark binary.
#
# nvcc picks up cuBLAS via -lcublas; the runtime needs libcublas.so on
# LD_LIBRARY_PATH, which the project's `nix develop .#benchmark` shell
# arranges automatically.
#
# Targets RTX 3070 (Ampere consumer = SM_86) by default; override with
# `make ARCH=sm_75` etc. if rebuilding for a different GPU.

NVCC ?= nvcc
ARCH ?= sm_86
CXXFLAGS ?= -O3 -std=c++17

cublas_bench: cublas_bench.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) -lcublas -lcudart \
		-o $@ $<

.PHONY: clean
clean:
	rm -f cublas_bench
