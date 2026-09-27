#!/usr/bin/env bash
# 2026-09-27 merge-readiness gates on the 4060 (run under lgpu after lbuild).
cd ~/ta-wt/c-ptx
T=$(ls -t target/release/deps/correctness-* | grep -v '\.d$' | head -1)
echo "## binary $T"
echo "## ptx:: with ML_PTX=1 (ML_PTX_GEMM default = auto)"
ML_PTX=1 timeout 400 $T --ignored --test-threads=1 ptx:: 2>&1 | grep -E "^test |test result|panicked|rel-RMS|skipping"
echo "## ptx:: without ML_PTX (plain device creation; policy test takes the degrade branch)"
timeout 200 $T --ignored --test-threads=1 ptx:: 2>&1 | grep -E "^test |test result|panicked|skipping"
echo "## ptx:: with ML_PTX=1 ML_NO_PTX=1 (veto)"
ML_PTX=1 ML_NO_PTX=1 RUST_LOG=info timeout 200 $T --ignored --test-threads=1 ptx:: 2>&1 | grep -E "^test |test result|panicked|skipping|vetoed"
echo "## ptx:: with ML_PTX=1 ML_PTX_GEMM=off"
ML_PTX=1 ML_PTX_GEMM=off timeout 200 $T --ignored --test-threads=1 ptx:: 2>&1 | grep -E "^test |test result|panicked|skipping"
echo "## f16:: graph:: prepared:: replay:: with the PTX route serving f32 (ML_PTX_GEMM=all)"
ML_PTX=1 ML_PTX_GEMM=all timeout 500 $T --ignored --test-threads=1 f16:: graph:: prepared:: replay:: 2>&1 | grep -E "test result|panicked|FAILED"
echo "## same suites, control without PTX"
timeout 500 $T --ignored --test-threads=1 f16:: graph:: prepared:: replay:: 2>&1 | grep -E "test result|panicked|FAILED"
echo "## invalid settings are errors"
ML_PTX=1 ML_PTX_GEMM=bogus timeout 60 target/release/llama_ash bench -m /nonexistent 2>&1 | tail -2
ML_COOPMAT_ACC=bf16 timeout 60 target/release/llama_ash bench -m /nonexistent 2>&1 | tail -2
echo "## deprecated ML_PTX_GEMM=f16p still reaches the f16p class (warning expected)"
ML_PTX=1 ML_PTX_GEMM=f16p timeout 100 target/release/llama_ash bench -m ~/models/tinyllama-1.1b-chat-v1.0.f16.gguf --pp 512 --tg 8 2>&1 | grep -E "deprecated|accumulator policy|pp512"
