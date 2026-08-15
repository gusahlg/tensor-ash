//! llama-ash as a library: GGUF weight loading and the llama-family
//! forward passes (prefill / decode) built on tensor-ash ops.
//!
//! Shared by the `llama_ash` CLI binary and the `ml_bench thesis`
//! performance-thesis harness, which loads real models to validate the
//! campaign's performance model (decode bandwidth accounting, prefill
//! time accounting, cross-mode token exactness).

pub mod gguf;
pub mod model;
