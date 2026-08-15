//! Matmul API and CPU-side shape resolution.
//!
//! Public operation types are kept separate from the validation and shader-ABI
//! details used by executors. Re-exports here preserve the `crate::matmul` API.

mod api;
mod resolution;

pub use api::{
    Activation, Epilogue, EpilogueBinary, MatmulCall, MatmulOp, MatmulStore, MatmulStoreDesc,
    RunStats,
};
pub use resolution::MatrixShape;
pub(crate) use resolution::{ResolvedMatmul, ResolvedMatmulBatch, total_flops};

#[cfg(test)]
mod tests;
