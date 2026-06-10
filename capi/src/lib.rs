//! C ABI for `tensor-ash`.
//!
//! The exported functions are intentionally thin wrappers over the Rust API:
//! opaque handles own the Rust objects, operations return `0` on success and
//! `-1` on failure, and `ta_last_error()` exposes the last per-thread error.

mod api;
mod error;
mod handles;
mod types;
