//! End-to-end correctness suite.  Every test compares GPU output against
//! an f64-accumulated CPU reference.
//!
//! All tests are `#[ignore]`d so `cargo test` works on a GPU-less host.
//! Run with:
//!
//!     cargo test --release --test correctness -- --ignored --test-threads=1
//!
//! `--test-threads=1` keeps to one Vulkan instance at a time, which
//! avoids flaky driver behavior on certain platforms.

mod common;

#[path = "correctness/batch.rs"]
mod batch;
#[path = "correctness/epilogue.rs"]
mod epilogue;
#[path = "correctness/graph.rs"]
mod graph;
#[path = "correctness/kernels.rs"]
mod kernels;
#[path = "correctness/ops.rs"]
mod ops;
#[path = "correctness/property.rs"]
mod property;
#[path = "correctness/shapes.rs"]
mod shapes;
#[path = "correctness/splitk2.rs"]
mod splitk2;
#[path = "correctness/streamk.rs"]
mod streamk;
#[path = "correctness/submission.rs"]
mod submission;
#[path = "correctness/tuning.rs"]
mod tuning;
