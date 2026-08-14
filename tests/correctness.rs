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

#[path = "correctness/attn_decode.rs"]
mod attn_decode;
#[path = "correctness/batch.rs"]
mod batch;
#[path = "correctness/decoder.rs"]
mod decoder;
#[path = "correctness/elementwise.rs"]
mod elementwise;
#[path = "correctness/epilogue.rs"]
mod epilogue;
#[path = "correctness/f16.rs"]
mod f16;
#[path = "correctness/flash.rs"]
mod flash;
#[path = "correctness/fusion.rs"]
mod fusion;
#[path = "correctness/graph.rs"]
mod graph;
#[path = "correctness/kernels.rs"]
mod kernels;
#[path = "correctness/ops.rs"]
mod ops;
#[path = "correctness/outer.rs"]
mod outer;
#[path = "correctness/prepared.rs"]
mod prepared;
#[path = "correctness/property.rs"]
mod property;
#[path = "correctness/replay.rs"]
mod replay;
#[path = "correctness/row.rs"]
mod row;
#[path = "correctness/shapes.rs"]
mod shapes;
#[path = "correctness/splitk2.rs"]
mod splitk2;
#[path = "correctness/submission.rs"]
mod submission;
#[path = "correctness/token_loop.rs"]
mod token_loop;
#[path = "correctness/tuning.rs"]
mod tuning;
