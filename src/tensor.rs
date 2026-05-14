//! Tensor — shape + owned device buffer.
//!
//! Row-major, contiguous, f32.  Rank 2 (`[M, K]`) or rank 3 (`[B, M, K]`).
//! All host-side I/O goes through `Executor::upload` / `Executor::download`,
//! so this type intentionally has no host-visible variant.

use std::sync::Arc;

use anyhow::{Result, bail};
use ash::vk;

use crate::buffer::{Buffer, BufferLocation};
use crate::context::VulkanContext;

pub struct Tensor {
    pub shape:  Vec<u32>,
    pub buffer: Buffer,
}

impl Tensor {
    /// Number of f32 elements implied by `shape`.
    #[inline]
    pub fn numel(shape: &[u32]) -> u64 {
        shape.iter().map(|&d| d as u64).product()
    }

    /// Allocate a device-local tensor.  Contents are uninitialized — use
    /// `Executor::upload` to fill it.
    pub fn zeros_device(ctx: &Arc<VulkanContext>, shape: &[u32]) -> Result<Self> {
        let size = Self::numel(shape) * 4;
        let buffer = Buffer::new(
            ctx,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            BufferLocation::Device,
        )?;
        Ok(Self { shape: shape.to_vec(), buffer })
    }

    /// Returns `(batch, m, k)` for rank-2 (`batch=1`) or rank-3 shapes.
    pub fn as_3d(&self) -> Result<(u32, u32, u32)> {
        match self.shape.len() {
            2 => Ok((1, self.shape[0], self.shape[1])),
            3 => Ok((self.shape[0], self.shape[1], self.shape[2])),
            r => bail!("tensor rank {r} not supported by matmul (must be 2 or 3)"),
        }
    }

    #[inline] pub fn raw_buffer(&self) -> vk::Buffer        { self.buffer.raw }
    #[inline] pub fn size_bytes(&self) -> vk::DeviceSize    { self.buffer.size }
}
