//! Tensor — shape + owned device buffer.
//!
//! Row-major, contiguous, f32 storage.  GEMM accepts rank 2 (`[M, K]`) or
//! rank 3 (`[B, M, K]`); utility paths such as transfer benchmarks may use
//! other non-empty shapes.  All host-side I/O goes through `Executor::upload`
//! / `Executor::download`, so this type intentionally has no host-visible
//! variant.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;

use crate::buffer::{Buffer, BufferLocation};
use crate::context::VulkanContext;
use crate::matmul::MatrixShape;

pub struct Tensor {
    pub shape: Vec<u32>,
    pub buffer: Buffer,
}

impl Tensor {
    /// Number of f32 elements implied by `shape`.
    #[inline]
    pub fn numel(shape: &[u32]) -> u64 {
        shape.iter().map(|&d| d as u64).product()
    }

    /// Checked number of f32 elements implied by `shape`.
    pub fn numel_checked(shape: &[u32]) -> Result<u64> {
        if shape.is_empty() {
            bail!("tensor shape must have at least one dimension");
        }
        shape
            .iter()
            .enumerate()
            .try_fold(1u64, |acc, (index, &dim)| {
                if dim == 0 {
                    bail!("tensor dimension {index} must be non-zero");
                }
                acc.checked_mul(dim as u64)
                    .ok_or_else(|| anyhow!("tensor element count overflows u64"))
            })
    }

    /// Allocate a device-local tensor.  Contents are uninitialized — use
    /// `Executor::upload` to fill it.
    pub fn zeros_device(ctx: &Arc<VulkanContext>, shape: &[u32]) -> Result<Self> {
        let size = Self::numel_checked(shape)?
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .context("tensor byte size overflows u64")?;
        let buffer = Buffer::new(
            ctx,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            BufferLocation::Device,
        )?;
        Ok(Self {
            shape: shape.to_vec(),
            buffer,
        })
    }

    /// Returns `(batch, m, k)` for rank-2 (`batch=1`) or rank-3 shapes.
    pub fn as_3d(&self) -> Result<(u32, u32, u32)> {
        let shape = MatrixShape::from_tensor_shape(&self.shape)?;
        Ok((shape.batch, shape.rows, shape.cols))
    }

    #[inline]
    pub fn raw_buffer(&self) -> vk::Buffer {
        self.buffer.raw
    }
    #[inline]
    pub fn size_bytes(&self) -> vk::DeviceSize {
        self.buffer.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_numel_rejects_empty_or_zero_shapes() {
        assert!(Tensor::numel_checked(&[]).is_err());
        assert!(Tensor::numel_checked(&[4, 0]).is_err());
    }

    #[test]
    fn checked_numel_reports_overflow() {
        let err = Tensor::numel_checked(&[u32::MAX, u32::MAX, u32::MAX])
            .unwrap_err()
            .to_string();

        assert!(err.contains("overflows"));
    }
}
