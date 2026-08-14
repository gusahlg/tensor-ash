//! Tensor — shape + owned device buffer.
//!
//! Row-major, contiguous storage, f32 by default with optional f16 (see
//! [`DType`]).  GEMM accepts rank 2 (`[M, K]`) or rank 3 (`[B, M, K]`);
//! utility paths such as transfer benchmarks may use other non-empty
//! shapes.  All host-side I/O goes through `Executor::upload` /
//! `Executor::download` as `&[f32]` regardless of storage type, so this
//! type intentionally has no host-visible variant.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;

use crate::buffer::{Buffer, BufferLocation};
use crate::context::VulkanContext;
use crate::dtype::DType;
use crate::matmul::MatrixShape;

pub struct Tensor {
    shape: Vec<u32>,
    dtype: DType,
    buffer: Buffer,
}

impl Tensor {
    pub(crate) fn belongs_to(&self, ctx: &Arc<VulkanContext>) -> bool {
        self.buffer.belongs_to(ctx)
    }

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

    /// Allocate a device-local tensor with **uninitialized** contents.
    /// Use `Executor::upload` (or a matmul that writes the whole tensor
    /// without `accumulate`) before reading the data.
    pub fn uninit_device(ctx: &Arc<VulkanContext>, shape: &[u32]) -> Result<Self> {
        Self::uninit_device_with(ctx, shape, DType::F32)
    }

    /// Allocate an f16-storage device tensor.  Contents are written via
    /// the same `&[f32]` upload path (values round to the nearest f16);
    /// kernels read it as half-precision and accumulate in f32.
    /// Requires `shaderFloat16` + `storageBuffer16BitAccess` (checked
    /// here so the failure is early and explicit).
    pub fn uninit_device_f16(ctx: &Arc<VulkanContext>, shape: &[u32]) -> Result<Self> {
        if !ctx.f16_storage_enabled {
            bail!("device does not support f16 storage (shaderFloat16 + storageBuffer16BitAccess)");
        }
        Self::uninit_device_with(ctx, shape, DType::F16)
    }

    fn uninit_device_with(ctx: &Arc<VulkanContext>, shape: &[u32], dtype: DType) -> Result<Self> {
        let size = Self::numel_checked(shape)?
            .checked_mul(dtype.size())
            .context("tensor byte size overflows u64")?;
        // Round f16 allocations up to 16 bytes so vectorized uvec4
        // readers can never touch memory outside the allocation even
        // when the element count is odd.
        let size = size
            .checked_next_multiple_of(16)
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
            dtype,
            buffer,
        })
    }

    /// Storage element type.
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns `(batch, m, k)` for rank-2 (`batch=1`) or rank-3 shapes.
    pub fn as_3d(&self) -> Result<(u32, u32, u32)> {
        let shape = MatrixShape::from_tensor_shape(&self.shape)?;
        Ok((shape.batch, shape.rows, shape.cols))
    }

    /// Read-only view of the shape.  Prefer this over reaching into the
    /// `shape` field directly so future versions can swap the storage
    /// type without breaking call sites.
    #[inline]
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    /// Number of dimensions.
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Total f32 element count.  Always non-zero (constructors reject
    /// zero-valued dims).
    #[inline]
    pub fn len(&self) -> u64 {
        Self::numel(&self.shape)
    }

    /// Always `false` for a valid tensor — every dim must be non-zero.
    /// Provided to silence the clippy `len_without_is_empty` lint.
    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    #[inline]
    pub fn raw_buffer(&self) -> vk::Buffer {
        self.buffer.raw_buffer()
    }

    /// Cached GPU pointer for `buffer_reference` addressing; 0 when
    /// `bufferDeviceAddress` is disabled.
    #[inline]
    pub(crate) fn device_address(&self) -> u64 {
        self.buffer.device_address()
    }
    #[inline]
    pub fn size_bytes(&self) -> vk::DeviceSize {
        self.buffer.size_bytes()
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

    #[test]
    fn numel_matches_shape_product() {
        assert_eq!(Tensor::numel(&[1]), 1);
        assert_eq!(Tensor::numel(&[2, 3]), 6);
        assert_eq!(Tensor::numel(&[4, 5, 6]), 120);
    }
}
