use std::sync::Arc;

use anyhow::{Result, bail};
use tensor_ash_core::{Executor, MatmulPipeline, PreparedOps, Tensor, VulkanContext};

#[repr(C)]
pub struct ta_context {
    pub(crate) ctx: Arc<VulkanContext>,
}

#[repr(C)]
pub struct ta_executor {
    pub(crate) ctx: Arc<VulkanContext>,
    pub(crate) _pipeline: Arc<MatmulPipeline>,
    pub(crate) exec: Executor,
}

#[repr(C)]
pub struct ta_tensor {
    pub(crate) ctx: Arc<VulkanContext>,
    pub(crate) tensor: Tensor,
}

/// Opaque prepared-replay handle.
///
/// SAFETY reasoning for the erased lifetimes: the Rust type
/// `PreparedOps<'e, 't>` borrows the executor and every tensor whose
/// device address was baked into the recorded command buffer. Across
/// the C boundary those borrows cannot be expressed, so the handle
/// stores `PreparedOps<'static, 'static>`. Validity is guaranteed by
/// the documented C contract instead: the `ta_executor` and every
/// `ta_tensor` referenced by the prepared ops must outlive this handle
/// (destroying them first is undefined behavior), and
/// `ta_prepared_destroy` waits for in-flight work — `PreparedOps`'s
/// `Drop` impl fence-waits before releasing its Vulkan objects.
///
/// Field order matters: `prepared` is declared first so it drops (and
/// fence-waits) before the `Arc` below releases its context reference.
#[repr(C)]
pub struct ta_prepared {
    pub(crate) prepared: PreparedOps<'static, 'static>,
    pub(crate) _ctx: Arc<VulkanContext>,
}

impl ta_executor {
    pub(crate) fn checked_tensor<'a>(
        &self,
        ptr: *const ta_tensor,
        operation: &str,
        operand: &str,
    ) -> Result<&'a ta_tensor> {
        if ptr.is_null() {
            bail!("{operation}: {operand} is null");
        }
        let tensor = unsafe { &*ptr };
        if !Arc::ptr_eq(&self.ctx, &tensor.ctx) {
            bail!("{operation}: {operand} belongs to a different Vulkan context");
        }
        Ok(tensor)
    }
}
