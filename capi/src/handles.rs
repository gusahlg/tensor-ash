use std::sync::Arc;

use anyhow::{Result, bail};
use tensor_ash_core::{Executor, MatmulPipeline, Tensor, VulkanContext};

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
