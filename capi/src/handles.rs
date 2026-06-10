use std::sync::Arc;

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
    pub(crate) tensor: Tensor,
}
