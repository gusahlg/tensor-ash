//! Resource-ownership checks performed before raw Vulkan handles are used.

use anyhow::{Result, bail};

use crate::matmul::{MatmulCall, MatmulOp};
use crate::tensor::Tensor;

use super::Executor;

impl Executor {
    pub(super) fn validate_tensor_context(&self, tensor: &Tensor, role: &str) -> Result<()> {
        if !tensor.belongs_to(&self.ctx) {
            bail!("{role} tensor belongs to a different VulkanContext");
        }
        Ok(())
    }

    pub(super) fn validate_call_context(&self, call: &MatmulCall<'_>) -> Result<()> {
        self.validate_tensor_context(call.a, "A")?;
        self.validate_tensor_context(call.b, "B")?;
        self.validate_tensor_context(call.c, "C")?;
        if call.c.raw_buffer() == call.a.raw_buffer() || call.c.raw_buffer() == call.b.raw_buffer()
        {
            bail!("C must not alias A or B; in-place matmul is not supported");
        }
        Ok(())
    }

    pub(super) fn validate_op_context(&self, op: &MatmulOp<'_>) -> Result<()> {
        self.validate_call_context(&op.call)?;
        if let Some(bias) = op.epilogue.bias {
            self.validate_tensor_context(bias, "bias")?;
        }
        if let Some(d) = op.epilogue.d_tensor() {
            self.validate_tensor_context(d, "D")?;
        }
        Ok(())
    }
}
