//! Host-visible u32 cells used by replayable mixed-op graphs.

use anyhow::{Context, Result, bail};
use ash::vk;

use crate::buffer::{Buffer, BufferLocation};

use super::Executor;

/// Shared 4-byte host-visible cell.  [`PosBuffer`] and
/// [`HostU32Buffer`] are directional wrappers over the same storage.
struct U32Cell {
    buffer: Buffer,
}

fn create_u32_cell(exec: &Executor, label: &str) -> Result<Buffer> {
    if !exec.ctx.buffer_device_address_enabled {
        bail!("{label}: requires bufferDeviceAddress");
    }
    Buffer::new(
        &exec.ctx,
        std::mem::size_of::<u32>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        BufferLocation::Host,
    )
    .context(label.to_string())
}

impl U32Cell {
    fn new(exec: &Executor, label: &str) -> Result<Self> {
        Ok(Self {
            buffer: create_u32_cell(exec, label)?,
        })
    }

    fn set(&self, value: u32) -> Result<()> {
        self.buffer.write_pod_slice(&[value])
    }

    fn read(&self) -> Result<u32> {
        let mut value = [0u32];
        self.buffer.read_pod_slice(&mut value)?;
        Ok(value[0])
    }

    fn device_address(&self) -> u64 {
        self.buffer.device_address()
    }
}

/// A 4-byte host-visible, device-readable position cell for replayable
/// decode graphs (see [`Executor::prepare_exec_ops`]).  The ops that
/// depend on the token position take its [`device_address`]
/// (`RopeDesc::pos_addr`, `CopyDesc::pos_addr`,
/// `AttnDecodeDesc::pos_addr`); the recorded shaders read the current
/// value each execution, so the host just [`set`]s the new position
/// between replays instead of re-recording push constants.
///
/// Host writes made before a `vkQueueSubmit` are visible to that
/// submission (the submit performs the domain operation; `set` flushes
/// non-coherent memory), so no barrier is needed — only the usual
/// "don't write while a replay is in flight" discipline, which
/// [`super::PreparedOps::run`]'s fence wait already enforces.
///
/// [`device_address`]: Self::device_address
/// [`set`]: Self::set
pub struct PosBuffer {
    cell: U32Cell,
}

impl PosBuffer {
    /// Write a new position for the next submission.
    pub fn set(&self, value: u32) -> Result<()> {
        self.cell.set(value)
    }

    /// GPU pointer for the shaders' position indirection; store it in
    /// the descs' `pos_addr` fields.  The buffer must outlive every
    /// execution of any op that captured this address.
    pub fn device_address(&self) -> u64 {
        self.cell.device_address()
    }
}

/// A 4-byte u32 cell that is BOTH device-writable and host-readable —
/// the [`PosBuffer`] machinery pointed the other way.  The GPU-argmax
/// decode loop writes the chosen token id here
/// ([`super::ExecOp::Argmax`] / [`Executor::run_argmax`]), a chained
/// [`super::ExecOp::EmbedGather`] reads it back on-device for the next
/// token's embedding, and the host [`read`]s ONE u32 after the fence
/// instead of downloading the logits.
///
/// Device writes become visible to the host once the submission's fence
/// wait returns (the fence signal is the domain operation; `read`
/// invalidates non-coherent memory), so the usual "wait before you
/// read" discipline — which [`super::PreparedOps::run`]'s fence already
/// enforces — is the only requirement.
///
/// [`read`]: Self::read
pub struct HostU32Buffer {
    cell: U32Cell,
}

impl HostU32Buffer {
    /// Read the current value (call only after the writing submission's
    /// fence wait has returned).
    pub fn read(&self) -> Result<u32> {
        self.cell.read()
    }

    /// Host-side write, for seeding the cell before a submission that
    /// only reads it (e.g. a standalone embed-gather).
    pub fn set(&self, value: u32) -> Result<()> {
        self.cell.set(value)
    }

    /// GPU pointer for the shaders' indirection.  The buffer must
    /// outlive every execution of any op that captured this address.
    pub fn device_address(&self) -> u64 {
        self.cell.device_address()
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.cell.buffer
    }
}

/// Host-visible list of token ids for [`super::ExecOp::EmbedGatherRows`].
pub struct TokenIdBuffer {
    buffer: Buffer,
    cap: u32,
}

impl TokenIdBuffer {
    /// Overwrite the first `ids.len()` slots.  The gather shader reads
    /// `n_tokens` entries, so the caller must pass the same count.
    pub fn write(&self, ids: &[u32]) -> Result<()> {
        if ids.len() as u32 > self.cap {
            bail!(
                "TokenIdBuffer::write: {} ids exceeds cap {}",
                ids.len(),
                self.cap
            );
        }
        self.buffer.write_pod_slice(ids)
    }

    pub fn device_address(&self) -> u64 {
        self.buffer.device_address()
    }

    pub fn cap(&self) -> u32 {
        self.cap
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl Executor {
    /// Allocate a [`PosBuffer`] on this executor's device.
    pub fn create_pos_buffer(&self) -> Result<PosBuffer> {
        Ok(PosBuffer {
            cell: U32Cell::new(self, "create_pos_buffer")?,
        })
    }

    /// Allocate a [`HostU32Buffer`] on this executor's device.
    pub fn create_host_u32_buffer(&self) -> Result<HostU32Buffer> {
        Ok(HostU32Buffer {
            cell: U32Cell::new(self, "create_host_u32_buffer")?,
        })
    }

    /// Host-visible u32 list (prefill token ids).  `write` is a
    /// coherent/flushed CPU store — no staging submit — so a 512-token
    /// prompt is 2 KiB instead of a 4 MiB embedding upload.
    pub fn create_token_id_buffer(&self, cap: u32) -> Result<TokenIdBuffer> {
        if cap == 0 {
            bail!("create_token_id_buffer: cap must be non-zero");
        }
        if !self.ctx.buffer_device_address_enabled {
            bail!("create_token_id_buffer: requires bufferDeviceAddress");
        }
        let bytes = match (cap as u64).checked_mul(std::mem::size_of::<u32>() as u64) {
            Some(bytes) => bytes,
            None => bail!("create_token_id_buffer: cap overflows"),
        };
        Ok(TokenIdBuffer {
            buffer: Buffer::new(
                &self.ctx,
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                BufferLocation::Host,
            )
            .context("create_token_id_buffer")?,
            cap,
        })
    }
}
