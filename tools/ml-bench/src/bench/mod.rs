mod cases;
mod commands;
mod env;
mod report;

use std::env as std_env;
use std::sync::Arc;

use anyhow::Result;
use tensor_ash::{DevicePreference, Executor, KernelSelection, MatmulPipeline, VulkanContext};

use env::{env_bool, env_string, env_usize};

pub fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cmd = std_env::args().nth(1).unwrap_or_else(|| "all".into());
    let validate = env_bool("ML_VALIDATE");
    let n_slots = env_usize("ML_SLOTS", 2);
    let device_preference = DevicePreference::parse(&env_string("ML_DEVICE", "auto"))?;
    let raw_kernel = env_string("ML_KERNEL", "auto");

    let ctx = VulkanContext::new_with_device_preference(validate, device_preference)?;
    let pipe = Arc::new(MatmulPipeline::new_with_kernel_selection(
        &ctx,
        KernelSelection::from_env()?,
    )?);
    let exec = Arc::new(Executor::new(
        ctx.clone(),
        pipe,
        n_slots,
        /*max_calls=*/ 256,
    )?);

    log::info!(
        "{} slots={} kernel={}",
        ctx.diagnostics(),
        n_slots,
        raw_kernel.as_str(),
    );

    match cmd.as_str() {
        "self-check" => commands::self_check(&ctx, n_slots)?,
        "correctness" => commands::correctness(&ctx, &exec)?,
        "sweep" => commands::sweep(&ctx, &exec)?,
        "single" => commands::single(&ctx, &exec)?,
        "concurrent" => commands::concurrent(ctx.clone(), exec.clone())?,
        "transfer" => commands::transfer(&ctx, &exec)?,
        "all" => {
            commands::correctness(&ctx, &exec)?;
            commands::sweep(&ctx, &exec)?;
        }
        _ => {
            log::warn!("unknown subcommand '{cmd}', running default correctness+sweep");
            commands::correctness(&ctx, &exec)?;
            commands::sweep(&ctx, &exec)?;
        }
    }
    Ok(())
}
