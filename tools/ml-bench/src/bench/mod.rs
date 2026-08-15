mod cases;
mod commands;
mod env;
mod report;
mod thesis;

use std::env as std_env;
use std::sync::Arc;

use anyhow::Result;
use tensor_ash::{Executor, KernelSelection, MatmulPipeline, VulkanContext};

use env::{env_bool, env_string, env_usize};

pub fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut args = std_env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "all".into());
    let validate = env_bool("ML_VALIDATE");
    let n_slots = env_usize("ML_SLOTS", 2);
    let raw_kernel = env_string("ML_KERNEL", "auto");

    // `VulkanContext::new` reads ML_DEVICE itself.
    let ctx = VulkanContext::new(validate)?;
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
        "cases" => commands::cases(&ctx, &exec, args)?,
        "concurrent" => commands::concurrent(ctx.clone(), exec.clone())?,
        "transfer" => commands::transfer(&ctx, &exec)?,
        "prepared" => commands::prepared(&ctx, &exec)?,
        "thesis" => thesis::run(&ctx, &exec, args)?,
        "all" => {
            commands::correctness(&ctx, &exec)?;
            commands::sweep(&ctx, &exec)?;
        }
        _ => anyhow::bail!(
            "unknown subcommand '{cmd}'; expected self-check, correctness, sweep, single, cases, concurrent, transfer, prepared, thesis, or all"
        ),
    }
    Ok(())
}
