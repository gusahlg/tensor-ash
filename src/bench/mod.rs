mod cases;
mod commands;
mod env;
mod report;

use std::env as std_env;
use std::sync::Arc;

use anyhow::Result;
use tensor_ash::{
    DevicePreference, Executor, KernelSelection, MatmulPipeline, PersistentMatmul, VulkanContext,
};

use env::{env_bool, env_string, env_usize};

pub fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cmd = std_env::args().nth(1).unwrap_or_else(|| "all".into());
    let validate = env_bool("ML_VALIDATE");
    let n_slots = env_usize("ML_SLOTS", 2);
    let device_preference = DevicePreference::parse(&env_string("ML_DEVICE", "auto"))?;
    let raw_kernel = env_string("ML_KERNEL", "auto");
    let persistent_requested = raw_kernel.eq_ignore_ascii_case("persistent_v4");

    let ctx = VulkanContext::new_with_device_preference(validate, device_preference)?;
    // When the user asks for the persistent kernel we still need a
    // regular MatmulPipeline so that the executor (which the standard
    // benchmark plumbing depends on for upload/download) can be built.
    // The pipeline is built with `Auto` so it doesn't choke on the
    // unknown `persistent_v4` value; the persistent path runs on its
    // own pipeline alongside it.
    let pipe_selection = if persistent_requested {
        KernelSelection::Auto
    } else {
        KernelSelection::from_env()?
    };
    let pipe = Arc::new(MatmulPipeline::new_with_kernel_selection(
        &ctx,
        pipe_selection,
    )?);
    let exec = Arc::new(Executor::new(
        ctx.clone(),
        pipe,
        n_slots,
        /*max_calls=*/ 256,
    )?);

    let mut persistent: Option<PersistentMatmul> = if persistent_requested {
        Some(PersistentMatmul::new(&ctx)?)
    } else {
        None
    };

    log::info!(
        "{} slots={} kernel={}",
        ctx.diagnostics(),
        n_slots,
        if persistent_requested {
            "persistent_v4"
        } else {
            raw_kernel.as_str()
        },
    );

    match cmd.as_str() {
        "self-check" => commands::self_check(&ctx, n_slots)?,
        "correctness" => commands::correctness(&ctx, &exec)?,
        "sweep" => commands::sweep(&ctx, &exec)?,
        "single" => {
            // The `commands::single_persistent` helper was never finished;
            // until it's wired up we fall back to the regular `single`
            // even when ML_KERNEL=persistent_v4 was requested.
            let _ = &persistent;
            commands::single(&ctx, &exec)?;
        }
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
