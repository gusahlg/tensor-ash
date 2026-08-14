//! llama_ash: run an f16 llama-family GGUF end to end on tensor-ash
//! and measure prefill / decode throughput.
//!
//! Subcommands:
//!   bench    -m <gguf> [--pp 512] [--tg 128] [--ctx 2048]
//!   generate -m <gguf> --ids "1,15043,3186" -n 24 [--ctx 2048]

mod gguf;
mod model;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use tensor_ash::{DevicePreference, Executor, KernelSelection, MatmulPipeline, VulkanContext};

use model::Model;

struct Args {
    model: PathBuf,
    pp: u32,
    tg: u32,
    ctx: u32,
    ids: Vec<u32>,
    n: u32,
}

fn parse_args(mut args: std::env::Args) -> Result<(String, Args)> {
    let cmd = args.next().context(USAGE)?;
    let mut parsed = Args {
        model: PathBuf::new(),
        pp: 512,
        tg: 128,
        ctx: 2048,
        ids: Vec::new(),
        n: 24,
    };
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .with_context(|| format!("flag {flag} needs a value"))
        };
        match flag.as_str() {
            "-m" | "--model" => parsed.model = PathBuf::from(value()?),
            "--pp" => parsed.pp = value()?.parse().context("--pp")?,
            "--tg" => parsed.tg = value()?.parse().context("--tg")?,
            "--ctx" => parsed.ctx = value()?.parse().context("--ctx")?,
            "-n" => parsed.n = value()?.parse().context("-n")?,
            "--ids" => {
                parsed.ids = value()?
                    .split(',')
                    .map(|s| s.trim().parse::<u32>().context("--ids"))
                    .collect::<Result<_>>()?;
            }
            other => bail!("unknown flag {other}\n{USAGE}"),
        }
    }
    if parsed.model.as_os_str().is_empty() {
        bail!("missing -m <gguf>\n{USAGE}");
    }
    Ok((cmd, parsed))
}

const USAGE: &str = "usage:\n  llama_ash bench -m <gguf> [--pp 512] [--tg 128] [--ctx 2048]\n  llama_ash generate -m <gguf> --ids \"1,15043,3186\" -n 24 [--ctx 2048]";

fn make_model(path: &Path, ctx_len: u32) -> Result<Model> {
    let ctx = VulkanContext::new_with_device_preference(false, DevicePreference::Auto)?;
    if !ctx.buffer_device_address_enabled || !ctx.f16_storage_enabled {
        bail!("device lacks buffer-device-address or f16 storage support");
    }
    let pipe = Arc::new(MatmulPipeline::new_with_kernel_selection(
        &ctx,
        KernelSelection::from_env()?,
    )?);
    let exec = Arc::new(Executor::new(ctx.clone(), pipe, 2, 256)?);
    log::info!("{}", ctx.diagnostics());
    Model::load(&ctx, &exec, path, ctx_len)
}

fn bench(args: &Args) -> Result<()> {
    let load_start = Instant::now();
    let mut model = make_model(&args.model, args.ctx)?;
    let load_s = load_start.elapsed().as_secs_f64();
    if args.pp + args.tg > model.cfg.t_max {
        bail!(
            "pp {} + tg {} exceeds context {}",
            args.pp,
            args.tg,
            model.cfg.t_max
        );
    }

    // Fixed synthetic prompt: BOS then arbitrary in-vocab ids.
    let prompt: Vec<u32> = std::iter::once(1)
        .chain((1..args.pp).map(|i| 100 + (i % 1000)))
        .collect();

    // Warmup: burn in clocks with an untimed prefill plus decode steps
    // (decode-sized dispatches warm that phase's pipelines), then reset.
    let (warm_id, warm_logits) = model.prefill(&prompt)?;
    if !warm_logits.iter().all(|v| v.is_finite()) {
        bail!("warmup prefill produced non-finite logits");
    }
    let mut warm_next = warm_id;
    for _ in 0..16 {
        (warm_next, _) = model.decode(warm_next)?;
    }
    log::info!("warmup prefill + decode done (argmax {warm_id}); resetting caches");
    model.reset()?;

    let t0 = Instant::now();
    let (mut next, _) = model.prefill(&prompt)?;
    let prefill_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for _ in 0..args.tg {
        (next, _) = model.decode(next)?;
    }
    let decode_s = t1.elapsed().as_secs_f64();

    // Per-op-class GPU-time breakdown of one decode step (perop mode
    // records per-dispatch GPU timestamps; graph mode cannot split
    // them, so run one instrumented step on the side).
    if std::env::var("LLAMA_ASH_BREAKDOWN").as_deref() == Ok("1") {
        model.breakdown.borrow_mut().clear();
        let _ = model.decode(next)?;
        let mut per_class: std::collections::BTreeMap<&'static str, (u64, u32)> =
            std::collections::BTreeMap::new();
        for (class, ns) in model.breakdown.borrow().iter() {
            let entry = per_class.entry(class).or_default();
            entry.0 += ns;
            entry.1 += 1;
        }
        let total: u64 = per_class.values().map(|v| v.0).sum();
        println!(
            "decode GPU breakdown (one token, {} dispatches):",
            model.breakdown.borrow().len()
        );
        let mut rows: Vec<_> = per_class.into_iter().collect();
        rows.sort_by_key(|(_, (ns, _))| std::cmp::Reverse(*ns));
        for (class, (ns, count)) in rows {
            println!(
                "  {class:<16} {:>8.1} us  ({count:>3} calls, {:>4.1}%)",
                ns as f64 / 1e3,
                ns as f64 * 100.0 / total as f64
            );
        }
        println!(
            "  {:<16} {:>8.1} us  (GPU-timestamped total)",
            "sum",
            total as f64 / 1e3
        );
    }

    let pp_tps = args.pp as f64 / prefill_s;
    let tg_tps = args.tg as f64 / decode_s;
    println!("model    : {}", args.model.display());
    println!("load     : {load_s:.2} s");
    println!("| test        | tokens | time (s) | tokens/s |");
    println!("|-------------|--------|----------|----------|");
    println!(
        "| pp{:<9} | {:>6} | {prefill_s:>8.3} | {pp_tps:>8.2} |",
        args.pp, args.pp
    );
    println!(
        "| tg{:<9} | {:>6} | {decode_s:>8.3} | {tg_tps:>8.2} |",
        args.tg, args.tg
    );
    Ok(())
}

fn generate(args: &Args) -> Result<()> {
    if args.ids.is_empty() {
        bail!("generate needs --ids \"1,...\"\n{USAGE}");
    }
    let load_start = Instant::now();
    let mut model = make_model(&args.model, args.ctx)?;
    let load_s = load_start.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let (first, logits) = model.prefill(&args.ids)?;
    let prefill_s = t0.elapsed().as_secs_f64();
    if !logits.iter().all(|v| v.is_finite()) {
        bail!("prefill produced non-finite logits");
    }

    let mut generated = vec![first];
    let t1 = Instant::now();
    while generated.len() < args.n as usize {
        let (next, _) = model.decode(*generated.last().unwrap())?;
        generated.push(next);
    }
    let decode_s = t1.elapsed().as_secs_f64();

    let ids: Vec<String> = generated.iter().map(u32::to_string).collect();
    println!("prompt_ids   : {}", join(&args.ids));
    println!("generated_ids: {}", ids.join(","));
    println!(
        "timings      : load {load_s:.2} s | prefill {} tok in {prefill_s:.3} s ({:.2} t/s) | decode {} tok in {decode_s:.3} s ({:.2} t/s)",
        args.ids.len(),
        args.ids.len() as f64 / prefill_s,
        generated.len().saturating_sub(1),
        generated.len().saturating_sub(1) as f64 / decode_s.max(1e-9),
    );
    Ok(())
}

fn join(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut args = std::env::args();
    args.next(); // program name
    let (cmd, parsed) = parse_args(args)?;
    match cmd.as_str() {
        "bench" => bench(&parsed),
        "generate" => generate(&parsed),
        other => bail!("unknown subcommand {other}\n{USAGE}"),
    }
}
