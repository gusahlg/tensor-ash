//! Benchmark binary for `ml_project`.
//!
//! Reports GPU-time TFLOPS (via vkCmdWriteTimestamp) — separating actual
//! kernel time from CPU-side submission overhead.
//!
//! Subcommands:
//!   self-check    Print runtime/toolchain diagnostics and selected Vulkan device.
//!   correctness   Quick correctness sanity check vs CPU reference.
//!   sweep         Throughput sweep across representative shapes.
//!   single        One configurable run: B,M,N,K from ML_B/M/N/K env vars.
//!   concurrent    Many parallel run_matmuls from N host threads.
//!   transfer      Host/device transfer bandwidth smoke benchmark.
//!
//! Without a subcommand, runs `correctness` then `sweep`.
//!
//! Env knobs:
//!   ML_B,ML_M,ML_N,ML_K  shape for `single`
//!   ML_ITERS             timing iterations (default 20)
//!   ML_WARMUP            warm-up iterations (default 3)
//!   ML_SLOTS             executor slot count (default 2)
//!   ML_THREADS           threads for `concurrent` (default = cpus)
//!   ML_VALIDATE          set 1 to enable Vulkan validation
//!   ML_DEVICE            auto, discrete, integrated, cpu, index:N, name:TEXT
//!   ML_KERNEL            auto, large, small
//!   ML_OUTPUT            table or csv
//!   ML_SWEEP             smoke, standard, full
//!   ML_TRANSFER_MB       transfer benchmark size in MiB (default 16)
//!
//! Run:
//!   cargo run --release --bin ml_bench -- sweep

use std::env;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use ml_project::{
    DeviceKind, DevicePreference, Executor, MatmulCall, MatmulPipeline, Tensor, VulkanContext,
};

// ---------- env helpers ----------
fn env_u32(k: &str, d: u32) -> u32 {
    env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_usize(k: &str, d: usize) -> usize {
    env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_bool(k: &str) -> bool {
    env::var(k)
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}
fn env_string(k: &str, d: &str) -> String {
    env::var(k).unwrap_or_else(|_| d.into())
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum OutputMode {
    Table,
    Csv,
}

impl OutputMode {
    fn from_env() -> Self {
        match env_string("ML_OUTPUT", "table")
            .to_ascii_lowercase()
            .as_str()
        {
            "csv" => Self::Csv,
            _ => Self::Table,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SweepMode {
    Smoke,
    Standard,
    Full,
}

impl SweepMode {
    fn from_env(default: Self) -> Self {
        match env_string("ML_SWEEP", "").to_ascii_lowercase().as_str() {
            "smoke" => Self::Smoke,
            "standard" => Self::Standard,
            "full" => Self::Full,
            _ => default,
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cmd = env::args().nth(1).unwrap_or_else(|| "all".into());
    let validate = env_bool("ML_VALIDATE");
    let n_slots = env_usize("ML_SLOTS", 2);
    let device_preference = DevicePreference::parse(&env_string("ML_DEVICE", "auto"))?;

    let ctx = VulkanContext::new_with_device_preference(validate, device_preference)?;
    let pipe = Arc::new(MatmulPipeline::new(&ctx)?);
    let exec = Executor::new(ctx.clone(), pipe, n_slots, /*max_calls=*/ 256)?;

    log::info!("{} slots={}", ctx.diagnostics(), n_slots);

    match cmd.as_str() {
        "self-check" => self_check(&ctx, n_slots)?,
        "correctness" => correctness(&ctx, &exec)?,
        "sweep" => sweep(&ctx, &exec)?,
        "single" => single(&ctx, &exec)?,
        "concurrent" => concurrent(ctx.clone(), Arc::new(exec))?,
        "transfer" => transfer(&ctx, &exec)?,
        "all" => {
            correctness(&ctx, &exec)?;
            sweep(&ctx, &exec)?;
        }
        _ => {
            log::warn!("unknown subcommand '{cmd}', running default correctness+sweep");
            correctness(&ctx, &exec)?;
            sweep(&ctx, &exec)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 1. Correctness — small, fast, prints max abs err.
// ---------------------------------------------------------------------------
fn correctness(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    const B: u32 = 2;
    const M: u32 = 64;
    const N: u32 = 80;
    const K: u32 = 48;

    let a = Tensor::zeros_device(ctx, &[B, M, K])?;
    let b = Tensor::zeros_device(ctx, &[B, K, N])?;
    let c = Tensor::zeros_device(ctx, &[B, M, N])?;

    let mut ha = vec![0.0f32; (B * M * K) as usize];
    let mut hb = vec![0.0f32; (B * K * N) as usize];
    fill_det(&mut ha, 1);
    fill_det(&mut hb, 2);
    exec.upload(&ha, &a)?;
    exec.upload(&hb, &b)?;

    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    }])?;

    let mut hc = vec![0.0f32; (B * M * N) as usize];
    exec.download(&c, &mut hc)?;

    let cpu = cpu_bmm(&ha, &hb, B, M, N, K);
    let (e, idx) = max_abs_err(&hc, &cpu);
    let tol = 8.0 * (K as f32) * f32::EPSILON;
    log::info!(
        "correctness: max|err|={e:.3e}  tol={tol:.3e}  \
         at idx {idx}: gpu={:.6}  cpu={:.6}",
        hc[idx],
        cpu[idx],
    );
    anyhow::ensure!(e <= tol, "correctness failed: err {e:.3e} > tol {tol:.3e}");
    println!("CORRECTNESS OK (err={e:.3e}, tol={tol:.3e})");
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. Sweep — representative shapes.  Reports best-of-N GPU TFLOPS and the
//    wall-clock vs GPU-time delta (= CPU submit overhead).
// ---------------------------------------------------------------------------
fn sweep(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let iters = env_u32("ML_ITERS", 20).max(1);
    let warmup = env_u32("ML_WARMUP", 3);
    let default_sweep = if ctx.device_kind() == DeviceKind::Cpu {
        SweepMode::Smoke
    } else {
        SweepMode::Standard
    };
    let sweep_mode = SweepMode::from_env(default_sweep);

    // Theoretical peak — RTX 3070: 5888 CUDA cores * 2 FMA * 1.725 GHz ≈ 20.3 TFLOPS FP32.
    // Allow override via env so other GPUs report something sensible.
    let peak = env::var("ML_PEAK_TFLOPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(20.3);
    let mut reporter = BenchReporter::new(OutputMode::from_env(), peak, ctx);
    reporter.print_header();

    for case in sweep_cases(sweep_mode) {
        let result = run_case(ctx, exec, *case, iters, warmup)?;
        reporter.print_case(&result);
    }
    Ok(())
}

fn run_case(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    case: BenchCase,
    iters: u32,
    warmup: u32,
) -> Result<BenchResult> {
    let BenchCase {
        label: _,
        b: bsz,
        m,
        n,
        k,
    } = case;
    let a = Tensor::zeros_device(ctx, &[bsz, m, k])?;
    let b = Tensor::zeros_device(ctx, &[bsz, k, n])?;
    let c = Tensor::zeros_device(ctx, &[bsz, m, n])?;
    // Upload non-zero (peak FP32 throughput is unaffected by data values,
    // but doing so guards against denormal pathologies).
    let mut h_a = vec![0.0f32; host_len(&[bsz, m, k])?];
    let mut h_b = vec![0.0f32; host_len(&[bsz, k, n])?];
    fill_det(&mut h_a, 7);
    fill_det(&mut h_b, 11);
    exec.upload(&h_a, &a)?;
    exec.upload(&h_b, &b)?;

    let flops = 2.0f64 * bsz as f64 * m as f64 * n as f64 * k as f64;

    for _ in 0..warmup {
        exec.run_matmuls(&[MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])?;
    }

    let mut best_gpu_ns = u64::MAX;
    let mut best_wall_ns = u128::MAX;
    for _ in 0..iters {
        let t0 = Instant::now();
        let stats = exec.run_matmuls(&[MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])?;
        let wall_ns = t0.elapsed().as_nanos();
        best_wall_ns = best_wall_ns.min(wall_ns);
        if let Some(gpu_ns) = stats.gpu_time_ns {
            best_gpu_ns = best_gpu_ns.min(gpu_ns);
        }
    }
    let gpu_ms = if best_gpu_ns != u64::MAX {
        best_gpu_ns as f64 / 1e6
    } else {
        f64::NAN
    };
    let wall_ms = best_wall_ns as f64 / 1e6;
    let tflops = if best_gpu_ns != u64::MAX {
        flops / best_gpu_ns as f64 * 1e-3
    } else {
        f64::NAN
    };
    Ok(BenchResult {
        case,
        flops,
        wall_ms,
        gpu_ms,
        tflops,
    })
}

// ---------------------------------------------------------------------------
// 3. Single — one configurable matmul.
// ---------------------------------------------------------------------------
fn single(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let b = env_u32("ML_B", 1);
    let m = env_u32("ML_M", 4096);
    let n = env_u32("ML_N", 4096);
    let k = env_u32("ML_K", 4096);
    let iters = env_u32("ML_ITERS", 20).max(1);
    let warmup = env_u32("ML_WARMUP", 3);
    let peak = env::var("ML_PEAK_TFLOPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(20.3);
    let label = Box::leak(format!("B={b} M={m} N={n} K={k}").into_boxed_str());
    let case = BenchCase { label, b, m, n, k };
    let mut reporter = BenchReporter::new(OutputMode::from_env(), peak, ctx);
    reporter.print_header();
    let result = run_case(ctx, exec, case, iters, warmup)?;
    reporter.print_case(&result);
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. Concurrent — many threads issuing run_matmuls in parallel.
// ---------------------------------------------------------------------------
fn concurrent(ctx: Arc<VulkanContext>, exec: Arc<Executor>) -> Result<()> {
    let n_threads = env_usize(
        "ML_THREADS",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
    );
    let iters = env_u32("ML_ITERS", 50).max(1);
    let (m, n, k) = (512u32, 512u32, 512u32);

    // Each thread owns its triple of tensors.
    let triples: Vec<(Tensor, Tensor, Tensor)> = (0..n_threads)
        .map(|t| {
            let a = Tensor::zeros_device(&ctx, &[m, k]).unwrap();
            let b = Tensor::zeros_device(&ctx, &[k, n]).unwrap();
            let c = Tensor::zeros_device(&ctx, &[m, n]).unwrap();
            let mut ha = vec![0.0f32; (m * k) as usize];
            let mut hb = vec![0.0f32; (k * n) as usize];
            fill_det(&mut ha, 100 + t as u64);
            fill_det(&mut hb, 200 + t as u64);
            exec.upload(&ha, &a).unwrap();
            exec.upload(&hb, &b).unwrap();
            (a, b, c)
        })
        .collect();
    let triples = Arc::new(triples);

    let flops_per_call = 2.0f64 * m as f64 * n as f64 * k as f64;
    let total_calls = iters as u64 * n_threads as u64;
    let total_flops = flops_per_call * total_calls as f64;

    let t0 = Instant::now();
    let mut handles = Vec::new();
    for t in 0..n_threads {
        let exec = exec.clone();
        let triples = triples.clone();
        handles.push(std::thread::spawn(move || -> Result<()> {
            let (a, b, c) = &triples[t];
            for _ in 0..iters {
                exec.run_matmuls(&[MatmulCall {
                    a,
                    b,
                    c,
                    alpha: 1.0,
                    accumulate: false,
                }])
                .context("run_matmuls")?;
            }
            Ok(())
        }));
    }
    for h in handles {
        h.join().unwrap()?;
    }
    let dt = t0.elapsed().as_secs_f64();
    let tflops = total_flops / dt * 1e-12;
    println!(
        "concurrent: {n_threads} threads × {iters} iters of 512^3   \
         total {:.1} GFLOPs  in {dt:.3}s  →  {tflops:.2} TFLOPS (aggregate)",
        total_flops * 1e-9,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 5. Transfer — upload/download bandwidth through Executor staging path.
// ---------------------------------------------------------------------------
fn transfer(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let mb = env_usize("ML_TRANSFER_MB", 16).max(1);
    let iters = env_u32("ML_ITERS", 10).max(1);
    let bytes = mb
        .checked_mul(1024 * 1024)
        .context("ML_TRANSFER_MB overflows usize")?;
    let n_f32 = bytes / std::mem::size_of::<f32>();
    anyhow::ensure!(
        n_f32 <= u32::MAX as usize,
        "transfer tensor is too large for current u32 shape storage"
    );

    let tensor = Tensor::zeros_device(ctx, &[n_f32 as u32])?;
    let src = vec![1.0f32; n_f32];
    let mut dst = vec![0.0f32; n_f32];

    exec.upload(&src, &tensor)?;
    exec.download(&tensor, &mut dst)?;

    let mut best_upload = f64::INFINITY;
    let mut best_download = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        exec.upload(&src, &tensor)?;
        best_upload = best_upload.min(t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        exec.download(&tensor, &mut dst)?;
        best_download = best_download.min(t0.elapsed().as_secs_f64());
    }

    let gib = bytes as f64 / 1024.0 / 1024.0 / 1024.0;
    let upload_gibs = gib / best_upload;
    let download_gibs = gib / best_download;
    match OutputMode::from_env() {
        OutputMode::Table => {
            println!();
            println!(
                "transfer: {mb} MiB x {iters} iters  upload={upload_gibs:.2} GiB/s  download={download_gibs:.2} GiB/s"
            );
        }
        OutputMode::Csv => {
            println!("device,kind,bytes,iters,upload_gibs,download_gibs");
            println!(
                "{},{},{},{},{:.6},{:.6}",
                csv_escape(ctx.device_name()),
                ctx.device_kind().as_str(),
                bytes,
                iters,
                upload_gibs,
                download_gibs,
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct BenchCase {
    label: &'static str,
    b: u32,
    m: u32,
    n: u32,
    k: u32,
}

struct BenchResult {
    case: BenchCase,
    flops: f64,
    wall_ms: f64,
    gpu_ms: f64,
    tflops: f64,
}

struct BenchReporter<'a> {
    mode: OutputMode,
    peak_tflops: f64,
    ctx: &'a VulkanContext,
}

impl<'a> BenchReporter<'a> {
    fn new(mode: OutputMode, peak_tflops: f64, ctx: &'a VulkanContext) -> Self {
        Self {
            mode,
            peak_tflops,
            ctx,
        }
    }

    fn print_header(&mut self) {
        match self.mode {
            OutputMode::Table => {
                println!();
                println!(
                    "{:<28} {:>14} {:>10} {:>10} {:>9} {:>9}",
                    "shape", "FLOPs", "wall(ms)", "gpu(ms)", "TF/s", "%peak",
                );
                println!("{}", "-".repeat(80));
            }
            OutputMode::Csv => {
                println!("device,kind,label,b,m,n,k,flops,wall_ms,gpu_ms,tflops,percent_peak");
            }
        }
    }

    fn print_case(&mut self, result: &BenchResult) {
        let pct = result.tflops / self.peak_tflops * 100.0;
        match self.mode {
            OutputMode::Table => println!(
                "{:<28} {:>14.3} {:>10.3} {:>10.3} {:>9.2} {:>8.1}%",
                result.case.label,
                result.flops / 1e9,
                result.wall_ms,
                result.gpu_ms,
                result.tflops,
                pct,
            ),
            OutputMode::Csv => println!(
                "{},{},{},{},{},{},{},{:.0},{:.6},{:.6},{:.6},{:.3}",
                csv_escape(self.ctx.device_name()),
                self.ctx.device_kind().as_str(),
                csv_escape(result.case.label),
                result.case.b,
                result.case.m,
                result.case.n,
                result.case.k,
                result.flops,
                result.wall_ms,
                result.gpu_ms,
                result.tflops,
                pct,
            ),
        }
    }
}

fn sweep_cases(mode: SweepMode) -> &'static [BenchCase] {
    const SMOKE: &[BenchCase] = &[
        BenchCase {
            label: "smoke 128^3 B=1",
            b: 1,
            m: 128,
            n: 128,
            k: 128,
        },
        BenchCase {
            label: "smoke 256^3 B=1",
            b: 1,
            m: 256,
            n: 256,
            k: 256,
        },
    ];
    const STANDARD: &[BenchCase] = &[
        BenchCase {
            label: "square 512^3   B=1",
            b: 1,
            m: 512,
            n: 512,
            k: 512,
        },
        BenchCase {
            label: "square 1024^3  B=1",
            b: 1,
            m: 1024,
            n: 1024,
            k: 1024,
        },
        BenchCase {
            label: "square 2048^3  B=1",
            b: 1,
            m: 2048,
            n: 2048,
            k: 2048,
        },
        BenchCase {
            label: "batched 8x1024^2",
            b: 8,
            m: 1024,
            n: 1024,
            k: 1024,
        },
        BenchCase {
            label: "tall   4096x1024x1024",
            b: 1,
            m: 4096,
            n: 1024,
            k: 1024,
        },
        BenchCase {
            label: "wide   1024x4096x1024",
            b: 1,
            m: 1024,
            n: 4096,
            k: 1024,
        },
        BenchCase {
            label: "odd   1023x1025x1027",
            b: 1,
            m: 1023,
            n: 1025,
            k: 1027,
        },
    ];
    const FULL: &[BenchCase] = &[
        BenchCase {
            label: "square 512^3   B=1",
            b: 1,
            m: 512,
            n: 512,
            k: 512,
        },
        BenchCase {
            label: "square 1024^3  B=1",
            b: 1,
            m: 1024,
            n: 1024,
            k: 1024,
        },
        BenchCase {
            label: "square 2048^3  B=1",
            b: 1,
            m: 2048,
            n: 2048,
            k: 2048,
        },
        BenchCase {
            label: "square 4096^3  B=1",
            b: 1,
            m: 4096,
            n: 4096,
            k: 4096,
        },
        BenchCase {
            label: "batched 32x1024^2",
            b: 32,
            m: 1024,
            n: 1024,
            k: 1024,
        },
        BenchCase {
            label: "batched 8x2048^2",
            b: 8,
            m: 2048,
            n: 2048,
            k: 2048,
        },
        BenchCase {
            label: "tall   8192x1024x1024",
            b: 1,
            m: 8192,
            n: 1024,
            k: 1024,
        },
        BenchCase {
            label: "wide   1024x8192x1024",
            b: 1,
            m: 1024,
            n: 8192,
            k: 1024,
        },
        BenchCase {
            label: "thin K 4096x4096x512",
            b: 1,
            m: 4096,
            n: 4096,
            k: 512,
        },
        BenchCase {
            label: "fat  K 1024x1024x8192",
            b: 1,
            m: 1024,
            n: 1024,
            k: 8192,
        },
        BenchCase {
            label: "odd   1023x1025x1027",
            b: 1,
            m: 1023,
            n: 1025,
            k: 1027,
        },
    ];
    match mode {
        SweepMode::Smoke => SMOKE,
        SweepMode::Standard => STANDARD,
        SweepMode::Full => FULL,
    }
}

fn self_check(ctx: &VulkanContext, n_slots: usize) -> Result<()> {
    println!("ml_project self-check");
    println!("status: OK");
    println!("selected: {}", ctx.diagnostics());
    println!("executor_slots: {n_slots}");
    println!("glslc: {}", command_path("glslc"));
    println!("vulkaninfo: {}", command_path("vulkaninfo"));
    println!(
        "LD_LIBRARY_PATH: {}",
        env::var("LD_LIBRARY_PATH").unwrap_or_else(|_| "<unset>".into()),
    );
    println!(
        "VK_ICD_FILENAMES: {}",
        env::var("VK_ICD_FILENAMES").unwrap_or_else(|_| "<unset>".into()),
    );
    if ctx.device_kind() == DeviceKind::Cpu {
        println!(
            "warning: selected device is CPU/software Vulkan; benchmark results are not useful for GPU tuning"
        );
    }
    Ok(())
}

fn command_path(command: &str) -> String {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {command}"))
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "<missing>".into())
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.into()
    }
}

fn host_len(shape: &[u32]) -> Result<usize> {
    let numel = Tensor::numel_checked(shape)?;
    usize::try_from(numel).context("tensor element count does not fit usize")
}

fn fill_det(out: &mut [f32], seed: u64) {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    for v in out.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (s >> 40) as u32;
        *v = (bits as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
    }
}

fn cpu_bmm(a: &[f32], b: &[f32], bsz: u32, m: u32, n: u32, k: u32) -> Vec<f32> {
    let m_ = m as usize;
    let n_ = n as usize;
    let k_ = k as usize;
    let mk = m_ * k_;
    let kn = k_ * n_;
    let mn = m_ * n_;
    let mut out = vec![0.0f32; bsz as usize * mn];
    for bi in 0..bsz as usize {
        let a = &a[bi * mk..(bi + 1) * mk];
        let b = &b[bi * kn..(bi + 1) * kn];
        let c = &mut out[bi * mn..(bi + 1) * mn];
        for i in 0..m_ {
            for j in 0..n_ {
                let mut acc = 0.0f64;
                for kk in 0..k_ {
                    acc += (a[i * k_ + kk] as f64) * (b[kk * n_ + j] as f64);
                }
                c[i * n_ + j] = acc as f32;
            }
        }
    }
    out
}

fn max_abs_err(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut m = 0.0f32;
    let mut idx = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let e = (x - y).abs();
        if e > m {
            m = e;
            idx = i;
        }
    }
    (m, idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_modes_choose_expected_case_sets() {
        assert_eq!(sweep_cases(SweepMode::Smoke).len(), 2);
        assert_eq!(sweep_cases(SweepMode::Standard).len(), 7);
        assert_eq!(sweep_cases(SweepMode::Full).len(), 11);
    }

    #[test]
    fn csv_escape_only_quotes_when_needed() {
        assert_eq!(csv_escape("RTX 3070"), "RTX 3070");
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
        assert_eq!(csv_escape("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn host_len_rejects_bad_shapes() {
        assert!(host_len(&[]).is_err());
        assert!(host_len(&[1, 0, 4]).is_err());
    }

    #[test]
    fn max_abs_err_reports_largest_difference() {
        let (err, idx) = max_abs_err(&[1.0, 3.0, -5.0], &[1.5, 1.0, -4.0]);

        assert_eq!(err, 2.0);
        assert_eq!(idx, 1);
    }
}
