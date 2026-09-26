//! `ml_bench isa` — make the GPU assembly of routed kernels visible.
//!
//! For each `label,b,m,n,k[,f16w][,packed]` spec (the `cases` syntax):
//!
//! 1. route it exactly like a dispatch would (`Executor::dispatch_info_for`
//!    / `dispatch_info_packed`, so a tuned winner or `ML_KERNEL=` is
//!    honored) and derive the bound specialization (`KernelVariant::
//!    for_dispatch`, zero epilogue, `C = A@B`);
//! 2. compile that pipeline through the library's own creation path
//!    (`MatmulPipeline::create_probe_pipeline`) into a FRESH, empty
//!    `VkPipelineCache` — pipeline creation dispatches no GPU work;
//! 3. `vkGetPipelineCacheData` -> NVIDIA cache blob -> zstd -> the
//!    `NVuc` ISA container -> exact UCODE bytes (`nvcache.rs`);
//! 4. when `VK_KHR_pipeline_executable_properties` is available
//!    (`ML_PIPELINE_EXEC_INFO=1` is set automatically), recompile with
//!    `CAPTURE_STATISTICS | CAPTURE_INTERNAL_REPRESENTATIONS`, print
//!    every statistic and internal representation the driver reports,
//!    and check the capture flags did not change the code;
//! 5. if `nvdisasm` is available, disassemble and run the census
//!    (`census.rs`), writing `<stem>.sass` / `<stem>.census.txt` and a
//!    markdown summary row.
//!
//! `ml_bench isa-analyze <dir|file.meta>...` re-runs step 5 offline (no
//! GPU, no Vulkan) on artifacts copied from a host without nvdisasm.
//!
//! Environment: `ML_ISA_OUT` (output dir, default `isa-out`),
//! `ML_ISA_ARCH` (nvdisasm `-b` value, default derived from the device
//! name: RTX 30xx -> SM86, RTX 40xx -> SM89), `ML_ISA_NVDISASM` (path,
//! or `off`), `ML_ISA_SPV` (compile this SPIR-V file instead of the
//! embedded module — same interface required; e.g. a `glslc -Os`
//! build), `ML_ISA_TAG` (suffix for the output stem).

mod census;
mod nvcache;
mod occupancy;

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use ash::vk;
use tensor_ash::{
    EpilogueKey, Executor, KERNEL_SPECS, KernelVariant, MatmulPipeline, VulkanContext,
};

use super::cases::BenchCase;
use census::{Census, Mix};
use nvcache::{NvucContainer, find_containers, parse_cache_blob, parse_container};
use occupancy::{SmLimits, occupancy};

/// `isa` subcommand.
pub(super) fn isa(
    ctx: &Arc<VulkanContext>,
    pipe: &MatmulPipeline,
    exec: &Executor,
    specs: impl IntoIterator<Item = String>,
) -> Result<()> {
    let cases = specs
        .into_iter()
        .map(|s| BenchCase::parse(&s))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !cases.is_empty(),
        "isa requires at least one label,b,m,n,k[,f16w][,packed] argument"
    );
    let out_dir = PathBuf::from(std::env::var("ML_ISA_OUT").unwrap_or_else(|_| "isa-out".into()));
    std::fs::create_dir_all(&out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let arch = std::env::var("ML_ISA_ARCH")
        .ok()
        .or_else(|| arch_for_device(&ctx.device_summary.name))
        .context("cannot derive the SASS arch from the device name; set ML_ISA_ARCH=SM86|SM89")?;
    let spv_override = match std::env::var("ML_ISA_SPV") {
        Ok(path) => Some((
            std::fs::read(&path).with_context(|| format!("read ML_ISA_SPV={path}"))?,
            path,
        )),
        Err(_) => None,
    };
    let tag = std::env::var("ML_ISA_TAG").ok();
    let nvdisasm = nvdisasm_path();
    if !ctx.pipeline_executable_info_enabled {
        log::warn!(
            "isa: VK_KHR_pipeline_executable_properties unavailable; driver statistics skipped"
        );
    }
    let exec_props = ctx
        .pipeline_executable_info_enabled
        .then(|| ash::khr::pipeline_executable_properties::Device::new(&ctx.instance, &ctx.device));
    let mut prod_codes: Option<HashSet<u64>> = None;
    let mut rows = Vec::new();

    for case in cases {
        let info = if case.packed_b {
            exec.dispatch_info_packed(case.n, case.k)
        } else {
            exec.dispatch_info_for(case.b, case.m, case.n, case.k, case.b_f16)
        };
        if let Some(splits) = info.split_k2_splits {
            bail!(
                "{}: routes to two-stage split-K ({} x{splits}); its stage-1/reduce \
                 pipelines are outside the matmul catalog and not probed",
                case.label,
                info.kernel
            );
        }
        let index = pipe
            .kernel_index(info.kernel)
            .with_context(|| format!("routed kernel {} has no built slot", info.kernel))?;
        let spec = &KERNEL_SPECS[index];
        let variant = KernelVariant::for_dispatch(info.tile, case.m, case.n, case.k, false, 1.0);
        let epilogue = EpilogueKey::default();
        let spec_constants = pipe.pipeline_spec_constants(index, variant, epilogue);
        let spv: &[u8] = spv_override
            .as_ref()
            .map_or(spec.spv, |(b, _)| b.as_slice());
        let local_size = spirv_local_size(spv).unwrap_or([0, 0, 0]);
        let threads = local_size.iter().product::<u32>();
        let stem = format!(
            "{}.{}.v{}.{}{}",
            case.label,
            spec.name,
            variant.index(),
            arch.to_ascii_lowercase(),
            tag.as_deref().map(|t| format!(".{t}")).unwrap_or_default()
        );
        log::info!(
            "isa {}: kernel {} tile {:?} variant {:?} spec {:?} local_size {:?}",
            case.label,
            spec.name,
            info.tile,
            variant,
            spec_constants,
            local_size
        );

        // (a) the production-equivalent compile: no capture flags.
        let spv_arg = spv_override.as_ref().map(|(b, _)| b.as_slice());
        let container = compile_to_container(
            ctx,
            pipe,
            index,
            variant,
            epilogue,
            vk::PipelineCreateFlags::empty(),
            spv_arg,
            None,
        )?
        .0;
        let ucode = container.ucode().context("NVuc container without UCODE")?;
        let (regs, regs_cap) = container.registers().unwrap_or((0, 0));

        // (b) with capture flags: driver statistics + representations.
        let mut stats_text = String::new();
        let mut meta_stats = Vec::new();
        let mut capture_same = "n/a".to_string();
        if let Some(props) = exec_props.as_ref() {
            let flags = vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR
                | vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR;
            let (captured, report) = compile_to_container(
                ctx,
                pipe,
                index,
                variant,
                epilogue,
                flags,
                spv_arg,
                Some(props),
            )?;
            capture_same = if captured.ucode() == Some(ucode) {
                "identical".into()
            } else {
                "DIFFERENT".into()
            };
            let report = report.expect("exec props requested");
            stats_text = report.text;
            meta_stats = report.stats;
            for (i, (name, data)) in report.irs.iter().enumerate() {
                let path = out_dir.join(format!("{stem}.ir{i}.{}", sanitize(name)));
                std::fs::write(&path, data)?;
                let _ = writeln!(stats_text, "  wrote IR '{name}' -> {}", path.display());
            }
        }

        // Is this exactly the code the dispatcher's persistent cache holds?
        let codes = prod_codes.get_or_insert_with(|| production_codes(ctx));
        let in_prod = if codes.is_empty() {
            "n/a"
        } else if codes.contains(&hash(ucode)) {
            "yes"
        } else {
            "no"
        };

        let mut meta = String::new();
        let mut kv = |k: &str, v: String| {
            let _ = writeln!(meta, "{k} = {v}");
        };
        kv("label", case.label.to_string());
        kv(
            "shape",
            format!("b={} m={} n={} k={}", case.b, case.m, case.n, case.k),
        );
        kv("b_f16", case.b_f16.to_string());
        kv("packed_b", case.packed_b.to_string());
        kv("device", ctx.device_summary.name.clone());
        kv(
            "driver",
            format!("0x{:08x}", ctx.device_summary.driver_version),
        );
        kv("arch", arch.clone());
        kv("kernel", spec.name.to_string());
        kv("tile", format!("{:?}", info.tile));
        kv("variant", format!("{variant:?}"));
        kv("variant_index", variant.index().to_string());
        kv("spec_constants", format!("{spec_constants:?}"));
        kv(
            "spirv",
            spv_override
                .as_ref()
                .map_or("embedded".into(), |(_, p)| p.clone()),
        );
        kv("spirv_bytes", spv.len().to_string());
        kv("spirv_hash", format!("{:016x}", hash(spv)));
        kv("local_size", format!("{local_size:?}"));
        kv("threads", threads.to_string());
        kv("shared_bytes", spec.shared_memory_bytes().to_string());
        kv("registers", regs.to_string());
        kv("registers_cap", regs_cap.to_string());
        kv(
            "barriers",
            container.barrier_count().unwrap_or(0).to_string(),
        );
        kv("symbol", container.symbol().unwrap_or_default());
        kv("ucode_bytes", ucode.len().to_string());
        kv("ucode_hash", format!("{:016x}", hash(ucode)));
        kv("capture_flags_code", capture_same.clone());
        kv("in_production_cache", in_prod.to_string());
        for (name, value) in &meta_stats {
            kv(&format!("stat.{}", sanitize(name)), value.clone());
        }
        std::fs::write(out_dir.join(format!("{stem}.meta")), &meta)?;
        std::fs::write(out_dir.join(format!("{stem}.nvuc")), &container.bytes)?;
        std::fs::write(out_dir.join(format!("{stem}.ucode.bin")), ucode)?;
        let sections = container.describe();
        std::fs::write(
            out_dir.join(format!("{stem}.stats.txt")),
            format!("{sections}\n{stats_text}"),
        )?;
        println!("== {stem}\n{meta}{sections}{stats_text}");

        match nvdisasm.as_deref() {
            Some(tool) => rows.push(analyze(tool, &out_dir, &stem)?),
            None => println!(
                "(nvdisasm unavailable: copy {}/{stem}.* to a host with the CUDA toolkit and \
                 run `ml_bench isa-analyze`)",
                out_dir.display()
            ),
        }
    }
    print_rows(&rows);
    Ok(())
}

/// `isa-cache <pipeline_cache.bin> [out_dir]`: split a persisted
/// NVIDIA pipeline cache (e.g. `~/.cache/tensor-ash/pipeline_cache_*.bin`)
/// into one `.nvuc` + `.meta` per entry and census each (offline).  The
/// kernel identity of an entry is unknown here (the cache stores no
/// SPIR-V), so occupancy needs `isa` instead.
pub(super) fn isa_cache(mut args: impl Iterator<Item = String>) -> Result<()> {
    let path = args
        .next()
        .context("isa-cache <pipeline_cache.bin> [out_dir]")?;
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| "isa-cache-out".into()));
    std::fs::create_dir_all(&out_dir)?;
    let blob = std::fs::read(&path).with_context(|| format!("read {path}"))?;
    let arch = std::env::var("ML_ISA_ARCH")
        .ok()
        .or_else(|| nvcache::cache_device_id(&blob).and_then(arch_for_device_id))
        .context("unknown device id in cache header; set ML_ISA_ARCH")?;
    let entries = parse_cache_blob(&blob)?;
    let nvdisasm = nvdisasm_path();
    let mut rows = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        for (j, c) in find_containers(&e.payload)?.iter().enumerate() {
            let stem = format!("entry{i:03}.{j}");
            let (regs, _) = c.registers().unwrap_or((0, 0));
            let meta = format!(
                "label = {stem}\narch = {arch}\nkernel = ?\nregisters = {regs}\n\
                 barriers = {}\nucode_bytes = {}\nucode_hash = {:016x}\n",
                c.barrier_count().unwrap_or(0),
                c.ucode().map_or(0, <[u8]>::len),
                c.ucode().map_or(0, hash),
            );
            std::fs::write(out_dir.join(format!("{stem}.meta")), meta)?;
            std::fs::write(out_dir.join(format!("{stem}.nvuc")), &c.bytes)?;
            if let Some(tool) = nvdisasm.as_deref() {
                rows.push(analyze(tool, &out_dir, &stem)?);
            }
        }
    }
    println!("{} entries -> {}", entries.len(), out_dir.display());
    print_rows(&rows);
    Ok(())
}

/// `isa-analyze` subcommand: census of previously extracted artifacts.
pub(super) fn isa_analyze(args: impl IntoIterator<Item = String>) -> Result<()> {
    let tool = nvdisasm_path().context("isa-analyze needs nvdisasm (CUDA toolkit) on PATH")?;
    let mut rows = Vec::new();
    for arg in args {
        let path = PathBuf::from(&arg);
        let metas: Vec<PathBuf> = if path.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(&path)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "meta"))
                .collect();
            v.sort();
            v
        } else {
            vec![path]
        };
        for meta in metas {
            let dir = meta.parent().unwrap_or(Path::new(".")).to_path_buf();
            let stem = meta
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".meta"))
                .with_context(|| format!("{} is not a .meta file", meta.display()))?
                .to_string();
            rows.push(analyze(&tool, &dir, &stem)?);
        }
    }
    ensure!(!rows.is_empty(), "isa-analyze: no .meta files given");
    print_rows(&rows);
    Ok(())
}

/// Disassemble `<dir>/<stem>.nvuc` and write the census.  Returns the
/// markdown summary row.
fn analyze(nvdisasm: &str, dir: &Path, stem: &str) -> Result<String> {
    let meta = std::fs::read_to_string(dir.join(format!("{stem}.meta")))?;
    let get = |k: &str| {
        meta.lines()
            .find_map(|l| l.split_once(" = ").filter(|(key, _)| *key == k))
            .map(|(_, v)| v.to_string())
    };
    let arch = std::env::var("ML_ISA_ARCH")
        .ok()
        .or_else(|| get("arch"))
        .context("no arch in .meta and ML_ISA_ARCH unset")?;
    let container = parse_container(&std::fs::read(dir.join(format!("{stem}.nvuc")))?)?;
    let ucode = container.ucode().context("no UCODE section")?;
    let bin = dir.join(format!("{stem}.ucode.bin"));
    std::fs::write(&bin, ucode)?;
    let out = Command::new(nvdisasm)
        .args(["-b", &arch])
        .arg(&bin)
        .output()
        .with_context(|| format!("run {nvdisasm}"))?;
    ensure!(
        out.status.success(),
        "nvdisasm failed on {}: {}",
        bin.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout)?;
    std::fs::write(dir.join(format!("{stem}.sass")), &text)?;
    let census = Census::new(census::parse_sass(&text, ucode)?);

    let regs: u32 = get("registers").and_then(|v| v.parse().ok()).unwrap_or(0);
    let threads: u32 = get("threads").and_then(|v| v.parse().ok()).unwrap_or(0);
    let smem: u32 = get("shared_bytes")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let limits = SmLimits::for_arch(&arch);
    let occ = limits
        .filter(|_| threads > 0 && regs > 0)
        .map(|l| (occupancy(&l, threads, regs, smem), l));

    let whole = census.whole();
    let mut rep = String::new();
    let _ = writeln!(rep, "# ISA census: {stem}\n");
    for k in [
        "label",
        "shape",
        "device",
        "arch",
        "kernel",
        "variant",
        "spec_constants",
        "spirv",
        "local_size",
        "shared_bytes",
        "registers",
        "barriers",
        "capture_flags_code",
        "in_production_cache",
    ] {
        let _ = writeln!(rep, "{k:<20} {}", get(k).unwrap_or_default());
    }
    for line in meta.lines().filter(|l| l.starts_with("stat.")) {
        let _ = writeln!(rep, "{line}");
    }
    if let Some((o, l)) = occ {
        let _ = writeln!(
            rep,
            "occupancy            {} blocks/SM = {} warps/SM ({:.1}% of {}), limiter {} \
             (regs {} / smem {} / threads {} / block cap {}) [{} limits, max carveout]",
            o.blocks_per_sm,
            o.warps_per_sm,
            o.percent(&l),
            l.max_warps,
            o.limiter,
            o.by_regs,
            if o.by_smem == u32::MAX {
                "-".to_string()
            } else {
                o.by_smem.to_string()
            },
            o.by_threads,
            l.max_blocks,
            l.arch
        );
    }
    let [var, fixed, nops] = census.sanity();
    let _ = writeln!(
        rep,
        "control-bit sanity   var-latency writers with W-scoreboard {}/{}; fixed-latency \
         ALU with W-scoreboard {}/{}; NOP pad == B------:R-:W-:Y:S00 {}/{}",
        var.0, var.1, fixed.0, fixed.1, nops.0, nops.1
    );
    let _ = writeln!(
        rep,
        "\n## Whole kernel ({} live instructions, {} incl. padding)\n",
        census.live,
        census.instrs.len()
    );
    write_mix(&mut rep, &whole);
    let _ = writeln!(rep, "\n## Loops (backward branches)\n");
    for (i, l) in census.loops.iter().enumerate() {
        let m = census.loop_mix(l);
        let _ = writeln!(
            rep,
            "loop {i}: 0x{:04x}..0x{:04x} {} instrs{} math {} ({:.1}%) stall-sum {} \
             (excess {}) waits {}",
            l.start,
            l.end,
            m.total,
            if l.innermost { " innermost" } else { "" },
            m.math(),
            100.0 * m.math_density(),
            m.stall_sum,
            m.stall_sum as i64 - m.total as i64,
            m.waiting
        );
    }
    let main = census.main_loop.map(|i| &census.loops[i]);
    let main_mix = main.map(|l| census.loop_mix(l));
    if let (Some(l), Some(m)) = (main, main_mix.as_ref()) {
        let _ = writeln!(
            rep,
            "\n## Main loop 0x{:04x}..0x{:04x} (one iteration)\n",
            l.start, l.end
        );
        write_mix(&mut rep, m);
        let _ = writeln!(
            rep,
            "\nstatic issue cycles/iteration (sum of stall fields) {}; instructions {}; \
             dependency stalls (sum - n) {}; instrs waiting on a scoreboard {}; \
             yield hints {}; reuse-flagged {}",
            m.stall_sum,
            m.total,
            m.stall_sum as i64 - m.total as i64,
            m.waiting,
            m.yields,
            m.reuse_flagged
        );
        // Pipe-time lower bounds for one warp's iteration on one SM
        // sub-partition: 1 issue/clk; FP32 = 32 lanes/clk (one warp FFMA
        // per clk on GA10x/AD10x); tensor = 64 f16-in/f32-acc FMA/clk
        // (GeForce GA10x/AD10x rate, half the f16-acc rate).
        let _ = writeln!(
            rep,
            "pipe lower bounds (cycles/iter/warp on one SMSP): issue {} | FP32 {} | tensor {} \
             (HMMA FMAs/warp {} / 64)",
            m.total,
            m.count("FFMA") + m.class("fp32-other"),
            m.mma_fmas / 64,
            m.mma_fmas
        );
        let _ = writeln!(rep, "\n```\n{}```", census.loop_listing(l));
    }
    std::fs::write(dir.join(format!("{stem}.census.txt")), &rep)?;
    println!("{rep}");

    let occ_s = occ
        .map(|(o, l)| format!("{}w {:.0}% ({})", o.warps_per_sm, o.percent(&l), o.limiter))
        .unwrap_or_else(|| "?".into());
    let (lp_n, lp_math, lp_stall) = main_mix
        .as_ref()
        .map(|m| {
            (
                m.total.to_string(),
                format!("{:.0}%", 100.0 * m.math_density()),
                m.stall_sum.to_string(),
            )
        })
        .unwrap_or_else(|| ("-".into(), "-".into(), "-".into()));
    Ok(format!(
        "| {} | {} | v{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} | {} |",
        get("label").unwrap_or_default(),
        get("kernel").unwrap_or_default(),
        get("variant_index").unwrap_or_default(),
        regs,
        threads,
        smem,
        occ_s,
        whole.total,
        whole.count("FFMA"),
        whole.count("HMMA"),
        whole.widths("LDG"),
        whole.widths("STG"),
        whole.widths("LDS"),
        whole.widths("STS"),
        whole.count("LDSM"),
        whole.count("BAR"),
        whole.count("LDL"),
        whole.count("STL"),
        lp_n,
        lp_math,
        lp_stall,
    ))
}

fn write_mix(rep: &mut String, m: &Mix) {
    let _ = writeln!(rep, "classes: {}", {
        let mut v: Vec<_> = m.by_class.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        v.iter()
            .map(|(k, n)| format!("{k} {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    });
    let _ = writeln!(rep, "opcodes: {}", {
        let mut v: Vec<_> = m.by_base.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        v.iter()
            .map(|(k, n)| format!("{k} {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    });
    let _ = writeln!(
        rep,
        "memory (count x bits): LDG {} | STG {} | LDS {} | STS {} | LDSM {} | LDL {} | STL {}",
        m.widths("LDG"),
        m.widths("STG"),
        m.widths("LDS"),
        m.widths("STS"),
        m.widths("LDSM"),
        m.widths("LDL"),
        m.widths("STL"),
    );
    let _ = writeln!(
        rep,
        "bytes/thread: global load {} store {}; shared load {} (+LDSM {}) store {}",
        m.bytes("LDG"),
        m.bytes("STG"),
        m.bytes("LDS"),
        m.bytes("LDSM"),
        m.bytes("STS"),
    );
    let _ = writeln!(
        rep,
        "math FFMA {} HMMA {} HFMA2 {} -> density {:.1}%; int overhead (int+uniform) {} ({:.1}%); \
         BAR {} BRA {} DEPBAR {} S2R/CS2R/S2UR {}",
        m.count("FFMA"),
        m.count("HMMA"),
        m.count("HFMA2"),
        100.0 * m.math_density(),
        m.int_overhead(),
        100.0 * m.int_overhead() as f64 / m.total.max(1) as f64,
        m.count("BAR"),
        m.count("BRA"),
        m.count("DEPBAR"),
        m.class("special-reg"),
    );
}

fn print_rows(rows: &[String]) {
    if rows.is_empty() {
        return;
    }
    println!(
        "| case | kernel | var | regs | thr | smem B | occupancy | instrs | FFMA | HMMA | LDG | STG | \
         LDS | STS | LDSM | BAR | LDL/STL | loop instrs | loop math | loop stall |"
    );
    println!("|{}", "---|".repeat(20));
    for r in rows {
        println!("{r}");
    }
}

struct ExecReport {
    text: String,
    stats: Vec<(String, String)>,
    irs: Vec<(String, Vec<u8>)>,
}

/// Compile one probe pipeline into a fresh cache and pull its single
/// NVuc container out of the cache data; with `props`, also collect
/// executable statistics / internal representations.
#[allow(clippy::too_many_arguments)]
fn compile_to_container(
    ctx: &VulkanContext,
    pipe: &MatmulPipeline,
    index: usize,
    variant: KernelVariant,
    epilogue: EpilogueKey,
    flags: vk::PipelineCreateFlags,
    spv: Option<&[u8]>,
    props: Option<&ash::khr::pipeline_executable_properties::Device>,
) -> Result<(NvucContainer, Option<ExecReport>)> {
    let device = &ctx.device;
    let cache =
        unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }
            .context("create_pipeline_cache (probe)")?;
    let cache = scopeguard::guard(cache, |c| unsafe { device.destroy_pipeline_cache(c, None) });
    let pipeline = pipe.create_probe_pipeline(index, variant, epilogue, *cache, flags, spv)?;
    let pipeline = scopeguard::guard(pipeline, |p| unsafe { device.destroy_pipeline(p, None) });
    let report = match props {
        Some(props) => Some(unsafe { exec_report(props, *pipeline)? }),
        None => None,
    };
    let blob =
        unsafe { device.get_pipeline_cache_data(*cache) }.context("get_pipeline_cache_data")?;
    let entries = parse_cache_blob(&blob)?;
    let mut containers = Vec::new();
    for e in &entries {
        containers.extend(find_containers(&e.payload)?);
    }
    ensure!(
        containers.len() == 1,
        "fresh probe cache holds {} NVuc containers in {} entries (expected exactly 1)",
        containers.len(),
        entries.len()
    );
    Ok((containers.pop().expect("one container"), report))
}

unsafe fn exec_report(
    props: &ash::khr::pipeline_executable_properties::Device,
    pipeline: vk::Pipeline,
) -> Result<ExecReport> {
    let mut text = String::new();
    let mut stats = Vec::new();
    let mut irs = Vec::new();
    let executables = unsafe {
        props.get_pipeline_executable_properties(&vk::PipelineInfoKHR::default().pipeline(pipeline))
    }
    .context("vkGetPipelineExecutablePropertiesKHR")?;
    for (i, e) in executables.iter().enumerate() {
        let name = cstr(&e.name);
        let _ = writeln!(
            text,
            "executable {i}: '{name}' — {} (stages {:?}, subgroup {})",
            cstr(&e.description),
            e.stages,
            e.subgroup_size
        );
        let info = vk::PipelineExecutableInfoKHR::default()
            .pipeline(pipeline)
            .executable_index(i as u32);
        for s in unsafe { props.get_pipeline_executable_statistics(&info) }
            .context("vkGetPipelineExecutableStatisticsKHR")?
        {
            let value = unsafe {
                match s.format {
                    vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
                        (s.value.b32 != 0).to_string()
                    }
                    vk::PipelineExecutableStatisticFormatKHR::INT64 => s.value.i64.to_string(),
                    vk::PipelineExecutableStatisticFormatKHR::UINT64 => s.value.u64.to_string(),
                    vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => s.value.f64.to_string(),
                    other => format!("<format {other:?}>"),
                }
            };
            let sname = cstr(&s.name);
            let _ = writeln!(
                text,
                "  stat {:<32} = {:<12} ({})",
                sname,
                value,
                cstr(&s.description)
            );
            stats.push((sname, value));
        }
        // Internal representations: two-call pattern with data buffers.
        let mut reps = unsafe { props.get_pipeline_executable_internal_representations(&info) }
            .context("vkGetPipelineExecutableInternalRepresentationsKHR (sizes)")?;
        let _ = writeln!(text, "  internal representations: {}", reps.len());
        if !reps.is_empty() {
            let mut buffers: Vec<Vec<u8>> = reps.iter().map(|r| vec![0u8; r.data_size]).collect();
            for (r, b) in reps.iter_mut().zip(buffers.iter_mut()) {
                r.p_data = b.as_mut_ptr().cast();
            }
            let mut count = reps.len() as u32;
            let result = unsafe {
                (props
                    .fp()
                    .get_pipeline_executable_internal_representations_khr)(
                    props.device(),
                    &info,
                    &mut count,
                    reps.as_mut_ptr(),
                )
            };
            ensure!(
                result == vk::Result::SUCCESS || result == vk::Result::INCOMPLETE,
                "vkGetPipelineExecutableInternalRepresentationsKHR (data): {result:?}"
            );
            for (r, mut b) in reps.iter().zip(buffers) {
                b.truncate(r.data_size);
                if r.is_text != 0 {
                    while b.last() == Some(&0) {
                        b.pop();
                    }
                }
                let rname = cstr(&r.name);
                let _ = writeln!(
                    text,
                    "  IR '{rname}' ({}; {} B, text={})",
                    cstr(&r.description),
                    r.data_size,
                    r.is_text != 0
                );
                irs.push((format!("e{i}.{rname}"), b));
            }
        }
    }
    Ok(ExecReport { text, stats, irs })
}

/// Hashes of every UCODE in the context's persistent (production)
/// pipeline cache; empty if it cannot be read or decoded.
fn production_codes(ctx: &VulkanContext) -> HashSet<u64> {
    let blob = match unsafe { ctx.device.get_pipeline_cache_data(ctx.pipeline_cache) } {
        Ok(blob) => blob,
        Err(_) => return HashSet::new(),
    };
    let Ok(entries) = parse_cache_blob(&blob) else {
        return HashSet::new();
    };
    entries
        .iter()
        .filter_map(|e| find_containers(&e.payload).ok())
        .flatten()
        .filter_map(|c| c.ucode().map(hash))
        .collect()
}

/// nvdisasm `-b` value for a device name (NVIDIA GeForce naming).
fn arch_for_device(name: &str) -> Option<String> {
    let name = name.to_ascii_uppercase();
    if name.contains("RTX 30") || name.contains("RTX A") {
        Some("SM86".into())
    } else if name.contains("RTX 40") || name.contains(" L4") || name.contains("ADA") {
        Some("SM89".into())
    } else {
        None
    }
}

/// nvdisasm `-b` value for an NVIDIA PCI device id (GA10x: 0x22xx-0x25xx,
/// AD10x: 0x26xx-0x28xx).
fn arch_for_device_id(id: u32) -> Option<String> {
    match id {
        0x2200..=0x25ff => Some("SM86".into()),
        0x2600..=0x28ff => Some("SM89".into()),
        _ => None,
    }
}

fn nvdisasm_path() -> Option<String> {
    match std::env::var("ML_ISA_NVDISASM") {
        Ok(v) if v == "off" => None,
        Ok(v) => Some(v),
        Err(_) => Command::new("nvdisasm")
            .arg("--version")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|_| "nvdisasm".to_string()),
    }
}

/// `LocalSize` (or `LocalSizeId` with plain/spec-default constants) of
/// the first entry point.
fn spirv_local_size(spv: &[u8]) -> Option<[u32; 3]> {
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if words.first() != Some(&0x0723_0203) {
        return None;
    }
    let mut constants = std::collections::HashMap::new();
    let mut size_ids = None;
    let mut pc = 5;
    while pc < words.len() {
        let wc = (words[pc] >> 16) as usize;
        let op = words[pc] & 0xffff;
        if wc == 0 || pc + wc > words.len() {
            return None;
        }
        let o = &words[pc + 1..pc + wc];
        match op {
            // OpExecutionMode %entry LocalSize x y z
            16 if o.len() >= 5 && o[1] == 17 => return Some([o[2], o[3], o[4]]),
            // OpExecutionModeId %entry LocalSizeId %x %y %z
            331 if o.len() >= 5 && o[1] == 38 => size_ids = Some([o[2], o[3], o[4]]),
            // OpConstant / OpSpecConstant: [type, id, value]
            43 | 50 if o.len() >= 3 => {
                constants.insert(o[1], o[2]);
            }
            _ => {}
        }
        pc += wc;
    }
    let ids = size_ids?;
    Some([
        *constants.get(&ids[0])?,
        *constants.get(&ids[1])?,
        *constants.get(&ids[2])?,
    ])
}

fn cstr(chars: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = chars
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn hash(bytes: &[u8]) -> u64 {
    // FNV-1a: stable across runs/hosts (DefaultHasher is not).
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_size_reflects_embedded_kernels() {
        let by_name = |name: &str| {
            KERNEL_SPECS
                .iter()
                .find(|s| s.name == name)
                .map(|s| spirv_local_size(s.spv))
                .unwrap()
        };
        // Every matmul kernel declares a literal LocalSize.
        for spec in KERNEL_SPECS {
            let size = spirv_local_size(spec.spv).unwrap_or_else(|| panic!("{}", spec.name));
            assert!(size.iter().product::<u32>() > 0, "{}", spec.name);
        }
        assert_eq!(by_name("f16w_coopmat_aligned"), Some([256, 1, 1]));
        assert_eq!(by_name("f16w_coopmat_m64n64"), Some([128, 1, 1]));
        // BDA_V4: (BN/TN) x (BM/TM) = 16 x 16.
        assert_eq!(by_name("large_bda_v4"), Some([16, 16, 1]));
    }

    #[test]
    fn device_names_map_to_arch() {
        assert_eq!(
            arch_for_device("NVIDIA GeForce RTX 3070").as_deref(),
            Some("SM86")
        );
        assert_eq!(
            arch_for_device("NVIDIA GeForce RTX 4060").as_deref(),
            Some("SM89")
        );
        assert_eq!(arch_for_device("llvmpipe"), None);
    }
}
