//! Context-churn stress: reproduce fd/handle leaks across repeated
//! VulkanContext create/dispatch/destroy cycles (one per test in the
//! real suite).  Prints the open-fd count each cycle so exhaustion of
//! `RLIMIT_NOFILE` is visible before it turns into a crash.
use crate::common;

fn open_fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .map(|d| d.count())
        .unwrap_or(0)
}

fn fd_summary() -> String {
    use std::collections::BTreeMap;
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    if let Ok(dir) = std::fs::read_dir("/proc/self/fd") {
        for e in dir.flatten() {
            let target = std::fs::read_link(e.path())
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "?".into());
            // Collapse per-instance suffixes so classes aggregate.
            let class = if target.starts_with("/dev/nvidia") {
                target
            } else if target.starts_with("anon_inode:") || target.starts_with("/memfd") {
                target.split('(').next().unwrap_or(&target).to_string()
            } else if target.starts_with("socket:") {
                "socket".into()
            } else if target.starts_with("/dev/dri") {
                "/dev/dri/*".into()
            } else {
                target
            };
            *counts.entry(class).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .map(|(k, v)| format!("{v}x {k}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One create/dispatch/destroy cycle; returns the selected device name.
fn one_cycle() -> String {
    let (ctx, exec) = common::make_setup(2, 64);
    let name = ctx.device_name().to_string();
    let (gpu, cpu) = common::run_one(
        &ctx,
        &exec,
        &[8, 8],
        &[8, 8],
        &[8, 8],
        1.0,
        false,
        1,
        2,
        None,
    );
    common::assert_close(&gpu, &cpu, 8, "churn");
    name
}

/// The libtest harness runs every test on a fresh thread; each test
/// builds and drops a VulkanContext, so the Vulkan loader dlcloses and
/// re-dlopens the ICD once per test *from a new thread each time*.
/// This is the pattern that exhausted glibc's static-TLS surplus via
/// libnvidia-tls.so and silently demoted the suite to llvmpipe.
#[test]
#[ignore]
fn context_churn_across_threads_keeps_device() {
    let first = one_cycle();
    eprintln!("cycle  0: device = {first}");
    for i in 1..25 {
        let name = std::thread::spawn(one_cycle).join().expect("cycle thread");
        eprintln!("cycle {i:2}: device = {name}");
        assert_eq!(
            name, first,
            "device changed mid-churn at cycle {i} — ICD reload failure \
             (e.g. static TLS exhaustion) caused a silent fallback"
        );
    }
}

#[test]
#[ignore]
fn context_churn_does_not_leak_fds() {
    let mut baseline = None;
    for i in 0..40 {
        let (ctx, exec) = common::make_setup(2, 64);
        // One trivial dispatch so the device/queue actually does work.
        let (gpu, cpu) = common::run_one(
            &ctx,
            &exec,
            &[8, 8],
            &[8, 8],
            &[8, 8],
            1.0,
            false,
            1,
            2,
            None,
        );
        common::assert_close(&gpu, &cpu, 8, "churn");
        drop(exec);
        drop(ctx);
        let fds = open_fd_count();
        eprintln!("cycle {i:2}: {fds} open fds [{}]", fd_summary());
        if i == 1 {
            baseline = Some(fds);
        }
        if let Some(base) = baseline {
            assert!(
                fds <= base + 8,
                "fd leak: cycle {i} has {fds} open fds (baseline {base}); [{}]",
                fd_summary()
            );
        }
    }
}
