//! Build script: compiles every `shaders/*.comp` to a SPIR-V binary in OUT_DIR.
//! The resulting `.spv` files are embedded into the binary via `include_bytes!`.
//!
//! Requires `glslc` (from the Vulkan SDK / shaderc) on PATH.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir      = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shader_dir   = manifest_dir.join("shaders");

    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=build.rs");

    let entries = std::fs::read_dir(&shader_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", shader_dir.display()));

    let mut compiled = 0usize;
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path  = entry.path();
        let Some(ext) = path.extension().and_then(|s| s.to_str()) else { continue };
        // Compile all standard compute/vertex/fragment GLSL extensions; we only ship .comp today.
        if !matches!(ext, "comp" | "vert" | "frag") { continue; }

        let stem    = path.file_stem().unwrap().to_string_lossy().into_owned();
        let out     = out_dir.join(format!("{stem}.spv"));

        println!("cargo:rerun-if-changed={}", path.display());

        let status = Command::new("glslc")
            .args(["--target-env=vulkan1.2", "-O"])
            .arg(&path)
            .arg("-o")
            .arg(&out)
            .status()
            .unwrap_or_else(|e| panic!(
                "failed to invoke glslc (is the Vulkan SDK installed and on PATH?): {e}"
            ));

        if !status.success() {
            panic!("glslc failed for {}", path.display());
        }
        compiled += 1;
    }

    if compiled == 0 {
        panic!("no shaders found in {}", shader_dir.display());
    }
}
