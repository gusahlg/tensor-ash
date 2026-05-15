{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    cargo
    clippy
    rustc
    rustfmt
    shaderc
    vulkan-loader
    vulkan-tools
    vulkan-validation-layers
    python3Packages.numpy
    python3Packages.torch
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
    echo "ml_project dev shell: Vulkan loader and shader tools are on PATH."
    echo "Run: cargo run --release --bin ml_bench -- self-check"
  '';
}
