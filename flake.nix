{
  description = "tensor-ash development and CUDA benchmark shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      cuda = pkgs.cudaPackages;
      runtimeLibraryPath = pkgs.lib.makeLibraryPath [
        pkgs.vulkan-loader
        pkgs.stdenv.cc.cc.lib
      ];
      python = pkgs.python3.withPackages (ps: [
        ps.numpy
        ps.pip
        ps.virtualenv
      ]);
      commonPackages = with pkgs; [
        cargo
        clippy
        gcc
        pkg-config
        rustc
        rustfmt
        shaderc
        uv
        vulkan-loader
        vulkan-tools
        vulkan-validation-layers
        python
        cuda.cuda_nvcc
        cuda.cuda_cudart
        cuda.cuda_nvrtc
        cuda.libcublas
        cuda.libcurand
        cuda.libcusolver
        cuda.libcusparse
      ];
    in
    {
      devShells.${system} = {
        default = pkgs.mkShell {
          packages = commonPackages;

          shellHook = ''
            export LD_LIBRARY_PATH="${runtimeLibraryPath}:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
            export CUDA_PATH="${cuda.cuda_nvcc}"
            echo "tensor-ash shell: Rust, Vulkan, CUDA tools, Python, and uv are available."
            echo "GPU Python setup: uv venv .venv-bench && source .venv-bench/bin/activate && uv pip install -r requirements-benchmark.txt"
            echo "Benchmark: python3 scripts/bench_compare.py --case-set extended --iters 20 --warmup 5 --torch-threads 1"
          '';
        };

        benchmark = pkgs.mkShell {
          packages = commonPackages;

          shellHook = ''
            export LD_LIBRARY_PATH="${runtimeLibraryPath}:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
            export CUDA_PATH="${cuda.cuda_nvcc}"
            echo "tensor-ash benchmark shell: run the GPU Python setup command once, then run scripts/bench_compare.py."
          '';
        };
      };
    };
}
