use std::env;
use std::path::PathBuf;

mod gpu_build {
    use super::*;

    /// Detect the GPU architecture by querying `rocminfo` for the first gfx target,
    /// or use the `ROCMFORGE_GPU_ARCH` env variable as an override.
    fn detect_gpu_arch() -> Option<String> {
        if let Ok(arch) = env::var("ROCMFORGE_GPU_ARCH") {
            println!(
                "cargo:warning=Using GPU arch from ROCMFORGE_GPU_ARCH: {}",
                arch
            );
            return Some(arch);
        }

        let output = std::process::Command::new("rocminfo").output().ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("Name:") {
                let name = trimmed.strip_prefix("Name:")?.trim();
                if name.starts_with("gfx") {
                    println!("cargo:warning=Detected GPU arch: {}", name);
                    return Some(name.to_string());
                }
            }
        }

        None
    }

    pub fn compile_kernels() {
        compile_hip_kernels();
    }

    fn compile_hip_kernels() {
        use std::path::Path;
        use std::process::Command;

        let hip_path = find_rocm_path();
        let hip_path = hip_path
            .as_ref()
            .map(|p| p.as_path())
            .unwrap_or(Path::new("/opt/rocm"));

        let kernels = [
            ("norm", "hip_kernels/norm.hip"),
            ("norm_vulkan_style", "hip_kernels/norm_vulkan_style.hip"),
            ("rope", "hip_kernels/rope.hip"),
            ("matmul", "hip_kernels/matmul.hip"),
            ("elementwise", "hip_kernels/elementwise.hip"),
            ("attention", "hip_kernels/attention.hip"),
        ];

        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
        let hipcc = hip_path.join("bin/hipcc");
        let hip_include = hip_path.join("include");

        if !hipcc.exists() {
            println!(
                "cargo:warning=hipcc not found at {:?}, skipping kernel compilation",
                hipcc
            );
            println!("cargo:warning=GPU feature will use Memoria's libgpu.so as fallback");
            return;
        }

        for (name, source) in kernels {
            let source_file = Path::new(source);
            if !source_file.exists() {
                println!("cargo:warning=Kernel source {} not found, skipping", source);
                continue;
            }

            let obj_file = format!("{}/{}.o", out_dir, name);
            let lib_file = format!("{}/lib{}.a", out_dir, name);

            println!("cargo:warning=Compiling HIP kernel: {}", name);

            // Compile to object file
            let compile_status = Command::new(&hipcc)
                .arg(source_file)
                .arg("-o")
                .arg(&obj_file)
                .arg("-c")
                .arg("-fPIC")
                .arg("-O3")
                .arg(format!(
                    "--offload-arch={}",
                    detect_gpu_arch().unwrap_or_else(|| {
                        println!(
                            "cargo:warning=Could not detect GPU arch, falling back to gfx1100"
                        );
                        "gfx1100".to_string()
                    })
                ))
                .arg(format!("-I{}", hip_include.display()))
                .status();

            match compile_status {
                Ok(s) if s.success() => {
                    // Create static library from object file
                    let ar_status = Command::new("ar")
                        .arg("rcs")
                        .arg(&lib_file)
                        .arg(&obj_file)
                        .status();

                    match ar_status {
                        Ok(_) => {
                            println!("cargo:rustc-link-lib=static={}", name);
                            println!("cargo:rustc-link-search=native={}", out_dir);
                        }
                        Err(e) => {
                            println!("cargo:warning=Failed to archive kernel {}: {:?}", name, e);
                        }
                    }
                }
                Ok(s) => {
                    println!(
                        "cargo:warning=Kernel {} compilation returned non-zero exit code: {:?}",
                        name,
                        s.code()
                    );
                }
                Err(e) => {
                    println!("cargo:warning=Kernel {} compilation failed: {:?}", name, e);
                }
            }
        }
    }

    pub fn compile_quant_kernels() {
        use std::path::Path;
        use std::process::Command;

        // Find cmake executable
        let cmake = find_cmake();
        let cmake = cmake.as_deref().unwrap_or(Path::new("cmake"));

        // Quantization kernel paths
        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
        let lib_dest = Path::new(&out_dir);
        let quant_src = Path::new("hip_kernels/quant");
        let quant_build = PathBuf::from(&out_dir).join("quant-cmake-build");
        let rocm_path = find_rocm_path().unwrap_or_else(|| PathBuf::from("/opt/rocm"));
        let hip_dir = rocm_path.join("lib/cmake/hip");

        // Create build directory if it doesn't exist
        if !quant_build.exists() {
            if let Err(e) = std::fs::create_dir_all(&quant_build) {
                println!(
                    "cargo:warning=Failed to create quant build directory: {}",
                    e
                );
                return;
            }
        }

        println!("cargo:warning=Compiling quantization kernels with CMake");

        // Configure with CMake
        let config_status = Command::new(cmake)
            .arg("-S")
            .arg(quant_src)
            .arg("-B")
            .arg(&quant_build)
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg(format!("-DCMAKE_PREFIX_PATH={}", rocm_path.display()))
            .arg(format!("-Dhip_DIR={}", hip_dir.display()))
            .status();

        match config_status {
            Ok(s) if s.success() => {
                // Build the project
                let build_status = Command::new(cmake)
                    .arg("--build")
                    .arg(&quant_build)
                    .arg("--config")
                    .arg("Release")
                    .status();

                match build_status {
                    Ok(_) => {
                        // Copy libraries to output directory for Cargo linking
                        let libs_to_copy = vec![
                            ("libtest_quant.a", "test_quant"),
                            // Q4_K libraries
                            ("libq4_k_quantize.a", "q4_k_quantize"),
                            ("libq4_k_dequantize.a", "q4_k_dequantize"),
                            ("libq4_k_verify.a", "q4_k_verify"),
                            // Q8_0 libraries
                            ("libq8_0_quantize.a", "q8_0_quantize"),
                            ("libq8_0_dequantize.a", "q8_0_dequantize"),
                            ("libq8_0_verify.a", "q8_0_verify"),
                            // Q8_0 GEMV library
                            ("libq8_0_gemv.a", "q8_0_gemv"),
                            // Q4_K GEMV library
                            ("libq4_k_gemv.a", "q4_k_gemv"),
                            ("libq4_k_q8_inline.a", "q4_k_q8_inline"),
                            // Q5_K libraries
                            ("libq5_k_quantize.a", "q5_k_quantize"),
                            ("libq5_k_dequantize.a", "q5_k_dequantize"),
                            ("libq5_k_verify.a", "q5_k_verify"),
                            ("libq5_k_gemv.a", "q5_k_gemv"),
                            // Q6_K libraries
                            ("libq6_k_gemv.a", "q6_k_gemv"),
                            // Q4_0 libraries
                            ("libq4_0_quantize.a", "q4_0_quantize"),
                            ("libq4_0_dequantize.a", "q4_0_dequantize"),
                            ("libdequant_q4_0_to_f16.a", "dequant_q4_0_to_f16"),
                            ("libwmma_gemm_16x16.a", "wmma_gemm_16x16"),
                            ("libwmma_gemm_tiled.a", "wmma_gemm_tiled"),
                            ("libwmma_gemm_q4_0.a", "wmma_gemm_q4_0"),
                            (
                                "libwmma_gemm_q4_0_fused_gate_up.a",
                                "wmma_gemm_q4_0_fused_gate_up",
                            ),
                            ("libwmma_gemm_q4_1.a", "wmma_gemm_q4_1"),
                            ("libwmma_gemm_q4_k.a", "wmma_gemm_q4_k"),
                            ("libwmma_gemm_q6_k.a", "wmma_gemm_q6_k"),
                            ("libwmma_attention_prefill.a", "wmma_attention_prefill"),
                            ("libq4_0_verify.a", "q4_0_verify"),
                            ("libq4_0_gemv.a", "q4_0_gemv"),
                            ("libq4_0_gemv_batched.a", "q4_0_gemv_batched"),
                            ("libq4_0_gemv_batched_tiled.a", "q4_0_gemv_batched_tiled"),
                            // Q4_1 libraries
                            ("libq4_1_quantize.a", "q4_1_quantize"),
                            ("libq4_1_dequantize.a", "q4_1_dequantize"),
                            ("libq4_1_verify.a", "q4_1_verify"),
                            ("libq4_1_gemv.a", "q4_1_gemv"),
                            ("libq4_1_ffn_experimental.a", "q4_1_ffn_experimental"),
                            ("libq4_0_vulkan_style.a", "q4_0_vulkan_style"),
                            ("libq4_0_gemv_vulkan_style.a", "q4_0_gemv_vulkan_style"),
                            ("libq4_k_gemv_vulkan_style.a", "q4_k_gemv_vulkan_style"),
                            // GEMM libraries
                            ("libq4_0_gemm.a", "q4_0_gemm"),
                            ("libq4_0_fused.a", "q4_0_fused"),
                            ("libq4_1_gemm.a", "q4_1_gemm"),
                            ("libq8_0_gemm.a", "q8_0_gemm"),
                            ("libq4_k_gemm.a", "q4_k_gemm"),
                            ("libq5_k_gemm.a", "q5_k_gemm"),
                        ];

                        for (lib_name, link_name) in libs_to_copy {
                            let src_lib = quant_build.join("lib").join(lib_name);
                            let dst_lib = lib_dest.join(lib_name);

                            if src_lib.exists() {
                                if let Err(e) = std::fs::copy(&src_lib, &dst_lib) {
                                    println!("cargo:warning=Failed to copy {}: {}", lib_name, e);
                                    continue;
                                }
                                println!("cargo:rustc-link-lib=static={}", link_name);
                            } else {
                                println!("cargo:warning={} not found (skipping)", lib_name);
                            }
                        }

                        println!("cargo:rustc-link-search=native={}", out_dir);
                    }
                    Err(e) => {
                        println!("cargo:warning=Quantization CMake build failed: {:?}", e);
                    }
                }
            }
            Ok(s) => {
                println!(
                    "cargo:warning=CMake configuration returned non-zero: {:?}",
                    s.code()
                );
            }
            Err(e) => {
                println!("cargo:warning=CMake configuration failed: {:?}", e);
            }
        }
    }

    fn find_cmake() -> Option<PathBuf> {
        // Try common CMake locations
        let standard_paths = [
            PathBuf::from("/usr/bin/cmake"),
            PathBuf::from("/usr/local/bin/cmake"),
            PathBuf::from("/opt/homebrew/bin/cmake"),
        ];

        for path in standard_paths.iter() {
            if path.exists() {
                return Some(path.clone());
            }
        }

        None
    }

    /// v1.0 HIP kernel compilation via CMake. Builds only the subset
    /// under `hip_kernels_v1/` and does not touch the v0.x pipeline in
    /// `hip_kernels/`. Guarded at the call site by `CARGO_FEATURE_GPU`,
    /// so `cpu-only` builds never invoke cmake / hipcc / link amdhip64.
    pub fn compile_v1_kernels() {
        use std::path::Path;
        use std::process::Command;

        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
        let build_dir = PathBuf::from(&out_dir).join("hip_kernels_v1_build");

        if let Err(e) = std::fs::create_dir_all(&build_dir) {
            println!(
                "cargo:warning=Failed to create v1 kernel build directory: {}",
                e
            );
            return;
        }

        let cmake = find_cmake();
        let cmake = cmake.as_deref().unwrap_or(Path::new("cmake"));

        let rocm_path = find_rocm_path().unwrap_or_else(|| PathBuf::from("/opt/rocm"));
        let hip_dir = rocm_path.join("lib/cmake/hip");

        println!("cargo:warning=Compiling v1.0 HIP kernels (hip_kernels_v1/) with CMake");

        let config_status = Command::new(cmake)
            .arg("-S")
            .arg("hip_kernels_v1")
            .arg("-B")
            .arg(&build_dir)
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg(format!("-DCMAKE_PREFIX_PATH={}", rocm_path.display()))
            .arg(format!("-Dhip_DIR={}", hip_dir.display()))
            .status();

        match config_status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                println!(
                    "cargo:warning=v1 CMake configure returned non-zero: {:?}",
                    s.code()
                );
                return;
            }
            Err(e) => {
                println!("cargo:warning=v1 CMake configure failed: {:?}", e);
                return;
            }
        }

        let build_status = Command::new(cmake)
            .arg("--build")
            .arg(&build_dir)
            .arg("--parallel")
            .status();

        match build_status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                println!(
                    "cargo:warning=v1 CMake build returned non-zero: {:?}",
                    s.code()
                );
                return;
            }
            Err(e) => {
                println!("cargo:warning=v1 CMake build failed: {:?}", e);
                return;
            }
        }

        // Link Phase-1 v1 libraries. More land here as v1.0 kernels come online.
        println!("cargo:rustc-link-search=native={}", build_dir.display());
        for lib in &[
            "v1_smoke",
            "v1_device_info",
            "v1_dequant_parity",
            "v1_wmma_q4_0_fp16",
            "v1_wmma_q4_k_fp16",
            "v1_wmma_q4_k_fp16_tiled",
            "v1_wmma_q6_k_fp16",
            "v1_wmma_q8_0_fp16",
            "v1_wmma_q4_0_fp8",
            "v1_wmma_q4_k_fp8",
            "v1_wmma_q6_k_fp8",
            "v1_wmma_q8_0_fp8",
            "v1_gemv_q4_0_standard",
            "v1_gemv_q4_k_standard",
            "v1_gemv_q6_k_standard",
            "v1_gemv_q8_0_standard",
            "v1_gemv_q4_k_q8_inline",
            "v1_gemv_q4_k_q8_inline_residual",
            "v1_gemv_q6_k_q8_inline",
            "v1_gemv_q6_k_mmvq",
            "v1_gemv_q4_k_q8_inline_sudot4",
            "v1_gemv_q4_k_mmvq",
            "v1_gemv_q4_k_mmvq_residual",
            "v1_gemv_q4_k_mmvq_fused",
            "v1_gemv_q4_k_gate_up_swiglu",
            "v1_quantize_q8_1",
            "v1_quantize_q8_1_mmq",
            "v1_wmma_i32_smoke",
            "v1_elementwise_block_a",
            "v1_rope",
            "v1_attention",
            "v1_kv_cache_fp8",
        ] {
            let path = build_dir.join(format!("lib{lib}.a"));
            if path.exists() {
                println!("cargo:rustc-link-lib=static={lib}");
            } else {
                println!("cargo:warning=v1 library not found at {}", path.display());
            }
        }

        println!("cargo:rerun-if-changed=hip_kernels_v1");
    }

    fn find_rocm_path() -> Option<PathBuf> {
        if let Ok(hip_path) = env::var("HIP_PATH") {
            let path = PathBuf::from(&hip_path);
            if path.exists() {
                return Some(path);
            }
        }

        if let Ok(rocm_path) = env::var("ROCM_PATH") {
            let path = PathBuf::from(&rocm_path);
            if path.exists() {
                return Some(path);
            }
        }

        let standard_paths = [
            PathBuf::from("/opt/rocm"),
            PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        ];

        for path in standard_paths.iter() {
            if path.exists() && path.join("bin/hipcc").exists() {
                return Some(path.clone());
            }
        }

        None
    }
}

fn main() {
    if env::var("CARGO_FEATURE_GPU").is_ok() {
        // Compile HIP kernels
        gpu_build::compile_kernels();

        // Compile quantization kernels via CMake
        gpu_build::compile_quant_kernels();

        // Compile v1.0 HIP kernels (Phase 1 onwards).
        gpu_build::compile_v1_kernels();

        // Link HIP runtime
        let hip_path = env::var("HIP_PATH")
            .ok()
            .and_then(|p| {
                let path = PathBuf::from(&p);
                if path.exists() {
                    Some(path)
                } else {
                    None
                }
            })
            .or_else(|| {
                if PathBuf::from("/opt/rocm/lib").exists() {
                    Some(PathBuf::from("/opt/rocm"))
                } else {
                    None
                }
            });

        if let Some(path) = hip_path {
            println!("cargo:rustc-link-search=native={}/lib", path.display());
        }

        println!("cargo:rustc-link-lib=hiprtc");
        println!("cargo:rustc-link-lib=amdhip64");
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rerun-if-env-changed=HIP_PATH");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-env-changed=ROCMFORGE_GPU_ARCH");
        println!("cargo:rerun-if-changed=hip_kernels");
        println!("cargo:rerun-if-changed=hip_kernels_v1");
    }
}
