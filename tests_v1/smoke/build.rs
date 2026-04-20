use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src = manifest.join("fp8_wmma_smoke.hip");

    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed=build.rs");

    let rocm = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    let hipcc = format!("{rocm}/bin/hipcc");
    let arch = env::var("ROCMFORGE_SMOKE_ARCH").unwrap_or_else(|_| "gfx1201".into());

    let so_path = out.join("libfp8_smoke.so");

    let status = Command::new(&hipcc)
        .args([
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-shared",
            &format!("--offload-arch={arch}"),
            src.to_str().unwrap(),
            "-o",
            so_path.to_str().unwrap(),
        ])
        .status()
        .expect("failed to invoke hipcc");

    if !status.success() {
        panic!(
            "hipcc failed for {src} (arch={arch}); see stderr above",
            src = src.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=fp8_smoke");
    println!("cargo:rustc-link-search=native={rocm}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    // rpath so the binary finds libfp8_smoke.so at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rocm}/lib");
}
