//! Regeneration + drift-check for every Phase-1 GEMV kernel file.
//!
//! Mirrors `codegen_wmma_emit_test.rs`. Set `ROCMFORGE_REGEN_KERNELS=1`
//! to refresh the committed files in `hip_kernels_v1/gemv/`; the
//! always-on drift check fails if they diverge from the codegen output.

#![cfg(feature = "v1")]

use rocmforge::v1::ir::codegen_gpu::emit_all_gemv_files;

const OUT_ROOT: &str = "hip_kernels_v1";

#[test]
fn test_gemv_regenerate_on_demand() {
    if std::env::var("ROCMFORGE_REGEN_KERNELS").ok().as_deref() != Some("1") {
        return;
    }
    for (rel_path, src) in emit_all_gemv_files() {
        let full = std::path::Path::new(OUT_ROOT).join(rel_path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).expect("mkdir parent");
        }
        std::fs::write(&full, &src).expect("write generated GEMV kernel");
        println!("Regenerated {} ({} bytes)", full.display(), src.len());
    }
}

#[test]
fn test_gemv_matches_committed() {
    for (rel_path, expected) in emit_all_gemv_files() {
        let full = std::path::Path::new(OUT_ROOT).join(rel_path);
        let actual = match std::fs::read_to_string(&full) {
            Ok(s) => s,
            Err(_) => panic!(
                "{} missing — run `ROCMFORGE_REGEN_KERNELS=1 cargo test \
                 --features v1 --test v1_codegen_gemv_emit_test test_gemv_regenerate_on_demand`",
                full.display()
            ),
        };
        assert_eq!(
            actual, expected,
            "{} drifted from codegen output — regenerate with ROCMFORGE_REGEN_KERNELS=1",
            full.display()
        );
    }
}
