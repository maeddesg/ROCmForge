//! Regeneration + drift-check for the WMMA GEMM kernel file. Mirrors
//! `codegen_emit_test.rs` for the parity kernels.
//!
//! Set `ROCMFORGE_REGEN_KERNELS=1` and run `test_wmma_regenerate_on_demand`
//! to refresh `hip_kernels_v1/wmma/wmma_gemm_q4_0_fp16.hip`.
//! `test_wmma_matches_committed` always runs and fails if the committed
//! file drifts from codegen output.

#![cfg(feature = "v1")]

use rocmforge::v1::ir::codegen_gpu::emit_all_wmma_kernels;

const COMMITTED_PATH: &str = "hip_kernels_v1/wmma/wmma_gemm_q4_0_fp16.hip";

#[test]
fn test_wmma_regenerate_on_demand() {
    if std::env::var("ROCMFORGE_REGEN_KERNELS").ok().as_deref() != Some("1") {
        return;
    }
    let src = emit_all_wmma_kernels();
    std::fs::create_dir_all("hip_kernels_v1/wmma").expect("mkdir wmma");
    std::fs::write(COMMITTED_PATH, &src).expect("write generated WMMA kernel");
    println!("Regenerated {COMMITTED_PATH} ({} bytes)", src.len());
}

#[test]
fn test_wmma_matches_committed() {
    let expected = emit_all_wmma_kernels();
    let actual = match std::fs::read_to_string(COMMITTED_PATH) {
        Ok(s) => s,
        Err(_) => {
            panic!(
                "{COMMITTED_PATH} missing — run `ROCMFORGE_REGEN_KERNELS=1 cargo test \
                 --features v1 --test v1_codegen_wmma_emit_test test_wmma_regenerate_on_demand`"
            );
        }
    };
    assert_eq!(
        actual, expected,
        "committed {COMMITTED_PATH} drifted from codegen output — regenerate with \
         ROCMFORGE_REGEN_KERNELS=1"
    );
}
