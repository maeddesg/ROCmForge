//! One-shot test that re-runs the GPU codegen and **writes** the
//! generated HIP source to `hip_kernels_v1/dequant_parity.hip`. Enabled
//! with `ROCMFORGE_REGEN_KERNELS=1` — off in CI, on when the codegen
//! changes and we need to refresh the committed file.
//!
//! The companion consistency test (`test_codegen_matches_committed`)
//! always runs and fails if the committed file drifts from what the
//! codegen produces.

#![cfg(feature = "v1")]

use rocmforge::v1::ir::codegen_gpu::emit_all_parity_kernels;

const COMMITTED_PATH: &str = "hip_kernels_v1/dequant_parity.hip";

#[test]
fn test_codegen_regenerate_on_demand() {
    if std::env::var("ROCMFORGE_REGEN_KERNELS").ok().as_deref() != Some("1") {
        return;
    }
    let src = emit_all_parity_kernels();
    std::fs::write(COMMITTED_PATH, &src).expect("write generated kernels");
    println!("Regenerated {COMMITTED_PATH} ({} bytes)", src.len());
}

#[test]
fn test_codegen_matches_committed() {
    let expected = emit_all_parity_kernels();
    let actual = match std::fs::read_to_string(COMMITTED_PATH) {
        Ok(s) => s,
        Err(_) => {
            panic!(
                "{COMMITTED_PATH} missing — run `ROCMFORGE_REGEN_KERNELS=1 cargo test \
                 --features v1 --test v1_codegen_emit_test test_codegen_regenerate_on_demand`"
            );
        }
    };
    assert_eq!(
        actual, expected,
        "committed {COMMITTED_PATH} drifted from codegen output — regenerate with \
         ROCMFORGE_REGEN_KERNELS=1"
    );
}
