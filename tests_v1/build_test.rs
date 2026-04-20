//! Phase 1 / Schritt 1.1 — build-system smoke test.
//!
//! Gated at the [[test]] level by `required-features = ["v1"]`. The
//! inner `gpu` test exercises the HIP-kernel linking path; without
//! `--features gpu,v1` it compiles to a no-op because the extern "C"
//! declaration is absent.

#[test]
fn test_v1_directory_structure() {
    let must_exist = [
        "src_v1/lib.rs",
        "src_v1/core/mod.rs",
        "src_v1/ir/mod.rs",
        "src_v1/graph/mod.rs",
        "src_v1/runtime/mod.rs",
        "src_v1/monitor/mod.rs",
        "src_v1/introspection/mod.rs",
        "src_v1/backend/mod.rs",
        "src_v1/backend/gpu/mod.rs",
        "src_v1/backend/cpu/mod.rs",
        "src_v1/cli/mod.rs",
        "hip_kernels_v1/CMakeLists.txt",
        "hip_kernels_v1/smoke_test.hip",
        "tools/rf-forge/Cargo.toml",
        "tools/rf-forge/src/main.rs",
    ];
    for path in must_exist {
        assert!(
            std::path::Path::new(path).exists(),
            "expected {path} to exist under the workspace root"
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_v1_smoke_kernel_links() {
    extern "C" {
        fn launch_v1_smoke_test(out: *mut f32, n: i32, stream: usize) -> i32;
    }
    let fn_ptr = launch_v1_smoke_test as *const ();
    assert!(
        !fn_ptr.is_null(),
        "launch_v1_smoke_test did not resolve — check libv1_smoke.a linking"
    );
}
