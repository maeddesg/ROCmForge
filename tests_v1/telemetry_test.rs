//! Phase 1 / Schritt 1.4 — telemetry smoke tests.
//!
//! `required-features = ["v1"]` at the [[test]] level; GPU-only tests
//! are gated by `#[cfg(feature = "gpu")]` so `--features v1` still
//! compiles the CPU-only profiler tests.

#![cfg(feature = "v1")]

use rocmforge::v1::core::telemetry::{profiling_enabled, Profiler};

#[test]
fn test_profiler_disabled_zero_overhead() {
    // Without `ROCMFORGE_PROFILE=1`, profiling_enabled() must return
    // false and the profiler must drop every observation. The test
    // also accepts "env-var set and profiling active" so CI runs with
    // the flag on don't fail spuriously.
    let mut profiler = Profiler::new();
    let env_set = std::env::var("ROCMFORGE_PROFILE").map(|v| v == "1").unwrap_or(false);
    assert_eq!(profiler.is_enabled(), env_set);
    assert_eq!(profiling_enabled(), env_set);

    profiler.count_dispatch("test_kernel");
    profiler.record("test_op".to_string(), 100.0, 150.0);

    if !env_set {
        let report = profiler.report();
        assert_eq!(
            report.total_dispatches, 0,
            "disabled profiler must not count dispatches"
        );
        assert!(
            report.ops.is_empty(),
            "disabled profiler must not record timings"
        );
    }
}

#[test]
fn test_profiler_dispatch_counting() {
    let mut profiler = Profiler::new();
    profiler.enabled = true; // bypass env-var cache for test determinism

    for _ in 0..3 {
        profiler.count_dispatch("gemv_q4_k");
    }
    profiler.count_dispatch("wmma_q6_k");
    profiler.count_dispatch("rms_norm");
    assert_eq!(profiler.total_dispatches(), 5);

    profiler.record("gemv_q4_k".to_string(), 1000.0, 1100.0);
    profiler.record("gemv_q4_k".to_string(), 1050.0, 1150.0);
    profiler.record("wmma_q6_k".to_string(), 500.0, 600.0);
    profiler.record("rms_norm".to_string(), 50.0, 80.0);

    let report = profiler.report();
    assert_eq!(report.ops.len(), 3, "expected 3 unique op names");
    assert_eq!(report.ops[0].name, "gemv_q4_k");
    assert!((report.ops[0].total_gpu_us - 2050.0).abs() < 1.0);
    assert_eq!(report.ops[0].count, 2);
    assert!(report.total_gpu_us > 0.0);
    // total_wall_us uses real elapsed time — may be very small but
    // must be non-negative and finite.
    assert!(report.total_wall_us.is_finite());
    assert!(report.total_wall_us >= 0.0);

    profiler.reset();
    assert_eq!(profiler.total_dispatches(), 0);
    assert!(profiler.timings().is_empty());
}

#[test]
fn test_profile_report_format() {
    let mut profiler = Profiler::new();
    profiler.enabled = true;

    for i in 0..36 {
        profiler.record(format!("rms_norm_l{i}"), 10.0, 15.0);
        profiler.count_dispatch("rms_norm");
        profiler.record(format!("gemv_qkv_l{i}"), 200.0, 250.0);
        for _ in 0..3 {
            profiler.count_dispatch("gemv_q4_k");
        }
        profiler.record(format!("gate_up_l{i}"), 400.0, 450.0);
        profiler.count_dispatch("gate_up_swiglu");
    }

    let report = profiler.report();
    assert_eq!(report.total_dispatches, 36 * 5); // 1 + 3 + 1 per layer
    assert!(report.dispatch_overhead_us >= 0.0);

    report.print();
}

#[cfg(feature = "gpu")]
#[test]
fn test_measure_gpu_op_accuracy() {
    use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
    use rocmforge::v1::core::telemetry::measure_gpu_op;

    let stream = HipStream::new().expect("hipStreamCreate failed");
    let mut buf = HipBuffer::new(4 * 1024 * 1024).expect("hipMalloc failed");
    let data = vec![0u8; 4 * 1024 * 1024];

    let (gpu_us, wall_us) = measure_gpu_op(&stream, || {
        buf.copy_from_host_async(&data, &stream)?;
        Ok(())
    })
    .expect("measure_gpu_op failed");

    assert!(gpu_us > 0.0, "gpu_us must be positive, got {gpu_us}");
    assert!(
        gpu_us < 10_000.0,
        "4 MiB H2D should take <10 ms, got {gpu_us} µs"
    );
    assert!(
        wall_us >= gpu_us * 0.5,
        "wall_us ({wall_us}) should be >= half of gpu_us ({gpu_us})"
    );
    println!("4 MiB H2D: GPU={gpu_us:.1} µs, Wall={wall_us:.1} µs");
}

#[cfg(feature = "gpu")]
#[test]
fn test_dirty_flags_gpu_cpu_communication() {
    use rocmforge::v1::core::telemetry::DirtyFlags;

    let mut flags = DirtyFlags::new().expect("DirtyFlags::new failed");

    // Fresh buffer must be zeroed.
    assert_eq!(flags.layer_progress(), 0);
    assert!(!flags.nan_detected());
    assert!(!flags.inf_detected());
    assert!(!flags.magnitude_drop());
    assert_eq!(flags.quality_trigger(), 0);

    // Simulate a kernel write from the CPU side through the mapped
    // buffer — same physical memory the kernel would touch.
    {
        let slots = flags.slots_mut();
        slots[DirtyFlags::SLOT_LAYER_PROGRESS] = 17;
        slots[DirtyFlags::SLOT_NAN_DETECTED] = 1;
        slots[DirtyFlags::SLOT_QUALITY_TRIGGER] = 0b0000_0101;
    }

    // Reading requires no GPU sync — that's the whole point of mapped
    // memory.
    assert_eq!(flags.layer_progress(), 17);
    assert!(flags.nan_detected());
    assert!(!flags.inf_detected());
    assert_eq!(flags.quality_trigger(), 0b0000_0101);

    // device_ptr must be non-null and match the host pointer so a
    // kernel can be launched with it as a parameter.
    assert!(!flags.device_ptr().is_null());

    flags.reset();
    assert_eq!(flags.layer_progress(), 0);
    assert!(!flags.nan_detected());
    assert_eq!(flags.quality_trigger(), 0);
}
