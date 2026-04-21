//! Phase 1 / Schritt 1.14 — Quality Monitor tests.
//!
//! CPU-only unit tests for the drift + repetition logic; GPU
//! integration tests exercise calibration + the "no false positive"
//! invariant on a real decode.

#![cfg(feature = "v1")]

use rocmforge::v1::introspection::PrecisionHint;
use rocmforge::v1::monitor::{
    DriftReason, ExpectedRange, NodeId, QualityMonitor, OUTPUT_HIDDEN,
};

fn calibrated(mean: f32, stddev: f32, max_abs: f32) -> QualityMonitor {
    let mut m = QualityMonitor::new(32, 3.0);
    m.expected_ranges.insert(
        OUTPUT_HIDDEN,
        ExpectedRange {
            mean_abs_expected: mean,
            mean_abs_stddev: stddev,
            max_abs_expected: max_abs,
            nan_tolerance: 0,
        },
    );
    m
}

// ── Calibration ──────────────────────────────────────────────────────────

#[test]
fn test_install_calibration_populates_band() {
    let mut m = QualityMonitor::new(32, 3.0);
    let samples = vec![(0.5, 3.0), (0.55, 3.1), (0.45, 2.9), (0.52, 3.05), (0.48, 2.95)];
    m.install_calibration(OUTPUT_HIDDEN, &samples);
    let band = m
        .expected_ranges
        .get(&OUTPUT_HIDDEN)
        .expect("band installed");
    assert!(band.mean_abs_expected > 0.4 && band.mean_abs_expected < 0.6);
    assert!(band.mean_abs_stddev > 0.0);
    assert_eq!(band.nan_tolerance, 0);
}

// ── Drift-Detection primitives ───────────────────────────────────────────

#[test]
fn test_detect_nan() {
    let monitor = calibrated(0.5, 0.1, 5.0);
    let mut hidden = vec![0.5f32; 4096];
    hidden[100] = f32::NAN;
    let signal = monitor
        .check_hidden_state(OUTPUT_HIDDEN, &hidden)
        .expect("NaN should fire");
    assert!(matches!(signal.reason, DriftReason::NaNDetected { count: 1 }));
    assert_eq!(signal.recommended_precision, PrecisionHint::Fp32Scales);
}

#[test]
fn test_detect_inf() {
    let monitor = calibrated(0.5, 0.1, 5.0);
    let mut hidden = vec![0.5f32; 4096];
    hidden[100] = f32::INFINITY;
    let signal = monitor
        .check_hidden_state(OUTPUT_HIDDEN, &hidden)
        .expect("Inf should fire");
    assert!(matches!(signal.reason, DriftReason::InfDetected { count: 1 }));
}

#[test]
fn test_detect_magnitude_drift() {
    let monitor = calibrated(0.5, 0.1, 5.0);
    // mean_abs ≈ 0.5 — no drift expected.
    let normal = vec![0.5f32; 4096];
    assert!(monitor
        .check_hidden_state(OUTPUT_HIDDEN, &normal)
        .is_none());
    // mean_abs ≈ 0.01 — z_score ≈ -4.9, fires.
    let drifted = vec![0.01f32; 4096];
    let signal = monitor
        .check_hidden_state(OUTPUT_HIDDEN, &drifted)
        .expect("drift should fire");
    match signal.reason {
        DriftReason::MeanAbsExceedsStdTolerance { z_score, .. } => {
            assert!(z_score.abs() > 3.0, "z-score should exceed tolerance");
        }
        other => panic!("unexpected reason: {other:?}"),
    }
}

#[test]
fn test_detect_max_abs_overflow() {
    let monitor = calibrated(0.5, 0.1, 5.0);
    let mut hidden = vec![0.5f32; 4096];
    hidden[0] = 65_000.0;
    let signal = monitor
        .check_hidden_state(OUTPUT_HIDDEN, &hidden)
        .expect("overflow should fire");
    assert!(matches!(
        signal.reason,
        DriftReason::MaxAbsExceedsThreshold { .. }
    ));
}

#[test]
fn test_no_signal_when_band_absent() {
    // Without a calibrated band for the node the check must be a
    // no-op, not a false positive.
    let m = QualityMonitor::new(32, 3.0);
    let hidden = vec![1.0f32; 4096];
    assert!(m.check_hidden_state(NodeId(99), &hidden).is_none());
}

// ── Repetition ───────────────────────────────────────────────────────────

#[test]
fn test_detect_repetition_5x_same_token() {
    let mut m = QualityMonitor::new(32, 3.0);
    for _ in 0..4 {
        m.record_token(42);
        assert!(m.check_repetition(42).is_none());
    }
    m.record_token(42);
    match m.check_repetition(42) {
        Some(DriftReason::RepetitionDetected { token_id: 42, count: 5 }) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn test_no_repetition_on_normal_text() {
    let mut m = QualityMonitor::new(32, 3.0);
    for t in [10, 20, 30, 40, 50, 10, 20, 30, 40, 50] {
        m.record_token(t);
        assert!(m.check_repetition(t).is_none());
    }
}

// ── Sample rate ──────────────────────────────────────────────────────────

#[test]
fn test_sample_rate() {
    let mut m = QualityMonitor::new(32, 3.0);
    for _ in 0..31 {
        m.record_token(1);
        assert!(!m.should_check());
    }
    m.record_token(1);
    assert!(m.should_check());
    m.reset_check_counter();
    assert!(!m.should_check());
}

// ── GPU integration ──────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
mod gpu_tests {
    use super::*;
    use rocmforge::v1::backend::gpu::device::GpuDevice;
    use rocmforge::v1::core::gguf::GGUFFile;
    use rocmforge::v1::core::inference::InferencePipeline;
    use rocmforge::v1::core::model_config::ModelConfig;
    use rocmforge::v1::core::model_loader::LoadedModel;
    use rocmforge::v1::core::sampling::SamplingConfig;
    use rocmforge::v1::core::tensor_info::{
        group_tensors_by_layer, parse_tensor_name, TensorInfo, TensorRole,
    };
    use rocmforge::v1::graph::{BufferPlan, GraphBuildContext, GraphBuilder};
    use serial_test::serial;

    const QWEN3: &str = "Qwen3-8B-Q4_K_M.gguf";

    fn model_path(name: &str) -> std::path::PathBuf {
        dirs::home_dir().expect("HOME set").join("models").join(name)
    }

    fn load_pipeline_qwen3() -> InferencePipeline<'static> {
        let device = GpuDevice::detect(0).expect("gpu");
        let path = model_path(QWEN3);
        let model = LoadedModel::load(&path, &device).expect("load model");
        let gguf = GGUFFile::open(&path).expect("reopen gguf");
        let model_static: &'static LoadedModel = Box::leak(Box::new(model));
        let gguf_static: &'static GGUFFile = Box::leak(Box::new(gguf));

        let cfg = ModelConfig::from_metadata(gguf_static.metadata(), gguf_static.tensors())
            .expect("cfg");
        let layers = group_tensors_by_layer(gguf_static.tensors());
        let mut globals: std::collections::HashMap<TensorRole, &TensorInfo> =
            std::collections::HashMap::new();
        for t in gguf_static.tensors() {
            let (role, li) = parse_tensor_name(&t.name);
            if li.is_none() && !matches!(role, TensorRole::Unknown(_)) {
                globals.insert(role, t);
            }
        }
        let ctx = GraphBuildContext {
            config: &cfg,
            layers: &layers,
            global_tensors: globals,
        };
        let graph = GraphBuilder::build(&ctx).expect("build graph");
        let plan = BufferPlan::plan_phase1(&graph);
        InferencePipeline::new(graph, plan, model_static, gguf_static, 256)
            .expect("pipeline")
    }

    /// Calibration must populate `OUTPUT_HIDDEN` and pick a
    /// non-zero mean_abs. The actual values depend on the
    /// calibration prompt × model; we only assert plausibility.
    #[test]
    #[serial]
    fn test_calibration_populates_ranges() {
        let mut pipe = load_pipeline_qwen3();
        pipe.calibrate_monitor().expect("calibrate");
        let band = pipe
            .monitor
            .expected_ranges
            .get(&OUTPUT_HIDDEN)
            .expect("band installed");
        assert!(band.mean_abs_expected > 0.0);
        assert!(band.mean_abs_stddev > 0.0);
        assert!(band.max_abs_expected > band.mean_abs_expected);
    }

    /// With the calibrated band installed, a normal 128-token
    /// generation on a well-behaved prompt must log zero drift
    /// events. A false positive here would indicate the band is
    /// too tight or the stddev is too small.
    #[test]
    #[serial]
    fn test_monitor_no_false_positives_on_normal_inference() {
        let mut pipe = load_pipeline_qwen3();
        pipe.calibrate_monitor().expect("calibrate");
        pipe.reset().expect("reset");
        let result = pipe
            .generate(
                "Explain what a mutex is in one paragraph.",
                128,
                &SamplingConfig::greedy(),
                true,
            )
            .expect("generate");
        assert!(result.generated_tokens > 10);
        assert!(
            pipe.monitor.revision_log.is_empty(),
            "expected zero drift events on normal inference, got {}: {:?}",
            pipe.monitor.revision_log.len(),
            pipe.monitor
                .revision_log
                .iter()
                .map(|e| format!("{:?}", e.signal.reason))
                .collect::<Vec<_>>()
        );
    }
}
