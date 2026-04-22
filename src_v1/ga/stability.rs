//! Stability-Validation der Top-Winner (`ga_tuning_spec §2.9`).
//!
//! After the main GA loop converges, the Top-5 candidates run a
//! stricter check before any of them lands in the production cache:
//!
//! * 3 input sets (short/medium/long context) × 10 runs = 30 samples
//! * median-variance over all 30 runs must stay < 2 %
//! * parity across 1000 deterministic blocks (not just 10)
//!
//! Failing any of the three → the candidate is "brittle", drops out
//! of the Pareto front, and the next-best candidate takes its slot.

use std::time::Duration;

use super::parity::{
    check_parity_known_kernel, KnownKernel, ParityConfig, ParityResult,
};
use super::types::KernelShape;

/// Tunables for stability validation. Defaults match `§2.9` verbatim.
#[derive(Debug, Clone, Copy)]
pub struct StabilityConfig {
    pub n_input_sets: usize,
    pub n_runs_per_set: usize,
    pub max_variance_pct: f64,
    pub parity_n_blocks: usize,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            n_input_sets: 3,
            n_runs_per_set: 10,
            max_variance_pct: 2.0,
            parity_n_blocks: 1000,
        }
    }
}

/// Verdict for one Top-K candidate.
#[derive(Debug, Clone)]
pub struct StabilityResult {
    pub passed: bool,
    pub median_times_us: Vec<f64>,
    pub variance_pct: f64,
    pub parity_result: Option<ParityResult>,
    pub reject_reason: Option<String>,
}

/// Thermal cool-down before the Top-5 stability pass. The GA spends
/// ~8 min per shape hammering the GPU at full clock; running the
/// stability check immediately after can hit throttled frequencies
/// and falsely flag all 5 candidates as "variance > 2 %". 2 s is
/// conservative — Navi 48 returns to base clock in ~400 ms after
/// idle, so this is ~5× that ceiling.
pub const THERMAL_COOLDOWN: Duration = Duration::from_secs(2);

/// Simple median helper. Copy of `fitness::median` kept local to
/// avoid a cross-module cycle.
fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::INFINITY;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("NaN in stability times"));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// End-to-end stability check for a Phase-1 kernel. Called from the
/// GA engine's post-convergence pass for every Top-K candidate.
/// Pools device buffers + stream across the 30 runs so the timing
/// measures kernel execution, not allocator churn.
#[cfg(feature = "gpu")]
pub fn check_stability_known_kernel(
    kind: KnownKernel,
    shape: &KernelShape,
    cfg: &StabilityConfig,
) -> Result<StabilityResult, String> {
    use crate::v1::backend::gpu::wrappers::{HipBuffer, HipEvent, HipStream};

    let fmt = kind.quant_format();
    let stream = HipStream::new().map_err(|e| format!("stability stream: {e:?}"))?;

    // Three "input sets" are simulated by three distinct deterministic
    // seeds — short/medium/long context differentiation lands when
    // attention shapes come into the picture in `2.1.4`.
    let seeds: [u64; 3] = [1, 2, 3];

    let mut all_times = Vec::with_capacity(cfg.n_input_sets * cfg.n_runs_per_set);
    let mut median_per_set = Vec::with_capacity(cfg.n_input_sets);

    // GPU-wide clock warmup before any timed runs. Consumer gfx1201
    // drops its core clock under idle and needs ~100 ms of continuous
    // work to hit peak again. 30 dispatches on the first input set's
    // data brings the clock up to a steady state that subsequent
    // `n_runs_per_set` × `n_input_sets` samples can measure stably.
    {
        let warmup_blocks =
            super::parity::generate_deterministic_test_blocks(1, shape, &fmt, seeds[0]);
        let tb = &warmup_blocks[0];
        let mut d_w = HipBuffer::new(tb.weights.len())
            .map_err(|e| format!("global warmup d_w: {e:?}"))?;
        d_w.copy_from_host(&tb.weights)
            .map_err(|e| format!("global warmup d_w upload: {e:?}"))?;
        let in_bytes = unsafe {
            std::slice::from_raw_parts(tb.input.as_ptr() as *const u8, tb.input.len() * 4)
        };
        let mut d_in = HipBuffer::new(in_bytes.len())
            .map_err(|e| format!("global warmup d_in: {e:?}"))?;
        d_in.copy_from_host(in_bytes)
            .map_err(|e| format!("global warmup d_in upload: {e:?}"))?;
        let zero_res = vec![0.0f32; shape.n];
        let res_bytes = unsafe {
            std::slice::from_raw_parts(zero_res.as_ptr() as *const u8, zero_res.len() * 4)
        };
        let mut d_res = HipBuffer::new(res_bytes.len())
            .map_err(|e| format!("global warmup d_res: {e:?}"))?;
        d_res
            .copy_from_host(res_bytes)
            .map_err(|e| format!("global warmup d_res upload: {e:?}"))?;
        let mut d_out = HipBuffer::new(shape.n * 4)
            .map_err(|e| format!("global warmup d_out: {e:?}"))?;
        // Issue ~200 dispatches in bulk, then sync once. Submit rate
        // matters more than total count here — we need the HIP stream
        // queue to stay saturated so the clock governor holds peak
        // for the duration of the warmup.
        for _ in 0..200 {
            super::parity::run_known_kernel_pooled(
                kind,
                d_w.as_ptr(),
                d_in.as_ptr(),
                d_res.as_ptr(),
                d_out.as_mut_ptr(),
                shape,
                stream.raw(),
            )?;
        }
        stream
            .synchronize()
            .map_err(|e| format!("global warmup sync: {e:?}"))?;
    }

    for (set_idx, &seed) in seeds.iter().take(cfg.n_input_sets).enumerate() {
        let blocks = super::parity::generate_deterministic_test_blocks(1, shape, &fmt, seed);
        let tb = &blocks[0];

        // Allocate once per set, reuse for the warmup + 10 timed runs.
        let mut d_w = HipBuffer::new(tb.weights.len())
            .map_err(|e| format!("stability d_w alloc: {e:?}"))?;
        d_w.copy_from_host(&tb.weights)
            .map_err(|e| format!("stability d_w upload: {e:?}"))?;
        let in_bytes = unsafe {
            std::slice::from_raw_parts(tb.input.as_ptr() as *const u8, tb.input.len() * 4)
        };
        let mut d_in = HipBuffer::new(in_bytes.len())
            .map_err(|e| format!("stability d_in alloc: {e:?}"))?;
        d_in.copy_from_host(in_bytes)
            .map_err(|e| format!("stability d_in upload: {e:?}"))?;
        let zero_res = vec![0.0f32; shape.n];
        let res_bytes = unsafe {
            std::slice::from_raw_parts(zero_res.as_ptr() as *const u8, zero_res.len() * 4)
        };
        let mut d_res = HipBuffer::new(res_bytes.len())
            .map_err(|e| format!("stability d_res alloc: {e:?}"))?;
        d_res
            .copy_from_host(res_bytes)
            .map_err(|e| format!("stability d_res upload: {e:?}"))?;
        let mut d_out = HipBuffer::new(shape.n * 4)
            .map_err(|e| format!("stability d_out alloc: {e:?}"))?;

        // Warm-up: 10 dispatches to saturate the kernel's L2/VRAM
        // residency and settle GPU clock. Empirically on gfx1201:
        //   1 warmup  → 22 % variance
        //   3 warmups → 2.1 – 3.7 % variance (context-dependent)
        //  10 warmups → consistently < 3 %
        for _ in 0..10 {
            super::parity::run_known_kernel_pooled(
                kind,
                d_w.as_ptr(),
                d_in.as_ptr(),
                d_res.as_ptr(),
                d_out.as_mut_ptr(),
                shape,
                stream.raw(),
            )?;
        }
        stream.synchronize().map_err(|e| format!("warmup sync: {e:?}"))?;

        let mut times_us = Vec::with_capacity(cfg.n_runs_per_set);
        for _ in 0..cfg.n_runs_per_set {
            let start_ev = HipEvent::new().map_err(|e| format!("event start: {e:?}"))?;
            let stop_ev = HipEvent::new().map_err(|e| format!("event stop: {e:?}"))?;
            start_ev
                .record(&stream)
                .map_err(|e| format!("start.record: {e:?}"))?;
            super::parity::run_known_kernel_pooled(
                kind,
                d_w.as_ptr(),
                d_in.as_ptr(),
                d_res.as_ptr(),
                d_out.as_mut_ptr(),
                shape,
                stream.raw(),
            )?;
            stop_ev.record(&stream)
                .map_err(|e| format!("stop.record: {e:?}"))?;
            stream.synchronize().map_err(|e| format!("sync: {e:?}"))?;
            let ms = HipEvent::elapsed_ms(&start_ev, &stop_ev)
                .map_err(|e| format!("elapsed_ms: {e:?}"))?;
            times_us.push(ms as f64 * 1000.0);
        }
        let _ = set_idx;
        median_per_set.push(median(&times_us));
        all_times.extend_from_slice(&times_us);
    }

    let overall_median = median(&all_times);
    let variance_pct = if overall_median > 0.0 {
        let max_dev = all_times
            .iter()
            .map(|t| (t - overall_median).abs())
            .fold(0.0f64, f64::max);
        (max_dev / overall_median) * 100.0
    } else {
        100.0
    };

    // 1000-block parity — the tight gate. Parity failure alone
    // dumps the candidate even if timing was rock-solid.
    let parity_cfg = ParityConfig::for_stability();
    let parity = check_parity_known_kernel(kind, shape, &parity_cfg, 42)
        .map_err(|e| format!("stability parity: {e}"))?;

    let mut reject_reason: Option<String> = None;
    if variance_pct >= cfg.max_variance_pct {
        reject_reason = Some(format!(
            "variance {:.2}% ≥ {:.2}% threshold",
            variance_pct, cfg.max_variance_pct
        ));
    }
    if !parity.passed {
        reject_reason = Some(match reject_reason {
            Some(prev) => format!(
                "{prev}; parity fail: max_err={:.6} > tolerance={:.6}",
                parity.max_abs_err, parity_cfg.tolerance
            ),
            None => format!(
                "parity fail: max_err={:.6} > tolerance={:.6}",
                parity.max_abs_err, parity_cfg.tolerance
            ),
        });
    }

    let passed = reject_reason.is_none();

    Ok(StabilityResult {
        passed,
        median_times_us: median_per_set,
        variance_pct,
        parity_result: Some(parity),
        reject_reason,
    })
}

/// CPU-only stability verdict from pre-computed timing samples. Used by
/// the unit tests — they don't need to hit the GPU to prove that the
/// variance math + reject-reason formatting are correct.
pub fn stability_verdict_from_times(
    times_us: &[f64],
    parity: Option<ParityResult>,
    cfg: &StabilityConfig,
) -> StabilityResult {
    let overall_median = median(times_us);
    let variance_pct = if overall_median > 0.0 {
        let max_dev = times_us
            .iter()
            .map(|t| (t - overall_median).abs())
            .fold(0.0f64, f64::max);
        (max_dev / overall_median) * 100.0
    } else {
        100.0
    };

    let mut reject_reason: Option<String> = None;
    if variance_pct >= cfg.max_variance_pct {
        reject_reason = Some(format!(
            "variance {:.2}% ≥ {:.2}% threshold",
            variance_pct, cfg.max_variance_pct
        ));
    }
    if let Some(p) = &parity {
        if !p.passed {
            reject_reason = Some(match reject_reason {
                Some(prev) => format!("{prev}; parity fail: max_err={:.6}", p.max_abs_err),
                None => format!("parity fail: max_err={:.6}", p.max_abs_err),
            });
        }
    }

    StabilityResult {
        passed: reject_reason.is_none(),
        median_times_us: vec![overall_median],
        variance_pct,
        parity_result: parity,
        reject_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec() {
        let c = StabilityConfig::default();
        assert_eq!(c.n_input_sets, 3);
        assert_eq!(c.n_runs_per_set, 10);
        assert!((c.max_variance_pct - 2.0).abs() < 0.01);
        assert_eq!(c.parity_n_blocks, 1000);
    }

    #[test]
    fn low_variance_passes() {
        let times: Vec<f64> = (0..30).map(|i| 100.0 + (i % 3) as f64 * 0.5).collect();
        let v = stability_verdict_from_times(&times, None, &StabilityConfig::default());
        assert!(v.passed, "variance {:.2}% should pass", v.variance_pct);
        assert!(v.variance_pct < 2.0);
    }

    #[test]
    fn high_variance_rejects() {
        let mut times: Vec<f64> = vec![100.0; 29];
        times.push(110.0); // one outlier → 10 % deviation
        let v = stability_verdict_from_times(&times, None, &StabilityConfig::default());
        assert!(!v.passed);
        assert!(v.reject_reason.as_ref().unwrap().contains("variance"));
    }

    #[test]
    fn parity_fail_rejects_even_with_low_variance() {
        use super::super::parity::ParityResult;
        let times: Vec<f64> = vec![100.0; 30];
        let bad_parity = ParityResult {
            passed: false,
            max_abs_err: 0.05,
            mean_abs_err: 0.01,
            n_blocks_tested: 1000,
            violations: Vec::new(),
            effective_tolerance: 0.001,
        };
        let v =
            stability_verdict_from_times(&times, Some(bad_parity), &StabilityConfig::default());
        assert!(!v.passed);
        assert!(v.reject_reason.as_ref().unwrap().contains("parity"));
    }
}
