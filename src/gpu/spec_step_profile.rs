//! Spec-step cost breakdown profiler using HIP Events.
//!
//! Gated behind `ROCMFORGE_PROFILE_SPEC_STEP=1`. When enabled, instruments
//! `gpu_speculative_decode_step()` to measure 5 phases:
//!   1. draft_forward    — drafting N tokens with the small model
//!   2. target_verify    — batched verify through the target model
//!   3. host_device_sync — hipDeviceSynchronize / D2H copies between phases
//!   4. accept_reject    — greedy accept/reject comparison + KV correction
//!   5. host_overhead    — Vec allocation, token bookkeeping, env checks
//!
//! Output is printed to stderr in a machine-parseable format when the
//! generation completes.

use super::device::GpuDevice;
use super::error::GpuResult;
use super::ffi;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub(crate) const PROFILE_SPEC_STEP_ENV: &str = "ROCMFORGE_PROFILE_SPEC_STEP";

const ENV_UNKNOWN: u8 = 0;
const ENV_DISABLED: u8 = 1;
const ENV_ENABLED: u8 = 2;

static PROFILE_SPEC_STEP_FLAG: AtomicU8 = AtomicU8::new(ENV_UNKNOWN);

pub fn spec_step_profiling_enabled() -> bool {
    match PROFILE_SPEC_STEP_FLAG.load(Ordering::Relaxed) {
        ENV_DISABLED => false,
        ENV_ENABLED => true,
        _ => {
            let enabled = std::env::var_os(PROFILE_SPEC_STEP_ENV).is_some();
            PROFILE_SPEC_STEP_FLAG.store(
                if enabled { ENV_ENABLED } else { ENV_DISABLED },
                Ordering::Relaxed,
            );
            enabled
        }
    }
}

pub(crate) fn refresh_spec_step_profile_env_flag() {
    PROFILE_SPEC_STEP_FLAG.store(ENV_UNKNOWN, Ordering::Relaxed);
}

/// Accumulated timing for one phase across all spec steps.
#[derive(Clone, Copy, Debug, Default)]
struct PhaseTiming {
    total_us: f64,
    count: u64,
}

/// Accumulated spec-step profiling data.
#[derive(Clone, Debug, Default)]
pub struct SpecStepProfileData {
    pub draft_forward_us: f64,
    pub target_verify_us: f64,
    pub accept_reject_us: f64,
    pub host_overhead_us: f64,
    pub total_step_us: f64,
    pub n_steps: u64,
}

/// Internal accumulator — mutex-protected.
#[derive(Debug, Default)]
struct ProfileAccumulator {
    draft_forward: PhaseTiming,
    target_verify: PhaseTiming,
    accept_reject: PhaseTiming,
    host_overhead: PhaseTiming,
    total_step: PhaseTiming,
}

fn profile_store() -> &'static Mutex<ProfileAccumulator> {
    static STORE: OnceLock<Mutex<ProfileAccumulator>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(ProfileAccumulator::default()))
}

/// RAII timer for a single spec step. Records HIP events for GPU phases
/// and host Instants for CPU phases.
pub(crate) struct SpecStepTimer {
    // HIP events for GPU-side timing
    step_start: ffi::hipEvent_t,
    draft_end: ffi::hipEvent_t,
    verify_end: ffi::hipEvent_t,
    // Host-side timing for accept/reject + overhead
    accept_reject_us: f64,
    host_overhead_us: f64,
    host_phase_start: Instant,
    stream: ffi::hipStream_t,
}

impl SpecStepTimer {
    /// Create and record the step-start event. Returns None if profiling disabled.
    pub(crate) fn begin(device: &GpuDevice) -> GpuResult<Option<Self>> {
        if !spec_step_profiling_enabled() {
            return Ok(None);
        }

        let step_start = ffi::hip_event_create()?;
        let draft_end = ffi::hip_event_create()?;
        let verify_end = ffi::hip_event_create()?;
        let stream = device.stream();

        ffi::hip_event_record(step_start, stream)?;

        Ok(Some(Self {
            step_start,
            draft_end,
            verify_end,
            accept_reject_us: 0.0,
            host_overhead_us: 0.0,
            host_phase_start: Instant::now(),
            stream,
        }))
    }

    /// Mark end of draft forward phase (GPU event).
    pub(crate) fn mark_draft_end(&mut self) -> GpuResult<()> {
        ffi::hip_event_record(self.draft_end, self.stream)
    }

    /// Mark end of target verify phase (GPU event).
    pub(crate) fn mark_verify_end(&mut self) -> GpuResult<()> {
        ffi::hip_event_record(self.verify_end, self.stream)
    }

    /// Start timing a host-side phase (accept/reject or overhead).
    pub(crate) fn host_phase_begin(&mut self) {
        self.host_phase_start = Instant::now();
    }

    /// End accept/reject phase timing.
    pub(crate) fn host_phase_end_accept_reject(&mut self) {
        self.accept_reject_us += self.host_phase_start.elapsed().as_secs_f64() * 1_000_000.0;
    }

    /// End host overhead phase timing.
    pub(crate) fn host_phase_end_overhead(&mut self) {
        self.host_overhead_us += self.host_phase_start.elapsed().as_secs_f64() * 1_000_000.0;
    }

    /// Finalize: synchronize events, compute elapsed times, accumulate.
    pub(crate) fn finish(self) -> GpuResult<()> {
        // Synchronize on the last GPU event to ensure all work is done
        ffi::hip_event_synchronize(self.verify_end)?;

        let draft_ms = ffi::hip_event_elapsed_time(self.step_start, self.draft_end)?;
        let verify_ms = ffi::hip_event_elapsed_time(self.draft_end, self.verify_end)?;

        let draft_us = draft_ms as f64 * 1000.0;
        let verify_us = verify_ms as f64 * 1000.0;
        let total_us = draft_us + verify_us + self.accept_reject_us + self.host_overhead_us;

        let mut guard = profile_store()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        guard.draft_forward.total_us += draft_us;
        guard.draft_forward.count += 1;
        guard.target_verify.total_us += verify_us;
        guard.target_verify.count += 1;
        guard.accept_reject.total_us += self.accept_reject_us;
        guard.accept_reject.count += 1;
        guard.host_overhead.total_us += self.host_overhead_us;
        guard.host_overhead.count += 1;
        guard.total_step.total_us += total_us;
        guard.total_step.count += 1;

        // Destroy events
        ffi::hip_event_destroy(self.step_start)?;
        ffi::hip_event_destroy(self.draft_end)?;
        ffi::hip_event_destroy(self.verify_end)?;

        Ok(())
    }
}

/// Get a snapshot of accumulated profiling data.
pub fn spec_step_profile_snapshot() -> SpecStepProfileData {
    let guard = profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    SpecStepProfileData {
        draft_forward_us: guard.draft_forward.total_us,
        target_verify_us: guard.target_verify.total_us,
        accept_reject_us: guard.accept_reject.total_us,
        host_overhead_us: guard.host_overhead.total_us,
        total_step_us: guard.total_step.total_us,
        n_steps: guard.total_step.count,
    }
}

/// Reset accumulated profiling data.
pub fn reset_spec_step_profile() {
    let mut guard = profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    *guard = ProfileAccumulator::default();
}

/// Print a summary table to stderr.
pub fn print_spec_step_profile_summary() {
    if !spec_step_profiling_enabled() {
        return;
    }

    let data = spec_step_profile_snapshot();
    if data.n_steps == 0 {
        return;
    }

    let n = data.n_steps as f64;
    let avg_draft = data.draft_forward_us / n;
    let avg_verify = data.target_verify_us / n;
    let avg_accept = data.accept_reject_us / n;
    let avg_overhead = data.host_overhead_us / n;
    let avg_total = data.total_step_us / n;

    let pct = |v: f64| -> f64 { v / avg_total * 100.0 };

    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║           Spec-Step Cost Breakdown ({} steps)          ║", data.n_steps);
    eprintln!("╠══════════════════════════════════════════════════════════╣");
    eprintln!(
        "║ draft_forward     {:>8.0} μs  ({:>5.1}%)                 ║",
        avg_draft,
        pct(avg_draft)
    );
    eprintln!(
        "║ target_verify     {:>8.0} μs  ({:>5.1}%)                 ║",
        avg_verify,
        pct(avg_verify)
    );
    eprintln!(
        "║ accept_reject     {:>8.0} μs  ({:>5.1}%)                 ║",
        avg_accept,
        pct(avg_accept)
    );
    eprintln!(
        "║ host_overhead     {:>8.0} μs  ({:>5.1}%)                 ║",
        avg_overhead,
        pct(avg_overhead)
    );
    eprintln!("╠══════════════════════════════════════════════════════════╣");
    eprintln!(
        "║ TOTAL             {:>8.0} μs                            ║",
        avg_total
    );
    eprintln!("╚══════════════════════════════════════════════════════════╝");

    // Machine-parseable JSON line
    eprintln!(
        "[SPEC_PROFILE_JSON] {{\"n_steps\":{},\"avg_draft_us\":{:.1},\"avg_verify_us\":{:.1},\"avg_accept_us\":{:.1},\"avg_overhead_us\":{:.1},\"avg_total_us\":{:.1}}}",
        data.n_steps, avg_draft, avg_verify, avg_accept, avg_overhead, avg_total
    );
}
