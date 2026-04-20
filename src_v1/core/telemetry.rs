//! Telemetry for profiling and diagnostics.
//!
//! Two modes:
//! * **Disabled (default)** — every function is a no-op. No
//!   `hipDeviceSynchronize`, no event records, no allocations. Zero
//!   overhead on the forward-pass.
//! * **Enabled (`ROCMFORGE_PROFILE=1`)** — `measure_gpu_op` wraps each
//!   recorded op in an explicit sync + `hipEventRecord` pair, captures
//!   both GPU and wall-clock timings, and builds a per-op breakdown.
//!
//! The `Profiler.enabled` flag is `pub` on purpose so integration
//! tests can bypass the env-var cache.
//!
//! Architecture reference: `architecture_v1.2.0-draft.md` §3.7 (Zero-
//! Sync Pipeline, dirty-flag telemetry) and the "profiling invertiert
//! IMMER die Schätzungen" lesson from §10.1.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Instant;

/// Is `ROCMFORGE_PROFILE=1` set? Computed once and cached.
pub fn profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ROCMFORGE_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// One recorded measurement: GPU time from `hipEventElapsedTime`, wall
/// time from `Instant`. The difference between them attributes CPU
/// dispatch overhead.
#[derive(Debug, Clone)]
pub struct OpTiming {
    pub name: String,
    pub gpu_us: f64,
    pub wall_us: f64,
}

/// Collects timings and dispatch counts across a forward pass.
///
/// Zero-overhead contract
/// ----------------------
/// When `enabled == false`, every observer method must short-circuit.
/// In particular:
/// * `record` / `count_dispatch` must not allocate on the hot path.
/// * The caller-side `measure_gpu_op` must never be invoked; the
///   `profile_op!` macro encodes this by calling the op directly.
pub struct Profiler {
    pub enabled: bool,
    timings: Vec<OpTiming>,
    dispatch_counts: HashMap<String, u64>,
    pass_start: Instant,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            enabled: profiling_enabled(),
            timings: Vec::new(),
            dispatch_counts: HashMap::new(),
            pass_start: Instant::now(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn count_dispatch(&mut self, kernel_name: &str) {
        if !self.enabled {
            return;
        }
        *self
            .dispatch_counts
            .entry(kernel_name.to_string())
            .or_insert(0) += 1;
    }

    pub fn record(&mut self, name: String, gpu_us: f64, wall_us: f64) {
        if !self.enabled {
            return;
        }
        self.timings.push(OpTiming {
            name,
            gpu_us,
            wall_us,
        });
    }

    pub fn total_dispatches(&self) -> u64 {
        self.dispatch_counts.values().sum()
    }

    pub fn timings(&self) -> &[OpTiming] {
        &self.timings
    }

    pub fn dispatch_counts(&self) -> &HashMap<String, u64> {
        &self.dispatch_counts
    }

    /// Build a per-op summary report. Ops are returned sorted by
    /// total GPU time (descending).
    pub fn report(&self) -> ProfileReport {
        // Use nanosecond resolution and convert — `as_micros()` truncates
        // sub-microsecond durations to zero, which flakes on fast tests.
        let total_wall_us = self.pass_start.elapsed().as_nanos() as f64 / 1000.0;
        let total_gpu_us: f64 = self.timings.iter().map(|t| t.gpu_us).sum();

        let mut by_op: HashMap<String, (f64, f64, u64)> = HashMap::new();
        for t in &self.timings {
            let entry = by_op.entry(t.name.clone()).or_insert((0.0, 0.0, 0));
            entry.0 += t.gpu_us;
            entry.1 += t.wall_us;
            entry.2 += 1;
        }

        let mut ops: Vec<OpSummary> = by_op
            .into_iter()
            .map(|(name, (gpu, wall, count))| OpSummary {
                name,
                total_gpu_us: gpu,
                total_wall_us: wall,
                count,
                percent: if total_gpu_us > 0.0 {
                    gpu / total_gpu_us * 100.0
                } else {
                    0.0
                },
            })
            .collect();
        ops.sort_by(|a, b| b.total_gpu_us.partial_cmp(&a.total_gpu_us).unwrap_or(std::cmp::Ordering::Equal));

        ProfileReport {
            ops,
            total_gpu_us,
            total_wall_us,
            dispatch_overhead_us: (total_wall_us - total_gpu_us).max(0.0),
            dispatch_counts: self.dispatch_counts.clone(),
            total_dispatches: self.total_dispatches(),
        }
    }

    /// Reset counters and timings for a new forward pass. The `enabled`
    /// state is preserved.
    pub fn reset(&mut self) {
        self.timings.clear();
        self.dispatch_counts.clear();
        self.pass_start = Instant::now();
    }
}

#[derive(Debug, Clone)]
pub struct OpSummary {
    pub name: String,
    pub total_gpu_us: f64,
    pub total_wall_us: f64,
    pub count: u64,
    pub percent: f64,
}

#[derive(Debug, Clone)]
pub struct ProfileReport {
    pub ops: Vec<OpSummary>,
    pub total_gpu_us: f64,
    pub total_wall_us: f64,
    pub dispatch_overhead_us: f64,
    pub dispatch_counts: HashMap<String, u64>,
    pub total_dispatches: u64,
}

impl ProfileReport {
    /// Human-readable breakdown matching the v0.x-style decode profile
    /// output. Columns: operation · GPU µs · percent · count.
    pub fn print(&self) {
        println!("\n=== Profile Report ===");
        println!("{:<30} {:>12} {:>7} {:>8}", "Operation", "GPU µs", "%", "Count");
        println!("{}", "-".repeat(60));
        for op in &self.ops {
            println!(
                "{:<30} {:>12.0} {:>6.1}% {:>8}",
                op.name, op.total_gpu_us, op.percent, op.count
            );
        }
        println!("{}", "-".repeat(60));
        println!(
            "{:<30} {:>12.0}         {:>8}",
            "TOTAL (GPU)", self.total_gpu_us, self.total_dispatches
        );
        println!("{:<30} {:>12.0}", "Wall-Clock µs", self.total_wall_us);
        let pct = if self.total_wall_us > 0.0 {
            self.dispatch_overhead_us / self.total_wall_us * 100.0
        } else {
            0.0
        };
        println!(
            "{:<30} {:>12.0} ({:>5.1}%)",
            "Dispatch overhead (µs)", self.dispatch_overhead_us, pct
        );

        println!("\nDispatches per kernel:");
        let mut by_count: Vec<_> = self.dispatch_counts.iter().collect();
        by_count.sort_by(|a, b| b.1.cmp(a.1));
        for (name, count) in by_count {
            println!("  {:<40} {:>8}", name, count);
        }
    }
}

// ----------------------------------------------------------------------
// GPU-specific telemetry (timing + dirty flags). Gated on the `gpu`
// feature so a `--features v1` (no gpu) build still compiles the
// CPU-only Profiler above.
// ----------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod gpu_telemetry {
    use super::*;
    use crate::v1::backend::gpu::error::HipResult;
    use crate::v1::backend::gpu::wrappers::{HipEvent, HipHostBuffer, HipStream};

    /// Measure a GPU operation with real `hipEvent` timings.
    ///
    /// Calls `hipStreamSynchronize` **before** the op (so the start
    /// event is recorded on a quiesced stream) and after the op
    /// completes (so elapsed time includes only this op's kernels).
    /// That synchronisation is by design — it's what makes the timing
    /// accurate — so this function must not be invoked when profiling
    /// is disabled. The `profile_op!` macro enforces that contract.
    ///
    /// Returns `(gpu_us, wall_us)` so callers can attribute dispatch
    /// overhead by taking the difference.
    pub fn measure_gpu_op<F>(stream: &HipStream, op: F) -> HipResult<(f64, f64)>
    where
        F: FnOnce() -> HipResult<()>,
    {
        stream.synchronize()?;

        let start_event = HipEvent::new()?;
        let stop_event = HipEvent::new()?;

        let wall_start = Instant::now();
        start_event.record(stream)?;
        op()?;
        stop_event.record(stream)?;
        stop_event.synchronize()?;

        let wall_us = wall_start.elapsed().as_nanos() as f64 / 1000.0;
        let gpu_ms = HipEvent::elapsed_ms(&start_event, &stop_event)?;
        Ok((gpu_ms as f64 * 1000.0, wall_us))
    }

    /// Dirty-flag telemetry backed by `hipHostMallocMapped`: one
    /// allocation, 8 `u32` slots, reachable from both CPU and GPU
    /// without an explicit sync (on RDNA 4 the mapped pointer is
    /// directly usable as a device pointer).
    ///
    /// Slot assignment
    /// ---------------
    /// 0. layer progress (current layer index)
    /// 1. NaN detected (1 = yes)
    /// 2. Inf detected (1 = yes)
    /// 3. magnitude drop (1 = hidden-state norm below threshold)
    /// 4. quality monitor trigger bitfield
    /// 5..8. reserved for future use
    pub struct DirtyFlags {
        host_buffer: HipHostBuffer,
    }

    impl DirtyFlags {
        pub const SLOT_LAYER_PROGRESS: usize = 0;
        pub const SLOT_NAN_DETECTED: usize = 1;
        pub const SLOT_INF_DETECTED: usize = 2;
        pub const SLOT_MAGNITUDE_DROP: usize = 3;
        pub const SLOT_QUALITY_TRIGGER: usize = 4;
        pub const NUM_SLOTS: usize = 8;

        pub fn new() -> HipResult<Self> {
            let size = Self::NUM_SLOTS * std::mem::size_of::<u32>();
            let host_buffer = HipHostBuffer::new_mapped(size)?;
            let mut this = Self { host_buffer };
            this.reset();
            Ok(this)
        }

        pub fn layer_progress(&self) -> u32 {
            self.host_buffer.as_slice::<u32>()[Self::SLOT_LAYER_PROGRESS]
        }

        pub fn nan_detected(&self) -> bool {
            self.host_buffer.as_slice::<u32>()[Self::SLOT_NAN_DETECTED] != 0
        }

        pub fn inf_detected(&self) -> bool {
            self.host_buffer.as_slice::<u32>()[Self::SLOT_INF_DETECTED] != 0
        }

        pub fn magnitude_drop(&self) -> bool {
            self.host_buffer.as_slice::<u32>()[Self::SLOT_MAGNITUDE_DROP] != 0
        }

        pub fn quality_trigger(&self) -> u32 {
            self.host_buffer.as_slice::<u32>()[Self::SLOT_QUALITY_TRIGGER]
        }

        /// Raw slot read — for callers that need a specific flag the
        /// typed accessors above don't cover (e.g. reserved slots).
        pub fn slot(&self, idx: usize) -> u32 {
            self.host_buffer.as_slice::<u32>()[idx]
        }

        /// Direct mutable access to the 8 slots. The GPU kernel
        /// normally writes here via the mapped pointer — CPU writes are
        /// valid too and used by `reset`, test simulations, and any
        /// caller that needs to clear a single slot.
        pub fn slots_mut(&mut self) -> &mut [u32] {
            self.host_buffer.as_mut_slice::<u32>()
        }

        /// Raw host pointer. Passed to a kernel via its launch-arg
        /// list; on RDNA 4 the mapped pointer doubles as a device
        /// pointer.
        pub fn device_ptr(&self) -> *const std::ffi::c_void {
            self.host_buffer.as_ptr()
        }

        pub fn device_mut_ptr(&mut self) -> *mut std::ffi::c_void {
            self.host_buffer.as_mut_ptr()
        }

        pub fn reset(&mut self) {
            for slot in self.slots_mut().iter_mut() {
                *slot = 0;
            }
        }
    }
}

#[cfg(feature = "gpu")]
pub use gpu_telemetry::{measure_gpu_op, DirtyFlags};

/// Record a GPU operation's timing iff profiling is enabled, otherwise
/// execute it directly. Zero overhead in the disabled path — no sync,
/// no event allocation, no closure indirection (the op expression is
/// inlined as-is).
///
/// Usage:
/// ```ignore
/// profile_op!(profiler, stream, "rms_norm", {
///     launch_rms_norm(&stream)?;
///     Ok(())
/// });
/// ```
#[cfg(feature = "gpu")]
#[macro_export]
macro_rules! profile_op {
    ($profiler:expr, $stream:expr, $name:expr, $op:block) => {{
        let _result: $crate::v1::backend::gpu::error::HipResult<()> =
            if $profiler.is_enabled() {
                match $crate::v1::core::telemetry::measure_gpu_op($stream, || $op) {
                    Ok((gpu_us, wall_us)) => {
                        $profiler.record($name.to_string(), gpu_us, wall_us);
                        $profiler.count_dispatch($name);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            } else {
                (|| $op)()
            };
        _result
    }};
}
