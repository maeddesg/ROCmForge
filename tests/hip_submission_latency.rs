//! HIP Submission Latency Micro-Benchmark
//!
//! Measures the per-dispatch overhead of HIP kernel submissions on the default stream.
//! Uses a tiny elementwise add kernel (n=64) as the minimal workload.
//!
//! Three measurements:
//! 1. Host-side submission throughput: wall-clock for N dispatches (no sync between)
//! 2. GPU-side inter-kernel gap: HIP events around batches of dispatches
//! 3. Sync overhead: hipStreamSynchronize after each dispatch
//!
//! Usage: cargo test --release --features gpu --test hip_submission_latency -- --ignored --nocapture --test-threads=1

#[cfg(feature = "gpu")]
mod common;

#[cfg(feature = "gpu")]
use serial_test::serial;

#[cfg(feature = "gpu")]
#[test]
#[ignore]
#[serial]
fn bench_hip_submission_latency() {
    use rocmforge::gpu;
    use std::time::Instant;

    let device = gpu::GpuDevice::init(0).expect("GPU init");

    // Allocate tiny buffers (64 floats = 256 bytes)
    let n: usize = 64;
    let buf_size = n * std::mem::size_of::<f32>();

    let a_ptr = gpu::ffi::hip_malloc(buf_size).expect("malloc a");
    let b_ptr = gpu::ffi::hip_malloc(buf_size).expect("malloc b");
    let c_ptr = gpu::ffi::hip_malloc(buf_size).expect("malloc c");

    // Buffers are uninitialized — fine for latency measurement, we don't care about results.

    let stream = gpu::ffi::hipStream_t::null();
    let iterations = 10_000;

    // ── Warmup ──────────────────────────────────────────────────────────
    for _ in 0..100 {
        gpu::kernels::add_on_stream(
            a_ptr as *const f32,
            b_ptr as *const f32,
            c_ptr as *mut f32,
            n,
            stream,
        ).expect("warmup add");
    }
    gpu::ffi::hip_stream_synchronize(stream).expect("warmup sync");

    // ── Measurement 1: Host-side submission throughput ────────────────
    // Dispatch N kernels as fast as possible (no sync between), then sync once.
    // Wall-clock / N ≈ host-side submission cost.
    let t0 = Instant::now();
    for _ in 0..iterations {
        gpu::kernels::add_on_stream(
            a_ptr as *const f32,
            b_ptr as *const f32,
            c_ptr as *mut f32,
            n,
            stream,
        ).expect("add dispatch");
    }
    let host_submit_elapsed = t0.elapsed();
    gpu::ffi::hip_stream_synchronize(stream).expect("sync after batch");
    let total_with_sync = t0.elapsed();

    let host_submit_us = host_submit_elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let total_with_sync_us = total_with_sync.as_secs_f64() * 1_000_000.0 / iterations as f64;

    // ── Measurement 2: GPU-side inter-kernel gap via HIP Events ──────
    // Record event before and after a batch of dispatches.
    // (end - start) / N = avg GPU-side time per dispatch (execution + gap).
    let batch_sizes = [100, 1000, 10000];
    let mut gpu_side_results = Vec::new();

    for &batch in &batch_sizes {
        let start_ev = gpu::ffi::hip_event_create().expect("event create");
        let end_ev = gpu::ffi::hip_event_create().expect("event create");

        gpu::ffi::hip_event_record(start_ev, stream).expect("record start");
        for _ in 0..batch {
            gpu::kernels::add_on_stream(
                a_ptr as *const f32,
                b_ptr as *const f32,
                c_ptr as *mut f32,
                n,
                stream,
            ).expect("add dispatch");
        }
        gpu::ffi::hip_event_record(end_ev, stream).expect("record end");
        gpu::ffi::hip_event_synchronize(end_ev).expect("event sync");

        let elapsed_ms = gpu::ffi::hip_event_elapsed_time(start_ev, end_ev).expect("elapsed");
        let per_dispatch_us = elapsed_ms as f64 * 1000.0 / batch as f64;
        gpu_side_results.push((batch, per_dispatch_us));

        gpu::ffi::hip_event_destroy(start_ev).expect("destroy");
        gpu::ffi::hip_event_destroy(end_ev).expect("destroy");
    }

    // ── Measurement 3: Sync-after-each-dispatch ──────────────────────
    // Worst case: dispatch + sync each time.
    let sync_iterations = 1000;
    let t0 = Instant::now();
    for _ in 0..sync_iterations {
        gpu::kernels::add_on_stream(
            a_ptr as *const f32,
            b_ptr as *const f32,
            c_ptr as *mut f32,
            n,
            stream,
        ).expect("add");
        gpu::ffi::hip_stream_synchronize(stream).expect("sync");
    }
    let sync_each_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / sync_iterations as f64;

    // ── Measurement 4: Per-launch with individual HIP event pairs ────
    // Most accurate: measure each dispatch individually.
    let individual_iterations = 1000;
    let mut individual_times = Vec::with_capacity(individual_iterations);

    for _ in 0..individual_iterations {
        let s = gpu::ffi::hip_event_create().expect("ev");
        let e = gpu::ffi::hip_event_create().expect("ev");

        gpu::ffi::hip_event_record(s, stream).expect("rec");
        gpu::kernels::add_on_stream(
            a_ptr as *const f32,
            b_ptr as *const f32,
            c_ptr as *mut f32,
            n,
            stream,
        ).expect("add");
        gpu::ffi::hip_event_record(e, stream).expect("rec");
        gpu::ffi::hip_event_synchronize(e).expect("sync");

        let ms = gpu::ffi::hip_event_elapsed_time(s, e).expect("elapsed");
        individual_times.push(ms as f64 * 1000.0); // μs

        gpu::ffi::hip_event_destroy(s).expect("destroy");
        gpu::ffi::hip_event_destroy(e).expect("destroy");
    }

    individual_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = individual_times[individual_times.len() / 2];
    let p5 = individual_times[individual_times.len() * 5 / 100];
    let p95 = individual_times[individual_times.len() * 95 / 100];
    let mean = individual_times.iter().sum::<f64>() / individual_times.len() as f64;

    // ── Output ───────────────────────────────────────────────────────
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║       HIP Submission Latency Micro-Benchmark                 ║");
    eprintln!("║       Kernel: add(n=64), Stream: default                     ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════╣");
    eprintln!("║ 1. Host-side submission (no sync):                           ║");
    eprintln!("║    {:.2} μs/dispatch ({} dispatches)                   ║", host_submit_us, iterations);
    eprintln!("║    Total incl final sync: {:.2} μs/dispatch              ║", total_with_sync_us);
    eprintln!("║                                                               ║");
    eprintln!("║ 2. GPU-side (HIP Events, batched):                           ║");
    for &(batch, per_us) in &gpu_side_results {
        eprintln!("║    batch={:>5}: {:.2} μs/dispatch                         ║", batch, per_us);
    }
    eprintln!("║                                                               ║");
    eprintln!("║ 3. Dispatch+Sync each: {:.2} μs/dispatch                 ║", sync_each_us);
    eprintln!("║                                                               ║");
    eprintln!("║ 4. Individual HIP Event pairs ({} dispatches):            ║", individual_iterations);
    eprintln!("║    mean:   {:.2} μs                                       ║", mean);
    eprintln!("║    median: {:.2} μs                                       ║", median);
    eprintln!("║    p5:     {:.2} μs                                       ║", p5);
    eprintln!("║    p95:    {:.2} μs                                       ║", p95);
    eprintln!("╚═══════════════════════════════════════════════════════════════╝");

    // JSON output
    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let json = format!(
        r#"{{
  "git_sha": "{}",
  "kernel": "add(n=64)",
  "stream": "default",
  "iterations": {},
  "host_submit_us_per_dispatch": {:.3},
  "host_total_with_sync_us_per_dispatch": {:.3},
  "gpu_batched": [{}],
  "dispatch_plus_sync_us": {:.3},
  "individual_event_pairs": {{
    "n": {},
    "mean_us": {:.3},
    "median_us": {:.3},
    "p5_us": {:.3},
    "p95_us": {:.3}
  }}
}}"#,
        git_sha,
        iterations,
        host_submit_us,
        total_with_sync_us,
        gpu_side_results.iter()
            .map(|(b, us)| format!(r#"{{"batch": {}, "us_per_dispatch": {:.3}}}"#, b, us))
            .collect::<Vec<_>>()
            .join(", "),
        sync_each_us,
        individual_iterations,
        mean,
        median,
        p5,
        p95,
    );

    let output_path = format!("profiling/results/hip_submission_latency_{}.json", git_sha);
    std::fs::write(&output_path, &json).expect("write JSON");
    eprintln!("\nWritten: {}", output_path);

    // Cleanup
    gpu::ffi::hip_free(a_ptr);
    gpu::ffi::hip_free(b_ptr);
    gpu::ffi::hip_free(c_ptr);
}
