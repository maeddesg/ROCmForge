//! Phase 1 / Schritt 1.2 — FFI layer smoke tests.
//!
//! Runs against the real GPU: exercises device detection, buffer
//! alloc/free, H2D/D2H, streams, events, and mapped host memory.
//! Gated by `required-features = ["v1", "gpu"]` in the root Cargo.toml.

#![cfg(all(feature = "v1", feature = "gpu"))]

use rocmforge::v1::backend::gpu::device::GpuDevice;
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipEvent, HipHostBuffer, HipStream};

#[test]
fn test_hip_device_detection() {
    let device = GpuDevice::detect(0).expect("no HIP-capable device visible at index 0");
    assert_eq!(
        device.warp_size, 32,
        "expected wave32 on RDNA 4, got {}",
        device.warp_size
    );
    assert!(device.compute_units > 0, "compute_units must be positive");
    assert!(device.total_memory > 0, "total_memory must be positive");
    println!(
        "GPU: {} (gcnArch='{}', CUs={}, VRAM={} MB, warp={})",
        device.name,
        device.gcn_arch_name,
        device.compute_units,
        device.total_memory / 1024 / 1024,
        device.warp_size
    );
    device.verify_gfx1201().expect("expected gfx1201 target");
}

#[test]
fn test_hip_malloc_free() {
    let buf = HipBuffer::new(1024 * 1024).expect("hipMalloc failed");
    assert_eq!(buf.size(), 1024 * 1024);
    assert!(!buf.as_ptr().is_null(), "hipMalloc returned null ptr");
    // Drop runs hipFree.
}

#[test]
fn test_hip_memcpy_roundtrip() {
    let data: Vec<u8> = (0..=255).collect();
    let mut buf = HipBuffer::new(data.len()).expect("hipMalloc failed");
    buf.copy_from_host(&data).expect("H2D failed");
    let mut readback = vec![0u8; data.len()];
    buf.copy_to_host(&mut readback).expect("D2H failed");
    assert_eq!(data, readback, "memcpy roundtrip lost data");
}

#[test]
fn test_hip_stream_create_destroy() {
    let stream = HipStream::new().expect("hipStreamCreate failed");
    stream.synchronize().expect("hipStreamSynchronize failed");
}

#[test]
fn test_hip_event_timing() {
    let stream = HipStream::new().expect("hipStreamCreate failed");
    let start = HipEvent::new().expect("hipEventCreate failed (start)");
    let stop = HipEvent::new().expect("hipEventCreate failed (stop)");

    // Surround a tiny H2D copy with event records so the timer reads
    // something non-degenerate but the test stays hardware-cheap.
    let src = vec![0u8; 4096];
    let mut buf = HipBuffer::new(4096).expect("hipMalloc failed");

    start.record(&stream).expect("event record (start) failed");
    buf.copy_from_host_async(&src, &stream)
        .expect("async H2D failed");
    stop.record(&stream).expect("event record (stop) failed");
    stop.synchronize().expect("event sync failed");

    let elapsed = HipEvent::elapsed_ms(&start, &stop).expect("elapsed failed");
    assert!(
        elapsed >= 0.0 && elapsed.is_finite(),
        "elapsed_ms must be finite and non-negative, got {elapsed}"
    );
    println!("Event timing over 4 KiB H2D: {elapsed:.3} ms");
}

#[test]
fn test_hip_host_buffer_mapped() {
    let mut host_buf = HipHostBuffer::new_mapped(4096).expect("hipHostMalloc mapped failed");
    assert_eq!(host_buf.size(), 4096);
    let last_idx = 4096 / std::mem::size_of::<u32>() - 1;
    {
        let slice = host_buf.as_mut_slice::<u32>();
        assert_eq!(slice.len(), last_idx + 1);
        slice[0] = 42;
        slice[last_idx] = 0xDEAD_BEEF;
    }
    let view = host_buf.as_slice::<u32>();
    assert_eq!(view[0], 42);
    assert_eq!(view[last_idx], 0xDEAD_BEEF);
}
