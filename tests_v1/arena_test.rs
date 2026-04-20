//! Phase 1 / Schritt 1.3 — VRAM arena smoke tests.
//!
//! Eight tests exercising zone layout, alignment, ping-pong, OOM and a
//! real GPU memcpy through an `ArenaSlice`. All GPU-allocating tests
//! are `#[serial]` because they reserve multi-GB slices on a single
//! consumer GPU.

#![cfg(all(feature = "v1", feature = "gpu"))]

use rocmforge::v1::backend::gpu::arena::{ArenaConfig, VramArena};
use rocmforge::v1::backend::gpu::hip_ffi::{
    hipMemcpy, hipMemcpyDeviceToHost, hipMemcpyHostToDevice,
};
use serial_test::serial;

/// Qwen3-8B Q4_K_M shape — used by every test that needs a full arena.
fn test_config_8b() -> ArenaConfig {
    ArenaConfig::from_model(
        5_000_000_000,  // ~4.7 GB weights
        36,             // n_layers
        8,              // n_kv_heads (GQA)
        128,            // head_dim
        8192,           // max_context (8 K for tests)
        2,              // kv element size (FP16)
        4096,           // hidden_dim
        12288,          // ffn_dim
        512,            // max_batch_size
        15_000_000_000, // ~14 GB free VRAM (16 GB minus compositor)
    )
}

#[test]
fn test_arena_config_validation() {
    let config = test_config_8b();
    assert!(
        config.validate().is_ok(),
        "8B config should fit in 14 GB: {:?}",
        config.validate()
    );

    // Way too big: 14 GB of weights plus KV plus scratch cannot fit.
    let bad_config = ArenaConfig::from_model(
        14_000_000_000,
        36,
        8,
        128,
        8192,
        2,
        4096,
        12288,
        512,
        15_000_000_000,
    );
    assert!(
        bad_config.validate().is_err(),
        "14 GB weights in 15 GB VRAM must not validate"
    );
}

#[test]
#[serial]
fn test_arena_allocate_zones() {
    let config = test_config_8b();
    let arena = VramArena::new(config).expect("arena allocation failed");

    assert!(arena.total_size() > 0);
    assert!(arena.weights_remaining() > 0);
    assert!(arena.kv_cache_remaining() > 0);
    assert!(arena.scratch_size() > 0);

    arena.print_layout();
}

#[test]
#[serial]
fn test_arena_zone_no_overlap() {
    let config = test_config_8b();
    let mut arena = VramArena::new(config).expect("arena allocation failed");

    let w1 = arena.alloc_weights(1024 * 1024).expect("w1 alloc");
    let w2 = arena.alloc_weights(2048 * 1024).expect("w2 alloc");

    assert!(
        w1.offset + w1.size <= w2.offset,
        "weight slices overlap: w1=[{}, {}), w2=[{}, {})",
        w1.offset,
        w1.offset + w1.size,
        w2.offset,
        w2.offset + w2.size
    );

    let kv = arena.kv_cache_slice();
    let last_weight_end = w2.offset + w2.size;
    assert!(
        last_weight_end <= kv.offset,
        "weights overlap KV-Cache: weight_end={}, kv_start={}",
        last_weight_end,
        kv.offset
    );

    let pp = arena.ping_pong();
    let kv_end = kv.offset + kv.size;
    assert!(
        kv_end <= pp.input().offset,
        "KV-Cache overlaps scratch: kv_end={}, scratch_start={}",
        kv_end,
        pp.input().offset
    );
    assert!(
        kv_end <= pp.output().offset,
        "KV-Cache overlaps scratch output: kv_end={}, scratch_start={}",
        kv_end,
        pp.output().offset
    );
}

#[test]
#[serial]
fn test_arena_scratch_ping_pong() {
    let config = test_config_8b();
    let mut arena = VramArena::new(config).expect("arena allocation failed");

    let (initial_input, initial_output) = {
        let pp = arena.ping_pong();
        (pp.input(), pp.output())
    };
    assert_ne!(
        initial_input.offset, initial_output.offset,
        "ping-pong buffers must be distinct"
    );

    arena.ping_pong_mut().swap();
    {
        let pp = arena.ping_pong();
        assert_eq!(pp.input().offset, initial_output.offset);
        assert_eq!(pp.output().offset, initial_input.offset);
    }

    arena.ping_pong_mut().swap();
    {
        let pp = arena.ping_pong();
        assert_eq!(pp.input().offset, initial_input.offset);
    }

    arena.ping_pong_mut().swap();
    arena.ping_pong_mut().reset();
    {
        let pp = arena.ping_pong();
        assert_eq!(pp.input().offset, initial_input.offset);
        assert_eq!(pp.output().offset, initial_output.offset);
    }
}

#[test]
#[serial]
fn test_arena_oom_graceful() {
    let config = test_config_8b();
    let mut arena = VramArena::new(config).expect("arena allocation failed");

    let w_remaining = arena.weights_remaining();
    assert!(
        arena.alloc_weights(w_remaining + 1).is_err(),
        "over-allocation in Zone A must fail gracefully"
    );

    let kv_remaining = arena.kv_cache_remaining();
    assert!(
        arena.kv_cache_grow(kv_remaining + 1).is_err(),
        "KV-Cache overflow must fail gracefully"
    );
}

#[test]
#[serial]
fn test_arena_alignment() {
    let config = test_config_8b();
    let mut arena = VramArena::new(config).expect("arena allocation failed");

    for &size in &[137usize, 1000, 4096, 7777, 65536] {
        let slice = arena.alloc_weights(size).expect("alloc failed");
        assert_eq!(
            slice.offset % 256,
            0,
            "slice offset {} not 256-aligned for size {}",
            slice.offset,
            size
        );
    }

    let kv = arena.kv_cache_slice();
    assert_eq!(kv.offset % 4096, 0, "KV-Cache offset not 4096-aligned");

    let pp = arena.ping_pong();
    assert_eq!(pp.input().offset % 4096, 0, "scratch A not 4096-aligned");
    assert_eq!(pp.output().offset % 4096, 0, "scratch B not 4096-aligned");
}

#[test]
#[serial]
fn test_arena_kv_cache_grow_reset() {
    let config = test_config_8b();
    let mut arena = VramArena::new(config).expect("arena allocation failed");

    assert_eq!(arena.kv_cache_used(), 0);

    let per_token = 2 * 36 * 8 * 128 * 2; // K+V × layers × kv-heads × head_dim × fp16
    arena.kv_cache_grow(per_token).expect("grow 1 failed");
    assert_eq!(arena.kv_cache_used(), per_token);

    arena.kv_cache_grow(per_token).expect("grow 2 failed");
    assert_eq!(arena.kv_cache_used(), 2 * per_token);

    arena.kv_cache_reset();
    assert_eq!(arena.kv_cache_used(), 0);
}

#[test]
#[serial]
fn test_arena_gpu_memcpy_through_slices() {
    let config = test_config_8b();
    let mut arena = VramArena::new(config).expect("arena allocation failed");

    let slice = arena.alloc_weights(1024).expect("alloc failed");
    let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();

    // Host → Device through the arena slice.
    let base_mut = arena.base_mut_ptr();
    let rc = unsafe {
        hipMemcpy(
            slice.as_mut_ptr(base_mut),
            data.as_ptr() as *const _,
            1024,
            hipMemcpyHostToDevice,
        )
    };
    assert_eq!(rc, 0, "hipMemcpy H2D returned {rc}");

    // Device → Host readback.
    let mut readback = vec![0u8; 1024];
    let base = arena.base_ptr();
    let rc = unsafe {
        hipMemcpy(
            readback.as_mut_ptr() as *mut _,
            slice.as_ptr(base),
            1024,
            hipMemcpyDeviceToHost,
        )
    };
    assert_eq!(rc, 0, "hipMemcpy D2H returned {rc}");

    assert_eq!(data, readback, "roundtrip through arena slice lost data");
}
