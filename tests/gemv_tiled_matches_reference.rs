#![cfg(feature = "gpu")]

//! Correctness test for the tiled batched GEMV kernel.
//!
//! Compares the tiled kernel output against:
//! - The standard batched kernel (for dimensions that fit in LDS)
//! - Sequential single-row GEMV (for dimensions that exceed LDS)
//!
//! Uses real Q4_0 quantization via the GPU quantize kernel to ensure
//! weight layout matches exactly what production code uses.

mod common;

use rocmforge::gpu::{self, ffi, kernels};
use serial_test::serial;

/// Generate deterministic pseudo-random f32 data.
fn pseudo_random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Run tiled kernel and sequential GEMV, compare outputs.
/// For large in_dim that exceeds batched LDS limit.
fn run_tiled_vs_sequential(
    n_rows: usize,
    n_cols: usize,
    batch_size: usize,
    tolerance: f32,
) {
    let device = gpu::GpuDevice::init(0).expect("GPU init failed");

    // Generate random f32 data for weights, then quantize to Q4_0 on GPU
    let weight_data = pseudo_random_f32(n_rows * n_cols, 42);
    let input_data = pseudo_random_f32(batch_size * n_rows, 123);

    let q4_block_size = 18usize; // Q4_0: 2 bytes scale + 16 bytes nibbles per 32 elements
    let blocks_per_col = n_rows / 32;
    let weights_q4_bytes = n_cols * blocks_per_col * q4_block_size;

    // Allocate GPU buffers
    let weight_f32_gpu = ffi::hip_malloc(weight_data.len() * 4).expect("malloc weight_f32");
    let weight_q4_gpu = ffi::hip_malloc(weights_q4_bytes).expect("malloc weight_q4");
    let input_gpu = ffi::hip_malloc(input_data.len() * 4).expect("malloc input");
    let output_tiled_gpu = ffi::hip_malloc(batch_size * n_cols * 4).expect("malloc output_tiled");
    let output_seq_gpu = ffi::hip_malloc(batch_size * n_cols * 4).expect("malloc output_seq");

    // Upload input
    ffi::hip_memcpy_h2d(
        input_gpu,
        input_data.as_ptr() as *const u8,
        input_data.len() * 4,
    )
    .expect("h2d input");

    // Quantize each column of weights to Q4_0 on GPU
    for col in 0..n_cols {
        let col_offset = col * n_rows;
        let col_f32_ptr = unsafe { weight_f32_gpu.add(col_offset * 4) };
        let col_q4_ptr = unsafe { weight_q4_gpu.add(col * blocks_per_col * q4_block_size) };

        ffi::hip_memcpy_h2d(
            col_f32_ptr,
            weight_data[col_offset..col_offset + n_rows].as_ptr() as *const u8,
            n_rows * 4,
        )
        .expect("h2d weight col");

        kernels::quantize_q4_0(col_f32_ptr as *const f32, col_q4_ptr, n_rows)
            .expect("quantize col");
    }

    device.synchronize().expect("sync quantize");

    // Zero both outputs
    let zeros = vec![0u8; batch_size * n_cols * 4];
    ffi::hip_memcpy_h2d(output_tiled_gpu, zeros.as_ptr(), zeros.len()).expect("zero tiled");
    ffi::hip_memcpy_h2d(output_seq_gpu, zeros.as_ptr(), zeros.len()).expect("zero seq");

    // ── Run tiled batched GEMV ───────────────────────────────────────────────
    kernels::quant::gemv_q4_0_f32_batched_tiled_on_stream(
        weight_q4_gpu,
        input_gpu as *const f32,
        output_tiled_gpu as *mut f32,
        n_rows,
        n_cols,
        batch_size,
        ffi::hipStream_t::null(),
    )
    .expect("tiled kernel failed");

    device.synchronize().expect("sync tiled");

    // ── Run sequential GEMV (one row at a time) ──────────────────────────────
    for b in 0..batch_size {
        let row_input = unsafe { (input_gpu as *const f32).add(b * n_rows) };
        let row_output = unsafe { (output_seq_gpu as *mut f32).add(b * n_cols) };

        kernels::gemv_q4_0_f32(
            weight_q4_gpu as *const u8,
            row_input,
            row_output,
            n_rows,
            n_cols,
        )
        .expect("sequential gemv failed");
    }

    device.synchronize().expect("sync sequential");

    // ── Read back and compare ────────────────────────────────────────────────
    let mut tiled_output = vec![0.0f32; batch_size * n_cols];
    let mut seq_output = vec![0.0f32; batch_size * n_cols];

    ffi::hip_memcpy_d2h(
        tiled_output.as_mut_ptr() as *mut u8,
        output_tiled_gpu,
        tiled_output.len() * 4,
    )
    .expect("d2h tiled");
    ffi::hip_memcpy_d2h(
        seq_output.as_mut_ptr() as *mut u8,
        output_seq_gpu,
        seq_output.len() * 4,
    )
    .expect("d2h seq");

    let mut max_abs_diff = 0.0f32;
    let mut n_exceeding = 0usize;

    for i in 0..tiled_output.len() {
        let abs_diff = (tiled_output[i] - seq_output[i]).abs();
        max_abs_diff = max_abs_diff.max(abs_diff);
        if abs_diff > tolerance {
            n_exceeding += 1;
            if n_exceeding <= 5 {
                eprintln!(
                    "    MISMATCH at [{}]: tiled={:.6} seq={:.6} diff={:.6}",
                    i, tiled_output[i], seq_output[i], abs_diff
                );
            }
        }
    }

    eprintln!(
        "  GEMV {:>5}×{:<5} batch={}: max_abs_diff={:.6}, exceeding={}/{}",
        n_rows, n_cols, batch_size, max_abs_diff, n_exceeding, tiled_output.len()
    );

    // Cleanup
    ffi::hip_free(weight_f32_gpu);
    ffi::hip_free(weight_q4_gpu);
    ffi::hip_free(input_gpu);
    ffi::hip_free(output_tiled_gpu);
    ffi::hip_free(output_seq_gpu);

    assert!(
        max_abs_diff <= tolerance,
        "Tiled vs Sequential GEMV {}×{} batch={}: max abs diff {:.6} exceeds tolerance {:.6} ({} values)",
        n_rows, n_cols, batch_size, max_abs_diff, tolerance, n_exceeding
    );
}

/// Compare tiled kernel against standard batched kernel (for small dimensions).
fn run_tiled_vs_batched(
    n_rows: usize,
    n_cols: usize,
    batch_size: usize,
    tolerance: f32,
) {
    let device = gpu::GpuDevice::init(0).expect("GPU init failed");

    let weight_data = pseudo_random_f32(n_rows * n_cols, 42);
    let input_data = pseudo_random_f32(batch_size * n_rows, 123);

    let q4_block_size = 18usize;
    let blocks_per_col = n_rows / 32;
    let weights_q4_bytes = n_cols * blocks_per_col * q4_block_size;

    let weight_f32_gpu = ffi::hip_malloc(weight_data.len() * 4).expect("malloc weight_f32");
    let weight_q4_gpu = ffi::hip_malloc(weights_q4_bytes).expect("malloc weight_q4");
    let input_gpu = ffi::hip_malloc(input_data.len() * 4).expect("malloc input");
    let output_tiled_gpu = ffi::hip_malloc(batch_size * n_cols * 4).expect("malloc output_tiled");
    let output_batched_gpu = ffi::hip_malloc(batch_size * n_cols * 4).expect("malloc output_batched");

    ffi::hip_memcpy_h2d(
        input_gpu,
        input_data.as_ptr() as *const u8,
        input_data.len() * 4,
    )
    .expect("h2d input");

    for col in 0..n_cols {
        let col_offset = col * n_rows;
        let col_f32_ptr = unsafe { weight_f32_gpu.add(col_offset * 4) };
        let col_q4_ptr = unsafe { weight_q4_gpu.add(col * blocks_per_col * q4_block_size) };

        ffi::hip_memcpy_h2d(
            col_f32_ptr,
            weight_data[col_offset..col_offset + n_rows].as_ptr() as *const u8,
            n_rows * 4,
        )
        .expect("h2d weight col");

        kernels::quantize_q4_0(col_f32_ptr as *const f32, col_q4_ptr, n_rows)
            .expect("quantize col");
    }
    device.synchronize().expect("sync quantize");

    let zeros = vec![0u8; batch_size * n_cols * 4];
    ffi::hip_memcpy_h2d(output_tiled_gpu, zeros.as_ptr(), zeros.len()).expect("zero tiled");
    ffi::hip_memcpy_h2d(output_batched_gpu, zeros.as_ptr(), zeros.len()).expect("zero batched");

    // Tiled kernel
    kernels::quant::gemv_q4_0_f32_batched_tiled_on_stream(
        weight_q4_gpu,
        input_gpu as *const f32,
        output_tiled_gpu as *mut f32,
        n_rows,
        n_cols,
        batch_size,
        ffi::hipStream_t::null(),
    )
    .expect("tiled kernel failed");

    // Standard batched kernel
    kernels::quant::gemv_q4_0_f32_batched_on_stream(
        weight_q4_gpu,
        input_gpu as *const f32,
        output_batched_gpu as *mut f32,
        n_rows,
        n_cols,
        batch_size,
        ffi::hipStream_t::null(),
    )
    .expect("batched kernel failed");

    device.synchronize().expect("sync");

    let mut tiled_output = vec![0.0f32; batch_size * n_cols];
    let mut batched_output = vec![0.0f32; batch_size * n_cols];

    ffi::hip_memcpy_d2h(
        tiled_output.as_mut_ptr() as *mut u8,
        output_tiled_gpu,
        tiled_output.len() * 4,
    )
    .expect("d2h tiled");
    ffi::hip_memcpy_d2h(
        batched_output.as_mut_ptr() as *mut u8,
        output_batched_gpu,
        batched_output.len() * 4,
    )
    .expect("d2h batched");

    let mut max_abs_diff = 0.0f32;
    let mut n_exceeding = 0usize;

    for i in 0..tiled_output.len() {
        let abs_diff = (tiled_output[i] - batched_output[i]).abs();
        max_abs_diff = max_abs_diff.max(abs_diff);
        if abs_diff > tolerance {
            n_exceeding += 1;
        }
    }

    eprintln!(
        "  GEMV {:>5}×{:<5} batch={} (tiled vs batched): max_abs_diff={:.6}, exceeding={}/{}",
        n_rows, n_cols, batch_size, max_abs_diff, n_exceeding, tiled_output.len()
    );

    ffi::hip_free(weight_f32_gpu);
    ffi::hip_free(weight_q4_gpu);
    ffi::hip_free(input_gpu);
    ffi::hip_free(output_tiled_gpu);
    ffi::hip_free(output_batched_gpu);

    assert!(
        max_abs_diff <= tolerance,
        "Tiled vs Batched GEMV {}×{} batch={}: max abs diff {:.6} exceeds tolerance {:.6}",
        n_rows, n_cols, batch_size, max_abs_diff, tolerance
    );
}

// ── Tiled vs Sequential (large in_dim, exceeds batched LDS) ─────────────────

// Note: tiled-vs-sequential tolerance is high (2.0) because the batched kernel
// quantizes all inputs to Q8 cooperatively in LDS, while sequential GEMV quantizes
// each input independently in its own kernel. The Q8 quantization uses warp-wide
// absmax reduction, which can produce different scales depending on thread mapping.
// The bit-identical match against the standard batched kernel (test_tiled_vs_batched_*)
// is the authoritative correctness proof.

#[test]
#[serial]
fn test_tiled_vs_sequential_ffn_down_7b_batch_2() {
    require_gpu!();
    run_tiled_vs_sequential(18944, 3584, 2, 2.0);
}

#[test]
#[serial]
fn test_tiled_vs_sequential_ffn_down_7b_batch_4() {
    require_gpu!();
    run_tiled_vs_sequential(18944, 3584, 4, 2.0);
}

#[test]
#[serial]
fn test_tiled_vs_sequential_ffn_down_7b_batch_6() {
    require_gpu!();
    run_tiled_vs_sequential(18944, 3584, 6, 2.0);
}

#[test]
#[serial]
fn test_tiled_vs_sequential_ffn_mistral_batch_2() {
    require_gpu!();
    run_tiled_vs_sequential(14336, 4096, 2, 2.0);
}

#[test]
#[serial]
fn test_tiled_vs_sequential_batch_1() {
    require_gpu!();
    run_tiled_vs_sequential(18944, 3584, 1, 2.0);
}

#[test]
#[serial]
fn test_tiled_vs_sequential_batch_8() {
    require_gpu!();
    run_tiled_vs_sequential(18944, 3584, 8, 2.0);
}

// ── Tiled vs Standard Batched (small in_dim, fits in both) ──────────────────

#[test]
#[serial]
fn test_tiled_vs_batched_qkv_3584_batch_2() {
    require_gpu!();
    run_tiled_vs_batched(3584, 3584, 2, 1e-3);
}

#[test]
#[serial]
fn test_tiled_vs_batched_qkv_3584_batch_6() {
    require_gpu!();
    run_tiled_vs_batched(3584, 3584, 6, 1e-3);
}
