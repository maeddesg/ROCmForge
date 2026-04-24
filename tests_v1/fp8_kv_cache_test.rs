//! Phase 2.2A — FP8-E5M2 (bf8) KV-cache tests.
//!
//! What we validate:
//!   1. **Roundtrip sanity** — `__builtin_amdgcn_cvt_pk_bf8_f32` +
//!      `__builtin_amdgcn_cvt_f32_bf8` work on gfx1201. Known values
//!      converge within the E5M2 quantum (2 mantissa bits → ~25 % rel
//!      error budget, matching the prompt's expectation).
//!   2. **Append/read layout** — writing FP32 values into the bf8 cache
//!      via `kv_cache_append_fp8` and reading them back through a
//!      direct device→host copy yields the expected byte pattern.
//!   3. **Attention parity** — attention_decode_fp8 vs attention_decode
//!      (FP32) on the same random Q and cache contents. Because the
//!      cache is lossy, outputs differ, but directionally they should
//!      match (cosine similarity > 0.99 on decode-length sequences).
//!   4. **Max-context math** — `KvCacheLayout::bytes_per_layer` halves
//!      then halves again (1/4) for bf8 vs FP32.
//!
//! End-to-end quality (decode coherence, 15-prompt quality gate) is
//! covered by a separate manual run since it's expensive and gated
//! by `ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1`.

#![cfg(all(feature = "v1", feature = "gpu"))]

use rocmforge::v1::backend::gpu::attention::{
    rocmforge_launch_attention_decode, rocmforge_launch_attention_decode_fp8,
    rocmforge_launch_kv_cache_append, rocmforge_launch_kv_cache_append_fp8,
};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
use serial_test::serial;
use std::ffi::c_void;

// ─── Struct-level sanity ───────────────────────────────────────────────────

#[test]
fn test_kv_precision_bytes_per_element() {
    use rocmforge::v1::graph::KvPrecision;
    assert_eq!(KvPrecision::Fp32.bytes_per_element(), 4);
    assert_eq!(KvPrecision::Fp8E5M2.bytes_per_element(), 1);
}

#[test]
fn test_kv_layout_fp8_quarter_size() {
    use rocmforge::v1::graph::{KvCacheLayout, KvPrecision};

    // KvCacheLayout holds its fields directly — we can build one
    // without going through the full ModelConfig::from_metadata path.
    // Qwen3-8B-ish dims: 4 KV heads × 128 head_dim × 2048 max_seq.
    let layout_fp32 = KvCacheLayout {
        num_kv_heads: 4,
        head_dim: 128,
        max_seq: 2048,
        head_stride: 2048 * 128,
        precision: KvPrecision::Fp32,
    };
    let layout_fp8 = KvCacheLayout {
        precision: KvPrecision::Fp8E5M2,
        ..layout_fp32
    };

    assert_eq!(layout_fp32.head_stride, layout_fp8.head_stride);
    assert_eq!(
        layout_fp8.bytes_per_layer() * 4,
        layout_fp32.bytes_per_layer()
    );
    assert_eq!(layout_fp8.bytes_per_side() * 4, layout_fp32.bytes_per_side());
}

// ─── bf8 roundtrip ─────────────────────────────────────────────────────────

fn append_fp8_then_read(values: &[f32]) -> Vec<u8> {
    let n_elems = values.len();
    let stream = HipStream::new().expect("stream");

    // Allocate a 1-byte-per-element cache, big enough for a single
    // "head" of head_dim = n_elems at pos=0.
    let mut d_cache = HipBuffer::new(n_elems).expect("d_cache");
    let dummy = vec![0u8; n_elems];
    d_cache.copy_from_host(&dummy).expect("zero-init");

    let in_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, n_elems * 4) };
    let mut d_in = HipBuffer::new(in_bytes.len()).expect("d_in");
    d_in.copy_from_host(in_bytes).expect("up in");

    // v_cache is required but unused for K-only tests — share with k_cache for simplicity.
    unsafe {
        rocmforge_launch_kv_cache_append_fp8(
            d_cache.as_mut_ptr(),
            d_cache.as_mut_ptr(),
            d_in.as_ptr() as *const f32,
            d_in.as_ptr() as *const f32,
            1,                // num_kv_heads
            n_elems as i32,   // head_dim
            0,                // pos
            n_elems as i32,   // head_stride (elements)
            stream.raw(),
        )
    };
    stream.synchronize().expect("sync");

    let mut out = vec![0u8; n_elems];
    unsafe {
        hipMemcpy(
            out.as_mut_ptr() as *mut _,
            d_cache.as_ptr(),
            n_elems,
            hipMemcpyDeviceToHost,
        );
    }
    out
}

#[test]
#[serial]
fn test_fp8_e5m2_zero_and_symmetry() {
    // Zero should round-trip to zero bit pattern (bf8 0x00 = +0).
    let out = append_fp8_then_read(&[0.0, -0.0]);
    assert_eq!(out[0], 0x00, "positive zero");
    // E5M2 has a sign bit, so -0 encodes as 0x80.
    assert!(
        out[1] == 0x00 || out[1] == 0x80,
        "negative zero: got 0x{:02X}",
        out[1]
    );
}

#[test]
#[serial]
fn test_fp8_e5m2_roundtrip_magnitude() {
    // Use attention_decode_fp8 as the "read side" — it converts per
    // element via cvt_f32_bf8. We reconstruct FP32 and check |Δ|/|x|.
    //
    // Build a 1-head, 1-token "cache" with known values, a Q vector
    // of all-ones (so output becomes normalised sum of V × softmax).
    // Use seq_len=1 so softmax is a passthrough and output = V.
    let values = vec![
        1.0f32, -1.0, 0.5, -0.5, 10.0, -10.0, 100.0, 0.001, 4.0, -4.0, 16.0, 256.0, 1024.0, 0.25,
        0.125, 0.0625,
    ];
    let head_dim = values.len();

    let stream = HipStream::new().expect("stream");
    let mut d_cache = HipBuffer::new(head_dim).expect("d_cache");
    d_cache.copy_from_host(&vec![0u8; head_dim]).expect("zero");

    // quantize-and-write values via kv_cache_append_fp8 at pos=0
    let in_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, head_dim * 4)
    };
    let mut d_fp = HipBuffer::new(in_bytes.len()).expect("d_fp");
    d_fp.copy_from_host(in_bytes).expect("up fp");
    unsafe {
        rocmforge_launch_kv_cache_append_fp8(
            d_cache.as_mut_ptr(),
            d_cache.as_mut_ptr(),
            d_fp.as_ptr() as *const f32,
            d_fp.as_ptr() as *const f32,
            1,
            head_dim as i32,
            0,
            head_dim as i32,
            stream.raw(),
        )
    };

    // Read back: attention with Q = e_d (unit vector per d), seq_len=1,
    // scale=1. Output equals V[d] (bf8 round-tripped).
    let mut q = vec![0f32; head_dim];
    let mut d_q = HipBuffer::new(head_dim * 4).expect("d_q");
    let mut d_out = HipBuffer::new(head_dim * 4).expect("d_out");
    let mut reconstructed = vec![0f32; head_dim];

    for i in 0..head_dim {
        q.iter_mut().for_each(|v| *v = 0.0);
        q[i] = 1.0;
        let q_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(q.as_ptr() as *const u8, head_dim * 4)
        };
        d_q.copy_from_host(q_bytes).expect("up q");

        unsafe {
            rocmforge_launch_attention_decode_fp8(
                d_q.as_ptr() as *const f32,
                d_cache.as_ptr(),
                d_cache.as_ptr(),
                d_out.as_mut_ptr() as *mut f32,
                1,                // num_heads
                1,                // num_kv_heads
                head_dim as i32,
                1,                // seq_len
                head_dim as i32,  // head_stride
                1.0,              // scale
                stream.raw(),
            )
        };
        stream.synchronize().expect("sync");

        let mut out_vec = vec![0f32; head_dim];
        unsafe {
            hipMemcpy(
                out_vec.as_mut_ptr() as *mut _,
                d_out.as_ptr(),
                head_dim * 4,
                hipMemcpyDeviceToHost,
            );
        }
        // The i-th output element = V[i] (round-tripped through bf8).
        // For seq_len=1 and scale=1, softmax is 1 → output = V.
        reconstructed[i] = out_vec[i];
    }

    // Check each value. E5M2 has 2 mantissa bits → ~25 % relative error
    // in the worst case; in practice most values round to within 15 %.
    println!("  FP8-E5M2 roundtrip (| input → output | rel-err):");
    for (i, (&src, &dst)) in values.iter().zip(reconstructed.iter()).enumerate() {
        let denom = src.abs().max(1e-6);
        let rel = (src - dst).abs() / denom;
        println!("    [{i:2}] {src:>10.4} → {dst:>10.4}  ({rel:>6.2}%)");
        assert!(
            rel < 0.30,
            "bf8 rel err {rel:.3} > 30 % for input {src} → {dst}"
        );
    }
}

// ─── Attention parity (FP32 vs FP8-KV) ─────────────────────────────────────

#[test]
#[serial]
fn test_attention_fp32_vs_fp8_kv_cosine() {
    // Realistic-sized single-head attention: head_dim=128, seq_len=32.
    // Populate the cache via the append kernel (FP32 path for both,
    // then FP8 path). Q is random. Compare outputs by cosine sim.
    const HEAD_DIM: usize = 128;
    const SEQ: usize = 32;

    let mut rng = fastrand::Rng::with_seed(0xA55AA55A);
    let q: Vec<f32> = (0..HEAD_DIM).map(|_| rng.f32() * 2.0 - 1.0).collect();
    // Each cache entry is one (k, v) pair per sequence position.
    let all_k: Vec<f32> = (0..SEQ * HEAD_DIM).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let all_v: Vec<f32> = (0..SEQ * HEAD_DIM).map(|_| rng.f32() * 2.0 - 1.0).collect();

    let stream = HipStream::new().expect("stream");
    let q_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, HEAD_DIM * 4) };
    let mut d_q = HipBuffer::new(q_bytes.len()).expect("d_q");
    d_q.copy_from_host(q_bytes).expect("up q");

    // FP32 cache
    let mut d_k_fp32 = HipBuffer::new(SEQ * HEAD_DIM * 4).expect("k fp32");
    let mut d_v_fp32 = HipBuffer::new(SEQ * HEAD_DIM * 4).expect("v fp32");
    d_k_fp32
        .copy_from_host(&vec![0u8; SEQ * HEAD_DIM * 4])
        .expect("zero k");
    d_v_fp32
        .copy_from_host(&vec![0u8; SEQ * HEAD_DIM * 4])
        .expect("zero v");

    // FP8 cache
    let mut d_k_fp8 = HipBuffer::new(SEQ * HEAD_DIM).expect("k fp8");
    let mut d_v_fp8 = HipBuffer::new(SEQ * HEAD_DIM).expect("v fp8");
    d_k_fp8.copy_from_host(&vec![0u8; SEQ * HEAD_DIM]).expect("zero");
    d_v_fp8.copy_from_host(&vec![0u8; SEQ * HEAD_DIM]).expect("zero");

    // Append each token through both paths.
    let row_bytes = HEAD_DIM * 4;
    let mut d_k_row = HipBuffer::new(row_bytes).expect("k row");
    let mut d_v_row = HipBuffer::new(row_bytes).expect("v row");
    for j in 0..SEQ {
        let k_row = unsafe {
            std::slice::from_raw_parts(all_k[j * HEAD_DIM..].as_ptr() as *const u8, row_bytes)
        };
        let v_row = unsafe {
            std::slice::from_raw_parts(all_v[j * HEAD_DIM..].as_ptr() as *const u8, row_bytes)
        };
        d_k_row.copy_from_host(k_row).expect("up k row");
        d_v_row.copy_from_host(v_row).expect("up v row");

        unsafe {
            rocmforge_launch_kv_cache_append(
                d_k_fp32.as_mut_ptr() as *mut f32,
                d_v_fp32.as_mut_ptr() as *mut f32,
                d_k_row.as_ptr() as *const f32,
                d_v_row.as_ptr() as *const f32,
                1, HEAD_DIM as i32, j as i32, HEAD_DIM as i32,
                stream.raw(),
            )
        };
        unsafe {
            rocmforge_launch_kv_cache_append_fp8(
                d_k_fp8.as_mut_ptr(),
                d_v_fp8.as_mut_ptr(),
                d_k_row.as_ptr() as *const f32,
                d_v_row.as_ptr() as *const f32,
                1, HEAD_DIM as i32, j as i32, HEAD_DIM as i32,
                stream.raw(),
            )
        };
    }

    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let mut d_out_fp32 = HipBuffer::new(HEAD_DIM * 4).expect("out fp32");
    let mut d_out_fp8 = HipBuffer::new(HEAD_DIM * 4).expect("out fp8");

    unsafe {
        rocmforge_launch_attention_decode(
            d_q.as_ptr() as *const f32,
            d_k_fp32.as_ptr() as *const f32,
            d_v_fp32.as_ptr() as *const f32,
            d_out_fp32.as_mut_ptr() as *mut f32,
            1, 1, HEAD_DIM as i32, SEQ as i32, HEAD_DIM as i32, scale,
            stream.raw(),
        )
    };
    unsafe {
        rocmforge_launch_attention_decode_fp8(
            d_q.as_ptr() as *const f32,
            d_k_fp8.as_ptr(),
            d_v_fp8.as_ptr(),
            d_out_fp8.as_mut_ptr() as *mut f32,
            1, 1, HEAD_DIM as i32, SEQ as i32, HEAD_DIM as i32, scale,
            stream.raw(),
        )
    };
    stream.synchronize().expect("sync");

    let mut out_fp32 = vec![0f32; HEAD_DIM];
    let mut out_fp8 = vec![0f32; HEAD_DIM];
    unsafe {
        hipMemcpy(
            out_fp32.as_mut_ptr() as *mut _,
            d_out_fp32.as_ptr(),
            HEAD_DIM * 4,
            hipMemcpyDeviceToHost,
        );
        hipMemcpy(
            out_fp8.as_mut_ptr() as *mut _,
            d_out_fp8.as_ptr(),
            HEAD_DIM * 4,
            hipMemcpyDeviceToHost,
        );
    }

    // Cosine similarity — direction matters more than magnitude here.
    let dot: f32 = out_fp32.iter().zip(&out_fp8).map(|(a, b)| a * b).sum();
    let na: f32 = out_fp32.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = out_fp8.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cos = dot / (na * nb);

    let diff_norm: f32 = out_fp32
        .iter()
        .zip(&out_fp8)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    let rel = diff_norm / na.max(1e-6);

    println!("  attention FP32 vs FP8-KV: cosine={cos:.5}, ||diff||/||fp32||={rel:.4}");
    assert!(cos > 0.99, "cosine {cos} below 0.99 — FP8-KV drift too large");
    assert!(rel < 0.20, "relative L2 diff {rel} > 20 %");
}
