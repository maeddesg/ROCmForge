//! Phase 1 / Schritt 1.9 Block A — elementwise kernels.
//!
//! Embedding lookup, RMSNorm, residual add. Verified against CPU
//! references and, where available, against v0.x launchers. All GPU
//! tests run `#[serial]`.

#![cfg(all(feature = "v1", feature = "gpu"))]

use rocmforge::v1::backend::gpu::elementwise::{
    rocmforge_launch_embedding_lookup, rocmforge_launch_residual_add,
    rocmforge_launch_residual_add_inplace, rocmforge_launch_rms_norm,
    rocmforge_launch_rms_norm_batched,
};
use rocmforge::v1::backend::gpu::error::{check, HipResult};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
use serial_test::serial;

// ── Helpers ────────────────────────────────────────────────────────────────

fn upload_f32(values: &[f32]) -> HipResult<HipBuffer> {
    let mut buf = HipBuffer::new(values.len() * 4)?;
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
    buf.copy_from_host(bytes)?;
    Ok(buf)
}

fn download_f32(buf: &HipBuffer, len: usize) -> HipResult<Vec<f32>> {
    let mut host_bytes = vec![0u8; len * 4];
    let rc = unsafe {
        hipMemcpy(
            host_bytes.as_mut_ptr() as *mut _,
            buf.as_ptr(),
            len * 4,
            hipMemcpyDeviceToHost,
        )
    };
    check(rc, "D2H readback")?;
    Ok(host_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ── Embedding lookup ───────────────────────────────────────────────────────

#[test]
#[serial]
fn test_embedding_lookup_bit_exact() {
    let vocab_size = 1000usize;
    let hidden_dim = 128usize;
    let seq_len = 8usize;
    let mut rng = fastrand::Rng::with_seed(0xE1);

    let table: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|_| rng.f32() * 2.0 - 1.0)
        .collect();
    let token_ids: Vec<u32> = (0..seq_len).map(|_| rng.u32(..(vocab_size as u32))).collect();

    let mut d_tok = HipBuffer::new(token_ids.len() * 4).unwrap();
    let tok_bytes = unsafe {
        std::slice::from_raw_parts(token_ids.as_ptr() as *const u8, token_ids.len() * 4)
    };
    d_tok.copy_from_host(tok_bytes).unwrap();
    let d_tbl = upload_f32(&table).unwrap();
    let mut d_out = HipBuffer::new(seq_len * hidden_dim * 4).unwrap();

    let stream = HipStream::new().unwrap();
    let rc = unsafe {
        rocmforge_launch_embedding_lookup(
            d_tok.as_ptr() as *const u32,
            d_tbl.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut f32,
            seq_len as i32,
            hidden_dim as i32,
            stream.raw(),
        )
    };
    check(rc, "embedding_lookup").unwrap();
    stream.synchronize().unwrap();

    let gpu = download_f32(&d_out, seq_len * hidden_dim).unwrap();

    // CPU reference: pure memcpy — bit-exact.
    for (pos, &tid) in token_ids.iter().enumerate() {
        let src = &table[(tid as usize) * hidden_dim..(tid as usize + 1) * hidden_dim];
        let dst = &gpu[pos * hidden_dim..(pos + 1) * hidden_dim];
        for (i, (&s, &g)) in src.iter().zip(dst.iter()).enumerate() {
            assert_eq!(
                s.to_bits(),
                g.to_bits(),
                "pos={pos} tid={tid} dim={i}: src={s} gpu={g}"
            );
        }
    }
    println!("Embedding lookup: bit-exact for {seq_len} tokens × {hidden_dim} dims");
}

#[test]
#[serial]
fn test_embedding_boundary_tokens() {
    // Tokens 0 and vocab_size-1 — make sure boundary indexing works.
    let vocab_size = 100usize;
    let hidden_dim = 64usize;
    let mut rng = fastrand::Rng::with_seed(0xE2);
    let table: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|_| rng.f32() * 2.0 - 1.0)
        .collect();
    let token_ids: Vec<u32> = vec![0, (vocab_size as u32) - 1];

    let mut d_tok = HipBuffer::new(token_ids.len() * 4).unwrap();
    let tok_bytes = unsafe {
        std::slice::from_raw_parts(token_ids.as_ptr() as *const u8, token_ids.len() * 4)
    };
    d_tok.copy_from_host(tok_bytes).unwrap();
    let d_tbl = upload_f32(&table).unwrap();
    let mut d_out = HipBuffer::new(2 * hidden_dim * 4).unwrap();

    let stream = HipStream::new().unwrap();
    let rc = unsafe {
        rocmforge_launch_embedding_lookup(
            d_tok.as_ptr() as *const u32,
            d_tbl.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut f32,
            2,
            hidden_dim as i32,
            stream.raw(),
        )
    };
    check(rc, "embedding_lookup boundary").unwrap();
    stream.synchronize().unwrap();

    let gpu = download_f32(&d_out, 2 * hidden_dim).unwrap();

    // pos 0 → table[0..hidden_dim]; pos 1 → table[(vocab-1)*hidden..]
    for i in 0..hidden_dim {
        assert_eq!(gpu[i].to_bits(), table[i].to_bits());
        assert_eq!(
            gpu[hidden_dim + i].to_bits(),
            table[(vocab_size - 1) * hidden_dim + i].to_bits()
        );
    }
    println!("Embedding boundary tokens: first and last vocab entries correct");
}

// ── RMSNorm ────────────────────────────────────────────────────────────────

fn cpu_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = weight.len();
    let mut output = vec![0.0f32; input.len()];
    for row in 0..(input.len() / dim) {
        let slice = &input[row * dim..(row + 1) * dim];
        let sum_sq: f32 = slice.iter().map(|x| x * x).sum();
        let inv_rms = (sum_sq / dim as f32 + eps).sqrt().recip();
        for i in 0..dim {
            output[row * dim + i] = slice[i] * inv_rms * weight[i];
        }
    }
    output
}

#[test]
#[serial]
fn test_rms_norm_single_token() {
    let n = 4096usize;
    let eps = 1e-5f32;
    let mut rng = fastrand::Rng::with_seed(0x70);
    let x: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let w: Vec<f32> = (0..n).map(|_| rng.f32() * 0.5 + 0.5).collect();

    let cpu = cpu_rms_norm(&x, &w, eps);

    let d_x = upload_f32(&x).unwrap();
    let d_w = upload_f32(&w).unwrap();
    let mut d_out = HipBuffer::new(n * 4).unwrap();
    let stream = HipStream::new().unwrap();
    let rc = unsafe {
        rocmforge_launch_rms_norm(
            d_x.as_ptr() as *const f32,
            d_w.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut f32,
            n as i32,
            eps,
            stream.raw(),
        )
    };
    check(rc, "rms_norm").unwrap();
    stream.synchronize().unwrap();

    let gpu = download_f32(&d_out, n).unwrap();
    let err = max_abs_err(&cpu, &gpu);
    let max_mag = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let tol = max_mag * 1e-5 + 1e-6;
    assert!(err < tol, "RMSNorm single-token err={err} tol={tol}");
    println!("RMSNorm single-token n={n}: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_rms_norm_batched() {
    let n = 4096usize;
    let seq_len = 64usize;
    let eps = 1e-5f32;
    let mut rng = fastrand::Rng::with_seed(0x71);
    let x: Vec<f32> = (0..seq_len * n).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let w: Vec<f32> = (0..n).map(|_| rng.f32() * 0.5 + 0.5).collect();

    let cpu = cpu_rms_norm(&x, &w, eps);

    let d_x = upload_f32(&x).unwrap();
    let d_w = upload_f32(&w).unwrap();
    let mut d_out = HipBuffer::new(seq_len * n * 4).unwrap();
    let stream = HipStream::new().unwrap();
    let rc = unsafe {
        rocmforge_launch_rms_norm_batched(
            d_x.as_ptr() as *const f32,
            d_w.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut f32,
            n as i32,
            eps,
            seq_len as i32,
            stream.raw(),
        )
    };
    check(rc, "rms_norm_batched").unwrap();
    stream.synchronize().unwrap();

    let gpu = download_f32(&d_out, seq_len * n).unwrap();
    let err = max_abs_err(&cpu, &gpu);
    let max_mag = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let tol = max_mag * 1e-5 + 1e-6;
    assert!(err < tol, "RMSNorm batched err={err} tol={tol}");
    println!(
        "RMSNorm batched seq={seq_len} n={n}: max_abs_err = {err:.4e} (tol {tol:.4e})"
    );
}

#[test]
#[serial]
fn test_rms_norm_eps_sensitivity() {
    // All inputs near zero — without eps this would divide by zero.
    let n = 128usize;
    let x = vec![1e-7f32; n];
    let w = vec![1.0f32; n];
    let eps = 1e-5f32;

    let d_x = upload_f32(&x).unwrap();
    let d_w = upload_f32(&w).unwrap();
    let mut d_out = HipBuffer::new(n * 4).unwrap();
    let stream = HipStream::new().unwrap();
    let rc = unsafe {
        rocmforge_launch_rms_norm(
            d_x.as_ptr() as *const f32,
            d_w.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut f32,
            n as i32,
            eps,
            stream.raw(),
        )
    };
    check(rc, "rms_norm eps").unwrap();
    stream.synchronize().unwrap();
    let gpu = download_f32(&d_out, n).unwrap();

    for &v in &gpu {
        assert!(v.is_finite(), "rms_norm produced non-finite value {v}");
    }
    println!("RMSNorm eps sensitivity: all {n} outputs finite with x ≈ 1e-7, eps = 1e-5");
}

// v0.x comparison — call the v0.x kernel via its extern symbol.
extern "C" {
    fn gpu_rms_norm(
        d_x: *const f32,
        d_weight: *const f32,
        d_out: *mut f32,
        n: i32,
        eps: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

#[test]
#[serial]
fn test_rms_norm_vs_v0x() {
    let n = 4096usize;
    let eps = 1e-5f32;
    let mut rng = fastrand::Rng::with_seed(0x72);
    let x: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let w: Vec<f32> = (0..n).map(|_| rng.f32() * 0.5 + 0.5).collect();

    let d_x = upload_f32(&x).unwrap();
    let d_w = upload_f32(&w).unwrap();
    let mut d_out_v1 = HipBuffer::new(n * 4).unwrap();
    let mut d_out_v0 = HipBuffer::new(n * 4).unwrap();
    let stream = HipStream::new().unwrap();

    // v1
    unsafe {
        rocmforge_launch_rms_norm(
            d_x.as_ptr() as *const f32,
            d_w.as_ptr() as *const f32,
            d_out_v1.as_mut_ptr() as *mut f32,
            n as i32,
            eps,
            stream.raw(),
        );
    }
    // v0.x
    unsafe {
        gpu_rms_norm(
            d_x.as_ptr() as *const f32,
            d_w.as_ptr() as *const f32,
            d_out_v0.as_mut_ptr() as *mut f32,
            n as i32,
            eps,
            stream.raw() as *mut std::ffi::c_void,
        );
    }
    stream.synchronize().unwrap();

    let v1 = download_f32(&d_out_v1, n).unwrap();
    let v0 = download_f32(&d_out_v0, n).unwrap();
    let err = max_abs_err(&v1, &v0);
    assert!(err < 1e-6, "RMSNorm v1 vs v0.x err={err} (tol 1e-6)");
    println!("RMSNorm v1 vs v0.x: max_abs_err = {err:.4e}");
}

// ── Residual add ───────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_residual_add() {
    let n = 4096usize;
    let mut rng = fastrand::Rng::with_seed(0x50);
    let a: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();

    let d_a = upload_f32(&a).unwrap();
    let d_b = upload_f32(&b).unwrap();
    let mut d_out = HipBuffer::new(n * 4).unwrap();
    let stream = HipStream::new().unwrap();
    let rc = unsafe {
        rocmforge_launch_residual_add(
            d_out.as_mut_ptr() as *mut f32,
            d_a.as_ptr() as *const f32,
            d_b.as_ptr() as *const f32,
            n as i32,
            stream.raw(),
        )
    };
    check(rc, "residual_add").unwrap();
    stream.synchronize().unwrap();
    let gpu = download_f32(&d_out, n).unwrap();

    for i in 0..n {
        let expected = a[i] + b[i];
        assert_eq!(
            gpu[i].to_bits(),
            expected.to_bits(),
            "idx {i}: gpu={} expected={}",
            gpu[i],
            expected
        );
    }
    println!("Residual add: bit-exact over {n} elements");
}

#[test]
#[serial]
fn test_residual_add_inplace() {
    let n = 2048usize;
    let mut rng = fastrand::Rng::with_seed(0x51);
    let a_init: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();

    let mut d_a = upload_f32(&a_init).unwrap();
    let d_b = upload_f32(&b).unwrap();
    let stream = HipStream::new().unwrap();
    let rc = unsafe {
        rocmforge_launch_residual_add_inplace(
            d_a.as_mut_ptr() as *mut f32,
            d_b.as_ptr() as *const f32,
            n as i32,
            stream.raw(),
        )
    };
    check(rc, "residual_add_inplace").unwrap();
    stream.synchronize().unwrap();
    let gpu = download_f32(&d_a, n).unwrap();

    for i in 0..n {
        let expected = a_init[i] + b[i];
        assert_eq!(gpu[i].to_bits(), expected.to_bits());
    }
    println!("Residual add in-place: bit-exact over {n} elements");
}
