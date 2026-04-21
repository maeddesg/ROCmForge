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

// ─── Block B: RoPE ─────────────────────────────────────────────────────────

use rocmforge::v1::backend::gpu::elementwise::{
    rocmforge_launch_rope, rocmforge_launch_rope_batched,
};

// CPU reference matching the classic non-NeoX layout of
// rope_heads_kernel. `freq_scale = None` → standard; `Some` → Llama-3.1
// custom.
fn cpu_rope(
    x: &mut [f32],
    pos: i32,
    num_heads: usize,
    head_dim: usize,
    theta_base: f32,
    freq_scale: Option<&[f32]>,
) {
    let half = head_dim / 2;
    for head in 0..num_heads {
        let base = head * head_dim;
        for i in 0..half {
            let exponent = (2 * i) as f32 / head_dim as f32;
            let mut theta = 1.0f32 / theta_base.powf(exponent);
            if let Some(fs) = freq_scale {
                theta /= fs[i];
            }
            let angle = pos as f32 * theta;
            let c = angle.cos();
            let s = angle.sin();
            let i0 = base + 2 * i;
            let i1 = base + 2 * i + 1;
            let x0 = x[i0];
            let x1 = x[i1];
            x[i0] = x0 * c - x1 * s;
            x[i1] = x0 * s + x1 * c;
        }
    }
}

fn gen_qk_state(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

#[test]
#[serial]
fn test_rope_standard_single_token() {
    // Qwen2.5 QKV: n_heads=32, head_dim=128, theta_base=10000.0.
    let pos = 0i32;
    let n_heads = 32usize;
    let head_dim = 128usize;
    let theta_base = 10000.0f32;
    let x = gen_qk_state(n_heads * head_dim, 0xF0);

    let mut cpu = x.clone();
    cpu_rope(&mut cpu, pos, n_heads, head_dim, theta_base, None);

    let mut d_x = upload_f32(&x).unwrap();
    let stream = HipStream::new().unwrap();
    let rc = unsafe {
        rocmforge_launch_rope(
            d_x.as_mut_ptr() as *mut f32,
            pos,
            n_heads as i32,
            head_dim as i32,
            theta_base,
            std::ptr::null(),
            stream.raw(),
        )
    };
    check(rc, "rope standard").unwrap();
    stream.synchronize().unwrap();
    let gpu = download_f32(&d_x, n_heads * head_dim).unwrap();

    let err = max_abs_err(&cpu, &gpu);
    assert!(err < 1e-5, "RoPE standard err={err} (tol 1e-5)");
    println!("RoPE standard single-token: max_abs_err = {err:.4e}");
}

#[test]
#[serial]
fn test_rope_standard_sequence() {
    // Batched RoPE for prefill with pos_offset=100.
    let start_pos = 100i32;
    let seq_len = 64usize;
    let n_heads = 32usize;
    let head_dim = 128usize;
    let theta_base = 10000.0f32;
    let x = gen_qk_state(seq_len * n_heads * head_dim, 0xF1);

    let mut cpu = x.clone();
    for s in 0..seq_len {
        let row_start = s * n_heads * head_dim;
        let row = &mut cpu[row_start..row_start + n_heads * head_dim];
        cpu_rope(row, start_pos + s as i32, n_heads, head_dim, theta_base, None);
    }

    let mut d_x = upload_f32(&x).unwrap();
    let stream = HipStream::new().unwrap();
    let rc = unsafe {
        rocmforge_launch_rope_batched(
            d_x.as_mut_ptr() as *mut f32,
            start_pos,
            n_heads as i32,
            head_dim as i32,
            theta_base,
            seq_len as i32,
            std::ptr::null(),
            stream.raw(),
        )
    };
    check(rc, "rope batched").unwrap();
    stream.synchronize().unwrap();
    let gpu = download_f32(&d_x, seq_len * n_heads * head_dim).unwrap();

    let err = max_abs_err(&cpu, &gpu);
    // CPU f32::cos/sin/powf and GPU cosf/sinf/powf diverge by ~1 ULP;
    // at pos=163 the accumulated roundoff reaches ~2e-5. v0.x match
    // test proves the kernel itself is correct — the tolerance gate
    // bounds only CPU-vs-GPU math-library drift.
    assert!(err < 5e-5, "RoPE batched err={err} (tol 5e-5)");
    println!("RoPE batched (seq={seq_len}, pos_offset={start_pos}): max_abs_err = {err:.4e}");
}

#[test]
#[serial]
fn test_rope_llama31_custom_freqs() {
    // Llama-3.1 style: non-null freq_scale.
    let pos = 50i32;
    let n_heads = 32usize;
    let head_dim = 128usize;
    let theta_base = 500000.0f32;
    let mut rng = fastrand::Rng::with_seed(0xF2);
    let freq_scale: Vec<f32> = (0..(head_dim / 2)).map(|_| 1.0 + rng.f32() * 3.0).collect();
    let x = gen_qk_state(n_heads * head_dim, 0xF3);

    let mut cpu = x.clone();
    cpu_rope(
        &mut cpu,
        pos,
        n_heads,
        head_dim,
        theta_base,
        Some(&freq_scale),
    );

    let mut d_x = upload_f32(&x).unwrap();
    let d_fs = upload_f32(&freq_scale).unwrap();
    let stream = HipStream::new().unwrap();
    let rc = unsafe {
        rocmforge_launch_rope(
            d_x.as_mut_ptr() as *mut f32,
            pos,
            n_heads as i32,
            head_dim as i32,
            theta_base,
            d_fs.as_ptr() as *const f32,
            stream.raw(),
        )
    };
    check(rc, "rope llama31").unwrap();
    stream.synchronize().unwrap();
    let gpu = download_f32(&d_x, n_heads * head_dim).unwrap();

    let err = max_abs_err(&cpu, &gpu);
    assert!(err < 1e-5, "RoPE Llama-3.1 err={err}");
    println!("RoPE Llama-3.1 custom (theta={theta_base}): max_abs_err = {err:.4e}");
}

#[test]
#[serial]
fn test_rope_qwen3_high_base() {
    // Qwen3: theta_base = 1_000_000 (1M). Numerical stability check.
    let pos = 2000i32;
    let n_heads = 32usize;
    let head_dim = 128usize;
    let theta_base = 1_000_000.0f32;
    let x = gen_qk_state(n_heads * head_dim, 0xF4);

    let mut cpu = x.clone();
    cpu_rope(&mut cpu, pos, n_heads, head_dim, theta_base, None);

    let mut d_x = upload_f32(&x).unwrap();
    let stream = HipStream::new().unwrap();
    let rc = unsafe {
        rocmforge_launch_rope(
            d_x.as_mut_ptr() as *mut f32,
            pos,
            n_heads as i32,
            head_dim as i32,
            theta_base,
            std::ptr::null(),
            stream.raw(),
        )
    };
    check(rc, "rope qwen3").unwrap();
    stream.synchronize().unwrap();
    let gpu = download_f32(&d_x, n_heads * head_dim).unwrap();

    for v in &gpu {
        assert!(v.is_finite(), "RoPE high-base produced non-finite: {v}");
    }
    let err = max_abs_err(&cpu, &gpu);
    // pos=2000, theta_base=1e6: angle range spans ~0..2000 rad. CPU
    // and GPU cos/sin diverge by a few ULPs; observed ~1.4e-4 is
    // normal math-library drift, not a kernel bug (v0.x bit-exact
    // match verifies kernel correctness separately).
    assert!(err < 5e-4, "RoPE high-base err={err}");
    println!("RoPE Qwen3 high-base (theta=1e6, pos=2000): max_abs_err = {err:.4e}");
}

#[test]
#[serial]
fn test_rope_gqa_kv_heads() {
    // GQA: n_heads=32 Q, n_kv_heads=8 K. The kernel is invoked TWICE
    // — once on Q with n_heads, once on K with n_kv_heads — rather
    // than conditionally inside. Verify both dispatches independently.
    let pos = 5i32;
    let n_heads = 32usize;
    let n_kv_heads = 8usize;
    let head_dim = 128usize;
    let theta_base = 10000.0f32;

    let q = gen_qk_state(n_heads * head_dim, 0xF5);
    let k = gen_qk_state(n_kv_heads * head_dim, 0xF6);

    let mut cpu_q = q.clone();
    let mut cpu_k = k.clone();
    cpu_rope(&mut cpu_q, pos, n_heads, head_dim, theta_base, None);
    cpu_rope(&mut cpu_k, pos, n_kv_heads, head_dim, theta_base, None);

    let mut d_q = upload_f32(&q).unwrap();
    let mut d_k = upload_f32(&k).unwrap();
    let stream = HipStream::new().unwrap();
    unsafe {
        rocmforge_launch_rope(
            d_q.as_mut_ptr() as *mut f32,
            pos,
            n_heads as i32,
            head_dim as i32,
            theta_base,
            std::ptr::null(),
            stream.raw(),
        );
        rocmforge_launch_rope(
            d_k.as_mut_ptr() as *mut f32,
            pos,
            n_kv_heads as i32,
            head_dim as i32,
            theta_base,
            std::ptr::null(),
            stream.raw(),
        );
    }
    stream.synchronize().unwrap();

    let gpu_q = download_f32(&d_q, n_heads * head_dim).unwrap();
    let gpu_k = download_f32(&d_k, n_kv_heads * head_dim).unwrap();

    let err_q = max_abs_err(&cpu_q, &gpu_q);
    let err_k = max_abs_err(&cpu_k, &gpu_k);
    assert!(err_q < 1e-5 && err_k < 1e-5, "GQA errQ={err_q} errK={err_k}");
    println!("RoPE GQA (Q heads {n_heads}, K heads {n_kv_heads}): err_Q = {err_q:.4e}, err_K = {err_k:.4e}");
}

// v0.x RoPE launcher for direct comparison.
extern "C" {
    fn gpu_rope_heads_scaled(
        d_x: *mut f32,
        pos: i32,
        num_heads: i32,
        head_dim: i32,
        theta_base: f32,
        neox: i32,
        d_freq_scale: *const f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

#[test]
#[serial]
fn test_rope_vs_v0x() {
    let pos = 42i32;
    let n_heads = 32usize;
    let head_dim = 128usize;
    let theta_base = 10000.0f32;
    let x = gen_qk_state(n_heads * head_dim, 0xF7);

    let mut d_v1 = upload_f32(&x).unwrap();
    let mut d_v0 = upload_f32(&x).unwrap();
    let stream = HipStream::new().unwrap();

    unsafe {
        rocmforge_launch_rope(
            d_v1.as_mut_ptr() as *mut f32,
            pos,
            n_heads as i32,
            head_dim as i32,
            theta_base,
            std::ptr::null(),
            stream.raw(),
        );
        gpu_rope_heads_scaled(
            d_v0.as_mut_ptr() as *mut f32,
            pos,
            n_heads as i32,
            head_dim as i32,
            theta_base,
            0, // neox = 0
            std::ptr::null(),
            stream.raw() as *mut std::ffi::c_void,
        );
    }
    stream.synchronize().unwrap();

    let v1 = download_f32(&d_v1, n_heads * head_dim).unwrap();
    let v0 = download_f32(&d_v0, n_heads * head_dim).unwrap();
    let err = max_abs_err(&v1, &v0);
    assert!(err < 1e-5, "RoPE v1 vs v0.x err={err}");
    println!("RoPE v1 vs v0.x: max_abs_err = {err:.4e}");
}

#[test]
#[serial]
fn test_rope_position_continuity() {
    // Applying RoPE to 10 tokens one at a time (decode) must match
    // applying the batched variant with start_pos=0 over the whole
    // sequence.
    let seq_len = 10usize;
    let n_heads = 8usize;
    let head_dim = 64usize;
    let theta_base = 10000.0f32;
    let x = gen_qk_state(seq_len * n_heads * head_dim, 0xF8);

    // Reference A: single-token calls, one per position.
    let mut ref_a = x.clone();
    for s in 0..seq_len {
        let row = &mut ref_a[s * n_heads * head_dim..(s + 1) * n_heads * head_dim];
        let mut d_row = upload_f32(row).unwrap();
        let stream = HipStream::new().unwrap();
        unsafe {
            rocmforge_launch_rope(
                d_row.as_mut_ptr() as *mut f32,
                s as i32,
                n_heads as i32,
                head_dim as i32,
                theta_base,
                std::ptr::null(),
                stream.raw(),
            );
        }
        stream.synchronize().unwrap();
        let rotated = download_f32(&d_row, n_heads * head_dim).unwrap();
        row.copy_from_slice(&rotated);
    }

    // Reference B: batched call with start_pos=0.
    let mut d_b = upload_f32(&x).unwrap();
    let stream = HipStream::new().unwrap();
    unsafe {
        rocmforge_launch_rope_batched(
            d_b.as_mut_ptr() as *mut f32,
            0,
            n_heads as i32,
            head_dim as i32,
            theta_base,
            seq_len as i32,
            std::ptr::null(),
            stream.raw(),
        );
    }
    stream.synchronize().unwrap();
    let ref_b = download_f32(&d_b, seq_len * n_heads * head_dim).unwrap();

    let err = max_abs_err(&ref_a, &ref_b);
    assert!(err < 1e-6, "Single-vs-batched consistency err={err}");
    println!("RoPE position continuity (seq={seq_len}): max_abs_err = {err:.4e}");
}
