// Milestone 0 — FP8 WMMA Smoke Test Harness (gfx1201 / RDNA 4)
//
// Runs five tests against the HIP kernel in `fp8_wmma_smoke.hip` and writes
// `fp8_smoke_results.md` next to the source. See tests/smoke/README in the
// architecture doc (section 9.0) for the Go/No-Go decision matrix.

use std::ffi::{c_char, c_int};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

// --- FFI ---------------------------------------------------------------------

extern "C" {
    fn smoke_get_device_info(
        out_name: *mut c_char,
        out_cap: c_int,
        out_major: *mut c_int,
        out_minor: *mut c_int,
    ) -> c_int;

    fn smoke_run_fp8_wmma(h_a: *const u8, h_b: *const u8, h_d: *mut f32) -> c_int;

    fn smoke_run_fp16_wmma(h_a_bits: *const u16, h_b_bits: *const u16, h_d: *mut f32) -> c_int;

    fn smoke_benchmark_wmma(
        kind: c_int,
        warmup: c_int,
        iters: c_int,
        out_median_us: *mut f32,
        out_min_us: *mut f32,
    ) -> c_int;

    fn smoke_fp32_to_fp8_roundtrip(h_src: *const f32, h_dst: *mut f32, n: c_int) -> c_int;
}

// --- FP8 E4M3 (OCP) host-side encode / decode --------------------------------
//
// E4M3 layout: S.EEEE.MMM, exponent bias 7, max = 448 (0x7E / 0xFE),
// NaN only at 0x7F / 0xFF, no infinities, denormals at exp = 0.

fn e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = (b >> 3) & 0xF;
    let mant = b & 0x7;
    let s = if sign == 0 { 1.0 } else { -1.0 };
    if exp == 0xF && mant == 0x7 {
        return f32::NAN;
    }
    if exp == 0 {
        if mant == 0 {
            return s * 0.0;
        }
        return s * (mant as f32) * 2f32.powi(-9); // 2^(-6) * 2^(-3)
    }
    s * (1.0 + (mant as f32) / 8.0) * 2f32.powi(exp as i32 - 7)
}

// Round-to-nearest-even encoder. Saturates to max-finite on overflow.
fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    // Bit-level RNE via scanning all 256 codes is fine for a smoke reference.
    let mut best_code: u8 = sign;
    let mut best_diff = f32::INFINITY;
    for code in 0..=0x7Fu8 {
        let b = sign | code;
        if (b & 0x7F) == 0x7F {
            continue; // NaN
        }
        let v = e4m3_to_f32(b);
        if v.is_nan() {
            continue;
        }
        let d = (v - x).abs();
        if d < best_diff || (d == best_diff && (b < best_code)) {
            best_diff = d;
            best_code = b;
        }
    }
    best_code
}

// --- FP16 IEEE-754 host-side encode -----------------------------------------

fn f32_to_f16_bits(x: f32) -> u16 {
    // Crude RN encoder (good enough for smoke tests; Rust stdlib has no f16).
    let xu = x.to_bits();
    let sign = ((xu >> 16) & 0x8000) as u16;
    let mag = xu & 0x7FFF_FFFF;
    if mag >= 0x7F80_0000 {
        // inf/NaN
        let nan_bit: u16 = if mag > 0x7F80_0000 { 0x0200 } else { 0 };
        return sign | 0x7C00 | nan_bit;
    }
    if mag < 0x3880_0000 {
        // subnormal or zero in fp16
        if mag < 0x3300_0000 {
            return sign;
        }
        let shift = 0x3880_0000u32.wrapping_sub(mag) >> 23;
        let mantissa = ((mag & 0x007F_FFFF) | 0x0080_0000) >> (shift + 1);
        return sign | mantissa as u16;
    }
    let exp = ((mag >> 23) as i32) - 127 + 15;
    if exp >= 31 {
        return sign | 0x7C00;
    }
    let mant = (mag >> 13) & 0x3FF;
    // RNE
    let round_bits = mag & 0x1FFF;
    let mut out = sign | ((exp as u16) << 10) | mant as u16;
    if round_bits > 0x1000 || (round_bits == 0x1000 && (mant & 1) == 1) {
        out = out.wrapping_add(1);
    }
    out
}

// --- CPU reference matmul ---------------------------------------------------

fn matmul_16x16x16(a: &[f32; 256], b: &[f32; 256], out: &mut [f32; 256]) {
    for i in 0..16 {
        for j in 0..16 {
            let mut acc = 0.0f32;
            for k in 0..16 {
                acc += a[i * 16 + k] * b[k * 16 + j];
            }
            out[i * 16 + j] = acc;
        }
    }
}

fn max_rel_err(ref_vals: &[f32], test_vals: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (r, t) in ref_vals.iter().zip(test_vals.iter()) {
        let abs = (r - t).abs();
        let rel = if r.abs() > 1e-6 { abs / r.abs() } else { abs };
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    (max_abs, max_rel)
}

// --- deterministic LCG so reports reproduce exactly -------------------------

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (self.0 >> 33) as u32;
        ((u as f32) / (u32::MAX as f32 / 2.0)) - 1.0 // [-1, 1)
    }
}

// --- tests ------------------------------------------------------------------

#[derive(Default)]
struct TestResult {
    name: &'static str,
    pass: bool,
    detail: String,
}

fn test1_compilability() -> TestResult {
    // The build already succeeded if we got this far, but double-check the
    // device reports as gfx1201 at runtime.
    let mut name = [0i8; 64];
    let mut major: c_int = 0;
    let mut minor: c_int = 0;
    let rc = unsafe {
        smoke_get_device_info(
            name.as_mut_ptr() as *mut c_char,
            name.len() as c_int,
            &mut major,
            &mut minor,
        )
    };
    if rc != 0 {
        return TestResult {
            name: "Test 1 — Compilability & device detection",
            pass: false,
            detail: format!("hipGetDeviceProperties failed, hipError={rc}"),
        };
    }
    let arch = unsafe {
        std::ffi::CStr::from_ptr(name.as_ptr() as *const c_char)
            .to_string_lossy()
            .into_owned()
    };
    let pass = arch.starts_with("gfx1201");
    TestResult {
        name: "Test 1 — Compilability & device detection",
        pass,
        detail: format!(
            "hipcc built fp8_wmma_smoke.hip for gfx1201; runtime device: {arch} ({major}.{minor})"
        ),
    }
}

fn test2_correctness() -> TestResult {
    let mut rng = Lcg(0xC0FFEE_0000_1234);
    // Build random FP32 matrices in [-2, 2], encode to E4M3, decode back.
    let mut a_f32 = [0.0f32; 256];
    let mut b_f32 = [0.0f32; 256];
    let mut a_bytes = [0u8; 256];
    let mut b_bytes = [0u8; 256];
    let mut a_q = [0.0f32; 256];
    let mut b_q = [0.0f32; 256];

    for i in 0..256 {
        a_f32[i] = rng.next_f32() * 2.0;
        b_f32[i] = rng.next_f32() * 2.0;
        a_bytes[i] = f32_to_e4m3(a_f32[i]);
        b_bytes[i] = f32_to_e4m3(b_f32[i]);
        a_q[i] = e4m3_to_f32(a_bytes[i]);
        b_q[i] = e4m3_to_f32(b_bytes[i]);
    }

    let mut cpu_ref = [0.0f32; 256];
    matmul_16x16x16(&a_q, &b_q, &mut cpu_ref);

    let mut gpu_out = [0.0f32; 256];
    let rc = unsafe {
        smoke_run_fp8_wmma(a_bytes.as_ptr(), b_bytes.as_ptr(), gpu_out.as_mut_ptr())
    };
    if rc != 0 {
        return TestResult {
            name: "Test 2 — FP8 WMMA correctness",
            pass: false,
            detail: format!("smoke_run_fp8_wmma failed, hipError={rc}"),
        };
    }

    // Sanity: also run the trivial all-1.0 case. E4M3 1.0 = 0x38.
    let ones_in = [0x38u8; 256];
    let mut ones_out = [0.0f32; 256];
    let rc2 = unsafe {
        smoke_run_fp8_wmma(ones_in.as_ptr(), ones_in.as_ptr(), ones_out.as_mut_ptr())
    };
    let mut worst_ones = 0.0f32;
    if rc2 == 0 {
        for &v in &ones_out {
            let e = (v - 16.0).abs();
            if e > worst_ones {
                worst_ones = e;
            }
        }
    }

    let (abs_err, rel_err) = max_rel_err(&cpu_ref, &gpu_out);
    // FP8 E4M3 has 3-bit mantissa; 16-accumulation in FP32 means relative
    // error ≲ 0.1. Any higher suggests a layout bug (wrong K-ordering in A/B).
    let pass = rc2 == 0 && worst_ones < 1e-4 && abs_err.is_finite() && rel_err < 0.1;
    TestResult {
        name: "Test 2 — FP8 WMMA correctness",
        pass,
        detail: format!(
            "all-1.0 sanity: max |out-16|={worst_ones:.2e} (expect 0) · random: max_abs_err={abs_err:.3e}, max_rel_err={rel_err:.3e} (tol 0.1)"
        ),
    }
}

fn test3_performance() -> TestResult {
    let mut fp8_med = 0.0f32;
    let mut fp8_min = 0.0f32;
    let mut fp16_med = 0.0f32;
    let mut fp16_min = 0.0f32;
    let rc1 = unsafe { smoke_benchmark_wmma(0, 32, 200, &mut fp8_med, &mut fp8_min) };
    let rc2 = unsafe { smoke_benchmark_wmma(1, 32, 200, &mut fp16_med, &mut fp16_min) };
    if rc1 != 0 || rc2 != 0 {
        return TestResult {
            name: "Test 3 — FP8 vs FP16 performance",
            pass: false,
            detail: format!("benchmark failed, fp8 rc={rc1}, fp16 rc={rc2}"),
        };
    }
    let ratio = if fp8_med > 0.0 { fp16_med / fp8_med } else { 0.0 };
    // A single-tile 16×16×16 matmul is dispatch-bound, not compute-bound, so
    // the FP8/FP16 speedup does NOT reflect the ~2× WMMA throughput difference.
    // A pass here is: FP8 is at least not meaningfully slower (ratio > 0.8).
    let pass = ratio > 0.8;
    TestResult {
        name: "Test 3 — FP8 vs FP16 performance (dispatch-bound single-tile)",
        pass,
        detail: format!(
            "FP8 median={fp8_med:.2} µs (min {fp8_min:.2}), FP16 median={fp16_med:.2} µs (min {fp16_min:.2}), ratio FP16/FP8={ratio:.2}× (single-tile is launch-bound; real speedup verified in Phase 1 on larger shapes)"
        ),
    }
}

fn test4_roundtrip() -> TestResult {
    let n = 1024;
    let mut src = vec![0.0f32; n];
    let mut rng = Lcg(0xD00D_BEEF_0000_4242);
    for i in 0..n {
        // Range [-4, 4] — inside E4M3's representable range.
        src[i] = rng.next_f32() * 4.0;
    }
    let mut dst = vec![0.0f32; n];
    let rc = unsafe { smoke_fp32_to_fp8_roundtrip(src.as_ptr(), dst.as_mut_ptr(), n as c_int) };
    if rc != 0 {
        return TestResult {
            name: "Test 4 — FP32 → E4M3 → FP32 roundtrip",
            pass: false,
            detail: format!("smoke_fp32_to_fp8_roundtrip failed, hipError={rc}"),
        };
    }
    // E4M3 smallest positive denormal ≈ 2^-9 ≈ 1.95e-3; anything below that
    // legitimately quantises to 0, so split the check into two criteria:
    //   · abs error below denormal step, OR
    //   · rel error within one 3-bit mantissa step (≤ 0.125).
    let denorm_step = 2.0f32.powi(-9);
    let mut max_abs = 0.0f32;
    let mut max_rel_large = 0.0f32;
    let mut violations = 0usize;
    for (r, t) in src.iter().zip(dst.iter()) {
        let abs = (r - t).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        if r.abs() > denorm_step {
            let rel = abs / r.abs();
            if rel > max_rel_large {
                max_rel_large = rel;
            }
            if rel > 0.15 {
                violations += 1;
            }
        } else if abs > denorm_step {
            violations += 1;
        }
    }
    let pass = violations == 0 && max_abs.is_finite();
    TestResult {
        name: "Test 4 — FP32 → E4M3 → FP32 roundtrip",
        pass,
        detail: format!(
            "n={n}, max_abs_err={max_abs:.3e}, max_rel_err (|r|>{denorm_step:.2e})={max_rel_large:.3e} (tol 0.15), violations={violations}"
        ),
    }
}

fn test5_fp8_vs_fp16_quality() -> TestResult {
    let mut rng = Lcg(0x5EED_5EED_5EED_5EED);
    let mut a_f32 = [0.0f32; 256];
    let mut b_f32 = [0.0f32; 256];
    let mut a_bytes = [0u8; 256];
    let mut b_bytes = [0u8; 256];
    let mut a_f16 = [0u16; 256];
    let mut b_f16 = [0u16; 256];
    let mut a_q = [0.0f32; 256];
    let mut b_q = [0.0f32; 256];
    for i in 0..256 {
        a_f32[i] = rng.next_f32();
        b_f32[i] = rng.next_f32();
        a_bytes[i] = f32_to_e4m3(a_f32[i]);
        b_bytes[i] = f32_to_e4m3(b_f32[i]);
        a_q[i] = e4m3_to_f32(a_bytes[i]);
        b_q[i] = e4m3_to_f32(b_bytes[i]);
        a_f16[i] = f32_to_f16_bits(a_f32[i]);
        b_f16[i] = f32_to_f16_bits(b_f32[i]);
    }
    let mut cpu_ref_fp32 = [0.0f32; 256];
    matmul_16x16x16(&a_f32, &b_f32, &mut cpu_ref_fp32);
    let mut cpu_ref_e4m3 = [0.0f32; 256];
    matmul_16x16x16(&a_q, &b_q, &mut cpu_ref_e4m3);

    let mut gpu_fp8 = [0.0f32; 256];
    let mut gpu_fp16 = [0.0f32; 256];
    let rc1 = unsafe { smoke_run_fp8_wmma(a_bytes.as_ptr(), b_bytes.as_ptr(), gpu_fp8.as_mut_ptr()) };
    let rc2 = unsafe { smoke_run_fp16_wmma(a_f16.as_ptr(), b_f16.as_ptr(), gpu_fp16.as_mut_ptr()) };
    if rc1 != 0 || rc2 != 0 {
        return TestResult {
            name: "Test 5 — FP8 vs FP16 numerical quality",
            pass: false,
            detail: format!("kernel failed, fp8 rc={rc1}, fp16 rc={rc2}"),
        };
    }
    let (_, rel_fp8_vs_fp32) = max_rel_err(&cpu_ref_fp32, &gpu_fp8);
    let (_, rel_fp16_vs_fp32) = max_rel_err(&cpu_ref_fp32, &gpu_fp16);
    let (_, rel_fp8_vs_quantref) = max_rel_err(&cpu_ref_e4m3, &gpu_fp8);
    // Consistency check: GPU FP8 must closely track the quantised CPU reference
    // (the unquantised gap is an intrinsic FP8 property, not a correctness issue).
    let pass = rel_fp8_vs_quantref < 0.1 && rel_fp16_vs_fp32 < 0.01;
    TestResult {
        name: "Test 5 — FP8 vs FP16 numerical quality",
        pass,
        detail: format!(
            "GPU-FP8 vs CPU-E4M3 ref: {rel_fp8_vs_quantref:.3e} (tol 0.1) · GPU-FP8 vs unquantised FP32: {rel_fp8_vs_fp32:.3e} (intrinsic FP8 gap) · GPU-FP16 vs FP32: {rel_fp16_vs_fp32:.3e} (tol 0.01)"
        ),
    }
}

// --- report generation ------------------------------------------------------

fn render_report(results: &[TestResult]) -> (String, bool) {
    let all_pass = results.iter().all(|r| r.pass);
    let decision = if all_pass {
        "**GO — FP8 as v1.0 default WMMA input path.**"
    } else if results.first().map_or(false, |r| !r.pass) {
        "**NO-GO FP8, FALLBACK to FP16.** Compiler cannot target the intrinsic; FP16 architecture applies."
    } else {
        "**PARTIAL — review individual test results.** FP8 is available but some criteria failed. Consult detail below before committing to the default path."
    };

    let mut md = String::new();
    md.push_str("# Milestone 0 — FP8 WMMA Smoke Test Results\n\n");
    md.push_str("Target: **gfx1201** (AMD Radeon RX 9070 XT, RDNA 4) · ROCm 7.2.2 · Wave32\n\n");
    md.push_str("Intrinsic under test: `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`\n\n");
    md.push_str("| # | Test | Result |\n|---|------|--------|\n");
    for (i, r) in results.iter().enumerate() {
        md.push_str(&format!(
            "| {} | {} | {} |\n",
            i + 1,
            r.name,
            if r.pass { "✅ PASS" } else { "❌ FAIL" }
        ));
    }
    md.push_str("\n## Details\n\n");
    for r in results {
        md.push_str(&format!(
            "### {}\n\n- Result: **{}**\n- {}\n\n",
            r.name,
            if r.pass { "PASS" } else { "FAIL" },
            r.detail
        ));
    }
    md.push_str("## Decision\n\n");
    md.push_str(decision);
    md.push('\n');
    (md, all_pass)
}

fn main() -> ExitCode {
    println!("ROCmForge v1.0 — Milestone 0: FP8 WMMA Smoke Test");
    println!("Target: gfx1201 (RX 9070 XT), ROCm 7.2.2\n");

    let results = [
        test1_compilability(),
        test2_correctness(),
        test3_performance(),
        test4_roundtrip(),
        test5_fp8_vs_fp16_quality(),
    ];

    for r in &results {
        let mark = if r.pass { "PASS" } else { "FAIL" };
        println!("[{mark}] {}", r.name);
        println!("       {}", r.detail);
    }

    let (md, all_pass) = render_report(&results);
    let out_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fp8_smoke_results.md");
    if let Err(e) = fs::write(&out_path, &md) {
        eprintln!("warn: could not write report to {}: {e}", out_path.display());
    } else {
        println!("\nReport written to {}", out_path.display());
    }

    println!(
        "\nDecision: {}",
        if all_pass {
            "GO — FP8 as default WMMA input path."
        } else if !results[0].pass {
            "NO-GO FP8 — FALLBACK to FP16 architecture."
        } else {
            "PARTIAL — manual review required."
        }
    );

    if all_pass {
        ExitCode::SUCCESS
    } else if !results[0].pass {
        ExitCode::from(10) // hard gate failed
    } else {
        ExitCode::from(20) // soft failure — review
    }
}
