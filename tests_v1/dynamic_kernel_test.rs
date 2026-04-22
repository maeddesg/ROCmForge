//! Phase 2 / Schritt 2.1.3 Block B — dynamic compile + load + launch.
//!
//! End-to-end proof of the GA infrastructure pipeline:
//!   Rust emits a HIP source with a parametric symbol + num_waves
//!   → hipcc compiles to a gfx1201 code object
//!   → HipModule::load binds the code object into the HIP context
//!   → HipModule::get_function resolves the unmangled symbol
//!   → DynamicKernel::launch_gate_up_swiglu dispatches it
//!   → output matches the statically-linked Phase-1 kernel bit-for-bit
//!
//! Test matrix:
//!   * CPU-only: hipcc-path detection, parametric source contains the
//!     expected symbol per num_waves, compile error surfaces stderr
//!   * GPU-gated: trivial compile+launch smoke, dynamic==static parity
//!     at num_waves=8, different num_waves produce different timings,
//!     VGPR readable from the dynamically-produced .co

#![cfg(feature = "v1")]

use rocmforge::v1::ga::compile::{compile_hip_source, find_hipcc, CompileError};
use rocmforge::v1::ir::codegen_gpu::{
    emit_q4_k_gemv_gate_up_swiglu_parametric, ga_gate_up_swiglu_symbol,
};

// ── CPU-only ────────────────────────────────────────────────────────────

#[test]
fn test_hipcc_available() {
    let hipcc = find_hipcc().expect("hipcc must be installed in the dev env");
    println!("hipcc located at: {}", hipcc.display());
    assert!(
        hipcc.is_file() || hipcc.to_string_lossy() == "hipcc",
        "find_hipcc returned a non-file: {}",
        hipcc.display()
    );
}

#[test]
fn test_parametric_symbol_unique_per_num_waves() {
    let s2 = ga_gate_up_swiglu_symbol(2);
    let s4 = ga_gate_up_swiglu_symbol(4);
    let s8 = ga_gate_up_swiglu_symbol(8);
    assert_ne!(s2, s4);
    assert_ne!(s4, s8);
    assert!(s8.contains("w8"));
    assert!(s2.contains("w2"));
}

#[test]
fn test_parametric_source_contains_symbol_and_num_waves() {
    for &w in &[1u32, 2, 4, 8] {
        let (src, sym) = emit_q4_k_gemv_gate_up_swiglu_parametric(w);
        assert!(src.contains(&sym), "emitted source missing its own symbol");
        assert!(
            src.contains(&format!("#define Q4_K_FIXED_WAVES       {w}")),
            "emitted source for num_waves={w} missing the expected #define"
        );
        assert!(
            src.contains("extern \"C\" __launch_bounds__"),
            "kernel must be `extern \"C\"` so hipModuleGetFunction sees an unmangled symbol"
        );
    }
}

// ── GPU-gated ───────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
mod gpu_tests {
    use super::*;

    use std::ffi::c_void;
    use std::time::Instant;

    use half::f16;
    use rocmforge::v1::backend::gpu::error::{check, HipResult};
    use rocmforge::v1::backend::gpu::gemv::rocmforge_launch_gemv_q4_k_gate_up_swiglu;
    use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
    use rocmforge::v1::backend::gpu::module::HipModule;
    use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
    use rocmforge::v1::ga::compile::parse_amdgpu_metadata;
    use rocmforge::v1::ga::dynamic::{DynamicKernel, GateUpSwigluGeometry};
    use serial_test::serial;

    // ── Smoke: trivial compile + launch ─────────────────────────────

    #[test]
    #[serial]
    fn test_compile_trivial_kernel() {
        let src = r#"
            #include <hip/hip_runtime.h>
            extern "C" __global__ void rf_test_noop(int* x) { (void)x; }
        "#;
        let co = compile_hip_source(src, "rf_test_noop").expect("compile noop");
        assert!(!co.is_empty(), "hipcc produced empty .co");
        println!("noop .co: {} bytes", co.len());
    }

    #[test]
    fn test_compile_error_reported() {
        let src = "this is not valid HIP source code";
        let result = compile_hip_source(src, "rf_test_bad");
        match result {
            Err(CompileError::HipccFailed { stderr, .. }) => {
                assert!(
                    !stderr.is_empty(),
                    "HipccFailed must carry compiler diagnostics"
                );
            }
            other => panic!("expected HipccFailed, got {other:?}"),
        }
    }

    #[test]
    #[serial]
    fn test_module_launch_trivial_kernel() {
        // Launch a kernel that writes a constant into every output
        // slot. Confirms the whole compile → load → get_function →
        // launch → readback path works end-to-end.
        let src = r#"
            #include <hip/hip_runtime.h>
            extern "C" __global__ void rf_test_fill(int* out, int n, int value) {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                if (i < n) out[i] = value;
            }
        "#;
        let co = compile_hip_source(src, "rf_test_fill").expect("compile fill");
        let module = HipModule::load(&co).expect("load fill");
        let func = module.get_function("rf_test_fill").expect("get fill");

        let n: i32 = 128;
        let value: i32 = 0xCAFE;
        let bytes = (n as usize) * 4;
        let mut d_out = HipBuffer::new(bytes).expect("alloc d_out");
        let stream = HipStream::new().expect("stream");

        let out_ptr = d_out.as_mut_ptr();
        let mut args: [*mut c_void; 3] = [
            &out_ptr as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
            &value as *const _ as *mut c_void,
        ];
        func.launch((1, 1, 1), (n as u32, 1, 1), 0, &stream, &mut args)
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut host = vec![0u8; bytes];
        let rc = unsafe {
            hipMemcpy(
                host.as_mut_ptr() as *mut c_void,
                d_out.as_ptr(),
                bytes,
                hipMemcpyDeviceToHost,
            )
        };
        check(rc, "readback").expect("readback");

        for (i, chunk) in host.chunks_exact(4).enumerate() {
            let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            assert_eq!(v, value, "slot {i} expected {value:#x}, got {v:#x}");
        }
    }

    // ── Parametric compile: all num_waves values land a valid .co ───

    #[test]
    #[serial]
    fn test_parametric_codegen_hipcc_compiles() {
        for &w in &[1u32, 2, 4, 8] {
            let (src, sym) = emit_q4_k_gemv_gate_up_swiglu_parametric(w);
            let co = compile_hip_source(&src, &format!("test_w{w}"))
                .unwrap_or_else(|e| panic!("hipcc failed for num_waves={w}: {e}"));
            let module = HipModule::load(&co).expect("load");
            let _ = module.get_function(&sym).expect("symbol lookup");
            // VGPR sanity on the extracted gfx1201 ELF.
            let res = parse_amdgpu_metadata(&std::env::temp_dir()
                .join("rocmforge_ga")
                .join(format!("test_w{w}.gfx1201.co"))).ok();
            println!(
                "num_waves={w}  .co {} B  symbol={sym}  resources={:?}",
                co.len(),
                res
            );
        }
    }

    // ── CRITICAL: dynamic (num_waves=8) == static Phase-1 kernel ────

    fn gen_q4_k_weights(n_rows: usize, ncols: usize, seed: u64) -> Vec<u8> {
        assert_eq!(n_rows % 256, 0);
        let blocks_per_col = n_rows / 256;
        let total = ncols * blocks_per_col;
        let mut buf = vec![0u8; total * 144];
        let mut rng = fastrand::Rng::with_seed(seed);
        for b in 0..total {
            let d = (rng.f32() * 2.0 - 1.0) * 0.05;
            let dmin = (rng.f32() * 2.0 - 1.0) * 0.05;
            buf[b * 144..b * 144 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            buf[b * 144 + 2..b * 144 + 4]
                .copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for i in 4..16 {
                buf[b * 144 + i] = rng.u8(..) & 0x3F;
            }
            for i in 16..144 {
                buf[b * 144 + i] = rng.u8(..);
            }
        }
        buf
    }

    fn gen_input(k: usize, seed: u64) -> Vec<f32> {
        let mut rng = fastrand::Rng::with_seed(seed);
        (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect()
    }

    fn run_static_gate_up(
        weights_gate: &[u8],
        weights_up: &[u8],
        input: &[f32],
        n_rows: i32,
        ncols: i32,
    ) -> HipResult<Vec<f32>> {
        let stream = HipStream::new()?;
        let mut d_wg = HipBuffer::new(weights_gate.len())?;
        d_wg.copy_from_host(weights_gate)?;
        let mut d_wu = HipBuffer::new(weights_up.len())?;
        d_wu.copy_from_host(weights_up)?;
        let in_bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        let mut d_in = HipBuffer::new(in_bytes.len())?;
        d_in.copy_from_host(in_bytes)?;
        let mut d_out = HipBuffer::new((ncols as usize) * 4)?;

        let rc = unsafe {
            rocmforge_launch_gemv_q4_k_gate_up_swiglu(
                d_wg.as_ptr() as *const u8,
                d_wu.as_ptr() as *const u8,
                d_in.as_ptr() as *const f32,
                d_out.as_mut_ptr() as *mut f32,
                n_rows,
                ncols,
                stream.raw(),
            )
        };
        check(rc, "static gate_up_swiglu")?;
        stream.synchronize()?;

        let out_bytes = (ncols as usize) * 4;
        let mut host = vec![0u8; out_bytes];
        let rc = unsafe {
            hipMemcpy(
                host.as_mut_ptr() as *mut c_void,
                d_out.as_ptr(),
                out_bytes,
                hipMemcpyDeviceToHost,
            )
        };
        check(rc, "static readback")?;
        Ok(host
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn run_dynamic_gate_up(
        kernel: &DynamicKernel,
        weights_gate: &[u8],
        weights_up: &[u8],
        input: &[f32],
        n_rows: i32,
        ncols: i32,
    ) -> HipResult<Vec<f32>> {
        let stream = HipStream::new()?;
        let mut d_wg = HipBuffer::new(weights_gate.len())?;
        d_wg.copy_from_host(weights_gate)?;
        let mut d_wu = HipBuffer::new(weights_up.len())?;
        d_wu.copy_from_host(weights_up)?;
        let in_bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        let mut d_in = HipBuffer::new(in_bytes.len())?;
        d_in.copy_from_host(in_bytes)?;
        let mut d_out = HipBuffer::new((ncols as usize) * 4)?;

        kernel.launch_gate_up_swiglu(&d_wg, &d_wu, &d_in, &mut d_out, n_rows, ncols, &stream)?;
        stream.synchronize()?;

        let out_bytes = (ncols as usize) * 4;
        let mut host = vec![0u8; out_bytes];
        let rc = unsafe {
            hipMemcpy(
                host.as_mut_ptr() as *mut c_void,
                d_out.as_ptr(),
                out_bytes,
                hipMemcpyDeviceToHost,
            )
        };
        check(rc, "dynamic readback")?;
        Ok(host
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn compile_dynamic_gate_up(num_waves: u32) -> DynamicKernel {
        let (src, sym) = emit_q4_k_gemv_gate_up_swiglu_parametric(num_waves);
        let co = compile_hip_source(&src, &format!("dyn_gate_up_w{num_waves}"))
            .expect("compile dynamic gate_up");
        DynamicKernel::from_code_object(&co, sym, GateUpSwigluGeometry::for_num_waves(num_waves))
            .expect("load dynamic kernel")
    }

    #[test]
    #[serial]
    fn test_dynamic_equals_static_kernel() {
        // The static Phase-1 kernel uses Q4_K_FIXED_WAVES=8. A dynamic
        // kernel emitted with num_waves=8 runs bit-identical code —
        // `max_abs_err` must be exactly 0.0.
        let n_rows = 4096; // K
        let ncols = 256; // N — divisible by 8×4 so every tile is full
        let weights_gate = gen_q4_k_weights(n_rows, ncols, 0xA1);
        let weights_up = gen_q4_k_weights(n_rows, ncols, 0xA2);
        let input = gen_input(n_rows, 0xA3);

        let stat = run_static_gate_up(&weights_gate, &weights_up, &input, n_rows as i32, ncols as i32)
            .expect("static");
        let kernel = compile_dynamic_gate_up(8);
        let dyn_out = run_dynamic_gate_up(
            &kernel,
            &weights_gate,
            &weights_up,
            &input,
            n_rows as i32,
            ncols as i32,
        )
        .expect("dynamic");

        let max_err = stat
            .iter()
            .zip(dyn_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("dynamic(w=8) vs static: max_abs_err = {max_err:.3e}");
        assert_eq!(
            max_err, 0.0,
            "dynamic kernel with num_waves=8 must match the static Phase-1 kernel bit-for-bit"
        );
    }

    // ── Parametrization actually affects runtime ────────────────────

    fn bench_us(
        kernel: &DynamicKernel,
        weights_gate: &[u8],
        weights_up: &[u8],
        input: &[f32],
        n_rows: i32,
        ncols: i32,
    ) -> f64 {
        let stream = HipStream::new().unwrap();
        let mut d_wg = HipBuffer::new(weights_gate.len()).unwrap();
        d_wg.copy_from_host(weights_gate).unwrap();
        let mut d_wu = HipBuffer::new(weights_up.len()).unwrap();
        d_wu.copy_from_host(weights_up).unwrap();
        let in_bytes =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
        let mut d_in = HipBuffer::new(in_bytes.len()).unwrap();
        d_in.copy_from_host(in_bytes).unwrap();
        let mut d_out = HipBuffer::new((ncols as usize) * 4).unwrap();

        // Warmup.
        for _ in 0..5 {
            kernel
                .launch_gate_up_swiglu(&d_wg, &d_wu, &d_in, &mut d_out, n_rows, ncols, &stream)
                .unwrap();
        }
        stream.synchronize().unwrap();

        let mut samples = Vec::with_capacity(20);
        for _ in 0..20 {
            let t0 = Instant::now();
            kernel
                .launch_gate_up_swiglu(&d_wg, &d_wu, &d_in, &mut d_out, n_rows, ncols, &stream)
                .unwrap();
            stream.synchronize().unwrap();
            samples.push(t0.elapsed().as_micros() as f64);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2]
    }

    #[test]
    #[serial]
    fn test_different_num_waves_different_timings() {
        // num_waves=2 and num_waves=8 launch different grid/block
        // geometries. Their median timings on the same shape must
        // differ measurably — if they don't, something in the
        // substitution path is silently collapsing both configs to
        // the same binary.
        let n_rows = 4096;
        let ncols = 512;
        let weights_gate = gen_q4_k_weights(n_rows, ncols, 0xB1);
        let weights_up = gen_q4_k_weights(n_rows, ncols, 0xB2);
        let input = gen_input(n_rows, 0xB3);

        let k2 = compile_dynamic_gate_up(2);
        let k8 = compile_dynamic_gate_up(8);

        let us2 = bench_us(&k2, &weights_gate, &weights_up, &input, n_rows as i32, ncols as i32);
        let us8 = bench_us(&k8, &weights_gate, &weights_up, &input, n_rows as i32, ncols as i32);

        println!("gate_up_swiglu median: num_waves=2 → {us2:.0} µs, num_waves=8 → {us8:.0} µs");
        let ratio = if us2 > us8 { us2 / us8 } else { us8 / us2 };
        assert!(
            ratio > 1.05,
            "num_waves=2 vs 8 should differ by >5 %; got us2={us2} us8={us8}"
        );
    }

    // ── Post-compile VGPR readable on dynamic .co ──────────────────

    #[test]
    #[serial]
    fn test_post_compile_vgpr_from_dynamic_co() {
        // Block B's compile_hip_source runs clang-offload-bundler
        // automatically, so the extracted gfx1201 ELF lives at
        // `<tmp>/rocmforge_ga/<kernel>.gfx1201.co`. The 2.1.1 VGPR
        // reader works on that file directly — no extra
        // llvm-objdump step needed.
        let (src, _sym) = emit_q4_k_gemv_gate_up_swiglu_parametric(4);
        let _co = compile_hip_source(&src, "vgpr_probe_w4").expect("compile vgpr probe");
        let gpu_elf = std::env::temp_dir()
            .join("rocmforge_ga")
            .join("vgpr_probe_w4.gfx1201.co");
        assert!(gpu_elf.is_file(), "expected extracted gfx1201 ELF at {:?}", gpu_elf);

        let res = parse_amdgpu_metadata(&gpu_elf)
            .expect("llvm-readobj on dynamic .co")
            .expect("AMDGPU metadata present");
        println!(
            "dynamic num_waves=4 VGPRs={} SGPRs={} LDS={} (waves/CU={})",
            res.vgpr_count,
            res.sgpr_count,
            res.lds_bytes,
            res.max_waves_per_cu()
        );
        assert!(res.vgpr_count > 0 && res.vgpr_count < 256);
    }
}
