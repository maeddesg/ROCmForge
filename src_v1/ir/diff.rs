//! Element-level diff between CPU-interpreter and GPU-kernel block
//! dequant outputs.
//!
//! Phase-1 goal (dequant_ir_spec §7.4): CPU and GPU must produce
//! **bit-identical** FP32 for the same input block. `DiffMode::BitExact`
//! compares via `f32::to_bits` so +0.0 vs −0.0 and NaN-pattern drift
//! are caught. `DiffMode::Tolerance` is provided for v1.1 transcendentals
//! but not used by any current op.

/// Per-element diff entry.
#[derive(Debug, Clone, Copy)]
pub struct IrDiff {
    pub element_index: usize,
    pub cpu_value: f32,
    pub gpu_value: f32,
    pub abs_diff: f32,
    pub rel_diff: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum DiffMode {
    /// Byte-wise FP32 comparison; any bit difference is a diff.
    BitExact,
    /// `abs(cpu - gpu) < max_abs_err` passes silently.
    Tolerance { max_abs_err: f32 },
}

/// Compute per-element diffs. Elements that agree under the chosen
/// mode are omitted from the result; an empty vec means "fully equal".
pub fn diff_block(cpu: &[f32], gpu: &[f32], mode: DiffMode) -> Vec<IrDiff> {
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "CPU/GPU output length mismatch: {} vs {}",
        cpu.len(),
        gpu.len()
    );
    let mut diffs = Vec::new();
    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let equal = match mode {
            DiffMode::BitExact => c.to_bits() == g.to_bits(),
            DiffMode::Tolerance { max_abs_err } => (c - g).abs() < max_abs_err,
        };
        if !equal {
            let abs = (c - g).abs();
            let rel = if c.abs() > 0.0 { abs / c.abs() } else { abs };
            diffs.push(IrDiff {
                element_index: i,
                cpu_value: c,
                gpu_value: g,
                abs_diff: abs,
                rel_diff: rel,
            });
        }
    }
    diffs
}

/// Pretty-print a diff report for one block.
pub fn format_diff_report(format_name: &str, diffs: &[IrDiff], total: usize) -> String {
    let mut s = String::new();
    s.push_str(&format!("{format_name} Block Diff Report:\n"));
    if diffs.is_empty() {
        s.push_str(&format!("  PASS: {total}/{total} elements bit-identical\n"));
    } else {
        s.push_str(&format!(
            "  FAIL: {}/{} elements differ\n",
            diffs.len(),
            total
        ));
        // Show up to 10 diffs; hide the rest behind a count.
        for d in diffs.iter().take(10) {
            s.push_str(&format!(
                "    elem {:>4}: CPU=0x{:08X} ({}) GPU=0x{:08X} ({}) abs={:.3e} rel={:.3e}\n",
                d.element_index,
                d.cpu_value.to_bits(),
                d.cpu_value,
                d.gpu_value.to_bits(),
                d.gpu_value,
                d.abs_diff,
                d.rel_diff,
            ));
        }
        if diffs.len() > 10 {
            s.push_str(&format!("    … ({} more)\n", diffs.len() - 10));
        }
    }
    s
}
