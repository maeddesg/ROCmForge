//! Kernel-Variant-Registry — Shape → list of candidate kernels.
//!
//! Phase-1 variants are hand-registered: each shape has the 1–2
//! kernels already built in `hip_kernels_v1/`. Phase-2 replaces this
//! with GA-generated variants cached to disk, but the interface stays
//! the same: the Bandit picks between `KernelVariant`s, Executor
//! dispatches the chosen `KernelId`.

use std::collections::HashMap;

use crate::v1::core::tensor_info::GgmlType;

/// Opaque handle for a variant inside a `ShapeBandit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VariantId(pub u32);

/// Hash key that groups variants by what they compute. Only shapes
/// with more than one registered variant get a Bandit; identical
/// (op_type, format, n, k) from different layers share the same
/// statistics, which is exactly what we want — a GEMV at 4096×4096
/// has the same hot path no matter which block it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeKey {
    pub op_type: OpType,
    pub format: GgmlType,
    pub n: u32,
    pub k: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    Gemv,
    Wmma,
}

/// Identifies the concrete kernel to launch. The executor matches on
/// this enum and calls the right `rocmforge_launch_*` FFI function.
/// Extending the set is a single-line enum + dispatch arm — no
/// plumbing through trait objects or unsafe `fn()` pointers, which
/// would drag the whole HIP type system into the runtime crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelId {
    // GEMV standard (one per quant format).
    GemvQ40Standard,
    GemvQ4KStandard,
    GemvQ6KStandard,
    GemvQ80Standard,
    // GEMV specialised paths.
    GemvQ4KQ8Inline,
    // WMMA GEMM variants (registered for Phase-1 completeness; wired
    // into batched prefill in Phase-2).
    WmmaQ40Fp16,
    WmmaQ40Fp8,
    WmmaQ4KFp16,
    WmmaQ4KFp8,
    WmmaQ6KFp16,
    WmmaQ6KFp8,
    WmmaQ80Fp16,
    WmmaQ80Fp8,
}

#[derive(Debug, Clone)]
pub struct KernelVariant {
    pub id: VariantId,
    pub name: &'static str,
    pub kernel: KernelId,
}

#[derive(Default)]
pub struct VariantRegistry {
    pub variants: HashMap<ShapeKey, Vec<KernelVariant>>,
    next_variant_id: u32,
}

impl VariantRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a kernel to the registry for `shape`. Idempotent: if the
    /// same `(shape, kernel)` pair is already registered, the existing
    /// variant id is returned. Callers (graph builder, test fixtures)
    /// can walk every layer and register without fearing duplicates.
    pub fn register(&mut self, shape: ShapeKey, name: &'static str, kernel: KernelId) -> VariantId {
        let entry = self.variants.entry(shape).or_default();
        if let Some(existing) = entry.iter().find(|v| v.kernel == kernel) {
            return existing.id;
        }
        let id = VariantId(self.next_variant_id);
        self.next_variant_id += 1;
        entry.push(KernelVariant { id, name, kernel });
        id
    }

    pub fn get_variants(&self, shape: &ShapeKey) -> &[KernelVariant] {
        self.variants
            .get(shape)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Look up the kernel behind a variant id in a given shape.
    pub fn lookup(&self, shape: &ShapeKey, variant: VariantId) -> Option<KernelId> {
        self.get_variants(shape)
            .iter()
            .find(|v| v.id == variant)
            .map(|v| v.kernel)
    }

    /// Register every variant available in Phase 1. Called once at
    /// pipeline construction for every GEMV shape the graph produces;
    /// the Runtime then builds a Bandit per shape with ≥2 entries.
    ///
    /// For GEMV shapes we register:
    ///   * Q4_K: `standard` + `q8_inline`       (2 variants → Bandit)
    ///   * Q4_0 / Q6_K / Q8_0: `standard` only  (1 variant → no Bandit)
    ///
    /// WMMA variants are registered globally by `register_wmma_shape`
    /// once the graph exposes the prefill shapes (Phase 2).
    pub fn register_gemv_shape(&mut self, format: GgmlType, n: u32, k: u32) {
        let shape = ShapeKey { op_type: OpType::Gemv, format, n, k };
        match format {
            GgmlType::Q4_0 => {
                self.register(shape, "q4_0_standard", KernelId::GemvQ40Standard);
            }
            GgmlType::Q4_K => {
                self.register(shape, "q4_k_standard", KernelId::GemvQ4KStandard);
                self.register(shape, "q4_k_q8_inline", KernelId::GemvQ4KQ8Inline);
            }
            GgmlType::Q6_K => {
                self.register(shape, "q6_k_standard", KernelId::GemvQ6KStandard);
            }
            GgmlType::Q8_0 => {
                self.register(shape, "q8_0_standard", KernelId::GemvQ80Standard);
            }
            _ => {
                // Non-quantised formats use hipBLAS or FP32 kernels,
                // which the Phase-1 graph does not dispatch through
                // this code path — no variants to register.
            }
        }
    }

    /// Register the WMMA FP16 + FP8 variants for a prefill shape.
    /// Phase 1 does sequential decode for prefill, so this is wired
    /// up but not yet called on the hot path.
    pub fn register_wmma_shape(&mut self, format: GgmlType, n: u32, k: u32) {
        let shape = ShapeKey { op_type: OpType::Wmma, format, n, k };
        match format {
            GgmlType::Q4_0 => {
                self.register(shape, "q4_0_wmma_fp16", KernelId::WmmaQ40Fp16);
                self.register(shape, "q4_0_wmma_fp8", KernelId::WmmaQ40Fp8);
            }
            GgmlType::Q4_K => {
                self.register(shape, "q4_k_wmma_fp16", KernelId::WmmaQ4KFp16);
                self.register(shape, "q4_k_wmma_fp8", KernelId::WmmaQ4KFp8);
            }
            GgmlType::Q6_K => {
                self.register(shape, "q6_k_wmma_fp16", KernelId::WmmaQ6KFp16);
                self.register(shape, "q6_k_wmma_fp8", KernelId::WmmaQ6KFp8);
            }
            GgmlType::Q8_0 => {
                self.register(shape, "q8_0_wmma_fp16", KernelId::WmmaQ80Fp16);
                self.register(shape, "q8_0_wmma_fp8", KernelId::WmmaQ80Fp8);
            }
            _ => {}
        }
    }

    pub fn num_shapes(&self) -> usize {
        self.variants.len()
    }
}
