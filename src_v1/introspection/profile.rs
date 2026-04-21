//! ModelProfile — output of Säule 1 (architecture_v1.2.0-draft §2.2).
//!
//! Names, fields, and semantics are verbatim from the doc. Phase 1
//! fills every field; Phase 2 precision-GA consumes the
//! `precision_recommendation` as its seed population and the
//! `critical_embedding_tokens` list as the FP32-overlay target set.

pub type TokenId = u32;

/// Per-tensor magnitude statistics from the Stichprobe (sample) scan.
#[derive(Debug, Clone)]
pub struct LayerStats {
    pub layer_index: usize,
    pub tensor_name: String,
    pub mean_abs: f32,
    pub max_abs: f32,
    pub std_abs: f32,
    pub element_count: usize,
}

/// Precision seed for the Phase-2 Allocation-GA. Names match §2.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionHint {
    /// FP8 E4M3 — default weight path in v1.1. Range ±448,
    /// halved VGPR footprint, 2× WMMA throughput vs. FP16.
    Fp8E4M3,
    /// FP8 E5M2 — activations and KV-cache (larger dynamic range).
    Fp8E5M2,
    /// FP16 scales — fallback tier after FP8.
    Fp16Scales,
    /// FP32 scales — high dynamic range weights (max_abs ≫ mean_abs).
    Fp32Scales,
    /// BF16 — similar speed to FP16, larger dynamic range, picked
    /// when the SNR risk signals that FP16 might under-represent
    /// critical tokens.
    Bf16Scales,
}

/// Result of the one-shot introspection pass run at model-load time.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// (min, max) L2-norm across the embedding rows.
    pub embedding_magnitude_range: (f32, f32),

    /// Embedding-row indices with L2 norm below 10 % of the mean.
    /// These are the SNR candidates for Phase-2 FP32-overlay.
    pub critical_embedding_tokens: Vec<TokenId>,

    /// Per-tensor stats from the layer Stichprobe.
    pub layer_magnitude_stats: Vec<LayerStats>,

    /// Dequant error estimate in L2 units, derived from the
    /// dominant quant format's per-value noise ratio times the
    /// mean magnitude observed.
    pub quantization_noise_estimate: f32,

    /// Signal-to-noise ratio risk. < 1.0 critical, 1.0–2.0 warn,
    /// > 2.0 unproblematic.
    pub snr_risk_score: f32,

    /// Per-layer precision recommendation — seed for the GA.
    pub precision_recommendation: Vec<PrecisionHint>,
}

impl ModelProfile {
    /// True if any embedding row triggered the "critical" threshold
    /// or the SNR risk score dropped into the warn band.
    pub fn has_risk(&self) -> bool {
        self.snr_risk_score < 2.0 || !self.critical_embedding_tokens.is_empty()
    }

    /// Pretty-print a one-shot summary table to stdout. Format is
    /// chosen so the operator can tell at a glance whether a model
    /// needs extra precision.
    pub fn print_summary(&self) {
        let (emb_min, emb_max) = self.embedding_magnitude_range;
        let band = if self.snr_risk_score < 1.0 {
            "CRITICAL"
        } else if self.snr_risk_score < 2.0 {
            "warn"
        } else {
            "ok"
        };
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                    Model Introspection                      ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║ Embedding L2 range     : {:>10.5}  …  {:>10.5}         ║",
            emb_min, emb_max
        );
        println!(
            "║ Critical tokens        : {:>5}                               ║",
            self.critical_embedding_tokens.len()
        );
        println!(
            "║ Quant noise estimate   : {:>10.5} (L2)                    ║",
            self.quantization_noise_estimate
        );
        println!(
            "║ SNR risk score         : {:>10.3}  [{:<8}]                ║",
            self.snr_risk_score, band
        );
        println!(
            "║ Layer stats rows       : {:>5}                               ║",
            self.layer_magnitude_stats.len()
        );
        println!(
            "║ Precision recommend    : FP8={:<3} FP16={:<3} BF16={:<3} FP32={:<3}    ║",
            self.count_hint(PrecisionHint::Fp8E4M3),
            self.count_hint(PrecisionHint::Fp16Scales),
            self.count_hint(PrecisionHint::Bf16Scales),
            self.count_hint(PrecisionHint::Fp32Scales),
        );
        println!("╚══════════════════════════════════════════════════════════════╝");

        if !self.critical_embedding_tokens.is_empty() {
            let preview: Vec<String> = self
                .critical_embedding_tokens
                .iter()
                .take(16)
                .map(|t| t.to_string())
                .collect();
            let more = if self.critical_embedding_tokens.len() > 16 {
                format!(" (+{} more)", self.critical_embedding_tokens.len() - 16)
            } else {
                String::new()
            };
            println!(
                "  critical token ids: {}{}",
                preview.join(", "),
                more
            );
        }
        println!();
    }

    fn count_hint(&self, hint: PrecisionHint) -> usize {
        self.precision_recommendation
            .iter()
            .filter(|h| **h == hint)
            .count()
    }
}
