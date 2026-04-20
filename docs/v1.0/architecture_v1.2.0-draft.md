# ROCmForge v1.0 — Architektur-Dokument

**Status:** Design-Dokument, finale Freigabe zur Implementierung
**Scope:** Kompletter Neubau der Architektur auf Basis der v0.x-Erkenntnisse
**Hardware:** AMD RDNA 4 (gfx1201, 64 CU / 128 AI Accelerators) + AMD Zen4 (AVX-512 Double-Pumped)
**Sprache:** Rust (Host + FFI), HIP/C++ (GPU-Kernel via hipcc)
**Modell-Klassen:** 8B (Q4_K_M, ~4,5 GB) und 14B (Q4_K_M, ~9,0 GB) auf 16 GB VRAM
**Zeitrahmen:** 5–6 Wochen Entwicklung, Meilenstein 0 + 3 Phasen
**Autor:** mg
**Version:** 1.2.0-draft (Rust-Native, 14B-Squeeze, FP8-Gate, Single-Stream)

---

## Inhaltsverzeichnis

1. [Executive Summary](#1-executive-summary)
2. [Architektur-Stack — Die 6 Säulen](#2-architektur-stack--die-6-säulen)
3. [Hardware-Backend-Spezifikation](#3-hardware-backend-spezifikation)
4. [Genetische Algorithmen](#4-genetische-algorithmen)
5. [Datenflüsse](#5-datenflüsse)
6. [Schnittstellen-Definitionen](#6-schnittstellen-definitionen)
7. [Konkrete Walk-Through-Beispiele](#7-konkrete-walk-through-beispiele)
8. [Migration von v0.x](#8-migration-von-v0x)
9. [Roadmap mit Meilensteinen](#9-roadmap-mit-meilensteinen)
10. [Schlüssel-Erkenntnisse aus v0.x](#10-schlüssel-erkenntnisse-aus-v0x)
11. [Anhänge — Glossar, Referenzen, CLI-Tooling](#anhang-a--glossar)

---

## 1. Executive Summary

### 1.1 Was ist ROCmForge v1.0

ROCmForge v1.0 ist eine **selbst-optimierende, Hardware-native LLM-Inference-Engine** für AMD-Hardware, geschrieben in Rust. Im Gegensatz zu v0.x ist v1.0 kein Werkzeugkasten handgeschriebener Kernel, sondern ein **System, das seine eigenen Kernel-, Precision- und Fusion-Strategien für ein gegebenes Modell und eine gegebene Hardware findet** — und das konsequent die nativen Hardware-Features der Zielplattform ausreizt.

Die Engine zielt ausschließlich auf zwei Hardware-Targets:

- **GPU:** AMD Radeon RX 9070 XT (`gfx1201`, RDNA 4) — **64 Compute Units, 128 AI Accelerators** (2 pro CU), natives WMMA mit **FP8-Support (E4M3/E5M2)**, 16 GB GDDR6-VRAM, ~644 GB/s Bandbreite.
- **CPU:** AMD Ryzen 9 7945HX (Zen4) — 16 Kerne / 32 Threads, AVX-512 **Double-Pumped** (512-Bit-Operationen über 256-Bit-Datenpfade, **ohne** die Takt-Downclocking-Probleme der älteren Intel-AVX-512-Implementierung), VNNI, BF16.

Dieser Fokus ist eine bewusste strategische Entscheidung, keine Limitierung. Er ermöglicht einen Ansatz, den generische Engines wie llama.cpp oder vLLM aus Kompatibilitätsgründen nicht wählen können: **jeder Kernel wird genau auf diese beiden Targets zugeschnitten, ohne Fallbacks, ohne Abstraktionslayer, ohne Kompromisse — und nutzt die RDNA-4-spezifischen FP8-WMMA-Instruktionen als Standardpfad, nicht als Experiment**.

### 1.2 Warum ein Neubau statt Weiterentwicklung

v0.x hat bewiesen, dass WMMA auf `gfx1201` funktioniert und dass ROCmForge bis auf 50–60 % der llama.cpp-Performance herankommt (v0.3.0: 43 tok/s Decode, 1 383 tok/s Prefill vs. llama.cpp 87 tok/s / 3 661 tok/s für Qwen3-8B Q4_K_M). Aber die Architektur von v0.x skaliert nicht.

Das zentrale Problem ist die kombinatorische Explosion:

```
    Aufwand(v0.x) = O(Modelle × Quant-Formate × Fusions-Varianten)
```

Jedes neue Modell erfordert architekturspezifische Dispatch-Logik. Jedes neue Quant-Format erfordert neue WMMA- und GEMV-Kernel (Q4_K, Q6_K, Q4_0, Q8_0 sind bereits vier separate Kernelpaare). Jede neue Fusion (Norm+GEMV, Gate+Up+SwiGLU, ...) muss pro Quant-Format neu geschrieben werden. Das Ergebnis ist eine Matrix aus `N × M × F` Kernel-Dateien, die alle einzeln getestet, profiled und gewartet werden müssen.

v1.0 löst das durch **Orthogonalisierung**: Modellarchitektur, Quant-Format und Fusion werden zu unabhängigen Dimensionen, die nicht mehr multiplikativ kombiniert werden müssen.

```
    Aufwand(v1.0) = O(Modelle) + O(Quant-Formate) + O(Fusions-Varianten)
```

Konkret: ein neues Quant-Format (z. B. Q5_K) wird zu einer Datenstruktur-Definition im Dequant-IR — rund 30 Minuten Arbeit — statt eines neuen Kernels. Ein neues Modell (z. B. Llama-4) wird automatisch aus den GGUF-Metadaten als Computation Graph rekonstruiert, ohne modellspezifischen Code.

### 1.3 Hardware-Fokus

Der Verzicht auf Hardware-Portabilität ist der Schlüssel zur Performance-Führung auf `gfx1201`. llama.cpp muss auf 50+ GPU-Architekturen laufen (gfx908 CDNA, gfx1030 RDNA 2, gfx1100 RDNA 3, gfx1201 RDNA 4, ...) und nutzt auf RDNA 4 weder die nativen WMMA-Intrinsics noch die FP8-Pfade, weil die Kernel-Fallbacks auf Wave64-CDNA-Code zurückgreifen. ROCmForge v1.0 adressiert die WMMA- und FP8-Einheiten direkt und hat dadurch einen strukturellen Performance-Vorteil, der nicht von Software-Optimierung abhängt, sondern von Hardware-Zugang.

Konkrete fixe Hardware-Annahmen:

| Parameter | Wert | Begründung |
|---|---|---|
| GPU Compute Units | **64 CU** | RX 9070 XT Vollausbau |
| GPU AI Accelerators | **128** (2 pro CU) | RDNA-4-spezifische WMMA-Einheiten |
| GPU Wavefront-Größe | 32 Lanes | RDNA 4 native, kein Wave64-Fallback |
| GPU LDS pro Workgroup | 64 KB | `gfx1201` Hardware-Limit |
| GPU Register-Budget (VGPRs/Wave) | **104 (Occupancy-Target)** | bewusste Limitierung trotz 256 Hardware-Max (siehe Abschnitt 3.3) |
| WMMA-Tile | 16×16×16 | FP16, BF16, **FP8 (E4M3/E5M2)** → FP32 |
| **WMMA FP8-Pfad** | **nativ unterstützt** | RDNA-4-Kern-Feature, Standardpfad für v1.0 |
| GPU Memory Bandwidth | ~644 GB/s | RX 9070 XT GDDR6 |
| CPU SIMD | AVX-512 + VNNI + BF16 | Zen4 Instruktions-Set |
| CPU AVX-512-Implementierung | **Double-Pumped (256b×2)** | kein Takt-Downclocking bei 512-Bit-Last |
| CPU Cache-Line | 64 Byte | fix |
| CPU Threads (physisch/SMT) | 16 / 32 | 7945HX |

### 1.4 Ziel-Performance

**Physikalisches Grundprinzip.** Decode (Batch=1) ist **bandbreiten-gebunden**: jedes generierte Token erfordert das Lesen aller Modellgewichte. Das BW-Limit ergibt sich direkt aus `Gewichtsgröße / Bandbreite`. Kein Software-Trick umgeht das. Prefill ist **compute-gebunden**: hier schlagen die 64 CUs und FP8-WMMA voll durch.

**Performance-Matrix nach Modellklasse:**

| Modell | Gewichte | BW-Limit (644 GB/s) | Konservativ | Realistisch | Optimistisch |
|---|---:|---:|---:|---:|---:|
| **8B Q4_K_M** | ~4,5 GB | 143 tok/s | 110 tok/s | **125 tok/s** | 140 tok/s |
| **8B Q4_0** | ~4,0 GB | 161 tok/s | 130 tok/s | **140 tok/s** | 155 tok/s |
| **14B Q4_K_M** | ~9,0 GB | 72 tok/s | 55 tok/s | **63 tok/s** | 68 tok/s |
| **14B Q4_0** | ~8,0 GB | 81 tok/s | 62 tok/s | **72 tok/s** | 77 tok/s |

| Modell | Prefill pp256 konserv. | Prefill pp256 realist. | Referenz llama.cpp |
|---|---:|---:|---|
| **8B Q4_K_M** | 4 500 tok/s | **7 500 tok/s** | 3 661 |
| **14B Q4_K_M** | 2 500 tok/s | **4 000 tok/s** | ~2 000 (geschätzt) |

Die realistischen Decode-Ziele entsprechen 87–89 % der theoretischen Bandbreiten-Effizienz. Das ist ambitioniert — llama.cpp erreicht typischerweise 60–70 %. Der Vorsprung entsteht durch: native WMMA-Nutzung (eliminiert VALU-Overhead), aggressive Kernel-Fusion (weniger Dispatch-Overhead und Memory-Roundtrips), und GA-optimierte Tile-Sizes (nicht manuell, sondern gemessen).

**Kontextlängen-Abhängigkeit.** Die Decode-Ziele gelten bei kurzen Kontexten (< 2k). Bei 8k-Kontext sinkt die Decode-Rate um ~5–10 % durch KV-Cache-Read-Overhead; bei 32k um ~15–20 %. Diese Kontextlängen-Sensitivität wird im Benchmark-Report pro Modell ausgewiesen.

**Fallback bei FP8-Nicht-Verfügbarkeit** (Meilenstein 0, siehe Abschnitt 9.0): Falls die FP8-WMMA-Intrinsic auf `gfx1201` nicht funktioniert, fallen die Prefill-Ziele auf ~50 % (FP16-WMMA: halber Durchsatz, doppelte VGPR-Belegung). Decode-Ziele bleiben identisch, da sie bandbreiten-gebunden sind und nicht von der WMMA-Precision abhängen.

### 1.5 Warum v1.0 llama.cpp auf RDNA 4 schlagen kann

Das ist keine Marketing-Aussage, sondern ein strukturelles Argument:

1. **llama.cpp hat auf `gfx1201` keine WMMA-Kernel.** hipBLAS/Tensile liefert keine WMMA-Lösungen für diese Architektur; stattdessen wird auf generische VALU-Kernel zurückgegriffen. Jede WMMA-Operation in ROCmForge ist damit strukturell überlegen, nicht nur optimierter.
2. **llama.cpp nutzt die FP8-Pfade der RDNA 4 nicht.** Die nativen `v_wmma_f32_16x16x16_fp8`-Instruktionen bringen auf RDNA 4 einen 2×-Durchsatz gegenüber FP16-WMMA, bei halbierter VGPR-Belegung. v1.0 nutzt FP8 als Standard-Input-Format für WMMA; llama.cpp nicht.
3. **GA-Tuning findet Konfigurationen, die Menschen nicht findet.** In v0.x haben wir gesehen, dass Direct-Global-Loads 5× schneller sind als LDS-basiertes Staging für bestimmte Shapes — ein Ergebnis, das jeder Lehrbuch-Ansatz verworfen hätte. Ein Genetischer Algorithmus hat keine Vorurteile und misst einfach.
4. **Automatische Fusion ohne Kombinatorik.** Während llama.cpp pro Quant-Format neue Fused-Kernel benötigt, erzeugt ROCmForge v1.0 Fusionen algorithmisch über den Computation Graph — einmal implementiert, funktioniert es für jedes Format.

Der Nachteil: v1.0 läuft *nur* auf `gfx1201`+Zen4. Wer eine andere GPU hat, benutzt llama.cpp. Das ist die akzeptierte Tradeoff-Entscheidung.

---

## 2. Architektur-Stack — Die 6 Säulen

### 2.1 Überblick

Der Architektur-Stack besteht aus sechs unabhängigen, aber kooperierenden Säulen. Jede Säule hat einen klar abgegrenzten Zweck und kommuniziert mit den anderen über schmale, explizite Schnittstellen. Säule 6 (Safety & Debug) wirkt als **orthogonale Validierungsschicht** über alle Kernel-Aufrufe und ist nicht Teil des linearen Datenflusses — sie prüft, dass die WMMA-basierten Ergebnisse gegen einen unabhängigen VALU-Pfad bit-nahe übereinstimmen.

```
┌─────────────────────────────────────────────────────────────┐
│                    GGUF-Datei (Input)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Säule 1: Model Introspection                               │
│  Magnitude-Scan, SNR-Risk-Score, Precision-Empfehlung       │
│  Output: ModelProfile                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Säule 2: Computation Graph + Fusion-Passes                 │
│  Operator-Graph aus GGUF, regelbasierte + GA-Fusion         │
│  Output: FusedGraph                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Säule 3: Dequant IR (FP8-native)                           │
│  Quant-Format als DequantOp-Programm, WMMA-Input = FP8      │
│  Codegen für gfx1201 (FP8-WMMA) und Zen4 (AVX-512)          │
│  Output: CompiledKernel (GPU-Binary + CPU-Binary)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Säule 4: Self-Tuning Runtime                               │        ┌──────────────────────────────┐
│  Bandit/UCB1 zwischen GA-optimierten Kernel-Varianten       │◄──────►│ Säule 6: Safety & Debug      │
│  Output: Selected KernelVariant pro Call                    │        │ VALU-Parity-Pfad             │
└──────────────────────────┬──────────────────────────────────┘        │ WMMA vs. VALU Bit-Check      │
                           │                                           │ Sampling-Rate konfigurierbar │
┌──────────────────────────▼──────────────────────────────────┐        └──────────────────────────────┘
│  Säule 5: Quality Monitor (FP8-Drift)                       │
│  Hidden-State-Check, Precision-Fallback bei Numerik-Drift   │
│  Standardpfad: FP8 → FP16-Fallback (nicht FP32)             │
│  Output: Inference-Output oder Precision-Revision-Signal    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Token (Output)                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Säule 1: Model Introspection

**Was sie tut.** Beim Laden einer GGUF-Datei scannt diese Säule die Gewichtstensoren statistisch. Für jeden Layer und für die Embedding-Tabelle werden Magnitude-Verteilungen, Varianz, und das Verhältnis zwischen den schwächsten und stärksten Gewichten berechnet. Daraus entsteht ein **ModelProfile**, das beschreibt, wie numerisch anspruchsvoll dieses konkrete Modell in dieser konkreten Quantisierung ist.

**Warum sie nötig ist.** In v0.x hat Llama-3.1-8B Q4_K_M im Multi-Turn-Chat zusammenhangslose Tokens produziert, während Qwen3-8B Q4_K_M einwandfrei funktionierte. Ursache: die Special-Token-Embeddings von Llama-3.1 (`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`) haben L2-Normen um `0.034`, während normale Text-Tokens bei `0.55` liegen — eine 16× kleinere Signalstärke. Das Dequantisierungs-Rauschen von Q4_K (~ `0.064` L2 über 4096 Dimensionen) ist **größer als das Signal** dieser Tokens. Das SNR fällt unter 1, und über 32 Layer eskaliert der Fehler.

Keine andere Engine erkennt das. llama.cpp, vLLM und TensorRT-LLM behandeln Gewichte als opake Blobs. v1.0 tut das nicht — es misst vor der ersten Inference, wo Präzisionsprobleme drohen, und aktiviert gezielt höhere Genauigkeit nur dort, wo sie nötig ist.

**Schnittstelle.** Input ist das GGUF-File (bzw. dessen bereits in VRAM/RAM geladene Tensoren). Output ist ein `ModelProfile`, das an die Precision-Allocation-GA (Abschnitt 4.2) weitergereicht wird.

**Datenstrukturen.**

```rust
/// Ergebnis der Model-Introspection, ein einmaliger Scan beim Laden.
pub struct ModelProfile {
    /// Bereich der L2-Norm der Embedding-Rows (min, max).
    pub embedding_magnitude_range: (f32, f32),
    /// Indizes von Embedding-Rows mit weniger als 10 % der mittleren L2-Norm.
    /// Diese sind SNR-Kandidaten für Precision-Hochstufung.
    pub critical_embedding_tokens: Vec<TokenId>,
    /// Pro-Layer-Statistik: (mean_abs, max_abs, std_abs) der Gewichtsmagnituden.
    pub layer_magnitude_stats: Vec<LayerStats>,
    /// Geschätzter Dequant-Fehler in L2-Einheiten, basiert auf Quant-Format.
    pub quantization_noise_estimate: f32,
    /// Signal-to-Noise-Ratio Risk Score.
    /// < 1.0 = kritisch, 1.0-2.0 = Warnung, > 2.0 = unkritisch.
    pub snr_risk_score: f32,
    /// Empfehlung pro Layer, welche Precision für Akkumulation benutzt werden soll.
    /// Wird als Startpunkt für die Precision-GA verwendet (nicht finaler Output).
    pub precision_recommendation: Vec<PrecisionHint>,
}

pub struct LayerStats {
    pub layer_index: usize,
    pub tensor_name: String,
    pub mean_abs: f32,
    pub max_abs: f32,
    pub std_abs: f32,
    pub element_count: usize,
}

pub enum PrecisionHint {
    /// FP8 E4M3 — Standardpfad für Gewichte in v1.1. Range ±448,
    /// halbierte VGPR-Belegung, 2× WMMA-Durchsatz vs. FP16.
    Fp8E4M3,
    /// FP8 E5M2 — für Aktivierungen und KV-Cache (größerer Dynamikbereich).
    Fp8E5M2,
    /// FP16-Skalen reichen aus — Fallback-Stufe nach FP8.
    Fp16Scales,
    /// FP32-Skalen-Akkumulation nötig (kleine Magnituden, hoher Dynamikbereich).
    Fp32Scales,
    /// BF16 ist sicherer als FP16 (ähnliche Speed, größerer Dynamikbereich).
    Bf16Scales,
}

pub type TokenId = u32;
```

**Algorithmus.**

```
fn introspect(gguf: &GgufModel) -> ModelProfile:
    # 1. Embedding-Scan
    embedding = gguf.tensor("token_embd")
    l2_per_row = für jede Row in embedding: sqrt(sum(row² dequantisiert))
    critical = indices wo l2_per_row < 0.1 × mean(l2_per_row)

    # 2. Pro-Layer-Scan
    für jeden Layer i:
        für jeden Gewichtstensor (qkv, attn_proj, gate, up, down, ...):
            mean_abs, max_abs, std_abs = statistik(tensor dequantisiert)
            speichere als LayerStats

    # 3. Noise-Estimate pro Quant-Format
    #    Q4_K: ~1/16 der Magnituden-Range
    #    Q6_K: ~1/64 der Magnituden-Range
    #    Q8_0: ~1/256 der Magnituden-Range
    noise_estimate = quant_noise_table[format] × mean_magnitude

    # 4. SNR-Risk-Score
    min_magnitude = min(l2_per_row für critical Tokens) falls critical nicht leer
                    sonst min(mean_abs über alle Layer)
    snr_risk = min_magnitude / noise_estimate

    # 5. Precision-Empfehlung (Startpunkt für GA)
    für jeden Layer:
        wenn max_abs / mean_abs > 50: Fp32Scales  # hoher Dynamikbereich
        sonst wenn snr_risk < 2.0: Bf16Scales
        sonst: Fp16Scales

    return ModelProfile { ... }
```

Der Scan dauert auf einem 7945HX rund 3–5 Sekunden für ein 8B-Modell (ca. 290 Tensoren, ~4 GB Gewichte) und kann parallel zum Weight-Preloading auf die GPU laufen.

### 2.3 Säule 2: Computation Graph + Fusion-Passes

**Was sie tut.** Diese Säule konstruiert aus den GGUF-Metadaten einen typisierten Operator-Graph — einen gerichteten azyklischen Graph (DAG), dessen Knoten Transformer-Operationen sind (RMSNorm, QKV-GEMV, RoPE, FlashAttention, O-Proj, SwiGLU, FFN-Down, ...) und dessen Kanten Datenabhängigkeiten darstellen. Anschließend laufen mehrere **Fusion-Passes** über diesen Graph, die benachbarte Operationen zu einem einzigen Kernel verschmelzen.

**Warum sie nötig ist.** In v0.x war jede Fused-Operation ein eigener handgeschriebener Kernel pro Quant-Format: `fused_norm_qkv_q4_k.hip`, `fused_norm_qkv_q4_0.hip`, `fused_gate_up_swiglu_q4_k.hip`, `fused_residual_norm_q4_k.hip`, usw. Das ist die oben beschriebene `O(Quants × Fusions)`-Explosion. Gleichzeitig war die Reihenfolge der Fusionen hardcodiert, was bedeutete, dass neue Fusionen nur durch Codeänderungen im Dispatcher möglich waren.

Der Graph-Ansatz macht Fusion zu einem Optimierungsproblem über eine Datenstruktur. Eine Fusion wie "Norm+GEMV+RoPE" ist eine **Graph-Transformation**, die sagt: "wenn ein RMSNorm-Knoten direkt vor einem GEMV-Knoten steht, der wiederum vor einem RoPE-Knoten steht, und keine anderen Konsumenten existieren, dann ersetze die drei Knoten durch einen FusedOp." Diese Regel gilt **unabhängig vom Quant-Format** — die Dequantisierung ist ein Detail des GEMV-Kernels, nicht der Fusions-Logik.

**Schnittstelle.** Input ist die GGUF-Datei plus das `ModelProfile` aus Säule 1 (um Precision-Hints in die Graph-Knoten zu propagieren). Output ist ein `FusedGraph`, dessen Knoten zur Ausführung durch die Runtime (Säule 4) bereit sind.

**Datenstrukturen.**

```rust
pub struct ComputeGraph {
    pub nodes: Vec<OpNode>,
    pub edges: Vec<Edge>,
    pub inputs: Vec<NodeId>,   // Token-Embeddings, KV-Cache, usw.
    pub outputs: Vec<NodeId>,  // Logits
    pub model_profile: ModelProfile,
}

pub struct OpNode {
    pub id: NodeId,
    pub op: Operator,
    pub precision: PrecisionHint,
    pub layer_index: Option<usize>,
    pub input_shape: Vec<TensorShape>,
    pub output_shape: TensorShape,
}

pub enum Operator {
    // Basisoperationen
    RmsNorm { eps: f32 },
    Gemv { weight: WeightRef, bias: Option<WeightRef> },
    Gemm { weight: WeightRef, bias: Option<WeightRef> },
    Rope { theta_base: f32, variant: RopeVariant },
    FlashAttention { head_dim: u32, causal: bool },
    Softmax,
    SwiGLU,
    Add,         // Residual
    Mul,         // Element-wise
    Reshape,
    // Fusionierte Operationen (Output der Fusion-Passes)
    FusedNormGemv { norm_eps: f32, weight: WeightRef },
    FusedGateUpSwiglu { gate: WeightRef, up: WeightRef },
    FusedResidualNorm { norm_eps: f32 },
    FusedNormQkvRope { norm_eps: f32, qkv: WeightRef, rope: RopeParams },
    // ... weitere, von Fusion-Passes erzeugt
}

pub enum RopeVariant {
    Standard,
    NtkAware { alpha: f32 },
    YaRN { scaling_factor: f32 },
    Llama3 { factor: f32, low_freq: f32, high_freq: f32, orig_ctx: u32 },
}

pub struct WeightRef {
    pub tensor_name: String,
    pub quant_format: QuantFormatId,
    pub shape: TensorShape,
}

pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub operand_index: usize,
}

pub type NodeId = u32;
pub type TensorShape = Vec<usize>;

/// Output nach allen Fusion-Passes. Strukturgleich mit ComputeGraph,
/// aber mit reduzierter Knotenzahl und FusedOp-Varianten.
pub struct FusedGraph(pub ComputeGraph);
```

**Algorithmus.** Die Graph-Konstruktion liest die GGUF-Metadaten (Architektur-Typ, `n_layers`, `n_heads`, `n_kv_heads`, `head_dim`, RoPE-Parameter, ...) und verwendet einen kleinen Satz Architektur-Templates, um den initialen Graph aufzubauen. Templates decken die Standard-Transformer-Variationen ab: Llama-Style (RMSNorm, SwiGLU, RoPE), Mistral-Style (dto. + Sliding-Window-Attention), Qwen-Style (dto. + Q/K-Norm), und MoE-Varianten. Ein neues Modell wie Llama-4 wird in der Regel durch Parameter-Variation eines bestehenden Templates abgedeckt; nur bei fundamental neuen Architekturen wird ein neues Template ergänzt.

Anschließend laufen die Fusion-Passes in zwei Stufen:

1. **Regelbasierte Passes (deterministisch, schnell).** Pattern-Matching gegen den Graph:
   - `RmsNorm → Gemv → Rope` → `FusedNormQkvRope`, wenn das Gemv das QKV-Weight ist und keine anderen Konsumenten zwischen den Knoten liegen.
   - `Add (Residual) → RmsNorm` → `FusedResidualNorm`.
   - `Gemv(gate) + Gemv(up) → SwiGLU → Gemv(down)` → `FusedGateUpSwiglu` gefolgt von separatem FFN-Down.
   - Mehrere weitere Standard-Patterns.

2. **GA-Pass (optional, einmalig pro Modell-Architektur).** Die Fusion-GA (Abschnitt 4.3) sucht über den Raum gültiger Fusions-Kombinationen nach der global besten Variante. Input: regelbasiert-fusionierter Graph. Output: optimaler Fusions-Plan. Ergebnis wird gecached.

Nach beiden Stufen steht ein `FusedGraph` bereit, der typischerweise 40–60 % weniger Knoten hat als der initiale Graph.

### 2.4 Säule 3: Dequant IR

**Was sie tut.** Statt für jedes Quant-Format handgeschriebene Dequant-Kernel zu pflegen, beschreibt v1.0 jedes Format als **Dequant-Programm** in einer Zwischensprache (Intermediate Representation, IR). Ein einziger generischer WMMA-Kernel und ein einziger generischer GEMV-Kernel sind über diese IR parametrisiert; der Codegen übersetzt das Dequant-Programm zur Build-Zeit oder Erst-Lade-Zeit in nativen GPU-Code (RDNA 4 Assembly mit nativen FP8-WMMA-Instruktionen) und CPU-Code (AVX-512 Intrinsics).

**Primäres WMMA-Input-Format: FP8 (nicht FP16).** Eine zentrale Architektur-Entscheidung der v1.1-Revision: der Dequant-Pfad produziert als Default **FP8 (E4M3)** in LDS bzw. in den A/B-Fragmenten, nicht FP16. Das verdoppelt den WMMA-Durchsatz (RDNA 4 unterstützt FP8-WMMA mit doppelter Rate gegenüber FP16), halbiert den VGPR-Bedarf für A/B-Fragmente, und erlaubt höhere Occupancy. FP16 und BF16 bleiben als alternative Output-Targets des Dequant-Programms erhalten und werden vom Precision-GA selektiv für kritische Layer aktiviert.

**Warum sie nötig ist.** Die bisher größte Zeitsenke in v0.x war das Hinzufügen neuer Quant-Formate. Q6_K-WMMA in v0.3.0 hat 5 Tage gedauert, brauchte sieben neue Kernel-Dateien (WMMA + GEMV + fused-variants) und produzierte drei Numerik-Bugs, bevor es stabil lief. Das ist nicht nachhaltig. Die Lösung ist, die Dequantisierung von der GEMM/GEMV-Logik zu trennen: GEMM/GEMV ist ein generisches Problem, Dequantisierung ist ein format-spezifisches Detail. Der FP8-Pfad kommt als zusätzlicher Multiplikator obendrauf: ohne IR müsste *jeder* Format-Kernel sowohl einen FP16- als auch einen FP8-Pfad haben; mit IR ist FP8 ein Emitter-Flag.

**Schnittstelle.** Input ist eine `QuantFormat`-Definition (deklarativ, ~30 Zeilen Rust pro Format) plus ein Target (`Gfx1201` oder `Zen4Plus`). Output ist ein `CompiledKernel`: für GPU eine HIP-Kernel-Binary, für CPU eine JIT-kompilierte Funktion mit AVX-512-Intrinsics.

**Datenstrukturen.**

```rust
/// Deklarative Definition eines Quant-Formats.
/// Ein neues Format hinzufügen = eine Instanz dieser Struct füllen.
pub struct QuantFormat {
    pub id: QuantFormatId,
    pub name: &'static str,
    pub block_bytes: usize,            // z.B. 144 für Q4_K, 210 für Q6_K
    pub elements_per_block: usize,     // meist 256
    pub sub_blocks_per_block: usize,   // z.B. 8 für Q4_K, 16 für Q6_K
    pub sub_block_size: usize,         // elements_per_block / sub_blocks_per_block

    /// Das Dequant-Programm: eine Sequenz von DequantOps, die aus einem
    /// Block-Pointer einen Vector aus FP16/BF16-Werten erzeugt.
    pub dequant_program: Vec<DequantOp>,

    /// Wo im Block liegt der Block-Scale (d)?
    pub block_scale_offset: usize,
    pub block_scale_type: ScalarType,

    /// Wo liegt der Block-Min (dmin)? None für Formate ohne Min (Q4_0, Q6_K).
    pub block_min_offset: Option<usize>,
    pub block_min_type: Option<ScalarType>,

    /// Sub-Block-Scales: Layout und Entpackung.
    pub sub_scales_layout: SubScalesLayout,
}

pub enum DequantOp {
    /// Lade N Bytes aus dem Block bei Offset.
    LoadBytes { offset: usize, count: usize, reg: RegId },

    /// Lade FP8-Werte (E4M3 oder E5M2) direkt aus dem Block bei Offset.
    /// Nutzbar für Formate, die bereits FP8 speichern (Q8-FP8 Mixed, native FP8-Modelle).
    LoadFP8 { offset: usize, count: usize, variant: Fp8Variant, reg: RegId },

    /// Extrahiere Nibble (4-Bit) aus einem Byte-Register.
    /// high=true → upper 4 bits, high=false → lower 4 bits.
    ExtractNibble { src: RegId, high: bool, dst: RegId },

    /// 6-Bit-Rekonstruktion aus ql + qh (für Q6_K, Q5_K).
    /// dst = (ql[i] & lower_mask) | ((qh[i] >> shift) << 4)
    Combine6Bit { ql: RegId, qh: RegId, shift: u8, dst: RegId },

    /// Konvertiere Integer → Float, optional mit Offset (z.B. -32 für Q6_K).
    IntToFloat { src: RegId, offset: i32, dst: RegId },

    /// FP32-Multiplikation. Wird in FP32 durchgeführt, selbst wenn Input/Output FP16.
    MulF32 { a: RegId, b: RegId, dst: RegId },

    /// Fused Multiply-Add in FP32: dst = a * b + c
    FmaF32 { a: RegId, b: RegId, c: RegId, dst: RegId },

    /// Runtime-Downcast FP32 → FP16/BF16/FP8, je nach Precision-Hint des Layers.
    /// FP8 ist der Default-Pfad für WMMA-Inputs in v1.1.
    DowncastToHalf { src: RegId, dst: RegId, target: HalfType },

    /// Spezialisierter FP8-Downcast mit Saturation (E4M3: ±448, E5M2: ±57 344).
    /// Bei Overflow: clamp auf max_val, Flag setzen für Quality Monitor.
    DowncastToFP8 { src: RegId, dst: RegId, variant: Fp8Variant, saturate: bool },

    /// Schreibe dst in LDS (GPU) bzw. in SIMD-Register (CPU).
    StoreHalf { src: RegId, lds_offset_expr: Expr },

    /// FP8-spezifischer LDS-Store (halbiertes Byte-Budget ggü. FP16-Variante).
    StoreFP8 { src: RegId, variant: Fp8Variant, lds_offset_expr: Expr },

    // --- Erweiterungen aus dequant_ir_spec.md §2 ---

    /// Generische Bit-Extraktion: dst = (src >> shift) & mask.
    /// Verallgemeinert ExtractNibble und wird für Q4_K-Scale-Entpackung benötigt.
    ExtractBits { src: RegId, shift: u8, mask: u32, dst: RegId },

    /// Generische Bit-Rekombination: dst = lo | (hi << hi_shift).
    /// Verallgemeinert Combine6Bit und deckt Q5_K / Q4_K-Split-Scales ab.
    CombineBits { lo: RegId, hi: RegId, hi_shift: u8, dst: RegId },

    /// FP32-Subtraktion: dst = a - b. Nötig u. a. für Q4_K-Fma-mit-Subtraktion.
    SubF32 { a: RegId, b: RegId, dst: RegId },

    /// FP32-Addition: dst = a + b.
    AddF32 { a: RegId, b: RegId, dst: RegId },

    /// FP32-Negation: dst = -src.
    NegF32 { src: RegId, dst: RegId },

    /// Marker-Op: Beginn der Verarbeitung eines Sub-Blocks (Q6_K-Zentrierung,
    /// K-getreue Akkumulation). Semantik siehe dequant_ir_spec.md §2.8.
    ScaleBlockStart { sub_block_idx: u8 },

    /// Kompilierzeit-Konstante in ein Register schreiben (z. B. -32 für Q6_K-Offset).
    Const { value: f32, dst: RegId },
}

pub enum Fp8Variant {
    /// E4M3: 4 Exponent-Bits, 3 Mantissen-Bits. Range ±448, höhere Präzision,
    /// Standardwahl für Gewichte (kleiner Dynamikbereich).
    E4M3,
    /// E5M2: 5 Exponent-Bits, 2 Mantissen-Bits. Range ±57 344, größerer
    /// Dynamikbereich, geeignet für Aktivierungen und KV-Cache.
    E5M2,
}

pub enum SubScalesLayout {
    /// Int8-Array (einfachster Fall, z.B. Q6_K: scales[16]).
    Int8Array { offset: usize, count: usize },

    /// 6-Bit gepackt (z.B. Q4_K: 8 scales + 8 mins in 12 Bytes).
    /// Das Unpacking-Schema wird als zusätzliches DequantOp-Subprogramm angegeben.
    Packed6Bit {
        offset: usize,
        count: usize,
        unpack_program: Vec<DequantOp>,
    },

    /// FP16-Array (für nicht-K-Formate wie Q4_0: nur ein d pro Block, keine Sub-Scales).
    None,
}

pub enum ScalarType { Fp16, Bf16, Fp32, Int8, Fp8E4M3, Fp8E5M2 }
pub enum HalfType { Fp16, Bf16 }

pub type RegId = u32;
pub type Expr = String; // kleines Ausdrucks-Sublanguage, vom Codegen geparst
pub type QuantFormatId = u32;
```

**SSA-Invariante.** Jede `RegId` wird genau einmal als `dst` eines Ops produziert; der Typ eines `RegId` steht beim Producer fest und ändert sich nicht. Der Codegen mappt `RegId`s per Linear-Scan auf physische Register (VGPRs auf GPU, ZMM auf CPU). Die Lebensdauer endet beim letzten Consumer. Mehrfach-Assignments werden vom Validator abgelehnt.

```rust

/// Kompilierte Repräsentation — Binary und Metadaten.
pub struct CompiledKernel {
    pub target: KernelTarget,
    pub kind: KernelKind, // Gemm, Gemv, FusedNormGemv, ...
    pub quant_format_id: QuantFormatId,
    pub shape: KernelShape,
    pub binary: Vec<u8>,
    pub entry_symbol: String,
    pub constants: KernelConstants,
}

pub enum KernelTarget { Gfx1201, Zen4Plus }
pub enum KernelKind {
    Gemm,
    Gemv,
    FusedNormGemv,
    FusedGateUpSwiglu,
    FusedResidualNorm,
    FlashAttention,
    // ... weitere
}
```

**Kernel-Cache-Key-Layout und Cache-Invalidierungsstrategie sind in `dequant_ir_spec.md` §5.6 spezifiziert.**

**GEMV-Kernel-Varianten (`q8_inline`, `fuse_residual`, `fuse_norm`, `x_via_lds`) und ihre Invarianten sind in `dequant_ir_spec.md` §5.3 und §5.5 spezifiziert.**

**Algorithmus (Codegen).** Der Codegen ist ein regelbasiertes Übersetzungsmodul, kein Optimierer. Das Optimierungswissen steckt in der `QuantFormat`-Definition und im Precision-Hint des konsumierenden `OpNode`; der Codegen spuckt nur die entsprechenden Instruktionen aus.

```
fn generate_gpu_kernel(fmt: &QuantFormat, shape: KernelShape, target_dtype: HalfType)
    -> CompiledKernel:

    // 1. Prolog: LDS-Layout, Workgroup-Size, Register-Allocation
    //    LDS-Budget hängt von target_dtype ab:
    //    - FP8 (E4M3/E5M2): 1 Byte/Element → doppelte Tile-Size möglich
    //    - FP16/BF16:       2 Byte/Element
    emit_prolog(shape, target_dtype)

    // 2. Main-Loop-Header: K-Chunk-Iteration
    emit_k_loop_header(shape.k_chunk)

    // 3. Block-Load: lade einen Block in Register
    emit_load_block(fmt.block_bytes)

    // 4. Sub-Scales entpacken (nur einmal pro Block)
    emit_program(fmt.sub_scales_layout.unpack_program)

    // 5. Dequant-Programm für alle Elemente im Block
    //    Letzte Op ist StoreHalf bzw. StoreFP8, je nach target_dtype.
    für jedes Element im Block:
        emit_program(fmt.dequant_program)

    // 6. WMMA-Aufruf — instruktionsspezifisch je nach target_dtype:
    match target_dtype:
        Fp8E4M3 | Fp8E5M2 =>
            emit("v_wmma_f32_16x16x16_fp8_fp8 ...")  // 2× Durchsatz
        Fp16 =>
            emit("v_wmma_f32_16x16x16_f16 ...")
        Bf16 =>
            emit("v_wmma_f32_16x16x16_bf16 ...")

    // 7. K-Loop-Ende
    emit_k_loop_end()

    // 8. Output-Store
    emit_output_store()

    // 9. Assembler-Aufruf auf HIP-Toolchain (hipcc --offload-arch=gfx1201 -O3)
    return compile_hip(emitted_code, target = "gfx1201")

fn generate_cpu_kernel(fmt: &QuantFormat, shape: KernelShape) -> CompiledKernel:
    // Analog, aber mit AVX-512-Intrinsics:
    // LoadBytes → _mm512_loadu_si512
    // ExtractNibble → _mm512_and_si512 + _mm512_srli_epi16
    // MulF32 → _mm512_mul_ps
    // FmaF32 → _mm512_fmadd_ps
    // VNNI-Pfad für Q8 → _mm512_dpbusd_epi32
    // Zen4 führt 512-Bit-Ops via 2× 256-Bit-Slots aus — keine Takt-Reduktion.
    ...
```

Weil der Codegen regelbasiert ist und die `QuantFormat`-Definition deklarativ, ist das Hinzufügen eines neuen Formats eine Datenänderung, keine Codeänderung. Q5_K würde aussehen wie Q4_K mit einem zusätzlichen `qh` und einer `Combine5Bit`-Op — geschätzter Aufwand 30–60 Minuten, vs. 3–5 Tage in v0.x. Der FP8-Pfad wird automatisch mit generiert, ohne Zusatzaufwand pro Format.

### 2.5 Säule 4: Self-Tuning Runtime

**Was sie tut.** Die GA (Säule 4) liefert pro Shape eine kleine Menge **Kernel-Varianten** (typischerweise 3–5 Pareto-optimale Konfigurationen: die schnellste, die bandbreiten-effizienteste, eine LDS-arme Variante, eine Direct-Global-Variante, eine mit hohem K-Unroll). Die Self-Tuning Runtime entscheidet zur Laufzeit, welche dieser Varianten für den konkreten Call verwendet wird. Die Entscheidung wird durch einen Multi-Armed-Bandit-Algorithmus (UCB1) getroffen, der Exploration und Exploitation balanciert.

**Warum sie nötig ist.** In v0.x war die Dispatch-Entscheidung hardcodiert: "für Shape X nimm Kernel Y". Jedes Mal, wenn ein Profiling-Durchlauf neue Daten lieferte, mussten wir den Dispatcher manuell anpassen. Und: die optimale Variante hängt nicht nur von der Shape ab, sondern auch vom konkreten Runtime-Zustand (VRAM-Auslastung durch andere Prozesse, aktuelle GPU-Taktfrequenz, Thermal-Throttling). Ein statischer Dispatch kann das nicht berücksichtigen; ein Bandit-Algorithmus passt sich innerhalb der ersten ~100 Calls an den Zustand an.

**Warum Bandit statt GA hier.** GA ist für die Offline-Suche optimal: weitgehender Suchraum, Hunderte von Evaluierungen, Minuten-Budget. Bandit ist für Online-Auswahl optimal: kleiner Kandidatensatz (5 Varianten), jeder Call ist eine Evaluierung (Mikrosekunden), Entscheidungs-Budget ist Null. Man nimmt den GA, um den Kandidatensatz zu finden, und den Bandit, um zwischen den Kandidaten zur Laufzeit zu wählen.

**Schnittstelle.** Input pro Call sind Shape, Quant-Format und die Liste der verfügbaren Kernel-Varianten (aus dem GA-Cache). Output ist die ausgewählte Variante. Messung (tatsächliche Laufzeit) wird in den Bandit zurückgefüttert.

**Datenstrukturen.**

```rust
/// Eine vom GA gelieferte Kernel-Variante. Mehrere pro Shape möglich.
pub struct KernelVariant {
    pub id: VariantId,
    pub compiled: Arc<CompiledKernel>,
    pub ga_predicted_time_us: f32,  // GA-Schätzung aus Offline-Tuning
    pub description: &'static str,   // z.B. "LDS-based, K_UNROLL=4"
}

/// Eine Shape hat einen Satz Varianten und einen Bandit-Zustand.
pub struct ShapeBandit {
    pub shape_key: ShapeKey,
    pub variants: Vec<KernelVariant>,
    pub stats: Vec<BanditArmStats>,
    pub total_calls: u64,
}

pub struct BanditArmStats {
    pub variant_id: VariantId,
    pub call_count: u64,
    pub mean_time_us: f32,   // laufendes Mittel
    pub m2: f64,             // Welford-Online-Varianz
}

pub struct ShapeKey {
    pub m: u32, pub n: u32, pub k: u32,
    pub quant_format: QuantFormatId,
    pub kind: KernelKind,
}

pub type VariantId = u16;
```

**Algorithmus (UCB1).** Für jeden Call wählt UCB1 den Arm mit dem höchsten Score:

```
score(arm) = - mean_time(arm) + c × sqrt(ln(total_calls) / arm.call_count)
```

Minus vor `mean_time`, weil wir minimieren wollen. Der Explorations-Term sorgt dafür, dass wenig-genutzte Arme gelegentlich nochmal getestet werden. `c` ist ein Trade-off-Parameter (Standard `1.4`).

```
fn select_variant(bandit: &mut ShapeBandit) -> VariantId:
    # Erste Runde: alle Arme mindestens einmal
    für arm in bandit.variants:
        wenn arm.call_count == 0:
            return arm.variant_id

    # UCB1
    best = argmax_{arm} (-arm.mean_time_us
                        + 1.4 × sqrt(ln(bandit.total_calls) / arm.call_count))
    return best.variant_id

fn record_measurement(bandit: &mut ShapeBandit, variant: VariantId, time_us: f32):
    # Welford-Update für Online-Mittel und -Varianz
    arm = bandit.variants.find(variant)
    arm.call_count += 1
    delta = time_us - arm.mean_time_us
    arm.mean_time_us += delta / arm.call_count
    delta2 = time_us - arm.mean_time_us
    arm.m2 += delta × delta2
    bandit.total_calls += 1
```

Innerhalb von ~100 Calls konvergiert UCB1 zur besten Variante. In unseren Shape-Sets (typisch 20–40 unique Shapes pro Modell) bedeutet das ~2 000–4 000 Calls Gesamt-Exploration — bei 50 tok/s Decode sind das 40–80 Sekunden Warmup, danach ist die Runtime stabil optimal.

### 2.6 Säule 5: Quality Monitor

**Was sie tut.** Während der Inference sampelt der Quality Monitor periodisch Hidden-State-Statistiken (Mean, Max-Abs, Anteil NaN/Inf, **FP8-Saturation-Count**) an strategisch wichtigen Punkten im Graph: nach jeder Attention-Block, nach jeder FFN-Block, vor dem Logit-Head. Weicht eine Statistik signifikant vom erwarteten Bereich ab (bekannt aus dem ersten Calibration-Pass während Model-Introspection), wird ein **Precision-Fallback-Signal** an die Runtime gesendet.

**Standard-Eskalations-Pfad in v1.1:** FP8 → FP16 → BF16 → FP32. Für die überwältigende Mehrheit der Layer reicht FP8; bei detektierter Drift wird gezielt auf FP16 hochgestuft (statt wie in v1.0-Draft direkt FP32), was deutlich günstiger ist und in 80–90 % der Fälle ausreicht. FP32 bleibt der letzte Ausweg für extrem dynamikbereichs-kritische Layer (LM-Head, Embedding-Lookup für Special Tokens).

**Warum sie nötig ist.** Numerische Drift ist in quantisierten Modellen kein seltenes Problem, sondern ein graduelles. Sie tritt typischerweise bei langen Kontexten (KV-Cache-Akkumulation), bei bestimmten Input-Kombinationen (seltene Tokens, die unter Quantisierungsrauschen verschwinden) oder bei bestimmten Layer-Tensoren (Outlier-Werte, die FP8/FP16 überschreiten) auf. Der Quality Monitor erkennt diese Fälle **während** der Inference, nicht erst wenn der User beschwert, dass der Output Müll ist.

Dies ist die Laufzeit-Ergänzung zur Model Introspection: Säule 1 findet statische Präzisionsrisiken vor der Inference; Säule 5 findet dynamische Risiken während der Inference. Bei FP8 als Default ist ein laufender Monitor besonders wichtig — FP8-E4M3 saturiert bei ±448, was in einzelnen Outlier-Fällen eine reale Gefahr ist.

**Schnittstelle.** Input sind Hidden-State-Tensoren an definierten Graph-Knoten. Output ist entweder "alles OK, weitermachen" oder ein `PrecisionRevisionSignal`, das die Runtime anweist, einen Layer auf höhere Precision umzustellen und den Inference-Schritt zu wiederholen.

**Datenstrukturen.**

```rust
pub struct QualityMonitor {
    pub expected_ranges: HashMap<NodeId, ExpectedRange>,
    pub tolerance_factor: f32,   // z.B. 3.0 = Drift bis 3σ erlaubt
    pub sample_rate: u32,        // Check alle N Tokens; N=1 teuer, N=32 üblich
    pub revision_log: Vec<RevisionEvent>,
}

pub struct ExpectedRange {
    pub mean_abs_expected: f32,
    pub mean_abs_stddev: f32,
    pub max_abs_expected: f32,
    pub nan_tolerance: u32,       // normalerweise 0
}

pub struct PrecisionRevisionSignal {
    pub affected_node: NodeId,
    pub current_precision: PrecisionHint,
    pub recommended_precision: PrecisionHint,
    pub reason: DriftReason,
}

pub enum DriftReason {
    MeanAbsExceedsStdTolerance { observed: f32, expected: f32, z_score: f32 },
    MaxAbsExceedsThreshold { observed: f32, threshold: f32 },
    NaNDetected { count: u32 },
    InfDetected { count: u32 },
    /// FP8-spezifisch: Anzahl saturierter Werte im Output überschreitet Toleranz.
    /// E4M3 saturiert bei ±448; ein einzelner Saturation-Event ist OK, aber
    /// > 0.1 % der Elemente saturiert → Precision-Hochstufung auf FP16 nötig.
    Fp8SaturationExceeded { count: u32, total: u32, fraction: f32 },
}

pub struct RevisionEvent {
    pub token_index: u64,
    pub node_id: NodeId,
    pub signal: PrecisionRevisionSignal,
    pub resolved: bool,
}
```

**Algorithmus.**

```
fn check(monitor: &QualityMonitor, node: NodeId, hidden_state: &Tensor)
    -> Option<PrecisionRevisionSignal>:

    range = monitor.expected_ranges[node]
    mean_abs = tensor_mean_abs(hidden_state)
    max_abs = tensor_max_abs(hidden_state)
    nan_count = tensor_count_nan(hidden_state)

    # Schnellprüfung: NaN/Inf ist immer kritisch
    wenn nan_count > range.nan_tolerance:
        return Some(Signal { reason: NaNDetected { count: nan_count }, ... })

    # Statistische Drift
    z = (mean_abs - range.mean_abs_expected) / range.mean_abs_stddev
    wenn |z| > monitor.tolerance_factor:
        return Some(Signal { reason: MeanAbsExceedsStdTolerance { z_score: z }, ... })

    # Max-Abs-Threshold (FP16-Overflow-Guard)
    wenn max_abs > 60000.0 and current_precision == Fp16:
        return Some(Signal { reason: MaxAbsExceedsThreshold, ... })

    return None
```

Die Kalibrierungsdaten (`expected_ranges`) werden beim ersten Inference-Durchlauf mit einem kurzen Prompt ("The quick brown fox jumps over the lazy dog") aufgenommen — eine Sekunde Warmup, einmalig pro Modell. Für Robustheit gegen input-abhängige Varianz werden die Toleranzen konservativ gesetzt (3σ statt 2σ).

**Implementierung als Fused-Epilog (nicht als separater Kernel).** Der Quality-Check läuft als **Epilog innerhalb des letzten Kernels jedes Layer-Blocks**, nicht als eigener Kernel-Launch. Das eliminiert den Dispatch-Overhead eines separaten Kernels (~5 μs × 48 Layer = 240 μs pro Token bei 14B) und vermeidet Race-Conditions bei Multi-Stream-Ansätzen. Der Epilog umfasst ca. 20 GPU-Instruktionen (Mean-Abs-Reduktion, Max-Abs-Reduktion, NaN-Scan, FP8-Saturation-Count) und schreibt bei Drift-Detektion ein Dirty-Flag in Pinned Host Memory. Die CPU pollt dieses Flag am Token-Ende — nach dem ohnehin nötigen `hipStreamSynchronize` für die Logits (siehe Section 3.7).

**Revisions-Fallback.** Wenn ein Signal ausgelöst wird, markiert die Runtime den betroffenen Knoten für die nächste Precision-Stufe (FP8 → FP16 → BF16 → FP32) und wiederholt den Inference-Schritt ab dem betroffenen Layer. Der Overhead ist gering, weil Revision selten auftritt (im Normalbetrieb 0 Mal pro 1000 Tokens). Ohne Quality Monitor würde der User stattdessen Kauderwelsch bekommen, wie im v0.x-Llama-3.1-Fall.


### 2.7 Säule 6: Safety & Debug (VALU-Parity)

**Was sie tut.** Säule 6 ist eine **orthogonale Validierungsschicht**, die nicht Teil des linearen Datenflusses ist, sondern parallel zu den WMMA-basierten Kerneln einen unabhängigen Referenzpfad bereitstellt: den **VALU-Parity-Pfad**. Für jede WMMA-Operation kann optional die äquivalente Operation über die skalaren Vector-ALUs (VALU) berechnet und die Ergebnisse bit-nahe verglichen werden. Stimmen sie nicht innerhalb einer definierten Toleranz überein, wird ein Safety-Event geloggt und die Runtime kann den WMMA-Pfad für den betroffenen Kernel deaktivieren.

**Warum sie nötig ist.** WMMA auf `gfx1201` ist Hardware, die zum Zeitpunkt der v1.0-Entwicklung jung ist. Silicon-Bugs, Driver-Regressions und Toolchain-Fehler sind nicht hypothetisch — die ROCm-Historie kennt mehrere Fälle (`gfx1100` Wave32/Wave64-Mismatch, Illegal-Opcode-Crashes in RDNA 3, Composable-Kernel-Inkompatibilitäten). Ein eigenständiger Referenzpfad erlaubt uns:

1. **Silicon-Probleme zu detektieren, bevor sie als Modell-Qualitätsprobleme erscheinen.** Wenn FP8-WMMA auf einer bestimmten Tile-Größe systematisch um 1 ULP abweicht, sehen wir das im Log, nicht erst in der Perplexity-Regression.
2. **Regressionen bei ROCm-Updates sofort zu erkennen.** Bei jedem Treiber-Update läuft ein kurzer Parity-Check-Durchlauf; Abweichungen gegenüber dem bisherigen Verhalten sind ein klares Warnsignal.
3. **Die GA-Kandidaten hart zu validieren.** Ein GA-generierter Kernel, der schnell aber subtil falsch ist, darf nicht in die Pareto-Front gelangen — Parity-Check ist Pflichtkriterium.
4. **Debugging-Support bei neuen Quant-Formaten.** Wenn ein neu hinzugefügtes Q5_K-Format falsch dequantisiert, zeigt der Parity-Pfad, ob das Problem im Dequant-Programm (VALU-Pfad ebenfalls falsch) oder im WMMA-Pfad liegt (VALU korrekt, WMMA abweichend) — eine sonst extrem schwer zu lokalisierende Bug-Klasse.

**Schnittstelle.** Input ist eine WMMA-Operation mit konkreten Input-Tensoren. Output ist entweder "OK" oder ein `ParityViolation`-Event mit Details.

**Datenstrukturen.**

```rust
pub struct ValuParityChecker {
    /// Pro Shape + Quant-Format: letzte N Parity-Checks (Ring-Buffer).
    pub history: HashMap<ShapeKey, VecDeque<ParityCheckResult>>,
    /// Sampling-Rate: wie viele Kernel-Aufrufe werden mit VALU-Pfad gespiegelt?
    /// Default: 1/1000 im Produktion-Modus, 1/1 im Debug-Modus.
    pub sample_rate: ParitySampleRate,
    /// Toleranz für FP-Vergleich. Strikte Bit-Identität ist wegen FMA-
    /// Reihenfolge und FP8-Rounding nicht garantiert; typ. 2⁻¹⁰ relativ.
    pub tolerance_rel: f32,
    pub tolerance_abs: f32,
    /// Aktionen bei Violation: nur loggen, Kernel disablen, oder crash.
    pub on_violation: ViolationAction,
}

pub enum ParitySampleRate {
    Never,                     // Produktion, maximale Performance
    Every(u32),                // 1/N Calls; Default 1/1000
    Always,                    // Debug-Modus, ~2× Laufzeit
    OnCompile,                 // nur einmal pro neu kompiliertem Kernel
}

pub enum ViolationAction {
    LogOnly,
    DisableWmmaFor { shape: ShapeKey, fallback: KernelVariant },
    AbortInference,
}

pub struct ParityCheckResult {
    pub timestamp: Instant,
    pub shape: ShapeKey,
    pub wmma_time_us: f32,
    pub valu_time_us: f32,
    pub max_rel_error: f32,
    pub max_abs_error: f32,
    pub violated: bool,
    pub element_count_violations: u32,
}

pub struct ParityViolation {
    pub result: ParityCheckResult,
    pub first_violation_index: usize,
    pub wmma_value: f32,
    pub valu_value: f32,
}
```

**Algorithmus.**

```
fn check_parity(
    checker: &mut ValuParityChecker,
    wmma_kernel: &CompiledKernel,
    inputs: &[Tensor],
    wmma_output: &Tensor,
) -> Option<ParityViolation>:

    if not should_sample(checker.sample_rate): return None

    // VALU-Pfad: dieselbe Operation ohne WMMA-Intrinsics, nur v_fma_f32 etc.
    // Ist 5–10× langsamer, aber deterministisch korrekt.
    let valu_output = run_valu_reference(wmma_kernel, inputs)

    let (max_rel, max_abs, violations) = compare(
        wmma_output, valu_output,
        checker.tolerance_rel, checker.tolerance_abs,
    )

    let result = ParityCheckResult { ... }
    checker.history[shape].push_back(result)

    if violations > 0:
        return Some(ParityViolation {
            result,
            first_violation_index: ...,
            wmma_value: wmma_output[idx],
            valu_value: valu_output[idx],
        })

    return None
```

**VALU-Referenz-Kernel-Generierung.** Der Codegen (Säule 3) erzeugt bei jeder WMMA-Kernel-Kompilierung automatisch einen **Zwilling**: denselben Kernel, aber mit `v_fma_f32` statt `v_wmma_*`. Der VALU-Kernel läuft 5–10× langsamer, wird aber deterministisch ausgeführt und dient als Goldstandard. Beide landen im `CompiledKernel`-Objekt; die Runtime kann je nach Sampling-Rate zwischen ihnen wechseln.

**Speicher- und Laufzeit-Kosten.** Zweites Output-Buffer (~wenige MB VRAM pro Shape). CPU-Time für Vergleich vernachlässigbar. Laufzeit-Overhead: bei `Every(1000)` rund 0,5–1 % zusätzliche GPU-Zeit; bei `Always` rund 2×. Default in Produktion ist `Every(1000)` — quasi kostenlos, erkennt aber systematische Probleme in wenigen Sekunden Betrieb.

**Korrektheits-Validierung zur Kernel-GA.** Ein Kernel-Kandidat aus der GA (Abschnitt 4.2) wird nur in die Pareto-Front aufgenommen, wenn sein Parity-Check-Durchlauf (fest `Always` während GA-Evaluierung) keine Violations zeigt. Das verhindert, dass ein schnell-aber-falsch-Kandidat die Laufzeit kontaminiert. Der VALU-Pfad ist damit integraler Teil der Fitness-Funktion.


---

## 3. Hardware-Backend-Spezifikation

### 3.1 GPU-Backend: gfx1201 (RDNA 4)

Die RX 9070 XT ist eine RDNA-4-Karte mit **64 Compute Units (CUs)** und **128 AI Accelerators** (2 pro CU), 16 GB GDDR6-VRAM und einer Peak-Bandbreite von ~644 GB/s. Sie unterstützt native WMMA-Instruktionen (Wave Matrix Multiply Accumulate), die 16×16×16-Matrix-Multiplikationen in wenigen Takten durchführen — **inklusive nativer FP8-Pfade (E4M3 und E5M2)** mit doppeltem Durchsatz gegenüber FP16. Das ist der zentrale Performance-Hebel von v1.0.

**Fixe Hardware-Constraints** (wird in `KernelGenome`-Validierung fest verdrahtet, keine Laufzeit-Erkennung):

```rust
pub const GFX1201: Gfx1201Constraints = Gfx1201Constraints {
    compute_units: 64,              // Vollausbau RX 9070 XT
    ai_accelerators: 128,           // 2 pro CU, WMMA-fähig
    wave_size: 32,                  // RDNA-native, kein Wave64
    vgpr_occupancy_target: 104,     // bewusste Limitierung, NICHT Hardware-Max
    vgpr_hardware_max: 256,         // nur in Notfall-Kerneln
    lds_per_workgroup: 65536,       // 64 KB, fix
    l1_cache_per_cu: 32768,         // 32 KB, fix
    l2_cache_total: 4194304,        // 4 MB, fix
    memory_bw_mbps: 644_000,        // ~644 GB/s
    wmma_tile: (16, 16, 16),        // M, N, K
    wmma_types: &[
        WmmaType { a: Fp16,    b: Fp16,    c: Fp32, throughput_relative: 1.0 },
        WmmaType { a: Bf16,    b: Bf16,    c: Fp32, throughput_relative: 1.0 },
        WmmaType { a: Fp8E4M3, b: Fp8E4M3, c: Fp32, throughput_relative: 2.0 },
        WmmaType { a: Fp8E5M2, b: Fp8E5M2, c: Fp32, throughput_relative: 2.0 },
    ],
    // Dual Issue: 2 WMMA pro Takt pro CU, wenn Register sauber partitioniert
    wmma_per_cu_per_cycle: 2,
};
```

**WMMA-Register-Layout.** Das Register-Layout für `gfx1201` WMMA wurde in v0.x für den FP16-Pfad in `docs/wmma_register_layout_gfx12.md` byte-genau dokumentiert (wird nach v1.0 migriert). Entscheidende Punkte für FP16/BF16/FP8:

- **A-Matrix (FP16/BF16):** 16×16 Werte, verteilt auf 8 VGPRs pro Lane (Wave32 × 8 = 256 Halfs = 16×16).
- **A-Matrix (FP8):** 16×16 Werte, verteilt auf **4 VGPRs pro Lane** (Wave32 × 4 = 128 Bytes = 16×16 Bytes, bei 1 Byte/FP8). **Halbierte VGPR-Belegung** ist der zweite Performance-Hebel.
- **B-Matrix:** analog zu A, mit transponierter Lane-Zuordnung.
- **C/D-Akkumulator:** 16×16 FP32-Werte, verteilt auf 8 VGPRs pro Lane — identisch für alle Input-Formate.
- **Nutzung:** Intrinsics `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, `..._bf16_w32`, `..._fp8_fp8_w32`, `..._bf8_bf8_w32` (letztere = E5M2).

**LDS-Layout.** 64 KB pro Workgroup reichen für Tile-Größen bis 128×128 FP16 A-Kachel + 128×128 FP16 B-Kachel + Padding. **Bei FP8 verdoppelt sich die effektive Tile-Size** auf 256×128 oder 128×256 — ein zusätzlicher Fusion-Spielraum. Für Decode (GEMV) nutzen typische Konfigurationen nur einen Bruchteil des LDS; für Prefill (GEMM) kann LDS der Bottleneck sein, was in v0.x überraschenderweise zu der Erkenntnis führte, dass Direct-Global-Loads für einige Shapes schneller sind als LDS-Staging.

**WMMA-Intrinsics in HIP-Code (FP8-Pfad):**

```cpp
// Beispiel-Usage in einem generierten Kernel mit FP8-Input:
using fp8x16_t  __attribute__((ext_vector_type(16))) = unsigned char;
using float32x8_t __attribute__((ext_vector_type(8))) = float;

fp8x16_t   a_frag, b_frag;    // 16 FP8 per Lane × 32 Lanes = 16×32 Bytes = 16×16
float32x8_t c_frag = {0};

// ... A und B aus LDS oder Direct-Global in a_frag/b_frag laden ...

// FP8 E4M3 × E4M3 → FP32 Akkumulator, 2× Durchsatz vs. FP16-WMMA
c_frag = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32(a_frag, b_frag, c_frag);

// ... c_frag nach Global Memory schreiben ...
```

Der Codegen (Säule 3) generiert je nach `target_dtype` die entsprechende Intrinsic-Variante und übergibt sie dem System-HIP-Compiler (`hipcc` mit Target `gfx1201`).

### 3.2 Register-Pressure-Management (VGPR-Budget 104)

Eine der wichtigsten Design-Entscheidungen auf `gfx1201` ist die **bewusste Beschränkung auf 104 VGPRs pro Wave**, obwohl die Hardware bis zu 256 erlaubt. Diese Sektion erklärt warum, und wie die Entscheidung trotz 64 CUs aufrechterhalten wird.

**Der Trade-off zwischen VGPRs pro Wave und Waves pro CU.** Jede CU hat einen festen Pool von Registern (1 536 VGPRs auf RDNA 4). Dieser Pool wird statisch zwischen den aktiven Waves partitioniert. Mehr VGPRs pro Wave bedeuten:

| VGPRs/Wave | Max Waves/CU | Nutzbar für Latency-Hiding |
|---|---|---|
| 64 | 24 | extrem gut, aber schlechte pro-Wave-Performance |
| 104 | ~15 | **Sweet-Spot**, genug Waves um Memory-Latenz zu verstecken |
| 128 | 12 | akzeptabel für Compute-gebundene Kernel |
| 256 | 6 | nur für spezielle Kernel, wo Waves keine Latenz haben |

Mit 104 VGPRs pro Wave und 64 CUs haben wir theoretisch `15 × 64 = 960` parallele Waves auf der GPU — genug, um Memory-Latenzen bei praktisch jedem Shape zu verstecken. Mit 256 VGPRs wären es nur `6 × 64 = 384` Waves, was bei Decode-Workloads (hohe Memory-Latenz) zu Auslastungs-Problemen führt.

**Warum die Verdopplung auf 64 CUs das Budget nicht lockert.** Eine intuitive Annahme wäre: "mehr CUs = mehr Parallelität = können es uns leisten, pro Wave mehr Register zu verbrauchen". Das ist falsch. Die VGPR-Budget-Entscheidung ist **pro CU**, nicht pro GPU. Die Register werden nicht über CUs geteilt. 64 CUs × 15 Waves ist besser als 64 CUs × 6 Waves — die Skalierung der CUs verstärkt das Argument für kleine Waves, sie schwächt es nicht.

**FP8 macht das Budget leichter einzuhalten.** FP8-WMMA belegt 4 VGPRs pro Fragment statt 8 (wie FP16). Dadurch schrumpft der VGPR-Bedarf eines Kernels, der vorher z. B. 112 VGPRs brauchte, auf 96 — innerhalb des 104-Budgets, ohne Spill oder Re-Allocation. Das ist ein weiterer Grund, warum FP8 in v1.1 der Default-Pfad ist.

**Enforcement im Kernel-GA.** Die `validate()`-Funktion des Kernel-Genome (Abschnitt 4.2) schätzt den VGPR-Bedarf vor der Kompilierung und verwirft Kandidaten, die über 104 gehen. Kandidaten, die zwischen 104 und 128 VGPRs brauchen, werden als **"Compute-Only"-Variante** markiert und nur für Prefill (hoch compute-bound, niedrig memory-bound) in die Pareto-Front aufgenommen. Die Runtime selektiert über den Bandit zwischen "Standard" (104 VGPRs, gut für Decode) und "Compute-Only" (128 VGPRs, gut für Prefill-Batches) je nach Workload.

**Praktische Implikation.** Fused-Kernel werden im Raum der 104-VGPR-Budgets designed. Ein großer Fused-Kernel wie `FusedNormQkvRopeAttention` (theoretisch attraktiv) wird in der Regel das Budget sprengen und ist daher nicht Teil der Standard-Fusions-Kandidaten — er landet als optionaler Kandidat im Fusion-GA (Abschnitt 4.4), wird aber nur als "Compute-Only" akzeptiert, falls er das Pareto-Front-Kriterium schafft.

### 3.3 CPU-Backend: Zen4 (AVX-512 Double-Pumped)

Der Ryzen 9 7945HX hat 16 physische Zen4-Kerne mit SMT (32 Threads), 64 MB L3-Cache und volle AVX-512-Unterstützung inklusive VNNI (Vector Neural Network Instructions) und BF16-Dot-Products. Zen4 implementiert AVX-512 **Double-Pumped**: 512-Bit-Operationen laufen über zwei 256-Bit-Execution-Ports in zwei aufeinanderfolgenden Takten. Entscheidend ist der Unterschied zu einigen Intel-AVX-512-Implementierungen der Skylake-X/Ice-Lake-Generation: **Zen4 reduziert den CPU-Takt bei 512-Bit-Last nicht**. AVX-512-Code läuft auf Zen4 mit voller Basis-/Boost-Frequenz weiter, was ein signifikanter Performance-Vorteil gegenüber diesen Intel-Chips ist.

**Fixe CPU-Constraints:**

```rust
pub const ZEN4: Zen4Constraints = Zen4Constraints {
    simd_width_bits: 512,           // AVX-512, Double-Pumped via 2× 256b
    simd_fp32_per_vec: 16,          // 512 / 32
    simd_fp16_per_vec: 32,          // 512 / 16 (AVX-512 FP16)
    simd_bf16_per_vec: 32,
    simd_int8_per_vec: 64,
    cache_line: 64,
    l1d_per_core: 32768,            // 32 KB
    l2_per_core: 1048576,           // 1 MB
    l3_shared_mb: 64,               // 7945HX spezifisch
    vnni_int8_dot: true,            // _mm512_dpbusd_epi32
    vnni_bf16_dot: true,            // _mm512_dpbf16_ps
    physical_cores: 16,
    logical_threads: 32,
    avx512_downclock: false,        // Zen4-spezifisch: kein Throttling
};
```

**AVX-512-GEMV-Strategie.** Für Decode (Batch-Size 1, Shape M×K × K×1 = M×1) nutzt das CPU-Backend einen breiten Reduktions-Kernel:

```cpp
// Konzept: 1 Output-Lane braucht K FMAs. 16 Output-Lanes parallel (eine AVX-512-Vec).
// Gewichtsmatrix wird in 16-Spalten-Blöcken durchlaufen.
__m512 acc[4] = { _mm512_setzero_ps() };  // 4 Akkumulatoren für ILP

for (int k = 0; k < K; k += 16) {
    __m512 x = _mm512_loadu_ps(input + k);          // 16 Input-Werte
    // Dequant-IR generiert hier die Dequant-Instruktionen für das Format:
    __m512 w0 = dequant_16_lanes(weights + ..., k + 0*N);
    __m512 w1 = dequant_16_lanes(weights + ..., k + 1*N);
    // ... usw., 4-fach unrolled
    acc[0] = _mm512_fmadd_ps(x, w0, acc[0]);
    acc[1] = _mm512_fmadd_ps(x, w1, acc[1]);
    // ... usw.
}
// Horizontale Reduktion der 4 Akkumulatoren
```

**VNNI-Pfad für Q8-Dot-Products.** Wenn Input in Q8 quantisiert vorliegt (Inline-Quantisierung der Activations), nutzt der Kernel `_mm512_dpbusd_epi32` für Int8×Int8 → Int32-Dot-Products mit 64 MACs pro Instruktion. Das ist bei Q8-Gewichten und Q8-Aktivierungen rund 3× schneller als der FP-Pfad. Zen4's Double-Pumped-Ausführung bedeutet hier: 2 Takte pro 512-Bit-VNNI-Instruktion, **ohne** Taktreduktion des Kerns — der arithmetische Durchsatz ist real, nicht nur nominal.

**FP8 auf CPU.** Zen4 hat keine dedizierten FP8-Instruktionen (das ist ein RDNA-4-Feature). Für den CPU-Pfad wird FP8-Dequant auf FP16 erweitert, bevor die FMAs laufen. Das ist akzeptabel, weil das CPU-Backend primär für GA-Läufe und Fallback genutzt wird, nicht als primärer Inference-Pfad.

**Thread-Modell.** GEMV-Operationen werden über die Output-Dimension parallelisiert: 16 Threads × 4096 Output-Lanes = 256 Lanes pro Thread. Die KV-Cache-Aktualisierung läuft thread-lokal (kein Lock nötig, weil jeder Thread einen disjunkten Slice aktualisiert). Für GA-Läufe wird der Thread-Pool hingegen sequentiell genutzt (ein GA-Evaluation zur Zeit), um stabile Messungen zu bekommen.

### 3.4 Codegen für beide Targets

Der Dequant-IR-Codegen (Säule 3) hat zwei Emitter-Backends:

**GPU-Emitter** erzeugt HIP-Code und ruft `hipcc --offload-arch=gfx1201 -O3 -shared -fPIC` auf. Ergebnis ist eine `.co`-Datei (Code Object), die zur Laufzeit über `hipModuleLoad` in die Runtime geladen wird. Für Meilenstein 0 wird zuerst verifiziert, dass `hipcc` die FP8-WMMA-Intrinsics für `gfx1201` überhaupt kompiliert (siehe Abschnitt 9.0).

**CPU-Emitter** erzeugt C mit AVX-512-Intrinsics und kompiliert über das systemeigene Clang/GCC (einmalig beim Build). Für v1.0 Phase 1 ist AOT-Kompilierung ausreichend; JIT wird in Phase 3 ergänzt, falls Bedarf besteht.

### 3.5 Kein Abstraktionslayer

Ein zentrales Design-Prinzip: **zwischen Kernel-Code und Hardware liegt keine Abstraktions-Schicht**. Kein HAL (Hardware Abstraction Layer), keine generische "CUDA-or-HIP"-Wrapper, keine Runtime-Architektur-Erkennung. Die Kernel wissen, dass sie auf `gfx1201` bzw. Zen4 laufen, und nutzen Hardware-spezifische Intrinsics direkt.

Das ist der strukturelle Unterschied zu llama.cpp: dort müssen Kernel auf einer gemeinsamen Abstraktion über alle unterstützten Hardware laufen, was den kleinsten gemeinsamen Nenner zum Ziel-Set macht. v1.0 hat keinen kleinsten gemeinsamen Nenner — nur genau eine GPU und eine CPU-Familie.

### 3.6 VRAM-Arena 2.0 (Static Tight-Fit Allocation)

14B-Modelle (~9,0 GB Gewichte in Q4_K_M) sättigen die 16 GB VRAM fast vollständig. Klassisches dynamisches Memory-Management (viele `hipMalloc`/`hipFree`-Aufrufe) führt auf AMD-GPUs zu Fragmentierung — ein in v0.x beobachtetes Problem, bei dem nach mehreren Modell-Lade/Entlade-Zyklen weniger zusammenhängende Blöcke verfügbar waren, obwohl der freie VRAM nominal ausreichte.

v1.0 eliminiert dieses Problem durch eine **monolithische Arena-Allokation**: ein einziger `hipMalloc`-Aufruf beim Start, danach ausschließlich Offset-Arithmetik innerhalb dieses Blocks.

**Arena-Zonierung:**

```rust
/// Die Arena wird beim Programmstart allokiert und lebt bis zum Exit.
/// Alle GPU-Daten leben innerhalb dieses einen Blocks.
pub struct VramArena {
    base_ptr: *mut u8,         // hipMalloc-Ergebnis
    total_bytes: usize,        // z.B. 14,5 GB (freier VRAM - 500 MB Sicherheitsreserve)
    layout: ArenaLayout,
}

pub struct ArenaLayout {
    // Zone A: Gewichte (Read-Only nach dem Laden)
    pub weights: ArenaZone,    // Offset 0, Größe = GGUF-Gewichte
    // Zone B: KV-Cache (dynamisch wachsend, max beim Start berechnet)
    pub kv_cache: ArenaZone,   // Offset = nach Zone A, Größe = Rest - Zone C
    // Zone C: Scratchpad (rotierend, Ping-Pong)
    pub scratchpad: ArenaZone, // Offset = Ende - Scratchpad-Size, fix
}

pub struct ArenaZone {
    pub offset: usize,
    pub size: usize,
}
```

**VRAM-Budget-Berechnung beim Start:**

```
fn compute_arena_layout(model: &GgufModel, free_vram: usize) -> ArenaLayout:
    let weights_size = model.total_weight_bytes()           // 8B: ~4,5 GB; 14B: ~9,0 GB
    let scratchpad_size = compute_scratchpad_size(model)     // typ. 2–50 MB
    let overhead = 100_MB                                     // QM-Checkpoints, Parity, Metadata
    let safety_reserve = 500_MB                               // für Treiber + Desktop-Compositor

    let available = free_vram - safety_reserve
    let kv_budget = available - weights_size - scratchpad_size - overhead
    let max_context = kv_budget / model.kv_bytes_per_token()  // → max Kontext-Tokens

    log::info!("Max context: {} tokens (KV-Cache: {} MB, FP8-E5M2)",
               max_context, kv_budget / 1_000_000);

    ArenaLayout {
        weights:    ArenaZone { offset: 0, size: weights_size },
        kv_cache:   ArenaZone { offset: weights_size, size: kv_budget },
        scratchpad: ArenaZone { offset: weights_size + kv_budget, size: scratchpad_size },
    }
```

**Beispiel-Budget 14B Q4_K_M, 16 GB VRAM, Desktop läuft (~1,5 GB Compositor):**

```
Freier VRAM:         14,5 GB
- Safety-Reserve:     0,5 GB
= Verfügbar:         14,0 GB
- Gewichte (Zone A):  9,0 GB
- Scratchpad (Zone C): 0,05 GB  (50 MB, Ping-Pong + Attention-Scores)
- Overhead:           0,1 GB
= KV-Cache (Zone B):  4,85 GB
÷ 96 KB/Token (14B FP8-E5M2)
= Max Kontext:       ~51 700 Tokens ≈ 50k

→ 14B mit 50k-Kontext auf 16 GB VRAM. Komfortabel.
→ Bei FP16-KV-Cache (192 KB/Token): max ~25k. Immer noch nutzbar.
```

**Scratchpad-Ping-Pong (Zone C).** Der Scratchpad ist in zwei gleich große Hälften aufgeteilt. Layer N schreibt sein Ergebnis in Hälfte A; Layer N+1 liest aus Hälfte A und schreibt in Hälfte B; Layer N+2 liest aus B und schreibt in A — die Aktivierungen von Layer N-2 werden dabei automatisch überschrieben. Kein Freigabe-Aufruf nötig, kein Lifecycle-Tracking, kein Leak möglich.

```rust
impl Scratchpad {
    /// Gibt den Buffer für Layer i zurück. Alterniert automatisch.
    pub fn buffer_for_layer(&self, layer_idx: usize) -> GpuSlice {
        let half = layer_idx % 2;
        let offset = self.zone.offset + half * self.half_size;
        GpuSlice { ptr: self.base.offset(offset), size: self.half_size }
    }
}
```

**Rust-Ownership-Garantien.** Die `VramArena` wird als `Owned`-Struct auf dem Rust-Heap gehalten. `Drop` ruft `hipFree` auf. Zonen werden als `GpuSlice`-Borrows ausgegeben, die den Borrow-Checker durchlaufen:

- Gewichte: `&GpuSlice` (immutable, read-only nach dem Laden)
- KV-Cache Write: `&mut GpuSlice` (exklusiv, Compile-Time-Check gegen doppelte Writes)
- Scratchpad: `&mut GpuSlice` (exklusiv pro Layer, Ping-Pong über Index)

Das eliminiert die Klasse von Bugs, bei der ein Layer versehentlich in den Speicher eines anderen Layers schreibt — ein Fehler, der in C++ erst zur Laufzeit als Daten-Korruption sichtbar wird und Tage zum Debuggen braucht.

**Null Laufzeit-Overhead.** Alle Offsets werden beim Graph-Compilation (vor der ersten Inference) zu fixen `usize`-Werten aufgelöst. Zur Inference-Laufzeit gibt es keine Allokation, keine Pointer-Arithmetik-Berechnung, kein Locking — nur vorkompilierte Offsets, die als Kernel-Argumente übergeben werden.

### 3.7 Zero-Sync Pipeline (Single-Stream Dispatch)

Das Runtime-Modell von v1.0 ist konsequent auf **einen einzigen HIP-Stream pro Token** ausgelegt. Alle Kernel-Launches eines Decode-Steps landen sequentiell in diesem Stream; die GPU führt sie in Reihenfolge aus; die CPU synchronisiert genau einmal: am Token-Ende, um die Logits zu lesen.

**Warum Single-Stream statt Multi-Stream:**

Multi-Stream erlaubt Overlap von Compute und Memory-Transfer — aber bei LLM-Decode (Batch=1) gibt es praktisch nichts zu überlappen. Die Gewichte liegen in VRAM (kein Host→Device-Transfer), der KV-Cache-Write ist winzig (96 KB pro Token bei 14B FP8), und Layer N+1 braucht das Ergebnis von Layer N (strikt sequentiell). Multi-Stream würde hipEvent-Synchronisation, Race-Condition-Risiken bei geteilten Buffern und schwer debugbaren nicht-deterministischen Ausführungsreihenfolgen einführen — für einen Gewinn von ~0 % bei Decode und ~1 % bei Prefill.

**2-Layer-Lookahead Dispatch:**

Die CPU schiebt Kernel-Kommandos **2 Layer voraus** in die GPU-Queue. Das eliminiert "GPU starvation" — die GPU muss nie auf die CPU warten, weil der nächste Kernel bereits in der Queue liegt.

```
CPU:  [Submit L0] [Submit L1] [Submit L2] [Submit L3] ... [Sync]
GPU:              [Execute L0]            [Execute L1]     [Execute L2] ...
                   ↑ keine Lücke — L1 war schon in der Queue
```

Ohne Lookahead würde nach jedem Kernel eine ~5–15 μs Lücke entstehen (CPU-Dispatch-Overhead). Bei 14B mit ~200 Launches pro Token sind das 1 000–3 000 μs — bis zu 20 % des Token-Budgets. Mit Lookahead: 0.

**Quality-Monitor als Fused-Epilog (nicht separater Kernel):**

Der Quality-Monitor (Säule 5) läuft **nicht** als eigener Kernel, sondern als Epilog im letzten Kernel jedes Layers — ca. 20 GPU-Instruktionen für Mean-Abs + Max-Abs + NaN-Check + FP8-Saturation-Count. Der Overhead ist vernachlässigbar; es gibt keine Race-Condition, weil alles im selben Stream und sogar im selben Kernel läuft.

Bei Drift-Detektion schreibt der Epilog ein **Dirty-Flag** in Pinned Host Memory:

```rust
// Setup (einmal beim Start):
let dirty_flag: *mut u32 = hip_host_malloc_mapped(4)?;  // hipHostMallocMapped

// Im GPU-Kernel-Epilog (generierter HIP-Code):
// if (mean_abs > threshold || nan_count > 0 || fp8_saturations > tolerance)
//     atomicExch(dirty_flag_device_ptr, 1);

// Am Token-Ende in Rust (nach dem einzigen hipStreamSynchronize):
unsafe {
    if *dirty_flag != 0 {
        *dirty_flag = 0;
        self.handle_precision_revision()?;
    }
}
```

Der `hipStreamSynchronize` am Token-Ende ist **ohnehin nötig** (die CPU muss die Logits lesen, um das nächste Token zu samplen). Das Dirty-Flag wird durch dieselbe Sync-Barriere sichtbar — kein zusätzlicher Sync-Point, kein Overhead.

**Timing für den Bandit (Zero-Sync):**

Der Bandit braucht pro Kernel-Call die Ausführungszeit. Statt nach jedem Kernel `hipEventElapsedTime` aufzurufen (Sync-Point!), werden die Events in die Queue geschoben und **am Token-Ende batch-weise ausgelesen**:

```rust
// Vor jedem Kernel-Call:
hip_event_record(start_events[i], stream);
launch_kernel(..., stream);
hip_event_record(end_events[i], stream);

// Am Token-Ende (nach Sync):
for i in 0..num_kernels {
    let time_us = hip_event_elapsed(start_events[i], end_events[i]);
    self.bandits[i].record(time_us);
}
```

Alle Events werden in einem Array vorallokiert (kein Runtime-Malloc). Die Abfrage erfolgt nach der Sync-Barriere, die ohnehin existiert. Ergebnis: exakte Timing-Daten ohne einen einzigen zusätzlichen Sync-Point im Hot-Path.

### 3.8 Rust-FFI-Strategie (Eigene HIP-Bindings)

v1.0 verzichtet auf externe Crates für HIP-Interaktion (`hip-sys`, `cust`, etc.) und generiert stattdessen eigene minimale Bindings über `bindgen` gegen die ROCm-7.2-Header. Das sind ~20 Funktionen, die in einer halben Tag implementierbar sind und vollständige Kontrolle über ABI und Versionskompatibilität geben.

**Benötigte HIP-Funktionen (vollständige Liste):**

```rust
// Memory-Management (6 Funktionen):
hipMalloc, hipFree, hipMemcpy, hipMemcpyAsync,
hipHostMalloc, hipHostFree

// Kernel-Launch (3 Funktionen):
hipModuleLoad, hipModuleGetFunction, hipModuleLaunchKernel

// Stream-Management (3 Funktionen):
hipStreamCreate, hipStreamSynchronize, hipStreamDestroy

// Event-Timing (4 Funktionen):
hipEventCreate, hipEventRecord, hipEventElapsedTime, hipEventDestroy

// Device-Info (2 Funktionen):
hipGetDeviceProperties, hipMemGetInfo

// Mapped Memory (1 Funktion):
hipHostRegister   // für Dirty-Flag Pinned Memory
```

**Generierung via bindgen:**

```fish
# build.rs oder einmaliges Setup:
bindgen /opt/rocm/include/hip/hip_runtime_api.h \
  --allowlist-function "hip(Malloc|Free|Memcpy|MemcpyAsync)" \
  --allowlist-function "hip(Module|Stream|Event|Host|GetDevice|MemGetInfo).*" \
  --allowlist-type "hipDeviceProp_t|hipStream_t|hipEvent_t|hipModule_t|hipFunction_t" \
  --no-layout-tests \
  -o src/hip/bindings.rs
```

**RAII-Wrapper:**

```rust
/// GPU-Memory mit Rust-Ownership. Drop ruft hipFree.
pub struct GpuBuffer {
    ptr: *mut u8,
    size: usize,
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe { hipFree(self.ptr as *mut c_void); }
    }
}

/// HIP-Kernel-Modul. Drop ruft hipModuleUnload.
pub struct GpuKernel {
    module: hipModule_t,
    function: hipFunction_t,
}

impl Drop for GpuKernel {
    fn drop(&mut self) {
        unsafe { hipModuleUnload(self.module); }
    }
}

/// Kernel-Launch: Argumente einzeln als *const c_void übergeben.
/// Keine Structs als Kernel-Args (Padding/Alignment-Risiko).
impl GpuKernel {
    pub unsafe fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: hipStream_t,
        args: &mut [*mut c_void],
    ) -> Result<(), HipError> {
        let err = hipModuleLaunchKernel(
            self.function,
            grid.0, grid.1, grid.2,
            block.0, block.1, block.2,
            shared_mem, stream,
            args.as_mut_ptr(), std::ptr::null_mut(),
        );
        if err != 0 { Err(HipError(err)) } else { Ok(()) }
    }
}
```

**Kernel-Module-Lifecycle.** Kernel-Binaries (`.co`-Dateien) belegen VRAM für den Code-Cache. Bei 100+ Varianten können das 50–100 MB sein — auf einer 14B-geladenen GPU relevant. v1.0 hält maximal 30 Module gleichzeitig geladen (LRU-Eviction). Da der Bandit in der Steady-State nur 3–5 Varianten pro Shape aktiv nutzt, ist das ausreichend.

**Warum eigene Bindings statt Crate:**

1. ROCm 7.2 + RDNA 4 ist so neu, dass existierende Crates die nötigen Funktionen möglicherweise nicht abdecken.
2. Keine transitive Dependency auf ein möglicherweise unmaintained Crate.
3. 19 Funktionen + 5 Typen = minimal, vollständig kontrollierbar, kein Überraschungs-Risiko bei Updates.

---

## 4. Genetische Algorithmen

### 4.1 Überblick: Wo GA eingesetzt wird und wo nicht

GA ist nicht universell, sondern das richtige Werkzeug für **diskrete Optimierungsprobleme mit großem Suchraum, nicht-differenzierbarer Fitness und verrauschter Messung**. In v1.0 gibt es genau drei solcher Probleme:

| Einsatzort | Warum GA | Alternative verworfen |
|---|---|---|
| Kernel-Auto-Tuning | ~100 000 gültige Konfigurationen, Performance nur messbar | Grid-Search zu langsam, Gradient-Descent unmöglich (diskret) |
| Precision-Allocation | 3^n Kombinationen (n Layer), Multi-Objective | NSGA-II explizit dafür gebaut |
| Fusion-Pass-Kombinatorik | Constraint-abhängige Kombinatorik | SAT-Solver theoretisch möglich, aber schlecht für Performance-Targets |

Wo GA **nicht** eingesetzt wird:

| Einsatzort | Warum kein GA | Tatsächliche Lösung |
|---|---|---|
| Self-Tuning Runtime | Online, einzelne μs-Entscheidungen, kleiner Kandidaten-Satz | Multi-Armed Bandit (UCB1) |
| Dequant IR Codegen | Deterministische Übersetzung, keine Suche | Regelbasierter Emitter |
| Model Introspection | Statistische Analyse auf Daten | Direkte Berechnung |
| Graph-Konstruktion | Deterministisch aus GGUF-Metadaten | Template-basiert |

### 4.2 Kernel-Auto-Tuning-GA

**Problem.** Für eine gegebene Shape `(M, N, K)` und ein gegebenes Quant-Format gibt es tausende gültige Kernel-Konfigurationen: Tile-Größen, Wave-Counts, K-Unroll-Faktoren, LDS-Nutzung ja/nein, Prefetch-Tiefe. Welche ist auf `gfx1201` die schnellste? Das lässt sich nicht theoretisch ausrechnen — jede Konfiguration muss auf der Hardware ausgeführt und gemessen werden.

**Genome-Definition.**

```rust
pub struct KernelGenome {
    // Tile-Struktur
    pub tile_m: u8,          // 16, 32, 64, 128 (muss Vielfaches von 16 sein für WMMA)
    pub tile_n: u8,          // dto.
    pub tile_k: u8,          // 16, 32, 64
    pub tiles_per_wave: u8,  // 1, 2, 4
    pub waves_per_block: u8, // 1, 2, 4, 8

    // Speicher-Strategie
    pub use_lds_for_a: bool,
    pub use_lds_for_b: bool,
    pub prefetch_depth: u8,  // 0, 1, 2

    // K-Loop
    pub k_unroll: u8,        // 1, 2, 4, 8
    pub double_buffer: bool,

    // Quant-spezifisch
    pub dequant_strategy: DequantStrategy,
}

pub enum DequantStrategy {
    Inline,                  // Dequant pro Element im Kernel
    PrePass { lds_bytes: u16 }, // Separater Pre-Pass in LDS
    Batched { batch_size: u8 }, // Batched Dequant für mehrere Blöcke
}
```

**Constraints** (Fitness=0 falls verletzt, verhindert ungültige Kandidaten):

```rust
fn validate(g: &KernelGenome, fmt: &QuantFormat) -> bool {
    // LDS-Budget: 64 KB
    let lds_bytes = g.tile_m * g.tile_k * 2 * (g.use_lds_for_a as u32)
                  + g.tile_n * g.tile_k * 2 * (g.use_lds_for_b as u32)
                  + g.dequant_strategy.lds_overhead();
    if lds_bytes > 65536 { return false; }

    // VGPR-Budget: ~104 pro Wave
    let vgpr_estimate = estimate_vgprs(g, fmt);
    if vgpr_estimate > 104 { return false; }

    // WMMA-Alignment
    if g.tile_m % 16 != 0 || g.tile_n % 16 != 0 || g.tile_k % 16 != 0 {
        return false;
    }

    // Waves × Tiles muss Workgroup-Size matchen
    if g.tiles_per_wave * g.waves_per_block > MAX_TILES_PER_BLOCK { return false; }

    true
}
```

**Fitness-Funktion.** Für einen Kandidaten wird der Kernel generiert, kompiliert, und gemessen:

```
fn fitness(g: KernelGenome, shape: KernelShape, fmt: &QuantFormat) -> f32:
    if not validate(g, fmt): return -INFINITY

    kernel = codegen_gpu(g, shape, fmt)

    // Warmup
    für 5 Iterationen: kernel.run(dummy_input)

    // Messung
    times = []
    für 20 Iterationen:
        t0 = gpu_timestamp()
        kernel.run(real_input)
        t1 = gpu_timestamp()
        times.push(t1 - t0)

    return -median(times)  // Minimierung → negative Zeit
```

**GA-Parameter.**

- Population: 100 Individuen
- Generationen: 50 (→ 5 000 Evaluierungen pro Shape)
- Selection: Tournament-Size 3
- Crossover: Uniform, Rate 0.7
- Mutation: Gauss-Perturbation einzelner Gene, Rate 0.1 pro Gen
- Elitism: Top 5 % unverändert in nächste Generation

**Zeitbudget.** Pro Evaluierung ~100 ms (5 ms Messung + 95 ms Kompilierung bei erstem Treffer; Wiederholungen mit gleichem Kernel nur 5 ms). 5 000 Evaluierungen × 100 ms = 500 s ≈ 8 Minuten pro Shape. Ein 8B-Modell hat ~20 unique Shapes (QKV, O-Proj, Gate, Up, Down × Prefill/Decode) → Erstbau ~2,5 Stunden, einmalig, danach Cache-Hit.

**Caching.**

```
~/.rocmforge/kernels/
├── gfx1201/
│   ├── gemm_4096x4096_q4_k_m.json     # Top-5 Varianten + Genome
│   ├── gemm_4096x4096_q4_k_m.co       # Compiled Code Objects
│   ├── gemv_4096x12288_q4_k_m.json
│   ├── gemv_4096x12288_q4_k_m.co
│   └── ...
└── zen4_plus/
    └── ...
```

**Output.** Nicht nur der beste Kernel, sondern die Pareto-Front der Top 3–5 Varianten, die an den Bandit weitergereicht werden. Warum? Weil ein GA im Offline-Mittel den besten findet, aber zur Runtime (mit anderer GPU-Auslastung, anderer Taktfrequenz) eine andere Variante besser sein kann.

### 4.3 Precision-Allocation-GA (NSGA-II)

**Problem.** Gegeben ein Modell mit n Layern und nun **vier** möglichen Precision-Stufen pro Layer (FP8-E4M3, FP8-E5M2, FP16, BF16 — FP32 nur als Safety-Fallback außerhalb der GA-Domäne), gibt es 4^n Kombinationen. Bei n=32 sind das rund 10^19. Zusätzlich gibt es **drei dedizierte Precision-Entscheidungen**, die nicht pro Layer sondern einmalig pro Modell getroffen werden: Embedding-Lookup, LM-Head, und **KV-Cache**. Jede Kombination ist ein Trade-off: höhere Precision → bessere Qualität (niedrigere Perplexity auf Validation-Set), aber niedrigere Speed und höherer Memory-Footprint. Das ist ein Multi-Objective-Problem — klassischer NSGA-II-Case.

**FP8 als Default-Suchraum.** In v1.1 ist FP8-E4M3 der Startpunkt für jeden Layer. Die GA sucht damit **nicht nach Gründen, Precision zu senken**, sondern nach den wenigen Layern, wo sie erhöht werden muss. Das invertiert die Problemstellung gegenüber klassischen Quantization-Aware-Ansätzen und ist deutlich effizienter: in der Praxis brauchen typisch nur 10–20 % der Layer eine Hochstufung, während der Rest auf FP8 bleiben kann.

**KV-Cache-Precision als eigene Dimension.** Der KV-Cache ist in Transformer-Inference der größte einzelne Memory-Block zur Laufzeit (bei 8B-Modellen und 8k-Kontext: mehrere GB). Seine Precision hat direkten Einfluss auf maximal mögliche Kontext-Länge und auf den Attention-Berechnungsaufwand. **FP8-E5M2 ist die Standardwahl** für den KV-Cache: der größere Exponent-Bereich (±57 344) ist robust gegen Outlier-Werte, die bei K/V häufiger auftreten als bei normalen Gewichten. Bei detektierter Drift (Quality Monitor) wird auf FP16-KV-Cache hochgestuft, was den KV-Cache-Footprint verdoppelt aber innerhalb der VRAM-Reserve bleibt.

**Genome.**

```rust
pub struct PrecisionGenome {
    /// Pro Layer: FP8-E4M3 (Standard), FP8-E5M2, FP16, BF16.
    /// FP32 ist kein GA-Kandidat — nur Safety-Monitor-Fallback.
    pub per_layer: Vec<PrecisionHint>,
    /// Embedding-Lookup: meist FP8-E4M3, aber bei kritischen Tokens hochstufen.
    pub embedding: PrecisionHint,
    /// LM-Head (Logits): häufig FP16 oder BF16 wegen Dynamikbereich.
    pub lm_head: PrecisionHint,
    /// KV-Cache-Precision: Standard FP8-E5M2, bei langem Kontext kritisch.
    pub kv_cache: PrecisionHint,
    /// Attention-Akkumulator: FP32 fast immer, kein GA-Suchraum.
    pub attn_acc_fp32: bool,  // typisch true, nur bei schwacher Präzision false
}
```

**NSGA-II.** Non-dominated Sorting Genetic Algorithm II liefert nicht eine einzelne beste Lösung, sondern die **Pareto-Front**: die Menge der Lösungen, für die keine andere Lösung existiert, die in beiden Objectives besser ist. Der User (oder automatisch eine Regel) wählt aus der Pareto-Front die gewünschte Position.

**Objective 1: Qualität.** Wird approximiert durch Kullback-Leibler-Divergenz zwischen Logits-Output des Kandidaten und einer FP32-Referenz auf einem kleinen Prompt-Set (10–20 Prompts, ~100 Tokens each). Niedrigere KL = höhere Qualität. Die Referenz läuft einmal beim Modell-Laden und wird gecached.

```
fn fitness_quality(g: PrecisionGenome, ref_logits: &[Logits], prompts: &[Prompt]) -> f32:
    engine = build_engine_with_precision(g)
    candidate_logits = für jeden Prompt: engine.forward(Prompt)
    kl = mean([kl_divergence(c, r) for (c, r) in zip(candidate_logits, ref_logits)])
    return -kl  // Maximierung von -KL
```

**Objective 2: Speed & Memory.** Dies ist in v1.1 ein kombiniertes Objective:

```
fn fitness_speed_and_memory(g: PrecisionGenome) -> f32:
    engine = build_engine_with_precision(g)

    // Speed-Komponente
    t0 = time()
    engine.decode(256 Tokens)
    t1 = time()
    speed = 256.0 / (t1 - t0)

    // Memory-Komponente (KV-Cache bei 8k-Kontext)
    kv_mem_bytes = kv_cache_size_bytes(g.kv_cache, context=8192)
    memory_efficiency = (16e9 - kv_mem_bytes) / 16e9  // relative VRAM-Reserve

    // Gewichtete Kombination — Gewichte sind Hyperparameter, pro Use-Case tunebar
    return 0.7 × speed + 0.3 × memory_efficiency × SPEED_SCALE
```

KV-Cache-Memory ist explizit Teil der Fitness, weil reine Decode-Speed ohne KV-Memory-Berücksichtigung zu Lösungen führt, die bei langen Kontexten OOM gehen.

**Integration mit Model Introspection.** Das `ModelProfile` (Säule 1) liefert `precision_recommendation: Vec<PrecisionHint>` — dies wird als **Startpunkt der initialen Population** verwendet (das erste Individuum ist exakt die Empfehlung, das zweite variiert einzelne Layer, usw.). Das beschleunigt die GA-Konvergenz erheblich, weil die Introspection bereits die kritischen Layer identifiziert hat. In v1.1 startet die Empfehlung mit FP8-E4M3 für "sichere" Layer und FP16 für Layer mit SNR-Risk < 2.0.

**GA-Parameter.**

- Population: 60
- Generationen: 30 → 1 800 Evaluierungen
- Jede Evaluierung ~500 ms (kurze Decode-Messung + KL-Berechnung)
- Gesamt: ~15 Minuten pro Modell, einmalig, gecached

**Output: Pareto-Front.** Typisch 5–10 Punkte auf der Front. Standard-Wahl: das Individuum, das 95 % der FP32-Qualität bei maximaler Speed-Memory-Balance liefert — das ist das "realistische Performance"-Ziel aus Abschnitt 1.4.

**Cache.**

```
~/.rocmforge/profiles/
├── qwen3-8b-q4km.json       # Pareto-Front + gewählter Punkt
├── llama-3.1-8b-q4km.json
└── ...
```

### 4.4 Fusion-Pass-GA

**Problem.** Der regelbasierte Fusion-Pass (Säule 2) fusioniert nur lokale, eindeutig sichere Patterns. Aber es gibt nicht-lokale Fusionen, die manchmal besser sind und manchmal schlechter, je nach Register-Druck und Cache-Verhalten. Beispiel: "Fused Residual+Norm+QKV+RoPE" ist in einem großen Kernel — spart Memory-Roundtrips, aber verbraucht viele VGPRs und könnte Occupancy killen.

**Genome.**

```rust
pub struct FusionGenome {
    /// Pro Fusion-Kandidat: aktivieren oder nicht.
    pub active_fusions: Vec<FusionDecision>,
}

pub struct FusionDecision {
    pub pattern_id: FusionPatternId,
    pub active: bool,
    pub target_nodes: Vec<NodeId>,
}

pub enum FusionPatternId {
    NormQkv,
    NormQkvRope,
    ResidualNorm,
    GateUpSwiglu,
    SwigluDown,
    NormQkvRopeAttention,  // großer Fused-Kernel
    // ... weitere
}
```

**Constraints.** Nicht jede Fusions-Kombination ist gültig:

- Zwei Fusionen dürfen nicht denselben Knoten beanspruchen.
- Manche Fusionen setzen andere voraus (z. B. `NormQkvRope` schließt separates `NormQkv` aus).
- Der resultierende Kernel muss VGPR- und LDS-Budgets respektieren (geschätzt vor der Evaluierung).

**Fitness.** End-to-End-Decode-Speed auf einem Standard-Benchmark, gleiche Messung wie Precision-GA. Optional mit Qualitäts-Check (nur falls die Fusion Numerik ändert, z. B. Reihenfolge der FMAs).

**GA-Parameter.**

- Population: 50
- Generationen: 20 → 1 000 Evaluierungen
- Pro Evaluierung ~300 ms
- Gesamt: ~5 Minuten pro Modell-Architektur, gecached (nicht pro Modell-Instanz, sondern pro Architektur-Template)

### 4.5 Warum GA für diese drei, Bandit für Runtime

Die Entscheidung GA vs. Bandit vs. Regelbasiert folgt aus der Problem-Struktur:

**GA eignet sich wenn:**
- Suchraum ist groß und diskret (10^4 bis 10^15 Kandidaten).
- Fitness ist teuer, aber die Offline-Budget erlaubt es (Minuten).
- Ergebnis wird gecached und oft wiederverwendet.

**Bandit eignet sich wenn:**
- Kandidaten-Satz ist klein (≤ 10).
- Entscheidung muss schnell sein (μs).
- Fitness ist noisy, aber günstig pro Sample (einzelner Call).
- Zustand ändert sich zur Laufzeit (GPU-Takt, VRAM-Druck).

**Regelbasiert eignet sich wenn:**
- Mapping ist deterministisch.
- Korrektheit ist wichtiger als Optimalität.
- Beispiele: Dequant-IR → Instruktionen, GGUF → Graph.

Diese Trennung verhindert ein klassisches Anti-Pattern: "alles ist ein Nagel, wenn man einen Hammer hat". GA überall einzusetzen würde die Runtime zum Rätsel machen; Bandit für Kernel-Tuning hätte keinen Weg, systematisch den Suchraum abzusuchen. Jedes Problem bekommt den passenden Algorithmus.

### 4.6 Caching-Strategie

Alle GA-Ergebnisse werden in versioniertem JSON gecached. Cache-Keys enthalten:

- Hardware-Fingerprint (für GPU: `gfx1201-ROCm7.2.2`; für CPU: `Zen4-Linux6.11`).
- Quant-Format-Version.
- Modell-Hash (SHA256 über GGUF-Metadaten, nicht über Gewichte — gleiche Architektur + gleiche Shapes = gleicher Cache-Eintrag).
- GA-Version (für Cache-Invalidierung bei Algorithmus-Änderung).

Bei Cache-Miss läuft der GA. Bei Cache-Hit lädt die Runtime die Top-Varianten direkt. Nutzer können die Cache-Inspektion und gezieltes Re-Tuning über das `rf-forge`-CLI-Tool steuern (siehe 4.7).

### 4.7 Offline-GA-Tuning CLI: `rf-forge`

**Trennung von Inference- und Tuning-Workflow.** Die GA-Läufe (Kernel-Tuning, Precision-Allocation, Fusion-Pass) sind lang (Stunden pro Modell im Cold Start), stören den regulären Inference-Pfad und sollten reproduzierbar und scriptbar sein. Statt GA-Logik in das Haupt-Binary `rocmforge` einzubetten und dort nur implizit zu starten, stellt v1.0 ein dediziertes Offline-Tuning-Tool bereit: **`rf-forge`**.

Das `rocmforge`-Haupt-Binary führt *nur Inference* aus. Fehlt ein Cache-Eintrag, erhält der User eine klare Meldung mit dem entsprechenden `rf-forge`-Aufruf — kein stundenlanges Warten auf einen unerwarteten GA-Lauf mitten im Chat.

**Kommando-Struktur.**

```
rf-forge <subcommand> [options]

Subcommands:
  tune-kernels       Kernel-Auto-Tuning-GA für ein Modell
  tune-precision     Precision-Allocation-GA (NSGA-II)
  tune-fusion        Fusion-Pass-GA für eine Architektur
  tune-all           Voller Cold-Start-Workflow (alle drei GA-Phasen)
  cache              Cache-Inspektion und Wartung
  bench              Benchmark-Suite auf gecachten Konfigurationen
  validate           VALU-Parity-Check für alle gecachten Kernel
  export             Export Config (z.B. für CI-Reproduzierbarkeit)
```

**Typische Workflows.**

```fish
# Ein komplett neues Modell vorbereiten (Cold Start, ~2,5 h)
rf-forge tune-all --model ~/models/Qwen3-8B-Q4_K_M.gguf \
                  --target gfx1201 \
                  --log ./tune-qwen3.log

# Nur Kernel-Tuning neu laufen lassen (nach ROCm-Update)
rf-forge tune-kernels --model ~/models/Qwen3-8B-Q4_K_M.gguf \
                      --shapes "gemm_*,gemv_*" \
                      --budget 30m

# Precision-GA mit benutzerdefiniertem Qualitäts-Ziel
rf-forge tune-precision --model ~/models/Llama-3.1-8B-Q4_K_M.gguf \
                        --min-quality 0.98 \
                        --prompts ./eval-prompts.txt

# Cache-Übersicht
rf-forge cache list
rf-forge cache inspect --shape "gemv_4096x12288_q4_k_m"
rf-forge cache invalidate --older-than 30d
rf-forge cache size  # Gesamtgröße in MB

# Komplett-Validierung nach Toolchain-Änderung (alle Kernel gegen VALU-Pfad)
rf-forge validate --all --tolerance 1e-4

# Benchmark-Suite mit aktuellem Cache
rf-forge bench --model ~/models/Qwen3-8B-Q4_K_M.gguf \
               --runs 100 \
               --report ./bench-report.md
```

**Relevante Optionen für `tune-kernels`.**

```
--model <PATH>          Pfad zur GGUF-Datei
--target <gfx1201|zen4> Hardware-Target (beide → beides nacheinander)
--shapes <GLOB>         Welche Shapes tunen (Default: alle unique Shapes)
--budget <TIME>         Zeit-Limit pro Shape (z.B. 10m, 1h)
--population <N>        GA-Population (Default: 100)
--generations <N>       GA-Generationen (Default: 50)
--seed <N>              RNG-Seed für Reproduzierbarkeit
--dry-run               Nur Plan ausgeben, nichts ausführen
--parallel-shapes <N>   Wie viele Shapes gleichzeitig (Default: 1, stabile Messung)
--log <PATH>            Log-Ausgabe
--force-rebuild         Cache ignorieren, neu tunen
```

**Log-Output-Format.** `rf-forge` schreibt strukturiertes JSONL zur Log-Datei, plus menschenlesbaren Progress-Output an stdout:

```json
{"ts":"2026-04-20T05:12:03Z","event":"shape_start","shape":"gemv_4096x12288_q4_k_m",
 "population":100,"generations":50,"budget_seconds":600}
{"ts":"2026-04-20T05:12:04Z","event":"generation_complete","shape":"gemv_...",
 "generation":1,"best_time_us":87.3,"median_time_us":112.5,"invalid":12}
{"ts":"2026-04-20T05:20:45Z","event":"shape_complete","shape":"gemv_...",
 "best_time_us":71.2,"pareto_front":5,"total_evaluations":4832,
 "elapsed_seconds":522}
```

Diese Logs sind maschinenauswertbar — z. B. für CI-Regressions-Checks, wenn das Tuning auf einem Build-Server läuft und Ergebnisse gegen eine Baseline verglichen werden.

**Reproducible Builds.** `rf-forge export --model X > tune.toml` schreibt die vollständige Konfiguration (GA-Seeds, Hardware-Fingerprint, Quant-Format-Versionen, Pareto-Front) in eine Config-Datei. Ein zweites System kann mit `rf-forge tune-all --from-config tune.toml` denselben Tuning-Lauf reproduzieren. Das ist wichtig für CI und für Team-Workflows.

**Warum separates Binary statt Subcommand.** Drei Gründe:

1. **Dependency-Minimierung** für das Inference-Binary. `rocmforge` muss keine GA-Infrastruktur, kein NSGA-II, keine Fitness-Messungs-Orchestrierung enthalten — kleineres Binary, weniger Attack-Surface.
2. **Operational Clarity.** Ein User, der `rocmforge chat ...` aufruft, soll nie überrascht werden: entweder Cache-Hit → sofort bereit, oder Cache-Miss → klare Fehlermeldung. Keine "dauert jetzt 2,5 Stunden"-Überraschung.
3. **Server-Workflow.** Produktionsserver laden fertige Caches; Tuning findet auf Entwickler-Workstations oder dedizierten Tuning-Maschinen statt. Die Trennung der Binaries macht diese Pipeline explizit.

---

## 5. Datenflüsse

### 5.1 First-Run-Flow (Cold Start)

Erstes Laden eines neuen Modells, ohne Cache:

```
┌─────────────────────────────────────────────────────────────┐
│  1. CLI: rocmforge chat --model qwen3-8b-q4km.gguf          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  2. GGUF-Parser                                             │
│     → Metadaten lesen (n_layers, n_heads, ...)              │
│     → Tensor-Tabelle aufbauen                               │
│     → Weight-Preloading in VRAM (4.5 s parallel zu Step 3)  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  3. Säule 1: Model Introspection                            │
│     → Magnitude-Scan über alle Tensoren (3–5 s)             │
│     → ModelProfile erzeugen                                 │
│     → ⚠ Warnung ausgeben wenn snr_risk < 2.0                │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  4. Säule 2: Graph-Konstruktion                             │
│     → Template-Lookup (Llama/Qwen/Mistral-Style)            │
│     → Knoten+Kanten aus GGUF-Metadaten aufbauen             │
│     → Precision-Hints aus ModelProfile propagieren          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  5. Säule 2: Regelbasierter Fusion-Pass                     │
│     → Pattern-Matching über den Graph                       │
│     → Lokale Fusionen anwenden                              │
│     → ~40–60 % weniger Knoten                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ├─────────► Cache-Hit? ──► skip 6, 7, 8
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  6. Fusion-GA (5 min, einmalig pro Architektur)             │
│     → Non-lokale Fusionen evaluieren                        │
│     → Fitness: End-to-End-Decode-Speed                      │
│     → Output: finaler FusedGraph                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  7. Kernel-Auto-Tuning-GA (8 min × N Shapes)                │
│     → Für jede unique Shape im FusedGraph:                  │
│       - 5 000 Kandidaten evaluieren (100 ms each)           │
│       - Top-5 als KernelVariants extrahieren                │
│     → Parallelisierung: 1 Shape zur Zeit (sonst Thermal)    │
│     → Typisch: 20 Shapes × 8 min = 2,5 h cold start         │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  8. Precision-GA (NSGA-II, 15 min)                          │
│     → Startpopulation aus ModelProfile.recommendation       │
│     → Pareto-Front finden                                   │
│     → Standard: 95 %-Qualitäts-Punkt wählen                 │
│     → Finale PrecisionConfig                                │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  9. Cache schreiben                                         │
│     → ~/.rocmforge/kernels/gfx1201/*.co                     │
│     → ~/.rocmforge/profiles/qwen3-8b-q4km.json              │
│     → Fusions-Plan, Precision-Config                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│ 10. Quality Monitor: Calibration-Pass                       │
│     → Dummy-Prompt "The quick brown fox jumps"              │
│     → Hidden-State-Ranges pro Knoten aufnehmen              │
│     → ExpectedRange-Map befüllen (1 s)                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│ 11. Runtime initialisieren                                  │
│     → ShapeBandit pro Shape (mit GA-Varianten als Arms)     │
│     → Chat-Prompt anzeigen, bereit für User-Input           │
└─────────────────────────────────────────────────────────────┘
```

**Gesamt-Zeit Cold Start:** ~2,75 Stunden (einmalig für ein neues Modell auf neuer Hardware).

### 5.2 Cached-Run-Flow (Warm Start)

Zweites und alle weiteren Starts:

```
┌─────────────────────────────────────────────────────────────┐
│  1. CLI-Aufruf (identisch)                                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  2. GGUF-Parser + Weight-Preloading (parallel, ~4,5 s)      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  3. Cache-Lookup (~100 ms)                                  │
│     → Hash (GGUF-Metadaten) berechnen                       │
│     → Kernel-Cache laden                                    │
│     → Precision-Profile laden                               │
│     → FusedGraph rekonstruieren                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  4. Quality Monitor: Calibration-Pass (~1 s)                │
│     (Calibration wird nicht gecached, weil kurz, und hilft  │
│      bei Treiber-Änderungen, die Hidden-States verschieben) │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  5. Runtime initialisieren (~100 ms)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                         READY (6 Sekunden)
```

### 5.3 Inference-Flow (pro Token)

```
┌─────────────────────────────────────────────────────────────┐
│  Eingehende Tokens (Prompt) oder letztes generiertes Token  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Embedding-Lookup                                           │
│  (mit FP32-Overlay für critical_embedding_tokens,           │
│   falls ModelProfile solche identifiziert hat)              │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Pro Layer i (meist 32):                                    │
│                                                             │
│    ┌─────────────────────────────────────────────────┐      │
│    │ ShapeBandit.select(shape) → variant_id          │      │
│    │ (UCB1-Auswahl aus Top-5 GA-Varianten)           │      │
│    └──────────────────┬──────────────────────────────┘      │
│                       │                                     │
│    ┌──────────────────▼──────────────────────────────┐      │
│    │ Kernel ausführen (FusedNormQkvRope, Attention,  │      │
│    │ FusedGateUpSwiglu, FFN-Down, ResidualNorm)      │      │
│    │ Laufzeit messen (GPU-Timestamp)                 │      │
│    └──────────────────┬──────────────────────────────┘      │
│                       │                                     │
│    ┌──────────────────▼──────────────────────────────┐      │
│    │ ShapeBandit.record(variant_id, time_us)         │      │
│    └──────────────────┬──────────────────────────────┘      │
│                       │                                     │
│    ┌──────────────────▼──────────────────────────────┐      │
│    │ Quality Monitor.check(node_id, hidden_state)    │      │
│    │ alle N Tokens (N=32 Standard)                   │      │
│    └──────────────────┬──────────────────────────────┘      │
│                       │                                     │
│                       ├─► Drift erkannt? ──► Revision      │
│                       │                                     │
│                      (next layer)                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  LM-Head (GEMV, meist FP32 für Logits-Präzision)            │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Sampling (Temperature, Top-P, Top-K)                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                         Token
```

**Budget pro Decode-Token bei 100 tok/s:** 10 ms = 10 000 μs für 32 Layer × 4 Haupt-GEMV + Attention = ~75 μs pro GEMV-Operation. Bandit-Overhead: ~100 ns pro Call, vernachlässigbar.

### 5.4 Revision-Flow (Quality-Drift)

Wenn der Quality Monitor Drift erkennt:

```
Check: Drift bei Layer 18 Attention, z-Score = 4.2

  ┌─► RevisionEvent loggen
  │
  ├─► PrecisionHint für Layer 18 von Fp16 → Bf16 hochstufen
  │
  ├─► Runtime für Layer 18 neue Kernel-Variante wählen
  │   (BF16-Variante liegt in Cache; falls nicht → FP32-Fallback)
  │
  └─► Inference-Schritt WIEDERHOLEN ab Layer 18
       (letzter sauberer Hidden-State war am Eingang von Layer 18
        im Checkpoint-Buffer — zusätzlicher VRAM ~10 MB pro Layer)
```

Der Checkpoint-Buffer ist ein Ring-Puffer der letzten 4 Layer-Inputs. Das kostet ~40 MB VRAM bei 8B-Modellen — vernachlässigbar bei 16 GB VRAM.


---

## 6. Schnittstellen-Definitionen

Die Säulen kommunizieren über klar typisierte Schnittstellen. Keine Säule greift direkt auf die interne Implementierung einer anderen zu; alle Interaktion läuft über die hier definierten Structs.

### 6.1 GGUF-Loader → Model Introspection

```rust
// Input:
pub struct LoadedGguf {
    pub metadata: GgufMetadata,
    pub tensors: HashMap<String, Tensor>,
    pub quant_format: QuantFormatId,
}

// Output:
pub struct ModelProfile { /* siehe 2.2 */ }

// Aufruf:
pub fn introspect(gguf: &LoadedGguf) -> Result<ModelProfile, IntrospectError>;
```

**Fehlerbehandlung.** Fehlschlag von `introspect` ist **nicht fatal**: wenn z. B. ein Tensor eine unerwartete Shape hat, wird eine Default-`ModelProfile` mit `precision_recommendation = [Fp16Scales; n_layers]` zurückgegeben und eine Warnung geloggt. Inferenz soll möglich bleiben, auch wenn die Optimierung suboptimal ist.

### 6.2 Model Introspection → Graph-Builder

```rust
// Input:
pub struct GraphBuildRequest<'a> {
    pub gguf: &'a LoadedGguf,
    pub profile: &'a ModelProfile,
}

// Output:
pub struct ComputeGraph { /* siehe 2.3 */ }

// Aufruf:
pub fn build_graph(req: GraphBuildRequest) -> Result<ComputeGraph, GraphBuildError>;
```

**Konvention.** Precision-Hints aus `profile.precision_recommendation` werden pro Layer in die entsprechenden `OpNode.precision`-Felder propagiert. Die Embedding-Node erhält zusätzlich den Hinweis, welche Token-IDs als `critical_embedding_tokens` markiert sind.

### 6.3 Graph-Builder → Fusion-Pipeline

```rust
// Input:
pub fn rule_based_fuse(graph: ComputeGraph) -> ComputeGraph;

// Fusion-GA:
pub struct FusionRequest {
    pub graph: ComputeGraph,
    pub cache: &FusionCache,
}
pub fn ga_fuse(req: FusionRequest) -> Result<FusedGraph, FusionError>;
```

**Fehlerfall.** Wenn die Fusion-GA zu keinem validen Ergebnis konvergiert (z. B. wegen zu enger VGPR-Budgets bei allen Kandidaten), wird der rein regelbasiert fusionierte Graph als `FusedGraph` zurückgegeben. Die Regelfusion ist immer ein sicherer Fallback.

### 6.4 Fusion-Pipeline → Dequant-IR-Codegen

```rust
pub struct KernelBuildRequest<'a> {
    pub op_node: &'a OpNode,
    pub quant_format: &'a QuantFormat,
    pub target: KernelTarget,
    pub shape: KernelShape,
    pub genome: &'a KernelGenome,  // aus GA oder Default
}

pub fn codegen(req: KernelBuildRequest) -> Result<CompiledKernel, CodegenError>;
```

**Konvention.** `codegen` ist deterministisch: gleiche Inputs → gleicher `CompiledKernel`. Das Cache-System nutzt dies, um teure Kompilierungen zu vermeiden.

### 6.5 Kernel-GA → Self-Tuning Runtime

```rust
// Input an Runtime beim Modell-Laden:
pub struct KernelInventory {
    pub variants_per_shape: HashMap<ShapeKey, Vec<KernelVariant>>,
}

// Pro Call:
impl ShapeBandit {
    pub fn select(&mut self) -> VariantId;
    pub fn record(&mut self, id: VariantId, time_us: f32);
}
```

**Konvention.** Das `KernelInventory` wird vollständig zu Modell-Lade-Zeit befüllt. Zur Runtime werden keine neuen Varianten mehr kompiliert (außer bei expliziter Precision-Revision). Das eliminiert Kompilierungs-Latenz im Hot-Path.

### 6.6 Runtime → Quality Monitor

```rust
// Installation von Probes beim Modell-Laden:
pub struct QualityProbeSet {
    pub probes: Vec<NodeId>,
    pub sample_rate: u32,
}

// Pro Check:
pub fn check_state(
    monitor: &mut QualityMonitor,
    node: NodeId,
    hidden_state: &Tensor,
) -> Option<PrecisionRevisionSignal>;

// Bei Revision-Signal:
pub fn apply_revision(
    runtime: &mut Runtime,
    signal: PrecisionRevisionSignal,
) -> Result<(), RevisionError>;
```

**Konvention.** `apply_revision` ist ein "soft restart" des Inference-Schritts ab dem betroffenen Layer. Der Runtime hält einen Ring-Puffer der letzten 4 Layer-Inputs in VRAM als Checkpoints vor.

### 6.7 Runtime ↔ Safety & Debug (Säule 6)

```rust
// Installation beim Modell-Laden:
pub struct ParityCheckerConfig {
    pub sample_rate: ParitySampleRate,
    pub tolerance_rel: f32,
    pub tolerance_abs: f32,
    pub on_violation: ViolationAction,
}

// Bei jedem WMMA-Call (wenn Sampling triggert):
pub fn check_parity(
    checker: &mut ValuParityChecker,
    wmma_kernel: &CompiledKernel,
    inputs: &[Tensor],
    wmma_output: &Tensor,
) -> Option<ParityViolation>;

// Bei Violation:
pub fn handle_violation(
    runtime: &mut Runtime,
    violation: ParityViolation,
) -> Result<(), SafetyError>;
```

**Konvention.** Säule 6 ist nicht Teil des linearen Datenflusses — sie sitzt "seitlich" am Runtime und greift nur bei Sampling-Triggern oder während GA-Evaluierung ein. Im Produktions-Modus ist sie praktisch kostenlos (1/1000 Sampling, < 1 % Overhead). Im GA-Evaluations-Modus ist sie Pflicht: ein Kernel-Kandidat mit Parity-Violation wird aus der Pareto-Front ausgeschlossen, egal wie schnell er ist.

### 6.8 Laufzeit-Fehler und -Fallbacks

Alle Säulen folgen dem Prinzip **"graceful degradation"**:

| Fehler | Fallback |
|---|---|
| Introspection schlägt fehl | Default-Profile, FP8-E4M3 überall |
| Fusion-GA konvergiert nicht | Regelbasierte Fusion |
| Kernel-GA findet keinen valid Kandidaten | Default-Genome (llama.cpp-ähnlich, FP16-Pfad) |
| Kompilierung schlägt fehl | nächstbester Genome aus Pareto-Front |
| Precision-GA schlägt fehl | Profile-Recommendation direkt nutzen |
| Quality-Monitor meldet FP8-Saturation | Precision-Hochstufung FP8 → FP16 für betroffene Schicht |
| Quality-Monitor meldet NaN | FP32-Fallback für ganze Layer-Gruppe |
| VALU-Parity-Violation (Produktion) | WMMA-Kernel für diese Shape deaktivieren, VALU-Pfad nutzen |
| VALU-Parity-Violation (GA-Phase) | Kandidat aus Pareto-Front entfernen, Fitness=-∞ |

Die Engine soll *nie* abstürzen wegen Optimierungsproblemen — nur suboptimal laufen. Das ist explizit wichtig, weil die GA-Komponenten lang dauern und User sie zur Not auch überspringen können sollen (`--no-autotune`).

---

## 7. Konkrete Walk-Through-Beispiele

### 7.1 Beispiel A — Neues Quant-Format Q5_K hinzufügen

**Szenario:** Die Community entwickelt ein neues Quant-Format Q5_K mit 5-Bit-Quantisierung, 32 Elementen pro Sub-Block, Int8-Scales und einem zusätzlichen `qh`-Teil für das 5. Bit.

**Was der Entwickler tut in v1.0:**

```rust
// Neue Datei: src/dequant_ir/formats/q5_k.rs
pub const Q5_K: QuantFormat = QuantFormat {
    id: 15,
    name: "Q5_K",
    block_bytes: 176,               // definiert durch Format-Spec
    elements_per_block: 256,
    sub_blocks_per_block: 8,
    sub_block_size: 32,
    block_scale_offset: 0,          // d: FP16 an Offset 0
    block_scale_type: ScalarType::Fp16,
    block_min_offset: Some(2),      // dmin: FP16 an Offset 2
    block_min_type: Some(ScalarType::Fp16),
    sub_scales_layout: SubScalesLayout::Packed6Bit {
        offset: 4,
        count: 16,                  // 8 scales + 8 mins, je 6 Bit
        unpack_program: q5_k_unpack_scales(),
    },
    dequant_program: vec![
        // Byte-Layout: qs[128] low 4-bit, qh[32] high 5. bit, scales[12]
        DequantOp::LoadBytes { offset: 16, count: 128, reg: r_qs },
        DequantOp::LoadBytes { offset: 144, count: 32, reg: r_qh },
        DequantOp::ExtractNibble { src: r_qs, high: false, dst: r_low4 },
        DequantOp::Combine5Bit { low4: r_low4, qh: r_qh, shift: 0, dst: r_q5 },
        DequantOp::IntToFloat { src: r_q5, offset: 0, dst: r_qf },
        DequantOp::FmaF32 { a: r_scale, b: r_qf, c: r_dmin_neg, dst: r_val },
        DequantOp::DowncastToHalf { src: r_val, dst: r_half, target: HalfType::Fp16 },
        DequantOp::StoreHalf { src: r_half, lds_offset_expr: "i*TILE_N+col".into() },
    ],
};

// In der Registry:
// src/dequant_ir/registry.rs
pub fn all_formats() -> &'static [QuantFormat] {
    &[Q4_0, Q4_K, Q6_K, Q8_0, Q5_K /* ← neu */]
}
```

**Das ist der gesamte Code-Beitrag.** Rund 30–40 Zeilen, strukturell identisch zu den bestehenden Format-Definitionen. Zusätzlich braucht es:

- Eine neue `Combine5Bit`-DequantOp, falls sie noch nicht existiert (5 Zeilen für Op-Definition + Codegen-Eintrag).
- Tests: Dequant-Korrektheit gegen llama.cpp-Referenz (wie in v0.x).

**Was automatisch passiert:**

1. Beim nächsten `cargo build`: der GPU- und CPU-Emitter erzeugen WMMA- und GEMV-Kernel für Q5_K — **inklusive FP8-, FP16- und BF16-Varianten aus derselben Format-Definition**. Das wird durch Makro-Expansion über `all_formats()` getrieben — kein manueller Build-Eintrag nötig. Keine zusätzliche Arbeit für den FP8-Pfad.
2. Beim ersten Laden eines Q5_K-Modells: der Entwickler ruft `rf-forge tune-all --model ~/models/X-q5km.gguf` auf (Cold Start ~2,5 h). Top-5-Varianten aller Shapes landen im Cache.
3. Precision-GA und Fusion-GA brauchen keine Änderung — sie arbeiten auf dem Graph-Level, nicht auf dem Format-Level. Der FP8-Default-Pfad gilt auch für Q5_K automatisch.
4. VALU-Parity-Check (Säule 6) validiert alle generierten Kernel vor ihrer Aufnahme in die Pareto-Front — keine stillen Numerik-Bugs.

**Vergleich mit v0.x:**

| Aufgabe | v0.x | v1.0 |
|---|---|---|
| Format-Spec dokumentieren | 1 Tag | 1 Tag (unverändert, nötig für Korrektheit) |
| WMMA-Kernel FP16 schreiben | 2 Tage | **0 Min** (generiert) |
| WMMA-Kernel FP8 schreiben | (wäre 2 weitere Tage) | **0 Min** (automatisch aus Spec) |
| GEMV-Kernel schreiben | 1 Tag | **0 Min** (generiert) |
| Fused-Gate-Up-Swiglu-Kernel | 1 Tag | **0 Min** (generiert) |
| Fused-Residual-Norm-Kernel | 0,5 Tage | **0 Min** (generiert) |
| Dispatch-Logik anpassen | 0,5 Tage | **0 Min** (automatisch über Registry) |
| Debugging & Numerik-Fixes | 2–3 Tage | **0,5 Tag** (VALU-Parity + Quality-Monitor fangen Bugs) |
| **Gesamt** | **10–12 Tage** | **~1,5 Tage** |

Der Hauptnutzen ist nicht die 6–8× Zeitersparnis, sondern dass die Bug-Fläche drastisch kleiner wird: statt sieben neuer Kernel mit potentiellen Indexing-, Scaling- und Rundungsfehlern (und FP8 würde diese Anzahl in v0.x verdoppeln), gibt es nur noch eine deklarative Datenstruktur zu überprüfen. Der VALU-Parity-Check (Säule 6) fängt die klassischen Numerik-Bugs automatisch ab.

### 7.2 Beispiel B — Neues Modell Llama-4 laden

**Szenario:** Meta veröffentlicht Llama-4-8B als GGUF, Q4_K_M-Quantisierung. Die Architektur variiert leicht: andere Head-Dimensionen, neue Tokenizer-Spezial-Tokens, vielleicht Grouped-Query-Attention mit neuer Group-Size.

**Was der Entwickler tut:** **nichts** in den meisten Fällen.

**Was automatisch passiert:**

1. GGUF-Parser liest die Metadaten:
   ```
   general.architecture = "llama"
   llama.block_count = 32
   llama.attention.head_count = 32
   llama.attention.head_count_kv = 8   (GQA 4:1)
   llama.attention.layer_norm_rms_epsilon = 1e-5
   llama.rope.freq_base = 500000.0
   llama.rope.scaling.type = "llama3"
   llama.rope.scaling.factor = 8.0
   ```

2. Graph-Builder matcht `general.architecture = "llama"` und wählt das `llama_template`. Alle Parameter werden automatisch aus den Metadaten in die `OpNode`-Felder propagiert. Die neue RoPE-Variante `Llama3 { factor: 8.0, ... }` existiert bereits als `RopeVariant` (weil Llama-3 sie eingeführt hat); falls nicht, wird eine Warnung ausgegeben und die Standard-RoPE als Fallback genutzt.

3. Model Introspection läuft und entdeckt Llama-4-spezifische Eigenheiten — vielleicht neue kritische Special-Tokens. Das wird im `ModelProfile` abgelegt.

4. Kernel-Cache wird geprüft: Shapes für 32 Head-Heads, Head-Dim 128, GQA-4:1 wurden vielleicht schon für Llama-3 durchgerechnet (identische Shape-Signatur) → Cache-Hit. Falls nicht → Kernel-GA läuft automatisch.

5. Precision-GA läuft (15 min, spezifisch für diese Modell-Instanz).

6. Chat startet nach 6 s (Warm) oder ~2,5 h (Cold).

**Fall: Llama-4 bringt eine fundamental neue Architektur-Komponente.** Z. B. einen neuen Normalisierungs-Typ "GeLUNorm". Dann muss im Graph-Builder ein neuer `Operator::GeLUNorm` + Template-Eintrag hinzugefügt werden, plus der entsprechende Dequant-IR-Operator im Codegen. Das ist ein ~1-Tag-Eingriff und betrifft nur diese eine Komponente — alle anderen Kernel bleiben unberührt.

**Vergleich mit v0.x:** in v0.x wäre jedes neue Llama-Modell eine Auditierung aller Dispatch-Pfade (hardcodierte Head-Counts, RoPE-Varianten im `ModelTraits`-Enum, etc.). Realistischer Aufwand 1–2 Tage, selbst bei "gleicher Architektur, nur andere Gewichte". In v1.0 sind es Minuten, weil Metadaten-Parsing und Graph-Konstruktion modell-agnostisch sind.

### 7.3 Beispiel C — Neues Hardware-Target RDNA 5 (hypothetisch)

**Szenario:** AMD veröffentlicht RDNA 5 mit neuer `gfx1301`-Architektur, erweiterten WMMA-Instruktionen (FP8-Support) und größerem LDS.

**Was ändern muss:**

- **Hardware-Konstanten** (`src/hw/gfx1301.rs`): neue Constraints (LDS-Größe, VGPR-Budget, WMMA-Typen).
- **GPU-Codegen-Backend** (`src/dequant_ir/codegen/gpu/gfx1301.rs`): neue Instruktionen emittieren, neue Intrinsics nutzen.
- **Kernel-GA**: die Genome-Validierung referenziert jetzt `GFX1301` statt `GFX1201`.

**Was identisch bleibt:**

- **Alle `QuantFormat`-Definitionen** (Q4_K, Q6_K, ...): unverändert — das sind format-, nicht hardware-spezifische Beschreibungen.
- **Computation Graph**: vollständig hardware-agnostisch.
- **Dequant-IR**: die `DequantOp`-Enum bleibt dieselbe; nur der Emitter ändert sich.
- **Fusion-Pass-Logik**: unverändert.
- **Model Introspection**: unverändert.
- **Self-Tuning Runtime (Bandit)**: unverändert.
- **Quality Monitor**: unverändert.
- **GA-Logik (Selection, Crossover, Mutation)**: unverändert.

**Aufwand-Schätzung für RDNA-5-Support:**

| Komponente | Aufwand |
|---|---|
| Hardware-Konstanten-Datei | 0,5 Tage |
| GPU-Codegen-Emitter | 1 Woche (neue Instruktionen, Register-Layout) |
| Kernel-GA-Anpassung | 0,5 Tage |
| Testing & Bench auf realer Hardware | 1 Woche |
| **Gesamt** | **~2,5 Wochen** |

Entscheidend ist: der Großteil der Engine bleibt unangetastet. In v0.x wäre jeder Kernel einzeln neu zu schreiben (Dutzende von `.hip`-Dateien); in v1.0 ist es ein einziger Emitter, der alle Formate abdeckt.

**Anmerkung:** v1.0 ist *nicht* für multi-hardware-concurrent-operation designed. Jeder Build ist für **ein** Target. Wer RDNA 4 und RDNA 5 unterstützen will, baut zwei getrennte Binaries.

---

## 8. Migration von v0.x

### 8.1 Was übernommen wird

Die v0.x-Basis enthält mehrere Artefakte, die in v1.0 weitergenutzt werden:

**Kernel als Backend-Referenz:**

| v0.x-Artefakt | v1.0-Verwendung |
|---|---|
| `hip_kernels/wmma/wmma_gemm_q4_0.hip` | Referenz-Implementierung, gegen die der generierte Q4_0-Kernel validiert wird |
| `hip_kernels/wmma/wmma_gemm_q4_k_m.hip` | dto. für Q4_K |
| `hip_kernels/wmma/wmma_gemm_q6_k.hip` | dto. für Q6_K |
| `hip_kernels/quant/q4_k_m_gemv.hip` | Referenz für GEMV-Codegen |
| `hip_kernels/quant/q6_k_gemv.hip` | dto. |

Diese Kernel werden nicht direkt weiterverwendet, sondern dienen als **Goldstandard**: der generierte Kernel muss für gleiche Inputs bit-identische Outputs liefern (innerhalb FP-Toleranz). Das ist eine harte Test-Anforderung, die sicherstellt, dass der Codegen korrekt arbeitet.

**Dokumentation** (wird direkt nach `docs/` übernommen, ggf. in `docs/legacy/` für Historie):

- `docs/wmma_register_layout_gfx12.md` — WMMA-Register-Layout-Spezifikation, zentrale Referenz für den GPU-Codegen.
- `docs/q4_k_m_block_format.md` — byte-genaues Q4_K-Block-Layout, dient als Vorlage für die `QuantFormat::Q4_K`-Definition.
- `docs/q6_k_block_format.md` — dto. für Q6_K.
- `profiling/results/phase8_q4_k_m_analysis.md` — Performance-Analyse-Methodik, wird zur Validierung der v1.0-Performance verwendet.

**Test-Infrastruktur:**

- Tokenizer-Parity-Tests (v0.x validiert Tokenizer gegen `tokenizer.json`-Referenz).
- Numerik-Validierung gegen llama.cpp-Outputs für einen festen Prompt-Satz.
- Perplexity-Messung auf WikiText-2.

**GGUF-Parser:**

- Die v0.x-GGUF-Parser-Komponente (`src/gguf/`) wird vollständig übernommen — sie ist modell-agnostisch und testfest.

**HTTP-Server und CLI:**

- Die OpenAI-kompatible HTTP-Schicht (`src/server/`) und die CLI (`src/cli/`) werden übernommen. Sie interagieren mit der Inference-Engine über ein stabiles Interface, das in v1.0 identisch bleibt.

### 8.2 Was nicht übernommen wird

**Handgeschriebene Fused-Kernel:**

- `hip_kernels/fused/fused_norm_qkv_q4_k.hip`, `fused_gate_up_swiglu_q4_k.hip`, `fused_residual_norm_q4_k.hip`, usw. — alle Fused-Varianten werden durch Graph-Fusion + Codegen ersetzt.

**Hardcodierte Dispatch-Logik:**

- `src/gpu/ops.rs::gpu_dispatch_gemm` mit seinen Match-Armen für jedes Quant-Format wird durch den Graph-basierten Ausführungs-Runner ersetzt.

**Modell-spezifische Config:**

- `ModelTraits`-Enum mit hardcodierten Werten pro Architektur (Llama-Style, Qwen-Style, Mistral-Style) wird durch Template-Lookup + Metadaten-Propagation ersetzt.

**Ad-hoc Precision-Fixes:**

- Die in Phase 8a eingeführten FP32-Skalen-Akkumulations-Flags werden durch die systematische Precision-Allocation-GA ersetzt. Keine hardcodierten Overrides mehr.

**Manuelles Profiling-Tooling:**

- Das manuelle `profile-*`-Binary-Ökosystem (mehrere Phasen von Profiling-Scripts) wird durch die integrierte Bandit-Runtime ersetzt. Profiling wird zum First-Class-Feature der Engine selbst.

### 8.3 Migration-Strategie

Die Migration erfolgt **nicht** als Big-Bang-Rewrite. Stattdessen läuft v0.x weiter (als `rocmforge-legacy`-Binary), während v1.0 in einem Subverzeichnis `rocmforge-v1/` parallel entwickelt wird. Sobald v1.0 Feature-Parität mit v0.x erreicht (Phase 3 der Roadmap), wird v1.0 zum Default; v0.x wird ein weiteres Release für Sicherheits-Fixes unterstützt und dann archiviert.

Das Repository-Layout während der Übergangszeit:

```
ROCmForge/
├── src/                   # v0.x (aktuell)
├── rocmforge-v1/
│   ├── src/               # v1.0 (neu)
│   │   ├── introspection/
│   │   ├── graph/
│   │   ├── dequant_ir/
│   │   ├── runtime/
│   │   ├── quality/
│   │   └── hw/
│   └── Cargo.toml
├── docs/
│   ├── architecture_v1.0.md   # dieses Dokument
│   ├── legacy/                # v0.x-Docs
│   └── ...
├── hip_kernels/               # v0.x-Kernel als Referenz
└── tests/                     # gemeinsame Tests, wo möglich
```

Gemeinsame Tests (Tokenizer, GGUF-Parser) testen beide Implementierungen und verlangen identische Outputs — das ist die praktische Sicherheit während der Migration.

---

## 9. Roadmap mit Meilensteinen

### 9.0 Meilenstein 0 — FP8 Go/No-Go Gate (Tag 1, ~2–4 Stunden)

**Zweck.** Bevor die 6-wöchige Entwicklung startet, wird die zentrale Hardware-Annahme der gesamten v1.2-Architektur verifiziert: **funktioniert `v_wmma_f32_16x16x16_fp8_fp8` auf `gfx1201` korrekt?** Falls nicht, muss die Strategie sofort auf FP16-WMMA umgestellt werden — vor Phase 1, nicht mitten in Phase 2.

**Implementierung.** Ein isoliertes Rust-Crate (`rocmforge-smoke`), das:

1. Eigene `bindgen`-Bindings gegen ROCm 7.2 generiert (Section 3.8).
2. Einen handgeschriebenen HIP-Kernel kompiliert und lädt (`hipModuleLoad`).
3. Die FP8-WMMA-Intrinsic ausführt und gegen Referenzwerte prüft.

**Test-Architektur (5 Tests, aufeinanderaufbauend):**

```
Test 1: Intrinsic-Kompilierbarkeit           ← HARD GATE
  → hipcc --offload-arch=gfx1201 kompiliert einen Kernel mit
    __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32
  → Falls Compiler-Fehler: STOP. FP8 nicht verfügbar.
    → Fallback: FP16-WMMA. Performance-Ziele: Prefill halbiert.

Test 2: Numerische Korrektheit (50 Runs, diverse Inputs)
  → WMMA-Output vs. CPU-Referenz (Rust FP32-Matmul)
  → Max relative error ≤ 1e-3 (inherente FP8-Ungenauigkeit)
  → Input-Diversität: Normalverteilt, Uniform, Near-Zero, Near-Saturation,
    Mixed-Magnitude, reale Q4_K-Dequant-Gewichte
  → Zusätzlich: WMMA vs. VALU-Referenz (max rel error ≤ 1e-5)

Test 3: Stability (1000 Runs, Timeout 100ms/Run)
  → Kein Hang, kein Crash, kein GPU-Reset
  → Poll-Loop mit exponential Backoff statt blocking Sync

Test 4: Sustained Load (60 Sekunden, Parity alle 5s)
  → Keine Drift über Zeit (thermische Stabilität)

Test 5: Edge-Cases
  → ±448 (E4M3 Saturation), ±57344 (E5M2 Saturation)
  → NaN-Propagation, Subnormal-Handling
  → Alle-Null, Alle-Max, Alternierende Vorzeichen
```

**Entscheidungsmatrix:**

| Test-Ergebnis | Entscheidung | Auswirkung auf Architektur |
|---|---|---|
| Test 1 PASS, alle weiteren PASS | **GO FP8** | v1.2-Architektur wie geplant |
| Test 1 PASS, Test 2 FAIL (Numerik) | **GO FP16 + Investigation** | FP16-WMMA als Default, FP8 als experimentell markiert |
| Test 1 PASS, Test 3 FAIL (Stability) | **GO FP16 + ROCm Bug-Report** | FP8 deaktiviert bis Treiber-Fix |
| Test 1 FAIL (Compiler) | **NO-GO FP8, GO FP16** | Performance-Ziele Prefill halbiert; Architektur ansonsten identisch |

**FP16-Fallback-Pfad.** Falls FP8 nicht verfügbar ist, ändert sich:

- `target_dtype` im Codegen: `Fp8E4M3` → `Fp16`
- WMMA-Intrinsic: `..._fp8_fp8_w32` → `..._f16_w32`
- VGPR-Budget pro Fragment: 4 → 8 (doppelt, niedrigere Occupancy)
- Prefill-Ziele: ~50 % der FP8-Werte
- Decode-Ziele: **identisch** (bandbreiten-gebunden, nicht von WMMA-Precision abhängig)
- Dequant IR: `DowncastToFP8` → `DowncastToHalf { target: Fp16 }`
- KV-Cache: FP16 statt FP8-E5M2 (doppelter Speicherbedarf, max Kontext halbiert)
- Alles andere (Graph, Fusion, GA, Bandit, Quality Monitor, VRAM Arena): **unverändert**

Der Fallback-Pfad ist kein "Plan B" — er ist die v1.0-Draft-Architektur ohne die v1.1-FP8-Erweiterung. Funktionsfähig, getestet im Konzept, nur langsamer bei Prefill und enger im VRAM.

**Lieferobjekt.** Das `rocmforge-smoke`-Crate bleibt im Repository als permanenter Regression-Test. Bei jedem ROCm-Update (`pacman -Syu`) wird es als erster Check ausgeführt.

### 9.1 Phase 1 — Fundament mit FP8-Pfad (3 Wochen)

**Ziel.** End-to-End-Inferenz mit dem neuen Graph + Dequant IR + Runtime, **mit FP8-WMMA-Pfad als Default**. Die Performance ist zu Beginn schlechter als v0.3.0 (Default-Genomes, keine Fusion-GA), aber die Architektur funktioniert durchgehend und der FP8-Pfad steht als Grundpfeiler. FP8 ist **Kern der v1.0**, kein nachträgliches Feature — daher in Phase 1, nicht Phase 2.

**Deliverables.**

| Woche | Deliverable | Akzeptanz |
|---|---|---|
| 1 | Computation-Graph-Modul | GGUF → Graph für Qwen3, Llama-3.1; unit-getestet |
| 1 | Dequant-IR-Datentypen | `QuantFormat`, `DequantOp` (inkl. `LoadFP8`/`DowncastToFP8`), `CompiledKernel` definiert; Q4_0 und Q4_K als Spec |
| 2 | GPU-Codegen für Q4_0, Q4_K — **FP16 UND FP8** | FP16-Pfad bit-identisch zu v0.x (Toleranz 1e-4); FP8-Pfad gegen VALU-Referenz innerhalb 2⁻¹⁰ relativ |
| 2 | Regelbasierte Fusion-Passes | `FusedResidualNorm`, `FusedNormQkv`, `FusedGateUpSwiglu` erzeugt |
| 2 | VALU-Referenz-Codegen | jeder WMMA-Kernel kommt mit VALU-Zwilling (Säule 6) |
| 3 | Runtime-Runner ohne Bandit | Graph-Execution, Default-Variant pro Shape, FP8-Pfad aktiv |
| 3 | Model Introspection (statisch) | `ModelProfile` korrekt für Qwen3-8B, Llama-3.1-8B |
| 3 | End-to-End Decode Qwen3-8B (FP8-Pfad) | ≥ 60 tok/s, Output-Qualität innerhalb 3 % Perplexity-Regression zu v0.3.0 |

**Risiken.**

- FP8-Rundung führt zu Qualitäts-Regressions bei bestimmten Layern → VALU-Parity-Check lokalisiert Fälle, Precision-GA (Phase 2) wird sie auflösen.
- Codegen-Komplexität mit zwei Pfaden (FP8 + FP16) höher als ein Pfad → Puffer von 3 Tagen pro Woche eingeplant.
- Quality-Monitor (Säule 5) wird in dieser Phase noch nicht aktiviert — Numerik-Tests basieren auf statischen Referenz-Outputs und VALU-Parity.

**Exit-Kriterium Phase 1.** v1.0-Binary startet, lädt Qwen3-8B Q4_K_M, generiert kohärente Texte via FP8-WMMA, erreicht ≥ 60 tok/s Decode. Keine GA-Optimierung. VALU-Parity-Check schlägt bei keinem Standard-Kernel fehl.

### 9.2 Phase 2 — Optimierung und Safety-Schicht (2 Wochen)

**Ziel.** GA-Integration, Caching, Safety & Debug vollständig. Performance auf ≥ 100 tok/s Decode bringen. Q6_K-Format wird ergänzt.

**Deliverables.**

| Woche | Deliverable | Akzeptanz |
|---|---|---|
| 4 | Kernel-Auto-Tuning-GA | läuft auf einer Shape, produziert Top-5-Varianten in ≤ 10 min, mit VALU-Parity-Validierung jedes Kandidaten |
| 4 | Self-Tuning Runtime (Bandit/UCB1) | konvergiert zur besten Variante innerhalb 100 Calls |
| 4 | Fusion-Pass-GA | findet valide Fusions-Kombinationen unter 104-VGPR-Constraint |
| 4 | Säule 6 Safety & Debug vollständig | `ValuParityChecker` im Produktions-Pfad (1/1000 Sampling), Violation-Handling |
| 5 | Precision-Allocation-GA (NSGA-II) mit FP8-Suchraum | Pareto-Front für Qwen3-8B in 15 min gefunden; FP8→FP16→BF16 als Fallback-Kette |
| 5 | Quality Monitor mit FP8-Saturation-Detection | Llama-3.1 Multi-Turn funktioniert ohne hardcodierten Override |
| 5 | Q6_K-Format-Definition | Q4_K_M (Mixed mit Q6_K-Layern) funktioniert im FP8-Pfad |
| 5 | Cache-System (vollständig) | zweiter Start eines bereits getunten Modells < 10 s |
| 5 | End-to-End Decode Qwen3-8B | ≥ 100 tok/s mit GA + FP8, ≥ 5 000 tok/s Prefill |

**Risiken.**

- GA-Konvergenz-Zeit länger als erwartet → GA-Parameter adaptiv tunen, frühes Abbruch-Kriterium einbauen.
- VALU-Parity-Check zu streng bei FP8 (Rundungsunterschiede) → Toleranzen pro Shape kalibrieren, nicht global.
- KV-Cache-FP8 könnte bei sehr langen Kontexten (> 4k) Drift zeigen → Quality Monitor im Kontext-Sampling-Mode.

**Exit-Kriterium Phase 2.** v1.0 erreicht Qwen3-8B Q4_K_M Decode ≥ 100 tok/s, Prefill ≥ 5 000 tok/s. Llama-3.1-8B Multi-Turn ohne Precision-Override funktioniert. VALU-Parity-Check erkennt eingebauten Bug-Testcase korrekt.

### 9.3 Phase 3 — CPU-Backend, rf-forge CLI, Release (1 Woche)

**Ziel.** CPU-Backend aktivieren, `rf-forge`-CLI als offizielles Tuning-Tool, Performance auf v1.1-realistisches Niveau bringen (125 tok/s Decode, 7 500 tok/s Prefill), Release-Vorbereitung.

**Deliverables.**

| Tag | Deliverable | Akzeptanz |
|---|---|---|
| 36 | AVX-512-Codegen-Emitter | CPU-Kernel für Q4_K GEMV erzeugt, korrekte Outputs, Zen4-Double-Pumped getestet |
| 37 | VNNI-Pfad für Q8-Dot | 2× Speedup gegenüber FP-Pfad auf Q8-Activation |
| 38 | CPU-only Inferenz (`--gpu false`) | Qwen3-8B läuft end-to-end auf CPU, ≥ 8 tok/s Decode |
| 39 | `rf-forge` CLI-Binary | `tune-all`, `tune-kernels`, `tune-precision`, `cache`, `validate`, `bench` implementiert |
| 39 | Hybrid-Dispatch (GA auf CPU, Inference auf GPU) | GA läuft während Inference ohne GPU-Kontention |
| 40 | Performance-Bench-Suite | Qwen3, Llama-3.1, Qwen2.5 via `rf-forge bench` dokumentiert |
| 41 | Release-Notes & Migrations-Guide | `docs/release_v1.0.md` fertig, Upgrade-Pfad beschrieben |
| 42 | v1.0.0-rc1 Tag | Agent fertig, nur manuelle Bestätigung pending |

**Exit-Kriterium Phase 3.** Benchmark-Tabelle reproduzierbar (realistische Ziele aus Abschnitt 1.4):

**Klasse 8B (Decode bandbreiten-begrenzt bei ~143 tok/s):**

| Modell | Decode | Prefill pp256 |
|---|---:|---:|
| Qwen3-8B Q4_K_M | ≥ 125 tok/s | ≥ 7 500 tok/s |
| Llama-3.1-8B Q4_K_M | ≥ 125 tok/s (stabil Multi-Turn) | ≥ 7 500 tok/s |
| Qwen2.5-7B Q4_0 | ≥ 140 tok/s | ≥ 8 500 tok/s |

**Klasse 14B (Decode bandbreiten-begrenzt bei ~72 tok/s):**

| Modell | Decode | Prefill pp256 | Max Kontext (16 GB, FP8-KV) |
|---|---:|---:|---:|
| Qwen2.5-14B Q4_K_M | ≥ 63 tok/s | ≥ 4 000 tok/s | ≥ 32k |

**Hinweis:** 14B @ 63 tok/s ist kein Rückschritt, sondern physikalisch nahe am Optimum (87 % BW-Effizienz). llama.cpp erreicht auf derselben Hardware ~45 tok/s für 14B (geschätzt) — v1.0 wäre damit +40 % schneller.

**Falls Meilenstein 0 den FP16-Fallback aktiviert:** Prefill-Ziele halbieren sich; Decode bleibt identisch; 14B max Kontext sinkt auf ~16k (FP16-KV-Cache, doppelter Speicher).

Qualitäts-Check: Perplexity auf WikiText-2 innerhalb 2 % der FP16-Referenz.
VALU-Parity-Regression-Check: 0 Violations über 10 000 Inference-Steps.

### 9.4 Gesamt-Aufwand und Puffer

- Summe der aktiven Entwicklung: 6 Wochen (3 + 2 + 1).
- Enthaltener Puffer: ~20 % (3 Tage pro Woche Phase 1, 2 Tage pro Woche Phase 2).
- Nicht enthalten: externe Blocker (ROCm-Bugs, neue GGUF-Varianten, unerwartete Hardware-Quirks). Diese werden ad-hoc adressiert.

### 9.5 Abhängigkeiten zwischen Phasen

```
Phase 1 [Fundament]
   │
   ├─► Phase 2.Kernel-GA (braucht Codegen + Runtime)
   ├─► Phase 2.Fusion-GA (braucht Graph + Regel-Fusion)
   ├─► Phase 2.Precision-GA (braucht Runtime + End-to-End Inferenz)
   │
   └─► Phase 3.CPU-Backend (braucht Dequant-IR + Codegen-Emitter-Struktur)
```

Phase 2 kann intern parallelisiert werden: Kernel-GA, Fusion-GA und Precision-GA haben unabhängige Datenstrukturen. Wenn zwei Entwickler zur Verfügung stehen, spart das bis zu 4 Tage.

---

## 10. Schlüssel-Erkenntnisse aus v0.x

Die Architektur-Entscheidungen in v1.0 sind keine theoretischen Ideale, sondern Antworten auf konkrete Probleme, die in der v0.x-Entwicklung aufgetreten sind. Diese Sektion macht die kausale Beziehung zwischen Erkenntnis und Design-Entscheidung explizit.

### 10.1 "Profiling invertiert IMMER die Schätzungen"

**Beobachtung.** In den Phasen 4, 6, 7 und 8 der v0.x-Entwicklung wurden jedes Mal Performance-Hypothesen vor dem Profiling aufgestellt ("der Q4_K-GEMV ist LDS-bound, weil…") und jedes Mal hat das Profiling die Hypothese widerlegt oder stark relativiert. In Phase 7 stellte sich heraus, dass Direct-Global-Loads **5× schneller** sind als das aufwendig optimierte LDS-Staging für eine bestimmte Shape-Klasse — ein Ergebnis, das keine Lehrbuch-Heuristik vorhergesagt hätte.

In Phase 8b zeigte sich, dass der Gate-Up-Swiglu-Kernel nach Q8-Inline **LDS-bound statt ALU-bound** wurde — wieder ein Shift des Bottlenecks, der manuelle Re-Analyse erforderte.

**Design-Konsequenz.** Die Self-Tuning Runtime (Säule 4, UCB1) und der Kernel-Auto-Tuning-GA (Säule 4) existieren als direkte Antwort: wir vertrauen der Theorie nicht mehr, sondern messen. Die GA erkundet systematisch den Konfigurationsraum; der Bandit wählt zur Laufzeit die zustandsabhängig beste Variante. Kein Bauchgefühl, keine Heuristik, keine statische Dispatch-Tabelle.

**Praktische Implikation.** Die GA findet Konfigurationen, die ein menschlicher Entwickler verwerfen würde ("Direct Global statt LDS? Das ist doch kontraintuitiv!"). Das ist Feature, nicht Bug. GAs haben keine Vorurteile.

### 10.2 "O(Modelle × Quants × Fusions) skaliert nicht"

**Beobachtung.** In v0.x gab es bei Abschluss von Phase 8b separate Kernel-Dateien für:

- 4 Quant-Formate (Q4_0, Q4_K, Q6_K, Q8_0)
- × 3 Operator-Typen (WMMA-GEMM Prefill, GEMV Decode, Fused Gate-Up-Swiglu)
- × 4 Fused-Varianten (Norm+GEMV, Residual+Norm, Gate+Up, ...)
- ≈ 48 Kernel-Dateien

Jedes neue Quant-Format Q5_K oder Q5_K_M würde 12 neue Kernel-Dateien bedeuten. Jede neue Fused-Variante (z. B. GEMV-mit-integriertem-RoPE) würde pro Quant-Format einen neuen Kernel brauchen. Der Aufwand ist multiplikativ.

**Design-Konsequenz.** Das Dequant IR (Säule 3) trennt die Dequantisierung (format-spezifisch) von der GEMM/GEMV-Logik (generisch). Der Computation Graph + Fusion-Passes (Säule 2) trennen die Fusions-Entscheidung (graph-level) von der Kernel-Implementierung. Das Ergebnis: Formate und Fusionen werden orthogonal; der Aufwand ist additiv, nicht multiplikativ.

**Praktische Implikation.** Ein neues Quant-Format hinzufügen ist eine `QuantFormat`-Struct-Definition (~30 Zeilen), nicht 12 Kernel-Dateien. Eine neue Fusion ist ein Graph-Transformations-Pattern, kein format-spezifischer Kernel.

### 10.3 "Llama-3.1 Special Tokens haben SNR < 1 bei Q4_K"

**Beobachtung.** Llama-3.1-8B Q4_K_M im Multi-Turn-Chat produzierte Garbage. Einzelne Prompts und sogar lange nicht-template-formatierte Prompts funktionierten. Der Unterschied war die Template-Formatierung mit Special Tokens (`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`). Diese Tokens haben Embedding-L2-Normen um `0.034`, während normale Text-Tokens bei `0.55` liegen. Das Q4_K-Dequant-Rauschen beträgt etwa `0.064` L2 über 4096 Dimensionen — **größer als das Signal selbst**. Über 32 Layer eskaliert der Fehler.

Der Bug war nicht modell-spezifisch, sondern **numerik-spezifisch**. Er wurde in v0.x durch FP32-Skalen-Akkumulation (Phase 8a) gepatcht, aber das ist ein hardcodierter Override — nicht skalierbar für andere Modelle oder Formate.

**Design-Konsequenz.** Zwei Säulen adressieren dieses Problem systematisch:

1. **Model Introspection (Säule 1)** scannt die Gewichte beim Laden und identifiziert Tokens/Layer mit ungewöhnlich kleinen Magnituden. Das `ModelProfile` meldet SNR-Risk < 2.0 und schlägt gezielt höhere Precision für betroffene Layer/Tokens vor.

2. **Precision-Allocation-GA (Säule 4)** nutzt diese Hinweise als Startpunkt und findet eine Pareto-optimale Präzisions-Verteilung, die Qualität (KL-Divergenz zu FP32) gegen Speed (tok/s) abwägt.

**Praktische Implikation.** Der Llama-3.1-Bug verschwindet in v1.0 vermutlich automatisch: Introspection erkennt die kleinen Magnituden, Precision-GA allokiert BF16 oder FP32 für die Embedding-Dequant, Quality-Monitor fängt Rest-Drift während der Inference ab. Kein Format-spezifischer Override mehr nötig.

### 10.4 "Direct-Global 5× schneller als LDS für bestimmte Shapes"

**Beobachtung.** In Phase 7 der v0.x-Entwicklung wurde ein WMMA-GEMV-Kernel mit sorgfältig designtem LDS-Staging implementiert. Ein Profiling-Experiment, das den LDS-Pfad deaktivierte und stattdessen direkt aus Global Memory lud, brachte einen **5× Speedup** für bestimmte Shapes (schmale GEMV, wo die Wiederverwendung der LDS-Daten gering ist). Das widerspricht der klassischen GPU-Optimierungs-Lehre ("LDS > Global").

**Design-Konsequenz.** Der Kernel-Auto-Tuning-GA (Säule 4) behandelt `use_lds_for_a` und `use_lds_for_b` als Boolean-Gene. Er findet daher Konfigurationen, die Lehrbuch-Heuristiken nie vorschlagen würden. Der Bandit zur Laufzeit schließt den Loop, indem er die GA-gefundenen Varianten gegen die tatsächliche Hardware-Realität testet.

**Praktische Implikation.** v1.0 wird voraussichtlich Kernel-Konfigurationen produzieren, die in einem Paper als "falsch" eingestuft würden — aber messbar schneller sind. Das ist OK. Korrektheit wird durch Bit-Identität gegen Referenz-Kernel validiert; Performance wird durch Messung validiert.

### 10.5 "v0.x Kernel sind Gold, Architektur drumherum ist das Problem"

**Beobachtung.** Bei der Evaluation von v0.x für v1.0 wurde klar, dass die handgeschriebenen WMMA-Kernel (Q4_0, Q4_K, Q6_K) tatsächlich **weltklasse** sind — sie erreichen 40–50 % der theoretischen Bandbreite auf `gfx1201`, was auf einer Architektur ohne hipBLAS-WMMA-Support außergewöhnlich ist. Die Dokumentation (Register-Layout, Block-Formate) ist ebenfalls extrem wertvoll und nirgendwo anders in der öffentlichen AMD-Welt in dieser Qualität verfügbar.

Das eigentliche Problem von v0.x war nicht die Kernel-Qualität, sondern die **Architektur drumherum**: hardcodierte Dispatch-Logik, modell-spezifische Config-Enums, handgeschriebene Fused-Varianten, keine systematische Präzisions-Behandlung.

**Design-Konsequenz.** v1.0 wirft die Kernel nicht weg. Sie werden als **Referenz-Backend** migriert: der Dequant-IR-Codegen generiert Kernel, die bit-identisch zu den v0.x-Kerneln sind (modulo Precision-Revisions). Die Korrektheits-Tests vergleichen Codegen-Output mit den v0.x-Kerneln. Die 6+ Monate Arbeit an den Kerneln sind damit nicht verloren — sie bilden die Messlatte für v1.0.

**Praktische Implikation.** v1.0 startet nicht mit "schlechterer" Kernel-Qualität. Die erste Phase von v1.0 reproduziert exakt die v0.x-Performance; alle Verbesserungen (GA-Tuning, Fusion-GA, Precision-GA) addieren sich obendrauf. Im schlimmsten Fall ist v1.0 so gut wie v0.3.0; im realistischen Fall deutlich besser.

### 10.6 "Hardware-Specs ernst nehmen, auch spät im Prozess" (v1.1-Revision)

**Beobachtung.** Während der v1.0-Draft-Erstellung wurde die Zielplattform `gfx1201` mit **32 Compute Units** angenommen. Eine nachträgliche Hardware-Verifikation zeigte: die RX 9070 XT hat im Vollausbau **64 CUs mit 128 AI Accelerators** — das Doppelte. Zusätzlich hat RDNA 4 **native FP8-WMMA-Instruktionen (E4M3/E5M2)**, die im ursprünglichen Draft als Future-Feature eingestuft waren statt als Kern-Feature.

Das ist kein kleines Versehen — es verändert die Architektur-Strategie fundamental. Mit 32 CUs und nur FP16-WMMA war v1.0 als "defensive Universal-Architektur" gestaltet, die sich langsam dem llama.cpp-Niveau nähert. Mit 64 CUs und FP8 ist v1.1 eine "aggressive Hardware-Native-Architektur", die llama.cpp strukturell überholen *kann* — nicht nur in einzelnen Benchmarks, sondern in Peak-Durchsatz (FP8-WMMA ist schlicht 2× FP16-WMMA, und llama.cpp nutzt weder).

**Design-Konsequenz.** Die v1.1-Revision hat:
- Performance-Ziele nach oben kalibriert (Decode 125 statt 110 tok/s, Prefill 7 500 statt 4 000 tok/s)
- FP8 vom optionalen zum Default-Pfad gemacht (Säule 3)
- Den KV-Cache-FP8-Pfad als dedizierte GA-Dimension eingeführt (Abschnitt 4.3)
- Register-Pressure-Management trotz verdoppelter CU-Zahl bei 104 VGPRs festgehalten (Abschnitt 3.2) — mehr CUs erlauben es nicht, pro Wave mehr Register zu verschwenden

**Praktische Implikation.** Jeder Architektur-Draft sollte einen expliziten Hardware-Verifikations-Schritt haben, bevor Design-Entscheidungen mit hoher Trägheit fallen. Ein fehlerhafter CU-Count oder ein übersehener WMMA-Pfad kann die Strategie von "vorsichtig nachziehen" auf "strukturell überholen" verschieben — oder umgekehrt.

### 10.7 "Silicon ist nicht immer korrekt" (Safety & Debug als eigene Säule)

**Beobachtung.** In der v0.x-Entwicklung trat mehrfach das Problem auf, dass ein WMMA-Kernel mathematisch korrekt aussah, gegen llama.cpp-Outputs dennoch abwich, und die Fehlerlokalisierung Tage kostete (Phase 5, Phase 7). Im ROCm-Ökosystem sind WMMA-bezogene Driver-Regressions, Silicon-Errata und Toolchain-Bugs dokumentierte Realität (Wave32/Wave64-Mismatches auf gfx1100, Illegal-Opcode-Crashes, Composable-Kernel-Inkompatibilitäten). Die einzige zuverlässige Debug-Methode war stets: eine zweite unabhängige Implementierung zur Parity-Validierung.

**Design-Konsequenz.** In v1.1 wird dies zur ersten Klasse befördert: **Säule 6 (Safety & Debug)** mit dem VALU-Parity-Pfad. Jeder WMMA-Kernel bekommt einen automatisch generierten VALU-Zwilling; die Runtime sampelt im Produktions-Pfad (1/1000) und im Debug/GA-Pfad (1/1), und Abweichungen werden als `ParityViolation` geloggt. Das ist kein "nice to have" — es ist die Infrastruktur, die uns erlaubt, auf junger Silicon-Generation (RDNA 4) und mit einem neuen Compute-Modell (FP8) zu arbeiten, ohne uns bei jedem Zweifel in tagelange Fehlersuche zu verlieren.

**Praktische Implikation.** Parity-Checks sind kein Overhead, sondern Versicherung. Bei 1/1000-Sampling ist der Performance-Impact unter 1 %. Im Gegenzug erkennen wir Regressions bei ROCm-Updates *während* sie auftreten, nicht Wochen später durch Perplexity-Drift.

### 10.8 Zusammenfassung: Die Design-Philosophie in einem Satz

> v1.0 ist keine bessere Kernel-Sammlung, sondern ein System, das seine eigenen Kernel-, Fusion- und Präzisions-Strategien aus Messdaten ableitet — das die Hardware-nativen FP8-WMMA-Pfade als Standard nutzt, die 104-VGPR-Occupancy-Grenze auch auf 64 CUs respektiert, und jede WMMA-Operation gegen einen unabhängigen VALU-Referenzpfad prüft.

Das ist der unfaire Vorteil gegenüber llama.cpp: nicht eine Instanz besserer Software-Engineering, sondern ein struktureller Unterschied in der Herangehensweise — und eine Architektur, die die Silicon-Realität von RDNA 4 ernst nimmt, statt sie hinter Abstraktionen zu verstecken.

---

## Anhang A — Glossar

| Begriff | Bedeutung |
|---|---|
| **AI Accelerator** | WMMA-Einheit auf RDNA 4; 2 pro CU auf gfx1201 (128 total auf RX 9070 XT) |
| **Bandit** | Multi-Armed-Bandit-Algorithmus (hier UCB1) für Online-Entscheidungen zwischen Varianten |
| **BF16** | Bfloat16, 16-Bit-Float mit FP32-Exponent-Range, FP16-ähnlicher Geschwindigkeit |
| **BW-Effizienz** | Anteil der theoretischen Speicher-Bandbreite, den ein Kernel tatsächlich nutzt |
| **CU** | Compute Unit — Basis-Ausführungsblock auf AMD-GPUs, enthält Wavefronts und AI Accelerators |
| **Dequant IR** | Intermediate Representation für Dequantisierungs-Programme |
| **Dispatch** | Die Auswahl, welcher Kernel für eine bestimmte Operation aufgerufen wird |
| **Double-Pumped AVX-512** | Zen4-Implementierung: 512-Bit-Operationen über 2× 256-Bit-Ports, kein Takt-Downclocking |
| **E4M3** | FP8-Variante mit 4 Exponent- und 3 Mantissen-Bits. Range ±448. Default für Gewichte |
| **E5M2** | FP8-Variante mit 5 Exponent- und 2 Mantissen-Bits. Range ±57 344. Default für KV-Cache/Aktivierungen |
| **FP8** | 8-Bit-Float-Format, nativ unterstützt in RDNA 4 WMMA (E4M3 und E5M2) |
| **FP16/FP32** | IEEE-754-Floats mit 16 bzw. 32 Bit |
| **GA** | Genetischer Algorithmus |
| **GEMV** | General Matrix-Vector-Multiply (Matrix × Spaltenvektor) |
| **GEMM** | General Matrix-Matrix-Multiply |
| **gfx1201** | AMD-Compiler-Target-Code für RDNA 4 (Radeon RX 9070 XT, 64 CUs) |
| **GGUF** | GPT-Generated Unified Format, von llama.cpp definiert |
| **HIP** | AMDs CUDA-äquivalentes Programmiermodell |
| **IR** | Intermediate Representation |
| **KL-Divergenz** | Kullback-Leibler-Divergenz, Maß für Unterschied zwischen zwei Wahrscheinlichkeits-Verteilungen |
| **KV-Cache** | Key-Value-Cache in Transformer-Attention, speichert Attention-Zustand |
| **LDS** | Local Data Share — schneller Shared-Memory auf AMD-GPUs (64 KB auf gfx1201) |
| **NSGA-II** | Non-dominated Sorting Genetic Algorithm II — Multi-Objective-GA |
| **Pareto-Front** | Menge der Lösungen, die in keinem Objective von einer anderen dominiert werden |
| **Parity-Check** | Bit-nahe Vergleich zwischen WMMA-Ausführung und unabhängigem VALU-Referenzpfad |
| **Q4_0, Q4_K, Q6_K, Q8_0** | GGUF-Quantisierungs-Formate mit 4, 4, 6, 8 Bit |
| **RDNA 4** | AMD-GPU-Architektur der Radeon-RX-9000-Serie; bringt natives FP8-WMMA |
| **rf-forge** | Offline-GA-Tuning-CLI, getrennt vom Inference-Binary `rocmforge` |
| **RoPE** | Rotary Position Embedding |
| **SNR** | Signal-to-Noise-Ratio |
| **UCB1** | Upper Confidence Bound 1 — Standard-Bandit-Algorithmus |
| **VALU** | Vector ALU — skalare Vector-Ausführungseinheit auf AMD-GPUs; unabhängig von WMMA |
| **VALU-Parity** | Validierungs-Strategie: WMMA-Ergebnis gegen unabhängig berechnetes VALU-Ergebnis prüfen |
| **VGPR** | Vector General Purpose Register — eine pro Wavefront-Lane |
| **VNNI** | Vector Neural Network Instructions — Intel/AMD-SIMD-Erweiterungen für Int8-Dot-Products |
| **Wavefront** | Gruppe von GPU-Lanes, die gemeinsam ausgeführt werden (32 Lanes auf RDNA 4) |
| **WMMA** | Wave Matrix Multiply Accumulate — AMDs Matrix-Core-Instruktionen |
| **Zen4** | AMD-CPU-Architektur der Ryzen-7000-Serie, AVX-512 Double-Pumped |

---

## Anhang B — Referenzen zu v0.x-Artefakten

| Referenz | Ort | Zweck in v1.0 |
|---|---|---|
| WMMA-Register-Layout | `docs/wmma_register_layout_gfx12.md` | GPU-Codegen-Spezifikation (FP16-Pfad); FP8-Layout wird in Phase 1 ergänzt |
| Q4_K-Block-Format | `docs/q4_k_m_block_format.md` | Vorlage für `QuantFormat::Q4_K` |
| Q6_K-Block-Format | `docs/q6_k_block_format.md` | Vorlage für `QuantFormat::Q6_K` |
| Phase 8 Profiling-Analyse | `profiling/results/phase8_q4_k_m_analysis.md` | Performance-Validierungs-Referenz |
| v0.3.0 Release-Notes | `docs/release_v0.3.0.md` | Performance-Baseline |
| WMMA-GEMM-Kernel Q4_0 | `hip_kernels/wmma/wmma_gemm_q4_0.hip` | Bit-Identität-Referenz + VALU-Parity-Referenz |
| WMMA-GEMM-Kernel Q4_K | `hip_kernels/wmma/wmma_gemm_q4_k_m.hip` | Bit-Identität-Referenz + VALU-Parity-Referenz |
| WMMA-GEMM-Kernel Q6_K | `hip_kernels/wmma/wmma_gemm_q6_k.hip` | Bit-Identität-Referenz + VALU-Parity-Referenz |

---

## Anhang C — Changelog (v1.0.0 → v1.1.0 → v1.2.0)

Dieses Dokument wurde am 20. April 2026 von v1.0.0-draft (5-Säulen-Architektur, FP16-WMMA-Default, 32-CU-Annahme) auf v1.1.0-draft (6-Säulen-Architektur, FP8-WMMA-Default, 64-CU-Verifikation) revidiert. Die Änderungen betreffen fundamentale strategische Entscheidungen, keine Kosmetik — die Umstellung von einer defensiven Universal-Architektur auf eine aggressive Hardware-Native-Architektur.

### C.1 Hardware-Korrekturen (verifizierte Fakten)

| Parameter | v1.0-Draft | v1.1-Draft | Auswirkung |
|---|---|---|---|
| GPU Compute Units | 32 CU | **64 CU** | Verdoppelung der Compute-Ressourcen |
| GPU AI Accelerators | nicht quantifiziert | **128** (2 pro CU) | Explizites Performance-Budget für WMMA |
| GPU FP8-WMMA | "zukünftig optional" | **nativ, Standardpfad** | 2× WMMA-Durchsatz, halbierte VGPR-Belegung |
| GPU Memory Bandwidth | 640 GB/s (geschätzt) | ~644 GB/s (verifiziert) | marginale Korrektur |
| CPU AVX-512 | "nahezu nativ-schnell" | **Double-Pumped, ohne Takt-Reduktion** | präzise Formulierung, relevant für Planung |
| CPU-Generation | "Zen4+" | **Zen4** (explizit, nicht "+") | Zielplattform geklärt |

### C.2 Strategische Neuausrichtungen

**Performance-Ziele** (Abschnitt 1.4):

| Szenario | v1.0-Draft | v1.1-Draft |
|---|---:|---:|
| Konservativ Decode | 95 tok/s | **110 tok/s** |
| Konservativ Prefill | 3 000 tok/s | **4 500 tok/s** |
| Realistisch Decode | 110 tok/s | **125 tok/s** |
| Realistisch Prefill | 4 000 tok/s | **7 500 tok/s** |
| Optimistisch Decode | 125 tok/s | **140+ tok/s** |
| Optimistisch Prefill | 5 500 tok/s | **9 000+ tok/s (VRAM-Limit)** |

**Architektur-Stack:** 5 Säulen → **6 Säulen** (Neu: Safety & Debug).

**Präzisions-Hierarchie:** FP16 als Default → **FP8-E4M3 als Default**. Eskalations-Kette FP16 → BF16 → FP32 wird zu FP8 → FP16 → BF16 → FP32.

**KV-Cache-Behandlung:** implizit FP16 → **explizit FP8-E5M2 als Default-Dimension im Precision-GA**.

### C.3 Abschnittsweise Änderungsübersicht

| Abschnitt | Änderungsart | Kurzbeschreibung |
|---|---|---|
| Metadaten | überarbeitet | Version 1.1.0-draft, Hardware-Targets präzisiert |
| Inhaltsverzeichnis | erweitert | 6-Säulen-Referenz, Anhang-Verweis |
| 1.1 Was ist v1.0 | überarbeitet | 64 CU / 128 AI Accelerators, FP8 als Kern-Feature |
| 1.3 Hardware-Fokus | komplett neu kalibriert | neue Tabelle mit allen Hardware-Konstanten |
| 1.4 Performance-Ziele | neu kalibriert | Zahlen verdoppelt bis versechsfacht |
| 1.5 llama.cpp-Vergleich | erweitert | 3 Argumente → 4 Argumente (FP8-Pfad ergänzt) |
| 2.1 Stack-Überblick | überarbeitet | Diagramm mit 6 Säulen, Säule 6 als orthogonale Schicht |
| 2.2 Säule 1 Introspection | überarbeitet | `PrecisionHint` um Fp8E4M3/Fp8E5M2 erweitert |
| 2.4 Säule 3 Dequant IR | stark erweitert | FP8 als primäres WMMA-Input-Format, neue DequantOps (`LoadFP8`, `DowncastToFP8`, `StoreFP8`), `Fp8Variant`-Enum, Codegen-Pfad mit `target_dtype` |
| 2.6 Säule 5 Quality Monitor | überarbeitet | FP8→FP16-Eskalation statt direkt FP32, `Fp8SaturationExceeded`-DriftReason |
| 2.7 Säule 6 Safety & Debug | **NEU** | Komplette Sektion: VALU-Parity-Checker, Datenstrukturen, Algorithmus, Integration in Kernel-GA |
| 3.1 GPU-Backend | komplett überarbeitet | 64 CU, 128 AI-Accelerators, FP8-WMMA-Intrinsics, `throughput_relative` in Wmma-Types |
| 3.2 Register-Pressure | **NEU** | Eigene Sub-Section zum VGPR-Budget 104 trotz 64 CUs |
| 3.3 CPU-Backend | überarbeitet | Zen4 (ohne "+"), Double-Pumped ohne Takt-Reduktion, FP8-auf-CPU-Strategie |
| 3.4 Codegen | unverändert | |
| 4.3 Precision-GA | stark erweitert | `4^n`-Suchraum, FP8-Startpunkt, KV-Cache als eigene Dimension, Speed+Memory-Objective |
| 4.7 rf-forge CLI | **NEU** | Komplette Sektion: Subcommands, Optionen, JSONL-Log-Format, reproducible builds |
| 5 Datenflüsse | unverändert | (Cold-Start-Zeiten bleiben gültig, FP8-Compile-Overhead ist vernachlässigbar) |
| 6.7 Safety-Schnittstelle | **NEU** | Runtime ↔ Säule 6 Interface |
| 6.8 Fallback-Tabelle | erweitert | FP8-Saturation und VALU-Parity-Violations ergänzt |
| 7.1 Walk-Through Q5_K | überarbeitet | FP8 automatisch mitgeneriert, `rf-forge tune-all`, VALU-Parity-Validierung |
| 9.1 Phase 1 Roadmap | überarbeitet | FP8-Pfad in Phase 1, nicht Phase 2; VALU-Referenz-Codegen, Exit ≥ 60 tok/s |
| 9.2 Phase 2 Roadmap | überarbeitet | Säule 6 vollständig, FP8-Saturation-Detection, Exit ≥ 100 tok/s |
| 9.3 Phase 3 Roadmap | überarbeitet | `rf-forge`-CLI als Deliverable, neue Zielzahlen 125/7 500 bzw. 140/8 500 |
| 10.6 Hardware-Specs | **NEU** | Lektion aus der v1.1-Revision selbst |
| 10.7 Silicon-Vertrauen | **NEU** | Rationale für Säule 6 als eigene Säule |
| 10.8 Design-Philosophie | umformuliert | Ein-Satz-Zusammenfassung mit FP8/VGPR/VALU |
| Anhang A Glossar | stark erweitert | 8 neue Einträge (FP8, E4M3, E5M2, VALU, Parity-Check, rf-forge, AI Accelerator, Double-Pumped) |
| Anhang B v0.x-Referenzen | erweitert | "VALU-Parity-Referenz" als Zweck ergänzt |
| Anhang C Changelog | **NEU** | Dieses Dokument |
| Anhang D rf-forge Quick-Reference | **NEU** | Kompakte CLI-Referenz |

### C.4 Was sich NICHT geändert hat

Explizite Nicht-Änderungen sind wichtig, weil sie zeigen, dass die grundlegenden Design-Prinzipien der v1.0-Draft sich bewährt haben:

- **Kein Abstraktionslayer** zwischen Kernel-Code und Hardware.
- **Getrennte Verantwortung** zwischen GA (Offline-Suche), Bandit (Online-Auswahl) und regelbasierten Komponenten (Codegen, Graph-Konstruktion).
- **Caching-Strategie** mit Hardware-Fingerprint, Quant-Format-Version, Modell-Hash, GA-Version.
- **Graceful Degradation** als universelles Fehlerprinzip.
- **Bit-Identität gegen v0.x-Kernel** als Korrektheits-Messlatte für die Migration.
- **Zeitrahmen** 5–6 Wochen in 3 Phasen — die Mehrarbeit durch FP8 und Säule 6 wird durch kürzere CPU-Backend-Arbeit in Phase 3 kompensiert, weil der FP8-Pfad in Phase 1 die Performance-Ziele bereits erreicht.
- **Migrations-Strategie** (v0.x parallel weiterbetreiben, kein Big-Bang).

### C.5 Offene Punkte für v1.3 oder später

Folgende Themen wurden auch in der v1.2-Revision nicht vertieft:

- **Multi-GPU-Support.** Aktuell: single-GPU. Multi-GPU würde einen Distribution-Layer über dem Graph erfordern.
- **Speculative Decoding.** Aktuell nicht unterstützt; wäre eine Erweiterung der Runtime, die mehrere Forward-Passes parallel laufen lässt.
- **Streaming-VRAM-Offload** bei sehr großen Modellen (> 16 GB Gewichte). Aktuell: Modell muss komplett in VRAM passen.
- **Fine-Tuning-Support** (LoRA). Aktuell: reine Inference-Engine.
- **Public-API für externe Fitness-Metriken** im Precision-GA, falls User eigene Qualitätsdefinitionen einbringen wollen.

### C.6 Changelog v1.1.0-draft → v1.2.0-draft

Die v1.2-Revision ist das Ergebnis eines dreiteiligen Senior-Audits (Strategie-Review, 14B-Squeeze-Analyse, Rust-Integration-Review). Sie schließt die Design-Phase ab und macht das Dokument zum finalen Implementierungs-Blueprint.

**Neue Abschnitte:**

| Abschnitt | Inhalt |
|---|---|
| 1.4 Performance-Matrix | Modellgrößen-abhängige Tabelle statt Pauschalzahl; 8B ≤ 125 tok/s, 14B ≤ 63 tok/s; BW-Limit-Physik erklärt |
| 3.6 VRAM-Arena 2.0 | Monolithische Arena mit Zonen A/B/C, Rust-Ownership, Offset-Arithmetik, Budget-Rechnung für 8B und 14B |
| 3.7 Zero-Sync Pipeline | Single-Stream Dispatch, 2-Layer-Lookahead, Dirty-Flag-Telemetrie, Batch-Timing für Bandit |
| 3.8 Rust-FFI-Strategie | Eigene bindgen-Bindings (~19 Funktionen), RAII-Wrapper, Kernel-Module-Lifecycle mit LRU |
| 9.0 Meilenstein 0 | FP8 Go/No-Go Gate: 5-Test-Architektur, Entscheidungsmatrix, FP16-Fallback-Pfad vollständig spezifiziert |

**Überarbeitete Abschnitte:**

| Abschnitt | Änderung |
|---|---|
| Metadaten | Version 1.2.0-draft, Rust/HIP als Sprachen, 14B als Modellklasse |
| 2.6 Quality Monitor | Fused-Epilog-Implementierung statt separater Kernel; Dirty-Flag-Referenz auf Section 3.7 |
| 3.4 Codegen | hipcc-Flags präzisiert (`-shared -fPIC`), Meilenstein-0-Referenz |
| 3.5 (ex 3.4) | Nummerierung korrigiert (Duplikat behoben) |
| 9.3 Phase 3 Exit | 14B-Benchmark-Tabelle ergänzt; Meilenstein-0-Fallback-Clause |

**Strategische Entscheidungen (fixiert durch Audit):**

1. **Performance ist modellgrößen-abhängig.** 125 tok/s gilt für 8B; 14B ist physikalisch auf ~72 tok/s begrenzt. Kein Marketing, sondern Bandbreiten-Physik.
2. **Single-Stream statt Multi-Stream.** 0,05 % Overhead durch einen Sync am Token-Ende; Multi-Stream brächte ~0 % Gewinn bei erheblichem Race-Condition-Risiko.
3. **Quality-Monitor als Fused-Epilog.** Kein separater Kernel, keine Race-Condition, kein Dispatch-Overhead. ~20 Instruktionen im Epilog, vernachlässigbar.
4. **Eigene HIP-Bindings statt externe Crates.** 19 Funktionen, halber Tag Aufwand, vollständige Kontrolle.
5. **FP8 erst verifizieren, dann bauen.** Meilenstein 0 als Hard-Gate am Tag 1, nicht als optimistisches Hoffen während Phase 2.

---

## Anhang D — `rf-forge` Quick-Reference

Kompakte Übersicht über die Offline-Tuning-CLI. Vollständige Spezifikation in Abschnitt 4.7.

### D.1 Installation und Voraussetzungen

```fish
# Binary wird mit dem Haupt-Release mitgebaut
cargo build --release --bin rf-forge
sudo install -m 755 target/release/rf-forge /usr/local/bin/

# Voraussetzungen:
# - gfx1201 GPU vorhanden (rocminfo | grep gfx1201)
# - ROCm 7.2+ installiert
# - Mindestens 20 GB freier VRAM während Tuning (GA-Population in VRAM)
# - Mindestens 4 CPU-Kerne für GA-Evaluations-Parallelität
```

### D.2 Standard-Workflows

**Komplett neues Modell vorbereiten (Cold Start):**

```fish
rf-forge tune-all --model ~/models/Qwen3-8B-Q4_K_M.gguf \
                  --target gfx1201 \
                  --log ~/tune-qwen3.log \
                  --budget 3h
```

**Nach ROCm-Update alle Kernel re-validieren:**

```fish
rf-forge validate --all --tolerance 1e-4
# Falls Violations: gezielt neu tunen
rf-forge tune-kernels --model ~/models/X.gguf --force-rebuild
```

**Cache verwalten:**

```fish
rf-forge cache list                                  # alle Einträge
rf-forge cache inspect --shape "gemv_4096x12288_*"   # Details
rf-forge cache invalidate --older-than 30d            # Aufräumen
rf-forge cache size                                   # Gesamtgröße
```

**Benchmark-Lauf auf bestehendem Cache:**

```fish
rf-forge bench --model ~/models/Qwen3-8B-Q4_K_M.gguf \
               --runs 100 \
               --report ~/bench-qwen3.md
```

**Reproducible Tuning für CI:**

```fish
# Auf Dev-Maschine:
rf-forge export --model X.gguf > tune-X.toml
# In CI oder auf Build-Server:
rf-forge tune-all --from-config tune-X.toml
```

### D.3 Subcommand-Matrix

| Command | Input | Output | Typ. Dauer |
|---|---|---|---|
| `tune-all` | GGUF-Modell | vollständiger Cache | 2,5 h |
| `tune-kernels` | GGUF + optional Shapes | Kernel-Cache | 8 min × N Shapes |
| `tune-precision` | GGUF + Kernel-Cache | Precision-Profile | 15 min |
| `tune-fusion` | GGUF + Kernel-Cache | Fusions-Plan | 5 min |
| `cache list` | — | Tabelle | < 1 s |
| `cache inspect` | Shape-Pattern | Details | < 1 s |
| `cache invalidate` | Filter | Anzahl gelöschter Einträge | < 5 s |
| `cache size` | — | MB | < 1 s |
| `bench` | GGUF | Markdown-Report | 1–10 min |
| `validate` | optional Shape-Filter | Violations-Report | 1–30 min |
| `export` | GGUF | TOML-Config | < 1 s |

### D.4 Exit-Codes

```
0    Erfolgreich
1    Allgemeiner Fehler
2    GA-Konvergenz gescheitert (Timeout oder keine valid Kandidaten)
3    VALU-Parity-Violation (Korrektheitsproblem)
4    Hardware nicht unterstützt (falsche GPU, ROCm fehlt, etc.)
5    Cache-Fehler (Korruption, Permissions)
6    User-Abbruch (Ctrl-C)
```

Skriptbare Automatisierung nutzt diese Codes, um zwischen "bauen, trotzdem warnen" (Exit 2, Performance suboptimal) und "nicht releasen" (Exit 3, Korrektheitsproblem) zu unterscheiden.

---

**Ende des Architektur-Dokuments v1.2.0-draft.**

Dokument-Historie:
- 2026-04-20 v1.0.0-draft: Initialer 5-Säulen-Entwurf (FP16-Default, 32-CU-Annahme)
- 2026-04-20 v1.1.0-draft: Hardware-Native Revision (64 CU verifiziert, FP8-WMMA als Default, Säule 6 Safety & Debug, rf-forge CLI)
- 2026-04-20 v1.2.0-draft: Finale Freigabe (Rust-Native, 14B-Squeeze, FP8 Go/No-Go Gate, VRAM-Arena 2.0, Single-Stream Pipeline, eigene HIP-Bindings)

Der Code-Aufbau beginnt mit Meilenstein 0 (FP8 Smoke Test) nach schriftlicher Freigabe dieses Dokuments durch mg.
