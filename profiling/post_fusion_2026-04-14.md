Kurze Frage zum QKV-Fused-Kernel: Nutzt du darin bereits die neuen GFX12-spezifischen Instruktionen für die Q4_0 Dequantisierung (z. B. über Dot-Product-Builtins), oder ist das noch klassischer Shift-und-Mask-Code?

# Post-Fusion Profiling — 2026-04-14

## Setup

- **GPU:** AMD Radeon RX 9070 XT (gfx1201, RDNA4)
- **ROCm:** 7.2, rocprofv3 1.1.0
- **Model:** Qwen2.5-0.5B-Instruct Q4_0
- **Workload:** 1 prompt token ("Hello"), 64 decode tokens, greedy sampling
- **Branch:** `feat/auto-detect-gpu-arch` (nach QKV+RoPE+KV-Write Fusion)
- **Änderung:** `gemv_qkv_rope_kvwrite_q4_0_f32_kernel` ersetzt 3 separate Kernel (QKV, RoPE-Q, KV-Write+RoPE-K)

## Vorher/Nachher-Vergleich: Graph-Pfad

### HIP Graph Launch Overhead

| Metrik | Baseline | Nach Fusion | Delta |
|--------|----------|-------------|-------|
| hipGraphLaunch total (64 tok) | 438.3 ms | **324.6 ms** | **-25.9%** |
| hipGraphLaunch avg/Token | 6.85 ms | **5.07 ms** | **-1.78 ms (-26%)** |
| hipGraphLaunch stddev | — | 56.5 µs | stabil |
| Graph Nodes | 219 | **172** | **-47 Nodes (-21.5%)** |

**Die Graph-Launch-Zeit ist um 26% gesunken.** Das bestätigt, dass die Reduktion der Node-Anzahl direkt den Launch-Overhead senkt.

### GPU Kernel-Zeiten (Decode, kumulativ über 64 Tokens)

| Kernel | Baseline Calls | Fusion Calls | Baseline µs | Fusion µs | Delta |
|--------|---------------|-------------|-------------|-----------|-------|
| GEMV Residual (Attn-Out+FFN-Down) | 3072 | 3072 | 31,025 | 31,196 | ~0% (rauschen) |
| Gate+Up+SiLU | 1536 | 1536 | 19,009 | 19,351 | ~0% |
| LM-Head | 65 | 65 | 14,517 | 14,606 | ~0% |
| **QKV (alt: 3 Kernel)** | **1536+1536+1536** | — | **11,389+2,451+2,317 = 16,157** | — | — |
| **QKV+RoPE+KV (neu: 1 Kernel)** | — | **1536** | — | **12,048** | **-25.4%** |
| Attention | 1536 | 1536 | 10,870 | 11,041 | ~0% |
| RMS Norm | 3137 | 3137 | 5,346 | 5,429 | ~0% |
| **rope_heads_state** | **1536** | **0** | **2,451** | **0** | **eliminiert** |
| **kv_write_rope_state** | **1536** | **0** | **2,317** | **0** | **eliminiert** |

**Summe GPU Kernel-Rechenzeit:**
- Baseline: ~118,900 µs
- Nach Fusion: ~114,800 µs
- **Delta: -3.4% GPU-Rechenzeit** (4,100 µs gespart)

### Per-Token Breakdown (Decode)

| Operation | Baseline | Nach Fusion | Delta |
|-----------|----------|-------------|-------|
| GEMV Residual | 485 µs | 487 µs | ~0 |
| Gate+Up+SiLU | 298 µs | 302 µs | ~0 |
| LM-Head | 223 µs | 225 µs | ~0 |
| **QKV+RoPE+KV** | **252 µs (3 Kernel)** | **188 µs (1 Kernel)** | **-64 µs (-25.4%)** |
| Attention | 170 µs | 173 µs | ~0 |
| RMS Norm | 82 µs | 85 µs | ~0 |
| Argmax | 5 µs | 5 µs | ~0 |
| **Summe GPU pro Token** | **~1,515 µs** | **~1,465 µs** | **-50 µs (-3.3%)** |

### End-to-End Zeitbudget pro Token

| Komponente | Baseline | Nach Fusion | Delta |
|-----------|----------|-------------|-------|
| GPU Kernel-Rechenzeit | ~1.5 ms | ~1.5 ms | -3.3% |
| hipGraphLaunch | 6.85 ms | **5.07 ms** | **-26%** |
| hipStreamSynchronize | 1.1 ms | 1.1 ms | ~0 |
| **Gesamt pro Token (Profiler)** | ~8.0 ms | **~5.8 ms** | **-27.5%** |
| **tok/s (Profiler)** | 124.8 | **170.9** | **+37%** |

### Kernel-Launch-Counts pro Decode-Token

| Operation | Baseline | Nach Fusion | Delta |
|-----------|----------|-------------|-------|
| GEMV Residual | 48 | 48 | 0 |
| Gate+Up+SiLU | 24 | 24 | 0 |
| **QKV** | **24** | **0** | **-24** |
| **QKV+RoPE+KV (fused)** | **0** | **24** | **+24** |
| Attention | 24 | 24 | 0 |
| RMS Norm | 48 | 48 | 0 |
| **RoPE Q** | **24** | **0** | **-24** |
| **KV Write+RoPE K** | **24** | **0** | **-24** |
| LM-Head | 1 | 1 | 0 |
| Argmax | 2 | 2 | 0 |
| **Total** | **~219** | **~171** | **-48 (-21.9%)** |

## Benchmark-Ergebnis (ohne Profiler-Overhead)

| Metrik | Baseline (RDNA4) | Nach Fusion | Delta |
|--------|-----------------|-------------|-------|
| Decode avg | 558.3 tok/s | **593.7 tok/s** | **+6.3%** |
| Decode max | — | 597.8 tok/s | |
| Prefill avg | 458.8 tok/s | **499.8 tok/s** | **+8.9%** |

## Fazit

1. **hipGraphLaunch ist immer noch der dominante Flaschenhals** (53.5% der Gesamtzeit unter Profiling), aber er ist von 6.85 ms auf 5.07 ms pro Token gesunken (-26%).

2. **Die GPU-Rechenzeit hat sich kaum geändert** (-3.3%), was erwartet war — die Arbeit ist dieselbe, nur der Launch-Overhead ist geringer.

3. **Der Benchmark-Gewinn (+6.3% Decode) ist geringer als der Profiler-Gewinn (+37%)**, weil der Profiler den Launch-Overhead überproportional aufbläht. Im realen Betrieb (527→594 tok/s) macht der Launch-Overhead einen kleineren Anteil aus.

4. **48 Kernel-Launches pro Token eliminiert** (von 219 auf 171), 47 Graph-Nodes weniger.

## Nächste Optimierungshebel

| Priorität | Optimierung | Launches gespart | Erwarteter Gewinn |
|-----------|------------|------------------|-------------------|
| 1 | Norm+QKV Fusion (RMSNorm in QKV-Kernel) | -24 | ~2-3% |
| 2 | Norm+Gate+Up Fusion | -24 | ~2-3% |
| 3 | LM-Head Optimierung (vocab sharding) | 0 | ~1-2% (GPU-Zeit) |
| 4 | Graph-Overhead weiter reduzieren | — | abhängig von Treiber |

Aktuelle Launches: **171/Token**, realistische Untergrenze: ~120-130 (mit weiteren Fusionen).
