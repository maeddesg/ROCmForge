# ROCmForge RDNA4 Optimization Plan

## Ausgangslage

**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA4), 16 GB VRAM, Wave32
**ROCm:** 7.2, LLVM 22 (AMD clang)
**Modelle:** Qwen2.5-0.5B Q4_0, Qwen2.5-7B Q4_0

### Aktuelle Performance (nach Bug-Fixes vom 2026-04-14)

| Modell | Pfad | Prefill | Decode |
|--------|------|---------|--------|
| 0.5B Q4_0 | GPU + Graph | ~409 tok/s | ~527 tok/s |
| 0.5B Q4_0 | GPU non-graph | — | ~514 tok/s |
| 7B Q4_0 | GPU | ~32 tok/s | ~107 tok/s |

**Referenz:** llama.cpp Vulkan auf gfx1100: ~619 tok/s (0.5B decode)

### Hauptengpass

Batch-1 Decode ist **memory-bandwidth-bound**, nicht compute-bound. Der Engpass sind:
1. **~240-288 Kernel-Launches pro Token** (10-12 pro Layer × 24 Layer)
2. **Redundante VRAM-Roundtrips** zwischen Kernels (Zwischenergebnisse → VRAM → nächster Kernel)
3. **Launch-Overhead** dominiert bei kleinen Kernels (< 10µs Rechenzeit)

---

## Phase 1: Kernel-Fusion (Ziel: +15-25%)

Reduktion der Kernel-Launches pro Token von ~240 auf ~120-144.

### Schritt 1.1: Baseline-Profiling mit rocprofv3

**Ziel:** Messen wo die Zeit tatsächlich hingeht, bevor etwas optimiert wird.

**Vorgehen:**
1. Decode-Timeline mit `rocprofv3` aufnehmen (64 Token, 0.5B)
2. Pro-Kernel Laufzeiten + Idle-Gaps zwischen Launches messen
3. Memory-Traffic pro Kernel identifizieren
4. Top-10 zeitfressende Kernels und Top-10 größte Idle-Gaps dokumentieren

**Kommandos:**
```bash
./.rocprofv3/profile_decode.sh runtime
./.rocprofv3/profile_decode.sh runtime-graph
```

**Ergebnis:** Messdaten als Baseline für alle weiteren Optimierungen. Ohne das optimieren wir blind.

**Dateien:** keine Code-Änderung, nur Analyse

---

### Schritt 1.2: Residual-Add in FFN-Down und Attn-Output GEMV einbauen

**Ziel:** Separaten `add_on_stream`-Kernel nach GEMV eliminieren.

**Aktuell (forward.rs):**
```
GEMV(attn_output) → separater add(residual)    # 2 Launches
GEMV(ffn_down)    → separater add(residual)    # 2 Launches
```

**Danach:**
```
GEMV_residual(attn_output)   # 1 Launch, Add ist im Kernel
GEMV_residual(ffn_down)      # 1 Launch, Add ist im Kernel
```

**Hinweis:** `gpu_dispatch_gemv_residual_on_stream` existiert bereits — prüfen ob alle Pfade (Graph + Hybrid) diesen nutzen oder ob Fallback-Pfade noch separate Adds machen.

**Dateien:**
- `src/gpu/forward.rs` — Hybrid-Pfad prüfen, ggf. Fallback-Pfade auf Residual-GEMV umstellen
- `src/gpu/ops.rs` — ggf. neue Dispatch-Variante

**Erwarteter Gewinn:** -48 Launches/Token (2 pro Layer × 24), ~3-5%

---

### Schritt 1.3: QKV + RoPE + KV-Write zu einem Kernel fusionieren

**Ziel:** 4 separate Kernel-Launches pro Layer → 1.

**Aktuell:**
```
1. fused_qkv(Q,K,V)         → VRAM schreiben
2. rope_q(Q)                 → VRAM lesen + schreiben
3. rope_k(K) + kv_write(K,V) → VRAM lesen + schreiben
```

**Danach:**
```
1. fused_qkv_rope_kvwrite(Q,K,V,RoPE,KV-Cache)  → alles in einem Kernel
   - Q,K,V projizieren
   - RoPE in Registern anwenden (kein VRAM-Roundtrip)
   - K,V direkt in Cache schreiben
   - Q bleibt in VRAM für Attention
```

**Dateien (neu + geändert):**
- `hip_kernels/quant/q4_0_fused_qkv_rope_kvwrite.hip` — **neuer Kernel**
- `src/gpu/kernels/quant.rs` — Rust FFI-Wrapper
- `src/gpu/ops.rs` — Dispatch-Funktion
- `src/gpu/forward.rs` — Neue Fused-Funktion aufrufen (hinter Feature-Flag)

**Vorgehen:**
1. Bestehenden `q4_0_fused.hip` als Basis nehmen (QKV-Fusion existiert)
2. RoPE-Logik aus `rope.hip` inline einfügen
3. KV-Write-Logik aus `attention.hip` (kv_write_rope) inline einfügen
4. Validieren: GPU-Decode-Output muss bit-identisch zum unfusionierten Pfad sein
5. Mit `ROCMFORGE_ENABLE_EXPERIMENTAL_FUSED_QKV_ROPE=1` gaten

**Erwarteter Gewinn:** -72 Launches/Token, ~8-12% (größter Einzelgewinn)

**Ergebnis (2026-04-14):** ✅ Implementiert und verifiziert.
- Kernel: `hip_kernels/quant/q4_0_fused_qkv_rope.hip`
- FFI+Dispatch: `src/gpu/kernels/quant.rs`, `src/gpu/ops.rs`
- Forward (Graph-Pfad): `src/gpu/forward.rs` — fused-first mit Fallback
- Hybrid-Pfad: nicht aktualisiert (nutzt Host-`pos`, kein Device-Pointer)
- Correctness: GPU vs CPU greedy token identisch ✅
- Performance (0.5B Q4_0, RX 9070 XT, Graph):
  - **Decode: 558 → 594 tok/s (+6.3%)**
  - **Prefill: 459 → 500 tok/s (+8.9%)**
  - Decode min/max: 590-598 tok/s, stddev 2.9

**Risiko:** Mittel — neuer Kernel muss korrekt sein, aber kann hinter Feature-Flag stehen

---

### Schritt 1.4: Fused Gate+Up + SiLU prüfen

**Ziel:** Sicherstellen dass der existierende Fused-Gate-Up-Kernel wirklich SiLU inline macht und kein separater SiLU-Launch stattfindet.

**Vorgehen:**
1. `gpu_dispatch_fused_gate_up_on_stream` in ops.rs prüfen
2. Trace in rocprofv3 auf separate `silu_kernel` Launches prüfen
3. Falls separater SiLU-Launch: in den Fused-Kernel integrieren

**Dateien:** `src/gpu/ops.rs`, ggf. `hip_kernels/quant/q4_0_fused_q8.hip`

**Erwarteter Gewinn:** 0-3% (vermutlich bereits fusioniert, nur Verifikation)

---

### Schritt 1.5: Phase-1-Profiling wiederholen

Nach Schritt 1.2-1.4 nochmal `rocprofv3` laufen lassen und mit Baseline vergleichen:
- Kernel-Launches pro Token zählen
- Idle-Gaps messen
- tok/s Benchmark (0.5B + 7B)

---

## Phase 2: Attention & Graph-Optimierung (Ziel: +8-12%)

### Schritt 2.1: Decode-Attention mit Online-Softmax

**Ziel:** Attention-Kernel für lange Sequenzen effizienter machen.

**Aktuell:** Alle Q·K Scores werden in LDS gespeichert (max ~2048 Sequenzlänge), dann ein Pass über V.

**Optimierung:**
- Online-Softmax (Tile-basiert, z.B. 64er Tiles)
- Pro Tile: lokales max + sum berechnen, mit globalem Zustand mergen
- V-Akkumulation läuft parallel zur Score-Berechnung
- Weniger LDS-Verbrauch → höhere Occupancy

**Dateien:**
- `hip_kernels/attention.hip` — neuer Kernel `flash_attn_decode_online_softmax`
- `src/gpu/kernels/attention.rs` — FFI-Wrapper
- `src/gpu/forward.rs` — Dispatch hinter Feature-Flag

**Vorgehen:**
1. Bestehenden Kernel kopieren als Basis
2. Score-Loop auf Tile-basiert umbauen
3. Online-Softmax implementieren (running max + running sum)
4. Numerische Validierung gegen bestehenden Kernel
5. Benchmark: kurze Sequenzen (< 128) vs. lange Sequenzen (> 512)

**Erwarteter Gewinn:** 5-8% bei langen Sequenzen, minimal bei kurzen

**Risiko:** Hoch — Attention-Rewrites sind in der Vergangenheit gescheitert (siehe improvements.md: "reduced decode to about 141-159 tok/s"). Unbedingt sorgfältig validieren.

---

### Schritt 2.2: Multi-Iteration HIP Graph Capture

**Ziel:** Mehrere Decode-Steps in einem einzigen Graph-Launch zusammenfassen.

**Aktuell:** 1 Graph = 1 Decode-Iteration (~240 Nodes). Jeder Token-Step erfordert einen Graph-Launch.

**Optimierung:**
- Graph für N=4 Decode-Iterationen aufnehmen
- Zwischen Iterationen nur Position + Seq-Length per Graph-Parameter-Update ändern
- Graph-Launch-Overhead wird über 4 Tokens amortisiert

**Vorgehen:**
1. `DecodeGraphScope` erweitern für Multi-Iteration Capture
2. Position-Counter als Device-Pointer (existiert bereits: `decode_pos_ptr`)
3. Pro Iteration: Position inkrementieren via `state_increment_on_stream` (existiert)
4. Graph-Parameter-Update API für die sich ändernden Werte nutzen
5. Fallback auf Single-Iteration wenn N nicht passt (z.B. letzte Tokens vor max_tokens)

**Dateien:**
- `src/gpu/graph.rs` — Multi-Iteration Capture-Logik
- `src/gpu/forward.rs` — Aufrufer anpassen

**Erwarteter Gewinn:** 3-5% (weniger Graph-Launch-Overhead)

**Risiko:** Niedrig — additive Änderung, Single-Iteration bleibt als Fallback

---

### Schritt 2.3: Phase-2-Profiling

Erneut messen und mit Phase-1-Ergebnissen vergleichen. Fokus auf:
- Attention-Kernel Laufzeit (kurz vs. lang)
- Graph-Launch-Overhead (Single vs. Multi-Iteration)

---

## Phase 3: Feinschliff (Ziel: +3-5%)

### Schritt 3.1: Vektorisierte Speicherzugriffe im Q4_0 GEMV

**Ziel:** Weight-Loading mit 128-Bit Loads statt skalarem Zugriff.

**Aktuell:** `Q4_0_block` (18 Bytes) wird einzeln geladen und Nibble-weise entpackt.

**Optimierung:**
- 4 Blöcke gleichzeitig laden via `int4` / `float4`
- Nibble-Unpacking per Bitwise-Ops in SIMD-Stil
- Bessere Cache-Line-Auslastung (128 Byte Lines auf RDNA4)

**Dateien:** `hip_kernels/quant/q4_0_gemv.hip`

**Erwarteter Gewinn:** 3-5% durch bessere Speicherbandbreite-Nutzung

---

### Schritt 3.2: Block-Size Tuning für gfx1201

**Ziel:** Optimale Block-Größe per Hardware-Profiling finden.

**Vorgehen:**
1. Launch-Autotune aktivieren: `ROCMFORGE_ENABLE_LAUNCH_AUTOTUNE=1`
2. Block-Größen {128, 192, 256, 320} systematisch testen
3. Ergebnisse in Autotune-Cache persistieren
4. Ggf. gfx1201-spezifische Defaults in `arch.rs` hinterlegen

**Dateien:** `src/gpu/launch_autotune.rs`, `src/gpu/arch.rs`

**Erwarteter Gewinn:** 1-3%

---

### Schritt 3.3: Optional — fp16 KV-Cache

**Ziel:** Speicherbandbreite für Attention halbieren.

**Trade-off:** Minimaler Genauigkeitsverlust vs. 50% weniger Bandbreite für KV-Zugriffe.

**Vorgehen:**
1. KV-Cache Buffertyp parametrisieren (f32 vs. f16)
2. Attention-Kernel für f16-KV anpassen
3. Accuracy-Vergleich: Greedy-Token-Sequenz muss identisch bleiben
4. Gaten via `ROCMFORGE_KV_PRECISION=fp16`

**Dateien:** `src/gpu/cache.rs`, `hip_kernels/attention.hip`

**Erwarteter Gewinn:** 3-5% bei langen Sequenzen (mehr KV-Cache-Daten → mehr Bandbreiteneffekt)

**Risiko:** Mittel — muss sorgfältig gegen Accuracy-Regression getestet werden

---

## Zusammenfassung & erwartete Ergebnisse

| Phase | Maßnahme | Launches/Token | Erwarteter Gewinn |
|-------|----------|---------------|-------------------|
| **1.1** | Baseline-Profiling | 240 | — (Messung) |
| **1.2** | Residual in GEMV | 192 | +3-5% |
| **1.3** | QKV+RoPE+KV Fusion | 120 | +8-12% |
| **1.4** | Gate+Up+SiLU Check | 120 | +0-3% |
| **2.1** | Online-Softmax Attention | 120 | +5-8% |
| **2.2** | Multi-Iteration Graph | 120 (amortisiert) | +3-5% |
| **3.1** | Vektorisierte Loads | 120 | +3-5% |
| **3.2** | Block-Size Tuning | 120 | +1-3% |
| **3.3** | fp16 KV-Cache | 120 | +3-5% |

### Geschätzte Zielwerte

| Modell | Aktuell | Nach Phase 1 | Nach Phase 2 | Nach Phase 3 |
|--------|---------|-------------|-------------|-------------|
| 0.5B Decode | ~527 tok/s | ~630-660 tok/s | ~700-740 tok/s | ~730-770 tok/s |
| 7B Decode | ~107 tok/s | ~130-140 tok/s | ~145-160 tok/s | ~155-170 tok/s |

---

## Validierungsstrategie

Jeder Schritt muss folgende Tests bestehen bevor er gemergt wird:

1. **Korrektheit:** `cargo test --release --features gpu --test gpu_decode_real -- --test-threads=1`
   - Greedy-Token-Sequenz muss identisch zum vorherigen Stand sein
2. **Stabilität:** 10-Run Benchmark mit < 3% Standardabweichung
3. **Regression:** Kein Pfad (Graph/Non-Graph) darf langsamer werden
4. **Profiling:** rocprofv3 Vorher/Nachher-Vergleich dokumentieren

```bash
# Korrektheit
ROCMFORGE_MODEL_PATH=~/models/qwen2.5-0.5b-instruct-q4_0.gguf \
cargo test --release --features gpu --test gpu_decode_real \
  test_gpu_decode_real_model_matches_cpu_greedy_token \
  -- --nocapture --test-threads=1

# Performance (10 Runs, 1 Warmup, 128 Tokens)
ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 \
ROCMFORGE_MODEL_PATH=~/models/qwen2.5-0.5b-instruct-q4_0.gguf \
ROCMFORGE_BENCH_RUNS=10 ROCMFORGE_BENCH_WARMUP=1 ROCMFORGE_BENCH_TOKENS=128 \
cargo test --release --features gpu --test gpu_decode_real \
  test_gpu_greedy_decode_benchmark_real_model_multi_run \
  -- --ignored --nocapture --test-threads=1
```

---

## Prinzipien

- **Messen vor Optimieren.** Kein Kernel-Rewrite ohne rocprofv3-Daten.
- **Feature-Flags für experimentelle Pfade.** Neue Kernels hinter `ROCMFORGE_ENABLE_*` Env-Vars.
- **Keine Model-Name-Branches.** Alles shape-driven oder metadata-driven (siehe CLAUDE.md).
- **Inkrementell.** Jeder Schritt ist ein eigener Commit/PR mit Vorher/Nachher-Messung.
- **Regressionstests.** Greedy-Token-Sequenz darf sich nie ändern.
