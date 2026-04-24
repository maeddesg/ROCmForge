# Multi-Modell-Kompatibilitätstest

**Date:** 2026-04-24
**Branch:** v1.0-dev @ `7ee28af`
**Hardware:** AMD Radeon RX 9070 XT (gfx1201)
**Binary:** `target/release/rocmforge-v1` (arch-aware sampling active)

## Kompatibilitäts-Matrix

| Modell | Laden | Arch-Check | Single-Turn | Multi-Turn | 5-Prompt | tok/s |
|---|:---:|:---:|:---:|:---:|:---:|---:|
| Qwen3-8B-Q4_K_M (Ref) | ✅ | ✅ | ✅ 15/15 | ✅ Alice | ✅ | 96.2 |
| Meta-Llama-3.1-8B-Q4_K_M | ✅ | ✅ | ⚠ 1–3/15 | ❌ Alice | ⚠ | 105.4 |
| **DeepSeek-R1-Distill-Llama-8B-Q4_K_M** | ✅ | ✅ (llama) | ❌ Müll | — SKIP | — SKIP | 60.1 |
| **Mistral-7B-Instruct-v0.3-Q4_K_M** | ✅ | ✅ (llama) | ❌ Müll | — SKIP | — SKIP | 60.1 |
| **gemma-4-E4B-it-Q4_K_M** | ❌ Crash | ⚠ teil. | — SKIP | — SKIP | — SKIP | — |

Legende: ✅ = funktioniert, ⚠ = teilweise/mit Einschränkung, ❌ = scheitert, — = übersprungen.

---

## Pro-Modell-Details

### DeepSeek-R1-Distill-Llama-8B-Q4_K_M

**Stufe 1 (Laden):** ✅ OK. 292 Tensoren, 32 Metadata-Einträge.

**Stufe 2 (Arch-Check):**
- `architecture = llama` (identisch zu Llama-3.1 im GGUF)
- `n_layers=32  n_heads=32  n_kv_heads=8  GQA=true`
- `hidden=4096  ffn=14336  head_dim=128  vocab=128256`
- `rope_freq_base=500000  rope_freqs=present (NTK-ramp)  qk_norm=false`
- **Identisches Tensor-Layout wie Meta-Llama-3.1-8B-Instruct.** Keine neuen Tensor-Rollen.
- SNR-Risk: **0.04** (CRITICAL band, 159 von 128256 kritische Embedding-Tokens)
- `sampling_for(profile)` wählt `repeat_penalty=1.1` ✓

**Stufe 3 (Single-Turn Decode):** Kein Crash, 60.1 tok/s, 100 Output-Tokens. Output ist *englisch-förmig aber inkohärent* — kein Mutex-Content:
> *Okay, I'm here to explain the process. Alright, I need to explain how can you're thinking about the user's response. Okay, but i'm trying to understand that i think of the user's …*

Das matched exakt das Llama-3.1-Muster (Instruction-Following tot, flüssig aber off-topic). Zusätzlich reasoning-tangent durch DeepSeek's `<think>`-Training. StreamingEmitter filtert keine `<think>`-Tags weil das Modell sie in diesem Template nicht rendert.

**Stufe 4 (Alice):** SKIP (Stufe 3 nicht kohärent).

**Stufe 5 (5-Prompt):** SKIP.

**Verdikt:** **Stufe 3 gibt dasselbe Muster wie Llama-3.1** — Infrastruktur lädt ohne Probleme, aber die Q4_K-Embedding-SNR-Degradation plus der reasoning-Fine-Tune produzieren kein brauchbares Output. `known_limitation` gleicher Klasse wie Llama-3.1.

---

### Mistral-7B-Instruct-v0.3-Q4_K_M

**Stufe 1 (Laden):** ✅ OK. 291 Tensoren, 29 Metadata-Einträge.

**Stufe 2 (Arch-Check):**
- `architecture = llama` (Mistral exportiert unter dem `llama`-Arch-Tag im GGUF)
- `n_layers=32  n_heads=32  n_kv_heads=8  GQA=true` (GQA, nicht MHA — korrigiert den Prompt)
- `hidden=4096  ffn=14336  head_dim=128  vocab=32768` (Llama-2-era Tokenizer, **NICHT** 128k)
- `rope_freq_base=1000000  rope_freqs=false  qk_norm=false`
- **Identisches Tensor-Layout wie Llama-3.1**, aber:
  - kein `rope_freqs.weight`
  - vocab 32768 (4× kleiner)
  - `context_length` = 32768 (im Vergleich zu Llama-3.1's 131072)
- SNR-Risk: **0.27** (CRITICAL band, 35 kritische Tokens — deutlich besser als Llama-3.1/DeepSeek)

**Stufe 3 (Single-Turn Decode):** Kein Crash, 60.1 tok/s, 100 Output-Tokens.
- **⚠ Prompt-Tokens = 240** für "Explain what a mutex is in one paragraph." — 8× zu hoch.
- Ursache: Unser Chat-Template verzweigt auf `architecture = "llama"` und emittiert Llama-3 Header-Blöcke (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`, …). Diese Token-Literale sind im Mistral-Vocab **nicht als Einzel-IDs** vorhanden → unser BPE-Encoder fallt sie byte-weise in ~240 Garbage-Tokens auseinander.
- Output: flüssiges Englisch, aber nonsense-semantik:
  > *It appears that the input text has been encoded using a shifted ASCII encoding, which is a form of Base64 encoding …* (und danach Unicode-Garbage)
- Model versucht den Garbage zu "erklären" — klassisches OOD-Verhalten.

**Stufe 4 (Alice):** SKIP.

**Stufe 5 (5-Prompt):** SKIP.

**Verdikt:** **Architektur lädt, Decode-Kernel-Pfad läuft sauber bei 60 tok/s**, aber der *Chat-Template-Mismatch* (Llama-3-Format für Mistral) macht den Output unbrauchbar. Ein einzelner v1.1-Fix (korrekter `[INST]...[/INST]`-Template-Pfad für Mistral) würde vermutlich genügen — die Kernel-Infrastruktur selbst ist kompatibel.

---

### gemma-4-E4B-it-Q4_K_M

**Stufe 1 (Laden):** ❌ **CRASH**.

```
Error: model load: HIP error -1
  (Zone-A alloc failed for 'blk.41.proj.weight':
   Zone A overflow: need 368640 bytes at cursor 5319107072,
   zone size 5319467008)
```

VRAM-Arena wurde zu klein bemessen weil ~6 Tensor-Rollen pro Layer UNBEKANNT sind → sie gehen nicht in die Arena-Sizer-Schätzung ein. Der Loader läuft bis Layer 41, dann Overflow.

**Stufe 2 (Arch-Check — aus `--list-tensors` extrahiert):**
- `architecture = gemma4` (wird als DISTINKT erkannt, nicht auf llama gemappt)
- `n_layers=42  n_heads=8  n_kv_heads=2  GQA=true` (viel kleinere Heads)
- `hidden=2560  ffn=10240  **head_dim=512**  vocab=262144` (vocab 2× Llama-3, head_dim 4× Llama)
- `rope_freq_base=1000000  rope_freqs=present  qk_norm=true`
- 720 Tensoren gesamt (!) — Llama-3.1 hat 292.

**Unique Gemma-4-Tensoren** die v1.0 nicht kennt:

| Tensor | Shape | Zweck |
|---|---|---|
| `per_layer_token_embd.weight` | 10752 × 262144 | **Per-Layer Embeddings** (PLE) — 2.2 GB! Eigenes Embedding pro Layer |
| `per_layer_model_proj.weight` | 2560 × 10752 | PLE global projection |
| `per_layer_proj_norm.weight` | 256 | PLE norm |
| `blk.N.inp_gate.weight` | 2560 × 256 | Per-Layer Input Gate |
| `blk.N.proj.weight` | 256 × 2560 | Per-Layer Projection |
| `blk.N.layer_output_scale.weight` | 1 (scalar) | Per-Layer Output-Scaling |
| `blk.N.post_attention_norm.weight` | 2560 | Zusätzlicher Post-Attention Norm |
| `blk.N.post_ffw_norm.weight` | 2560 | Zusätzlicher Post-FFN Norm |
| `blk.N.post_norm.weight` | 2560 | Zusätzlicher Post-Norm |

**Stufe 3–5:** SKIP (Laden gescheitert).

**Verdikt:** **Erwartungsgemäß gescheitert**, wie im Prompt vorausgesagt. Kein einzelner Fix — Gemma-4 braucht einen v1.1-Architektur-Neubau:
- PLE-Pfad (Zusatz-Embedding pro Layer mit Global-Projection)
- 9 neue Tensor-Rollen in `TensorRole` + Parser
- Hybrid-Attention (Sliding-Window-Layers + Global-Layers abwechselnd) — aus der rocmforge-Tensorliste **noch nicht sichtbar**, könnte im Metadata-JSON stehen
- Shared-KV-Cache über Global-Layers
- Gemma-3-Chat-Template (`<start_of_turn>user\n…<end_of_turn>`)
- Arena-Sizer-Override für unbekannte Architekturen (generell robuster)
- head_dim=512 — alle Decode-Kernel nehmen aktuell head_dim ∈ {64, 128}; bräuchten Kompatibilitätsprüfung

---

## v1.1 Feature-Backlog

Aus den drei neuen Modellen identifiziert:

### Sofort-Lücken mit hoher Wirkung

1. **Chat-Template-Disambiguation für `arch=llama`** (blockt Mistral komplett, teilweise auch andere Llama-2-Derivate).
   - `general.architecture = "llama"` ist nicht eindeutig: Llama-2 / Llama-3 / Mistral / DeepSeek-Distill teilen sich denselben Tag.
   - Fix: Disambiguierung über `tokenizer.ggml.bos_token_id` + `tokenizer.ggml.model` + vocab-size. Llama-3 = bos 128000, Mistral = bos 1, Llama-2 = bos 1 vocab 32000.
   - Mistral-Template: `[INST] {prompt} [/INST]`, Llama-2: `<s>[INST] <<SYS>> … <</SYS>> {prompt} [/INST]`.
   - Geschätzter Aufwand: 2-3 h für Detection + 2 Template-Branches.

2. **Arena-Sizer-Robustheit** (blockt Gemma-4 sofort).
   - Aktuell nimmt der Sizer eine feste Tensor-Rollen-Liste an. Unbekannte Tensoren kippen in Overflow.
   - Fix: Über **tatsächliche GGUF-Tensor-Byte-Summen** sizen statt über Rollen-Schätzung. Arena = `sum(tensor.byte_size) + margin`.
   - Löst nicht das Semantik-Problem (unbekannte Tensoren wissen wir noch nicht wie zu benutzen), aber mindestens crasht der Loader nicht mehr.
   - Aufwand: 1-2 h.

### Architektur-Feature-Lücken (mittel-große Blöcke für v1.1)

3. **Sliding-Window-Attention** (blockt Mistral bei >4k Kontext, Gemma-3 bei den Local-Layers).
   - Attention-Kernel ergänzen, pro-Layer-Flag.

4. **Per-Layer-Embeddings (PLE)** (blockt Gemma-4 komplett).
   - Neues Graph-Node + Kernel, Zusatz-Vocab-Tabelle pro Layer.
   - Global-Projection-Pfad.

5. **Hybrid-Attention** (blockt Gemma-4 komplett).
   - Layer-spezifisches Attention-Pattern (Sliding vs Global abwechselnd).
   - Shared-KV-Cache über Global-Layers (nicht alle Layer bekommen eigenen Cache).

6. **head_dim ∈ {64, 128, 512}** (blockt Gemma-4).
   - Aktuell hardkodiert auf 128 in LDS-Tiling der Attention-Decode. Dyn. Tile-Größe oder expandierte Kernel-Variante.

7. **p-RoPE (Proportional RoPE)** (Gemma-3/4 Feature).
   - Formulieren als `freq_scale * base ^ (2i / head_dim)` mit model-specific `proportional_factor`.

### Quality-Lücken (kein Arch-Block, aber Output-Qualität)

8. **Llama-3-Derivate mit SNR < 0.1** (betrifft Llama-3.1, DeepSeek-R1-Distill).
   - Gleiche Klasse wie `known_limitation` für Llama-3.1. Jetzt 2 Modelle betroffen.
   - Siehe `results/phase3_llama31_known_limitation.md` für die ausgeschlossenen Root-Causes.

---

## Gesamtstand

- **Model-Families die v1.0 robust unterstützt:** Qwen3-8B, Qwen2/2.5 (implizit, gleicher Arch-Tag).
- **Model-Families die v1.0 lädt, aber unbrauchbare Ausgabe produziert:** Llama-3.1 / DeepSeek-R1-Distill (SNR-Klasse), Mistral (Chat-Template).
- **Model-Families die v1.0 nicht lädt:** Gemma-4 (unbekannte Arch).

Die 2 Sofort-Lücken (**Chat-Template-Disambig + Arena-Sizer-Robustheit**) sind zusammen ~4 h und würden Mistral komplett entblocken + Gemma-4 vom Crash zu "Arch nicht unterstützt (erwartet)" bringen. Das sind die empfohlenen v1.1-Starter.
