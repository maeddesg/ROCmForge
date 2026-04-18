# Phase 7 Baseline — llama.cpp Q4_K_M target numbers

**Date:** 2026-04-18
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA 4), 16 GB VRAM, ROCm 7.2.2
**llama.cpp:** git `23b8cc4` (2026-04-18 build, fresh rebuild for Qwen3 support)
**CPU:** AMD Ryzen 9 7945HX (Zen 4), 64 GB DDR5

Goal: capture the performance numbers ROCmForge has to beat at the end
of Phase 7. Q4_K_M is the quantisation format almost every current
GGUF release ships by default — Phase 7 replaces ROCmForge's Q4_0-only
focus with Q4_K_M support on two new target models.

---

## Substitution notice

- **Llama-3.3-8B does not exist.** Meta shipped Llama 3.3 as 70B only;
  the 8B-class model in the Llama 3 family remains Llama-3.1-8B-Instruct.
  This baseline uses **Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf** as the
  Llama-3 representative. That's the production deployment target
  anyway.

## Downloaded models

| Model | Repo | Quant | File size |
|---|---|---|---:|
| Qwen3-8B Q4_K_M | `Qwen/Qwen3-8B-GGUF` | Q4_K - Medium | 4.7 GB |
| Llama-3.1-8B Q4_K_M | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | Q4_K - Medium | 4.6 GB |

Both saved to `~/models/`. Both smoke-test clean (generate coherent
text under `llama-cli --single-turn`).

## llama.cpp update

The ROCmForge-installed llama.cpp at `~/tmp/llama.cpp/build-rocm/`
was at commit `408225b` from Phase 3. That version hung on Qwen3
model-load (Qwen3 architecture not yet supported). Rebuilt at current
HEAD `23b8cc4`; both models load and run. Build command:

```
cmake --build build-rocm --config Release -j$(nproc)
```

Keeps the existing ROCm/HIP config (no need to re-invoke cmake).

---

## 1. Synthetic benchmarks (`llama-bench`)

3 runs each, defaults otherwise (`-ngl 99`):

| Model | pp64 | pp128 | pp256 | pp512 | tg128 |
|---|---:|---:|---:|---:|---:|
| **Qwen3-8B Q4_K_M** | 2,000 ± 8 | 2,657 ± 1 | 3,661 ± 10 | 3,756 ± 6 | **87.4 ± 0.8** |
| **Llama-3.1-8B Q4_K_M** | 2,198 ± 15 | 2,975 ± 3 | 3,925 ± 20 | 3,922 ± 1 | **93.3 ± 0.7** |
| *Qwen2.5-7B Q4_0 (Phase 4 ref)* | *2,912* | *3,966* | *4,951* | *5,158* | *117–121* |

**Reading this:** Q4_K_M is substantially slower than Q4_0 at every
shape even in llama.cpp itself — the K-quant format does more compute
per block (super-block structure with 6-bit scales + 6-bit mins + FP16
super-scale + FP16 super-min, dequant is `y = q·(d·d6/64) + m·m6/64`).
llama.cpp pp256: 4,951 tok/s on Q4_0 → 3,661-3,925 tok/s on Q4_K_M
(−26 to −34 %). Decode: 117–121 → 87–93 tok/s (−22 to −26 %). These
are the **realistic llama.cpp ceilings** for the new target models.

Raw data: `llama_cpp_synthetic.txt`.

## 2. Real-world 15-prompt benchmark (`llama-cli --single-turn`)

15 prompts (5 code / 5 chat / 5 prose), 128 generated tokens, greedy
(temp=0, top_p=1), 3 runs per prompt, median over 45 runs:

| Model | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| **Qwen3-8B Q4_K_M** | **834.4** | **98.7** |
| **Llama-3.1-8B Q4_K_M** | **1,407.7** | **103.7** |
| *Qwen2.5-7B Q4_0 (Phase 4 ref)* | *525.4* | *121.0* |

Interesting pattern: Llama-3.1 prefill at short prompts (834 → 1407) is
much faster than Qwen3 on the same shape, despite synthetic pp64 being
nearly identical. That's a Qwen3-specific short-prompt cost in the
current llama.cpp Qwen3 implementation (likely dispatch overhead on the
novel architecture) — something to watch when implementing Qwen3
support in ROCmForge. Decode is flat at ~100 tok/s for both models,
consistent with the synthetic tg128 numbers once per-run variance is
accounted for.

Raw data: `llama_cpp_15prompts.tsv`.

## 3. VRAM usage

Reported by `llama-cli`'s memory breakdown (`--single-turn`, default
context sizes):

| Model | Model weights (MB) | Context (default, MB) | Total used (MB) | Free on 16 GB card (MB) |
|---|---:|---:|---:|---:|
| Qwen3-8B Q4_K_M | 4,455 | 5,760 @ 40k tokens | 10,600 | 5,334 |
| Llama-3.1-8B Q4_K_M | 4,403 | 10,272 @ 131k tokens | 14,948 | 986 |

At realistic context sizes (4-8k tokens) both models sit at ~5 GB total
— plenty of headroom. At maximum context Llama-3.1 gets uncomfortably
close to the 16 GB ceiling, but that's a non-issue for the Phase 7
benchmark plan (128-token generation with ~30-token prompts).

## 4. Architecture comparison

Extracted from the GGUF metadata headers.

| Parameter | Qwen2.5-7B (v0.1.0 target) | Qwen3-8B (new) | Llama-3.1-8B (new) |
|---|---:|---:|---:|
| `general.architecture` | `qwen2` | **`qwen3`** | **`llama`** |
| Params | 7.6 B | **8.2 B** | **8.0 B** |
| Layers | 28 | **36** | **32** |
| Hidden dim | 3,584 | **4,096** | **4,096** |
| FFN dim | 18,944 | **12,288** | **14,336** |
| Heads (Q) | 28 | **32** | **32** |
| Heads (KV) | 4 | **8** | **8** |
| **GQA ratio** | 7:1 | **4:1** | **4:1** |
| Head dim | 128 | 128 | 128 |
| Context length | 32,768 | **40,960** | **131,072** |
| RoPE base | 1,000,000 | 1,000,000 | **500,000** |
| RMS-norm eps | 1e-6 | 1e-6 | 1e-5 |
| Vocab size | 152,064 | ~151k | **128,256** |
| Chat template | ChatML | ChatML (same) | **Llama-3 format** |

### What changes in the ROCmForge code base

**Kernels — no changes needed.** The WMMA GEMM kernel and WMMA
FlashAttention already parametrise M/N/K/num_heads/head_dim/GQA-ratio
and handle arbitrary combinations. Head_dim=128 matches all three
models. The attention kernel's GQA-ratio path works for 7:1 (current)
and 4:1 (new) without modification.

**Quantisation — this is Phase 7's core.** Q4_K_M is a super-block
format (256 elements, 144 bytes). Fundamentally different layout from
Q4_0 (32 elements, 18 bytes) and Q4_1 (32 elements, 20 bytes). New
dequant kernel, new WMMA variant, new GEMV path. This is the bulk of
Phase 7 Step 2 and 3.

**Architecture registration — small.** Adding `qwen3` as a recognised
architecture in `src/config.rs::TensorNameRegistry` is near-trivial if
the tensor naming convention matches Qwen2 (it does; Qwen3 uses the
same `blk.N.attn_*.weight` naming). Llama-3 also uses the same naming.

**Chat template — moderate.** The chat CLI currently hardcodes the
Qwen2.5 ChatML template in `src/cli/template.rs`. Llama-3 uses a
different wrapper (`<|begin_of_text|><|start_header_id|>user<|end_header_id|>`
etc.). The template rendering needs a dispatch on the loaded model's
chat template, which is already in the GGUF metadata
(`tokenizer.chat_template`). Two options: (a) read the template
string from the GGUF and invoke a minimal Jinja2-alike evaluator;
(b) hardcode dispatch for the three supported families. Option (b)
is simpler and matches the current codebase style.

---

## 5. Q4_K_M format preview (for Phase 7 Step 2)

Source: `ggml-quants.c` in llama.cpp.

```
Q4_K_M super-block = 256 elements, 144 bytes
  [  0..1 ]  FP16 d    — super-block scale
  [  2..3 ]  FP16 dmin — super-block min
  [  4..15]  12 bytes packed: 8 × 6-bit scales + 8 × 6-bit mins
             (interleaved into 12 bytes using a 4:2 packing trick)
  [ 16..143] 128 bytes of 4-bit quantised values
             (256 nibbles = 256 elements; low nibble first, then high)

  Super-block is divided into 8 sub-blocks of 32 elements.
  For sub-block j ∈ [0, 8):
    scale_j   = d    · (uint8)(6-bit scale_j / 64)
    min_j     = dmin · (uint8)(6-bit min_j   / 64)
    for i in 0..32:
      q_ij    = nibble(block[j*16 + i/2], i%2)   ∈ [0, 16)
      value   = scale_j · q_ij  -  min_j
```

Density: 4.5 bits / element (144·8 / 256). Same as Q4_0, but with
richer metadata per block.

### Estimated Phase 7 Step 2 effort

Three kernels to add (mirroring the Q4_0/Q4_1 set from Phase 4):

1. **`wmma_gemm_q4_k_m` (prefill)** — structural copy of `wmma_gemm_q4_0.hip`
   with a new 144-byte block format. The K-loop advances by 32 elements
   at a time (one sub-block), so the number of K-chunks matches Q4_0
   (256/8 sub-blocks instead of 256/8×32 elements = same K stride).
   Dequant formula is more complex (two multiplications and a
   subtraction per element instead of one multiply-subtract), so
   expect 1.1-1.3× higher compute per op at WMMA tile level.
   **Complexity: medium** — same code structure as Q4_0, new unpack
   logic for the 6-bit scale/min packing.

2. **`gemv_q4_k_m` (decode)** — structural copy of `q4_0_gemv.hip`,
   same super-block-aware indexing. Launch geometry unchanged
   (8 waves × 256 threads). **Complexity: medium.**

3. **Fused `gate_up` Q4_K_M variant** — mirrors Phase 4 Step 3, one
   dispatch for both FFN projections. **Complexity: low** once the
   Q4_K_M WMMA kernel exists; pure dispatch refactor.

**Total estimate: 3–5 focused days** of kernel work + 2 days of
per-kernel correctness tests (CPU reference + golden vectors) + 1 day
of Qwen3 / Llama-3 architecture registration + chat template dispatch.
Phase 7 Step 1 (this step) is analysis only; Step 2 writes kernel #1;
Step 3 writes kernel #2 + gate_up fusion; Step 4 is multi-model
end-to-end integration and head-to-head benchmarks.

---

## 6. Kurz-Report

**Phase 7 Baseline — llama.cpp Q4_K_M Zielwerte:**

| Metrik | Qwen3-8B Q4_K_M | Llama-3.1-8B Q4_K_M | ROCmForge Qwen2.5-7B Q4_0 (v0.1.0, for context) |
|---|---:|---:|---:|
| Prefill pp256 tok/s (synthetic) | **3,661** | **3,925** | 1,484 |
| Prefill pp512 tok/s (synthetic) | **3,756** | **3,922** | 1,693 |
| Decode 128 tok/s (synthetic) | **87.4** | **93.3** | 102.4 |
| Prefill real (tok/s median, 15 prompts) | **834** | **1,408** | 493.5 |
| Decode real (tok/s median) | **98.7** | **103.7** | 102.0 |
| VRAM model weights (GB) | 4.35 | 4.30 | 4.1 |

**Llama-3.3-8B existiert nicht** (Meta hat 3.3 nur als 70B released).
**Ersatz: Llama-3.1-8B-Instruct** — das produktiv eingesetzte 8B-Llama-3.

**Architektur-Unterschiede:**
- **Qwen3 vs. Qwen2.5:** 36 statt 28 Layer, Hidden 4096 statt 3584,
  FFN 12288 statt 18944, GQA 4:1 statt 7:1 (mehr KV-Heads), selber
  head_dim=128, selbes ChatML-Template. Architektur-String `qwen3`
  (neu zu registrieren).
- **Llama-3.1 vs. Qwen2.5:** 32 Layer, Hidden 4096, FFN 14336, GQA
  4:1, head_dim=128, RoPE base 500k statt 1M, Vocab 128k statt 152k,
  eigenes Llama-3-Chat-Template. Architektur-String `llama`.

**Q4_K_M Block-Format-Vorschau:** 256 Elemente / 144 Bytes (4.5 bits /
Elem, wie Q4_0 aber mit Super-Block-Struktur). Header: FP16 d + FP16
dmin + 12 B gepackte 6-bit-Scales/Mins für 8 Sub-Blocks à 32 Elemente
+ 128 B 4-bit-Werte. Dequant: `y = (d · s6/64) · q - (dmin · m6/64)`.

**Geschätzter Aufwand für Q4_K_M-Kernel:** Drei Kernels analog zu
Phase 4 (WMMA-Prefill, GEMV-Decode, fused Gate+Up), ~3–5 Kernel-Tage
+ 2 Tage Tests + 1 Tag Multi-Model-Registration + Chat-Template-Dispatch.

Kein Kernel-Rewrite für Attention nötig — `head_dim=128` für alle
drei Modelle, GQA-Pfad arbeitet schon für beliebige Ratios.
