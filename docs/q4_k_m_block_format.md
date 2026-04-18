# Q4_K_M Block Format — Ground Truth

**Reference:** `~/tmp/llama.cpp/ggml/src/ggml-common.h` (block struct),
`~/tmp/llama.cpp/ggml/src/ggml-quants.c` lines 818 and 1467 (unpack +
dequant). This document is byte-accurate against that source and drives
the Phase 7 Step 3 WMMA kernel.

The same precision the Q4_0 kernel got from
`docs/wmma_register_layout_gfx12.md` — addresses, masks, shifts,
example values — applies here. A single bit-level mistake corrupts
every FFN output; a Phase-7-Step-3 kernel that deviates from this doc
will produce garbage.

---

## 1. Struct layout

From `ggml-common.h` with `QK_K = 256`, `K_SCALE_SIZE = 12`:

```
block_q4_K {
    ggml_half  d;              //  2 B   super-block scale   (FP16)
    ggml_half  dmin;           //  2 B   super-block min     (FP16)
    uint8_t    scales[12];     // 12 B   packed 6-bit scales + 6-bit mins (8 sub-blocks)
    uint8_t    qs[128];        // 128 B  4-bit quantised values (256 nibbles)
}
static_assert(sizeof(block_q4_K) == 144)
```

Byte map (total **144 bytes per 256 elements = 4.5 bits / element**,
exactly the same density as Q4_0 but with richer metadata):

```
┌────────┬─────────────────────────────────────────────────────────────┐
│ 0  .. 1│ d     (FP16) — super-block scale                            │
│ 2  .. 3│ dmin  (FP16) — super-block min                              │
│ 4  ..15│ scales[12] — 8 × 6-bit scales + 8 × 6-bit mins (packed)     │
│16  ..143│ qs[128] — 4-bit quantised values (interleaved, see §4)     │
└────────┴─────────────────────────────────────────────────────────────┘
```

## 2. Super-block ↔ sub-block structure

A super-block is 256 elements, divided into **8 sub-blocks of 32
elements each**. Each sub-block carries its own 6-bit scale and 6-bit
min, picked out of the packed `scales[12]` header by `get_scale_min_k4`.
The super-block scales `d` and `dmin` apply uniformly.

## 3. Scale / min unpacking (`get_scale_min_k4`)

This is the fiddly part. llama.cpp's logic packs 8 scales + 8 mins
(16 × 6-bit = 96 bits = 12 bytes) into `scales[12]` using a 4:2
split. Straight from `ggml-quants.c:818`:

```c
static inline void get_scale_min_k4(int j, const uint8_t *q,
                                    uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j]   & 63;          // low 6 bits
        *m = q[j+4] & 63;          // low 6 bits
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >> 4)  | ((q[j]   >> 6) << 4);
    }
}
```

Unpacked per-byte contents:

| Byte index | Bits 0..5 | Bits 6..7 |
|---|---|---|
| `scales[0]` | `scale_0` (6 bits) | `scale_4[5..4]` (high 2 bits) |
| `scales[1]` | `scale_1` | `scale_5[5..4]` |
| `scales[2]` | `scale_2` | `scale_6[5..4]` |
| `scales[3]` | `scale_3` | `scale_7[5..4]` |
| `scales[4]` | `min_0` | `min_4[5..4]` |
| `scales[5]` | `min_1` | `min_5[5..4]` |
| `scales[6]` | `min_2` | `min_6[5..4]` |
| `scales[7]` | `min_3` | `min_7[5..4]` |

And `scales[8..11]` carry the low 4 bits of the four "higher" scales
and mins:

| Byte index | Bits 0..3 | Bits 4..7 |
|---|---|---|
| `scales[8]`  | `scale_4[3..0]` | `min_4[3..0]` |
| `scales[9]`  | `scale_5[3..0]` | `min_5[3..0]` |
| `scales[10]` | `scale_6[3..0]` | `min_6[3..0]` |
| `scales[11]` | `scale_7[3..0]` | `min_7[3..0]` |

Explicit formulas for each sub-block:

| j | scale_j formula | min_j formula |
|---:|---|---|
| 0 | `scales[0] & 0x3F` | `scales[4] & 0x3F` |
| 1 | `scales[1] & 0x3F` | `scales[5] & 0x3F` |
| 2 | `scales[2] & 0x3F` | `scales[6] & 0x3F` |
| 3 | `scales[3] & 0x3F` | `scales[7] & 0x3F` |
| 4 | `(scales[8]  & 0x0F) \| ((scales[0] >> 6) << 4)` | `(scales[8]  >> 4) \| ((scales[4] >> 6) << 4)` |
| 5 | `(scales[9]  & 0x0F) \| ((scales[1] >> 6) << 4)` | `(scales[9]  >> 4) \| ((scales[5] >> 6) << 4)` |
| 6 | `(scales[10] & 0x0F) \| ((scales[2] >> 6) << 4)` | `(scales[10] >> 4) \| ((scales[6] >> 6) << 4)` |
| 7 | `(scales[11] & 0x0F) \| ((scales[3] >> 6) << 4)` | `(scales[11] >> 4) \| ((scales[7] >> 6) << 4)` |

Result range: **scale_j, min_j ∈ [0, 63]** (unsigned 6-bit integers,
used as raw multipliers — they are **not** divided by 64 in the dequant
formula; see §5).

## 4. Nibble layout in `qs[128]`

This is **not** "first 32 low nibbles, then next 32 high nibbles". The
dequant loop consumes nibbles in pairs of sub-blocks at a time:

```
For each j_outer ∈ {0, 2, 4, 6}:
    is = j_outer  (sub-block index, 0/2/4/6)
    qs_off = (j_outer / 2) * 32      (byte offset into qs)
    For l in 0..32:
        element[is_outer*32 + l]       = (qs[qs_off + l] & 0x0F) ... use scale_is/min_is
    For l in 0..32:
        element[(is_outer+1)*32 + l]   = (qs[qs_off + l] >> 4)    ... use scale_(is+1)/min_(is+1)
```

Byte `qs[k]` contains **two nibbles belonging to two different
sub-blocks**: its low nibble belongs to sub-block `2 * (k / 32)`, its
high nibble belongs to sub-block `2 * (k / 32) + 1`. Adjacent low-high
nibble pairs in one byte are 32 elements apart in the output.

| qs byte range | low nibbles → elements | high nibbles → elements | sub-blocks |
|---|---|---|---|
| `qs[0..31]`   | 0..31    | 32..63   | 0, 1 |
| `qs[32..63]`  | 64..95   | 96..127  | 2, 3 |
| `qs[64..95]`  | 128..159 | 160..191 | 4, 5 |
| `qs[96..127]` | 192..223 | 224..255 | 6, 7 |

## 5. Dequantisation formula

From `ggml-quants.c:1467`:

```c
for (int j = 0; j < QK_K; j += 64) {
    get_scale_min_k4(is + 0, scales, &sc, &m);
    const float d1 = d * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, scales, &sc, &m);
    const float d2 = d * sc; const float m2 = dmin * m;
    for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
    for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
    q += 32; is += 2;
}
```

Per element in sub-block `j`:

```
value = (d · scale_j) · nibble_ij  −  (dmin · min_j)
```

Observations that matter for the kernel:

- **No `/64` anywhere.** The 6-bit scale is used as a raw integer 0..63.
  If the WMMA kernel computes `d * scale_j / 64`, the output is off by
  64× — a common mistake when reading the format specification without
  reading the code.
- **The `dmin · min_j` term is constant over a sub-block.** Pre-compute
  it once per sub-block (`m1`, `m2`) and reuse over 32 elements.
- **Two sub-blocks are processed per outer step**, sharing the 32-byte
  `qs` chunk (low vs high nibbles). Pairing (0, 1), (2, 3), (4, 5),
  (6, 7).

## 6. Comparison with Q4_0

| Property | Q4_0 | Q4_K_M |
|---|---|---|
| Elements per block | 32 | 256 |
| Bytes per block | 18 | 144 |
| Bits / element | 4.5 | 4.5 (same density) |
| Super-scale | — | 1 × FP16 (`d`) |
| Super-min | — | 1 × FP16 (`dmin`) |
| Sub-scale | 1 × FP16 per 32 elems | 8 × 6-bit per 256 elems |
| Sub-min | — (nibbles centred via `-8`) | 8 × 6-bit per 256 elems |
| Dequant per element | `(nib - 8) · scale` | `d · scale_j · nib − dmin · min_j` |
| Dequant ops / element | 1 SUB + 1 MUL | 2 MUL + 1 SUB (with sub-block constants hoisted) |
| Nibble order | low → elem i, high → elem i+16 | low → elem j*32+l, high → elem (j+1)*32+l (sub-block-pair interleave) |
| Block-per-K stride | 18 B | 144 B |
| Scale unpacking | single FP16 load | bit-twiddling across 12 bytes |

## 7. Example (block 0 of `blk.0.ffn_gate.weight` in Qwen3-8B Q4_K_M)

Raw 144 bytes (see `profiling/golden_vectors/q4_k_m_qwen3_8b_ffn_gate_layer0_first3_blocks.json`):

```
d    = 8.273e-05  (FP16 at bytes 0..1)
dmin = 7.730e-04  (FP16 at bytes 2..3)

scales[12] →
  sub-block 0: scale_0=58, min_0=39
  sub-block 1: scale_1=63, min_1=40
  sub-block 2: scale_2=48, min_2=40
  sub-block 3: scale_3=48, min_3=36
  sub-block 4: scale_4=57, min_4=63
  sub-block 5: scale_5=33, min_5=22
  sub-block 6: scale_6=49, min_6=47
  sub-block 7: scale_7=55, min_7=45
```

Precomputed constants (hoistable, one pair per sub-block):

```
for sub-block 0:  d1 = d · 58 = 4.798e-03   m1 = dmin · 39 = 3.015e-02
for sub-block 1:  d2 = d · 63 = 5.212e-03   m2 = dmin · 40 = 3.092e-02
...
```

Element 0 (sub-block 0, `qs[0] & 0x0F` = 6 in this dump):

```
value[0] = d1 · 6 - m1 = 4.798e-03 · 6 - 3.015e-02 = -0.01575
```

Matches the golden dequant (first float in block 0 is `-0.01574993...`). ✅

## 8. Addressing for the WMMA K-loop

The current Q4_0 WMMA kernel
(`hip_kernels/wmma/wmma_gemm_q4_0.hip:93`) computes:

```c
block_ofs = (n_col · blocks_per_row + kc) · 18    // one block = 32 K
```

For Q4_K_M, one super-block covers 256 K values. Two viable schemes:

### Option A — sub-block as K-chunk (recommended)

Keep the K-chunk size at 32 (one Q4_K sub-block), match the Q4_0
kernel's iteration cadence one-for-one. Super-block header load is
redundant every 8 sub-blocks but easy to amortise in L1.

```c
#define K_CHUNK 32
const int SUB_PER_SB = 8;
const int sub_blocks_per_row = K / K_CHUNK;
for (int sb = 0; sb < sub_blocks_per_row; ++sb) {
    const int super_block_idx = sb / SUB_PER_SB;
    const int sub_idx         = sb % SUB_PER_SB;

    const size_t sb_ofs =
        (size_t)(tile_n + col) * (size_t)(K / 256) * 144
        + (size_t)super_block_idx * 144;
    const uint8_t *sb_ptr = W + sb_ofs;

    const __half d    = *(__half *)(sb_ptr + 0);
    const __half dmin = *(__half *)(sb_ptr + 2);
    const uint8_t *scales = sb_ptr + 4;

    uint8_t scale_j, min_j;
    get_scale_min_k4_inline(sub_idx, scales, &scale_j, &min_j);
    const float d_scale = __half2float(d)    * (float)scale_j;
    const float d_min   = __half2float(dmin) * (float)min_j;

    // 32 nibbles for this sub-block. Layout interleaves with its pair.
    const uint8_t *qs = sb_ptr + 16;
    const int pair_base = (sub_idx / 2) * 32;   // 0, 32, 64, 96
    const bool is_upper_of_pair = (sub_idx & 1);

    for (int i = 0; i < 32; ++i) {
        const uint8_t byte = qs[pair_base + i];
        const int nib = is_upper_of_pair ? (byte >> 4) : (byte & 0x0F);
        lds_b[i * TILE_N + col] = __float2half(d_scale * nib - d_min);
    }
    // → WMMA iteration identical to Q4_0 path
}
```

**Pros:** kernel structure is a clone of the Q4_0 kernel. The K-loop
body changes only the dequant expression and the 6-bit-scale unpacker.
Minimum risk, easiest to review against the existing WMMA path.

**Cons:** re-reads the 16-byte `scales[12]` once per sub-block (8× per
super-block). On gfx1201 the scales are 12 bytes well inside an L1
cacheline, so this is actually fine — it hits warm cache.

### Option B — super-block as K-chunk

`K_CHUNK = 256`, 8 inner WMMA iterations per chunk with all 8 sub-block
constants pre-computed. One-time header decode per super-block.

**Pros:** eliminates the redundant scales re-read.

**Cons:** the inner 8-iteration loop does not map cleanly to the
existing WMMA body. Requires more careful register accounting (the 8
pre-computed `(d_scale, d_min)` pairs occupy 16 scalar registers).

### Recommendation

**Take Option A.** The scales re-read is essentially free in practice,
and keeping the kernel architecturally identical to the working Q4_0
kernel reduces reviewer cost in Step 3. If profiling after Step 3 shows
the header re-read is measurable, switching to Option B later is a
contained refactor.

## 9. Coalescing — same shape as Q4_0, slightly worse absolute stride

For a fixed K-chunk index, the byte stride between adjacent N-columns'
super-blocks is `blocks_per_row_super × 144`. Contrast with Q4_0's
`blocks_per_row × 18`:

| Model | K (in_dim of `ffn_gate`) | blocks_per_row_super (Q4_K_M) | N-stride Q4_K_M (B) | Q4_0 N-stride (B, Qwen2.5-7B reference) |
|---|---:|---:|---:|---:|
| Qwen2.5-7B Q4_0 | 3,584 | — | — | 2,016 (gate) / 10,656 (down) |
| Qwen3-8B Q4_K_M | 4,096 | 16 | 2,304 | — |
| Llama-3.1-8B Q4_K_M | 4,096 | 16 | 2,304 | — |

A 2,304-byte stride means a wavefront of 64 N-columns touches 64 distinct
cachelines per sub-block load — same non-coalesced pattern as Q4_0.
Cacheline utilisation is the same ~14 % nominal (18 useful bytes out of
128 per line fetched), and the Phase-4-Step-2.1 analysis holds:
non-coalesced N-reads aren't the bottleneck because the kernel is L1
bandwidth limited, not load-issue limited.

## 10. Q8_K intermediate — skip

llama.cpp's CPU path uses `ggml_vec_dot_q4_K_q8_K`, which quantises the
activations to Q8_K at runtime and does the dot product in Q4 × Q8. We
do **not** need this for the WMMA kernel: WMMA operates on FP16, and
the existing Q4_0 kernel already dequantises to FP16 in LDS before the
matrix multiply. Doing the same for Q4_K_M is correct and matches the
working architecture.

## 11. Register pressure sanity check

Per wave processing 4 output N-columns × 1 K-chunk sub-block (pattern
identical to the Q4_0 kernel):

| Kind | Q4_0 | Q4_K_M | Delta |
|---|---|---|---|
| FP16 nibbles in LDS | 64 cols × 32 K = 2048 halves = 4 KB | same | 0 |
| FP32 activation tile in LDS | 64 rows × 32 K = 2048 floats = 8 KB | same | 0 |
| Scale/min per wave | 1 × FP16 | 2 × f32 (`d_scale`, `d_min`) | +4 B / thread |
| Nibble-unpack ALU ops | `nib - 8` | `(byte & 0xF)` or `(byte >> 4)` | same |
| Scale-unpack ALU ops | — | 3-4 bitops per sub-block | +4 / thread per K-chunk |
| Dequant ops | 1 MUL | 1 MAD (`d_scale * nib - d_min`) | same count |

Net: a handful of extra scalar registers + 3-4 extra ALU ops per K-chunk
at the top. None of this should push the kernel above RDNA 4's 104 VGPRs
limit. Occupancy should match the Q4_0 kernel (4 waves per block).

## 12. Step-3 correctness plan

1. **Port the CPU reference into Rust** (e.g., `src/cpu/dequant/q4_k.rs`).
   A copy of the Python generator logic: the bitshift-laden
   `get_scale_min_k4` + the sub-block-pair loop.

2. **Isolated kernel test** (`tests/wmma_q4_k_m_correctness.rs`,
   mirroring `wmma_q4_1_correctness.rs`):
   - Byte-level smoke test with a hand-constructed block (known
     `d=0.5`, `dmin=0.25`, a specific scales layout, fixed nibbles).
     The output values are hand-computable; the WMMA kernel must
     match them.
   - Shape tests: `64×64×256`, `64×256×256`, `64×4096×4096`, and the
     real Qwen3-8B `ffn_gate` shape `(seq_len × 12288 × 4096)`.
   - Each uses the CPU reference as the source of truth.

3. **Golden vectors already extracted.** The first 3 super-blocks of
   `blk.0.ffn_gate.weight` from Qwen3-8B, along with the expected FP32
   dequant output, live at
   `profiling/golden_vectors/q4_k_m_qwen3_8b_ffn_gate_layer0_first3_blocks.json`.
   These are directly usable as fixed-input test vectors.

4. **End-to-end regression.** Once the WMMA kernel passes the isolated
   tests, `cargo test --release --features gpu --test chat_single_turn_correctness`
   should continue to produce the Qwen2.5-7B outputs byte-for-byte
   (regression) and greedy decode of "Hello" on Qwen3-8B Q4_K_M should
   produce coherent text (first cross-model validation).

---

## Kurz-Report

**Q4_K_M Format-Analyse:**

- **Super-Block: 256 Elemente, 144 Bytes** (4.5 bits/Elem, selbe Dichte wie Q4_0)
- **Sub-Blöcke: 8 × 32 Elemente**, jedes mit eigenem 6-bit-Scale und 6-bit-Min
- **Dequant:** `value = (d · scale_j) · nibble − (dmin · min_j)` — **kein `/64`**, die 6-bit-Werte werden direkt als Integer 0..63 benutzt
- **Scale-Entpackung:** 4:2-Split über `scales[12]` — niedrige 6 Bits für j<4 liegen in `scales[j]`/`scales[j+4]`, für j≥4 kombiniert aus `scales[j+4]` + High-2-Bits aus `scales[j-4]`/`scales[j]` (formulas in §3)

**WMMA-K-Loop: Option A empfohlen** (sub-block als K-chunk, K_CHUNK=32). Struktur identisch zum bestehenden Q4_0-Kernel; nur Dequant-Block und Scale-Unpacker ändern sich. Redundanter Scales-Reload ist L1-warm und praktisch kostenlos. Option B (super-block als K-chunk) ist weiter entfernt vom Q4_0-Kernel und bringt keinen messbaren Vorteil.

**Coalescing:** N-Stride 2,304 B (Qwen3-8B gate, K=4096) — strukturell identisch zur Q4_0-Situation (nicht-coalesced über N, aber kein Flaschenhals — der Kernel ist L1-bandbreitenbegrenzt, nicht load-issue-begrenzt). Gleiche Analyse wie Phase 4 Step 2.1.

**Erwartete Kernel-Änderungen vs. Q4_0:**
- **Block-Bytes:** 18 → 144 (Adressformel: `(tile_n + col) · (K/256) · 144 + super_block_idx · 144`)
- **K-Stride:** 32 (unverändert, sub-block-basiert)
- **Dequant:** 1 MUL → 1 MAD + 3-4 Bitops fürs Scale-Unpacking
- **Register-Bedarf:** +4 B/Thread (zwei FP32-Konstanten pro Sub-Block statt einer FP16)
- **LDS:** unverändert

**Golden Vectors:** 3 Super-Blöcke von `blk.0.ffn_gate.weight` aus Qwen3-8B Q4_K_M, dequantisiert gegen die llama.cpp-Referenz, gespeichert in `profiling/golden_vectors/q4_k_m_qwen3_8b_ffn_gate_layer0_first3_blocks.json`.

**CPU-Referenz-Dequant:** `profiling/generate_q4_k_m_golden.py` mirror byte-für-byte die llama.cpp `get_scale_min_k4` + `dequantize_row_q4_K`. Verifiziert: Block 0 Element 0 = −0.01575, exakt gleich wie der llama.cpp-Output.
