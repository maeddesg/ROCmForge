# Q6_K block format — byte-level ground truth

Authoritative source: `ggml/src/ggml-common.h::block_q6_K` and
`ggml/src/ggml-quants.c::dequantize_row_q6_K` in llama.cpp.

## 1. Struct layout (210 bytes, 256 elements)

```c
typedef struct {
    uint8_t ql[128];     // low 4 bits of quants           -- bytes   0..127
    uint8_t qh[64];      // high 2 bits of quants          -- bytes 128..191
    int8_t  scales[16];  // per-sub-block scales (signed)  -- bytes 192..207
    ggml_half d;         // super-block scale (FP16)       -- bytes 208..209
} block_q6_K;            // total 210 B
```

Effective rate: `210 * 8 / 256 = 6.5625` bits per weight.

**Note:** `d` sits at the end of the block, NOT the beginning. Reading `d`
requires `memcpy(&d_h, block + 208, 2)`.

## 2. Sub-block structure

A super-block of 256 elements is split into **16 sub-blocks of 16 elements
each**. Each sub-block `j ∈ [0, 16)` owns one signed int8 scale `scales[j]`.

For element `e ∈ [0, 256)`: `sub_idx = e / 16`, `scale = scales[sub_idx]`.

## 3. 6-bit quant reconstruction

Each quant is 6 bits. The low 4 bits live in `ql`, the high 2 bits in `qh`.
The layout is **not byte-consecutive** — `qh` packs the high bits of four
elements spread across 32-element strides.

### 3.1 Two 128-element halves

The 256-element super-block is processed in two halves:

```
half = 0:  elements 0..127    ql[0..63]    qh[0..31]   scales[0..7]
half = 1:  elements 128..255  ql[64..127]  qh[32..63]  scales[8..15]
```

Within each half, the dequant loop iterates `l ∈ [0, 32)` and produces
**four output positions per l**.

### 3.2 The four positions per l (ground truth, unchanged from llama.cpp)

For `l ∈ [0, 32)` inside a half, the four output positions are at
offsets `l + {0, 32, 64, 96}` within the half. Each uses a different
slice of `ql[l]` / `ql[l+32]` and a different 2-bit slice of `qh[l]`:

| Position | ql byte      | ql nibble  | qh bits from `qh[l]` | scale index `is` |
|----------|--------------|------------|----------------------|------------------|
| `l +  0` | `ql[l]`      | low 4 bits | bits 0..1            | `is + 0`         |
| `l + 32` | `ql[l + 32]` | low 4 bits | bits 2..3            | `is + 2`         |
| `l + 64` | `ql[l]`      | high 4 bits| bits 4..5            | `is + 4`         |
| `l + 96` | `ql[l + 32]` | high 4 bits| bits 6..7            | `is + 6`         |

Where `is = l / 16` (0 for `l < 16`, 1 for `l ≥ 16`), and the scale pointer
`sc` advances by 8 between halves.

The full reference loop (from `ggml-quants.c:1888`):

```c
for (int n = 0; n < QK_K; n += 128) {       // two halves
    for (int l = 0; l < 32; ++l) {
        int is = l / 16;
        const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l +  0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
    }
    y  += 128;
    ql += 64;
    qh += 32;
    sc += 8;
}
```

### 3.3 Per-element closed-form address (useful for cooperative dequant)

Given a target element index `e ∈ [0, 256)` relative to the super-block:

```
n_half   = e / 128                 // 0 or 1
e_half   = e % 128                 // 0..127
which_q  = e_half / 32             // 0..3
l        = e_half % 32             // 0..31
sub_idx  = e / 16                  // 0..15  (for scales[])

ql_idx   = n_half * 64  + (which_q & 1) * 32 + l    // 0..127
qh_idx   = 128 + n_half * 32 + l                    // 128..191
ql_byte  = block[ql_idx]
qh_byte  = block[qh_idx]

is_high  = (which_q >= 2)
ql_nib   = is_high ? (ql_byte >> 4) : (ql_byte & 0xF)
qh_bits  = (qh_byte >> (2 * which_q)) & 0x3
q6       = ql_nib | (qh_bits << 4)                  // 0..63
signed_q = (int)q6 - 32                              // -32..31

scale_j  = (int8_t)block[192 + sub_idx]
d        = *(ggml_half*)(block + 208)
value    = d * (float)scale_j * (float)signed_q
```

### 3.4 Dequant formula vs Q4_K

| | Q4_K | Q6_K |
|---|---|---|
| Bits per element | 4 | 6 |
| Quant range | 0..15 (unsigned nibble) | -32..31 (signed q6) |
| Scale type | 6-bit packed (`get_scale_min_k4`) | int8 direct |
| Sub-block count | 8 (32 elems each) | 16 (16 elems each) |
| Min (zero point) | Yes — `dmin * min_j` per sub-block | **No** |
| Formula | `d · scale_j · nib − dmin · min_j` | `d · scale_j · (q6 − 32)` |

Q6_K is numerically simpler (no affine offset) at the cost of 1.5× more
quant bits.

## 4. Addressing for a WMMA K_CHUNK of 32 elements

A WMMA kernel processing `K_CHUNK = 32` elements per iteration (same layout
as Q4_K) decomposes as follows. For chunk `kc` into super-block
`super_block_idx = kc / 8`, `chunk_in_sb = kc & 7`:

```
chunk_base = chunk_in_sb * 32              // 0, 32, 64, ..., 224 inside the super-block
```

The 32 elements of the chunk are `[chunk_base, chunk_base + 32)`.

With the Q4_K cooperative-thread scheme (128 threads, each thread owns
1 column × 1 half-of-chunk = 16 K-values), thread `tid` maps to:

```
col     = tid >> 1        // 0..63   (column within the TILE_N=64 N-tile)
half    = tid & 1         // 0 or 1
k_base  = half * 16       // 0 or 16 — start offset within the K_CHUNK
```

The 16 K-values this thread writes cover elements
`[chunk_base + k_base, chunk_base + k_base + 16)` inside the super-block.
Per the analysis in §3.3, these 16 elements have:

- Constant `sub_idx` = `(chunk_base + k_base) / 16`  →  **single scale lookup**
- Constant `n_half` and `which_q`  →  **single ql/qh strip**

So the thread does one scale load, then 16 byte-unpacks to produce
16 FP16 values and writes them to `lds_b[k * TILE_N + col]`.

## 5. Qwen3-8B / Llama-3.1-8B shapes (Q4_K_M mixture)

Tensors stored as Q6_K in the reference GGUFs:

| Model | Tensor | Shape (N × K) | K / 256 | N / 64 |
|---|---|---|---|---|
| Qwen3-8B | `blk.*.attn_v.weight` | 1024 × 4096 | 16 | 16 |
| Qwen3-8B | `blk.*.ffn_down.weight` | 4096 × 12288 | 48 | 64 |
| Qwen3-8B | `output.weight` (LM head) | 151936 × 4096 | 16 | 2374 |
| Llama-3.1-8B | `blk.*.attn_v.weight` | 1024 × 4096 | 16 | 16 |
| Llama-3.1-8B | `blk.*.ffn_down.weight` | 4096 × 14336 | 56 | 64 |

All Q6_K shapes satisfy `K % 256 == 0` and `N % 64 == 0`, so a WMMA GEMM
kernel with `TILE_M = TILE_N = 64`, `TILE_K = 16`, `K_CHUNK = 32` can
dispatch directly without padding beyond the usual prefill M padding to 64.
