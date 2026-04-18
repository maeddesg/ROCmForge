#!/usr/bin/env python3
"""Q4_K_M CPU reference dequantisation, byte-identical to ggml-quants.c.

Usage:
    python3 profiling/generate_q4_k_m_golden.py \
        ~/models/Qwen3-8B-Q4_K_M.gguf \
        blk.0.ffn_gate.weight \
        3

Generates a JSON file under `profiling/golden_vectors/` with the raw
bytes of the first N super-blocks plus the expected FP32 output of
`dequantize_row_q4_K`. This is the ground-truth reference against
which the Phase 7 Step 3 Q4_K_M WMMA kernel will be validated.

The dequant mirrors:
- `get_scale_min_k4` in `ggml-quants.c:818`
- `dequantize_row_q4_K` in `ggml-quants.c:1467`

If this generator ever drifts from llama.cpp's output, the kernel
tests will drift too — re-sync against the current ggml-quants.c.
"""
import argparse
import json
import struct
import sys

from gguf import GGUFReader


def fp16_to_fp32(b: bytes) -> float:
    return struct.unpack("<e", b)[0]


def get_scale_min_k4(j: int, q: bytes):
    """Unpack the 6-bit scale and 6-bit min for sub-block j ∈ [0, 8).

    Mirrors ggml-quants.c:818 byte-for-byte.
    """
    if j < 4:
        d = q[j] & 63
        m = q[j + 4] & 63
    else:
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)
        m = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4)
    return d, m


def dequant_block_q4_k(block: bytes):
    """Dequantise one 144-byte Q4_K super-block to 256 FP32 values.

    Mirrors ggml-quants.c:1467 byte-for-byte.
    """
    assert len(block) == 144, f"expected 144 bytes, got {len(block)}"
    d = fp16_to_fp32(block[0:2])
    dmin = fp16_to_fp32(block[2:4])
    scales = block[4:16]
    qs = block[16:144]

    out = []
    qs_off = 0
    is_ = 0
    for _ in range(0, 256, 64):
        sc0, m0 = get_scale_min_k4(is_ + 0, scales)
        d1, m1 = d * sc0, dmin * m0
        sc1, m1u = get_scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc1, dmin * m1u
        for l in range(32):
            out.append(d1 * (qs[qs_off + l] & 0x0F) - m1)
        for l in range(32):
            out.append(d2 * (qs[qs_off + l] >> 4) - m2)
        qs_off += 32
        is_ += 2
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("tensor")
    p.add_argument("n_blocks", type=int, default=3, nargs="?")
    p.add_argument("-o", "--output",
                   default="profiling/golden_vectors/q4_k_m_golden.json")
    args = p.parse_args()

    reader = GGUFReader(args.model)
    tensor = next((x for x in reader.tensors if x.name == args.tensor), None)
    if tensor is None:
        print(f"Tensor {args.tensor} not found", file=sys.stderr)
        sys.exit(1)

    raw = bytes(tensor.data.tobytes())
    n_blocks_total = tensor.n_elements // 256
    assert len(raw) == n_blocks_total * 144, (
        f"raw {len(raw)} B != expected {n_blocks_total * 144} B")

    blocks = []
    for bi in range(args.n_blocks):
        b = raw[bi * 144:(bi + 1) * 144]
        floats = dequant_block_q4_k(b)
        sub = [dict(j=j, scale_6bit=s[0], min_6bit=s[1])
               for j in range(8) for s in [get_scale_min_k4(j, b[4:16])]]
        blocks.append({
            "block_index": bi,
            "raw_hex": b.hex(),
            "d_fp32": fp16_to_fp32(b[0:2]),
            "dmin_fp32": fp16_to_fp32(b[2:4]),
            "scales_bytes": list(b[4:16]),
            "unpacked_scale_min_per_subblock": sub,
            "dequantized_first_64": floats[:64],
            "dequantized_last_8": floats[-8:],
        })

    with open(args.output, "w") as f:
        json.dump({
            "source_model": args.model,
            "source_tensor": args.tensor,
            "source_tensor_shape": [int(x) for x in tensor.shape],
            "quant_format": "Q4_K_M",
            "block_size_bytes": 144,
            "elements_per_block": 256,
            "reference": "ggml-quants.c::dequantize_row_q4_K + get_scale_min_k4",
            "blocks": blocks,
        }, f, indent=2)
    print(f"wrote {args.output} ({len(blocks)} blocks)")


if __name__ == "__main__":
    main()
