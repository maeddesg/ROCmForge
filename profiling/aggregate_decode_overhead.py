#!/usr/bin/env python3
"""Aggregate per-step decode profiling trace events emitted by
`ROCMFORGE_PROFILE_DECODE_OPS=1`.

Usage:
    python3 profiling/aggregate_decode_overhead.py \
        profiling/results/decode_profiling_128steps.log \
        --ground-truth-ms-per-token 9.76
"""
import argparse
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
KV = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([-0-9]+)")

STAGE_FIELDS = [
    "attn_norm_us", "qkv_us", "q_rope_us", "k_rope_us", "kv_write_us",
    "attention_us", "attn_proj_us", "attn_residual_us", "ffn_norm_us",
    "gate_up_us", "ffn_down_us", "ffn_residual_us", "logits_norm_us",
    "logits_proj_us", "argmax_us",
]
META_FIELDS = ["pos", "wall_us", "launches_approx",
               "layer_invocations", "tail_invocations"]


def parse(path):
    rows = []
    for line in open(path):
        line = ANSI.sub("", line)
        if "Decode step profiling" not in line:
            continue
        kv = dict(KV.findall(line))
        row = {k: int(kv.get(k, "0")) for k in STAGE_FIELDS + META_FIELDS}
        rows.append(row)
    return rows


def median_row(rows, field):
    vals = [r[field] for r in rows]
    return statistics.median(vals) if vals else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--ground-truth-ms-per-token", type=float, required=True,
                   help="Unprofiled ms/token — the reference for rescaling")
    p.add_argument("--warmup-steps", type=int, default=1,
                   help="Number of initial steps to discard as warmup")
    args = p.parse_args()

    rows = parse(args.log)
    if not rows:
        print("No 'Decode step profiling' events found.", file=sys.stderr)
        sys.exit(1)
    print(f"Steps parsed: {len(rows)}")

    hot = rows[args.warmup_steps:]
    print(f"Steady-state steps (after {args.warmup_steps} warmup): {len(hot)}")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Table 1 — per-stage medians, raw and rescaled to ground truth
    # ────────────────────────────────────────────────────────────────────
    profiled_wall_median = median_row(hot, "wall_us")
    ground_truth_us = int(args.ground_truth_ms_per_token * 1000)
    scale = ground_truth_us / profiled_wall_median if profiled_wall_median else 1.0
    sync_overhead_us = profiled_wall_median - ground_truth_us
    sync_overhead_pct = (sync_overhead_us / ground_truth_us) * 100.0 if ground_truth_us else 0

    print("## Table 1 — Per-stage breakdown (steady-state median)")
    print()
    print("Wall-clock per step:")
    print(f"  Profiled median:        {profiled_wall_median/1000:.2f} ms")
    print(f"  Ground truth:           {ground_truth_us/1000:.2f} ms")
    print(f"  Sync overhead:          +{sync_overhead_us/1000:.2f} ms  ({sync_overhead_pct:+.0f}%)")
    print(f"  Rescale factor applied: ×{scale:.3f}  (to distribute profile into GT budget)")
    print()
    print(f"| Stage | Profiled µs | Anteil % | **Rescaled µs** (ground truth) |")
    print(f"|---|---:|---:|---:|")
    sum_profiled = 0
    for f in STAGE_FIELDS:
        v = median_row(hot, f)
        sum_profiled += v
    for f in STAGE_FIELDS:
        v = median_row(hot, f)
        pct = (v / sum_profiled * 100.0) if sum_profiled else 0.0
        rescaled = int(v * scale)
        nice = f.replace("_us", "")
        print(f"| {nice} | {v} | {pct:.1f} | **{rescaled}** |")
    rescaled_total = int(sum_profiled * scale)
    print(f"| **Sum** | **{sum_profiled}** | **100.0** | **{rescaled_total}** |")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Table 2 — categories
    # ────────────────────────────────────────────────────────────────────
    cats = {
        "GEMV (Q/K/V fused + O + Gate+Up + Down + LM-head)": [
            "qkv_us", "attn_proj_us", "gate_up_us", "ffn_down_us", "logits_proj_us"],
        "Attention (decode)":         ["attention_us"],
        "Norm (attn + FFN + logits)": ["attn_norm_us", "ffn_norm_us", "logits_norm_us"],
        "RoPE":                       ["q_rope_us", "k_rope_us"],
        "KV-Cache Write":             ["kv_write_us"],
        "Residual (post-attn + post-FFN)": ["attn_residual_us", "ffn_residual_us"],
        "Sampling (argmax)":          ["argmax_us"],
    }
    print("## Table 2 — Categories (rescaled to ground truth)")
    print()
    print("| Category | Rescaled µs | Share % |")
    print("|---|---:|---:|")
    covered = 0
    for label, fields in cats.items():
        v = sum(int(median_row(hot, f) * scale) for f in fields)
        covered += v
        pct = (v / ground_truth_us * 100.0)
        print(f"| {label} | {v} | {pct:.1f} |")
    unaccounted = max(ground_truth_us - covered, 0)
    print(f"| Unaccounted (launch overhead + host-side orchestration) | {unaccounted} | "
          f"{unaccounted/ground_truth_us*100:.1f} |")
    print(f"| **Total** | **{ground_truth_us}** | **100.0** |")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Table 3 — launch overhead
    # ────────────────────────────────────────────────────────────────────
    launches = median_row(hot, "launches_approx")
    print("## Table 3 — Launch-overhead analysis")
    print()
    print(f"| Metric | Value |")
    print(f"|---|---:|")
    print(f"| Launches per token (approx) | {launches} |")
    for latency_us in [3, 5]:
        est = launches * latency_us
        pct = est / ground_truth_us * 100
        print(f"| Launch overhead @ {latency_us} µs each | {est} µs ({pct:.1f}%) |")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Table 5 — KV-cache scaling
    # ────────────────────────────────────────────────────────────────────
    print("## Table 5 — Scaling vs. KV-cache length")
    print()
    print("| Step pos | Wall µs | Attention µs |")
    print("|---:|---:|---:|")
    for target_pos in [2, 10, 32, 64, 96, 127]:
        best = min(rows, key=lambda r: abs(r["pos"] - target_pos)) if rows else None
        if best:
            print(f"| {best['pos']} | {best['wall_us']} | {best['attention_us']} |")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Table 6 — sync-overhead quantification
    # ────────────────────────────────────────────────────────────────────
    print("## Table 6 — Sync-overhead quantification")
    print()
    print(f"| Measurement | ms/token | Δ vs ground truth |")
    print(f"|---|---:|---:|")
    print(f"| Ground truth (unprofiled) | {args.ground_truth_ms_per_token:.2f} | — |")
    print(f"| Per-op profiling | {profiled_wall_median/1000:.2f} | "
          f"{sync_overhead_us/1000:+.2f} ms ({sync_overhead_pct:+.0f}%) |")


if __name__ == "__main__":
    main()
