#!/usr/bin/env python3
"""
Parse the per-layer + top-level tracing events written by
`ROCMFORGE_PROFILE_PREFILL_OPS=1` and produce aggregated tables for the
Phase 4 Step 1 analysis report.

Usage:
    python3 profiling/aggregate_prefill_overhead.py \
        profiling/results/prefill_overhead_pp256.log
    python3 profiling/aggregate_prefill_overhead.py \
        profiling/results/prefill_overhead_pp256.log --json
"""
import argparse
import json
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
KV = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([-0-9A-Za-z_.]+)")

LAYER_OPS = [
    "norm_pre_attn_us",
    "q_proj_us",
    "k_proj_us",
    "v_proj_us",
    "qkv_bias_us",
    "rope_q_us",
    "rope_k_us",
    "kv_write_us",
    "attention_us",
    "o_proj_us",
    "residual_attn_us",
    "norm_pre_ffn_us",
    "gate_proj_us",
    "up_proj_us",
    "silu_mul_us",
    "down_proj_us",
    "residual_ffn_us",
]

TOPLEVEL_OPS = ["embed_us", "layers_us", "final_norm_us", "lm_head_us",
                "total_us", "unaccounted_us"]


def parse_log(path):
    layers = []
    toplevel = None
    with open(path, "r") as f:
        for line in f:
            line = ANSI.sub("", line).strip()
            if "Prefill layer profiling" in line:
                kv = dict(KV.findall(line))
                layers.append({k: int(kv[k]) for k in kv
                               if k in LAYER_OPS + ["layer", "seq_len",
                                                     "total_us"]})
            elif "Prefill top-level profiling" in line:
                kv = dict(KV.findall(line))
                toplevel = {k: int(kv[k]) for k in kv
                            if k in TOPLEVEL_OPS + ["seq_len"]}
    return layers, toplevel


def sum_op(layers, op):
    return sum(l.get(op, 0) for l in layers)


def median_op(layers, op):
    vals = [l.get(op, 0) for l in layers]
    return int(statistics.median(vals)) if vals else 0


def exclude_layer0(layers):
    # Layer 0 carries JIT + lazy buffer init; the hot-loop median is a better
    # representative of the per-layer steady-state cost.
    return [l for l in layers if l.get("layer") != 0]


def print_table1(layers, label):
    print(f"\n### Tabelle 1 — Per-Operation Aggregation ({label}, 28 Layer summiert)\n")
    print("| Operation | Summe (ms) | Anteil (%) | Pro Layer median (µs) |")
    print("|---|---:|---:|---:|")
    total_layer_sum = 0
    for op in LAYER_OPS:
        total_layer_sum += sum_op(layers, op)
    hot = exclude_layer0(layers)
    for op in LAYER_OPS:
        s_us = sum_op(layers, op)
        s_ms = s_us / 1000.0
        pct = (s_us / total_layer_sum * 100) if total_layer_sum else 0.0
        med = median_op(hot, op) if hot else 0
        pretty = op.replace("_us", "")
        print(f"| {pretty} | {s_ms:.2f} | {pct:.1f} | {med} |")
    print(f"| **Layer-Summe** | **{total_layer_sum/1000:.2f}** | **100.0** | — |")


def print_table2(toplevel, label):
    print(f"\n### Tabelle 2 — Top-Level-Breakdown ({label})\n")
    print("| Komponente | Zeit (ms) | Anteil (%) |")
    print("|---|---:|---:|")
    total = toplevel.get("total_us", 0)
    for name, key in [("Embedding", "embed_us"),
                      ("28 Layer (gesamt)", "layers_us"),
                      ("Final Norm", "final_norm_us"),
                      ("LM Head", "lm_head_us"),
                      ("Unaccounted", "unaccounted_us")]:
        us = toplevel.get(key, 0)
        pct = (us / total * 100) if total else 0.0
        print(f"| {name} | {us/1000:.2f} | {pct:.1f} |")
    print(f"| **Gesamt** | **{total/1000:.2f}** | **100.0** |")


def print_table3(layers, toplevel, label):
    print(f"\n### Tabelle 3 — Overhead-Kategorien ({label})\n")
    groups = {
        "GEMM (Q/K/V/O/Gate/Up/Down + LM Head)": (
            ["q_proj_us", "k_proj_us", "v_proj_us", "o_proj_us",
             "gate_proj_us", "up_proj_us", "down_proj_us"],
            toplevel.get("lm_head_us", 0),
        ),
        "Attention (WMMA)": (["attention_us"], 0),
        "Norm (RMS × 2 pro Layer + Final)": (
            ["norm_pre_attn_us", "norm_pre_ffn_us"],
            toplevel.get("final_norm_us", 0),
        ),
        "RoPE (Q + K)": (["rope_q_us", "rope_k_us"], 0),
        "Residual (post-attn + post-FFN)": (
            ["residual_attn_us", "residual_ffn_us"], 0),
        "Activation (SiLU + Mul)": (["silu_mul_us"], 0),
        "KV-Cache-Write": (["kv_write_us"], 0),
        "QKV-Bias": (["qkv_bias_us"], 0),
        "Embedding": ([], toplevel.get("embed_us", 0)),
    }
    total = toplevel.get("total_us", 0)
    print("| Kategorie | Summe (ms) | Anteil Prefill (%) |")
    print("|---|---:|---:|")
    covered = 0
    for label_g, (op_list, extra_us) in groups.items():
        s_us = sum(sum_op(layers, o) for o in op_list) + extra_us
        covered += s_us
        pct = (s_us / total * 100) if total else 0.0
        print(f"| {label_g} | {s_us/1000:.2f} | {pct:.1f} |")
    launch = max(total - covered, 0)
    print(f"| Launch/Dispatch + Unaccounted | {launch/1000:.2f} | "
          f"{(launch/total*100 if total else 0.0):.1f} |")
    print(f"| **Gesamt** | **{total/1000:.2f}** | **100.0** |")


def print_scaling_table(layers64, top64, layers256, top256):
    print("\n### Tabelle 4 — pp64 vs. pp256 Skalierung\n")
    print("| Operation | pp64 (ms) | pp256 (ms) | Faktor | Linear wäre 4× |")
    print("|---|---:|---:|---:|---|")
    rows = [("Embedding", top64["embed_us"], top256["embed_us"])]
    for op in LAYER_OPS:
        s64 = sum_op(layers64, op)
        s256 = sum_op(layers256, op)
        rows.append((op.replace("_us", ""), s64, s256))
    rows.append(("Final Norm", top64["final_norm_us"], top256["final_norm_us"]))
    rows.append(("LM Head", top64["lm_head_us"], top256["lm_head_us"]))
    rows.append(("Gesamt", top64["total_us"], top256["total_us"]))
    for name, a, b in rows:
        fac = (b / a) if a else float("nan")
        marker = "≈ linear" if a and 3.0 <= fac <= 5.0 else (
            "sub-linear" if a and fac < 3.0 else (
                "super-linear" if a and fac > 5.0 else "—"))
        print(f"| {name} | {a/1000:.2f} | {b/1000:.2f} | "
              f"{fac:.2f}× | {marker} |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="+")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    datasets = {}
    for path in args.log:
        layers, top = parse_log(path)
        datasets[path] = (layers, top)
        label = "pp64" if "pp64" in path else (
            "pp256" if "pp256" in path else path)
        if args.json:
            continue
        print(f"\n## {label} ({path})")
        print(f"- layers parsed: {len(layers)}")
        print(f"- top-level: {top}")
        if layers:
            print_table1(layers, label)
            print_table2(top, label)
            print_table3(layers, top, label)
    if args.json:
        out = {}
        for path, (layers, top) in datasets.items():
            out[path] = {"layers": layers, "toplevel": top}
        print(json.dumps(out, indent=2))
        return
    # Cross-prompt scaling if both present.
    pp64 = [v for k, v in datasets.items() if "pp64" in k]
    pp256 = [v for k, v in datasets.items() if "pp256" in k]
    if pp64 and pp256:
        print_scaling_table(pp64[0][0], pp64[0][1], pp256[0][0], pp256[0][1])


if __name__ == "__main__":
    main()
