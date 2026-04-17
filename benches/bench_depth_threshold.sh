#!/usr/bin/env bash
# bench_depth_threshold.sh — Sweep adaptive depth-down EMA thresholds
#
# Tests: {1.2, 1.0, 0.9} × {depth=3, depth=5} × 5 code prompts + baseline
# Tiled GEMV is default-on (no env override needed)

set -euo pipefail

BINARY="target/release/rocmforge"
TARGET_MODEL="${ROCMFORGE_BENCH_TARGET_MODEL:-$HOME/models/Qwen2.5-7B-Instruct-Q4_0.gguf}"
DRAFT_MODEL="${ROCMFORGE_BENCH_DRAFT_MODEL:-$HOME/models/qwen2.5-0.5b-instruct-q4_0.gguf}"
MAX_TOKENS=128
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
OUTPUT_DIR="benches/results"

if [ ! -f "$BINARY" ]; then
    echo "Error: $BINARY not found. Run: cargo build --release --features gpu"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Adaptive Depth Threshold Sweep ==="
echo "Git SHA: $GIT_SHA"
echo "Thresholds: 1.2 (current), 1.0, 0.9"
echo "Depths: 3, 5"
echo "Prompts: code_01..code_05"
echo ""

run_single() {
    local prompt_file="$1"
    local depth="$2"
    local threshold="$3"

    local prompt
    prompt=$(cat "$prompt_file")

    local stderr
    if [ "$depth" = "0" ]; then
        stderr=$(ROCMFORGE_PROFILE_SPEC_STEP=1 "$BINARY" --model "$TARGET_MODEL" --prompt "$prompt" --max-tokens "$MAX_TOKENS" --gpu 2>&1 >/dev/null) || true
    else
        stderr=$(ROCMFORGE_SPEC_DEPTH_DOWN_EMA="$threshold" ROCMFORGE_PROFILE_SPEC_STEP=1 "$BINARY" --model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" --prompt "$prompt" --max-tokens "$MAX_TOKENS" --spec-depth "$depth" --gpu 2>&1 >/dev/null) || true
    fi

    local tok_s="0" acceptance="0" final_depth="$depth" verify_us="0"

    local stats_line
    stats_line=$(echo "$stderr" | grep "tok/s" | tail -1) || true

    if [ -n "$stats_line" ]; then
        tok_s=$(echo "$stats_line" | grep -oP '[\d.]+(?= tok/s)') || true

        if echo "$stats_line" | grep -q "speculative"; then
            acceptance=$(echo "$stats_line" | grep -oP '[\d.]+(?=%)') || true
            local d_match
            d_match=$(echo "$stats_line" | grep -oP 'depth=\d+' | grep -oP '\d+') || true
            if [ -n "$d_match" ]; then
                final_depth="$d_match"
            fi
        fi
    fi

    local profile_line
    profile_line=$(echo "$stderr" | grep "SPEC_PROFILE_JSON" | sed 's/.*\] //') || true
    if [ -n "$profile_line" ]; then
        verify_us=$(echo "$profile_line" | python3 -c "import json,sys; print(json.load(sys.stdin).get('avg_verify_us',0))" 2>/dev/null) || verify_us="0"
    fi

    echo "${tok_s}|${acceptance}|${final_depth}|${verify_us}"
}

# Baseline first
echo "--- baseline (no spec) ---"
for prompt_file in benches/prompts/code_*.txt; do
    prompt_name=$(basename "$prompt_file" .txt)
    echo -n "  $prompt_name... "
    result=$(run_single "$prompt_file" "0" "0")
    IFS='|' read -r tok_s acceptance final_depth verify_us <<< "$result"
    echo "${tok_s} tok/s"
done
echo ""

# Threshold sweep
for threshold in 1.2 1.0 0.9; do
    for depth in 3 5; do
        echo "--- depth=${depth} threshold=${threshold} ---"
        for prompt_file in benches/prompts/code_*.txt; do
            prompt_name=$(basename "$prompt_file" .txt)
            echo -n "  $prompt_name... "
            result=$(run_single "$prompt_file" "$depth" "$threshold")
            IFS='|' read -r tok_s acceptance final_depth verify_us <<< "$result"
            echo "${tok_s} tok/s (α=${acceptance}%, verify=${verify_us} μs, depth→${final_depth})"
        done
        echo ""
    done
done

echo "Done."
