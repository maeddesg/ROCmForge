#!/usr/bin/env bash
# bench_tiled_sweep.sh — Full benchmark sweep: tiled vs non-tiled across depths
#
# 15 prompts × {baseline, tiled-off d1/d3/d5, tiled-on d1/d3/d5} = 105 runs
# Output: benches/results/tiled_sweep_{sha}.json

set -euo pipefail

BINARY="target/release/rocmforge"
TARGET_MODEL="${ROCMFORGE_BENCH_TARGET_MODEL:-$HOME/models/Qwen2.5-7B-Instruct-Q4_0.gguf}"
DRAFT_MODEL="${ROCMFORGE_BENCH_DRAFT_MODEL:-$HOME/models/qwen2.5-0.5b-instruct-q4_0.gguf}"
MAX_TOKENS=128
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +%s)
OUTPUT_DIR="benches/results"
OUTPUT_FILE="${OUTPUT_DIR}/tiled_sweep_${GIT_SHA}.json"

if [ ! -f "$BINARY" ]; then
    echo "Error: $BINARY not found. Run: cargo build --release --features gpu"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Tiled GEMV Benchmark Sweep ==="
echo "Git SHA: $GIT_SHA"
echo "Target: $TARGET_MODEL"
echo "Draft: $DRAFT_MODEL"
echo "Max tokens: $MAX_TOKENS"
echo "Output: $OUTPUT_FILE"
echo ""

run_single() {
    local prompt_file="$1"
    local depth="$2"
    local tiled="$3"
    local prompt
    prompt=$(cat "$prompt_file")

    local env_vars=""
    if [ "$tiled" = "0" ]; then
        env_vars="ROCMFORGE_DISABLE_TILED_GEMV=1"
    fi

    local stderr
    if [ "$depth" = "0" ]; then
        stderr=$(env $env_vars "$BINARY" --model "$TARGET_MODEL" --prompt "$prompt" --max-tokens "$MAX_TOKENS" --gpu 2>&1 >/dev/null) || true
    else
        stderr=$(env $env_vars ROCMFORGE_PROFILE_SPEC_STEP=1 "$BINARY" --model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" --prompt "$prompt" --max-tokens "$MAX_TOKENS" --spec-depth "$depth" --gpu 2>&1 >/dev/null) || true
    fi

    local tok_s="0" acceptance="0" avg_accepted="0" n_steps="0" n_generated="0" final_depth="$depth" verify_us="0"

    local stats_line
    stats_line=$(echo "$stderr" | grep "tok/s" | tail -1) || true

    if [ -n "$stats_line" ]; then
        n_generated=$(echo "$stats_line" | grep -oP '^\d+') || true
        tok_s=$(echo "$stats_line" | grep -oP '[\d.]+(?= tok/s)') || true

        if echo "$stats_line" | grep -q "speculative"; then
            acceptance=$(echo "$stats_line" | grep -oP '[\d.]+(?=%)') || true
            avg_accepted=$(echo "$stats_line" | grep -oP 'avg [\d.]+' | grep -oP '[\d.]+') || true
            n_steps=$(echo "$stats_line" | grep -oP 'over \d+ steps' | grep -oP '\d+') || true
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

    echo "${tok_s}|${acceptance}|${avg_accepted}|${n_steps}|${n_generated}|${final_depth}|${verify_us}"
}

RESULTS="["
FIRST=true

# mode_name:depth:tiled
MODES="baseline:0:0 spec_d1:1:0 spec_d3:3:0 spec_d5:5:0 spec_d1_tiled:1:1 spec_d3_tiled:3:1 spec_d5_tiled:5:1"

for mode_spec in $MODES; do
    IFS=':' read -r mode_name depth tiled <<< "$mode_spec"
    echo "--- $mode_name ---"

    for prompt_file in benches/prompts/*.txt; do
        prompt_name=$(basename "$prompt_file" .txt)
        echo -n "  $prompt_name... "

        result=$(run_single "$prompt_file" "$depth" "$tiled")
        IFS='|' read -r tok_s acceptance avg_accepted n_steps n_generated final_depth verify_us <<< "$result"

        echo "${tok_s} tok/s (α=${acceptance}%, verify=${verify_us} μs, depth→${final_depth})"

        if [ "$FIRST" = true ]; then
            FIRST=false
        else
            RESULTS="${RESULTS},"
        fi

        RESULTS="${RESULTS}
    {\"mode\":\"${mode_name}\",\"prompt\":\"${prompt_name}\",\"spec_depth\":${depth},\"tiled\":${tiled},\"tok_s\":${tok_s},\"acceptance_pct\":${acceptance},\"avg_accepted\":${avg_accepted:-0},\"n_steps\":${n_steps:-0},\"n_generated\":${n_generated:-0},\"final_depth\":${final_depth},\"avg_verify_us\":${verify_us}}"
    done
    echo ""
done

RESULTS="${RESULTS}
]"

cat > "$OUTPUT_FILE" << ENDJSON
{
  "meta": {
    "git_sha": "${GIT_SHA}",
    "timestamp": ${TIMESTAMP},
    "date": "$(date -Iseconds)",
    "max_tokens": ${MAX_TOKENS},
    "target_model": "${TARGET_MODEL}",
    "draft_model": "${DRAFT_MODEL}"
  },
  "runs": ${RESULTS}
}
ENDJSON

echo "Results written to: $OUTPUT_FILE"
echo "Done."
