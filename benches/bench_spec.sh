#!/usr/bin/env bash
# bench_spec.sh — Speculative decoding benchmark harness
#
# Runs the rocmforge binary across 15 prompts × 4 modes (baseline + depth 1/3/5)
# and collects throughput, acceptance rate, and GPU power metrics.
#
# Usage: ./benches/bench_spec.sh [--output <path>]
# Requires: ROCMFORGE_RUN_GPU_BENCHES=1, target/release/rocmforge built
#
# Output: JSON file in benches/results/

set -euo pipefail

BINARY="target/release/rocmforge"
TARGET_MODEL="${ROCMFORGE_BENCH_TARGET_MODEL:-$HOME/models/Qwen2.5-7B-Instruct-Q4_0.gguf}"
DRAFT_MODEL="${ROCMFORGE_BENCH_DRAFT_MODEL:-$HOME/models/qwen2.5-0.5b-instruct-q4_0.gguf}"
PROMPT_DIR="benches/prompts"
MAX_TOKENS=128
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +%s)
ROCM_VERSION=$(rocminfo 2>/dev/null | grep -m1 "Runtime Version" | awk '{print $NF}' | tr -d '\n' || echo "unknown")

# GPU power reading
HWMON_POWER=""
for f in /sys/class/drm/card*/device/hwmon/*/power1_average; do
    if [ -f "$f" ]; then
        HWMON_POWER="$f"
        break
    fi
done

# Parse output file path
OUTPUT_DIR="benches/results"
OUTPUT_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/baseline_${GIT_SHA}_${TIMESTAMP}.json"
fi

mkdir -p "$OUTPUT_DIR"

# Check prerequisites
if [ ! -f "$BINARY" ]; then
    echo "Error: $BINARY not found. Run: cargo build --release --features gpu"
    exit 1
fi
if [ ! -f "$TARGET_MODEL" ]; then
    echo "Error: Target model not found: $TARGET_MODEL"
    exit 1
fi
if [ ! -f "$DRAFT_MODEL" ]; then
    echo "Error: Draft model not found: $DRAFT_MODEL"
    exit 1
fi

echo "=== Speculative Decode Benchmark ==="
echo "Git SHA: $GIT_SHA"
echo "ROCm: $ROCM_VERSION"
echo "Target: $TARGET_MODEL"
echo "Draft: $DRAFT_MODEL"
echo "Max tokens: $MAX_TOKENS"
echo "Output: $OUTPUT_FILE"
echo ""

# Read GPU power in microwatts, convert to watts
read_gpu_power() {
    if [ -n "$HWMON_POWER" ] && [ -f "$HWMON_POWER" ]; then
        echo "scale=1; $(cat "$HWMON_POWER") / 1000000" | bc 2>/dev/null || echo "null"
    else
        echo "null"
    fi
}

# Run a single benchmark and parse stderr output
# Returns: tok_s, acceptance_pct, avg_accepted, n_steps, n_generated, n_drafted, n_accepted
run_single() {
    local prompt_file="$1"
    local spec_depth="$2"  # "0" means baseline (no draft)
    local prompt
    prompt=$(cat "$prompt_file")

    local power_before
    power_before=$(read_gpu_power)

    local stdout stderr exit_code
    if [ "$spec_depth" = "0" ]; then
        # Baseline: no draft model
        stderr=$($BINARY --model "$TARGET_MODEL" --prompt "$prompt" --max-tokens "$MAX_TOKENS" --gpu 2>&1 >/dev/null) || true
    else
        stderr=$($BINARY --model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" --prompt "$prompt" --max-tokens "$MAX_TOKENS" --spec-depth "$spec_depth" --gpu 2>&1 >/dev/null) || true
    fi

    local power_after
    power_after=$(read_gpu_power)

    # Parse tok/s from stderr
    local tok_s="0"
    local acceptance_pct="0"
    local avg_accepted="0"
    local n_steps="0"
    local n_generated="0"

    # Baseline format: "128 tokens in 1234.5ms = 82.1 tok/s"
    # Spec format: "128 tokens in 1234.5ms = 65.9 tok/s (speculative: 31/163 accepted = 19.0%, avg 0.9/step over 34 steps)"
    local stats_line
    stats_line=$(echo "$stderr" | grep "tokens in.*tok/s" | tail -1)

    if [ -n "$stats_line" ]; then
        n_generated=$(echo "$stats_line" | grep -oP '^\d+')
        tok_s=$(echo "$stats_line" | grep -oP '[\d.]+(?= tok/s)')

        if echo "$stats_line" | grep -q "speculative"; then
            acceptance_pct=$(echo "$stats_line" | grep -oP '[\d.]+(?=%)')
            avg_accepted=$(echo "$stats_line" | grep -oP 'avg [\d.]+' | grep -oP '[\d.]+')
            n_steps=$(echo "$stats_line" | grep -oP 'over \d+ steps' | grep -oP '\d+')
        fi
    fi

    echo "${tok_s}|${acceptance_pct}|${avg_accepted}|${n_steps}|${n_generated}|${power_before}|${power_after}"
}

# Collect all results
RESULTS="["
FIRST=true

for depth in 0 1 3 5; do
    mode_label="baseline"
    if [ "$depth" != "0" ]; then
        mode_label="spec_depth_${depth}"
    fi
    echo "--- Mode: $mode_label ---"

    TOK_S_VALUES=""

    for prompt_file in "$PROMPT_DIR"/*.txt; do
        prompt_name=$(basename "$prompt_file" .txt)
        echo -n "  $prompt_name... "

        result=$(run_single "$prompt_file" "$depth")
        IFS='|' read -r tok_s acceptance_pct avg_accepted n_steps n_generated power_before power_after <<< "$result"

        # E[n] = expected committed tokens per target forward
        # For baseline: E[n] = 1
        # For spec: E[n] = (n_generated) / n_steps (if n_steps > 0)
        e_n="1.0"
        if [ "$depth" != "0" ] && [ "$n_steps" != "0" ] && [ -n "$n_steps" ]; then
            e_n=$(echo "scale=2; $n_generated / $n_steps" | bc 2>/dev/null || echo "1.0")
        fi

        echo "${tok_s} tok/s (α=${acceptance_pct}%, E[n]=${e_n})"

        TOK_S_VALUES="${TOK_S_VALUES}${tok_s} "

        if [ "$FIRST" = true ]; then
            FIRST=false
        else
            RESULTS="${RESULTS},"
        fi

        RESULTS="${RESULTS}
    {
      \"mode\": \"${mode_label}\",
      \"spec_depth\": ${depth},
      \"prompt\": \"${prompt_name}\",
      \"tok_s\": ${tok_s},
      \"acceptance_pct\": ${acceptance_pct},
      \"avg_accepted_per_step\": ${avg_accepted},
      \"e_n\": ${e_n},
      \"n_steps\": ${n_steps:-0},
      \"n_generated\": ${n_generated},
      \"gpu_power_before_w\": ${power_before},
      \"gpu_power_after_w\": ${power_after}
    }"
    done

    # Calculate median tok/s for this mode
    median=$(echo "$TOK_S_VALUES" | tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$1} END{if(NR%2==1)print a[(NR+1)/2]; else print (a[NR/2]+a[NR/2+1])/2}')
    echo "  Median: ${median} tok/s"
    echo ""
done

RESULTS="${RESULTS}
]"

# Write final JSON
cat > "$OUTPUT_FILE" << ENDJSON
{
  "meta": {
    "git_sha": "${GIT_SHA}",
    "timestamp": ${TIMESTAMP},
    "date": "$(date -Iseconds)",
    "rocm_version": "${ROCM_VERSION}",
    "target_model": "${TARGET_MODEL}",
    "draft_model": "${DRAFT_MODEL}",
    "max_tokens": ${MAX_TOKENS},
    "gpu": "$(rocm-smi --showproductname 2>/dev/null | grep -oP 'Card Series:\s*\K.*' | head -1 | tr -d '\n' || echo 'unknown')"
  },
  "runs": ${RESULTS}
}
ENDJSON

echo "Results written to: $OUTPUT_FILE"
echo "Done."
