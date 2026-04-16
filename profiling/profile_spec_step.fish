#!/usr/bin/env fish
# profile_spec_step.fish — Spec-step cost breakdown profiler
#
# Runs rocmforge with ROCMFORGE_PROFILE_SPEC_STEP=1 on selected prompts
# at depth=1 and collects the per-phase cost breakdown.
#
# Usage: fish profiling/profile_spec_step.fish
# Requires: target/release/rocmforge built with --features gpu

set -l BINARY target/release/rocmforge
set -l TARGET_MODEL (test -n "$ROCMFORGE_BENCH_TARGET_MODEL"; and echo $ROCMFORGE_BENCH_TARGET_MODEL; or echo "$HOME/models/Qwen2.5-7B-Instruct-Q4_0.gguf")
set -l DRAFT_MODEL (test -n "$ROCMFORGE_BENCH_DRAFT_MODEL"; and echo $ROCMFORGE_BENCH_DRAFT_MODEL; or echo "$HOME/models/qwen2.5-0.5b-instruct-q4_0.gguf")
set -l MAX_TOKENS 128
set -l SPEC_DEPTH 1
set -l GIT_SHA (git rev-parse --short HEAD 2>/dev/null; or echo "unknown")
set -l TIMESTAMP (date +%s)
set -l OUTPUT_DIR profiling/results

set -l PROMPTS code_01 chat_01 prose_03

if not test -f $BINARY
    echo "Error: $BINARY not found. Run: cargo build --release --features gpu"
    exit 1
end

mkdir -p $OUTPUT_DIR

echo "=== Spec-Step Cost Breakdown Profiler ==="
echo "Git SHA: $GIT_SHA"
echo "Target: $TARGET_MODEL"
echo "Draft: $DRAFT_MODEL"
echo "Depth: $SPEC_DEPTH"
echo "Max tokens: $MAX_TOKENS"
echo ""

set -l ALL_JSON "["
set -l first true

for prompt_name in $PROMPTS
    set -l prompt_file "benches/prompts/$prompt_name.txt"
    if not test -f $prompt_file
        echo "Warning: $prompt_file not found, skipping"
        continue
    end

    set -l prompt (cat $prompt_file)
    echo "--- $prompt_name ---"

    # Run with profiling enabled
    set -l stderr_output (ROCMFORGE_PROFILE_SPEC_STEP=1 $BINARY \
        --model $TARGET_MODEL \
        --draft-model $DRAFT_MODEL \
        --prompt "$prompt" \
        --max-tokens $MAX_TOKENS \
        --spec-depth $SPEC_DEPTH \
        --gpu 2>&1 >/dev/null)

    # Extract the JSON line
    set -l json_line (echo "$stderr_output" | grep "SPEC_PROFILE_JSON" | sed 's/.*\] //')

    # Extract decode tok/s (last line with tok/s, not prefill)
    set -l tok_s (echo "$stderr_output" | grep "tok/s" | tail -1 | grep -oP '[\d.]+(?= tok/s)')

    if test -n "$json_line"
        echo "  Profile: $json_line"
        echo "  Throughput: $tok_s tok/s"

        # Add prompt name and tok/s to the JSON
        set -l enriched (echo $json_line | sed "s/^{/{\"prompt\":\"$prompt_name\",\"tok_s\":$tok_s,/")

        if test "$first" = true
            set ALL_JSON "$ALL_JSON$enriched"
            set first false
        else
            set ALL_JSON "$ALL_JSON,$enriched"
        end
    else
        echo "  (no profile data — check ROCMFORGE_PROFILE_SPEC_STEP)"
        echo "  stderr: $stderr_output" | head -5
    end

    echo ""
end

set ALL_JSON "$ALL_JSON]"

# Write results JSON
set -l outfile "$OUTPUT_DIR/spec_step_profile_{$GIT_SHA}_{$TIMESTAMP}.json"
echo "{
  \"meta\": {
    \"git_sha\": \"$GIT_SHA\",
    \"timestamp\": $TIMESTAMP,
    \"date\": \"(date -Iseconds)\",
    \"spec_depth\": $SPEC_DEPTH,
    \"max_tokens\": $MAX_TOKENS,
    \"target_model\": \"$TARGET_MODEL\",
    \"draft_model\": \"$DRAFT_MODEL\"
  },
  \"profiles\": $ALL_JSON
}" > $outfile

echo "Results written to: $outfile"

# Print summary table
echo ""
echo "=== Summary (depth=$SPEC_DEPTH, avg μs per step) ==="
echo ""
printf "%-12s %10s %10s %10s %10s %10s %8s\n" "Prompt" "Draft" "Verify" "Accept" "Overhead" "Total" "tok/s"
printf "%-12s %10s %10s %10s %10s %10s %8s\n" "------" "-----" "------" "------" "--------" "-----" "-----"

for prompt_name in $PROMPTS
    set -l entry (echo $ALL_JSON | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data:
    if r.get('prompt') == '$prompt_name':
        n = r['n_steps']
        print(f\"{r['avg_draft_us']:.0f} {r['avg_verify_us']:.0f} {r['avg_accept_us']:.0f} {r['avg_overhead_us']:.0f} {r['avg_total_us']:.0f} {r.get('tok_s', 'N/A')}\")
        break
" 2>/dev/null)

    if test -n "$entry"
        set -l parts (string split " " $entry)
        printf "%-12s %10s %10s %10s %10s %10s %8s\n" $prompt_name $parts[1] $parts[2] $parts[3] $parts[4] $parts[5] $parts[6]
    end
end

echo ""
echo "Done."
