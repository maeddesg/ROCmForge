#!/usr/bin/env fish
# bench_tiled_sweep.fish — Full benchmark sweep: tiled vs non-tiled across depths
#
# 15 prompts × {baseline, tiled-off d1/d3/d5, tiled-on d1/d3/d5} = 105 runs
# Output: benches/results/tiled_sweep_{sha}.json
#
# Usage: fish benches/bench_tiled_sweep.fish

set -l BINARY target/release/rocmforge
set -l TARGET_MODEL (test -n "$ROCMFORGE_BENCH_TARGET_MODEL"; and echo $ROCMFORGE_BENCH_TARGET_MODEL; or echo "$HOME/models/Qwen2.5-7B-Instruct-Q4_0.gguf")
set -l DRAFT_MODEL (test -n "$ROCMFORGE_BENCH_DRAFT_MODEL"; and echo $ROCMFORGE_BENCH_DRAFT_MODEL; or echo "$HOME/models/qwen2.5-0.5b-instruct-q4_0.gguf")
set -l MAX_TOKENS 128
set -l GIT_SHA (git rev-parse --short HEAD 2>/dev/null; or echo "unknown")
set -l TIMESTAMP (date +%s)
set -l OUTPUT_DIR benches/results
set -l OUTPUT_FILE "$OUTPUT_DIR/tiled_sweep_$GIT_SHA.json"

if not test -f $BINARY
    echo "Error: $BINARY not found. Run: cargo build --release --features gpu"
    exit 1
end

mkdir -p $OUTPUT_DIR

echo "=== Tiled GEMV Benchmark Sweep ==="
echo "Git SHA: $GIT_SHA"
echo "Target: $TARGET_MODEL"
echo "Draft: $DRAFT_MODEL"
echo "Max tokens: $MAX_TOKENS"
echo "Output: $OUTPUT_FILE"
echo ""

# Collect all results as JSON array entries
set -l RESULTS ""
set -l first true

# Run a single benchmark, return tok_s and spec stats
function run_bench
    set -l prompt_file $argv[1]
    set -l depth $argv[2]        # 0 = baseline (no draft)
    set -l tiled_flag $argv[3]   # "0" or "1"
    set -l prompt (cat $prompt_file)

    set -l env_prefix ""
    if test "$tiled_flag" = "0"
        set env_prefix "ROCMFORGE_DISABLE_TILED_GEMV=1"
    end

    set -l stderr_out
    if test "$depth" = "0"
        # Baseline: no draft model
        set stderr_out (env $env_prefix $BINARY --model $TARGET_MODEL --prompt "$prompt" --max-tokens $MAX_TOKENS --gpu 2>&1 >/dev/null)
    else
        set stderr_out (env $env_prefix ROCMFORGE_PROFILE_SPEC_STEP=1 $BINARY --model $TARGET_MODEL --draft-model $DRAFT_MODEL --prompt "$prompt" --max-tokens $MAX_TOKENS --spec-depth $depth --gpu 2>&1 >/dev/null)
    end

    # Parse tok/s from last line containing "tok/s"
    set -l stats_line (echo "$stderr_out" | grep "tok/s" | tail -1)
    set -l tok_s "0"
    set -l acceptance "0"
    set -l avg_accepted "0"
    set -l n_steps "0"
    set -l n_generated "0"
    set -l final_depth "$depth"
    set -l verify_us "0"

    if test -n "$stats_line"
        set tok_s (echo "$stats_line" | grep -oP '[\d.]+(?= tok/s)')
        set n_generated (echo "$stats_line" | grep -oP '^\d+')

        if echo "$stats_line" | grep -q "speculative"
            set acceptance (echo "$stats_line" | grep -oP '[\d.]+(?=%)')
            set avg_accepted (echo "$stats_line" | grep -oP 'avg [\d.]+' | grep -oP '[\d.]+')
            set n_steps (echo "$stats_line" | grep -oP 'over \d+ steps' | grep -oP '\d+')
            # Extract final adaptive depth if shown
            set -l depth_match (echo "$stats_line" | grep -oP 'depth=\d+' | grep -oP '\d+')
            if test -n "$depth_match"
                set final_depth $depth_match
            end
        end
    end

    # Extract verify time from profiling JSON
    set -l profile_line (echo "$stderr_out" | grep "SPEC_PROFILE_JSON" | sed 's/.*\] //')
    if test -n "$profile_line"
        set verify_us (echo "$profile_line" | python3 -c "import json,sys; print(json.load(sys.stdin).get('avg_verify_us',0))" 2>/dev/null; or echo "0")
    end

    echo "$tok_s|$acceptance|$avg_accepted|$n_steps|$n_generated|$final_depth|$verify_us"
end

# Modes: baseline, then spec with tiled off/on
set -l modes "baseline:0:0" "spec_d1:1:0" "spec_d3:3:0" "spec_d5:5:0" "spec_d1_tiled:1:1" "spec_d3_tiled:3:1" "spec_d5_tiled:5:1"

for mode_spec in $modes
    set -l parts (string split ":" $mode_spec)
    set -l mode_name $parts[1]
    set -l depth $parts[2]
    set -l tiled $parts[3]

    echo "--- $mode_name ---"

    for prompt_file in benches/prompts/*.txt
        set -l prompt_name (basename $prompt_file .txt)
        echo -n "  $prompt_name... "

        set -l result (run_bench $prompt_file $depth $tiled)
        set -l vals (string split "|" $result)
        set -l tok_s $vals[1]
        set -l acceptance $vals[2]
        set -l avg_accepted $vals[3]
        set -l n_steps $vals[4]
        set -l n_generated $vals[5]
        set -l final_depth $vals[6]
        set -l verify_us $vals[7]

        echo "$tok_s tok/s (α=$acceptance%, verify=$verify_us μs, depth→$final_depth)"

        # Build JSON entry
        set -l entry "{\"mode\":\"$mode_name\",\"prompt\":\"$prompt_name\",\"spec_depth\":$depth,\"tiled\":$tiled,\"tok_s\":$tok_s,\"acceptance_pct\":$acceptance,\"avg_accepted\":$avg_accepted,\"n_steps\":$n_steps,\"n_generated\":$n_generated,\"final_depth\":$final_depth,\"avg_verify_us\":$verify_us}"

        if test "$first" = "true"
            set RESULTS "$entry"
            set first false
        else
            set RESULTS "$RESULTS,$entry"
        end
    end
    echo ""
end

# Write JSON
echo "{
  \"meta\": {
    \"git_sha\": \"$GIT_SHA\",
    \"timestamp\": $TIMESTAMP,
    \"date\": \"(date -Iseconds)\",
    \"max_tokens\": $MAX_TOKENS,
    \"target_model\": \"$TARGET_MODEL\",
    \"draft_model\": \"$DRAFT_MODEL\"
  },
  \"runs\": [$RESULTS]
}" > $OUTPUT_FILE

echo "Results written to: $OUTPUT_FILE"
echo "Done."
