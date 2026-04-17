#!/usr/bin/env fish
# bench_batched_lm_head.fish — Benchmark batched vs sequential lm_head
#
# Runs 15 prompts × {sequential, batched} × {depth 1, 3, 5} + baseline
#
# Usage: fish benches/bench_batched_lm_head.fish

set BINARY target/release/rocmforge
set TARGET_MODEL $HOME/models/Qwen2.5-7B-Instruct-Q4_0.gguf
set DRAFT_MODEL $HOME/models/qwen2.5-0.5b-instruct-q4_0.gguf
set PROMPT_DIR benches/prompts
set MAX_TOKENS 128
set GIT_SHA (git rev-parse --short HEAD 2>/dev/null; or echo unknown)
set TIMESTAMP (date +%s)
set OUTPUT_DIR benches/results
set OUTPUT_FILE $OUTPUT_DIR/batched_lm_head_sweep_{$GIT_SHA}_{$TIMESTAMP}.json

mkdir -p $OUTPUT_DIR

# Verify prerequisites
for f in $BINARY $TARGET_MODEL $DRAFT_MODEL
    if not test -f $f
        echo "Error: $f not found"
        exit 1
    end
end

echo "=== Batched lm_head Benchmark Sweep ==="
echo "Git SHA: $GIT_SHA"
echo "Output: $OUTPUT_FILE"
echo ""

# Collect results as JSON lines in a temp file
set TMPJSON (mktemp)
echo "[" > $TMPJSON
set -l first true

for mode in baseline sequential batched
    for depth in 1 3 5
        # Skip baseline at depth > 1
        if test "$mode" = "baseline"; and test "$depth" != "1"
            continue
        end

        set -l mode_label $mode
        if test "$mode" = "baseline"
            set mode_label "baseline_no_spec"
        else
            set mode_label {$mode}_depth_{$depth}
        end

        echo "--- Mode: $mode_label ---"

        for prompt_file in $PROMPT_DIR/*.txt
            set -l prompt_name (basename $prompt_file .txt)
            echo -n "  $prompt_name... "

            # Write prompt to temp file to avoid shell quoting issues
            set -l tmpfile (mktemp)
            cp $prompt_file $tmpfile

            set -l stderr_file (mktemp)
            set -l spec_depth_val 0

            if test "$mode" = "baseline"
                $BINARY --model $TARGET_MODEL --prompt (cat $tmpfile) --max-tokens $MAX_TOKENS --gpu 2>$stderr_file >/dev/null; or true
            else if test "$mode" = "sequential"
                set spec_depth_val $depth
                env ROCMFORGE_DISABLE_BATCHED_LM_HEAD=1 $BINARY --model $TARGET_MODEL --draft-model $DRAFT_MODEL --prompt (cat $tmpfile) --max-tokens $MAX_TOKENS --spec-depth $depth --gpu 2>$stderr_file >/dev/null; or true
            else
                set spec_depth_val $depth
                env ROCMFORGE_DISABLE_BATCHED_LM_HEAD=0 $BINARY --model $TARGET_MODEL --draft-model $DRAFT_MODEL --prompt (cat $tmpfile) --max-tokens $MAX_TOKENS --spec-depth $depth --gpu 2>$stderr_file >/dev/null; or true
            end

            rm -f $tmpfile

            # Parse output
            set -l stats_line (grep "tokens in.*tok/s" $stderr_file | tail -1)
            set -l tok_s 0
            set -l acceptance 0
            set -l n_steps 0
            set -l n_generated 0

            if test -n "$stats_line"
                set tok_s (echo $stats_line | grep -oP '[\d.]+(?= tok/s)')
                set n_generated (echo $stats_line | grep -oP '^\d+')
                if echo $stats_line | grep -q "speculative"
                    set acceptance (echo $stats_line | grep -oP '[\d.]+(?=%)' | head -1)
                    set n_steps (echo $stats_line | grep -oP 'over \d+ steps' | grep -oP '\d+')
                end
            end

            # E[n]
            set -l e_n 1.0
            if test "$mode" != "baseline"; and test -n "$n_steps"; and test "$n_steps" != "0"
                set e_n (python3 -c "print(round($n_generated / $n_steps, 2))")
            end

            echo "$tok_s tok/s (α={$acceptance}%, E[n]=$e_n)"

            rm -f $stderr_file

            if test "$first" = "true"
                set first false
            else
                echo "," >> $TMPJSON
            end

            echo -n "    {\"mode\": \"$mode_label\", \"spec_depth\": $spec_depth_val, \"prompt\": \"$prompt_name\", \"tok_s\": $tok_s, \"acceptance_pct\": $acceptance, \"e_n\": $e_n, \"n_steps\": $n_steps, \"n_generated\": $n_generated}" >> $TMPJSON
        end
        echo ""
    end
end

echo "" >> $TMPJSON
echo "]" >> $TMPJSON

# Assemble final JSON
echo "{" > $OUTPUT_FILE
echo "  \"meta\": {" >> $OUTPUT_FILE
echo "    \"git_sha\": \"$GIT_SHA\"," >> $OUTPUT_FILE
echo "    \"timestamp\": $TIMESTAMP," >> $OUTPUT_FILE
echo "    \"target_model\": \"$TARGET_MODEL\"," >> $OUTPUT_FILE
echo "    \"draft_model\": \"$DRAFT_MODEL\"," >> $OUTPUT_FILE
echo "    \"max_tokens\": $MAX_TOKENS" >> $OUTPUT_FILE
echo "  }," >> $OUTPUT_FILE
echo "  \"runs\":" >> $OUTPUT_FILE
cat $TMPJSON >> $OUTPUT_FILE
echo "}" >> $OUTPUT_FILE

rm -f $TMPJSON

echo "Results written to: $OUTPUT_FILE"
