#!/usr/bin/fish
# ROCmForge vs llama.cpp — 15-Prompt Benchmark on gfx1201 (RX 9070 XT).

# Anchor to the ROCmForge checkout so relative paths work regardless
# of where `fish bench_llamacpp_15prompt.fish` is invoked from.
cd /home/maeddes/projects/ROCmForge

#
# Runs the SAME 15 prompts from benches_v1/inference_test_prompts_15.json
# against llama.cpp's llama-completion binary, so the two inference
# stacks can be compared apples-to-apples.
#
# Why llama-completion, not llama-cli?  On our test machine llama-cli
# consistently hangs at 100% CPU with no output (likely an upstream
# issue with the interactive/conversation mode on this build).
# llama-completion is the non-interactive equivalent: it reads the
# prompt via -p, applies the GGUF-embedded chat template, decodes
# up to -n tokens, and prints standard `common_perf_print` timing at
# exit. That timing block is what we parse.
#
# Output: results/bench_llamacpp_15prompt.md.

set -l LLAMA /home/maeddes/tmp/llama.cpp/build-rocm/bin/llama-completion
set -l MODELS \
    ~/models/Qwen3-8B-Q4_K_M.gguf \
    ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
set -l SUITE /home/maeddes/projects/ROCmForge/benches_v1/inference_test_prompts_15.json
set -l REPORT /home/maeddes/projects/ROCmForge/results/bench_llamacpp_15prompt.md
set -l SCRATCH /tmp/llamacpp_bench_$fish_pid

# Sanity.
if not test -x "$LLAMA"
    echo "error: llama-completion not found at $LLAMA" >&2
    exit 1
end
if not test -f "$SUITE"
    echo "error: suite JSON not found: $SUITE" >&2
    exit 1
end
for m in $MODELS
    if not test -f "$m"
        echo "error: model not found: $m" >&2
        exit 1
    end
end
mkdir -p $SCRATCH
mkdir -p (dirname $REPORT)

echo "# llama.cpp 15-Prompt Benchmark (RX 9070 XT, gfx1201)" > $REPORT
echo "" >> $REPORT
echo "**Date:** "(date +%Y-%m-%d) >> $REPORT
echo "**Tool:** "$LLAMA >> $REPORT
echo "**Build:** "(cd /home/maeddes/tmp/llama.cpp && git rev-parse --short HEAD 2>/dev/null) >> $REPORT
echo "**Sampling:** greedy (temp=0, seed=42), GPU-only (-ngl 99)" >> $REPORT
echo "**Suite:** "$SUITE" (same 15 prompts ROCmForge uses)" >> $REPORT
echo "" >> $REPORT

# Run every prompt on every model. We keep both raw stderr log files
# (so the report reader can inspect them) and parsed per-prompt
# metrics.
for model in $MODELS
    set -l model_name (basename $model)
    echo "=== $model_name ==="
    echo "## $model_name" >> $REPORT
    echo "" >> $REPORT
    echo "| # | Name | max_tok | Prefill tok/s | Decode tok/s | Prefill tok | Decode tok | Wall ms |" >> $REPORT
    echo "|---:|---|---:|---:|---:|---:|---:|---:|" >> $REPORT

    set -l sum_prefill_ms 0
    set -l sum_decode_ms 0
    set -l sum_prefill_tok 0
    set -l sum_decode_tok 0
    set -l sum_wall 0
    set -l failures 0

    for i in (seq 0 14)
        # Extract prompt fields via jq — avoids brittle parsing.
        set -l id (jq -r ".prompts[$i].id" $SUITE)
        set -l name (jq -r ".prompts[$i].name" $SUITE)
        set -l max_tokens (jq -r ".prompts[$i].max_tokens" $SUITE)
        set -l prompt (jq -r ".prompts[$i].prompt" $SUITE)

        set -l log_file $SCRATCH/run_(basename $model .gguf)_$id.log
        set -l short_name (echo $name | head -c 22)
        printf "  [%2d/15] %-23s max=%4d ... " $id "$short_name" $max_tokens

        # Full wall-clock timing — `time -f %e` would be cleaner but
        # not every fish environment has GNU time in $PATH. Use
        # `date +%s%N` pre/post.
        set -l t0 (date +%s%N)
        # LC_ALL=C forces a decimal point rather than the German
        # comma. -no-cnv would be the llama-cli flag; llama-completion
        # is non-interactive by design.
        env LC_ALL=C $LLAMA \
            --model $model \
            --prompt "$prompt" \
            --n-predict $max_tokens \
            --temp 0 \
            --seed 42 \
            -ngl 99 \
            > $log_file 2>&1
        set -l rc $status
        set -l t1 (date +%s%N)
        set -l wall_ms (math "($t1 - $t0) / 1000000")

        if test $rc -ne 0
            printf "FAIL (rc=%d)\n" $rc
            set failures (math $failures + 1)
            echo "| $id | $name | $max_tokens | — | — | — | — | $wall_ms |" >> $REPORT
            continue
        end

        # Parse the `common_perf_print` timing block. Example line:
        #   common_perf_print: prompt eval time =     123,45 ms /     33 tokens (    3,74 ms per token,   267,38 tokens per second)
        # LC_ALL=C converts the commas to dots above, so we can match
        # plain numbers.
        set -l prompt_line (grep "prompt eval time" $log_file | tail -1)
        set -l eval_line (grep -E "^common_perf_print:\s+eval time" $log_file | tail -1)

        # Extract tok/s and token counts.
        set -l prefill_tps "n/a"
        set -l decode_tps "n/a"
        set -l prefill_tok "n/a"
        set -l decode_tok "n/a"
        set -l prefill_ms "n/a"
        set -l decode_ms "n/a"

        if test -n "$prompt_line"
            # prompt eval time =   X.XX ms / N tokens (...)
            set prefill_ms (echo $prompt_line | grep -oE '= *[0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+')
            set prefill_tok (echo $prompt_line | grep -oE '/ *[0-9]+ tokens' | head -1 | grep -oE '[0-9]+')
            set prefill_tps (echo $prompt_line | grep -oE '[0-9]+\.[0-9]+ tokens per second' | head -1 | grep -oE '[0-9]+\.[0-9]+')
        end
        if test -n "$eval_line"
            set decode_ms (echo $eval_line | grep -oE '= *[0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+')
            set decode_tok (echo $eval_line | grep -oE '/ *[0-9]+ runs' | head -1 | grep -oE '[0-9]+')
            set decode_tps (echo $eval_line | grep -oE '[0-9]+\.[0-9]+ tokens per second' | head -1 | grep -oE '[0-9]+\.[0-9]+')
        end

        printf "prefill %s tok/s, decode %s tok/s, %dms\n" $prefill_tps $decode_tps $wall_ms
        echo "| $id | $name | $max_tokens | $prefill_tps | $decode_tps | $prefill_tok | $decode_tok | $wall_ms |" >> $REPORT

        # Aggregate sums (require finite numbers).
        if test "$prefill_tps" != "n/a"
            set sum_prefill_ms (math "$sum_prefill_ms + $prefill_ms")
            set sum_decode_ms (math "$sum_decode_ms + $decode_ms")
            set sum_prefill_tok (math "$sum_prefill_tok + $prefill_tok")
            set sum_decode_tok (math "$sum_decode_tok + $decode_tok")
            set sum_wall (math "$sum_wall + $wall_ms")
        end
    end

    # Aggregate row.
    if test "$sum_prefill_ms" != "0"
        set -l agg_prefill_tps (math "$sum_prefill_tok / ($sum_prefill_ms / 1000.0)")
        set -l agg_decode_tps (math "$sum_decode_tok / ($sum_decode_ms / 1000.0)")
        set agg_prefill_tps (printf "%.1f" $agg_prefill_tps)
        set agg_decode_tps (printf "%.1f" $agg_decode_tps)
        echo "| **Agg** | **Total** | — | **$agg_prefill_tps** | **$agg_decode_tps** | **$sum_prefill_tok** | **$sum_decode_tok** | **$sum_wall** |" >> $REPORT
    end
    if test $failures -gt 0
        echo "" >> $REPORT
        echo "**Failures:** $failures prompt(s)" >> $REPORT
    end
    echo "" >> $REPORT
end

echo "" >> $REPORT
echo "## Raw-log archive" >> $REPORT
echo "Per-prompt stderr + stdout captured under $SCRATCH/" >> $REPORT
echo "" >> $REPORT

echo ""
echo "Report written to $REPORT"
echo "Raw logs in $SCRATCH"
