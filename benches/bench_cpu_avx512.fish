#!/usr/bin/env fish
#
# CPU AVX-512 Q4_0 GEMV end-to-end throughput sweep.
# Runs 3 prompts × 2 modes (AVX2 vs AVX-512 VNNI) on Qwen2.5-0.5B.

set -l here (dirname (status filename))
set -l root (realpath "$here/..")
set -l model "$HOME/models/qwen2.5-0.5b-instruct-q4_0.gguf"

if not test -f "$model"
    echo "Model not found at $model" >&2
    exit 1
end

set -l bin "$root/target/release/rocmforge"
if not test -x "$bin"
    cargo build --release --bin rocmforge
    or exit 1
end

set -l sha (git -C $root rev-parse --short HEAD 2>/dev/null; or echo dev)
set -l ts (date +%s)
set -l out "$root/benches/results/cpu_avx512_sweep_{$sha}_{$ts}.json"

set -l prompt_code "Write a Python function that sorts a list using quicksort"
set -l prompt_chat "What is the capital of France and what is its population?"
set -l prompt_prose "Once upon a time in a small village there lived a young baker"

# Two temp prompt files so fish doesn't mangle them
set -l tmp_code (mktemp /tmp/rocmforge_prompt.XXXXXX)
set -l tmp_chat (mktemp /tmp/rocmforge_prompt.XXXXXX)
set -l tmp_prose (mktemp /tmp/rocmforge_prompt.XXXXXX)
echo $prompt_code > $tmp_code
echo $prompt_chat > $tmp_chat
echo $prompt_prose > $tmp_prose

function run_one --argument-names bin model prompt_file mode max_tokens
    set -l env_prefix ""
    if test "$mode" = avx2
        set env_prefix ROCMFORGE_DISABLE_AVX512=1
    end
    set -l prompt (cat $prompt_file)
    set -l result
    if test -z "$env_prefix"
        set result ($bin --model $model --prompt "$prompt" --max-tokens $max_tokens \
            --temperature 0.0 --top-p 1.0 --no-template 2>&1)
    else
        set result (env $env_prefix $bin --model $model --prompt "$prompt" --max-tokens $max_tokens \
            --temperature 0.0 --top-p 1.0 --no-template 2>&1)
    end
    # Parse "N tokens in T ms = TPS tok/s" line
    set -l stats (string match -r '(\d+) tokens in (\d+\.\d+)ms = (\d+\.\d+) tok/s' -- $result)
    if test (count $stats) -ge 4
        set -l n $stats[2]
        set -l ms $stats[3]
        set -l tps $stats[4]
        echo "$n $ms $tps"
    else
        echo "0 0 0"
    end
end

echo "{"
echo "  \"git_sha\": \"$sha\","
echo "  \"timestamp\": $ts,"
echo "  \"model\": \"qwen2.5-0.5b-instruct-q4_0\","
echo "  \"max_tokens\": 128,"
echo "  \"runs\": ["

set -l first 1
for prompt_name in code chat prose
    switch $prompt_name
        case code
            set tmp $tmp_code
        case chat
            set tmp $tmp_chat
        case prose
            set tmp $tmp_prose
    end
    for mode in avx2 avx512
        for run in 1 2 3
            set -l r (run_one $bin $model $tmp $mode 128)
            set -l parts (string split " " $r)
            set -l n $parts[1]
            set -l ms $parts[2]
            set -l tps $parts[3]

            if test $first -eq 0
                echo "    ,"
            else
                set first 0
            end
            printf '    {"prompt": "%s", "mode": "%s", "run": %d, "tokens": %s, "ms": %s, "tps": %s}\n' \
                $prompt_name $mode $run $n $ms $tps
            echo "  $prompt_name/$mode/run$run: $tps tok/s  ($n tokens in $ms ms)" >&2
        end
    end
end

echo "  ]"
echo "}"

rm -f $tmp_code $tmp_chat $tmp_prose
