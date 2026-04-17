#!/usr/bin/env fish
#
# Prefill throughput sweep: compare the hipBLAS-backed prefill path
# against the custom GEMM path at several prompt lengths.

set -l here (dirname (status filename))
set -l root (realpath "$here/..")
set -l model "$HOME/models/Qwen2.5-7B-Instruct-Q4_0.gguf"
set -l bin "$root/target/release/rocmforge"

if not test -f "$model"
    echo "Model not found: $model" >&2
    exit 1
end

if not test -x "$bin"
    echo "Building release binary with --features gpu..."
    cargo build --release --features gpu
    or exit 1
end

set -l sha (git -C $root rev-parse --short HEAD 2>/dev/null; or echo dev)
set -l ts (date +%s)
set -l out "$root/benches/results/prefill_hipblas_{$sha}_{$ts}.json"

set -l lengths 19 64 128 256 512

function run_one --argument-names bin model prompt_file mode
    set -l ppp (cat $prompt_file)
    if test "$mode" = hipblas
        set result ($bin --model $model --prompt "$ppp" --max-tokens 1 \
            --temperature 0.0 --top-p 1.0 --no-template --gpu 2>&1)
    else
        set result (env ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1 $bin --model $model \
            --prompt "$ppp" --max-tokens 1 --temperature 0.0 --top-p 1.0 \
            --no-template --gpu 2>&1)
    end
    # Prefill line example: "Prefill: 2366.3ms (84.5 tok/s)"
    set -l line (string match -r 'Prefill: ([0-9.]+)ms \(([0-9.]+) tok/s\)' -- $result)
    if test (count $line) -ge 3
        echo "$line[2] $line[3]"
    else
        echo "0 0"
    end
end

echo "{"
echo "  \"git_sha\": \"$sha\","
echo "  \"timestamp\": $ts,"
echo "  \"model\": \"Qwen2.5-7B-Instruct-Q4_0\","
echo "  \"runs\": ["

set -l first 1
for len in $lengths
    # Build a prompt of exactly `len` tokens by repeating a simple word.
    # The binary tokenizer may not give exact counts, but the measured
    # "Prompt: N tokens" line reports actual length.
    set -l words (math "$len - 1")
    python3 -c "print(' '.join(['word'] * $words))" > /tmp/ppbench.txt

    for mode in hipblas gemv
        for run in 1 2 3
            set -l r (run_one $bin $model /tmp/ppbench.txt $mode)
            set -l parts (string split " " $r)
            set -l ms $parts[1]
            set -l tps $parts[2]
            if test $first -eq 0
                echo "    ,"
            else
                set first 0
            end
            printf '    {"target_len": %d, "mode": "%s", "run": %d, "ms": %s, "tps": %s}\n' \
                $len $mode $run $ms $tps
            echo "  len=$len mode=$mode run=$run: $tps tok/s ($ms ms)" >&2
        end
    end
end

echo "  ]"
echo "}"
