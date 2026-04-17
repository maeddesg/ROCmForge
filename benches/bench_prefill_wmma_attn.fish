#!/usr/bin/env fish
#
# Phase 3d end-to-end prefill throughput sweep comparing all four
# dispatch paths:
#   custom    — disable WMMA prefill + disable hipBLAS prefill
#   hipblas   — disable WMMA prefill (→ hipBLAS Hgemm + scalar attention)
#   wmma_gemm — WMMA GEMM with scalar attention (Phase 2d)
#   wmma_full — WMMA GEMM + WMMA GQA/causal attention (Phase 3d, default)

set -l here (dirname (status filename))
set -l root (realpath "$here/..")
set -l model "$HOME/models/Qwen2.5-7B-Instruct-Q4_0.gguf"
set -l bin "$root/target/release/rocmforge"

if not test -f "$model"
    echo "Model not found: $model" >&2; exit 1
end
if not test -x "$bin"
    cargo build --release --features gpu; or exit 1
end

set -l sha (git -C $root rev-parse --short HEAD 2>/dev/null; or echo dev)
set -l ts (date +%s)
set -l out "$root/benches/results/prefill_wmma_attn_e2e_{$sha}_{$ts}.json"

set -l lengths 64 128 192 256 384 512

function run_one --argument-names bin model prompt mode max_tokens
    set -l result
    switch $mode
        case wmma_full
            set result ($bin --model $model --prompt "$prompt" --max-tokens $max_tokens \
                --temperature 0.0 --top-p 1.0 --no-template --gpu 2>&1)
        case wmma_gemm
            set result (env ROCMFORGE_DISABLE_WMMA_ATTENTION=1 $bin --model $model \
                --prompt "$prompt" --max-tokens $max_tokens \
                --temperature 0.0 --top-p 1.0 --no-template --gpu 2>&1)
        case hipblas
            set result (env ROCMFORGE_DISABLE_WMMA_PREFILL=1 ROCMFORGE_DISABLE_WMMA_ATTENTION=1 \
                $bin --model $model \
                --prompt "$prompt" --max-tokens $max_tokens \
                --temperature 0.0 --top-p 1.0 --no-template --gpu 2>&1)
        case custom
            set result (env ROCMFORGE_DISABLE_WMMA_PREFILL=1 ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1 \
                ROCMFORGE_DISABLE_WMMA_ATTENTION=1 \
                $bin --model $model --prompt "$prompt" --max-tokens $max_tokens \
                --temperature 0.0 --top-p 1.0 --no-template --gpu 2>&1)
    end
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
    python3 -c "print(' '.join(['word'] * $len))" > /tmp/ppbench_attn.txt
    set -l prompt (cat /tmp/ppbench_attn.txt)

    for mode in custom hipblas wmma_gemm wmma_full
        for run in 1 2 3
            set -l r (run_one $bin $model "$prompt" $mode 1)
            set -l parts (string split " " $r)
            if test $first -eq 0
                echo "    ,"
            else
                set first 0
            end
            printf '    {"target_len": %d, "mode": "%s", "run": %d, "ms": %s, "tps": %s}\n' \
                $len $mode $run $parts[1] $parts[2]
            echo "  len=$len mode=$mode run=$run: $parts[2] tok/s ($parts[1] ms)" >&2
        end
    end
end

echo "  ]"
echo "}"
