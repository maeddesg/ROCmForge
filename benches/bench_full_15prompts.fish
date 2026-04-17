#!/usr/bin/env fish
#
# 15-prompt end-to-end benchmark: ROCmForge (3 configs × 3 runs) plus
# llama.cpp raw-completion reference. Emits a single JSON file with
# per-prompt timings and writes per-config answer dumps for later
# divergence analysis.

set -l here (dirname (status filename))
set -l root (realpath "$here/..")
set -l model "$HOME/models/Qwen2.5-7B-Instruct-Q4_0.gguf"
set -l draft "$HOME/models/qwen2.5-0.5b-instruct-q4_0.gguf"
set -l bin "$root/target/release/rocmforge"
set -l llama_simple "$HOME/tmp/llama.cpp/build-rocm/bin/llama-simple"

if not test -f "$model"
    echo "Model not found: $model" >&2; exit 1
end
if not test -x "$bin"
    cargo build --release --features gpu; or exit 1
end

set -l sha (git -C $root rev-parse --short HEAD 2>/dev/null; or echo dev)
set -l ts (date +%s)
set -l answers_dir "$root/benches/results/answers"
mkdir -p $answers_dir

set -l prompts_dir "$root/benches/prompts"
set -l names code_01 code_02 code_03 code_04 code_05 \
              chat_01 chat_02 chat_03 chat_04 chat_05 \
              prose_01 prose_02 prose_03 prose_04 prose_05

# ---------- Parse helpers ---------------------------------------------
function parse_rocmforge_out
    set -l ptok 0; set -l prefill_ms 0; set -l prefill_tps 0
    set -l dec_ms 0; set -l dec_tps 0; set -l alpha 0
    for line in $argv
        set -l m (string match -r '^Prompt: ([0-9]+) tokens' -- $line)
        if test (count $m) -ge 2; set ptok $m[2]; end
        set -l m (string match -r '^Prefill: ([0-9.]+)ms \(([0-9.]+) tok/s\)' -- $line)
        if test (count $m) -ge 3
            set prefill_ms $m[2]; set prefill_tps $m[3]
        end
        set -l m (string match -r '^([0-9]+) tokens in ([0-9.]+)ms = ([0-9.]+) tok/s' -- $line)
        if test (count $m) -ge 4
            set dec_ms $m[3]; set dec_tps $m[4]
        end
        set -l m (string match -r 'speculative: [0-9]+/[0-9]+ accepted = ([0-9.]+)%' -- $line)
        if test (count $m) -ge 2; set alpha $m[2]; end
    end
    echo "$ptok $prefill_ms $prefill_tps $dec_ms $dec_tps $alpha"
end

# ---------- ROCmForge -------------------------------------------------
function rocmforge_run_A --argument-names bin model prompt save_path
    set -l out ($bin --model $model --prompt "$prompt" --max-tokens 128 --gpu \
                     --temperature 0.0 --top-p 1.0 --no-template 2>&1)
    if test -n "$save_path"; printf '%s\n' $out > $save_path; end
    parse_rocmforge_out $out
end

function rocmforge_run_B --argument-names bin model draft prompt save_path
    set -l out ($bin --model $model --draft-model $draft --spec-depth 1 \
                     --prompt "$prompt" --max-tokens 128 --gpu \
                     --temperature 0.0 --top-p 1.0 --no-template 2>&1)
    if test -n "$save_path"; printf '%s\n' $out > $save_path; end
    parse_rocmforge_out $out
end

function rocmforge_run_C --argument-names bin model prompt save_path
    set -l out (env ROCMFORGE_DISABLE_WMMA_PREFILL=1 \
                    ROCMFORGE_DISABLE_WMMA_ATTENTION=1 \
                    ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1 \
                    ROCMFORGE_DISABLE_TILED_GEMV=1 \
                    ROCMFORGE_DISABLE_BATCHED_LM_HEAD=1 \
                    $bin --model $model --prompt "$prompt" --max-tokens 128 --gpu \
                         --temperature 0.0 --top-p 1.0 --no-template 2>&1)
    if test -n "$save_path"; printf '%s\n' $out > $save_path; end
    parse_rocmforge_out $out
end

# ---------- llama.cpp raw completion ----------------------------------
function llamacpp_run --argument-names llama_simple model prompt save_path
    set -l out ($llama_simple -m $model -n 128 -ngl 99 "$prompt" 2>&1)
    if test -n "$save_path"; printf '%s\n' $out > $save_path; end
    set -l pp_tps 0; set -l dec_tps 0; set -l pp_tokens 0
    for line in $out
        set -l m (string match -r 'prompt eval time = +([0-9.]+) ms / +([0-9]+) tokens .+, +([0-9.]+) tokens per second' -- $line)
        if test (count $m) -ge 4
            set pp_tokens $m[3]; set pp_tps $m[4]
        end
        set -l m (string match -r 'eval time = +([0-9.]+) ms / +([0-9]+) runs .+, +([0-9.]+) tokens per second' -- $line)
        if test (count $m) -ge 4
            set dec_tps $m[4]
        end
    end
    echo "$pp_tokens $pp_tps $dec_tps"
end

# ---------- JSON emitter ----------------------------------------------
set -l out_json "$root/benches/results/full_benchmark_wmma_{$sha}_{$ts}.json"
echo "{" > $out_json
echo "  \"git_sha\": \"$sha\"," >> $out_json
echo "  \"timestamp\": $ts," >> $out_json
echo "  \"model\": \"Qwen2.5-7B-Instruct-Q4_0\"," >> $out_json
echo "  \"runs\": [" >> $out_json

set -l first 1
for name in $names
    set -l prompt_file "$prompts_dir/$name.txt"
    set -l prompt (cat $prompt_file)
    echo ">>> $name" >&2

    for mode in A B C
        for run in 1 2 3
            set -l save ""
            if test $run -eq 1
                set save "$answers_dir/rocmforge_{$mode}_{$name}.txt"
            end
            set -l line
            switch $mode
                case A
                    set line (rocmforge_run_A $bin $model "$prompt" "$save")
                case B
                    set line (rocmforge_run_B $bin $model $draft "$prompt" "$save")
                case C
                    set line (rocmforge_run_C $bin $model "$prompt" "$save")
            end
            set -l p (string split " " $line)
            if test $first -eq 0; echo "    ," >> $out_json; else; set first 0; end
            printf '    {"prompt": "%s", "engine": "rocmforge", "mode": "%s", "run": %d, "prompt_tokens": %s, "prefill_ms": %s, "prefill_tps": %s, "decode_ms": %s, "decode_tps": %s, "alpha_pct": %s}\n' \
                $name $mode $run $p[1] $p[2] $p[3] $p[4] $p[5] $p[6] >> $out_json
            echo "  $name $mode run=$run: ptok=$p[1] prefill=$p[3] tok/s decode=$p[5] tok/s α=$p[6]%" >&2
        end
    end

    for run in 1 2 3
        set -l save ""
        if test $run -eq 1
            set save "$answers_dir/llamacpp_{$name}.txt"
        end
        set -l line (llamacpp_run $llama_simple $model "$prompt" "$save")
        set -l p (string split " " $line)
        echo "    ," >> $out_json
        printf '    {"prompt": "%s", "engine": "llamacpp", "mode": "raw", "run": %d, "prompt_tokens": %s, "prefill_tps": %s, "decode_tps": %s}\n' \
            $name $run $p[1] $p[2] $p[3] >> $out_json
        echo "  $name llamacpp run=$run: ptok=$p[1] prefill=$p[2] tok/s decode=$p[3] tok/s" >&2
    end
end

echo "  ]" >> $out_json
echo "}" >> $out_json

echo "Wrote $out_json" >&2
