#!/usr/bin/env fish
#
# ROCm validation suite — reusable before/after snapshot harness.
#
# Part A: build fingerprint (ROCm/hipcc/driver/WMMA intrinsic availability)
# Part B: synthetic benchmark (prefill pp64/128/256/512 + decode + layer timing)
# Part C: 15-prompt real benchmark (Config A = WMMA active, with answer dumps)
# Part D: summary.json aggregating key metrics for diffing
#
# All outputs land in benches/results/rocm_baseline/rocm_<ver>_<ts>/ and the
# same script can be re-run after a ROCm package upgrade to produce a second
# snapshot for benches/rocm_diff.fish.

set -l here (dirname (status filename))
set -l root (realpath "$here/..")
set -l model "$HOME/models/Qwen2.5-7B-Instruct-Q4_0.gguf"
set -l bin "$root/target/release/rocmforge"
set -l prompts_dir "$root/benches/prompts"

if not test -f "$model"
    echo "❌ Model not found: $model" >&2
    exit 1
end

# --- Configuration ----------------------------------------------------
set -l ROCM_VERSION (cat /opt/rocm/.info/version 2>/dev/null; or echo unknown)
set -l TIMESTAMP (date +%s)
set -l OUTDIR "$root/benches/results/rocm_baseline/rocm_"$ROCM_VERSION"_"$TIMESTAMP
mkdir -p "$OUTDIR/answers"

echo "=== ROCm Validation Suite ==="
echo "ROCm Version: $ROCM_VERSION"
echo "Output:       $OUTDIR"
echo ""

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

function strip_ansi
    # Strip ANSI CSI sequences (tracing-subscriber colors). Input on stdin.
    sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g'
end

function json_escape
    # Escape a single string for JSON value use. Input via argv.
    string replace -a '\\' '\\\\' -- $argv[1] \
        | string replace -a '"' '\\"' \
        | string replace -a \t '\\t' \
        | string replace -a \n '\\n'
end

function percentile
    # Usage: percentile <p> <numbers...>. Returns the requested percentile.
    # p = 50 for median, 90 for p90, etc. Empty input -> 0.
    set -l p $argv[1]
    set -l values $argv[2..-1]
    set -l n (count $values)
    if test $n -eq 0
        echo 0
        return
    end
    set -l sorted (printf '%s\n' $values | sort -g)
    set -l idx (math --scale=0 "ceil($n * $p / 100)")
    if test $idx -lt 1; set idx 1; end
    if test $idx -gt $n; set idx $n; end
    echo $sorted[$idx]
end

function median
    percentile 50 $argv
end

# ---------------------------------------------------------------------
# Part A: Build Fingerprint
# ---------------------------------------------------------------------
echo "--- Part A: Build Fingerprint ---"

set -l hipcc_path (command -v hipcc 2>/dev/null; or echo "")
set -l hipcc_version ""
if test -n "$hipcc_path"
    set hipcc_version (hipcc --version 2>/dev/null | head -1 | string trim)
end

set -l amdclang_path (command -v amdclang 2>/dev/null; or echo "")
set -l amdclang_version ""
if test -n "$amdclang_path"
    set amdclang_version (amdclang --version 2>/dev/null | head -1 | string trim)
end

set -l gpu_arch unknown
set -l rocminfo_bin (command -v rocminfo 2>/dev/null)
if test -n "$rocminfo_bin"
    set gpu_arch ($rocminfo_bin 2>/dev/null | grep -E "^\s+Name:\s+gfx" | head -1 | string trim | string replace -r '^Name:\s+' '')
end
if test -z "$gpu_arch"; set gpu_arch unknown; end

set -l gpu_marketing unknown
if test -n "$rocminfo_bin"
    # rocminfo lists CPU agents first; pick the Marketing Name that belongs
    # to the same Agent block as the gfx12xx Name.
    set gpu_marketing ($rocminfo_bin 2>/dev/null | awk '
        /^\*\*\*\*\*/ { in_block=1; name=""; mkt=""; next }
        /^  Name:[[:space:]]+gfx/ { sub(/^  Name:[[:space:]]+/, "", $0); name=$0 }
        /^  Marketing Name:/ { sub(/^  Marketing Name:[[:space:]]+/, "", $0); mkt=$0 }
        name ~ /^gfx1[0-9]+/ && mkt != "" { print mkt; exit }
    ' | string trim)
end
if test -z "$gpu_marketing"; set gpu_marketing unknown; end

set -l gpu_driver (cat /sys/module/amdgpu/version 2>/dev/null; or echo unknown)
if test -z "$gpu_driver"; set gpu_driver unknown; end

set -l wmma_intrinsic "not found"
if grep -rql "wmma_f32_16x16x16_f16_w32_gfx12" /opt/rocm/lib/llvm/ 2>/dev/null
    set wmma_intrinsic found
end

set -l hipblas_path (find /opt/rocm/lib -maxdepth 2 -name "libhipblas.so*" 2>/dev/null | head -1)
if test -z "$hipblas_path"; set hipblas_path "not found"; end
set -l hip_runtime_path (find /opt/rocm/lib -maxdepth 2 -name "libamdhip64.so*" 2>/dev/null | head -1)
if test -z "$hip_runtime_path"; set hip_runtime_path "not found"; end

set -l hipblas_pkg (pacman -Q hipblas 2>/dev/null | awk '{print $2}')
if test -z "$hipblas_pkg"; set hipblas_pkg unknown; end
set -l hip_runtime_pkg (pacman -Q hip-runtime-amd 2>/dev/null | awk '{print $2}')
if test -z "$hip_runtime_pkg"; set hip_runtime_pkg unknown; end
set -l rocm_core_pkg (pacman -Q rocm-core 2>/dev/null | awk '{print $2}')
if test -z "$rocm_core_pkg"; set rocm_core_pkg unknown; end
set -l rocm_hip_libraries_pkg (pacman -Q rocm-hip-libraries 2>/dev/null | awk '{print $2}')
if test -z "$rocm_hip_libraries_pkg"; set rocm_hip_libraries_pkg unknown; end

set -l kernel_count_hip 0
set -l hip_kernel_dir "$root/src/gpu/kernels"
if test -d "$hip_kernel_dir"
    set kernel_count_hip (find "$hip_kernel_dir" -name "*.hip" -o -name "*.cu" 2>/dev/null | wc -l | string trim)
end

set -l host_name (hostname)
set -l iso_timestamp (date --iso-8601=seconds)
set -l git_sha (git -C $root rev-parse --short HEAD 2>/dev/null; or echo dev)

set -l fp "$OUTDIR/fingerprint.json"
printf '{\n' > $fp
printf '  "rocm_version": "%s",\n' (json_escape $ROCM_VERSION) >> $fp
printf '  "hipcc_path": "%s",\n' (json_escape $hipcc_path) >> $fp
printf '  "hipcc_version": "%s",\n' (json_escape $hipcc_version) >> $fp
printf '  "amdclang_path": "%s",\n' (json_escape $amdclang_path) >> $fp
printf '  "amdclang_version": "%s",\n' (json_escape $amdclang_version) >> $fp
printf '  "gpu_arch": "%s",\n' (json_escape $gpu_arch) >> $fp
printf '  "gpu_marketing_name": "%s",\n' (json_escape $gpu_marketing) >> $fp
printf '  "gpu_driver_version": "%s",\n' (json_escape $gpu_driver) >> $fp
printf '  "wmma_intrinsic_available": %s,\n' (test "$wmma_intrinsic" = found; and echo true; or echo false) >> $fp
printf '  "hipblas_path": "%s",\n' (json_escape $hipblas_path) >> $fp
printf '  "hipblas_package_version": "%s",\n' (json_escape $hipblas_pkg) >> $fp
printf '  "hip_runtime_path": "%s",\n' (json_escape $hip_runtime_path) >> $fp
printf '  "hip_runtime_package_version": "%s",\n' (json_escape $hip_runtime_pkg) >> $fp
printf '  "rocm_core_package_version": "%s",\n' (json_escape $rocm_core_pkg) >> $fp
printf '  "rocm_hip_libraries_package_version": "%s",\n' (json_escape $rocm_hip_libraries_pkg) >> $fp
printf '  "kernel_count_hip": %s,\n' $kernel_count_hip >> $fp
printf '  "git_sha": "%s",\n' (json_escape $git_sha) >> $fp
printf '  "hostname": "%s",\n' (json_escape $host_name) >> $fp
printf '  "timestamp": "%s"\n' (json_escape $iso_timestamp) >> $fp
printf '}\n' >> $fp

echo "  ROCm:      $ROCM_VERSION"
echo "  GPU:       $gpu_arch  ($gpu_marketing)"
echo "  Driver:    $gpu_driver"
echo "  WMMA:      $wmma_intrinsic"
echo "  hipBLAS:   $hipblas_pkg"
echo "  HIP rt:    $hip_runtime_pkg"

# Correctness gate
set -l fp_ok 1
if test "$ROCM_VERSION" = unknown
    echo "  ⚠ ROCm version not detected"
    set fp_ok 0
end
if test "$wmma_intrinsic" = "not found"
    echo "  ⚠ WMMA intrinsic not found in LLVM headers"
end

# --- Build ------------------------------------------------------------
echo ""
echo "--- Build (cargo --release --features gpu) ---"
cargo build --release --features gpu --manifest-path "$root/Cargo.toml" 2>&1 | tee "$OUTDIR/build_verbose.log"
set -l build_status $pipestatus[1]
if test $build_status -ne 0
    echo "❌ Build FAILED — see $OUTDIR/build_verbose.log"
    exit 1
end
if not test -x "$bin"
    echo "❌ Binary not found at $bin after build"
    exit 1
end
echo "✅ Build OK"

# ---------------------------------------------------------------------
# Part B: Synthetic Benchmark (prefill pp64/128/256/512 + decode)
# ---------------------------------------------------------------------
echo ""
echo "--- Part B: Synthetic Benchmark ---"

set -l synth_raw "$OUTDIR/synthetic_bench_raw.txt"
: > $synth_raw

function parse_rocmforge_line_prompt
    # Return just the token count from "Prompt: N tokens". Empty on no match.
    set -l m (string match -r '^Prompt: ([0-9]+) tokens' -- $argv[1])
    if test (count $m) -ge 2; echo $m[2]; end
end

function parse_rocmforge_line_prefill
    # Return "ms tps" from "Prefill: Xms (Y tok/s)". Empty on no match.
    set -l m (string match -r '^Prefill: ([0-9.]+)ms \(([0-9.]+) tok/s\)' -- $argv[1])
    if test (count $m) -ge 3; echo "$m[2] $m[3]"; end
end

function parse_rocmforge_line_decode
    # Return "tokens ms tps" from "N tokens in Xms = Y tok/s". Empty on no match.
    set -l m (string match -r '^([0-9]+) tokens in ([0-9.]+)ms = ([0-9.]+) tok/s' -- $argv[1])
    if test (count $m) -ge 4; echo "$m[2] $m[3] $m[4]"; end
end

function run_synth --argument-names bin model prompt max_tokens
    # Merge stderr so "Prompt:", "Prefill:", and decode-summary lines are captured.
    $bin --model $model --prompt "$prompt" --max-tokens $max_tokens --gpu \
         --temperature 0.0 --top-p 1.0 --no-template 2>&1
end

# Build prompt strings of known token counts (~1 token per "word " via BPE).
# Token count is measured and reported by the binary, not assumed.
set -l pp_values 64 128 256 512
set -l pp_runs 3

for pp in $pp_values
    set -l prompt (string repeat -n $pp "word ")
    echo "  Prefill pp~=$pp (3 runs)..."
    for run in (seq 1 $pp_runs)
        echo "### pp=$pp run=$run" >> $synth_raw
        run_synth $bin $model "$prompt" 1 >> $synth_raw
        echo "" >> $synth_raw
    end
end

echo "  Decode 128 tokens (3 runs)..."
for run in (seq 1 3)
    echo "### decode run=$run" >> $synth_raw
    run_synth $bin $model "Hello" 128 >> $synth_raw
    echo "" >> $synth_raw
end

# --- Parse synthetic results -----------------------------------------
set -l synth_json "$OUTDIR/synthetic_bench.json"

# Aggregate per-pp prefill tok/s and ms; collect lists.
printf '{\n' > $synth_json
printf '  "rocm_version": "%s",\n' (json_escape $ROCM_VERSION) >> $synth_json
printf '  "git_sha": "%s",\n' (json_escape $git_sha) >> $synth_json
printf '  "prefill": {\n' >> $synth_json

set -l pp_count (count $pp_values)
set -l pp_idx 0
set -g pp64_tps_median 0
set -g pp128_tps_median 0
set -g pp256_tps_median 0
set -g pp512_tps_median 0
for pp in $pp_values
    set pp_idx (math $pp_idx + 1)
    set -l ptoks
    set -l pms
    set -l ptps
    set -l in_block 0
    for line in (cat $synth_raw)
        if string match -q "### pp=$pp run=*" -- $line
            set in_block 1
            continue
        else if string match -q '### *' -- $line
            set in_block 0
            continue
        end
        if test $in_block -eq 1
            set -l pt (parse_rocmforge_line_prompt $line)
            if test -n "$pt"; set -a ptoks $pt; end
            set -l pf (parse_rocmforge_line_prefill $line)
            if test -n "$pf"
                set -l parts (string split ' ' $pf)
                set -a pms $parts[1]
                set -a ptps $parts[2]
            end
        end
    end
    set -l ptok_median (median $ptoks)
    set -l pms_median (median $pms)
    set -l ptps_median (median $ptps)
    set -l ptps_max (percentile 100 $ptps)
    set -g "pp"$pp"_tps_median" $ptps_median
    set -l trailing_comma ,
    if test $pp_idx -eq $pp_count; set trailing_comma ""; end
    printf '    "pp_%s": {\n' $pp >> $synth_json
    printf '      "target_prompt_tokens": %s,\n' $pp >> $synth_json
    printf '      "measured_prompt_tokens_median": %s,\n' $ptok_median >> $synth_json
    printf '      "prefill_ms_median": %s,\n' $pms_median >> $synth_json
    printf '      "prefill_tps_median": %s,\n' $ptps_median >> $synth_json
    printf '      "prefill_tps_best": %s,\n' $ptps_max >> $synth_json
    printf '      "runs_ms": [%s],\n' (string join ', ' $pms) >> $synth_json
    printf '      "runs_tps": [%s]\n' (string join ', ' $ptps) >> $synth_json
    printf '    }%s\n' $trailing_comma >> $synth_json
end
printf '  },\n' >> $synth_json

# Decode parsing
set -l dec_ms_list
set -l dec_tps_list
set -l in_block 0
for line in (cat $synth_raw)
    if string match -q '### decode run=*' -- $line
        set in_block 1
        continue
    else if string match -q '### *' -- $line
        set in_block 0
        continue
    end
    if test $in_block -eq 1
        set -l dp (parse_rocmforge_line_decode $line)
        if test -n "$dp"
            set -l parts (string split ' ' $dp)
            set -a dec_ms_list $parts[2]
            set -a dec_tps_list $parts[3]
        end
    end
end
set -l dec_tps_median (median $dec_tps_list)
set -l dec_tps_max (percentile 100 $dec_tps_list)
set -l dec_ms_median (median $dec_ms_list)
printf '  "decode_128": {\n' >> $synth_json
printf '    "tps_median": %s,\n' $dec_tps_median >> $synth_json
printf '    "tps_best": %s,\n' $dec_tps_max >> $synth_json
printf '    "ms_median": %s,\n' $dec_ms_median >> $synth_json
printf '    "runs_tps": [%s]\n' (string join ', ' $dec_tps_list) >> $synth_json
printf '  }\n' >> $synth_json
printf '}\n' >> $synth_json

echo "  ✅ Synthetic parsed -> $synth_json"

# --- Layer-timing (single trace run) ---------------------------------
echo "  Layer timing (RUST_LOG=trace, 1 run at pp~=64)..."
set -l prompt64 (string repeat -n 64 "word ")
set -l layer_raw "$OUTDIR/layer_timing_raw.log"
set -l layer_txt "$OUTDIR/layer_timing.txt"

env RUST_LOG=trace $bin --model $model --prompt "$prompt64" \
    --max-tokens 1 --gpu --temperature 0.0 --top-p 1.0 --no-template \
    2> $layer_raw >/dev/null

# Strip ANSI, keep only "Prefill layer launched" lines
cat $layer_raw | strip_ansi | grep "Prefill layer launched" > $layer_txt

set -l layer_us_list
for line in (cat $layer_txt)
    set -l m (string match -r 'launch_us=([0-9]+)' -- $line)
    if test (count $m) -ge 2
        set -a layer_us_list $m[2]
    end
end

set -l layer_json "$OUTDIR/layer_timing.json"
set -l n_layers (count $layer_us_list)
set -l layer0_us 0
if test $n_layers -ge 1; set layer0_us $layer_us_list[1]; end
set -l rest_us
if test $n_layers -ge 2; set rest_us $layer_us_list[2..-1]; end
set -l rest_median (median $rest_us)
set -l rest_p90 (percentile 90 $rest_us)
set -l rest_max (percentile 100 $rest_us)
set -l ratio 0
if test "$rest_median" != "0" -a "$rest_median" != ""
    set ratio (math --scale=2 "$layer0_us / $rest_median")
end

printf '{\n' > $layer_json
printf '  "rocm_version": "%s",\n' (json_escape $ROCM_VERSION) >> $layer_json
printf '  "prompt_tokens_target": 64,\n' >> $layer_json
printf '  "layer_count": %s,\n' $n_layers >> $layer_json
printf '  "layer0_us": %s,\n' $layer0_us >> $layer_json
printf '  "layer1_to_n_median_us": %s,\n' $rest_median >> $layer_json
printf '  "layer1_to_n_p90_us": %s,\n' $rest_p90 >> $layer_json
printf '  "layer1_to_n_max_us": %s,\n' $rest_max >> $layer_json
printf '  "layer0_over_median_ratio": %s,\n' $ratio >> $layer_json
printf '  "all_layers_us": [%s]\n' (string join ', ' $layer_us_list) >> $layer_json
printf '}\n' >> $layer_json

echo "  ✅ Layer timing: L0=$layer0_us µs, median(L1..)=$rest_median µs, ratio=$ratio""×"

# Correctness gate on synthetic
set -l synth_ok 1
if test $n_layers -lt 28
    echo "  ⚠ Layer-timing captured only $n_layers layers (expected >=28)"
    set synth_ok 0
end

# ---------------------------------------------------------------------
# Part C: 15-prompt benchmark (Config A = WMMA on, with answer dumps)
# ---------------------------------------------------------------------
echo ""
echo "--- Part C: 15-Prompt Benchmark (Config A) ---"

set -l prompt_files
for f in $prompts_dir/code_*.txt $prompts_dir/chat_*.txt $prompts_dir/prose_*.txt
    set -a prompt_files $f
end
set -l prompt_count (count $prompt_files)
if test $prompt_count -ne 15
    echo "⚠ Expected 15 prompt files, found $prompt_count"
end

set -l prompt_raw "$OUTDIR/prompt_bench_raw.txt"
: > $prompt_raw

set -l prompt_json "$OUTDIR/prompt_bench.json"
printf '{\n' > $prompt_json
printf '  "rocm_version": "%s",\n' (json_escape $ROCM_VERSION) >> $prompt_json
printf '  "git_sha": "%s",\n' (json_escape $git_sha) >> $prompt_json
printf '  "config": "A (WMMA prefill + WMMA attention + hipBLAS + tiled GEMV + batched lm_head)",\n' >> $prompt_json
printf '  "prompts": [\n' >> $prompt_json

set -l pi 0
for f in $prompt_files
    set pi (math $pi + 1)
    set -l name (basename $f .txt)
    set -l prompt (cat $f)

    echo "  $name..."
    set -l ptok_list
    set -l pms_list
    set -l ptps_list
    set -l dec_tps_list2
    set -l dec_ms_list2
    # 3 runs, no saving — merge stderr so stats lines are captured
    for run in 1 2 3
        echo "### $name run=$run" >> $prompt_raw
        set -l out ($bin --model $model --prompt "$prompt" --max-tokens 128 --gpu \
                         --temperature 0.0 --top-p 1.0 2>&1)
        printf '%s\n' $out >> $prompt_raw
        echo "" >> $prompt_raw
        for line in $out
            set -l pt (parse_rocmforge_line_prompt $line)
            if test -n "$pt"; set -a ptok_list $pt; end
            set -l pf (parse_rocmforge_line_prefill $line)
            if test -n "$pf"
                set -l parts (string split ' ' $pf)
                set -a pms_list $parts[1]
                set -a ptps_list $parts[2]
            end
            set -l dp (parse_rocmforge_line_decode $line)
            if test -n "$dp"
                set -l parts (string split ' ' $dp)
                set -a dec_ms_list2 $parts[2]
                set -a dec_tps_list2 $parts[3]
            end
        end
    end

    # 4th run: save the generated answer (greedy -> deterministic)
    set -l answer_file "$OUTDIR/answers/rocmforge_"$name".txt"
    $bin --model $model --prompt "$prompt" --max-tokens 128 --gpu \
         --temperature 0.0 --top-p 1.0 2>/dev/null > $answer_file

    set -l ptok_m (median $ptok_list)
    set -l pms_m (median $pms_list)
    set -l ptps_m (median $ptps_list)
    set -l dec_tps_m (median $dec_tps_list2)
    set -l dec_ms_m (median $dec_ms_list2)
    set -l trailing_comma ,
    if test $pi -eq $prompt_count; set trailing_comma ""; end
    printf '    {\n' >> $prompt_json
    printf '      "name": "%s",\n' (json_escape $name) >> $prompt_json
    printf '      "prompt_tokens_median": %s,\n' $ptok_m >> $prompt_json
    printf '      "prefill_ms_median": %s,\n' $pms_m >> $prompt_json
    printf '      "prefill_tps_median": %s,\n' $ptps_m >> $prompt_json
    printf '      "decode_ms_median": %s,\n' $dec_ms_m >> $prompt_json
    printf '      "decode_tps_median": %s,\n' $dec_tps_m >> $prompt_json
    printf '      "runs_prefill_tps": [%s],\n' (string join ', ' $ptps_list) >> $prompt_json
    printf '      "runs_decode_tps": [%s]\n' (string join ', ' $dec_tps_list2) >> $prompt_json
    printf '    }%s\n' $trailing_comma >> $prompt_json
end

printf '  ]\n' >> $prompt_json
printf '}\n' >> $prompt_json

# Correctness gate on 15-prompt
set -l answer_count (count $OUTDIR/answers/rocmforge_*.txt)
set -l prompt_ok 1
if test $answer_count -ne 15
    echo "  ❌ Expected 15 answer files, found $answer_count"
    set prompt_ok 0
end
for f in $OUTDIR/answers/rocmforge_*.txt
    set -l sz (wc -c < $f | string trim)
    if test $sz -lt 10
        echo "  ⚠ $f is suspiciously small ($sz bytes)"
        set prompt_ok 0
    end
end
if test $prompt_ok -eq 1
    echo "  ✅ 15/15 answer files saved"
end

# ---------------------------------------------------------------------
# Part D: Summary
# ---------------------------------------------------------------------
echo ""
echo "--- Part D: Summary ---"

# Pull TTFT median across the 15 prompts (prefill_ms_median values).
set -l ttft_values
for f in $OUTDIR/answers/rocmforge_*.txt
    # no-op — keeps ordering deterministic while using the real values below
end
# Reparse prompt_json to collect prefill_ms_medians across prompts.
set -l prompt_ttfts (grep '"prefill_ms_median"' $prompt_json | string match -r '([0-9.]+)' | string match -r '^[0-9.]+$')
set -l ttft_median (median $prompt_ttfts)

# Decode tps across the 15 prompts
set -l prompt_decodes (grep '"decode_tps_median"' $prompt_json | string match -r '([0-9.]+)' | string match -r '^[0-9.]+$')
set -l prompt_decode_median (median $prompt_decodes)

set -l summary "$OUTDIR/summary.json"
printf '{\n' > $summary
printf '  "rocm_version": "%s",\n' (json_escape $ROCM_VERSION) >> $summary
printf '  "git_sha": "%s",\n' (json_escape $git_sha) >> $summary
printf '  "timestamp": "%s",\n' (json_escape $iso_timestamp) >> $summary
printf '  "gpu_arch": "%s",\n' (json_escape $gpu_arch) >> $summary
printf '  "wmma_intrinsic_available": %s,\n' (test "$wmma_intrinsic" = found; and echo true; or echo false) >> $summary
printf '  "build_ok": true,\n' >> $summary
printf '  "synthetic": {\n' >> $summary
printf '    "prefill_pp64_tps_median": %s,\n' $pp64_tps_median >> $summary
printf '    "prefill_pp128_tps_median": %s,\n' $pp128_tps_median >> $summary
printf '    "prefill_pp256_tps_median": %s,\n' $pp256_tps_median >> $summary
printf '    "prefill_pp512_tps_median": %s,\n' $pp512_tps_median >> $summary
printf '    "decode_128_tps_median": %s\n' $dec_tps_median >> $summary
printf '  },\n' >> $summary
printf '  "real_prompts": {\n' >> $summary
printf '    "count": %s,\n' $prompt_count >> $summary
printf '    "ttft_ms_median": %s,\n' $ttft_median >> $summary
printf '    "decode_tps_median": %s\n' $prompt_decode_median >> $summary
printf '  },\n' >> $summary
printf '  "layer_timing": {\n' >> $summary
printf '    "layer0_us": %s,\n' $layer0_us >> $summary
printf '    "layer1_to_n_median_us": %s,\n' $rest_median >> $summary
printf '    "layer0_over_median_ratio": %s\n' $ratio >> $summary
printf '  }\n' >> $summary
printf '}\n' >> $summary

echo ""
echo "=== Baseline Summary ==="
echo "ROCm:                $ROCM_VERSION"
echo "GPU:                 $gpu_arch ($gpu_marketing)"
echo "WMMA intrinsic:      $wmma_intrinsic"
echo "Build:               OK"
echo "Prefill pp256 (med): $pp256_tps_median tok/s"
echo "Decode 128 (med):    $dec_tps_median tok/s"
echo "TTFT 15-prompt med:  $ttft_median ms"
echo "Decode 15-prompt med:$prompt_decode_median tok/s"
echo "Layer-0 warmup:      $layer0_us µs"
echo "Layer 1..N median:   $rest_median µs"
echo "Ratio L0/L1..:       $ratio""×"
echo "Answer files:        $answer_count/15"
echo ""
echo "Baseline saved to:   $OUTDIR"
echo ""
echo "Next steps (after ROCm upgrade):"
echo "  fish benches/rocm_validate.fish"
echo "  fish benches/rocm_diff.fish $OUTDIR <new_dir>"
