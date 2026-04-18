#!/usr/bin/env fish
#
# ROCm baseline diff tool.
#
# Compares two baseline directories produced by rocm_validate.fish and emits
# a BUILD / CORRECTNESS / PERFORMANCE / VERDICT report to stdout plus a
# Markdown copy in benches/results/rocm_baseline/diff_<old>_vs_<new>.md.
#
# Usage:
#   fish benches/rocm_diff.fish <old_dir> <new_dir>

if test (count $argv) -lt 2
    echo "Usage: fish benches/rocm_diff.fish <old_dir> <new_dir>"
    exit 1
end

set -l OLD $argv[1]
set -l NEW $argv[2]

if not test -d "$OLD"
    echo "❌ Baseline dir not found: $OLD"; exit 1
end
if not test -d "$NEW"
    echo "❌ New dir not found: $NEW"; exit 1
end

# --- Tool selection ---------------------------------------------------
set -g JQ (command -v jq 2>/dev/null)
set -g PY (command -v python3 2>/dev/null)
if test -z "$JQ" -a -z "$PY"
    echo "❌ Neither jq nor python3 found — cannot parse JSON." >&2
    exit 1
end

function json_get --argument-names file field
    # Dotted-path lookup (e.g. "synthetic.decode_128_tps_median"). Prints "N/A"
    # when the field or file is missing.
    if not test -f "$file"
        echo "N/A"; return
    end
    if test -n "$JQ"
        set -l v ($JQ -r ".$field // \"N/A\"" $file 2>/dev/null)
        if test -z "$v"; set v "N/A"; end
        echo $v
    else
        python3 -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    for part in sys.argv[2].split('.'):
        d = d.get(part, 'N/A') if isinstance(d, dict) else 'N/A'
    print(d)
except Exception:
    print('N/A')
" $file $field
    end
end

# --- Output plumbing --------------------------------------------------
set -g root (realpath (dirname (status filename))/..)
set -g OLD_VERSION (json_get $OLD/fingerprint.json rocm_version)
set -g NEW_VERSION (json_get $NEW/fingerprint.json rocm_version)
set -g DIFF_FILE "$root/benches/results/rocm_baseline/diff_"$OLD_VERSION"_vs_"$NEW_VERSION".md"
: > $DIFF_FILE

function report
    echo $argv
    echo $argv >> $DIFF_FILE
end

# --- Deltas / rating --------------------------------------------------
function pct_delta --argument-names old new
    # Returns percentage change (new - old) / old * 100, to 2 dp.
    # "N/A" if either side is non-numeric.
    if not string match -qr '^-?[0-9]+(\.[0-9]+)?$' -- "$old"
        echo N/A; return
    end
    if not string match -qr '^-?[0-9]+(\.[0-9]+)?$' -- "$new"
        echo N/A; return
    end
    if test "$old" = "0"
        echo N/A; return
    end
    math --scale=2 "($new - $old) * 100 / $old"
end

function rate_delta --argument-names pct
    if test "$pct" = "N/A"
        echo "N/A"; return
    end
    awk -v p="$pct" 'BEGIN {
        a = (p < 0) ? -p : p
        if (a < 5)       print "✅"
        else if (a < 10) print "⚠️"
        else             print "❌"
    }'
end

function fmt_delta --argument-names old new unit
    set -l pct (pct_delta $old $new)
    set -l rating (rate_delta $pct)
    if test "$pct" = "N/A"
        printf '%s → %s %s (n/a)' $old $new $unit
    else
        set -l sign ""
        if string match -qr '^-' -- $pct
            # already has a leading minus
        else
            set sign "+"
        end
        printf '%s → %s %s (%s%s%%, %s)' $old $new $unit $sign $pct $rating
    end
end

# ======================================================================
# Header
# ======================================================================
set -l now (date --iso-8601=seconds)
report "# ROCm Upgrade Diff: $OLD_VERSION → $NEW_VERSION"
report ""
report "- **Date:** $now"
report "- **Baseline:** \`$OLD\`"
report "- **New:**      \`$NEW\`"
report ""

# ======================================================================
# [BUILD] — Fingerprint diff
# ======================================================================
report "## [BUILD]"
report ""
report '```'

set -g build_ok_overall 1

function cmp_field --argument-names label old_val new_val
    set -l mark "✅ unchanged"
    if test "$old_val" != "$new_val"
        set mark "⚠️ changed"
    end
    if test "$old_val" = "N/A" -o "$new_val" = "N/A"
        set mark "❌ missing"
    end
    printf '  %-22s %s → %s   %s\n' "$label:" "$old_val" "$new_val" "$mark"
end

set -l old_rocm (json_get $OLD/fingerprint.json rocm_version)
set -l new_rocm (json_get $NEW/fingerprint.json rocm_version)
function short_version
    # Extract "X.Y.Z[-suffix]" from a version banner line.
    string match -r '[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9._+-]*' -- $argv[1] | head -1
end
set -l old_hipcc_v (short_version (json_get $OLD/fingerprint.json hipcc_version))
set -l new_hipcc_v (short_version (json_get $NEW/fingerprint.json hipcc_version))
set -l old_amdclang_v (short_version (json_get $OLD/fingerprint.json amdclang_version))
set -l new_amdclang_v (short_version (json_get $NEW/fingerprint.json amdclang_version))
if test -z "$old_hipcc_v"; set old_hipcc_v "N/A"; end
if test -z "$new_hipcc_v"; set new_hipcc_v "N/A"; end
if test -z "$old_amdclang_v"; set old_amdclang_v "N/A"; end
if test -z "$new_amdclang_v"; set new_amdclang_v "N/A"; end
set -l old_hipblas (json_get $OLD/fingerprint.json hipblas_package_version)
set -l new_hipblas (json_get $NEW/fingerprint.json hipblas_package_version)
set -l old_hip_rt (json_get $OLD/fingerprint.json hip_runtime_package_version)
set -l new_hip_rt (json_get $NEW/fingerprint.json hip_runtime_package_version)
set -l old_rocm_core (json_get $OLD/fingerprint.json rocm_core_package_version)
set -l new_rocm_core (json_get $NEW/fingerprint.json rocm_core_package_version)
set -l old_arch (json_get $OLD/fingerprint.json gpu_arch)
set -l new_arch (json_get $NEW/fingerprint.json gpu_arch)
set -l old_driver (json_get $OLD/fingerprint.json gpu_driver_version)
set -l new_driver (json_get $NEW/fingerprint.json gpu_driver_version)
set -l old_wmma (json_get $OLD/fingerprint.json wmma_intrinsic_available)
set -l new_wmma (json_get $NEW/fingerprint.json wmma_intrinsic_available)

report (cmp_field "ROCm Version" $old_rocm $new_rocm)
report (cmp_field "hipcc" $old_hipcc_v $new_hipcc_v)
report (cmp_field "amdclang" $old_amdclang_v $new_amdclang_v)
report (cmp_field "hipBLAS pkg" $old_hipblas $new_hipblas)
report (cmp_field "HIP runtime pkg" $old_hip_rt $new_hip_rt)
report (cmp_field "rocm-core pkg" $old_rocm_core $new_rocm_core)
report (cmp_field "GPU Arch" $old_arch $new_arch)
report (cmp_field "Driver" $old_driver $new_driver)
report (cmp_field "WMMA Intrinsic" $old_wmma $new_wmma)

# Build OK detection: tail of build_verbose.log must contain cargo's "Finished"
# marker. Warnings are fine; rustc errors (error[EXXXX]: / error:) are not.
function build_ok --argument-names log
    if not test -f "$log"
        echo "❌"; return
    end
    if tail -5 $log | grep -q 'Finished `release` profile'
        echo "✅"
    else
        echo "❌"
    end
end
set -l old_build (build_ok "$OLD/build_verbose.log")
set -l new_build (build_ok "$NEW/build_verbose.log")
if test "$old_build" = "❌" -o "$new_build" = "❌"
    set build_ok_overall 0
end
report (cmp_field "Compilation" "$old_build OK" "$new_build OK")
report '```'
report ""

# ======================================================================
# [CORRECTNESS] — answer comparison
# ======================================================================
report "## [CORRECTNESS]"
report ""
report '```'

set -l identical 0
set -l diverged 0
set -l missing 0
set -l div_details

function tokenize_file --argument-names file
    # Whitespace-insensitive tokenisation: collapse any run of whitespace
    # (including newlines) into a single separator and emit one word per line.
    tr -s '[:space:]' '\n' < $file | grep -v '^$'
end

function compare_answers --argument-names old_file new_file
    set -l name (basename $old_file .txt)
    set -l stripped (string replace -r '^rocmforge_' '' $name)
    if not test -f "$old_file"
        echo "  $stripped:  ❌ missing in baseline"
        return 2
    end
    if not test -f "$new_file"
        echo "  $stripped:  ❌ missing in new run"
        return 2
    end
    set -l old_words (tokenize_file $old_file)
    set -l new_words (tokenize_file $new_file)
    set -l oc (count $old_words)
    set -l nc (count $new_words)
    set -l minc $oc
    if test $nc -lt $minc; set minc $nc; end

    set -l diverge 0
    for i in (seq 1 $minc)
        if test "$old_words[$i]" != "$new_words[$i]"
            set diverge $i
            break
        end
    end

    if test $diverge -eq 0 -a $oc -eq $nc
        echo "  $stripped:  ✅ identical ($oc words)"
        return 0
    else if test $diverge -eq 0
        echo "  $stripped:  ⚠️ length differs ($oc → $nc words)"
        return 1
    else
        echo "  $stripped:  ⚠️ diverges at word $diverge ($oc vs $nc words)"
        return 1
    end
end

for f in $OLD/answers/rocmforge_*.txt
    set -l name (basename $f)
    set -l nf "$NEW/answers/$name"
    set -l line (compare_answers $f $nf)
    set -l code $status
    report $line
    switch $code
        case 0
            set identical (math $identical + 1)
        case 1
            set diverged (math $diverged + 1)
            set -a div_details (basename $f .txt)
        case 2
            set missing (math $missing + 1)
    end
end

set -l total (math $identical + $diverged + $missing)
report ""
if test $diverged -eq 0 -a $missing -eq 0
    report "  Verdict: $identical/$total identical"
else
    set -l detail_str ""
    if test $diverged -gt 0
        set detail_str ", $diverged drifted"
    end
    if test $missing -gt 0
        set detail_str "$detail_str, $missing missing"
    end
    report "  Verdict: $identical/$total identical$detail_str"
end
report '```'
report ""

# ======================================================================
# [PERFORMANCE] — throughput + latency deltas with tolerance bands
# ======================================================================
report "## [PERFORMANCE]"
report ""
report '```'

set -g perf_ok_overall 1
set -g perf_warn 0
set -g perf_bad 0

function perf_row --argument-names label old new unit
    set -l delta (fmt_delta $old $new $unit)
    set -l pct (pct_delta $old $new)
    set -l rating (rate_delta $pct)
    switch $rating
        case "⚠️"
            set -g perf_warn (math $perf_warn + 1)
        case "❌"
            set -g perf_bad (math $perf_bad + 1)
            set -g perf_ok_overall 0
    end
    printf '    %-22s %s\n' "$label:" "$delta"
end

report "  Synthetic Benchmark:"
report (perf_row "Prefill pp64" \
    (json_get $OLD/summary.json synthetic.prefill_pp64_tps_median) \
    (json_get $NEW/summary.json synthetic.prefill_pp64_tps_median) "tok/s")
report (perf_row "Prefill pp128" \
    (json_get $OLD/summary.json synthetic.prefill_pp128_tps_median) \
    (json_get $NEW/summary.json synthetic.prefill_pp128_tps_median) "tok/s")
report (perf_row "Prefill pp256" \
    (json_get $OLD/summary.json synthetic.prefill_pp256_tps_median) \
    (json_get $NEW/summary.json synthetic.prefill_pp256_tps_median) "tok/s")
report (perf_row "Prefill pp512" \
    (json_get $OLD/summary.json synthetic.prefill_pp512_tps_median) \
    (json_get $NEW/summary.json synthetic.prefill_pp512_tps_median) "tok/s")
report (perf_row "Decode 128tok" \
    (json_get $OLD/summary.json synthetic.decode_128_tps_median) \
    (json_get $NEW/summary.json synthetic.decode_128_tps_median) "tok/s")
report ""
report "  TTFT Stability (trace):"
report (perf_row "Layer-0 Warmup" \
    (json_get $OLD/summary.json layer_timing.layer0_us) \
    (json_get $NEW/summary.json layer_timing.layer0_us) "µs")
report (perf_row "Layer 1..N Median" \
    (json_get $OLD/summary.json layer_timing.layer1_to_n_median_us) \
    (json_get $NEW/summary.json layer_timing.layer1_to_n_median_us) "µs")
report (perf_row "Ratio L0/L1" \
    (json_get $OLD/summary.json layer_timing.layer0_over_median_ratio) \
    (json_get $NEW/summary.json layer_timing.layer0_over_median_ratio) "x")
report ""
report "  15-Prompt (Median):"
report (perf_row "TTFT" \
    (json_get $OLD/summary.json real_prompts.ttft_ms_median) \
    (json_get $NEW/summary.json real_prompts.ttft_ms_median) "ms")
report (perf_row "Decode" \
    (json_get $OLD/summary.json real_prompts.decode_tps_median) \
    (json_get $NEW/summary.json real_prompts.decode_tps_median) "tok/s")
report '```'
report ""

# ======================================================================
# [VERDICT]
# ======================================================================
report "## [VERDICT]"
report ""
report '```'

# Build verdict
set -l build_verdict "✅ Both versions compile successfully"
if test $build_ok_overall -eq 0
    set build_verdict "❌ New version fails to compile (see build_verbose.log)"
end

# Correctness verdict
set -l correctness_verdict
if test $diverged -eq 0 -a $missing -eq 0
    set correctness_verdict "✅ $identical/$total answers identical"
else if test $missing -gt 0
    set correctness_verdict "❌ $missing/$total answer files missing"
else if test $diverged -le 2
    set correctness_verdict "⚠️ $identical/$total identical, $diverged minor drift"
else
    set correctness_verdict "❌ $identical/$total identical, $diverged drifted"
end

# Performance verdict
set -l perf_verdict
if test $perf_bad -gt 0
    set perf_verdict "❌ $perf_bad metrics outside ±10% tolerance"
else if test $perf_warn -gt 0
    set perf_verdict "⚠️ $perf_warn metrics in ±5..10% band (noise)"
else
    set perf_verdict "✅ All metrics within ±5% tolerance"
end

printf '  %-14s %s\n' "Build:" "$build_verdict" | tee -a $DIFF_FILE
printf '  %-14s %s\n' "Correctness:" "$correctness_verdict" | tee -a $DIFF_FILE
printf '  %-14s %s\n' "Performance:" "$perf_verdict" | tee -a $DIFF_FILE
report ""

if test $build_ok_overall -eq 0
    report "  ➜ DO NOT USE ROCm $NEW_VERSION — build broken. Rollback recommended."
else if test $perf_bad -gt 0 -o $missing -gt 0
    report "  ➜ Investigate before using ROCm $NEW_VERSION — see details above."
else if test $perf_warn -gt 0 -o $diverged -gt 0
    report "  ➜ ROCm $NEW_VERSION usable, but review drift/warnings above."
else
    report "  ➜ ROCm $NEW_VERSION is safe to use. No regressions detected."
end

report '```'
report ""
echo ""
echo "Report saved to: $DIFF_FILE"
