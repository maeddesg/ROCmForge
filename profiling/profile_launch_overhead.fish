#!/usr/bin/env fish
# profile_launch_overhead.fish — Kernel-counting and launch overhead analysis
#
# Method 1: rocprofv3 kernel tracing (SQLite DB output)
# Method 2: HIP Event sub-phase timing (via ROCMFORGE_PROFILE_VERIFY_BREAKDOWN)
#
# Usage: fish profiling/profile_launch_overhead.fish

set -l BINARY target/release/rocmforge
set -l TARGET_MODEL (test -n "$ROCMFORGE_BENCH_TARGET_MODEL"; and echo $ROCMFORGE_BENCH_TARGET_MODEL; or echo "$HOME/models/Qwen2.5-7B-Instruct-Q4_0.gguf")
set -l DRAFT_MODEL (test -n "$ROCMFORGE_BENCH_DRAFT_MODEL"; and echo $ROCMFORGE_BENCH_DRAFT_MODEL; or echo "$HOME/models/qwen2.5-0.5b-instruct-q4_0.gguf")
set -l MAX_TOKENS 64
set -l GIT_SHA (git rev-parse --short HEAD 2>/dev/null; or echo "unknown")
set -l TIMESTAMP (date +%s)
set -l OUTPUT_DIR profiling/results
set -l PROMPTS code_01 chat_01 prose_03

if not test -f $BINARY
    echo "Error: $BINARY not found. Run: cargo build --release --features gpu"
    exit 1
end

mkdir -p $OUTPUT_DIR

echo "=== Launch Overhead Profiling ==="
echo "Git SHA: $GIT_SHA"
echo "Target: $TARGET_MODEL"
echo "Draft: $DRAFT_MODEL"
echo "Max tokens: $MAX_TOKENS"
echo ""

for prompt_name in $PROMPTS
    set -l prompt_file "benches/prompts/$prompt_name.txt"
    if not test -f $prompt_file
        echo "Warning: $prompt_file not found, skipping"
        continue
    end
    set -l prompt (cat $prompt_file)
    set -l output_file "$OUTPUT_DIR/launch_overhead_{$GIT_SHA}_{$TIMESTAMP}_{$prompt_name}.json"

    echo "--- $prompt_name ---"

    # Method 2: HIP Event sub-phase breakdown
    echo "  [Method 2] HIP Event sub-phase timing..."
    set -l m2_stderr (env ROCMFORGE_PROFILE_SPEC_STEP=1 ROCMFORGE_PROFILE_VERIFY_BREAKDOWN=1 \
        $BINARY --model $TARGET_MODEL --draft-model $DRAFT_MODEL \
        --prompt "$prompt" --max-tokens $MAX_TOKENS --spec-depth 1 --gpu 2>&1 >/dev/null)

    set -l m2_spec_json (echo "$m2_stderr" | grep "SPEC_PROFILE_JSON" | sed 's/.*\] //')
    set -l m2_breakdown_json (echo "$m2_stderr" | grep "VERIFY_BREAKDOWN_JSON" | sed 's/.*\] //')

    # Method 1: rocprofv3 kernel tracing
    echo "  [Method 1] rocprofv3 kernel tracing..."
    set -l trace_dir (mktemp -d /tmp/rocprof_trace_XXXXXX)

    env ROCMFORGE_PROFILE_SPEC_STEP=1 \
        rocprofv3 --kernel-trace -d $trace_dir -- \
        $BINARY --model $TARGET_MODEL --draft-model $DRAFT_MODEL \
        --prompt "$prompt" --max-tokens $MAX_TOKENS --spec-depth 1 --gpu \
        >/dev/null 2>/dev/null

    # Find the results.db file (rocprofv3 outputs SQLite DB)
    set -l trace_db (find $trace_dir -name "*.db" -type f | head -1)
    set -l m1_json "{}"

    if test -n "$trace_db" -a -f "$trace_db"
        echo "  Parsing trace: $trace_db"

        set m1_json (python3 -c "
import sqlite3, json, sys

db = sqlite3.connect('$trace_db')
db.row_factory = sqlite3.Row

groups = {
    'attention': ['flash_attn', 'attention_verify'],
    'projections_qkv': ['gemv_qkv', 'qkv_q4_0'],
    'projections_ffn': ['gemv_gate_up_swiglu', 'gate_up_swiglu'],
    'projections_down': ['gemv_q4_0_f32_q8_inline_residual', 'gemv_q4_0_f32_batched_tiled'],
    'projections_other': ['gemm_q4_0', 'gemm_q4_1', 'gemv_q6_k', 'gemv_q4_1'],
    'normalization': ['rms_norm'],
    'rope': ['rope'],
    'kv_cache': ['kv_write'],
    'activation': ['silu_kernel'],
    'elementwise': ['add_kernel', 'add_batched', 'mul_kernel'],
    'sampling': ['argmax', 'softmax'],
    'copy': ['copyBuffer', 'copyImage'],
    'embedding': ['embed'],
    'quantize': ['quantize'],
}

results = {}
total_launches = 0
total_gpu_ns = 0
kernel_list = []

try:
    rows = db.execute('SELECT name, duration FROM kernels').fetchall()
except:
    rows = []

for row in rows:
    name = row['name'] or ''
    dur_ns = row['duration'] or 0

    name_lower = name.lower()
    group = 'other'
    for g, keywords in groups.items():
        if any(kw in name_lower for kw in keywords):
            group = g
            break

    if group not in results:
        results[group] = {'n_launches': 0, 'total_ns': 0, 'kernels': {}}
    results[group]['n_launches'] += 1
    results[group]['total_ns'] += dur_ns

    short_name = name.split('(')[0][:80]
    results[group]['kernels'][short_name] = results[group]['kernels'].get(short_name, 0) + 1

    total_launches += 1
    total_gpu_ns += dur_ns

for g in results:
    n = results[g]['n_launches']
    results[g]['total_us'] = round(results[g]['total_ns'] / 1000.0, 1)
    results[g]['avg_us_per_launch'] = round(results[g]['total_ns'] / 1000.0 / n, 1) if n > 0 else 0
    del results[g]['total_ns']

output = {
    'kernel_groups': {g: {
        'n_launches': results[g]['n_launches'],
        'total_us': results[g]['total_us'],
        'avg_us_per_launch': results[g]['avg_us_per_launch'],
        'kernels': results[g]['kernels'],
    } for g in sorted(results.keys())},
    'total_launches': total_launches,
    'total_gpu_time_us': round(total_gpu_ns / 1000.0, 1),
}
print(json.dumps(output))
db.close()
" 2>/dev/null; or echo "{}")
        echo "  Parsed $prompt_name"
    else
        echo "  Warning: no trace DB found in $trace_dir"
    end

    rm -rf $trace_dir

    # Combine into output JSON
    python3 -c "
import json

m1 = json.loads('''$m1_json''')
m2_spec = json.loads('''$m2_spec_json''') if '''$m2_spec_json''' else {}
m2_breakdown = json.loads('''$m2_breakdown_json''') if '''$m2_breakdown_json''' else {}

output = {
    'git_sha': '$GIT_SHA',
    'timestamp': $TIMESTAMP,
    'config': {
        'spec_depth': 1,
        'tiled_gemv': True,
        'model': 'Qwen2.5-7B-Q4_0',
        'max_tokens': $MAX_TOKENS,
    },
    'prompt_id': '$prompt_name',
    'method_1_rocprofv3': m1,
    'method_2_hip_events': {
        'spec_step': m2_spec,
        'verify_breakdown': m2_breakdown,
    },
}
print(json.dumps(output, indent=2))
" > $output_file

    echo "  Written: $output_file"
    echo ""
end

echo "Done."
