#!/usr/bin/env fish
#
# Build + run the intermediate-buffer-traffic micro-benchmark and drop a JSON
# artifact into profiling/results/.

set -l here (dirname (status filename))
cd $here

if not test -x ./bench
    echo "building bench..."
    make bench
    if test $status -ne 0
        echo "build failed" >&2
        exit 1
    end
end

set -l sha (git -C $here rev-parse --short HEAD 2>/dev/null)
if test -z "$sha"
    set sha "dev"
end
set -l ts (date +%s)
set -l out "$here/../results/buffer_traffic_validation_{$sha}_{$ts}.json"

echo "Running benchmark..."
./bench --warmup 10 --iters 100 > $out
if test $status -ne 0
    echo "bench failed" >&2
    exit 1
end

echo ""
echo "JSON: $out"
