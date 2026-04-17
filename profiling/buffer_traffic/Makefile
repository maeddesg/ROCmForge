# Intermediate-Buffer-Traffic Micro-Benchmark
#
# Build: make
# Run:   ./bench (or ./run.fish)

HIPCC    ?= hipcc
ARCH     ?= gfx1201
CXXFLAGS := -O3 -std=c++17 --offload-arch=$(ARCH) -Wno-unused-result

bench: bench.hip
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f bench

.PHONY: clean
