# Standalone WMMA proof-of-concept build.
#
# Builds a single binary that drives the 16×16 kernel and compares
# against a CPU reference.

HIPCC    ?= hipcc
ARCH     ?= gfx1201
CXXFLAGS := -O3 -std=c++17 --offload-arch=$(ARCH) -Wno-unused-result

test_wmma_16x16: test_wmma_16x16.hip ../../hip_kernels/wmma/wmma_gemm_16x16.hip
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f test_wmma_16x16

.PHONY: clean
