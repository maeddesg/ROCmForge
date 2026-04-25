# HIP Graph: stale device-pointer reads in large graphs on gfx1201

I'm hitting an issue where kernels inside a captured HIP graph read stale values from device pointers during replay. The values behind the pointers are updated via `hipMemcpyAsync` on the same stream before `hipGraphLaunch`, but the kernels still see whatever was there at capture time.

The confusing part: a simple 1-kernel test case works fine. The problem only shows up once the graph gets large (~200 kernel nodes). I've spent a few days narrowing this down and I'm fairly confident it's not a user error at this point, but I can't rule it out completely without more eyes on it.

## Setup

- ROCm 7.2.1, HIP 7.2.53211-9999, AMD clang 22.0.0git
- RX 9070 XT (gfx1201 / RDNA4)
- amdgpu driver, kernel 7.0.0-1-cachyos, CachyOS (Arch-based)
- Haven't been able to test on gfx1100 or older yet

## What I'm doing

I'm working on a small LLM inference engine ([ROCmForge](https://github.com/maeddesg/ROCmForge)) and I capture the full autoregressive decode step as a HIP graph. That's 28 transformer layers with ~7 kernels each (RMSNorm, GEMV, RoPE, attention, elementwise, etc.), plus lm_head and argmax at the end — roughly 200 kernel nodes total.

Two values change between replays: `pos` (current token position) and `seq_len` (how many KV cache entries to attend over). Since kernel args are frozen at capture time, I pass these as device pointers and update the pointed-to memory before each replay. Standard pattern, works the same way in CUDA.

Simplified kernel:

```cpp
__global__ void attn_decode_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const half*  __restrict__ k_cache,
    const half*  __restrict__ v_cache,
    const int*   __restrict__ pos_ptr,
    const int*   __restrict__ seq_len_ptr,
    int num_heads, int num_kv_heads, int head_dim, float scale
) {
    const int pos = *pos_ptr;
    const int seq_len = *seq_len_ptr;
    // ... attention over seq_len positions ...
}
```

Host side:

```cpp
// Capture with pos=0, seq_len=1
int h_state[2] = {0, 1};
hipMemcpyAsync(d_state, h_state, 8, hipMemcpyHostToDevice, stream);
hipStreamSynchronize(stream);

hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
for (int layer = 0; layer < 28; layer++) {
    rms_norm_kernel<<<...>>>(d_hidden, d_norm_weight, ...);
    qkv_gemv_kernel<<<...>>>(d_qkv, d_hidden, d_qkv_weight, ...);
    rope_kernel<<<...>>>(..., &d_state[0], ...);
    attn_decode_kernel<<<...>>>(..., &d_state[0], &d_state[1], ...);
    ffn_kernels<<<...>>>(...);
}
lm_head_gemv<<<...>>>(...);
argmax_kernel<<<...>>>(...);
hipStreamEndCapture(stream, &graph);
hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

// Later: update to pos=15, seq_len=16
h_state[0] = 15;  h_state[1] = 16;
hipMemcpyAsync(d_state, h_state, 8, hipMemcpyHostToDevice, stream);
hipGraphLaunch(exec, stream);
hipStreamSynchronize(stream);
// Kernel output is wrong — looks like it used seq_len=1, not 16
```

## What I tried to isolate it

I tested four configurations, same kernels, same data:

| Config | Result |
|--------|--------|
| Direct launch, seq_len as immediate kernel arg | correct |
| Direct launch, seq_len via device pointer | correct |
| Graph replay (~200 nodes), seq_len via device pointer | **wrong** |
| Graph replay (only 2 nodes: lm_head + argmax), layers direct | correct |

The third row is the problem. Same kernel, same pointer, same stream — just wrapped in a bigger graph.

I also added a readback right before `hipGraphLaunch` to confirm the memcpy landed:

```
pos=9: uploaded state=[9, 10]  -- device memory is correct
hidden before graph: [-0.023, 0.007, ...]  -- input looks fine  
hidden after graph: [1.61, 0.08, ...]  -- output is wrong (not NaN, just wrong values)
argmax token=52643  -- garbage
```

So the data is there on the device, the input to the graph is valid, but the output is wrong in a way that's consistent with the attention kernel using seq_len=1 instead of 10.

## The confusing part: simple test passes

I wrote a minimal standalone test — single kernel that reads an int from a device pointer, graph capture, update value, replay:

```
$ hipcc -o graph_ptr_test graph_ptr_test.hip && ./graph_ptr_test
Test 1 (original value 42): Got 42 PASS
Test 2 (updated to 99): Got 99 PASS
Test 3 (10 sequential replays): PASS
```

All fine. So it's not a blanket "device pointer reads are inlined" issue. Something about the graph size or topology triggers it.

I haven't managed to build a self-contained reproducer that fails yet — the bug seems tied to having many kernel nodes. If it helps I can provide the full project to reproduce against a GGUF model file.

## My guesses (probably wrong, but for context)

- Some graph-level optimization inlines or caches device memory reads when node count is high?
- Memory coherence issue when many kernels read from the same small allocation?
- Kernel reordering within the graph that races with the pre-launch memcpy?
- Something RDNA4-specific in the L2 / memory controller path?

No idea really. Could also be something dumb on my end that I'm not seeing.

## Workaround

I run the layer kernels directly (they're the ones that read the device pointers) and only capture the tail (lm_head + argmax) as a graph. Works correctly. Performance hit is noticeable on small models (~650 → 220 tok/s for 0.5B) but negligible on 7B since those layers are bandwidth-bound anyway.

## Standalone test (passes — included for reference)

```cpp
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void read_device_ptr_kernel(
    int* __restrict__ d_output,
    const int* __restrict__ d_value_ptr
) {
    if (threadIdx.x == 0) {
        d_output[0] = *d_value_ptr;
    }
}

int main() {
    int* d_value;
    int* d_output;
    hipMalloc(&d_value, sizeof(int));
    hipMalloc(&d_output, sizeof(int));

    hipStream_t stream;
    hipStreamCreate(&stream);

    int h_value = 42;
    hipMemcpyAsync(d_value, &h_value, sizeof(int), hipMemcpyHostToDevice, stream);
    hipStreamSynchronize(stream);

    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    read_device_ptr_kernel<<<1, 1, 0, stream>>>(d_output, d_value);
    hipGraph_t graph;
    hipStreamEndCapture(stream, &graph);

    hipGraphExec_t exec;
    hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    hipGraphLaunch(exec, stream);
    hipStreamSynchronize(stream);
    int h_output = 0;
    hipMemcpy(&h_output, d_output, sizeof(int), hipMemcpyDeviceToHost);
    printf("Test 1 (original value 42): Got %d %s\n",
           h_output, h_output == 42 ? "PASS" : "FAIL");

    h_value = 99;
    hipMemcpyAsync(d_value, &h_value, sizeof(int), hipMemcpyHostToDevice, stream);
    hipGraphLaunch(exec, stream);
    hipStreamSynchronize(stream);
    hipMemcpy(&h_output, d_output, sizeof(int), hipMemcpyDeviceToHost);
    printf("Test 2 (updated to 99): Got %d %s\n",
           h_output, h_output == 99 ? "PASS" : "FAIL");

    int all_pass = 1;
    for (int i = 1000; i < 1010; i++) {
        hipMemcpyAsync(d_value, &i, sizeof(int), hipMemcpyHostToDevice, stream);
        hipGraphLaunch(exec, stream);
        hipStreamSynchronize(stream);
        hipMemcpy(&h_output, d_output, sizeof(int), hipMemcpyDeviceToHost);
        if (h_output != i) {
            printf("Test 3 (expected %d): Got %d FAIL\n", i, h_output);
            all_pass = 0;
        }
    }
    if (all_pass) printf("Test 3 (10 sequential replays): PASS\n");

    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);
    hipFree(d_value);
    hipFree(d_output);
    return 0;
}
```

Happy to provide more info or test things if that helps track this down.
