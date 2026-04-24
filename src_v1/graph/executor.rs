//! Graph executor (Phase 1).
//!
//! Minimal forward-pass runtime. Each `GraphNode` dispatches exactly
//! one kernel (the kernels from Schritte 1.7–1.9); Phase-1 restrictions:
//!
//!   * Decode path only — every call processes one token. Prefill is
//!     a `seq_len`-step loop over the decode path. WMMA-based prefill
//!     batching is Phase-2.
//!   * Q4_K format expected on FFN gate+up for the fused Q4_K kernel.
//!     Other formats fall back to split `gate-GEMV + up-GEMV + SwiGLU`.
//!   * Biases are **ignored** for now — Qwen2.5 attention biases would
//!     need a separate add pass that Phase 1 doesn't implement.
//!   * KV cache is FP32, dedicated allocation (not arena Zone B) so we
//!     sidestep an Arena-layout change for Block B.
//!
//! Safeguards per the Block-B scope:
//!   1. `ROCMFORGE_TRACE=1`        → log each node's kind + output
//!                                    buffer magnitude.
//!   2. `ROCMFORGE_NAN_GUARD=1`    → `is_finite` check after every
//!                                    node; fail loudly with the
//!                                    offending kind.
//!   3. Logits are always sanity-checked before return (NaN/Inf there
//!      is an immediate error).

use std::collections::HashMap;
use std::ffi::c_void;

use super::super::backend::gpu::attention::rocmforge_launch_attention_prefill;
use super::super::backend::gpu::attention::{
    rocmforge_launch_attention_decode, rocmforge_launch_attention_decode_fp8,
    rocmforge_launch_kv_cache_append, rocmforge_launch_kv_cache_append_fp8,
};
use super::super::backend::gpu::elementwise::rocmforge_launch_rope_batched;
use super::super::backend::gpu::elementwise::{
    rocmforge_launch_embedding_lookup, rocmforge_launch_residual_add_inplace,
    rocmforge_launch_rms_norm, rocmforge_launch_rms_norm_batched, rocmforge_launch_rope,
    rocmforge_launch_swiglu,
};
use super::super::backend::gpu::error::{check, HipError, HipResult};
use super::super::backend::gpu::gemv::{
    rocmforge_launch_gemv_q4_0_standard, rocmforge_launch_gemv_q4_k_gate_up_swiglu,
    rocmforge_launch_gemv_q4_k_mmvq, rocmforge_launch_gemv_q4_k_mmvq_fused,
    rocmforge_launch_gemv_q4_k_mmvq_residual, rocmforge_launch_gemv_q4_k_q8_inline,
    rocmforge_launch_gemv_q4_k_q8_inline_residual, rocmforge_launch_gemv_q4_k_q8_inline_sudot4,
    rocmforge_launch_gemv_q4_k_standard, rocmforge_launch_gemv_q6_k_mmvq,
    rocmforge_launch_gemv_q6_k_q8_inline, rocmforge_launch_gemv_q6_k_standard,
    rocmforge_launch_gemv_q8_0_standard,
};
use super::super::backend::gpu::hip_ffi::{
    hipGraphDestroy, hipGraphExecDestroy, hipGraphExecKernelNodeSetParams, hipGraphExec_t,
    hipGraphGetNodes, hipGraphInstantiate, hipGraphKernelNodeGetParams, hipGraphLaunch,
    hipGraphNode_t, hipGraph_t, hipKernelNodeParams, hipMemcpy, hipMemcpyDeviceToDevice,
    hipMemcpyDeviceToHost, hipStreamBeginCapture, hipStreamCaptureModeGlobal, hipStreamEndCapture,
    HIP_SUCCESS,
};
use super::super::backend::gpu::quantize::{rocmforge_launch_quantize_q8_1, BlockQ81, QK8_1};
use super::super::backend::gpu::wmma::{
    rocmforge_launch_mmq_q4_k, rocmforge_launch_quantize_q8_1_mmq,
    rocmforge_launch_wmma_gemm_q4_0_fp16, rocmforge_launch_wmma_gemm_q4_0_fp8,
    rocmforge_launch_wmma_gemm_q4_k_fp16, rocmforge_launch_wmma_gemm_q4_k_fp8,
    rocmforge_launch_wmma_gemm_q6_k_fp16, rocmforge_launch_wmma_gemm_q6_k_fp8,
    rocmforge_launch_wmma_gemm_q8_0_fp16, rocmforge_launch_wmma_gemm_q8_0_fp8,
};
use super::super::backend::gpu::wrappers::{HipBuffer, HipStream};
use super::super::core::gguf::GGUFFile;
use super::super::core::model_loader::LoadedModel;
use super::super::core::tensor_info::{GgmlType, TensorRole};
use super::super::ga::dynamic::DynamicKernel;
use super::super::ir::formats::{q4_0, q4_k, q6_k, q8_0};
use super::super::ir::interpreter::dequant_block;
use super::super::runtime::{KernelId, OpType, Runtime, ShapeKey, VariantId};
use super::buffer_plan::{BufferPlan, KvCacheLayout, KvPrecision};
use super::nodes::{BufferId, GraphNode, WeightRef};
use super::ComputationGraph;
use std::sync::Arc;

/// Executor state: owns the graph, a reference to the loaded model
/// (weights + arena), one device buffer per BufferId, one FP32 K and V
/// cache per transformer layer, and a stream.
pub struct GraphExecutor<'m> {
    graph: ComputationGraph,
    plan: BufferPlan,
    model: &'m LoadedModel,

    // Transient per-run buffers, keyed by `BufferId`.
    buffers: HashMap<BufferId, HipBuffer>,

    // KV cache: one (K, V) pair per layer.
    kv_layout: KvCacheLayout,
    k_cache: Vec<HipBuffer>, // len == n_layers
    v_cache: Vec<HipBuffer>,

    /// FP32-dequantised copy of the embedding table. Phase-1 workaround
    /// for the "FP32-Embedding in Zone A" budget note — the GGUF stores
    /// embeddings quantised (Q4_K / Q6_K / Q4_0), but our embedding
    /// kernel expects FP32. We dequant once at init and use this buffer
    /// during the Embedding node dispatch.
    embedding_fp32: HipBuffer,

    stream: HipStream,
    trace: bool,
    nan_guard: bool,

    /// Optional Self-Tuning Runtime. When present, GEMV dispatches
    /// route through `runtime.select_variant()`, run the chosen
    /// kernel, sync the stream, and feed the elapsed time back into
    /// the Bandit. When absent (default), dispatches go straight to
    /// the fixed `_standard` kernel per quant format — byte-identical
    /// to the pre-1.12 path.
    runtime: Option<Runtime>,

    /// Event pool paired with `runtime`. During Bandit-exploration
    /// (Phase-2 §8.1), each tuned GEMV records `(start, stop)` events
    /// on the stream instead of calling `stream.synchronize()`. The
    /// pool is drained once at token end — one sync per token rather
    /// than one per kernel (Arch-Doc §3.7 Zero-Sync Pipeline). After
    /// Bandit-convergence (`runtime.all_exploiting() == true`) we
    /// skip event recording entirely.
    event_pool: Option<super::super::runtime::EventPool>,

    /// GA-tuned `gate_up_swiglu` kernel (Phase 2 step 2.1.3 Block D).
    /// When present and the node shape matches the hook's expected
    /// `(ffn_dim, hidden_dim)`, `dispatch_gate_up_swiglu` uses this
    /// dynamic kernel instead of the static
    /// `rocmforge_launch_gemv_q4_k_gate_up_swiglu` launcher.
    /// The static launcher remains the fallback — every FFN node
    /// whose shape doesn't match still takes the Phase-1 path.
    gate_up_dynamic: Option<DynamicGateUpHook>,

    /// Prefill-path WMMA precision (Phase 2 step 2.1.5 FP8 follow-up).
    /// Default `Fp16`; `Fp8` enables the FP8-pair-packed kernels
    /// from Block A. Read by `dispatch_prefill_wmma_gemm`. The env
    /// var `ROCMFORGE_PREFILL_FP8=1` flips the initial default.
    prefill_precision: PrefillPrecision,

    /// Decode-path scratch for the un-fused gate_up path (post-2.1.5
    /// follow-up). Two FP32 `[ffn_dim]` buffers sized on first use.
    /// The fused `gemv_q4_k_gate_up_swiglu` kernel hits only 20 %
    /// BW (rocprof deep-dive 2026-04-23); splitting into 2× Q4_K
    /// GEMV + SwiGLU lifts the BW to the 442 GB/s that the
    /// `q4_k_q8_inline_residual` kernel already shows on the same
    /// hardware, at the cost of 2 × `ffn_dim × 4 B` VRAM scratch.
    /// Opt-back to the fused path via `ROCMFORGE_FUSED_GATE_UP=1`.
    gate_scratch: Option<HipBuffer>,
    up_scratch: Option<HipBuffer>,
    /// Runtime toggle — mirrors the `ROCMFORGE_FUSED_GATE_UP` env
    /// var at construction; tests can flip it via
    /// `set_fused_gate_up`.
    fused_gate_up: bool,

    /// Flat `BufferId → device pointer` lookup for the decode hot
    /// path. Populated once at the end of `new()`; HipBuffer device
    /// pointers don't move for the lifetime of the buffer (hipMalloc
    /// returns a stable VA), so caching is safe. Indexed by
    /// `BufferId.0 as usize`. Entries for ids the plan doesn't
    /// mention stay `null`. Replaces the `self.buffers[&id]`
    /// HashMap lookup on every node dispatch.
    buffer_ptrs: Vec<*mut c_void>,

    /// Per-graph-node cache of resolved weight pointers and the
    /// Bandit-committed kernel choice. Populated lazily by
    /// `compile_fast_dispatch()` once `runtime.all_exploiting()`
    /// holds — i.e. once the Bandit has stopped flipping its
    /// selections. `None` → legacy path (HashMap lookups + Bandit
    /// query per call).
    ///
    /// Indexed by `node_idx`; length == `graph.nodes.len()` when
    /// populated. Invalidated (set to `None`) whenever a config
    /// field that would change the dispatched kernel changes —
    /// `set_gate_up_swiglu_dynamic_kernel`, `set_fused_gate_up`.
    node_fast_cache: Option<Vec<NodeFastEntry>>,

    /// Captured HIP-Graph for the decode hot path (post-Bandit-
    /// convergence). `None` → normal dispatch; `Some` → replay path.
    /// Invalidated (set to `None`) by the same triggers as
    /// `node_fast_cache`, plus on monitor-detected drift.
    hip_graph: Option<HipGraphDecodeCache>,

    /// Phase-2 Schritt 1/3 — llama.cpp MMVQ port pre-work. When enabled
    /// (via `ROCMFORGE_Q8_1_PREQUANT=1` or `set_q8_1_prequant(true)`),
    /// the executor pre-quantizes the per-token embedding row to Q8_1
    /// once per decode step, outside any HIP-Graph capture. The buffer
    /// is currently WRITE-ONLY — no kernel reads it. Consumer is the
    /// upcoming MMVQ kernel (Schritt 2/3). Lazy-allocated on first use.
    q8_1_buffer: Option<HipBuffer>,
    q8_1_prequant_enabled: bool,

    /// Phase 2 Schritt 5 — Q8_1 buffer sharing across consecutive MMVQ
    /// dispatches with the same input BufferId. Reset to `None` at the
    /// start of every decode pass and at every dispatch point that
    /// invalidates the currently-quantized buffer (FusedGemmResidual
    /// writes back in-place; non-MMVQ nodes implicitly reset by missing
    /// the cache). When `Some(id)` and the next MMVQ dispatch's input
    /// matches, the per-call `quantize_q8_1` is skipped and the
    /// existing `q8_1_buffer` contents are reused. Matches the graph's
    /// Q → K → V pattern (3 consecutive matmuls on the same RmsNorm
    /// output) — saves 2 quantize launches per layer × 36 layers =
    /// 72 calls per token.
    last_q8_1_input_id: Option<BufferId>,
    /// Env-controlled kill switch for the Q8_1-sharing optimisation.
    /// `ROCMFORGE_DISABLE_Q8_1_SHARING=1` forces a quantize on every
    /// MMVQ dispatch — used for A/B regression testing.
    q8_1_sharing_enabled: bool,

    // P0.2 Schritt 4 — Integer-WMMA MMQ prefill path.
    /// Pre-quantise buffer for activations on the MMQ path. Re-used
    /// across every Q4_K prefill GEMM in a forward pass. Lazy-allocated
    /// on first use at the observed `(M, K)`, grown if a later call
    /// sees a bigger shape. Only accessed when `prefill_mmq_enabled`.
    mmq_activation_buffer: Option<HipBuffer>,
    /// Env-gated opt-in (`ROCMFORGE_PREFILL_MMQ=1`). When true, the
    /// Q4_K prefill path quantises activations into `block_q8_1_mmq`
    /// and calls the integer-WMMA MMQ kernel; when false (default),
    /// the FP16-WMMA kernel runs as before.
    prefill_mmq_enabled: bool,
}

/// Per-node cache entry for the fast dispatch path. All pointers
/// resolved once; the hot-path dispatch just reads them directly.
#[derive(Debug, Clone, Copy)]
struct NodeFastEntry {
    /// Primary weight ptr (Gemm.weight, RmsNorm.weight,
    /// FusedGemmResidual.weight, GateUpSwiGLU.gate_weight).
    /// `null` when the node has no weight.
    weight_primary: *const c_void,
    /// Secondary weight ptr — GateUpSwiGLU.up_weight only.
    /// `null` otherwise.
    weight_secondary: *const c_void,
    /// RoPE-freqs pointer for Rope nodes (optional). `null` when
    /// using standard RoPE.
    rope_freqs: *const c_void,
    /// Bandit-committed kernel for Gemm / FusedGemmResidual nodes.
    /// `None` → fall back to the runtime's dynamic select_variant.
    committed_kernel: Option<KernelId>,
}

/// Precision of the WMMA kernel dispatched during batched prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillPrecision {
    /// Level-1: FP16 accumulate, FP16 tiles. Default.
    Fp16,
    /// Level-0: FP8 E4M3 pair-packed tiles. ~2× throughput target;
    /// numerical error grows vs FP16 but greedy top-1 typically
    /// matches because pair-packing preserves ordering on Q4_K
    /// super-blocks after the Block-A fix.
    Fp8,
}

impl PrefillPrecision {
    /// Read the default precision from the environment. Set once at
    /// executor construction; [`GraphExecutor::set_prefill_precision`]
    /// overrides it at runtime.
    pub fn from_env() -> Self {
        if std::env::var("ROCMFORGE_PREFILL_FP8").ok().as_deref() == Some("1") {
            Self::Fp8
        } else {
            Self::Fp16
        }
    }
}

/// One GA-compiled dynamic kernel bound to a specific
/// gate_up_swiglu shape. Held `Arc` so the same compiled module can
/// be shared across executors when a GA winner applies to multiple
/// models at runtime.
pub struct DynamicGateUpHook {
    /// GA-compiled kernel — unloaded when the `Arc` drops to zero.
    pub kernel: Arc<DynamicKernel>,
    /// Model's `hidden_dim` the kernel was tuned for (the K-dim of
    /// the GEMV). A node whose `hidden_dim` ≠ this is rejected —
    /// grid-X math bakes both dims into the launch geometry.
    pub hidden_dim: usize,
    /// Model's `ffn_dim` the kernel was tuned for (the N-dim).
    pub ffn_dim: usize,
}

impl<'m> GraphExecutor<'m> {
    /// Build an executor for the given model + graph. Allocates every
    /// transient buffer (sized for `max_seq = 1` — decode-only path)
    /// plus the full KV cache for `max_seq`.
    pub fn new(
        graph: ComputationGraph,
        plan: BufferPlan,
        model: &'m LoadedModel,
        gguf: &GGUFFile,
        max_seq: usize,
    ) -> HipResult<Self> {
        let stream = HipStream::new()?;
        // Phase 2.2A — KV-cache precision is env-selected. FP32 by
        // default (Phase 1 baseline); `ROCMFORGE_KV_FP8=1` switches to
        // bf8, allocating ¼ of the FP32 cache size and dispatching the
        // bf8-aware kv_cache_append + attention_decode kernel pair.
        let kv_precision = KvPrecision::from_env();
        let kv_layout = KvCacheLayout::from_config_with_precision(
            &graph.config, max_seq, kv_precision);

        // Transient buffers — sized for one token (decode).
        let mut buffers: HashMap<BufferId, HipBuffer> = HashMap::new();
        for (&id, spec) in &plan.specs {
            let bytes = spec.bytes(1).max(4); // never zero-size
            buffers.insert(id, HipBuffer::new(bytes)?);
        }

        // KV cache — one allocation per (layer, K or V). Buffer size
        // scales with the precision (4 bytes/element for FP32, 1 for bf8).
        let per_cache = kv_layout.bytes_per_side();
        let mut k_cache = Vec::with_capacity(graph.config.n_layers);
        let mut v_cache = Vec::with_capacity(graph.config.n_layers);
        for _ in 0..graph.config.n_layers {
            k_cache.push(HipBuffer::new(per_cache)?);
            v_cache.push(HipBuffer::new(per_cache)?);
        }

        // Dequantise the embedding table once, CPU-side. Phase-1 path:
        // the GGUF stores the table as Q4_K / Q6_K / Q4_0 / Q8_0; the
        // elementwise embedding kernel reads FP32. We re-read the raw
        // tensor bytes from the mmap, dequant block-by-block via the
        // IR interpreter, and upload a single FP32 buffer.
        let embed_tensor = gguf
            .tensors()
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or_else(|| HipError {
                code: -1,
                message: "GGUF missing token_embd.weight".into(),
                context: "executor init".into(),
            })?;
        let fmt = match embed_tensor.ggml_type {
            GgmlType::Q4_0 => q4_0(),
            GgmlType::Q4_K => q4_k(),
            GgmlType::Q6_K => q6_k(),
            GgmlType::Q8_0 => q8_0(),
            GgmlType::F32 => {
                // Rare — just upload as-is.
                let bytes = gguf.tensor_data_full(embed_tensor).map_err(|e| HipError {
                    code: -1,
                    message: format!("tensor_data_full: {e}"),
                    context: "embed".into(),
                })?;
                let mut buf = HipBuffer::new(bytes.len())?;
                buf.copy_from_host(bytes)?;
                let _ = (&graph, &plan); // silence unused-warning in this branch
                let trace = std::env::var("ROCMFORGE_TRACE").ok().as_deref() == Some("1");
                let nan_guard = std::env::var("ROCMFORGE_NAN_GUARD").ok().as_deref() == Some("1");
                let buffer_ptrs = build_buffer_ptrs_cache(&buffers);
                return Ok(Self {
                    graph,
                    plan,
                    model,
                    buffers,
                    kv_layout,
                    k_cache,
                    v_cache,
                    embedding_fp32: buf,
                    stream,
                    trace,
                    nan_guard,
                    runtime: None,
                    event_pool: None,
                    gate_up_dynamic: None,
                    prefill_precision: PrefillPrecision::from_env(),
                    gate_scratch: None,
                    up_scratch: None,
                    fused_gate_up: std::env::var("ROCMFORGE_FUSED_GATE_UP").ok().as_deref()
                        == Some("1"),
                    buffer_ptrs,
                    node_fast_cache: None,
                    hip_graph: None,
                    q8_1_buffer: None,
                    q8_1_prequant_enabled: std::env::var("ROCMFORGE_Q8_1_PREQUANT")
                        .ok()
                        .as_deref()
                        == Some("1"),
                    last_q8_1_input_id: None,
                    q8_1_sharing_enabled: std::env::var("ROCMFORGE_DISABLE_Q8_1_SHARING")
                        .ok()
                        .as_deref()
                        != Some("1"),
                    mmq_activation_buffer: None,
                    prefill_mmq_enabled: std::env::var("ROCMFORGE_PREFILL_MMQ")
                        .ok()
                        .as_deref()
                        == Some("1"),
                });
            }
            other => {
                return Err(HipError {
                    code: -1,
                    message: format!("embedding table format {other} not supported in Phase 1"),
                    context: "executor init".into(),
                })
            }
        };
        let vocab = graph.config.vocab_size;
        let hidden = graph.config.hidden_dim;
        let epb = fmt.elements_per_block;
        let bb = fmt.block_bytes;
        let blocks_per_row = hidden / epb;
        if hidden % epb != 0 {
            return Err(HipError {
                code: -1,
                message: format!(
                    "hidden_dim {} not divisible by embedding block size {}",
                    hidden, epb
                ),
                context: "executor init".into(),
            });
        }
        // Dequantise the whole embedding matrix: vocab rows × hidden cols.
        let mut fp32_table: Vec<f32> = Vec::with_capacity(vocab * hidden);
        for row in 0..vocab {
            for blk in 0..blocks_per_row {
                let offset = (row * blocks_per_row + blk) * bb;
                let block_bytes =
                    gguf.tensor_data(embed_tensor, offset + bb)
                        .map_err(|e| HipError {
                            code: -1,
                            message: format!("tensor_data: {e}"),
                            context: "embed block".into(),
                        })?;
                let block = &block_bytes[offset..offset + bb];
                let elems = dequant_block(&fmt, block).map_err(|e| HipError {
                    code: -1,
                    message: format!("dequant_block: {e}"),
                    context: "embed dequant".into(),
                })?;
                fp32_table.extend_from_slice(&elems);
            }
        }
        let mut embedding_fp32 = HipBuffer::new(fp32_table.len() * 4)?;
        let bytes = unsafe {
            std::slice::from_raw_parts(fp32_table.as_ptr() as *const u8, fp32_table.len() * 4)
        };
        embedding_fp32.copy_from_host(bytes)?;

        let trace = std::env::var("ROCMFORGE_TRACE").ok().as_deref() == Some("1");
        let nan_guard = std::env::var("ROCMFORGE_NAN_GUARD").ok().as_deref() == Some("1");
        let buffer_ptrs = build_buffer_ptrs_cache(&buffers);

        Ok(Self {
            graph,
            plan,
            model,
            buffers,
            kv_layout,
            k_cache,
            v_cache,
            embedding_fp32,
            stream,
            trace,
            nan_guard,
            runtime: None,
            event_pool: None,
            gate_up_dynamic: None,
            prefill_precision: PrefillPrecision::from_env(),
            gate_scratch: None,
            up_scratch: None,
            fused_gate_up: std::env::var("ROCMFORGE_FUSED_GATE_UP").ok().as_deref() == Some("1"),
            buffer_ptrs,
            node_fast_cache: None,
            hip_graph: None,
            q8_1_buffer: None,
            q8_1_prequant_enabled: std::env::var("ROCMFORGE_Q8_1_PREQUANT")
                .ok()
                .as_deref()
                == Some("1"),
            last_q8_1_input_id: None,
            q8_1_sharing_enabled: std::env::var("ROCMFORGE_DISABLE_Q8_1_SHARING")
                .ok()
                .as_deref()
                != Some("1"),
            mmq_activation_buffer: None,
            prefill_mmq_enabled: std::env::var("ROCMFORGE_PREFILL_MMQ")
                .ok()
                .as_deref()
                == Some("1"),
        })
    }

    /// Force the fused (pre-un-fusing) `gate_up_swiglu` kernel. Used
    /// by the A/B-regression tests; the default after the post-2.1.5
    /// follow-up is the un-fused path.
    pub fn set_fused_gate_up(&mut self, fused: bool) {
        self.fused_gate_up = fused;
        self.invalidate_fast_dispatch();
    }

    pub fn fused_gate_up(&self) -> bool {
        self.fused_gate_up
    }

    /// Phase-2 Schritt 1/3 — enable or disable the per-token Q8_1
    /// activation pre-quantizer. Off by default; the smoke test flips
    /// it on to verify the code path runs without regressing decode
    /// throughput. The buffer is WRITE-ONLY in this step (no consumer
    /// kernel yet), so toggling only affects wall time, not logits.
    pub fn set_q8_1_prequant(&mut self, enabled: bool) {
        self.q8_1_prequant_enabled = enabled;
    }

    pub fn q8_1_prequant_enabled(&self) -> bool {
        self.q8_1_prequant_enabled
    }

    /// Lazy-allocate / grow the Q8_1 activation scratch buffer so it
    /// can hold at least `n_elements` quantized values. The MMVQ
    /// dispatch arm calls this on every Q4_K GEMV — `n_elements` can
    /// be either `hidden_dim` (QKV / O / gate / up) or `ffn_dim` (down),
    /// so we size up to the largest seen so far.
    fn ensure_q8_1_buffer(&mut self, n_elements: usize) -> HipResult<()> {
        if n_elements % QK8_1 != 0 {
            return Err(HipError {
                code: -1,
                message: format!("n_elements {n_elements} not divisible by QK8_1={QK8_1}"),
                context: "ensure_q8_1_buffer".into(),
            });
        }
        let needed = (n_elements / QK8_1) * std::mem::size_of::<BlockQ81>();
        let grow = match &self.q8_1_buffer {
            Some(buf) => buf.size() < needed,
            None => true,
        };
        if grow {
            self.q8_1_buffer = Some(HipBuffer::new(needed)?);
        }
        Ok(())
    }

    /// Phase 2 Schritt 5 — ensure the Q8_1 buffer holds a current
    /// quantization of `input_id`'s contents. Returns the Q8_1 buffer
    /// pointer. If the last dispatch already quantized the same
    /// `input_id` (Q → K → V pattern), skips the quantize launch.
    ///
    /// Cache invariant: between two successful calls with the same
    /// `input_id`, no intervening dispatch may write to that buffer.
    /// Enforced by:
    ///   * Any non-MMVQ `dispatch_node` arm (RmsNorm, RoPE, ...) does
    ///     not call this helper, so the cache sits with the stale id
    ///     but will miss on the NEXT MMVQ dispatch (whose input will
    ///     be something else — the graph doesn't re-enter old buffers).
    ///   * `dispatch_fused_gemm_residual` explicitly invalidates the
    ///     cache after the kernel returns, because the residual buffer
    ///     it just wrote into could be the cached input of a later
    ///     MMVQ dispatch in a pathological graph layout.
    fn ensure_q8_1_quantized(
        &mut self,
        input_id: BufferId,
        in_ptr: *const c_void,
        in_dim: usize,
    ) -> HipResult<*mut c_void> {
        self.ensure_q8_1_buffer(in_dim)?;
        let q8_1_ptr = self
            .q8_1_buffer
            .as_mut()
            .expect("ensure_q8_1_buffer just ran")
            .as_mut_ptr();

        let skip = self.q8_1_sharing_enabled && self.last_q8_1_input_id == Some(input_id);
        if !skip {
            let rc = unsafe {
                rocmforge_launch_quantize_q8_1(
                    in_ptr as *const f32,
                    q8_1_ptr,
                    in_dim as i32,
                    self.stream.raw(),
                )
            };
            check(rc, "quantize_q8_1 (ensure_q8_1_quantized)")?;
            self.last_q8_1_input_id = Some(input_id);
        }
        Ok(q8_1_ptr)
    }

    /// Explicitly invalidate the Q8_1 cache. Called after any dispatch
    /// that writes in-place to a buffer which might be the currently-
    /// cached MMVQ input — notably `FusedGemmResidual`, which reads and
    /// writes the same `residual` BufferId.
    fn invalidate_q8_1_cache(&mut self) {
        self.last_q8_1_input_id = None;
    }

    /// Pre-quantize the FP32 embedding row of `token_id` into the
    /// Q8_1 buffer. Side-effect only — no kernel consumes this since
    /// MMVQ runs its own per-GEMV quantize. Kept behind the
    /// `ROCMFORGE_Q8_1_PREQUANT` env var for historical smoke testing.
    fn prequantize_embedding_row_q8_1(&mut self, token_id: u32) -> HipResult<()> {
        let hidden = self.graph.config.hidden_dim;
        self.ensure_q8_1_buffer(hidden)?;
        let row_offset_bytes = (token_id as usize) * hidden * std::mem::size_of::<f32>();
        let input_ptr = unsafe {
            (self.embedding_fp32.as_ptr() as *const u8).add(row_offset_bytes) as *const f32
        };
        let out_ptr = self
            .q8_1_buffer
            .as_mut()
            .expect("ensure_q8_1_buffer just ran")
            .as_mut_ptr();
        let rc = unsafe {
            rocmforge_launch_quantize_q8_1(
                input_ptr,
                out_ptr,
                hidden as i32,
                self.stream.raw(),
            )
        };
        check(rc, "rocmforge_launch_quantize_q8_1")
    }

    /// Drop the per-node fast-dispatch cache. Called whenever a
    /// config field that influences kernel selection changes
    /// (`fused_gate_up`, `gate_up_dynamic`). After the next
    /// `execute_decode`, if the Bandit has converged the cache is
    /// rebuilt.
    fn invalidate_fast_dispatch(&mut self) {
        self.node_fast_cache = None;
        // Any config change that affects kernel selection invalidates
        // the captured HIP-graph too — the graph holds the pre-
        // bandit-committed function pointers.
        if let Some(cache) = self.hip_graph.take() {
            cache.destroy();
        }
    }

    /// Build the per-node fast-dispatch cache. Called lazily at
    /// `execute_decode` entry once the Bandit has committed on
    /// every registered shape. Caches:
    ///   * `committed_kernel`: the `KernelId` the Bandit currently
    ///     exploits for this Gemm/FusedGemmResidual node — eliminates
    ///     two HashMap lookups per Gemm dispatch.
    ///   * `weight_primary` / `weight_secondary` / `rope_freqs`:
    ///     reserved for a future per-node weight pointer cache.
    ///     Populated as null here; a follow-up pass will wire them
    ///     through the dispatch match arms.
    ///
    /// Returns `Ok(true)` when the cache was built; `Ok(false)` when
    /// the Bandit isn't ready yet (legacy path stays active).
    fn compile_fast_dispatch(&mut self) -> HipResult<bool> {
        if std::env::var("ROCMFORGE_LEGACY_DISPATCH").ok().as_deref() == Some("1") {
            return Ok(false);
        }
        let runtime = match self.runtime.as_ref() {
            Some(r) if r.all_exploiting() => r,
            _ => return Ok(false),
        };

        let mut cache: Vec<NodeFastEntry> = Vec::with_capacity(self.graph.nodes.len());
        for node in &self.graph.nodes {
            let mut entry = NodeFastEntry {
                weight_primary: std::ptr::null(),
                weight_secondary: std::ptr::null(),
                rope_freqs: std::ptr::null(),
                committed_kernel: None,
            };
            if let GraphNode::Gemm {
                weight,
                out_dim,
                in_dim,
                ..
            } = node
            {
                let shape = ShapeKey {
                    op_type: OpType::Gemv,
                    format: weight.format,
                    n: *out_dim as u32,
                    k: *in_dim as u32,
                };
                if let Some(variant_id) = runtime.select_variant(&shape) {
                    if let Some(kernel) = runtime.kernel_for(&shape, variant_id) {
                        entry.committed_kernel = Some(kernel);
                    }
                }
            }
            cache.push(entry);
        }
        self.node_fast_cache = Some(cache);
        Ok(true)
    }

    /// Override the prefill WMMA precision at runtime. Tests use
    /// this to toggle FP8 / FP16 on the same loaded pipeline.
    pub fn set_prefill_precision(&mut self, p: PrefillPrecision) {
        self.prefill_precision = p;
    }

    /// Current prefill WMMA precision — reflects env default plus
    /// any `set_prefill_precision` override.
    pub fn prefill_precision(&self) -> PrefillPrecision {
        self.prefill_precision
    }

    /// Install a GA-compiled dynamic kernel for `gate_up_swiglu`
    /// (Phase 2 step 2.1.3 Block D). The kernel is only used when a
    /// graph node's `(hidden_dim, ffn_dim)` matches `hidden_dim` and
    /// `ffn_dim` here — the grid math bakes both dimensions in, so a
    /// shape-mismatched dispatch would produce wrong results.
    ///
    /// Passing `None` clears the hook; the executor reverts to the
    /// static Phase-1 launcher.
    pub fn set_gate_up_swiglu_dynamic_kernel(&mut self, hook: Option<DynamicGateUpHook>) {
        self.gate_up_dynamic = hook;
        self.invalidate_fast_dispatch();
    }

    /// Read-only view on the currently installed GA-tuned kernel, if
    /// any. Tests use this to verify that `set_*` wired correctly.
    pub fn gate_up_swiglu_dynamic_kernel(&self) -> Option<&DynamicGateUpHook> {
        self.gate_up_dynamic.as_ref()
    }

    /// Attach a Self-Tuning Runtime. The executor then routes all
    /// GEMV dispatches through the Bandit and feeds back timings.
    /// Pre-registers every unique GEMV shape the graph will dispatch
    /// so the Bandit knows the candidate set before the first token.
    pub fn attach_runtime(&mut self, mut runtime: Runtime) {
        for node in &self.graph.nodes {
            if let GraphNode::Gemm {
                weight,
                out_dim,
                in_dim,
                ..
            } = node
            {
                runtime.registry.register_gemv_shape(
                    weight.format,
                    *out_dim as u32,
                    *in_dim as u32,
                );
            }
        }
        // Rebuild the Bandit map so the shapes we just registered get
        // fresh ShapeBandits where applicable.
        let rebuilt = Runtime::new(runtime.registry);
        self.runtime = Some(rebuilt);
        // Allocate the event pool — capacity covers the largest
        // per-token GEMV count we expect (Qwen3: 32 layers × ~5
        // GEMV nodes + LM-head = ~165; round to 256 for headroom).
        // Pool allocation failure is non-fatal; we just skip timing.
        match super::super::runtime::EventPool::new(256) {
            Ok(pool) => self.event_pool = Some(pool),
            Err(e) => {
                tracing::warn!(error = %e.message, "event pool alloc failed; \
                    Bandit will fall back to wall-clock syncs");
                self.event_pool = None;
            }
        }
    }

    pub fn runtime(&self) -> Option<&Runtime> {
        self.runtime.as_ref()
    }

    pub fn runtime_mut(&mut self) -> Option<&mut Runtime> {
        self.runtime.as_mut()
    }

    /// Number of tokens currently residing in the KV cache.
    pub fn kv_cache_len(&self) -> usize {
        // Caller tracks; see `execute_decode` which advances via `pos`.
        0
    }

    /// Reset the KV cache for a fresh conversation. Required between
    /// unrelated prompts (e.g. the 15-prompt validation run) so
    /// attention doesn't see stale context from the previous round.
    /// Caller must also reset its own position counter to 0 — this
    /// method just zero-fills the cache buffers.
    pub fn reset_kv_cache(&mut self) -> HipResult<()> {
        // Precision-aware: bf8 buffers are 1/4 the size of FP32 buffers.
        // Zero-filling is safe for both — FP8-E5M2 bit pattern 0x00 is
        // positive zero, same as FP32 0x00000000.
        let per_cache_bytes = self.kv_layout.bytes_per_side();
        let zeros = vec![0u8; per_cache_bytes];
        for buf in &mut self.k_cache {
            buf.copy_from_host(&zeros)?;
        }
        for buf in &mut self.v_cache {
            buf.copy_from_host(&zeros)?;
        }
        Ok(())
    }

    /// Run one forward pass for a single token. Returns the full logits
    /// vector (length = vocab_size).
    pub fn execute_decode(&mut self, token_id: u32, pos: usize) -> HipResult<Vec<f32>> {
        // Phase 2 Schritt 5 — reset the Q8_1 sharing cache. Each decode
        // step starts with no quantized buffer; the first MMVQ dispatch
        // this step will quantize and cache, and consecutive MMVQ
        // dispatches with the same input (Q → K → V) skip quantize.
        self.last_q8_1_input_id = None;

        // Phase-2 Schritt 1/3 — optional Q8_1 activation pre-quantize.
        // Runs OUTSIDE any HIP-Graph capture (the input pointer is
        // token-dependent, so it must not be baked into a replay).
        // Write-only until the MMVQ kernel lands in Schritt 2/3; pure
        // side-effect on `self.q8_1_buffer`, no influence on logits.
        if self.q8_1_prequant_enabled {
            self.prequantize_embedding_row_q8_1(token_id)?;
        }

        // HIP-Graph fast path (Option C, 2026-04-23).
        //   * If we already have a captured graph, replay it.
        //   * Otherwise, if the Bandit has converged and graph capture
        //     is not env-disabled, capture now. Capture runs the first
        //     launch itself (producing this token's logits) so this
        //     method returns successfully on either path.
        //   * Any failure in capture/replay falls back to the legacy
        //     dispatch below, which always works.
        if self.hip_graph.is_some() {
            match self.replay_decode_graph(token_id, pos) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    tracing::warn!(
                        error = %e.message,
                        "HIP-graph replay failed, invalidating cache and falling back"
                    );
                    if let Some(cache) = self.hip_graph.take() {
                        cache.destroy();
                    }
                    // Fall through to legacy path for this token.
                }
            }
        } else if self.should_capture_hip_graph() {
            match self.capture_decode_graph(token_id, pos) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    tracing::warn!(
                        error = %e.message,
                        "HIP-graph capture failed, continuing on legacy path"
                    );
                    // `capture_decode_graph` cleans up on failure;
                    // just fall through.
                }
            }
        }

        // Seed the token-ids buffer.
        let tok_bytes = token_id.to_le_bytes();
        self.buffers
            .get_mut(&self.graph.token_ids_buffer)
            .expect("token-ids buffer missing")
            .copy_from_host(&tok_bytes)?;

        // Lazy fast-dispatch cache build. Once the Bandit has
        // converged on every shape, cache the committed KernelId
        // per Gemm node so the hot path skips two HashMap lookups
        // per Gemm launch (runtime.select_variant + kernel_for).
        if self.node_fast_cache.is_none() {
            let _ = self.compile_fast_dispatch()?;
        }

        // Walk the graph.
        // Iterate over indexes so we can mutably borrow `self.buffers`
        // while reading the graph structure.
        for i in 0..self.graph.nodes.len() {
            self.dispatch_node(i, pos)?;
            if self.nan_guard || self.trace {
                self.check_node(i)?;
            }
        }

        // Zero-Sync Pipeline (Arch-Doc §3.7): this is the one
        // explicit `hipStreamSynchronize` per token. It replaces
        // the ~500 syncs the Phase-1 bandit issued inside
        // `dispatch_gemv_tuned`. The sync is required because
        // `hipMemcpy` (blocking, no stream arg) does **not**
        // serialise against work on user-created streams — without
        // this line, `read_buffer` could race the last GEMV writing
        // into the logits buffer.
        self.stream.synchronize()?;
        let logits = self.read_buffer(self.graph.logits_buffer, self.graph.config.vocab_size)?;
        // Events are complete now — drain them into the Bandit.
        self.flush_event_pool()?;
        for (i, v) in logits.iter().enumerate() {
            if !v.is_finite() {
                return Err(HipError {
                    code: -1,
                    message: format!("logits[{i}] = {v} (non-finite)"),
                    context: "execute_decode".into(),
                });
            }
        }
        Ok(logits)
    }

    /// Run prefill + decode: prefill seeds the KV cache with
    /// `token_ids[..]`, then returns the logits for the *last* token.
    ///
    /// Phase-2 Schritt 2.1.5: when the prompt is long enough for
    /// WMMA to pay off, dispatches the batched WMMA-based prefill
    /// (see [`execute_prefill_wmma`]). Otherwise falls back to the
    /// Phase-1 decode-loop — a correct but slow `O(seq_len ×
    /// decode_cost)` implementation.
    pub fn execute_prefill(&mut self, token_ids: &[u32], pos_offset: usize) -> HipResult<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(HipError {
                code: -1,
                message: "execute_prefill called with 0 tokens".into(),
                context: "prefill".into(),
            });
        }
        // Phase 2.4 — multi-turn correctness gate. The WMMA prefill
        // kernel computes causal attention ONLY over the new prompt
        // tokens (`attention(q_new, k_new, v_new)`); it does not
        // attend to the K/V cache populated by prior turns. For any
        // follow-up turn the new tokens MUST attend to the cached
        // conversation history, so we route those through the
        // sequential decode-loop prefill path (uses `attention_decode`
        // which reads the full cache). First-turn / single-shot
        // prompts still get the WMMA fast path.
        if pos_offset > 0 {
            return self.execute_prefill_decode_loop(token_ids, pos_offset);
        }
        if self.should_use_wmma_prefill(token_ids.len()) {
            match self.execute_prefill_wmma(token_ids, pos_offset) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    // Hard policy: if WMMA prefill is eligible but fails,
                    // propagate the error instead of silently producing
                    // bogus output via the decode-loop fallback. A
                    // runtime-env opt-out is available via
                    // ROCMFORGE_DISABLE_WMMA_PREFILL.
                    if std::env::var("ROCMFORGE_DISABLE_WMMA_PREFILL")
                        .ok()
                        .as_deref()
                        != Some("1")
                    {
                        return Err(e);
                    }
                    tracing::warn!(
                        error = %e.message,
                        "WMMA prefill failed; falling back to decode loop (env override)"
                    );
                }
            }
        }
        self.execute_prefill_decode_loop(token_ids, pos_offset)
    }

    /// Phase-1 decode-loop prefill. Retained as an explicit entry
    /// point so tests can force the decode path even when the prompt
    /// is long enough for WMMA, and as the fallback when
    /// `should_use_wmma_prefill` returns false (short prompts).
    pub fn execute_prefill_decode_loop(
        &mut self,
        token_ids: &[u32],
        pos_offset: usize,
    ) -> HipResult<Vec<f32>> {
        let mut last = Vec::new();
        for (i, &tok) in token_ids.iter().enumerate() {
            last = self.execute_decode(tok, pos_offset + i)?;
        }
        Ok(last)
    }

    /// Decide whether a prompt of length `seq_len` should go through
    /// the WMMA-batched prefill.
    ///
    /// Rules:
    ///   * `ROCMFORGE_DISABLE_WMMA_PREFILL=1` → always false
    ///   * `seq_len < WMMA_PREFILL_MIN_SEQ_LEN (16)` → decode-loop
    ///   * Otherwise → WMMA. Kernel-shape constraints (e.g. Q4_K's
    ///     M must be a multiple of 64) are enforced in the dispatch
    ///     path via padding to `padded_m = round_up(seq_len, 64)`.
    pub fn should_use_wmma_prefill(&self, seq_len: usize) -> bool {
        if std::env::var("ROCMFORGE_DISABLE_WMMA_PREFILL")
            .ok()
            .as_deref()
            == Some("1")
        {
            return false;
        }
        seq_len >= WMMA_PREFILL_MIN_SEQ_LEN
    }

    // Kernel dispatch ---------------------------------------------------

    fn dispatch_node(&mut self, node_idx: usize, pos: usize) -> HipResult<()> {
        // Borrow nodes as an immutable snapshot; the node itself only
        // reads constants — all device memory touches go through
        // `self.buffers`, which we mutate.
        let node = &self.graph.nodes[node_idx] as *const GraphNode;
        // SAFETY: we don't touch `self.graph` mutably during dispatch.
        let node = unsafe { &*node };
        match node {
            GraphNode::Embedding {
                table: _table,
                input,
                output,
            } => {
                // Use the pre-dequantised FP32 table prepared in `new`.
                let table_ptr = self.embedding_fp32.as_ptr();
                let in_ptr = self.buf_ptr(*input);
                let out_ptr = self.buf_mut_ptr(*output);
                let rc = unsafe {
                    rocmforge_launch_embedding_lookup(
                        in_ptr as *const u32,
                        table_ptr as *const f32,
                        out_ptr as *mut f32,
                        1,
                        self.graph.config.hidden_dim as i32,
                        self.stream.raw(),
                    )
                };
                check(rc, "embedding")?;
            }
            GraphNode::RmsNorm {
                weight,
                input,
                output,
                eps,
                dim,
                num_rows,
            } => {
                let w_ptr = self.weight_ptr(weight);
                let in_ptr = self.buf_ptr(*input);
                let out_ptr = self.buf_mut_ptr(*output);
                let rc = if *num_rows == 1 {
                    unsafe {
                        rocmforge_launch_rms_norm(
                            in_ptr as *const f32,
                            w_ptr as *const f32,
                            out_ptr as *mut f32,
                            *dim as i32,
                            *eps,
                            self.stream.raw(),
                        )
                    }
                } else {
                    unsafe {
                        rocmforge_launch_rms_norm_batched(
                            in_ptr as *const f32,
                            w_ptr as *const f32,
                            out_ptr as *mut f32,
                            *dim as i32,
                            *eps,
                            *num_rows as i32,
                            self.stream.raw(),
                        )
                    }
                };
                check(rc, "rms_norm")?;
            }
            GraphNode::Gemm {
                weight,
                input,
                output,
                out_dim,
                in_dim,
                ..
            } => {
                self.dispatch_gemv(weight, *input, *output, *out_dim, *in_dim, Some(node_idx))?;
            }
            GraphNode::Rope {
                q_buffer,
                k_buffer,
                rope_freqs,
                n_heads,
                n_kv_heads,
                head_dim,
                theta_base,
            } => {
                let freq_ptr = rope_freqs
                    .as_ref()
                    .map(|w| self.weight_ptr(w) as *const f32)
                    .unwrap_or(std::ptr::null());
                let q_ptr = self.buf_mut_ptr(*q_buffer);
                let k_ptr = self.buf_mut_ptr(*k_buffer);
                // Q
                let rc = unsafe {
                    rocmforge_launch_rope(
                        q_ptr as *mut f32,
                        pos as i32,
                        *n_heads as i32,
                        *head_dim as i32,
                        *theta_base,
                        freq_ptr,
                        self.stream.raw(),
                    )
                };
                check(rc, "rope Q")?;
                // K
                let rc = unsafe {
                    rocmforge_launch_rope(
                        k_ptr as *mut f32,
                        pos as i32,
                        *n_kv_heads as i32,
                        *head_dim as i32,
                        *theta_base,
                        freq_ptr,
                        self.stream.raw(),
                    )
                };
                check(rc, "rope K")?;
            }
            GraphNode::KvCacheAppend {
                k_buffer,
                v_buffer,
                layer_idx,
            } => {
                let k_new = self.buf_ptr(*k_buffer);
                let v_new = self.buf_ptr(*v_buffer);
                let k_cache = self.k_cache[*layer_idx].as_mut_ptr();
                let v_cache = self.v_cache[*layer_idx].as_mut_ptr();
                // Phase 2.2A — precision-aware dispatch. bf8 path writes
                // one byte per element; FP32 path writes four.
                let rc = match self.kv_layout.precision {
                    KvPrecision::Fp32 => unsafe {
                        rocmforge_launch_kv_cache_append(
                            k_cache as *mut f32,
                            v_cache as *mut f32,
                            k_new as *const f32,
                            v_new as *const f32,
                            self.kv_layout.num_kv_heads as i32,
                            self.kv_layout.head_dim as i32,
                            pos as i32,
                            self.kv_layout.head_stride as i32,
                            self.stream.raw(),
                        )
                    },
                    KvPrecision::Fp8E5M2 => unsafe {
                        rocmforge_launch_kv_cache_append_fp8(
                            k_cache,
                            v_cache,
                            k_new as *const f32,
                            v_new as *const f32,
                            self.kv_layout.num_kv_heads as i32,
                            self.kv_layout.head_dim as i32,
                            pos as i32,
                            self.kv_layout.head_stride as i32,
                            self.stream.raw(),
                        )
                    },
                };
                check(rc, "kv_cache_append")?;
            }
            GraphNode::Attention {
                q_buffer,
                output,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
            } => {
                let q_ptr = self.buf_ptr(*q_buffer);
                let out_ptr = self.buf_mut_ptr(*output);
                let k_cache = self.k_cache[*layer_idx].as_ptr();
                let v_cache = self.v_cache[*layer_idx].as_ptr();
                let scale = 1.0f32 / (*head_dim as f32).sqrt();
                let rc = match self.kv_layout.precision {
                    KvPrecision::Fp32 => unsafe {
                        rocmforge_launch_attention_decode(
                            q_ptr as *const f32,
                            k_cache as *const f32,
                            v_cache as *const f32,
                            out_ptr as *mut f32,
                            *n_heads as i32,
                            *n_kv_heads as i32,
                            *head_dim as i32,
                            (pos + 1) as i32,
                            self.kv_layout.head_stride as i32,
                            scale,
                            self.stream.raw(),
                        )
                    },
                    KvPrecision::Fp8E5M2 => unsafe {
                        rocmforge_launch_attention_decode_fp8(
                            q_ptr as *const f32,
                            k_cache,
                            v_cache,
                            out_ptr as *mut f32,
                            *n_heads as i32,
                            *n_kv_heads as i32,
                            *head_dim as i32,
                            (pos + 1) as i32,
                            self.kv_layout.head_stride as i32,
                            scale,
                            self.stream.raw(),
                        )
                    },
                };
                check(rc, "attention_decode")?;
            }
            GraphNode::ResidualAdd { a, b } => {
                let a_ptr = self.buf_mut_ptr(*a);
                let b_ptr = self.buf_ptr(*b);
                let n = self.plan.specs[a].elem_count as i32;
                let rc = unsafe {
                    rocmforge_launch_residual_add_inplace(
                        a_ptr as *mut f32,
                        b_ptr as *const f32,
                        n,
                        self.stream.raw(),
                    )
                };
                check(rc, "residual_add")?;
                // Schritt 5: in-place write on `a` — if a MMVQ quantized
                // that buffer and its contents just changed, the cache
                // would serve stale data. Invalidate defensively.
                if self.last_q8_1_input_id == Some(*a) {
                    self.invalidate_q8_1_cache();
                }
            }
            GraphNode::GateUpSwiGLU {
                gate_weight,
                up_weight,
                input,
                output,
                ffn_dim,
                hidden_dim,
            } => {
                self.dispatch_gate_up_swiglu(
                    gate_weight,
                    up_weight,
                    *input,
                    *output,
                    *ffn_dim,
                    *hidden_dim,
                )?;
            }
            GraphNode::SwiGLU {
                gate_buffer,
                up_buffer,
                output,
            } => {
                let g_ptr = self.buf_ptr(*gate_buffer);
                let u_ptr = self.buf_ptr(*up_buffer);
                let o_ptr = self.buf_mut_ptr(*output);
                let n = self.plan.specs[output].elem_count as i32;
                let rc = unsafe {
                    rocmforge_launch_swiglu(
                        g_ptr as *const f32,
                        u_ptr as *const f32,
                        o_ptr as *mut f32,
                        n,
                        self.stream.raw(),
                    )
                };
                check(rc, "swiglu")?;
            }
            GraphNode::FusedGemmResidual {
                weight,
                input,
                residual,
                out_dim,
                in_dim,
            } => {
                // Saves one dispatch + one VRAM round-trip vs. the
                // unfused `Gemm → ResidualAdd` pair. See
                // `hip_kernels_v1/gemv/gemv_q4_k_q8_inline_residual.hip`.
                self.dispatch_fused_gemm_residual(weight, *input, *residual, *out_dim, *in_dim)?;
            }
        }
        // No per-node stream sync: HIP guarantees in-order
        // execution within a single stream, so the next node's
        // kernel will see this one's output without a CPU sync.
        // The only syncs per token now are: (1) logits readback in
        // `execute_decode`, (2) monitor hidden-state reads (every
        // sample_rate tokens), (3) the trace/NaN-guard `check_node`
        // path below, which opts in via env var.
        if self.nan_guard || self.trace {
            self.stream.synchronize()?;
        }
        Ok(())
    }

    fn dispatch_gemv(
        &mut self,
        weight: &WeightRef,
        input: BufferId,
        output: BufferId,
        out_dim: usize,
        in_dim: usize,
        node_idx: Option<usize>,
    ) -> HipResult<()> {
        // Path 1 — Bandit attached: ask which kernel to launch,
        // time the launch + sync, and feed the time back. The Bandit
        // uses wall-clock around the synchronize() call; Phase-2
        // replaces this with HIP-event timing batched at token end
        // so the extra sync goes away.
        if self.runtime.is_some() {
            return self.dispatch_gemv_tuned(weight, input, output, out_dim, in_dim, node_idx);
        }

        // Path 2 — no Bandit: straight to the fixed `_standard`
        // kernel for this quant format (pre-1.12 behaviour).
        let w_ptr = self.weight_ptr(weight);
        let in_ptr = self.buf_ptr(input);
        let out_ptr = self.buf_mut_ptr(output);
        let rc = match weight.format {
            GgmlType::Q4_0 => unsafe {
                rocmforge_launch_gemv_q4_0_standard(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    self.stream.raw(),
                )
            },
            GgmlType::Q4_K => unsafe {
                rocmforge_launch_gemv_q4_k_standard(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    self.stream.raw(),
                )
            },
            GgmlType::Q6_K => unsafe {
                rocmforge_launch_gemv_q6_k_standard(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    self.stream.raw(),
                )
            },
            GgmlType::Q8_0 => unsafe {
                rocmforge_launch_gemv_q8_0_standard(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    self.stream.raw(),
                )
            },
            other => {
                return Err(HipError {
                    code: -1,
                    message: format!("no GEMV kernel for format {other}"),
                    context: "dispatch_gemv".into(),
                })
            }
        };
        check(rc, "gemv")?;
        Ok(())
    }

    /// Bandit-driven GEMV dispatch. Separate from `dispatch_gemv`
    /// so the fixed-kernel path stays untouched when no Runtime is
    /// attached. Caller guarantees `self.runtime.is_some()`.
    ///
    /// Zero-Sync Pipeline (Arch-Doc §3.7 / Phase-2 §8.1):
    /// * **Exploration** — record `(start, stop)` HIP events on the
    ///   stream around the launch, push the `(shape, variant_id)`
    ///   onto the event pool. No stream sync here. Events are drained
    ///   once at token end (`flush_event_pool`), which brings total
    ///   syncs from ~500 per token down to 1.
    /// * **Exploitation** — after every Bandit shape has
    ///   `is_exploiting() == true`, skip event recording entirely.
    ///   The GPU pipelines kernels back-to-back with zero CPU
    ///   involvement until the logits readback forces a sync.
    fn dispatch_gemv_tuned(
        &mut self,
        weight: &WeightRef,
        input: BufferId,
        output: BufferId,
        out_dim: usize,
        in_dim: usize,
        node_idx: Option<usize>,
    ) -> HipResult<()> {
        // Fast path: if the per-node fast-dispatch cache is built
        // (which only happens once the Bandit has committed on
        // every shape), we can use the pre-resolved KernelId and
        // skip the two HashMap lookups + the all_exploiting() check
        // the legacy path does. ShapeKey / variant_id are also
        // unused in this branch; bandit lifetime-state is stable.
        let cached_kernel = node_idx
            .and_then(|idx| {
                self.node_fast_cache
                    .as_ref()
                    .map(|c| c[idx].committed_kernel)
            })
            .flatten();
        // Fall back to the legacy runtime query when no cache is
        // available (Bandit still exploring, or hot-path was
        // invalidated by a setter).
        let (kernel, want_timing, shape_for_timing, variant_for_timing) =
            if let Some(k) = cached_kernel {
                // Cache is only populated when runtime.all_exploiting()
                // holds → no timing is needed here by definition.
                (k, false, None, None)
            } else {
                let shape = ShapeKey {
                    op_type: OpType::Gemv,
                    format: weight.format,
                    n: out_dim as u32,
                    k: in_dim as u32,
                };
                let runtime = self
                    .runtime
                    .as_ref()
                    .expect("dispatch_gemv_tuned requires an attached runtime");
                let variant_id = runtime.select_variant(&shape).ok_or_else(|| HipError {
                    code: -1,
                    message: format!("no variant registered for {shape:?}"),
                    context: "dispatch_gemv_tuned".into(),
                })?;
                let kernel = runtime
                    .kernel_for(&shape, variant_id)
                    .ok_or_else(|| HipError {
                        code: -1,
                        message: format!("variant {variant_id:?} has no kernel for {shape:?}"),
                        context: "dispatch_gemv_tuned".into(),
                    })?;
                // Only record events while the Bandit is still learning the
                // shape's optimum. After convergence the arms stop moving
                // and new measurements add no information — skipping them
                // lets the GPU run without any CPU-visible interruption.
                let want_timing = !runtime.all_exploiting();
                (kernel, want_timing, Some(shape), Some(variant_id))
            };

        let w_ptr = self.weight_ptr(weight);
        let in_ptr = self.buf_ptr(input);
        let out_ptr = self.buf_mut_ptr(output);
        let stream_raw = self.stream.raw();

        let pair_idx = if want_timing {
            // `shape_for_timing` and `variant_for_timing` are populated
            // together with `want_timing = true` in the legacy path
            // above; unwrap is safe because the fast path sets
            // want_timing=false.
            let (shape, variant_id) = match (shape_for_timing, variant_for_timing) {
                (Some(s), Some(v)) => (s, v),
                _ => {
                    return Err(HipError {
                        code: -1,
                        message: "timing requested without shape/variant".into(),
                        context: "dispatch_gemv_tuned".into(),
                    })
                }
            };
            if let Some(pool) = self.event_pool.as_mut() {
                pool.record_start(&self.stream, shape, variant_id)?
            } else {
                None
            }
        } else {
            None
        };

        let rc = unsafe {
            match kernel {
                KernelId::GemvQ40Standard => rocmforge_launch_gemv_q4_0_standard(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    stream_raw,
                ),
                KernelId::GemvQ4KStandard => rocmforge_launch_gemv_q4_k_standard(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    stream_raw,
                ),
                KernelId::GemvQ4KQ8Inline => rocmforge_launch_gemv_q4_k_q8_inline(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    stream_raw,
                ),
                KernelId::GemvQ4KMmvq => {
                    // Phase-2 Schritt 2/3: per-GEMV Q8_1 pre-quantize +
                    // llama.cpp-style MMVQ. Schritt 5: `ensure_q8_1_quantized`
                    // skips the quantize call when the input BufferId
                    // matches the last cached one (Q → K → V chain).
                    let q8_1_ptr = self.ensure_q8_1_quantized(input, in_ptr, in_dim)?;
                    rocmforge_launch_gemv_q4_k_mmvq(
                        w_ptr,
                        q8_1_ptr as *const std::ffi::c_void,
                        out_ptr,
                        out_dim as i32,
                        in_dim as i32,
                        stream_raw,
                    )
                }
                KernelId::GemvQ4KQ8InlineSudot4 => rocmforge_launch_gemv_q4_k_q8_inline_sudot4(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    stream_raw,
                ),
                KernelId::GemvQ6KStandard => rocmforge_launch_gemv_q6_k_standard(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    stream_raw,
                ),
                KernelId::GemvQ6KQ8Inline => rocmforge_launch_gemv_q6_k_q8_inline(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    stream_raw,
                ),
                KernelId::GemvQ6KMmvq => {
                    // Same dispatch shape as GemvQ4KMmvq with Schritt 5
                    // Q8_1 sharing.
                    let q8_1_ptr = self.ensure_q8_1_quantized(input, in_ptr, in_dim)?;
                    rocmforge_launch_gemv_q6_k_mmvq(
                        w_ptr,
                        q8_1_ptr as *const std::ffi::c_void,
                        out_ptr,
                        out_dim as i32,
                        in_dim as i32,
                        stream_raw,
                    )
                }
                KernelId::GemvQ80Standard => rocmforge_launch_gemv_q8_0_standard(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    stream_raw,
                ),
                other => {
                    return Err(HipError {
                        code: -1,
                        message: format!("kernel {other:?} not a GEMV kernel"),
                        context: "dispatch_gemv_tuned".into(),
                    });
                }
            }
        };
        check(rc, "gemv_tuned_launch")?;

        if let Some(idx) = pair_idx {
            if let Some(pool) = self.event_pool.as_mut() {
                pool.record_stop(&self.stream, idx)?;
            }
        }
        Ok(())
    }

    /// Phase-2 fused GEMV + ResidualAdd dispatch. Q4_K-only for now
    /// — other quant formats fall back to the split path because no
    /// fused kernel is emitted yet. The weight format is asserted
    /// rather than returning an error: the graph-builder fuse-pass
    /// already gates on `format == Q4_K`, so anything else reaching
    /// this function is an invariant violation.
    fn dispatch_fused_gemm_residual(
        &mut self,
        weight: &WeightRef,
        input: BufferId,
        residual: BufferId,
        out_dim: usize,
        in_dim: usize,
    ) -> HipResult<()> {
        debug_assert_eq!(
            weight.format,
            GgmlType::Q4_K,
            "FusedGemmResidual only Q4_K in Phase 2 step 2.0.2"
        );
        let w_ptr = self.weight_ptr(weight);
        let in_ptr = self.buf_ptr(input);
        let residual_ptr = self.buf_mut_ptr(residual);

        // Phase 2 Schritt 2b (2026-04-24): route Q4_K FusedGemmResidual
        // through the MMVQ-residual kernel. Path: pre-quantize activation
        // to Q8_1, then MMVQ with residual-add epilog in the same launch.
        // Fallback env var `ROCMFORGE_DISABLE_MMVQ_RESIDUAL=1` reverts
        // to the old q8_inline_residual kernel for A/B / regression.
        let use_mmvq = std::env::var("ROCMFORGE_DISABLE_MMVQ_RESIDUAL")
            .ok()
            .as_deref()
            != Some("1");
        let rc = if use_mmvq {
            // Schritt 5: Q8_1 sharing — skip quantize when `input`
            // BufferId matches the last cached MMVQ input.
            let q8_1_ptr = self.ensure_q8_1_quantized(input, in_ptr, in_dim)?;
            let result = unsafe {
                rocmforge_launch_gemv_q4_k_mmvq_residual(
                    w_ptr,
                    q8_1_ptr as *const std::ffi::c_void,
                    residual_ptr as *const std::ffi::c_void,
                    residual_ptr,
                    out_dim as i32,
                    in_dim as i32,
                    self.stream.raw(),
                )
            };
            // After the residual-add epilog the `residual` buffer is
            // written in-place. If a later MMVQ dispatch happens to
            // read that same BufferId, the cached Q8_1 data would be
            // stale — invalidate defensively. (The cache also naturally
            // misses when the next MMVQ reads a different BufferId,
            // which is the common case.)
            self.invalidate_q8_1_cache();
            result
        } else {
            unsafe {
                rocmforge_launch_gemv_q4_k_q8_inline_residual(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    residual_ptr as *const f32,
                    residual_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    self.stream.raw(),
                )
            }
        };
        check(rc, if use_mmvq { "gemv_q4_k_mmvq_residual" } else { "gemv_q4_k_q8_inline_residual" })?;
        Ok(())
    }

    /// Drain the event pool into the Bandit. Called once per token
    /// after the logits readback syncs the stream — the events are
    /// guaranteed complete at that point. Cheap no-op when the pool
    /// is empty (e.g. Bandit is in exploitation).
    fn flush_event_pool(&mut self) -> HipResult<()> {
        let pool = match self.event_pool.as_mut() {
            Some(p) if !p.is_empty() => p,
            _ => return Ok(()),
        };
        let runtime = match self.runtime.as_mut() {
            Some(r) => r,
            None => {
                pool.clear_pending();
                return Ok(());
            }
        };
        pool.flush_into(runtime)
    }

    fn dispatch_gate_up_swiglu(
        &mut self,
        gate_weight: &WeightRef,
        up_weight: &WeightRef,
        input: BufferId,
        output: BufferId,
        ffn_dim: usize,
        hidden_dim: usize,
    ) -> HipResult<()> {
        // Only Q4_K has the fused Q4_K kernel from Schritt 1.8 Block C.
        // For other formats we fall back to split gate-GEMV + up-GEMV
        // + SwiGLU — temporary buffers borrowed from the layout by
        // allocating fresh `HipBuffer`s. This path is slower but fully
        // correct and exercises the SwiGLU kernel we added this block.
        if gate_weight.format == GgmlType::Q4_K && up_weight.format == GgmlType::Q4_K {
            // Block D: if a GA-tuned dynamic kernel is installed for
            // this exact shape, dispatch through it instead of the
            // static Phase-1 launcher. Mismatched shapes fall through
            // to the static path — the grid math depends on both
            // dimensions, so using the wrong kernel would silently
            // corrupt the output.
            let use_dynamic = self
                .gate_up_dynamic
                .as_ref()
                .map(|h| h.hidden_dim == hidden_dim && h.ffn_dim == ffn_dim)
                .unwrap_or(false);
            if use_dynamic {
                let gate_ptr = self.weight_ptr(gate_weight);
                let up_ptr = self.weight_ptr(up_weight);
                let in_ptr = self.buf_ptr(input);
                let out_ptr = self.buf_mut_ptr(output);
                let stream_raw = self.stream.raw();
                // Clone the Arc<DynamicKernel> to drop the borrow on
                // self before mutably-borrowing to launch. Arc-clone
                // is a ref-count bump, no heap work.
                let kernel = self
                    .gate_up_dynamic
                    .as_ref()
                    .expect("checked above")
                    .kernel
                    .clone();
                unsafe {
                    kernel.launch_gate_up_swiglu_raw(
                        gate_ptr,
                        up_ptr,
                        in_ptr,
                        out_ptr,
                        hidden_dim as i32,
                        ffn_dim as i32,
                        stream_raw,
                    )
                }?;
                return Ok(());
            }

            // ── MMVQ-fused path (Phase 2 Schritt 3, default) ─────
            // Two parallel Q4_K dots over the same pre-quantized Q8_1
            // activation + silu(gate)×up epilog in register. 1.40×
            // faster than the unfused 5-kernel sequence (measured:
            // unfused 109.87 µs, fused 78.55 µs on N=12288, K=4096 —
            // see `results/phase2_mmvq_gate_fusion.md`). Bit-exact vs
            // composite on parity tests. Opt-out via
            // `ROCMFORGE_DISABLE_MMVQ_FUSION=1` to force the unfused
            // path below for A/B measurement.
            let use_mmvq_fusion = std::env::var("ROCMFORGE_DISABLE_MMVQ_FUSION")
                .ok()
                .as_deref()
                != Some("1");
            if use_mmvq_fusion {
                let gate_ptr = self.weight_ptr(gate_weight);
                let up_ptr = self.weight_ptr(up_weight);
                let in_ptr = self.buf_ptr(input);
                let out_ptr = self.buf_mut_ptr(output);
                let stream_raw = self.stream.raw();
                // Schritt 5: Q8_1 sharing — skip quantize when `input`
                // BufferId matches the last cached MMVQ input.
                let q8_1_ptr = self.ensure_q8_1_quantized(input, in_ptr, hidden_dim)?;
                let rc = unsafe {
                    rocmforge_launch_gemv_q4_k_mmvq_fused(
                        gate_ptr,
                        up_ptr,
                        q8_1_ptr as *const std::ffi::c_void,
                        out_ptr,
                        ffn_dim as i32,
                        hidden_dim as i32,
                        stream_raw,
                    )
                };
                check(rc, "gemv_q4_k_mmvq_fused")?;
                return Ok(());
            }

            // ── Fused path (legacy, pre-MMVQ) ────────────────────
            // The fused `gemv_q4_k_gate_up_swiglu` kernel hits only
            // 20 % of the 640 GB/s peak BW on gfx1201 (rocprof
            // deep-dive 2026-04-23, 65 % of decode time sat here).
            // Ship as opt-in via ROCMFORGE_FUSED_GATE_UP=1 for A/B.
            if self.fused_gate_up {
                let gate_ptr = self.weight_ptr(gate_weight);
                let up_ptr = self.weight_ptr(up_weight);
                let in_ptr = self.buf_ptr(input);
                let out_ptr = self.buf_mut_ptr(output);
                let rc = unsafe {
                    rocmforge_launch_gemv_q4_k_gate_up_swiglu(
                        gate_ptr as *const u8,
                        up_ptr as *const u8,
                        in_ptr as *const f32,
                        out_ptr as *mut f32,
                        hidden_dim as i32,
                        ffn_dim as i32,
                        self.stream.raw(),
                    )
                };
                check(rc, "gemv_q4_k_gate_up_swiglu")?;
                return Ok(());
            }

            // ── Un-fused path (default post-2.1.5 follow-up) ─────
            // 2× q4_k_q8_inline + 1× swiglu. The standalone
            // q4_k_q8_inline_residual kernel already runs at
            // 442 GB/s (69 % of peak) on similar shapes, so
            // splitting recovers ~3× on the hottest decode kernel
            // at the cost of 2 × ffn_dim × 4 B VRAM scratch.

            // Ensure scratch buffers exist at this ffn_dim. Lazy
            // alloc because the graph-build ffn_dim is constant
            // for a given model — allocate once, reuse for every
            // layer for every token.
            let scratch_bytes = ffn_dim * 4;
            let need_alloc = self
                .gate_scratch
                .as_ref()
                .map(|b| b.size() < scratch_bytes)
                .unwrap_or(true);
            if need_alloc {
                self.gate_scratch = Some(HipBuffer::new(scratch_bytes)?);
                self.up_scratch = Some(HipBuffer::new(scratch_bytes)?);
            }

            let gate_w_ptr = self.weight_ptr(gate_weight) as *const u8;
            let up_w_ptr = self.weight_ptr(up_weight) as *const u8;
            let in_ptr = self.buf_ptr(input) as *const f32;
            let out_ptr = self.buf_mut_ptr(output) as *mut f32;
            let gate_scratch_ptr = self
                .gate_scratch
                .as_mut()
                .expect("gate scratch alloc'd")
                .as_mut_ptr() as *mut f32;
            let up_scratch_ptr = self
                .up_scratch
                .as_mut()
                .expect("up scratch alloc'd")
                .as_mut_ptr() as *mut f32;
            let stream_raw = self.stream.raw();

            // 1. Gate projection
            let rc = unsafe {
                rocmforge_launch_gemv_q4_k_q8_inline(
                    gate_w_ptr,
                    in_ptr,
                    gate_scratch_ptr,
                    hidden_dim as i32,
                    ffn_dim as i32,
                    stream_raw,
                )
            };
            check(rc, "unfused gate_up: gate GEMV")?;

            // 2. Up projection
            let rc = unsafe {
                rocmforge_launch_gemv_q4_k_q8_inline(
                    up_w_ptr,
                    in_ptr,
                    up_scratch_ptr,
                    hidden_dim as i32,
                    ffn_dim as i32,
                    stream_raw,
                )
            };
            check(rc, "unfused gate_up: up GEMV")?;

            // 3. SwiGLU — silu(gate) * up → output
            let rc = unsafe {
                rocmforge_launch_swiglu(
                    gate_scratch_ptr as *const f32,
                    up_scratch_ptr as *const f32,
                    out_ptr,
                    ffn_dim as i32,
                    stream_raw,
                )
            };
            check(rc, "unfused gate_up: swiglu")?;
            return Ok(());
        }

        // Fallback split path: gate GEMV → up GEMV → SwiGLU.
        let gate_buf = HipBuffer::new(ffn_dim * 4)?;
        let up_buf = HipBuffer::new(ffn_dim * 4)?;
        let in_ptr = self.buf_ptr(input);
        // Gate
        let rc = match gate_weight.format {
            GgmlType::Q4_0 => unsafe {
                rocmforge_launch_gemv_q4_0_standard(
                    self.weight_ptr(gate_weight) as *const u8,
                    in_ptr as *const f32,
                    gate_buf.as_ptr() as *mut f32,
                    hidden_dim as i32,
                    ffn_dim as i32,
                    self.stream.raw(),
                )
            },
            GgmlType::Q6_K => unsafe {
                rocmforge_launch_gemv_q6_k_standard(
                    self.weight_ptr(gate_weight) as *const u8,
                    in_ptr as *const f32,
                    gate_buf.as_ptr() as *mut f32,
                    hidden_dim as i32,
                    ffn_dim as i32,
                    self.stream.raw(),
                )
            },
            GgmlType::Q8_0 => unsafe {
                rocmforge_launch_gemv_q8_0_standard(
                    self.weight_ptr(gate_weight) as *const u8,
                    in_ptr as *const f32,
                    gate_buf.as_ptr() as *mut f32,
                    hidden_dim as i32,
                    ffn_dim as i32,
                    self.stream.raw(),
                )
            },
            other => {
                return Err(HipError {
                    code: -1,
                    message: format!("no GEMV for gate format {other}"),
                    context: "gate_up split".into(),
                })
            }
        };
        check(rc, "gate gemv")?;
        // Up
        let rc = match up_weight.format {
            GgmlType::Q4_0 => unsafe {
                rocmforge_launch_gemv_q4_0_standard(
                    self.weight_ptr(up_weight) as *const u8,
                    in_ptr as *const f32,
                    up_buf.as_ptr() as *mut f32,
                    hidden_dim as i32,
                    ffn_dim as i32,
                    self.stream.raw(),
                )
            },
            GgmlType::Q6_K => unsafe {
                rocmforge_launch_gemv_q6_k_standard(
                    self.weight_ptr(up_weight) as *const u8,
                    in_ptr as *const f32,
                    up_buf.as_ptr() as *mut f32,
                    hidden_dim as i32,
                    ffn_dim as i32,
                    self.stream.raw(),
                )
            },
            GgmlType::Q8_0 => unsafe {
                rocmforge_launch_gemv_q8_0_standard(
                    self.weight_ptr(up_weight) as *const u8,
                    in_ptr as *const f32,
                    up_buf.as_ptr() as *mut f32,
                    hidden_dim as i32,
                    ffn_dim as i32,
                    self.stream.raw(),
                )
            },
            other => {
                return Err(HipError {
                    code: -1,
                    message: format!("no GEMV for up format {other}"),
                    context: "gate_up split".into(),
                })
            }
        };
        check(rc, "up gemv")?;
        // SwiGLU
        let out_ptr = self.buf_mut_ptr(output);
        let rc = unsafe {
            rocmforge_launch_swiglu(
                gate_buf.as_ptr() as *const f32,
                up_buf.as_ptr() as *const f32,
                out_ptr as *mut f32,
                ffn_dim as i32,
                self.stream.raw(),
            )
        };
        check(rc, "swiglu fallback")?;
        // Suppress unused warning (integrates q8_inline launcher above, not used yet).
        let _ = rocmforge_launch_gemv_q4_k_q8_inline;
        Ok(())
    }

    // Helpers -----------------------------------------------------------

    fn buf_ptr(&self, id: BufferId) -> *const c_void {
        // Fast path: indexed array lookup (populated at new() time).
        // Dispatch-optimisation follow-up 2026-04-23 — replaces the
        // per-call HashMap lookup that cost ~100 ns × 3 buffers per
        // node × 600 nodes = ~180 µs per decode token.
        let idx = id.0 as usize;
        debug_assert!(
            idx < self.buffer_ptrs.len(),
            "BufferId {idx} out of range (cache len {})",
            self.buffer_ptrs.len()
        );
        // SAFETY: in-bounds (debug-asserted); entries initialised in
        // `build_buffer_ptrs_cache`. Non-registered ids stay `null`,
        // which a subsequent kernel launch will refuse.
        unsafe { *self.buffer_ptrs.get_unchecked(idx) as *const c_void }
    }

    fn buf_mut_ptr(&mut self, id: BufferId) -> *mut c_void {
        let idx = id.0 as usize;
        debug_assert!(idx < self.buffer_ptrs.len());
        unsafe { *self.buffer_ptrs.get_unchecked(idx) }
    }

    fn weight_ptr(&self, w: &WeightRef) -> *const c_void {
        let slice = self
            .model
            .tensor_map
            .get(&w.name)
            .unwrap_or_else(|| panic!("tensor {} not in tensor_map", w.name));
        let base = self.model.arena.base_ptr();
        unsafe { slice.as_ptr(base) }
    }

    /// RmsNorm on the Q/K post-projection has length `head_dim`; the
    /// hidden-state norm has length `hidden_dim`. Inspect the buffer's
    /// own spec (one entry per buffer in the planner).
    fn rms_norm_len(&self, input: BufferId) -> usize {
        self.plan
            .specs
            .get(&input)
            .map(|s| s.elem_count)
            .unwrap_or(self.graph.config.hidden_dim)
    }

    /// Quality-Monitor hook: copy the post-output-norm hidden state
    /// from device to host. Synchronises the stream first so the
    /// caller sees the state **after** the last kernel in the
    /// current decode step, not a stale copy from the prior token.
    pub fn read_hidden_state(&mut self) -> HipResult<Vec<f32>> {
        self.stream.synchronize()?;
        self.read_buffer(self.graph.hidden_state_buffer, self.graph.config.hidden_dim)
    }

    fn read_buffer(&self, id: BufferId, elems: usize) -> HipResult<Vec<f32>> {
        let mut host_bytes = vec![0u8; elems * 4];
        let rc = unsafe {
            hipMemcpy(
                host_bytes.as_mut_ptr() as *mut _,
                self.buffers[&id].as_ptr(),
                elems * 4,
                hipMemcpyDeviceToHost,
            )
        };
        check(rc, "executor read_buffer")?;
        Ok(host_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Trace + NaN-guard check for a given node index. On any non-finite
    /// value, returns a descriptive error naming the node kind and
    /// output-buffer magnitude.
    fn check_node(&mut self, node_idx: usize) -> HipResult<()> {
        // Pick the first output buffer to inspect.
        let (kind, out_buf) = {
            let n = &self.graph.nodes[node_idx];
            let outs = n.outputs();
            (n.kind(), outs.first().copied())
        };
        let Some(out_id) = out_buf else {
            return Ok(());
        };
        let elems = self.plan.specs[&out_id].elem_count;
        let sample = self.read_buffer(out_id, elems)?;
        let (min, max, any_nan) = sample.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY, false),
            |(mn, mx, nan), &v| (mn.min(v), mx.max(v), nan | !v.is_finite()),
        );
        if self.trace {
            eprintln!(
                "[trace] node {:>3} {:<16} out=BufferId({}) elems={} min={:.3} max={:.3} nan={}",
                node_idx, kind, out_id.0, elems, min, max, any_nan
            );
        }
        if self.nan_guard && any_nan {
            return Err(HipError {
                code: -1,
                message: format!(
                    "non-finite after node {} ({}): min={} max={}",
                    node_idx, kind, min, max
                ),
                context: "nan_guard".into(),
            });
        }
        Ok(())
    }
}

// ─── WMMA-batched prefill (Phase 2 step 2.1.5) ──────────────────────────────

/// Minimum seq_len to use WMMA prefill. Below this, padding to
/// `WMMA_M_TILE` (64) wastes more work than the decode-loop path.
/// 16 is a conservative bar — at seq_len=16 the pad factor is 4×
/// but WMMA still wins because each GEMM is a single tile instead
/// of 16 GEMV dispatches.
pub const WMMA_PREFILL_MIN_SEQ_LEN: usize = 16;

/// WMMA row-tile used by all four Phase-1 dequant emitters
/// (`docs/v1.0/dequant_ir_spec.md §5.2`). `M`, `N` and `K` must each
/// be a multiple of 64 (and K a multiple of the format's
/// `elements_per_block` — enforced by the launcher itself).
const WMMA_M_TILE: usize = 64;

/// Per-node transient buffer allocation for one prefill call.
///
/// Decodes re-use `self.buffers` (sized for M=1) across tokens. The
/// WMMA-prefill path can't: its buffers are sized for
/// `padded_m × elem_count` and are freed at the end of the call.
/// Keeping them in a separate `HashMap` means the decode path's
/// buffer pool stays untouched — decode after prefill sees the
/// exact same buffers it saw before prefill.
struct PrefillBuffers {
    buffers: HashMap<BufferId, HipBuffer>,
    padded_m: usize,
    seq_len: usize,
}

impl PrefillBuffers {
    fn allocate(plan: &BufferPlan, seq_len: usize, padded_m: usize) -> HipResult<Self> {
        let mut buffers = HashMap::with_capacity(plan.specs.len());
        for (&id, spec) in &plan.specs {
            let bytes = spec.bytes(padded_m).max(4);
            buffers.insert(id, HipBuffer::new(bytes)?);
        }
        Ok(Self {
            buffers,
            padded_m,
            seq_len,
        })
    }

    fn ptr(&self, id: BufferId) -> *const c_void {
        self.buffers[&id].as_ptr()
    }

    fn mut_ptr(&mut self, id: BufferId) -> *mut c_void {
        self.buffers
            .get_mut(&id)
            .expect("prefill buffer")
            .as_mut_ptr()
    }
}

impl<'m> GraphExecutor<'m> {
    /// WMMA-batched prefill. Seeds the KV cache with `token_ids[..]`
    /// over the positions `[pos_offset, pos_offset + token_ids.len())`
    /// and returns the logits for the *last* token.
    ///
    /// Every matrix multiply runs as a WMMA GEMM with `M = padded_m`
    /// (next multiple of 64). Elementwise ops (RMSNorm, RoPE,
    /// residual, SwiGLU, embedding) use their existing `*_batched`
    /// variants. Attention uses `rocmforge_launch_attention_prefill`
    /// (causal, FP32, O(seq²) — the Phase-1 scaffold).
    ///
    /// Caller contract:
    ///   * `token_ids.len() >= WMMA_PREFILL_MIN_SEQ_LEN`
    ///   * Every `Gemm` weight in the graph is Q4_0 / Q4_K / Q6_K / Q8_0
    ///     (FP16 WMMA). Unsupported formats → `hipErrorInvalidValue`
    ///     with a descriptive message.
    ///   * All Gemm bias fields are `None` (Phase-1 WMMA path doesn't
    ///     fold bias).
    ///
    /// Failure modes any test should verify:
    ///   * Unsupported weight format → explicit error, no silent
    ///     wrong-math.
    ///   * KV cache too small for `pos_offset + seq_len` → error from
    ///     the first `kv_cache_append` whose pos exceeds `head_stride / head_dim`.
    pub fn execute_prefill_wmma(
        &mut self,
        token_ids: &[u32],
        pos_offset: usize,
    ) -> HipResult<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(HipError {
                code: -1,
                message: "execute_prefill_wmma: empty token_ids".into(),
                context: "prefill".into(),
            });
        }
        let seq_len = token_ids.len();
        let padded_m = ((seq_len + WMMA_M_TILE - 1) / WMMA_M_TILE) * WMMA_M_TILE;

        // Upload token_ids to a prefill-only u32 buffer. The graph's
        // `token_ids_buffer` is reused from the decode pool so we
        // don't disturb it; use the prefill pool's copy instead.
        let mut prefill = PrefillBuffers::allocate(&self.plan, seq_len, padded_m)?;
        let token_buf_id = self.graph.token_ids_buffer;
        let token_bytes = unsafe {
            std::slice::from_raw_parts(token_ids.as_ptr() as *const u8, token_ids.len() * 4)
        };
        prefill
            .buffers
            .get_mut(&token_buf_id)
            .expect("token ids buffer")
            .copy_from_host(token_bytes)?;

        // Walk every node with padded-M dispatch.
        let n_nodes = self.graph.nodes.len();
        for node_idx in 0..n_nodes {
            self.dispatch_prefill_node(node_idx, &mut prefill, pos_offset)?;
        }

        // Read back the last token's logits. Layout:
        // logits_buffer = [padded_m × vocab_size] f32. We want row
        // `seq_len - 1`.
        let vocab = self.graph.config.vocab_size;
        let row_bytes = vocab * 4;
        let last_row_offset = (seq_len - 1) * row_bytes;
        let logits_ptr = prefill.ptr(self.graph.logits_buffer) as *const u8;
        let mut host = vec![0u8; row_bytes];
        let rc = unsafe {
            hipMemcpy(
                host.as_mut_ptr() as *mut c_void,
                logits_ptr.add(last_row_offset) as *const c_void,
                row_bytes,
                hipMemcpyDeviceToHost,
            )
        };
        check(rc, "prefill logits readback")?;
        self.stream.synchronize()?;

        let logits: Vec<f32> = host
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (i, v) in logits.iter().enumerate() {
            if !v.is_finite() {
                return Err(HipError {
                    code: -1,
                    message: format!("prefill logits[{i}] = {v} (non-finite)"),
                    context: "execute_prefill_wmma".into(),
                });
            }
        }
        Ok(logits)
    }

    fn dispatch_prefill_node(
        &mut self,
        node_idx: usize,
        prefill: &mut PrefillBuffers,
        pos_offset: usize,
    ) -> HipResult<()> {
        // Snapshot the node reference. See the decode-path comment
        // in `dispatch_node` — we read constants only.
        let node = &self.graph.nodes[node_idx] as *const GraphNode;
        let node = unsafe { &*node };
        let seq_len = prefill.seq_len as i32;
        let padded_m = prefill.padded_m;

        match node {
            GraphNode::Embedding { input, output, .. } => {
                let table_ptr = self.embedding_fp32.as_ptr();
                let in_ptr = prefill.ptr(*input);
                let out_ptr = prefill.mut_ptr(*output);
                let rc = unsafe {
                    rocmforge_launch_embedding_lookup(
                        in_ptr as *const u32,
                        table_ptr as *const f32,
                        out_ptr as *mut f32,
                        seq_len,
                        self.graph.config.hidden_dim as i32,
                        self.stream.raw(),
                    )
                };
                check(rc, "prefill embedding")?;
            }
            GraphNode::RmsNorm {
                weight,
                input,
                output,
                eps,
                dim,
                num_rows,
            } => {
                let w_ptr = self.weight_ptr(weight);
                let in_ptr = prefill.ptr(*input);
                let out_ptr = prefill.mut_ptr(*output);
                // Main norm has num_rows=1 in the graph → prefill uses
                // seq_len rows. QK-norm has num_rows=n_heads/n_kv_heads
                // → prefill uses seq_len × num_rows rows.
                let total_rows = (*num_rows * padded_m) as i32;
                let rc = unsafe {
                    rocmforge_launch_rms_norm_batched(
                        in_ptr as *const f32,
                        w_ptr as *const f32,
                        out_ptr as *mut f32,
                        *dim as i32,
                        *eps,
                        total_rows,
                        self.stream.raw(),
                    )
                };
                check(rc, "prefill rms_norm")?;
            }
            GraphNode::Gemm {
                weight,
                bias,
                input,
                output,
                out_dim,
                in_dim,
            } => {
                if bias.is_some() {
                    return Err(HipError {
                        code: -1,
                        message: "prefill WMMA: Gemm bias not yet supported".into(),
                        context: "prefill gemm".into(),
                    });
                }
                let in_ptr = prefill.ptr(*input);
                let out_ptr = prefill.mut_ptr(*output);
                self.dispatch_prefill_wmma_gemm(
                    weight,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    padded_m,
                    *out_dim,
                    *in_dim,
                )?;
            }
            GraphNode::Rope {
                q_buffer,
                k_buffer,
                rope_freqs,
                n_heads,
                n_kv_heads,
                head_dim,
                theta_base,
            } => {
                let freq_ptr = rope_freqs
                    .as_ref()
                    .map(|w| self.weight_ptr(w) as *const f32)
                    .unwrap_or(std::ptr::null());
                let q_ptr = prefill.mut_ptr(*q_buffer);
                let k_ptr = prefill.mut_ptr(*k_buffer);
                let rc = unsafe {
                    rocmforge_launch_rope_batched(
                        q_ptr as *mut f32,
                        pos_offset as i32,
                        *n_heads as i32,
                        *head_dim as i32,
                        *theta_base,
                        seq_len,
                        freq_ptr,
                        self.stream.raw(),
                    )
                };
                check(rc, "prefill rope Q")?;
                let rc = unsafe {
                    rocmforge_launch_rope_batched(
                        k_ptr as *mut f32,
                        pos_offset as i32,
                        *n_kv_heads as i32,
                        *head_dim as i32,
                        *theta_base,
                        seq_len,
                        freq_ptr,
                        self.stream.raw(),
                    )
                };
                check(rc, "prefill rope K")?;
            }
            GraphNode::KvCacheAppend {
                k_buffer,
                v_buffer,
                layer_idx,
            } => {
                // Loop seq_len times — simple but adds 2 × seq_len
                // dispatches per layer. For a 33-token Qwen3-8B prompt
                // that's 33 × 2 × 36 = 2376 extra dispatches, ~2 ms
                // total at 1 µs/dispatch. Worth a batched kernel in a
                // follow-up if prefill becomes the next bottleneck.
                // The row stride always addresses FP32 source rows —
                // k_base/v_base hold fresh K/V from the QKV projection
                // and are independent of the KV-cache element size.
                let k_base = prefill.ptr(*k_buffer) as *const u8;
                let v_base = prefill.ptr(*v_buffer) as *const u8;
                let k_cache = self.k_cache[*layer_idx].as_mut_ptr();
                let v_cache = self.v_cache[*layer_idx].as_mut_ptr();
                let row_stride_k =
                    (self.kv_layout.num_kv_heads * self.kv_layout.head_dim * 4) as isize;
                for i in 0..prefill.seq_len {
                    let pos = pos_offset + i;
                    let k_row = unsafe { k_base.offset(i as isize * row_stride_k) };
                    let v_row = unsafe { v_base.offset(i as isize * row_stride_k) };
                    let rc = match self.kv_layout.precision {
                        KvPrecision::Fp32 => unsafe {
                            rocmforge_launch_kv_cache_append(
                                k_cache as *mut f32,
                                v_cache as *mut f32,
                                k_row as *const f32,
                                v_row as *const f32,
                                self.kv_layout.num_kv_heads as i32,
                                self.kv_layout.head_dim as i32,
                                pos as i32,
                                self.kv_layout.head_stride as i32,
                                self.stream.raw(),
                            )
                        },
                        KvPrecision::Fp8E5M2 => unsafe {
                            rocmforge_launch_kv_cache_append_fp8(
                                k_cache,
                                v_cache,
                                k_row as *const f32,
                                v_row as *const f32,
                                self.kv_layout.num_kv_heads as i32,
                                self.kv_layout.head_dim as i32,
                                pos as i32,
                                self.kv_layout.head_stride as i32,
                                self.stream.raw(),
                            )
                        },
                    };
                    check(rc, "prefill kv_cache_append")?;
                }
            }
            GraphNode::Attention {
                q_buffer,
                output,
                layer_idx: _,
                n_heads,
                n_kv_heads,
                head_dim,
            } => {
                // attention_prefill runs on fresh Q/K/V in transient
                // buffers — it does NOT read from the KV cache. The
                // cache is populated by the preceding KvCacheAppend
                // node and is *only* needed for subsequent decode.
                // The prefill kernel is therefore self-contained.
                //
                // But: the graph's `Attention` node only exposes
                // `q_buffer`, not k/v. Find them via the preceding
                // `KvCacheAppend` node's buffers.
                let (k_id, v_id) = self.find_kv_buffers_for_attention(node_idx)?;
                let q_ptr = prefill.ptr(*q_buffer);
                let k_ptr = prefill.ptr(k_id);
                let v_ptr = prefill.ptr(v_id);
                let out_ptr = prefill.mut_ptr(*output);
                let scale = 1.0f32 / (*head_dim as f32).sqrt();
                let rc = unsafe {
                    rocmforge_launch_attention_prefill(
                        q_ptr as *const f32,
                        k_ptr as *const f32,
                        v_ptr as *const f32,
                        out_ptr as *mut f32,
                        seq_len,
                        *n_heads as i32,
                        *n_kv_heads as i32,
                        *head_dim as i32,
                        scale,
                        self.stream.raw(),
                    )
                };
                check(rc, "prefill attention")?;
            }
            GraphNode::ResidualAdd { a, b } => {
                let a_ptr = prefill.mut_ptr(*a);
                let b_ptr = prefill.ptr(*b);
                let per_row = self.plan.specs[a].elem_count as i32;
                let n = per_row * padded_m as i32;
                let rc = unsafe {
                    rocmforge_launch_residual_add_inplace(
                        a_ptr as *mut f32,
                        b_ptr as *const f32,
                        n,
                        self.stream.raw(),
                    )
                };
                check(rc, "prefill residual_add")?;
            }
            GraphNode::GateUpSwiGLU {
                gate_weight,
                up_weight,
                input,
                output,
                ffn_dim,
                hidden_dim,
            } => {
                // Unfuse for prefill (§5.3): two WMMAs + SwiGLU.
                // The scratch buffers come out of the regular transient
                // pool — we allocate them fresh since they aren't
                // BufferIds in the plan.
                let scratch_bytes = padded_m * *ffn_dim * 4;
                let mut gate_scratch = HipBuffer::new(scratch_bytes)?;
                let mut up_scratch = HipBuffer::new(scratch_bytes)?;
                let in_ptr = prefill.ptr(*input) as *const f32;

                self.dispatch_prefill_wmma_gemm(
                    gate_weight,
                    in_ptr,
                    gate_scratch.as_mut_ptr() as *mut f32,
                    padded_m,
                    *ffn_dim,
                    *hidden_dim,
                )?;
                self.dispatch_prefill_wmma_gemm(
                    up_weight,
                    in_ptr,
                    up_scratch.as_mut_ptr() as *mut f32,
                    padded_m,
                    *ffn_dim,
                    *hidden_dim,
                )?;

                let out_ptr = prefill.mut_ptr(*output);
                let n = (padded_m * *ffn_dim) as i32;
                let rc = unsafe {
                    rocmforge_launch_swiglu(
                        gate_scratch.as_ptr() as *const f32,
                        up_scratch.as_ptr() as *const f32,
                        out_ptr as *mut f32,
                        n,
                        self.stream.raw(),
                    )
                };
                check(rc, "prefill swiglu")?;
            }
            GraphNode::SwiGLU {
                gate_buffer,
                up_buffer,
                output,
            } => {
                let g_ptr = prefill.ptr(*gate_buffer);
                let u_ptr = prefill.ptr(*up_buffer);
                let o_ptr = prefill.mut_ptr(*output);
                let per_row = self.plan.specs[output].elem_count as i32;
                let n = per_row * padded_m as i32;
                let rc = unsafe {
                    rocmforge_launch_swiglu(
                        g_ptr as *const f32,
                        u_ptr as *const f32,
                        o_ptr as *mut f32,
                        n,
                        self.stream.raw(),
                    )
                };
                check(rc, "prefill swiglu")?;
            }
            GraphNode::FusedGemmResidual {
                weight,
                input,
                residual,
                out_dim,
                in_dim,
            } => {
                // Unfuse: WMMA writes to a scratch, residual_add
                // folds it into `residual` in-place.
                let scratch_bytes = padded_m * *out_dim * 4;
                let mut scratch = HipBuffer::new(scratch_bytes)?;
                let in_ptr = prefill.ptr(*input) as *const f32;
                self.dispatch_prefill_wmma_gemm(
                    weight,
                    in_ptr,
                    scratch.as_mut_ptr() as *mut f32,
                    padded_m,
                    *out_dim,
                    *in_dim,
                )?;
                let res_ptr = prefill.mut_ptr(*residual);
                let n = (padded_m * *out_dim) as i32;
                let rc = unsafe {
                    rocmforge_launch_residual_add_inplace(
                        res_ptr as *mut f32,
                        scratch.as_ptr() as *const f32,
                        n,
                        self.stream.raw(),
                    )
                };
                check(rc, "prefill fused_residual")?;
            }
        }
        Ok(())
    }

    /// Pick the matching WMMA launcher based on the weight's quant
    /// format. All four Phase-1 kernels share the same argument
    /// order `(input, weights, output, M, N, K, stream)`.
    fn dispatch_prefill_wmma_gemm(
        &mut self,
        weight: &WeightRef,
        input: *const f32,
        output: *mut f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        let weights = self.weight_ptr(weight) as *const u8;
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let stream = self.stream.raw();

        // Integer-WMMA MMQ opt-in path (Q4_K + FP16 precision only).
        // Pre-quantises the FP32 activation tile into block_q8_1_mmq and
        // dispatches the integer-WMMA kernel. Alignment requirements
        // (M % 16, N % 16, K % 256) are already satisfied by the prefill
        // padding (M to 64, N/K Qwen3 shapes) so no extra guards needed.
        if self.prefill_mmq_enabled
            && weight.format == GgmlType::Q4_K
            && self.prefill_precision == PrefillPrecision::Fp16
            && (m % 16) == 0 && (n % 16) == 0 && (k % 256) == 0
        {
            let mmq_bytes = ((m * k) / 128) * 144;
            let need_grow = self
                .mmq_activation_buffer
                .as_ref()
                .map(|b| b.size() < mmq_bytes)
                .unwrap_or(true);
            if need_grow {
                self.mmq_activation_buffer = Some(HipBuffer::new(mmq_bytes)?);
            }
            let mmq_buf = self.mmq_activation_buffer.as_mut().expect("mmq buf");
            let rc = unsafe {
                rocmforge_launch_quantize_q8_1_mmq(
                    input,
                    mmq_buf.as_mut_ptr(),
                    (m * k) as i32,
                    stream,
                )
            };
            check(rc, "prefill quantize_q8_1_mmq")?;
            let rc = unsafe {
                rocmforge_launch_mmq_q4_k(
                    weights as *const core::ffi::c_void,
                    mmq_buf.as_ptr(),
                    output,
                    m_i, n_i, k_i,
                    stream,
                )
            };
            return check(rc, "prefill mmq_q4_k");
        }

        let rc = unsafe {
            match (weight.format, self.prefill_precision) {
                (GgmlType::Q4_K, PrefillPrecision::Fp16) => rocmforge_launch_wmma_gemm_q4_k_fp16(
                    input, weights, output, m_i, n_i, k_i, stream,
                ),
                (GgmlType::Q4_K, PrefillPrecision::Fp8) => rocmforge_launch_wmma_gemm_q4_k_fp8(
                    input, weights, output, m_i, n_i, k_i, stream,
                ),
                (GgmlType::Q6_K, PrefillPrecision::Fp16) => rocmforge_launch_wmma_gemm_q6_k_fp16(
                    input, weights, output, m_i, n_i, k_i, stream,
                ),
                (GgmlType::Q6_K, PrefillPrecision::Fp8) => rocmforge_launch_wmma_gemm_q6_k_fp8(
                    input, weights, output, m_i, n_i, k_i, stream,
                ),
                (GgmlType::Q4_0, PrefillPrecision::Fp16) => rocmforge_launch_wmma_gemm_q4_0_fp16(
                    input, weights, output, m_i, n_i, k_i, stream,
                ),
                (GgmlType::Q4_0, PrefillPrecision::Fp8) => rocmforge_launch_wmma_gemm_q4_0_fp8(
                    input, weights, output, m_i, n_i, k_i, stream,
                ),
                (GgmlType::Q8_0, PrefillPrecision::Fp16) => rocmforge_launch_wmma_gemm_q8_0_fp16(
                    input, weights, output, m_i, n_i, k_i, stream,
                ),
                (GgmlType::Q8_0, PrefillPrecision::Fp8) => rocmforge_launch_wmma_gemm_q8_0_fp8(
                    input, weights, output, m_i, n_i, k_i, stream,
                ),
                (other, _) => {
                    return Err(HipError {
                        code: -1,
                        message: format!("prefill WMMA: unsupported weight format {other:?}"),
                        context: "wmma_gemm".into(),
                    })
                }
            }
        };
        check(rc, "prefill wmma_gemm")
    }

    /// Find the `(k_buffer, v_buffer)` that feed into the Attention
    /// node at `attention_idx`. Walks backwards to the preceding
    /// `KvCacheAppend`, which holds both references.
    fn find_kv_buffers_for_attention(
        &self,
        attention_idx: usize,
    ) -> HipResult<(BufferId, BufferId)> {
        for i in (0..attention_idx).rev() {
            if let GraphNode::KvCacheAppend {
                k_buffer, v_buffer, ..
            } = &self.graph.nodes[i]
            {
                return Ok((*k_buffer, *v_buffer));
            }
        }
        Err(HipError {
            code: -1,
            message: format!(
                "prefill: no KvCacheAppend found before Attention at node {attention_idx}"
            ),
            context: "find_kv_buffers".into(),
        })
    }
}

/// Construct the `BufferId → *mut c_void` lookup table used by the
/// decode hot-path. Length = max `BufferId.0` + 1 across `buffers`;
/// entries for unregistered ids stay null. Called once after the
/// `buffers` HashMap is populated in `new()`.
fn build_buffer_ptrs_cache(buffers: &HashMap<BufferId, HipBuffer>) -> Vec<*mut c_void> {
    let max_id = buffers.keys().map(|id| id.0).max().unwrap_or(0);
    let mut cache = vec![std::ptr::null_mut::<c_void>(); (max_id as usize) + 1];
    for (id, buf) in buffers {
        // HipBuffer::as_ptr() returns *const c_void; the decode
        // path casts back to *mut where needed (buf_mut_ptr).
        // Store as *mut so the cache is uniform.
        cache[id.0 as usize] = buf.as_ptr() as *mut c_void;
    }
    cache
}

// ─── HIP-Graph decode integration (Option C) ────────────────────────────────
//
// Captures the decode forward pass once (at the first decode token after
// the Bandit has converged on every shape) and replays it for every
// subsequent token. Pre-token work that cannot live inside the graph —
// the `copy_from_host` that writes `token_id` to the token-ids buffer —
// runs right before `hipGraphLaunch`.
//
// Only three kernel parameters vary per token:
//   * `pos` in `rope` (Q and K launches) — arg index 1
//   * `pos` in `kv_cache_append`         — arg index 6
//   * `seq_len` (= pos + 1) in `attention_decode` — arg index 7
//
// They are driven from two heap-stable `Box<i32>` slots so the address
// we pass to `hipGraphExecKernelNodeSetParams` doesn't move.

/// Per-token-mutable slots + captured kernel-node handles.
pub struct HipGraphDecodeCache {
    graph: hipGraph_t,
    exec: hipGraphExec_t,
    /// Persistent storage for the `pos` argument (Rope + KvCacheAppend).
    kv_pos_slot: Box<i32>,
    /// Persistent storage for the `seq_len` argument (Attention).
    seq_len_slot: Box<i32>,
    /// Kernel nodes that take `pos` as arg index 1 (Rope Q + K).
    rope_nodes: Vec<hipGraphNode_t>,
    /// Kernel nodes that take `pos` as arg index 6 (KvCacheAppend).
    kv_write_nodes: Vec<hipGraphNode_t>,
    /// Kernel nodes that take `seq_len` as arg index 7 (Attention).
    attention_nodes: Vec<hipGraphNode_t>,
}

impl HipGraphDecodeCache {
    /// Tear down graph + exec handles. Called from `Drop` and from
    /// `invalidate_fast_dispatch`.
    fn destroy(self) {
        unsafe {
            if !self.exec.is_null() {
                let _ = hipGraphExecDestroy(self.exec);
            }
            if !self.graph.is_null() {
                let _ = hipGraphDestroy(self.graph);
            }
        }
        // kv_pos_slot / seq_len_slot dropped normally — just Box<i32>.
    }
}

impl Drop for HipGraphDecodeCache {
    fn drop(&mut self) {
        unsafe {
            if !self.exec.is_null() {
                let _ = hipGraphExecDestroy(self.exec);
            }
            if !self.graph.is_null() {
                let _ = hipGraphDestroy(self.graph);
            }
        }
    }
}

/// Per-kernel arg signatures that the replay path patches.
const ROPE_POS_ARG_IDX: usize = 1;
const ROPE_N_ARGS: usize = 6;

const KV_WRITE_POS_ARG_IDX: usize = 6;
const KV_WRITE_N_ARGS: usize = 8;

const ATTENTION_SEQ_LEN_ARG_IDX: usize = 7;
const ATTENTION_N_ARGS: usize = 10;

impl<'m> GraphExecutor<'m> {
    /// Return `true` if HIP-Graph decode capture should be attempted
    /// right now.
    fn should_capture_hip_graph(&self) -> bool {
        if std::env::var("ROCMFORGE_DISABLE_HIP_GRAPH").ok().as_deref() == Some("1") {
            return false;
        }
        let converged = self
            .runtime
            .as_ref()
            .map(|r| r.all_exploiting())
            .unwrap_or(false);
        converged && self.hip_graph.is_none()
    }

    /// Count, per `GraphNode`, how many HIP kernel launches that node
    /// dispatches. The returned vector has `graph.nodes.len()` entries;
    /// each is the running launch-index base for that node. Used to
    /// translate per-node kernel kinds into indices into the captured
    /// graph's flat node array.
    fn launch_index_spans(&self) -> LaunchSpans {
        let mut spans = LaunchSpans::default();
        let mut idx = 0usize;

        // Schritt 5: simulate the Q8_1 sharing cache during the walk.
        // Every MMVQ dispatch checks `last_q8_1_input_id == Some(input)`;
        // a hit skips the quantize_q8_1 launch (1 launch instead of 2).
        // `capture_decode_graph` resets this cache before calling
        // `launch_index_spans` so both simulations start in the same
        // state. Disabled when `ROCMFORGE_DISABLE_Q8_1_SHARING=1`.
        let sharing_on = self.q8_1_sharing_enabled;
        let mut sim_q8_1_input: Option<BufferId> = None;

        for (node_idx, node) in self.graph.nodes.iter().enumerate() {
            match node {
                GraphNode::Embedding { .. } => idx += 1,
                GraphNode::RmsNorm { .. } => idx += 1,
                GraphNode::Gemm { input, .. } => {
                    // MMVQ variants emit `quantize_q8_1 + mmvq` = 2
                    // launches on a cache miss, but only `mmvq` = 1 on a
                    // cache hit (Q → K → V pattern). Non-MMVQ kernels
                    // always emit 1 launch and don't touch the cache.
                    let committed = self
                        .node_fast_cache
                        .as_ref()
                        .and_then(|cache| cache.get(node_idx).copied())
                        .and_then(|e| e.committed_kernel);
                    let is_mmvq = matches!(
                        committed,
                        Some(KernelId::GemvQ4KMmvq) | Some(KernelId::GemvQ6KMmvq)
                    );
                    if is_mmvq {
                        let cache_hit = sharing_on && sim_q8_1_input == Some(*input);
                        if cache_hit {
                            idx += 1; // mmvq only, quantize skipped
                        } else {
                            idx += 2; // quantize + mmvq
                            sim_q8_1_input = Some(*input);
                        }
                    } else {
                        idx += 1;
                    }
                }
                GraphNode::Rope { .. } => {
                    // Two launches: Q then K.
                    spans.rope.push(idx);
                    spans.rope.push(idx + 1);
                    idx += 2;
                }
                GraphNode::KvCacheAppend { .. } => {
                    spans.kv_write.push(idx);
                    idx += 1;
                }
                GraphNode::Attention { .. } => {
                    spans.attention.push(idx);
                    idx += 1;
                }
                GraphNode::ResidualAdd { a, .. } => {
                    idx += 1;
                    // In-place write on `a`; if it was the currently
                    // cached Q8_1 input, invalidate to stay consistent
                    // with the dispatch-time invalidation above.
                    if sim_q8_1_input == Some(*a) {
                        sim_q8_1_input = None;
                    }
                }
                GraphNode::GateUpSwiGLU { gate_weight, up_weight, input, .. } => {
                    // Launch count depends on which dispatch path runs
                    // (see `dispatch_gate_up_swiglu`):
                    //   * Dynamic GA hook (shape-match):       1 launch
                    //   * Legacy `fused_gate_up_swiglu`:       1 launch
                    //   * MMVQ-fused (Phase 2 Schritt 3):      2 launches
                    //     (quantize_q8_1 + mmvq_fused); with Schritt 5
                    //     Q8_1 sharing, drops to 1 launch when the
                    //     previous MMVQ already quantized this input.
                    //   * Unfused (pre-MMVQ / disable env):    3 launches
                    //   * Non-Q4_K fallback:                   3 launches
                    let has_dyn = self
                        .gate_up_dynamic
                        .as_ref()
                        .map(|h| h.hidden_dim > 0 && h.ffn_dim > 0)
                        .unwrap_or(false);
                    let q4k_both = gate_weight.format == GgmlType::Q4_K
                        && up_weight.format == GgmlType::Q4_K;
                    let mmvq_fusion_active = q4k_both
                        && !has_dyn
                        && !self.fused_gate_up
                        && std::env::var("ROCMFORGE_DISABLE_MMVQ_FUSION")
                            .ok()
                            .as_deref()
                            != Some("1");
                    if has_dyn || self.fused_gate_up {
                        idx += 1;
                    } else if mmvq_fusion_active {
                        let cache_hit = sharing_on && sim_q8_1_input == Some(*input);
                        if cache_hit {
                            idx += 1;
                        } else {
                            idx += 2;
                            sim_q8_1_input = Some(*input);
                        }
                    } else {
                        idx += 3;
                    }
                }
                GraphNode::SwiGLU { .. } => idx += 1,
                GraphNode::FusedGemmResidual { weight, input, .. } => {
                    // 2026-04-24: Q4_K FusedGemmResidual routes through
                    // the MMVQ-residual kernel (quantize_q8_1 + mmvq_res
                    // = 2 launches) unless the env override flips back
                    // to the 1-launch q8_inline_residual. With Schritt 5
                    // Q8_1 sharing, drops to 1 launch on a cache hit.
                    // After dispatch the cache is invalidated because
                    // the kernel writes the residual buffer in-place.
                    let mmvq_path = weight.format == GgmlType::Q4_K
                        && std::env::var("ROCMFORGE_DISABLE_MMVQ_RESIDUAL")
                            .ok()
                            .as_deref()
                            != Some("1");
                    if mmvq_path {
                        let cache_hit = sharing_on && sim_q8_1_input == Some(*input);
                        if cache_hit {
                            idx += 1;
                        } else {
                            idx += 2;
                            sim_q8_1_input = Some(*input);
                        }
                        // In-place residual write invalidates cache.
                        sim_q8_1_input = None;
                    } else {
                        idx += 1;
                    }
                }
            }
        }
        spans.total_launches = idx;
        spans
    }

    /// Capture the decode forward pass into a HIP-graph.
    ///
    /// Called from `execute_decode` on the first eligible call. Runs
    /// the same dispatch path that the legacy code takes, but wrapped
    /// in `hipStreamBeginCapture` / `hipStreamEndCapture`. After the
    /// capture the function instantiates the graph, extracts the
    /// updatable node handles, and (crucially) launches the exec
    /// once to actually produce the current token's output (capture
    /// itself is observation-only).
    fn capture_decode_graph(&mut self, token_id: u32, pos: usize) -> HipResult<Vec<f32>> {
        // 1) Seed the token-ids buffer BEFORE entering capture. The
        //    `copy_from_host` inside `hipStreamBeginCapture` would be
        //    recorded with a stale host-stack source pointer.
        let tok_bytes = token_id.to_le_bytes();
        self.buffers
            .get_mut(&self.graph.token_ids_buffer)
            .expect("token-ids buffer missing")
            .copy_from_host(&tok_bytes)?;

        // 1a) Reset the Q8_1 sharing cache so the capture-walk sees
        //     the same initial state as `launch_index_spans`'s
        //     simulation. If we enter capture with the cache set from
        //     a prior execute_decode call, the simulated count and the
        //     actual captured count would disagree and the validation
        //     check below fires, bailing to legacy dispatch.
        self.last_q8_1_input_id = None;

        // 2) Start capture.
        let stream = self.stream.raw();
        let rc = unsafe { hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) };
        check(rc, "hipStreamBeginCapture")?;

        // 3) Dispatch every graph node. These calls record into the
        //    capture instead of actually launching. Any error mid-
        //    capture MUST still call EndCapture; we defer via a
        //    per-call ok/err tracker.
        let mut dispatch_err: Option<HipError> = None;
        for i in 0..self.graph.nodes.len() {
            if let Err(e) = self.dispatch_node(i, pos) {
                dispatch_err = Some(e);
                break;
            }
        }

        // 4) End capture, even on dispatch error, to bring the stream
        //    back to normal mode.
        let mut graph: hipGraph_t = std::ptr::null_mut();
        let rc_end = unsafe { hipStreamEndCapture(stream, &mut graph) };
        if let Some(e) = dispatch_err {
            if !graph.is_null() {
                unsafe {
                    let _ = hipGraphDestroy(graph);
                }
            }
            return Err(e);
        }
        check(rc_end, "hipStreamEndCapture")?;
        if graph.is_null() {
            return Err(HipError {
                code: -1,
                message: "EndCapture returned null graph".into(),
                context: "hipStreamEndCapture".into(),
            });
        }

        // 5) Enumerate captured nodes in topological order.
        let mut n_nodes: usize = 0;
        let rc = unsafe { hipGraphGetNodes(graph, std::ptr::null_mut(), &mut n_nodes) };
        check(rc, "hipGraphGetNodes (count)")?;
        let mut nodes: Vec<hipGraphNode_t> = vec![std::ptr::null_mut(); n_nodes];
        let rc = unsafe { hipGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n_nodes) };
        check(rc, "hipGraphGetNodes (data)")?;
        nodes.truncate(n_nodes);

        // 6) Match per-node kernel-kinds against the captured nodes by
        //    index. The spans are built from the builder graph + our
        //    dispatch rules, and topological order of a single-stream
        //    capture equals capture order.
        let spans = self.launch_index_spans();
        if spans.total_launches != n_nodes {
            // Node count mismatch — defensively bail out. The actual
            // HIP-Graph node count may include one copyBuffer or
            // similar we didn't account for. Falling back to the
            // legacy path keeps correctness.
            unsafe {
                let _ = hipGraphDestroy(graph);
            }
            return Err(HipError {
                code: -1,
                message: format!(
                    "HIP-Graph capture: expected {} kernel nodes, got {} — falling back to legacy dispatch",
                    spans.total_launches, n_nodes
                ),
                context: "launch_index_spans".into(),
            });
        }

        let rope_nodes: Vec<_> = spans.rope.iter().map(|&i| nodes[i]).collect();
        let kv_write_nodes: Vec<_> = spans.kv_write.iter().map(|&i| nodes[i]).collect();
        let attention_nodes: Vec<_> = spans.attention.iter().map(|&i| nodes[i]).collect();

        // 7) Instantiate.
        let mut exec: hipGraphExec_t = std::ptr::null_mut();
        let mut err_node: hipGraphNode_t = std::ptr::null_mut();
        let rc = unsafe {
            hipGraphInstantiate(&mut exec, graph, &mut err_node, std::ptr::null_mut(), 0)
        };
        if rc != HIP_SUCCESS {
            unsafe {
                let _ = hipGraphDestroy(graph);
            }
            return Err(HipError {
                code: rc,
                message: format!("hipGraphInstantiate failed (err_node={:?})", err_node),
                context: "capture".into(),
            });
        }

        // 8) Build the cache. Slot values get overwritten each token;
        //    initial seeding here means the first patch-replay sees
        //    the right values.
        let cache = HipGraphDecodeCache {
            graph,
            exec,
            kv_pos_slot: Box::new(pos as i32),
            seq_len_slot: Box::new((pos + 1) as i32),
            rope_nodes,
            kv_write_nodes,
            attention_nodes,
        };
        self.hip_graph = Some(cache);

        // 9) Capture did NOT execute the kernels — launch the exec
        //    now to produce this token's output.
        self.replay_decode_graph(token_id, pos)
    }

    /// Per-token replay path: patch the three variable params, launch
    /// the exec, sync, read logits.
    fn replay_decode_graph(&mut self, token_id: u32, pos: usize) -> HipResult<Vec<f32>> {
        // Token-id memcpy outside the graph.
        let tok_bytes = token_id.to_le_bytes();
        self.buffers
            .get_mut(&self.graph.token_ids_buffer)
            .expect("token-ids buffer missing")
            .copy_from_host(&tok_bytes)?;

        // Update persistent slots. Because SetParams is a copy-in
        // operation, the slot pointers only need to be valid for the
        // duration of each SetParams call — but using stable heap
        // slots keeps the pointer arithmetic simple and mirrors how
        // llama.cpp / nvcc-style graph replays tend to be written.
        let cache = self
            .hip_graph
            .as_mut()
            .expect("replay_decode_graph called with no cache");
        *cache.kv_pos_slot = pos as i32;
        *cache.seq_len_slot = (pos + 1) as i32;

        let kv_pos_ptr: *mut c_void = &mut *cache.kv_pos_slot as *mut i32 as *mut c_void;
        let seq_len_ptr: *mut c_void = &mut *cache.seq_len_slot as *mut i32 as *mut c_void;

        // Helper to patch one node. Reads the HIP-owned params, builds
        // a local copy with one pointer swapped, then writes it back
        // with SetParams. All pointer math lives for exactly one
        // SetParams call.
        let patch_one = |exec: hipGraphExec_t,
                         node: hipGraphNode_t,
                         n_args: usize,
                         arg_idx: usize,
                         new_ptr: *mut c_void|
         -> HipResult<()> {
            let mut p = hipKernelNodeParams::default();
            let rc = unsafe { hipGraphKernelNodeGetParams(node, &mut p) };
            check(rc, "hipGraphKernelNodeGetParams")?;
            // Copy the HIP-owned pointer array into our own stack-
            // local Vec so we can swap a single entry without
            // mutating HIP-owned memory.
            let mut new_args: Vec<*mut c_void> = Vec::with_capacity(n_args);
            for i in 0..n_args {
                let p_i = unsafe { *p.kernelParams.add(i) };
                new_args.push(p_i);
            }
            new_args[arg_idx] = new_ptr;
            let new_params = hipKernelNodeParams {
                kernelParams: new_args.as_mut_ptr(),
                ..p
            };
            let rc = unsafe { hipGraphExecKernelNodeSetParams(exec, node, &new_params) };
            check(rc, "hipGraphExecKernelNodeSetParams")?;
            Ok(())
        };

        for &node in &cache.rope_nodes {
            patch_one(cache.exec, node, ROPE_N_ARGS, ROPE_POS_ARG_IDX, kv_pos_ptr)?;
        }
        for &node in &cache.kv_write_nodes {
            patch_one(
                cache.exec,
                node,
                KV_WRITE_N_ARGS,
                KV_WRITE_POS_ARG_IDX,
                kv_pos_ptr,
            )?;
        }
        for &node in &cache.attention_nodes {
            patch_one(
                cache.exec,
                node,
                ATTENTION_N_ARGS,
                ATTENTION_SEQ_LEN_ARG_IDX,
                seq_len_ptr,
            )?;
        }

        // Launch the exec and read logits.
        let stream = self.stream.raw();
        let rc = unsafe { hipGraphLaunch(cache.exec, stream) };
        check(rc, "hipGraphLaunch")?;
        self.stream.synchronize()?;

        let logits = self.read_buffer(self.graph.logits_buffer, self.graph.config.vocab_size)?;
        self.flush_event_pool()?;
        for (i, v) in logits.iter().enumerate() {
            if !v.is_finite() {
                return Err(HipError {
                    code: -1,
                    message: format!("replay logits[{i}] = {v} (non-finite)"),
                    context: "replay_decode_graph".into(),
                });
            }
        }
        Ok(logits)
    }
}

#[derive(Default)]
struct LaunchSpans {
    rope: Vec<usize>,
    kv_write: Vec<usize>,
    attention: Vec<usize>,
    total_launches: usize,
}
