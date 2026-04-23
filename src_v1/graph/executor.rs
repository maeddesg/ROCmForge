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
    rocmforge_launch_attention_decode, rocmforge_launch_kv_cache_append,
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
    rocmforge_launch_gemv_q4_k_q8_inline, rocmforge_launch_gemv_q4_k_q8_inline_residual,
    rocmforge_launch_gemv_q4_k_standard, rocmforge_launch_gemv_q6_k_standard,
    rocmforge_launch_gemv_q8_0_standard,
};
use super::super::backend::gpu::hip_ffi::{
    hipMemcpy, hipMemcpyDeviceToDevice, hipMemcpyDeviceToHost,
};
use super::super::backend::gpu::wmma::{
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
use super::buffer_plan::{BufferPlan, KvCacheLayout};
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
        let kv_layout = KvCacheLayout::from_config(&graph.config, max_seq);

        // Transient buffers — sized for one token (decode).
        let mut buffers: HashMap<BufferId, HipBuffer> = HashMap::new();
        for (&id, spec) in &plan.specs {
            let bytes = spec.bytes(1).max(4); // never zero-size
            buffers.insert(id, HipBuffer::new(bytes)?);
        }

        // KV cache — one allocation per (layer, K or V).
        let per_cache = kv_layout.num_kv_heads * kv_layout.head_stride * 4;
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
        })
    }

    /// Force the fused (pre-un-fusing) `gate_up_swiglu` kernel. Used
    /// by the A/B-regression tests; the default after the post-2.1.5
    /// follow-up is the un-fused path.
    pub fn set_fused_gate_up(&mut self, fused: bool) {
        self.fused_gate_up = fused;
    }

    pub fn fused_gate_up(&self) -> bool {
        self.fused_gate_up
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
        let per_cache_bytes = self.kv_layout.num_kv_heads * self.kv_layout.head_stride * 4;
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
        // Seed the token-ids buffer.
        let tok_bytes = token_id.to_le_bytes();
        self.buffers
            .get_mut(&self.graph.token_ids_buffer)
            .expect("token-ids buffer missing")
            .copy_from_host(&tok_bytes)?;

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
                self.dispatch_gemv(weight, *input, *output, *out_dim, *in_dim)?;
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
                let rc = unsafe {
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
                let rc = unsafe {
                    rocmforge_launch_attention_decode(
                        q_ptr as *const f32,
                        k_cache as *const f32,
                        v_cache as *const f32,
                        out_ptr as *mut f32,
                        *n_heads as i32,
                        *n_kv_heads as i32,
                        *head_dim as i32,
                        (pos + 1) as i32, // seq_len so far
                        self.kv_layout.head_stride as i32,
                        scale,
                        self.stream.raw(),
                    )
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
    ) -> HipResult<()> {
        // Path 1 — Bandit attached: ask which kernel to launch,
        // time the launch + sync, and feed the time back. The Bandit
        // uses wall-clock around the synchronize() call; Phase-2
        // replaces this with HIP-event timing batched at token end
        // so the extra sync goes away.
        if self.runtime.is_some() {
            return self.dispatch_gemv_tuned(weight, input, output, out_dim, in_dim);
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
    ) -> HipResult<()> {
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

        let w_ptr = self.weight_ptr(weight);
        let in_ptr = self.buf_ptr(input);
        let out_ptr = self.buf_mut_ptr(output);
        let stream_raw = self.stream.raw();

        let pair_idx = if want_timing {
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
                KernelId::GemvQ6KStandard => rocmforge_launch_gemv_q6_k_standard(
                    w_ptr as *const u8,
                    in_ptr as *const f32,
                    out_ptr as *mut f32,
                    in_dim as i32,
                    out_dim as i32,
                    stream_raw,
                ),
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
        let rc = unsafe {
            rocmforge_launch_gemv_q4_k_q8_inline_residual(
                w_ptr as *const u8,
                in_ptr as *const f32,
                // The fused kernel reads the old residual then
                // writes `residual + gemv` back to the same buffer.
                residual_ptr as *const f32,
                residual_ptr as *mut f32,
                in_dim as i32,
                out_dim as i32,
                self.stream.raw(),
            )
        };
        check(rc, "gemv_q4_k_q8_inline_residual")?;
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

            // ── Fused path (legacy) ──────────────────────────────
            // The fused `gemv_q4_k_gate_up_swiglu` kernel hits only
            // 20 % of the 640 GB/s peak BW on gfx1201 (rocprof
            // deep-dive 2026-04-23, 65 % of decode time sits here).
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
        self.buffers[&id].as_ptr()
    }

    fn buf_mut_ptr(&mut self, id: BufferId) -> *mut c_void {
        self.buffers.get_mut(&id).expect("buffer").as_mut_ptr()
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
                    let rc = unsafe {
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
        &self,
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
