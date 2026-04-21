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

use super::super::backend::gpu::attention::{
    rocmforge_launch_attention_decode, rocmforge_launch_kv_cache_append,
};
use super::super::backend::gpu::elementwise::{
    rocmforge_launch_embedding_lookup, rocmforge_launch_residual_add_inplace,
    rocmforge_launch_rms_norm, rocmforge_launch_rms_norm_batched, rocmforge_launch_rope,
    rocmforge_launch_swiglu,
};
use super::super::backend::gpu::error::{check, HipError, HipResult};
use super::super::backend::gpu::gemv::{
    rocmforge_launch_gemv_q4_0_standard, rocmforge_launch_gemv_q4_k_gate_up_swiglu,
    rocmforge_launch_gemv_q4_k_q8_inline, rocmforge_launch_gemv_q4_k_standard,
    rocmforge_launch_gemv_q6_k_standard, rocmforge_launch_gemv_q8_0_standard,
};
use super::super::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use super::super::backend::gpu::wrappers::{HipBuffer, HipStream};
use super::super::core::gguf::GGUFFile;
use super::super::core::model_loader::LoadedModel;
use super::super::core::tensor_info::{GgmlType, TensorRole};
use super::super::ir::interpreter::dequant_block;
use super::super::ir::formats::{q4_0, q4_k, q6_k, q8_0};
use super::buffer_plan::{BufferPlan, KvCacheLayout};
use super::nodes::{BufferId, GraphNode, WeightRef};
use super::ComputationGraph;
use super::super::runtime::{KernelId, OpType, Runtime, ShapeKey, VariantId};

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
                let trace =
                    std::env::var("ROCMFORGE_TRACE").ok().as_deref() == Some("1");
                let nan_guard =
                    std::env::var("ROCMFORGE_NAN_GUARD").ok().as_deref()
                        == Some("1");
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
                });
            }
            other => {
                return Err(HipError {
                    code: -1,
                    message: format!(
                        "embedding table format {other} not supported in Phase 1"
                    ),
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
                let block_bytes = gguf
                    .tensor_data(embed_tensor, offset + bb)
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
            std::slice::from_raw_parts(
                fp32_table.as_ptr() as *const u8,
                fp32_table.len() * 4,
            )
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
        })
    }

    /// Attach a Self-Tuning Runtime. The executor then routes all
    /// GEMV dispatches through the Bandit and feeds back timings.
    /// Pre-registers every unique GEMV shape the graph will dispatch
    /// so the Bandit knows the candidate set before the first token.
    pub fn attach_runtime(&mut self, mut runtime: Runtime) {
        for node in &self.graph.nodes {
            if let GraphNode::Gemm {
                weight, out_dim, in_dim, ..
            } = node
            {
                runtime
                    .registry
                    .register_gemv_shape(weight.format, *out_dim as u32, *in_dim as u32);
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
        let per_cache_bytes =
            self.kv_layout.num_kv_heads * self.kv_layout.head_stride * 4;
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
    /// Implemented as a loop over `execute_decode` — a true
    /// WMMA-batched prefill is Phase-2.
    pub fn execute_prefill(
        &mut self,
        token_ids: &[u32],
        pos_offset: usize,
    ) -> HipResult<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(HipError {
                code: -1,
                message: "execute_prefill called with 0 tokens".into(),
                context: "prefill".into(),
            });
        }
        let mut last = Vec::new();
        for (i, &tok) in token_ids.iter().enumerate() {
            last = self.execute_decode(tok, pos_offset + i)?;
        }
        Ok(last)
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
            GraphNode::Embedding { table: _table, input, output } => {
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
            GraphNode::Gemm { weight, input, output, out_dim, in_dim, .. } => {
                self.dispatch_gemv(weight, *input, *output, *out_dim, *in_dim)?;
            }
            GraphNode::Rope {
                q_buffer, k_buffer, rope_freqs, n_heads, n_kv_heads, head_dim, theta_base,
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
            GraphNode::KvCacheAppend { k_buffer, v_buffer, layer_idx } => {
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
                q_buffer, output, layer_idx, n_heads, n_kv_heads, head_dim,
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
                gate_weight, up_weight, input, output, ffn_dim, hidden_dim,
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
            GraphNode::SwiGLU { gate_buffer, up_buffer, output } => {
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
        let kernel = runtime.kernel_for(&shape, variant_id).ok_or_else(|| HipError {
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
            let gate_ptr = self.weight_ptr(gate_weight);
            let up_ptr = self.weight_ptr(up_weight);
            let in_ptr = self.buf_ptr(input);
            let out_ptr = self.buf_mut_ptr(output);
            // NOTE: the fused Q4_K gate+up+swiglu kernel outputs
            // `ffn_dim` values for the swiglu mid-activation (actually
            // `ffn_dim` is our "N" there — it's the output column
            // count). That matches the `ffn_mid` buffer sizing in the
            // planner.
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
        let Some(out_id) = out_buf else { return Ok(()); };
        let elems = self.plan.specs[&out_id].elem_count;
        let sample = self.read_buffer(out_id, elems)?;
        let (min, max, any_nan) = sample.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY, false),
            |(mn, mx, nan), &v| {
                (
                    mn.min(v),
                    mx.max(v),
                    nan | !v.is_finite(),
                )
            },
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
