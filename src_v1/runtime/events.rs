//! HIP-event pool for Bandit-timing during Exploration.
//!
//! Arch-Doc §3.7 "Zero-Sync Pipeline": instead of calling
//! `stream.synchronize()` around every GEMV launch to measure its
//! elapsed time (Phase-1 code, 83 260 syncs/100 tok), we record
//! pairs of `hipEvent_t`s **on the same stream** before and after
//! the launch and drain the measurements in one batch at token
//! end — exactly one sync per token during exploration, zero
//! syncs during exploitation.
//!
//! The pool holds a fixed number of event pairs (pre-allocated at
//! pipeline construction time, `hipEventCreate` is not cheap) and
//! recycles them: `record_start` / `record_stop` bump a cursor,
//! `flush` reads all queued pairs, then `clear_pending` resets the
//! cursor for the next token.

use super::super::backend::gpu::error::HipResult;
use super::super::backend::gpu::wrappers::{HipEvent, HipStream};
use super::{Runtime, ShapeKey, VariantId};

pub struct EventPair {
    pub start: HipEvent,
    pub stop: HipEvent,
}

#[derive(Debug, Clone, Copy)]
pub struct PendingMeasurement {
    pub shape: ShapeKey,
    pub variant_id: VariantId,
    pub pair_idx: usize,
}

pub struct EventPool {
    pairs: Vec<EventPair>,
    pending: Vec<PendingMeasurement>,
    /// Capacity hard-limit so a pathological graph can't silently
    /// outgrow the pool. `record_start` returns `None` if full, and
    /// the caller falls back to launching without timing — the
    /// Bandit just misses that data point for this token.
    capacity: usize,
}

impl EventPool {
    pub fn new(capacity: usize) -> HipResult<Self> {
        let mut pairs = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            pairs.push(EventPair {
                start: HipEvent::new()?,
                stop: HipEvent::new()?,
            });
        }
        Ok(Self {
            pairs,
            pending: Vec::with_capacity(capacity),
            capacity,
        })
    }

    /// Record a start event on the stream. Returns the pair index
    /// that `record_stop` must receive, or `None` if the pool is
    /// full for this token (caller skips timing).
    pub fn record_start(
        &mut self,
        stream: &HipStream,
        shape: ShapeKey,
        variant: VariantId,
    ) -> HipResult<Option<usize>> {
        let idx = self.pending.len();
        if idx >= self.capacity {
            return Ok(None);
        }
        self.pairs[idx].start.record(stream)?;
        self.pending.push(PendingMeasurement {
            shape,
            variant_id: variant,
            pair_idx: idx,
        });
        Ok(Some(idx))
    }

    pub fn record_stop(&mut self, stream: &HipStream, idx: usize) -> HipResult<()> {
        self.pairs[idx].stop.record(stream)
    }

    /// Drain pending measurements into the Bandit. Must be called
    /// after the stream is synchronised (either by `read_buffer` on
    /// the logits or by an explicit sync). Does **not** sync on its
    /// own — callers either rely on an implicit sync (the logits
    /// hipMemcpy) or sync once explicitly.
    pub fn flush_into(&mut self, runtime: &mut Runtime) -> HipResult<()> {
        for m in self.pending.drain(..) {
            let pair = &self.pairs[m.pair_idx];
            let ms = HipEvent::elapsed_ms(&pair.start, &pair.stop)?;
            // Bandit stores microseconds (Phase-1 semantics).
            // hipEventElapsedTime returns milliseconds, convert here
            // — missing this factor flattens every arm by 1000× and
            // UCB1's exploration term would dominate forever.
            runtime.record(&m.shape, m.variant_id, ms as f64 * 1000.0);
        }
        Ok(())
    }

    /// Discard any pending entries without feeding them to the
    /// Bandit (e.g. when a generation errors out).
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
