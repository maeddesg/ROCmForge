//! VRAM Arena Allocator — single `hipMalloc`, three zones.
//!
//! Architecture reference: `architecture_v1.2.0-draft.md` §3.6 (VRAM
//! Arena 2.0). A monolithic allocation avoids fragmentation, keeps
//! dispatch latency low (no per-tensor `hipMalloc`), and locks the
//! VRAM budget at program start.
//!
//! Zones
//! ```text
//! +------------------+---------------------+--------------------+
//! | Zone A: weights  | Zone B: KV-Cache    | Zone C: scratch    |
//! | (bump allocator) | (grows with ctx)    | (ping-pong 2 bufs) |
//! +------------------+---------------------+--------------------+
//! 0                 A_end              B_end                  end
//! ```
//!
//! Alignment: 256 B for individual slices (coalesced memory access),
//! 4 KiB for zone boundaries (TLB efficiency).

use std::ffi::c_void;

use super::error::{HipError, HipResult};
use super::wrappers::HipBuffer;

pub const ARENA_ALIGNMENT: usize = 256;
pub const ZONE_ALIGNMENT: usize = 4096;

/// Round `size` up to the next multiple of `alignment`. `alignment`
/// must be a power of two.
const fn align_up(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

/// Planned zone sizes. Derived from model shape via
/// [`ArenaConfig::from_model`]; callers that want to handcraft the
/// layout can instantiate directly.
#[derive(Debug, Clone, Copy)]
pub struct ArenaConfig {
    pub total_size: usize,
    pub weights_size: usize,
    pub kv_cache_max_size: usize,
    /// Size of **one** scratch buffer; the arena reserves two of these
    /// for the ping-pong.
    pub scratch_per_buffer: usize,
}

impl ArenaConfig {
    /// Plan an arena for a concrete model.
    ///
    /// Size arithmetic
    /// * weights: `model_size_bytes` rounded up to `ZONE_ALIGNMENT`
    /// * kv per token: `2 × n_layers × n_kv_heads × head_dim × kv_element_size`
    ///   (factor 2 covers Key and Value), multiplied by `max_context`
    /// * scratch per buffer: `max(hidden_dim, ffn_dim) × max_batch_size × 4`
    ///
    /// If the resulting `needed + safety_margin` exceeds `total_vram`,
    /// `total_size` is clamped to `total_vram - safety_margin` so that
    /// [`validate`] fails with a descriptive breakdown — the caller is
    /// expected to reduce `max_context` and retry, never the arena.
    #[allow(clippy::too_many_arguments)]
    pub fn from_model(
        model_size_bytes: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_context: usize,
        kv_element_size: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        max_batch_size: usize,
        total_vram: usize,
    ) -> Self {
        let weights_size = align_up(model_size_bytes, ZONE_ALIGNMENT);

        let kv_per_token = 2 * n_layers * n_kv_heads * head_dim * kv_element_size;
        let kv_cache_max_size = align_up(kv_per_token * max_context, ZONE_ALIGNMENT);

        let scratch_dim = std::cmp::max(hidden_dim, ffn_dim);
        let scratch_per_buffer = align_up(
            scratch_dim * max_batch_size * std::mem::size_of::<f32>(),
            ZONE_ALIGNMENT,
        );

        let needed = weights_size + kv_cache_max_size + 2 * scratch_per_buffer;
        let safety_margin: usize = 512 * 1024 * 1024;

        let total_size = if needed + safety_margin <= total_vram {
            needed
        } else {
            // Intentional: don't silently shrink zones — surface OOM via
            // validate(). The caller decides how to adapt (shorter ctx,
            // smaller batch, different precision).
            total_vram.saturating_sub(safety_margin)
        };

        Self {
            total_size,
            weights_size,
            kv_cache_max_size,
            scratch_per_buffer,
        }
    }

    /// Returns `Err(breakdown)` if the zone totals don't fit into
    /// `total_size`. The error string is the single most useful piece
    /// of information for an end user.
    pub fn validate(&self) -> Result<(), String> {
        let needed = self.weights_size + self.kv_cache_max_size + 2 * self.scratch_per_buffer;
        if needed > self.total_size {
            return Err(format!(
                "Arena too small: need {:.2} GB but {:.2} GB allocated. \
                 Weights={:.2} GB, KV={:.2} GB, Scratch=2×{:.1} MB",
                needed as f64 / 1e9,
                self.total_size as f64 / 1e9,
                self.weights_size as f64 / 1e9,
                self.kv_cache_max_size as f64 / 1e9,
                self.scratch_per_buffer as f64 / 1e6,
            ));
        }
        Ok(())
    }
}

/// A range inside the arena. No allocation of its own — just an offset
/// and a length relative to the arena's base pointer.
#[derive(Debug, Clone, Copy)]
pub struct ArenaSlice {
    pub offset: usize,
    pub size: usize,
}

impl ArenaSlice {
    /// SAFETY: `base_ptr` must be the arena's base pointer for this
    /// slice to be valid. The returned pointer inherits the lifetime
    /// and aliasing contract of the arena.
    pub unsafe fn as_ptr(&self, base_ptr: *const c_void) -> *const c_void {
        (base_ptr as *const u8).add(self.offset) as *const c_void
    }

    /// SAFETY: see [`as_ptr`].
    pub unsafe fn as_mut_ptr(&self, base_ptr: *mut c_void) -> *mut c_void {
        (base_ptr as *mut u8).add(self.offset) as *mut c_void
    }
}

/// Zone-C ping-pong: two scratch buffers that alternate between
/// "input" and "output" on each [`swap`]. Allows every transformer
/// layer to write into a region the next layer treats as read-only
/// without explicit reallocation.
pub struct PingPong {
    buffer_a: ArenaSlice,
    buffer_b: ArenaSlice,
    /// `true` = A is current input, B is current output.
    state: bool,
}

impl PingPong {
    pub fn new(buffer_a: ArenaSlice, buffer_b: ArenaSlice) -> Self {
        Self {
            buffer_a,
            buffer_b,
            state: true,
        }
    }

    pub fn input(&self) -> ArenaSlice {
        if self.state {
            self.buffer_a
        } else {
            self.buffer_b
        }
    }

    pub fn output(&self) -> ArenaSlice {
        if self.state {
            self.buffer_b
        } else {
            self.buffer_a
        }
    }

    pub fn swap(&mut self) {
        self.state = !self.state;
    }

    pub fn reset(&mut self) {
        self.state = true;
    }
}

/// The VRAM arena: one [`HipBuffer`], three zones, offset arithmetic.
/// Dropping the arena runs `hipFree` on the base allocation via
/// `HipBuffer`'s Drop.
pub struct VramArena {
    buffer: HipBuffer,
    zone_a: ArenaSlice,
    zone_b: ArenaSlice,
    zone_c_ping_pong: PingPong,
    config: ArenaConfig,
    zone_a_cursor: usize,
    zone_b_used: usize,
}

impl VramArena {
    pub fn new(config: ArenaConfig) -> HipResult<Self> {
        config.validate().map_err(|message| HipError {
            code: -1,
            message,
            context: "VramArena::new".to_string(),
        })?;

        let buffer = HipBuffer::new(config.total_size)?;

        let zone_a_offset = 0usize;
        let zone_a_size = config.weights_size;

        let zone_b_offset = align_up(zone_a_offset + zone_a_size, ZONE_ALIGNMENT);
        let zone_b_size = config.kv_cache_max_size;

        let zone_c_offset = align_up(zone_b_offset + zone_b_size, ZONE_ALIGNMENT);
        let scratch_a_offset = zone_c_offset;
        let scratch_b_offset = align_up(
            scratch_a_offset + config.scratch_per_buffer,
            ZONE_ALIGNMENT,
        );

        let zone_a = ArenaSlice {
            offset: zone_a_offset,
            size: zone_a_size,
        };
        let zone_b = ArenaSlice {
            offset: zone_b_offset,
            size: zone_b_size,
        };
        let scratch_a = ArenaSlice {
            offset: scratch_a_offset,
            size: config.scratch_per_buffer,
        };
        let scratch_b = ArenaSlice {
            offset: scratch_b_offset,
            size: config.scratch_per_buffer,
        };

        Ok(Self {
            buffer,
            zone_a,
            zone_b,
            zone_c_ping_pong: PingPong::new(scratch_a, scratch_b),
            config,
            zone_a_cursor: 0,
            zone_b_used: 0,
        })
    }

    // --- Zone A: weights (bump allocator, forward-only) ---------------------

    pub fn alloc_weights(&mut self, size: usize) -> Result<ArenaSlice, String> {
        let aligned_size = align_up(size, ARENA_ALIGNMENT);
        if self.zone_a_cursor + aligned_size > self.zone_a.size {
            return Err(format!(
                "Zone A overflow: need {} bytes at cursor {}, zone size {}",
                aligned_size, self.zone_a_cursor, self.zone_a.size
            ));
        }
        let slice = ArenaSlice {
            offset: self.zone_a.offset + self.zone_a_cursor,
            size: aligned_size,
        };
        self.zone_a_cursor += aligned_size;
        Ok(slice)
    }

    pub fn weights_used(&self) -> usize {
        self.zone_a_cursor
    }

    pub fn weights_remaining(&self) -> usize {
        self.zone_a.size - self.zone_a_cursor
    }

    // --- Zone B: KV-Cache ---------------------------------------------------

    pub fn kv_cache_slice(&self) -> ArenaSlice {
        self.zone_b
    }

    pub fn kv_cache_used(&self) -> usize {
        self.zone_b_used
    }

    pub fn kv_cache_remaining(&self) -> usize {
        self.zone_b.size - self.zone_b_used
    }

    pub fn kv_cache_grow(&mut self, additional_bytes: usize) -> Result<(), String> {
        if self.zone_b_used + additional_bytes > self.zone_b.size {
            return Err(format!(
                "KV-Cache overflow: need {} more bytes, only {} remaining",
                additional_bytes,
                self.kv_cache_remaining()
            ));
        }
        self.zone_b_used += additional_bytes;
        Ok(())
    }

    pub fn kv_cache_reset(&mut self) {
        self.zone_b_used = 0;
    }

    // --- Zone C: scratch (ping-pong) ---------------------------------------

    pub fn ping_pong(&self) -> &PingPong {
        &self.zone_c_ping_pong
    }

    pub fn ping_pong_mut(&mut self) -> &mut PingPong {
        &mut self.zone_c_ping_pong
    }

    pub fn scratch_size(&self) -> usize {
        self.config.scratch_per_buffer
    }

    // --- base pointer -------------------------------------------------------

    pub fn base_ptr(&self) -> *const c_void {
        self.buffer.as_ptr()
    }

    pub fn base_mut_ptr(&mut self) -> *mut c_void {
        self.buffer.as_mut_ptr()
    }

    // --- diagnostics --------------------------------------------------------

    pub fn total_size(&self) -> usize {
        self.config.total_size
    }

    pub fn config(&self) -> &ArenaConfig {
        &self.config
    }

    pub fn print_layout(&self) {
        println!(
            "VRAM Arena Layout ({:.2} GB total):",
            self.total_size() as f64 / 1e9
        );
        println!(
            "  Zone A (Weights):  offset={:#012x}, size={:.2} GB, used={:.2} GB",
            self.zone_a.offset,
            self.zone_a.size as f64 / 1e9,
            self.zone_a_cursor as f64 / 1e9,
        );
        println!(
            "  Zone B (KV-Cache): offset={:#012x}, size={:.2} GB, used={:.2} GB",
            self.zone_b.offset,
            self.zone_b.size as f64 / 1e9,
            self.zone_b_used as f64 / 1e9,
        );
        let pp = &self.zone_c_ping_pong;
        println!(
            "  Zone C (Scratch A): offset={:#012x}, size={:.2} MB",
            pp.buffer_a.offset,
            pp.buffer_a.size as f64 / 1e6,
        );
        println!(
            "  Zone C (Scratch B): offset={:#012x}, size={:.2} MB",
            pp.buffer_b.offset,
            pp.buffer_b.size as f64 / 1e6,
        );
    }
}
