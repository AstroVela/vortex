// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An array that patches another array with per-block exception values.
//!
//! # Background
//!
//! Patching is common when an encoding almost completely covers an array save a few exceptions.
//! In that case, rather than avoid the encoding entirely, it's preferable to
//!
//! * Replace unencodable values with fillers (zeros, frequent values, nulls, etc.)
//! * Wrap the array with a [`PatchesArray`] signaling that when the base array is executed,
//!   some of the decoded values must be combined with patch values.
//!
//! This encoding supersedes the transposed, lane-oriented `Patched` array and is modeled on the
//! block-residual layout: patches are grouped by the 1024-element block they fall into, and each
//! patch position is stored relative to the start of its block.
//!
//! # Layout
//!
//! The array has 4 children (no buffers):
//!
//! * `inner`: the base array containing encoded values, including the filler values that need to
//!   be patched at execution time.
//! * `skip_indices`: `n_blocks + 1` unsigned offsets into `indices`/`values`. Block `b` owns the
//!   patches `indices[skip_indices[b]..skip_indices[b + 1]]`, so any block's patches are found in
//!   constant time — the global bits of a patch position live here.
//! * `indices`: `u16` patch positions **relative to the start of the 1024-element block**, sorted
//!   ascending within each block — the local bits of a patch position.
//! * `values`: the patch values aligned with `indices`.
//!
//! ```text
//!                    block 0        block 1        block 2       block 3
//!                ┌─────────────┬─────────────┬─────────────┬─────────────┐
//! skip_indices   │      0      │      2      │      2      │      5      │  ...
//!                └──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
//!                       │             └──────┬──────┘             │
//!                       │                    │                    │
//!                ┌──────▼──────┬─────────────▼────────────┬───────▼─────┐
//!    indices     │  17  │ 903  │  4   │  4   │  511       │             │
//!                ├──────┼──────┼──────┼──────┼────────────┼─────────────┤
//!    values      │      │      │      │      │            │             │
//!                └──────┴──────┴──────┴──────┴────────────┴─────────────┘
//! ```
//!
//! Splitting a patch position into global bits (`skip_indices`) and block-local bits (`indices`)
//! has three payoffs over a single sorted list of absolute positions:
//!
//! * Random access is O(1) + O(log k) for k patches per block (usually one cache line of `u16`s),
//!   instead of an O(log n_patches) binary search whose every probe is a potential cache miss.
//! * The block-local `indices` always fit in `u16`, regardless of array length.
//! * Decompression fuses naturally with block-oriented codecs (e.g. FastLanes bit-packing):
//!   each 1024-element block is decoded while its patches are applied cache-hot, and each block's
//!   patches are a contiguous run — no searching, and trivially parallel.
//!
//! # Combine functions
//!
//! Unlike classic patching, the patch values do not have to overwrite the base values. Each array
//! carries a [`PatchFn`] describing how a patch is combined with the base value it lands on:
//!
//! * [`PatchFn::Overwrite`]: `out = patch` — classic exception patching.
//! * [`PatchFn::Add`]: `out = base.wrapping_add(patch)` — residual patching, where the base holds
//!   a lossy approximation and the patch stores the correction.
//! * [`PatchFn::Or`]: `out = base | patch` — split-bits patching, where the base holds the low
//!   bits (e.g. bit-packed) and the patch stores the pre-shifted high bits.

mod array;
mod compute;
mod vtable;

use std::env;
use std::sync::LazyLock;

pub use array::*;
pub use vtable::*;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
}

/// The number of elements covered by one entry of `skip_indices`.
pub const PATCH_BLOCK_SIZE: usize = 1024;

/// Flag indicating if experimental block-patches array support is enabled.
///
/// This is set using the environment variable `VORTEX_EXPERIMENTAL_PATCHES_ARRAY`.
///
/// When this is true, arrays with interior `Patches` are read as a [`PatchesArray`], eliminating
/// the interior patches, and the builtin compressor will also generate [`PatchesArray`]s.
///
/// This flag supersedes `VORTEX_EXPERIMENTAL_PATCHED_ARRAY` (the lane-transposed `Patched`
/// array); when both are set, this one wins.
pub fn use_experimental_patches_array() -> bool {
    static USE_EXPERIMENTAL_PATCHES_ARRAY: LazyLock<bool> =
        LazyLock::new(|| env::var("VORTEX_EXPERIMENTAL_PATCHES_ARRAY").is_ok_and(|v| v == "1"));
    *USE_EXPERIMENTAL_PATCHES_ARRAY
}
