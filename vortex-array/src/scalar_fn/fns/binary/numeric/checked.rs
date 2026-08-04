// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checked-lane execution for the decimal kernels, driven by the shared
//! `vortex-compute` lane kernels.
//!
//! The primitive widths do not come through here: they are computed one row at a time by
//! [`row`](super::row), which writes a value for every row and reduces failure as one bit rather
//! than scanning lanes.

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSource;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_mask::AllOr;
use vortex_mask::Mask;

/// Apply the fallible `f` over every lane of `source`, failing only when `f` returns `None`
/// on a valid lane.
///
/// `f` is also invoked on invalid lanes (their failures are masked out and their values are
/// unspecified), so it must be total: no panics or side effects on any stored lane value.
///
/// This drives the one-pass early-exit kernels: failures abort at the end of the enclosing
/// 64-lane chunk. It suits an operation whose per-lane failure handling is cheap relative to the
/// operation itself, which is what the decimal kernels and their per-lane casts are.
///
/// On failure returns `Err(first_failing_valid_lane)`.
///
/// `#[inline(always)]`: this wrapper and its kernel calls must inline into the caller that
/// constructs the closure, so the closure environment (e.g. a captured constant operand)
/// flattens into registers. Left to its own devices under `codegen-units > 1`, the compiler
/// keeps the environment behind a pointer, and reloading a captured constant on every lane
/// blocks vectorization of the whole loop.
#[inline(always)]
pub(super) fn checked_lanes<S, T, F>(source: S, valid_rows: &Mask, f: F) -> Result<Buffer<T>, usize>
where
    S: IndexedSource,
    T: Copy + Default,
    F: FnMut(S::Item) -> Option<T>,
{
    let len = source.len();
    debug_assert_eq!(len, valid_rows.len());

    let valid_bits = match valid_rows.bit_buffer() {
        AllOr::All => None,
        AllOr::None => return Ok(Buffer::zeroed(len)),
        AllOr::Some(valid_bits) => Some(valid_bits),
    };

    let mut values = BufferMut::<T>::with_capacity(len);
    let out = &mut values.spare_capacity_mut()[..len];
    match valid_bits {
        None => source.try_map_into(out, f)?,
        Some(valid_bits) => source.try_map_masked_into(valid_bits, out, f)?,
    }
    // SAFETY: the kernels initialize every lane in `out`.
    unsafe { values.set_len(len) };
    Ok(values.freeze())
}
