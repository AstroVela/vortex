// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::PrimitiveArray;
use crate::arrays::patches::PATCH_BLOCK_SIZE;
use crate::arrays::patches::Patches;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::half::f16;
use crate::legacy_session;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

/// How a patch value is combined with the base value it lands on.
///
/// [`PatchFn::Add`] and [`PatchFn::Or`] are only valid for integer-typed arrays; this is
/// enforced at construction.
#[derive(Copy, Clone, Debug, Default, Hash, PartialEq, Eq)]
#[repr(u32)]
pub enum PatchFn {
    /// Replace the base value with the patch value.
    #[default]
    Overwrite = 0,
    /// Wrapping-add the patch value to the base value (residual patches).
    Add = 1,
    /// Bitwise-or the patch value into the base value (pre-shifted high-bit patches).
    Or = 2,
}

impl PatchFn {
    /// Returns true if this combine function is valid for values of the given [`PType`].
    pub fn supports_ptype(self, ptype: PType) -> bool {
        match self {
            PatchFn::Overwrite => true,
            PatchFn::Add | PatchFn::Or => ptype.is_int(),
        }
    }
}

impl TryFrom<u32> for PatchFn {
    type Error = vortex_error::VortexError;

    fn try_from(value: u32) -> VortexResult<Self> {
        match value {
            0 => Ok(PatchFn::Overwrite),
            1 => Ok(PatchFn::Add),
            2 => Ok(PatchFn::Or),
            _ => Err(vortex_err!("invalid PatchFn value {value}")),
        }
    }
}

impl Display for PatchFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PatchFn::Overwrite => write!(f, "overwrite"),
            PatchFn::Add => write!(f, "add"),
            PatchFn::Or => write!(f, "or"),
        }
    }
}

/// Native types that can be combined with a patch value via a [`PatchFn`].
pub trait PatchCombine: NativePType {
    /// Combine `base` with `patch` according to `patch_fn`.
    fn combine(patch_fn: PatchFn, base: Self, patch: Self) -> Self;
}

macro_rules! impl_patch_combine_int {
    ($($T:ty),*) => {
        $(impl PatchCombine for $T {
            #[inline(always)]
            fn combine(patch_fn: PatchFn, base: Self, patch: Self) -> Self {
                match patch_fn {
                    PatchFn::Overwrite => patch,
                    PatchFn::Add => base.wrapping_add(patch),
                    PatchFn::Or => base | patch,
                }
            }
        })*
    };
}

macro_rules! impl_patch_combine_float {
    ($($T:ty),*) => {
        $(impl PatchCombine for $T {
            #[inline(always)]
            fn combine(patch_fn: PatchFn, _base: Self, patch: Self) -> Self {
                match patch_fn {
                    PatchFn::Overwrite => patch,
                    // Validation rejects Add/Or for float arrays at construction.
                    _ => vortex_panic!("PatchFn::{patch_fn} is not supported for float values"),
                }
            }
        })*
    };
}

impl_patch_combine_int!(u8, u16, u32, u64, i8, i16, i32, i64);
impl_patch_combine_float!(f16, f32, f64);

/// A [`Patches`]-encoded Vortex array.
pub type PatchesArray = Array<Patches>;

#[derive(Debug, Clone)]
pub struct PatchesArrayData {
    /// The offset into the first block that is considered in bounds.
    ///
    /// The patches of the first block at positions less than `offset` should be skipped, and the
    /// offset should be subtracted out of the remaining positions to get their final position in
    /// the executed array. Always in `0..1024`.
    pub(super) offset: usize,

    /// How patch values are combined with the base values.
    pub(super) patch_fn: PatchFn,
}

impl Display for PatchesArrayData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}, patch_fn: {}", self.offset, self.patch_fn)
    }
}

#[array_slots(Patches)]
pub struct PatchesSlots {
    /// The base array containing the unpatched values.
    #[slot(0)]
    pub inner: ArrayRef,
    /// Per-block offsets into `indices`/`values`: block `b` owns patches
    /// `skip_indices[b]..skip_indices[b + 1]`.
    #[slot(1)]
    pub skip_indices: ArrayRef,
    /// The `u16` positions of patches relative to the start of their 1024-element block, sorted
    /// ascending within each block.
    #[slot(2)]
    pub indices: ArrayRef,
    /// The patch values aligned with `indices`.
    #[slot(3)]
    pub values: ArrayRef,
}

impl PatchesArrayData {
    pub(crate) fn validate(
        &self,
        dtype: &DType,
        len: usize,
        slots: &PatchesSlotsView,
    ) -> VortexResult<()> {
        vortex_ensure!(
            dtype.is_primitive(),
            "PatchesArray only supports primitive dtypes, got {dtype}"
        );
        vortex_ensure!(
            slots.inner.dtype() == dtype,
            "PatchesArray base dtype {} does not match outer dtype {}",
            slots.inner.dtype(),
            dtype
        );
        vortex_ensure!(
            slots.inner.len() == len,
            "PatchesArray base len {} does not match outer len {}",
            slots.inner.len(),
            len
        );
        vortex_ensure!(
            slots.indices.len() == slots.values.len(),
            "PatchesArray indices len {} does not match values len {}",
            slots.indices.len(),
            slots.values.len()
        );
        vortex_ensure!(
            slots.indices.dtype() == &DType::Primitive(PType::U16, Nullability::NonNullable),
            "PatchesArray indices must be non-nullable u16, got {}",
            slots.indices.dtype()
        );
        vortex_ensure!(
            slots.skip_indices.dtype() == &DType::Primitive(PType::U32, Nullability::NonNullable),
            "PatchesArray skip_indices must be non-nullable u32, got {}",
            slots.skip_indices.dtype()
        );
        vortex_ensure!(
            self.offset < PATCH_BLOCK_SIZE,
            "PatchesArray offset {} must be less than {PATCH_BLOCK_SIZE}",
            self.offset
        );
        if len > 0 {
            let n_blocks = (self.offset + len).div_ceil(PATCH_BLOCK_SIZE);
            vortex_ensure!(
                slots.skip_indices.len() == n_blocks + 1,
                "PatchesArray skip_indices len {} does not match block count {} + 1",
                slots.skip_indices.len(),
                n_blocks
            );
        }
        vortex_ensure!(
            dtype.eq_with_nullability_superset(slots.values.dtype()),
            "PatchesArray values dtype {} does not match array dtype {}",
            slots.values.dtype(),
            dtype
        );
        vortex_ensure!(
            self.patch_fn.supports_ptype(dtype.as_ptype()),
            "PatchFn::{} is not supported for dtype {dtype}",
            self.patch_fn
        );
        Ok(())
    }
}

pub trait PatchesArrayExt: PatchesArraySlotsExt {
    /// Returns the offset into the first block that is considered in bounds.
    #[inline]
    fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the combine function applied between base and patch values.
    #[inline]
    fn patch_fn(&self) -> PatchFn {
        self.patch_fn
    }

    /// Returns the total number of patches.
    #[inline]
    fn n_patches(&self) -> usize {
        self.indices().len()
    }

    /// Returns the range of `indices`/`values` owned by the given block.
    ///
    /// Blocks are counted in padded coordinates: block 0 starts `offset` elements before the
    /// first in-bounds element.
    #[inline]
    #[allow(clippy::disallowed_methods)]
    fn block_patch_range(&self, block: usize) -> VortexResult<Range<usize>> {
        assert!(block * PATCH_BLOCK_SIZE <= self.as_ref().len() + self.offset());

        let mut ctx = legacy_session().create_execution_ctx();
        let start = self.skip_indices().execute_scalar(block, &mut ctx)?;
        let stop = self.skip_indices().execute_scalar(block + 1, &mut ctx)?;

        let start = start
            .as_primitive()
            .as_::<usize>()
            .ok_or_else(|| vortex_err!("could not cast skip_index to usize"))?;
        let stop = stop
            .as_primitive()
            .as_::<usize>()
            .ok_or_else(|| vortex_err!("could not cast skip_index to usize"))?;

        Ok(start..stop)
    }

    /// Slice the array to a whole-block range (in padded coordinates), keeping the indices and
    /// values children whole.
    fn slice_blocks(&self, blocks: Range<usize>) -> VortexResult<PatchesArray> {
        let sliced_skip_indices = self.skip_indices().slice(blocks.start..blocks.end + 1)?;

        let begin = (blocks.start * PATCH_BLOCK_SIZE).saturating_sub(self.offset());
        let end = (blocks.end * PATCH_BLOCK_SIZE)
            .saturating_sub(self.offset())
            .min(self.as_ref().len());

        let offset = if blocks.start == 0 { self.offset() } else { 0 };
        let inner = self.inner().slice(begin..end)?;
        let len = inner.len();
        let dtype = self.as_ref().dtype().clone();
        let slots = PatchesSlots {
            inner,
            skip_indices: sliced_skip_indices,
            indices: self.indices().clone(),
            values: self.values().clone(),
        }
        .into_slots();

        Ok(unsafe { Patches::new_unchecked(dtype, len, slots, offset, self.patch_fn()) })
    }
}

impl<T: TypedArrayRef<Patches>> PatchesArrayExt for T {}

impl Patches {
    /// Build a [`PatchesArray`] from a base array and a set of legacy [`crate::patches::Patches`],
    /// converting the globally-sorted absolute patch indices into the block-relative layout.
    ///
    /// Unlike the lane-transposed `Patched` array, this conversion preserves patch order: the
    /// values child is reused as-is, and only the indices are re-encoded.
    pub fn from_array_and_patches(
        inner: ArrayRef,
        patches: &crate::patches::Patches,
        patch_fn: PatchFn,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PatchesArray> {
        vortex_ensure!(
            inner.dtype().eq_with_nullability_superset(patches.dtype()),
            "array DType must match patches DType"
        );
        vortex_ensure!(
            inner.dtype().is_primitive(),
            "Creating PatchesArray from Patches only supported for primitive arrays"
        );
        vortex_ensure!(
            patches.num_patches() <= u32::MAX as usize,
            "PatchesArray does not support > u32::MAX patch values"
        );
        vortex_ensure!(
            patches.values().all_valid(ctx)?,
            "PatchesArray cannot be built from Patches with nulls"
        );
        vortex_ensure!(
            patch_fn.supports_ptype(inner.dtype().as_ptype()),
            "PatchFn::{patch_fn} is not supported for dtype {}",
            inner.dtype()
        );
        vortex_ensure!(
            patches.array_len() == inner.len(),
            "Patches array_len {} does not match inner len {}",
            patches.array_len(),
            inner.len()
        );

        let indices = patches
            .indices()
            .clone()
            .execute::<Canonical>(ctx)?
            .into_primitive();

        let array_len = patches.array_len();
        let offset = patches.offset();

        let (skip_indices, block_indices) =
            match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
                build_block_layout(indices.as_slice::<I>(), offset, array_len)
            });

        let skip_indices = PrimitiveArray::new(skip_indices, Validity::NonNullable).into_array();
        let block_indices = PrimitiveArray::new(block_indices, Validity::NonNullable).into_array();

        let dtype = inner.dtype().clone();
        let len = inner.len();
        // The values child keeps the (possibly still compressed) patch values as-is; only ensure
        // the dtype matches the array's.
        let values = if patches.values().dtype() == &dtype {
            patches.values().clone()
        } else {
            patches.values().cast(dtype.clone())?
        };

        let slots = PatchesSlots {
            inner,
            skip_indices,
            indices: block_indices,
            values,
        }
        .into_slots();
        Ok(unsafe { Self::new_unchecked(dtype, len, slots, 0, patch_fn) })
    }

    /// Construct a new [`PatchesArray`] without validating the layout invariants.
    ///
    /// # Safety
    ///
    /// The caller must uphold the invariants checked by [`PatchesArrayData::validate`], plus:
    /// * `indices` are sorted ascending within each block, with no duplicates.
    /// * Every index is less than 1024.
    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        len: usize,
        slots: ArraySlots,
        offset: usize,
        patch_fn: PatchFn,
    ) -> PatchesArray {
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Patches, dtype, len, PatchesArrayData { offset, patch_fn })
                    .with_slots(slots),
            )
        }
    }
}

/// Split sorted absolute patch indices into per-block skip offsets and block-relative positions.
///
/// `indices_in` are absolute positions offset by `offset` (the legacy `Patches` convention);
/// the returned layout is rebuilt with its first block aligned to the logical array start.
#[expect(clippy::cast_possible_truncation)]
fn build_block_layout<I: IntegerPType>(
    indices_in: &[I],
    offset: usize,
    array_len: usize,
) -> (Buffer<u32>, Buffer<u16>) {
    let n_blocks = array_len.div_ceil(PATCH_BLOCK_SIZE);
    let n_patches = indices_in.len();

    let mut skip_indices = BufferMut::<u32>::with_capacity(n_blocks + 1);
    let mut block_indices = BufferMut::<u16>::with_capacity(n_patches);

    let mut cursor = 0usize;
    for block in 0..n_blocks {
        skip_indices.push(cursor as u32);
        let block_start = block * PATCH_BLOCK_SIZE;
        let block_end = block_start + PATCH_BLOCK_SIZE;
        while cursor < n_patches {
            let index = indices_in[cursor].as_() - offset;
            if index >= block_end {
                break;
            }
            block_indices.push((index - block_start) as u16);
            cursor += 1;
        }
    }
    skip_indices.push(n_patches as u32);

    (skip_indices.freeze(), block_indices.freeze())
}

/// Apply block-relative patches on top of the existing values.
///
/// `skip_indices` values are absolute offsets into `indices`/`values` (they are not rebased when
/// an array is sliced), and blocks are walked in padded coordinates: positions below `offset` or
/// at or beyond `offset + len` are skipped.
pub(crate) fn apply_patches_primitive<V: PatchCombine>(
    output: &mut [V],
    offset: usize,
    len: usize,
    skip_indices: &[u32],
    indices: &[u16],
    values: &[V],
    patch_fn: PatchFn,
) {
    let n_blocks = (offset + len).div_ceil(PATCH_BLOCK_SIZE);
    debug_assert!(skip_indices.len() > n_blocks);

    for block in 0..n_blocks {
        let start = skip_indices[block] as usize;
        let stop = skip_indices[block + 1] as usize;
        let block_start = block * PATCH_BLOCK_SIZE;

        for patch_idx in start..stop {
            let index = block_start + indices[patch_idx] as usize;
            if index < offset || index >= offset + len {
                continue;
            }
            let out = &mut output[index - offset];
            *out = V::combine(patch_fn, *out, values[patch_idx]);
        }
    }
}

/// Search one block's sorted block-relative indices for `rel`, returning the absolute position
/// into `indices`/`values` if a patch exists at that position.
///
/// This is the block-relative counterpart of a global binary search over absolute patch indices:
/// the caller locates `block_range` in constant time via `skip_indices`, and this search then
/// only touches the (at most 1024, `u16`-typed) indices of one block.
pub fn search_block(indices: &[u16], block_range: Range<usize>, rel: u16) -> Option<usize> {
    let block_indices = &indices[block_range.clone()];
    block_indices
        .binary_search(&rel)
        .ok()
        .map(|found| block_range.start + found)
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::build_block_layout;

    #[test]
    fn block_layout_simple() -> VortexResult<()> {
        let (skip, rel) = build_block_layout(&[1u32, 2, 3], 0, 1024);
        assert_eq!(skip.as_slice(), &[0, 3]);
        assert_eq!(rel.as_slice(), &[1, 2, 3]);
        Ok(())
    }

    #[test]
    fn block_layout_multi_block() -> VortexResult<()> {
        let (skip, rel) = build_block_layout(&[100u64, 1500, 2500, 3500], 0, 4096);
        assert_eq!(skip.as_slice(), &[0, 1, 2, 3, 4]);
        assert_eq!(
            rel.as_slice(),
            &[100, 1500 - 1024, 2500 - 2048, 3500 - 3072]
        );
        Ok(())
    }

    #[test]
    fn block_layout_empty_blocks() -> VortexResult<()> {
        let (skip, rel) = build_block_layout(&[4000u32, 4001], 0, 5000);
        assert_eq!(skip.as_slice(), &[0, 0, 0, 0, 2, 2]);
        assert_eq!(rel.as_slice(), &[4000 - 3072, 4001 - 3072]);
        Ok(())
    }

    #[test]
    fn block_layout_with_offset() -> VortexResult<()> {
        // Legacy patches store absolute indices; offset 10 means logical position idx - 10.
        let (skip, rel) = build_block_layout(&[10u32, 1034], 10, 2000);
        assert_eq!(skip.as_slice(), &[0, 1, 2]);
        assert_eq!(rel.as_slice(), &[0, 0]);
        Ok(())
    }
}
