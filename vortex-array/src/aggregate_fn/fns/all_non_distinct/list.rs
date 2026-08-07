// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_error::VortexResult;

use super::all_non_distinct;
use super::filter::filter_valid_rows_if_needed;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::listview::list_from_list_view;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::NativePType;
use crate::match_each_unsigned_integer_ptype;

pub(super) fn check_list_identical(
    lhs: &ListViewArray,
    rhs: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if let Some((lhs, rhs)) =
        filter_valid_rows_if_needed(&lhs.clone().into_array(), &rhs.clone().into_array(), ctx)?
    {
        return all_non_distinct(&lhs, &rhs, ctx);
    }

    if lhs.is_zero_copy_to_list() && rhs.is_zero_copy_to_list() {
        return check_zero_copy_list_identical(lhs, rhs, ctx);
    }

    let lhs = list_from_list_view(lhs.clone(), ctx)?;
    let rhs = list_from_list_view(rhs.clone(), ctx)?;

    if !check_list_offsets_identical(&lhs, &rhs, ctx)? {
        return Ok(false);
    }

    all_non_distinct(lhs.elements(), rhs.elements(), ctx)
}

/// Materialize an integer offsets/sizes child as an unsigned primitive array.
///
/// The values are non-negative by the list invariants, so the unsigned reinterpret is lossless
/// and halves the ptype dispatch combinations.
fn materialize_unsigned(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
    Ok(primitive.reinterpret_cast(primitive.ptype().to_unsigned()))
}

/// Whether two non-negative integer slices are element-wise equal, widened to `u64`.
fn unsigned_slices_identical<L, R>(lhs: &[L], rhs: &[R]) -> bool
where
    L: NativePType + AsPrimitive<u64>,
    R: NativePType + AsPrimitive<u64>,
{
    lhs.iter().zip(rhs).all(|(&l, &r)| {
        let (l, r): (u64, u64) = (l.as_(), r.as_());
        l == r
    })
}

/// Whether two non-negative offset slices are element-wise equal relative to each slice's first
/// offset. The slices must be non-empty and sorted, as zero-copy list view offsets are.
fn relative_offsets_identical<L, R>(lhs: &[L], rhs: &[R]) -> bool
where
    L: NativePType + AsPrimitive<u64>,
    R: NativePType + AsPrimitive<u64>,
{
    let lhs_base: u64 = lhs[0].as_();
    let rhs_base: u64 = rhs[0].as_();
    lhs.iter().zip(rhs).all(|(&l, &r)| {
        let (l, r): (u64, u64) = (l.as_(), r.as_());
        l - lhs_base == r - rhs_base
    })
}

/// Compare the `len + 1` offsets of two list arrays for equality.
///
/// The offsets are materialized once and compared as slices, rather than paying `offset_at`'s
/// per-call downcast and ptype dispatch (or per-element scalar execution for non-primitive
/// offsets) inside the loop.
fn check_list_offsets_identical(
    lhs: &ListArray,
    rhs: &ListArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let lhs_offsets = materialize_unsigned(lhs.offsets(), ctx)?;
    let rhs_offsets = materialize_unsigned(rhs.offsets(), ctx)?;

    Ok(match_each_unsigned_integer_ptype!(
        lhs_offsets.ptype(),
        |L| {
            match_each_unsigned_integer_ptype!(rhs_offsets.ptype(), |R| {
                unsigned_slices_identical(
                    &lhs_offsets.as_slice::<L>()[..=lhs.len()],
                    &rhs_offsets.as_slice::<R>()[..=lhs.len()],
                )
            })
        }
    ))
}

fn check_zero_copy_list_identical(
    lhs: &ListViewArray,
    rhs: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    debug_assert!(lhs.is_zero_copy_to_list());
    debug_assert!(rhs.is_zero_copy_to_list());

    if lhs.is_empty() {
        return Ok(true);
    }

    // Materialize sizes and offsets once and compare them as slices, rather than paying
    // `size_at`/`offset_at`'s per-call downcast and ptype dispatch inside the loop.
    let len = lhs.len();

    let lhs_sizes = materialize_unsigned(lhs.sizes(), ctx)?;
    let rhs_sizes = materialize_unsigned(rhs.sizes(), ctx)?;
    let sizes_identical = match_each_unsigned_integer_ptype!(lhs_sizes.ptype(), |L| {
        match_each_unsigned_integer_ptype!(rhs_sizes.ptype(), |R| {
            unsigned_slices_identical(
                &lhs_sizes.as_slice::<L>()[..len],
                &rhs_sizes.as_slice::<R>()[..len],
            )
        })
    });
    if !sizes_identical {
        return Ok(false);
    }

    // Zero-copy views are ordered, so offsets are compared relative to each array's first offset.
    let lhs_offsets = materialize_unsigned(lhs.offsets(), ctx)?;
    let rhs_offsets = materialize_unsigned(rhs.offsets(), ctx)?;
    let offsets_identical = match_each_unsigned_integer_ptype!(lhs_offsets.ptype(), |L| {
        match_each_unsigned_integer_ptype!(rhs_offsets.ptype(), |R| {
            relative_offsets_identical(
                &lhs_offsets.as_slice::<L>()[..len],
                &rhs_offsets.as_slice::<R>()[..len],
            )
        })
    });
    if !offsets_identical {
        return Ok(false);
    }

    let lhs_base = lhs.offset_at(0);
    let rhs_base = rhs.offset_at(0);
    let lhs_end = lhs.offset_at(len - 1) + lhs.size_at(len - 1);
    let rhs_end = rhs.offset_at(len - 1) + rhs.size_at(len - 1);

    let lhs_elements = lhs.elements().slice(lhs_base..lhs_end)?;
    let rhs_elements = rhs.elements().slice(rhs_base..rhs_end)?;

    all_non_distinct(&lhs_elements, &rhs_elements, ctx)
}
