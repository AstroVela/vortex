// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lazy gather of contiguous child ranges.
//!
//! `TakeSlicesArray` represents the concatenation of
//! `values[starts[i]..starts[i] + lengths[i]]` for each range row. Ranges may overlap, repeat,
//! and appear in any order.

mod array;
mod vtable;

pub use array::TakeSlicesArrayExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
pub use vtable::*;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::dtype::IntegerPType;

pub(super) fn check_index_arrays(starts: &ArrayRef, lengths: &ArrayRef) -> VortexResult<()> {
    check_index_dtype("starts", starts)?;
    check_index_dtype("lengths", lengths)?;
    vortex_ensure!(
        starts.len() == lengths.len(),
        "TakeSlicesArray starts and lengths must have equal length, got starts {} and lengths {}",
        starts.len(),
        lengths.len()
    );
    Ok(())
}

fn check_index_dtype(name: &str, indices: &ArrayRef) -> VortexResult<()> {
    match indices.dtype() {
        DType::Primitive(ptype, nullability) if ptype.is_unsigned_int() => {
            vortex_ensure!(
                !nullability.is_nullable(),
                "TakeSlicesArray {name} must be non-nullable, got {}",
                indices.dtype()
            );
            Ok(())
        }
        other => vortex_bail!(
            "TakeSlicesArray {name} must be a non-nullable unsigned integer, got {other}"
        ),
    }
}

pub(super) fn constant_index_value<T: IntegerPType>(
    name: &str,
    indices: &ArrayRef,
) -> VortexResult<Option<T>> {
    indices
        .as_constant()
        .map(|scalar| {
            scalar
                .as_primitive()
                .try_typed_value::<T>()?
                .ok_or_else(|| vortex_err!("TakeSlicesArray {name} constant value is null"))
        })
        .transpose()
}

pub(super) fn index_value_to_usize<T: IntegerPType>(name: &str, value: T) -> VortexResult<usize> {
    value
        .to_usize()
        .ok_or_else(|| vortex_err!("TakeSlicesArray {name} value {value} does not fit in usize"))
}

#[cfg(test)]
mod tests;
