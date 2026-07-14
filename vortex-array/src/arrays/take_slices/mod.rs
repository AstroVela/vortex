// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lazy gather of contiguous child ranges.
//!
//! `TakeSlicesArray` represents the concatenation of
//! `values[starts[i]..starts[i] + lengths[i]]` for each range row. Ranges may overlap, repeat,
//! and appear in any order.
//!
//! This is equivalent to a `Take` with concatenated sequence codes, but keeps the selectors
//! proportional to the number of ranges instead of the output length. A run-end encoded code array
//! cannot represent a general range compactly because each output position within a range selects
//! a different child index.

mod array;
mod kernel;
mod vtable;

pub use array::TakeSlicesArrayExt;
use itertools::Itertools as _;
pub use kernel::TakeSlicesExecuteAdaptor;
pub use kernel::TakeSlicesKernel;
pub use kernel::TakeSlicesReduce;
pub use kernel::TakeSlicesReduceAdaptor;
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

pub(super) fn index_value_to_usize<T: IntegerPType>(name: &str, value: T) -> VortexResult<usize> {
    value
        .to_usize()
        .ok_or_else(|| vortex_err!("TakeSlicesArray {name} value {value} does not fit in usize"))
}

pub(super) fn checked_range_end(start: usize, length: usize) -> VortexResult<usize> {
    start.checked_add(length).ok_or_else(|| {
        vortex_err!("TakeSlicesArray range overflow for start {start} and length {length}")
    })
}

pub(super) fn validate_index_ranges<S, L>(
    child_len: usize,
    starts: &[S],
    lengths: &[L],
    output_len: usize,
) -> VortexResult<()>
where
    S: IntegerPType,
    L: IntegerPType,
{
    let mut produced_len = 0usize;
    for (&start, &length) in starts.iter().zip_eq(lengths) {
        let start = index_value_to_usize("start", start)?;
        let length = index_value_to_usize("length", length)?;
        let end = checked_range_end(start, length)?;
        vortex_ensure!(
            end <= child_len,
            "TakeSlicesArray range {start}..{end} exceeds child array length {child_len}",
        );
        produced_len = produced_len
            .checked_add(length)
            .ok_or_else(|| vortex_err!("TakeSlicesArray produced length overflow"))?;
    }
    vortex_ensure!(
        produced_len == output_len,
        "TakeSlicesArray produced length {produced_len} does not match declared length {output_len}",
    );
    Ok(())
}

#[cfg(test)]
mod tests;
