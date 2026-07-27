// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use crate::array::Array;
use crate::arrays::Primitive;
use crate::match_each_unsigned_integer_ptype;

pub(super) fn validate_interleave(array: &Array<Interleave>) -> VortexResult<()> {
    let array_indices = array.array_indices().as_::<Primitive>();
    let row_indices = array.row_indices().as_::<Primitive>();
    match_each_unsigned_integer_ptype!(array_indices.ptype(), |A| {
        match_each_unsigned_integer_ptype!(row_indices.ptype(), |R| {
            validate_selectors(
                array.num_values(),
                |branch| array.value(branch).len(),
                array_indices.as_slice::<A>(),
                row_indices.as_slice::<R>(),
            )
            .map(|_| ())
        })
    })
}

/// Validates selector lengths and bounds, returning the output length.
#[inline(always)]
pub(super) fn validate_selectors<A, R>(
    num_values: usize,
    value_len: impl Fn(usize) -> usize,
    branches: &[A],
    rows: &[R],
) -> VortexResult<usize>
where
    A: AsPrimitive<usize>,
    R: AsPrimitive<usize>,
{
    let len = branches.len();
    vortex_ensure!(
        rows.len() == len,
        "interleave selectors differ in length: array_indices {len}, row_indices {}",
        rows.len()
    );

    for i in 0..len {
        let branch = branches[i].as_();
        vortex_ensure!(branch < num_values, "interleave array index out of bounds");
        vortex_ensure!(
            rows[i].as_() < value_len(branch),
            "interleave row index out of bounds"
        );
    }

    Ok(len)
}
