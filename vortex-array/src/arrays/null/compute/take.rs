// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Null;
use crate::arrays::NullArray;
use crate::arrays::Primitive;
use crate::arrays::dict::TakeReduce;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::match_each_integer_ptype;
use crate::optimizer::rules::ParentRuleSet;

impl TakeReduce for Null {
    #[expect(clippy::cast_possible_truncation)]
    fn take(array: ArrayView<'_, Null>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // This rule is metadata-only: it can bounds-check indices that are already decoded, but
        // must defer encoded indices to the compute path rather than executing them here.
        let Some(indices) = indices.as_opt::<Primitive>() else {
            return Ok(None);
        };

        // Enforce all indices are valid
        match_each_integer_ptype!(indices.ptype(), |T| {
            for index in indices.as_slice::<T>() {
                if (*index as usize) >= array.len() {
                    vortex_bail!(OutOfBounds: *index as usize, 0, array.len());
                }
            }
        });

        Ok(Some(NullArray::new(indices.len()).into_array()))
    }
}

impl Null {
    pub const TAKE_RULES: ParentRuleSet<Self> =
        ParentRuleSet::new(&[ParentRuleSet::lift(&TakeReduceAdaptor::<Self>(Self))]);
}
