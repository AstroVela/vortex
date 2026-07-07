// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::trivial_take_slices;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::TakeSlices;
use crate::arrays::TakeSlicesArray;
use crate::arrays::take_slices::TakeSlicesArrayExt;
use crate::arrays::take_slices::TakeSlicesReduce;
use crate::arrays::take_slices::TakeSlicesReduceAdaptor;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::optimizer::rules::ReduceRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<TakeSlices> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&TakeSlicesReduceAdaptor(TakeSlices))]);

pub(super) const RULES: ReduceRuleSet<TakeSlices> = ReduceRuleSet::new(&[&TrivialTakeSlicesRule]);

#[derive(Debug)]
struct TrivialTakeSlicesRule;

impl ArrayReduceRule<TakeSlices> for TrivialTakeSlicesRule {
    fn reduce(&self, array: ArrayView<'_, TakeSlices>) -> VortexResult<Option<ArrayRef>> {
        trivial_take_slices(array.child(), array.slices())
    }
}

impl TakeSlicesReduce for TakeSlices {
    fn take_slices(
        array: ArrayView<'_, Self>,
        slices: &[(usize, usize)],
    ) -> VortexResult<Option<ArrayRef>> {
        let combined = project_slices(array.slices(), slices);
        Ok(Some(
            TakeSlicesArray::try_new(array.child().clone(), combined)?.into_array(),
        ))
    }
}

fn project_slices(inner: &[(usize, usize)], outer: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut projected = Vec::new();

    for &(outer_start, outer_end) in outer {
        let mut logical_start = 0usize;
        for &(inner_start, inner_end) in inner {
            let inner_len = inner_end - inner_start;
            let logical_end = logical_start + inner_len;

            if outer_start < logical_end && outer_end > logical_start {
                let overlap_start = outer_start.max(logical_start);
                let overlap_end = outer_end.min(logical_end);
                projected.push((
                    inner_start + (overlap_start - logical_start),
                    inner_start + (overlap_end - logical_start),
                ));
            }

            if logical_end >= outer_end {
                break;
            }
            logical_start = logical_end;
        }
    }

    projected
}
