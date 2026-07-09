// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::legacy_execution_ctx;
use super::trivial_take_slices;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::TakeSlices;
use crate::arrays::TakeSlicesArray;
use crate::arrays::take_slices::TakeSlicesArrayExt;
use crate::arrays::take_slices::TakeSlicesReduce;
use crate::arrays::take_slices::TakeSlicesReduceAdaptor;
use crate::arrays::take_slices::selector_slices;
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
        trivial_take_slices(array.child(), array.starts(), array.lengths())
    }
}

impl TakeSlicesReduce for TakeSlices {
    fn take_slices(
        array: ArrayView<'_, Self>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>> {
        let mut ctx = legacy_execution_ctx();
        let inner = selector_slices(
            array.child().len(),
            array.starts(),
            array.lengths(),
            &mut ctx,
        )?;
        let outer = selector_slices(array.len(), starts, lengths, &mut ctx)?;
        let combined = project_slices(&inner, &outer);
        let (starts, lengths) = selectors_from_slices(&combined)?;
        Ok(Some(
            TakeSlicesArray::try_new(array.child().clone(), starts, lengths)?.into_array(),
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

fn selectors_from_slices(slices: &[(usize, usize)]) -> VortexResult<(ArrayRef, ArrayRef)> {
    let starts = slices
        .iter()
        .map(|&(start, _)| selector_value(start, "start"))
        .collect::<VortexResult<Vec<_>>>()?;
    let lengths = slices
        .iter()
        .map(|&(start, end)| selector_value(end - start, "length"))
        .collect::<VortexResult<Vec<_>>>()?;
    Ok((
        PrimitiveArray::from_iter(starts).into_array(),
        PrimitiveArray::from_iter(lengths).into_array(),
    ))
}

fn selector_value(value: usize, name: &str) -> VortexResult<u64> {
    u64::try_from(value)
        .map_err(|_| vortex_err!("TakeSlicesArray projected {name} {value} does not fit in u64"))
}
