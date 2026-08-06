// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::once;
use std::ops::Range;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_layout::plan::ChunkedPlan;
use vortex_layout::plan::DictPlan;
use vortex_layout::plan::ExpressionPlan;
use vortex_layout::plan::PlanRef;
use vortex_layout::plan::RowIdxPartitionPlan;
use vortex_layout::plan::RowIdxPlan;
use vortex_layout::plan::StructPlan;
use vortex_layout::plan::ZonedPlan;
use vortex_scan::selection::Selection;

const IDEAL_SPLIT_SIZE: u64 = 100_000;
const MAX_RANGE_SIZE: u64 = IDEAL_SPLIT_SIZE / 25;
const MIN_GAP_BETWEEN_RANGES: u64 = IDEAL_SPLIT_SIZE / 2;

/// Defines how a plan scan is divided into independently executable row ranges.
#[derive(Default, Copy, Clone, Debug)]
pub enum SplitBy {
    /// Uses boundaries exposed by the optimized physical plan.
    #[default]
    Layout,
    /// Splits every `n` rows.
    RowCount(usize),
}

impl SplitBy {
    pub(crate) fn splits(
        &self,
        plans: &[&PlanRef],
        row_range: &Range<u64>,
    ) -> VortexResult<Vec<u64>> {
        let mut boundaries = match *self {
            Self::Layout => {
                let mut boundaries = vec![row_range.start];
                for plan in plans {
                    collect_plan_splits(plan, 0, row_range, &mut boundaries)?;
                }
                boundaries
            }
            Self::RowCount(row_count) => {
                vortex_ensure!(row_count > 0, "Row-count split size must be non-zero");
                row_range
                    .clone()
                    .step_by(row_count)
                    .chain(once(row_range.end))
                    .collect()
            }
        };
        boundaries.sort_unstable();
        boundaries.dedup();
        Ok(subdivide_large_spans(boundaries, IDEAL_SPLIT_SIZE))
    }
}

fn collect_plan_splits(
    plan: &PlanRef,
    row_offset: u64,
    row_range: &Range<u64>,
    boundaries: &mut Vec<u64>,
) -> VortexResult<()> {
    if plan.is::<ExpressionPlan>() || plan.is::<RowIdxPlan>() || plan.is::<ZonedPlan>() {
        if let Some(child) = plan.child(0)? {
            collect_plan_splits(&child, row_offset, row_range, boundaries)?;
        }
        return Ok(());
    }

    if plan.is::<DictPlan>() {
        if let Some(codes) = plan.child(0)? {
            collect_plan_splits(&codes, row_offset, row_range, boundaries)?;
        }
        return Ok(());
    }

    if plan.is::<StructPlan>() || plan.is::<RowIdxPartitionPlan>() {
        for index in 0..plan.child_count() {
            if let Some(child) = plan.child(index)?
                && child.row_count() == plan.row_count()
            {
                collect_plan_splits(&child, row_offset, row_range, boundaries)?;
            }
        }
        return Ok(());
    }

    if plan.is::<ChunkedPlan>() {
        let mut chunk_offset = 0_u64;
        for index in 0..plan.child_count() {
            let Some(chunk) = plan.child(index)? else {
                continue;
            };
            let chunk_end = chunk_offset
                .checked_add(chunk.row_count())
                .ok_or_else(|| vortex_error::vortex_err!("Chunk row offset overflow"))?;
            let start = row_range.start.max(chunk_offset);
            let end = row_range.end.min(chunk_end);
            if start < end {
                let child_range = start - chunk_offset..end - chunk_offset;
                collect_plan_splits(&chunk, row_offset + chunk_offset, &child_range, boundaries)?;
                boundaries.push(row_offset + end);
            }
            chunk_offset = chunk_end;
        }
        return Ok(());
    }

    boundaries.push(row_offset + row_range.end);
    Ok(())
}

fn subdivide_large_spans(boundaries: Vec<u64>, max_span: u64) -> Vec<u64> {
    if boundaries.len() < 2
        || boundaries
            .windows(2)
            .all(|window| window[1] - window[0] <= max_span)
    {
        return boundaries;
    }

    let mut output = Vec::with_capacity(boundaries.len() * 2);
    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        output.push(start);
        let span = end - start;
        if span > max_span {
            let split_count = span.div_ceil(max_span);
            let split_size = span.div_ceil(split_count);
            let mut point = start + split_size;
            while point < end {
                output.push(point);
                point = point.saturating_add(split_size);
            }
        }
    }
    if let Some(&last) = boundaries.last() {
        output.push(last);
    }
    output
}

pub(crate) enum Splits {
    Natural(Vec<u64>),
    Ranges(Vec<Range<u64>>),
}

pub(crate) fn attempt_split_ranges(
    selection: &Selection,
    row_range: Option<&Range<u64>>,
) -> Option<Vec<Range<u64>>> {
    let Selection::IncludeByIndex(buffer) = selection else {
        return None;
    };
    if row_range.is_some() {
        return None;
    }
    let indices = buffer.as_slice();
    if indices.is_empty() {
        return Some(Vec::new());
    }

    let mut ranges = Vec::with_capacity((indices.len() as u64 / MAX_RANGE_SIZE) as usize);
    let mut current_start = indices[0];
    let mut current_end = indices[0] + 1;
    for &index in &indices[1..] {
        let new_range_size = (index + 1) - current_start;
        let gap = (index + 1) - current_end;
        if new_range_size >= MAX_RANGE_SIZE {
            if gap < MIN_GAP_BETWEEN_RANGES {
                return None;
            }
            ranges.push(current_start..current_end);
            current_start = index;
        }
        current_end = index + 1;
    }
    ranges.push(current_start..current_end);
    Some(ranges)
}
