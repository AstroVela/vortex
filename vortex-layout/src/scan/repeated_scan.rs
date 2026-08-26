// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp;
use std::iter;
use std::ops::Range;
use std::sync::Arc;

use async_stream::try_stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use itertools::Either;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReaderRef;
use crate::scan::filter::FilterExpr;
use crate::scan::splits::Splits;
use crate::scan::tasks::TaskContext;
use crate::scan::tasks::filter_exec;
use crate::scan::tasks::project_exec;
use crate::scan::tasks::split_exec;

/// A projected subset (by indices, range, and filter) of rows from a Vortex data source.
///
/// The method of this struct enable, possibly concurrent, scanning of multiple row ranges of this
/// data source.
pub struct RepeatedScan<A: 'static + Send> {
    session: VortexSession,
    layout_reader: LayoutReaderRef,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    ordered: bool,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    selection: Selection,
    /// The natural splits of the file.
    splits: Splits,
    /// The number of splits to make progress on concurrently **per-thread**.
    concurrency: usize,
    /// Function to apply to each [`ArrayRef`] within the spawned split tasks.
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    /// Maximal number of rows to read (after filtering)
    limit: Option<u64>,
    /// The dtype of the projected arrays.
    dtype: DType,
}

impl RepeatedScan<ArrayRef> {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn execute_array_iter<B: BlockingRuntime>(
        &self,
        row_range: Option<Range<u64>>,
        runtime: &B,
    ) -> VortexResult<impl ArrayIterator + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream(row_range)?;
        let iter = runtime.block_on_stream(stream);
        Ok(ArrayIteratorAdapter::new(dtype, iter))
    }

    pub fn execute_array_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<impl ArrayStream + Send + 'static> {
        let dtype = self.dtype.clone();
        let stream = self.execute_stream(row_range)?;
        Ok(ArrayStreamAdapter::new(dtype, stream))
    }
}

impl<A: 'static + Send> RepeatedScan<A> {
    /// Constructor just to allow `scan_builder` to create a `RepeatedScan`.
    #[expect(
        clippy::too_many_arguments,
        reason = "all arguments are needed for scan construction"
    )]
    pub fn new(
        session: VortexSession,
        layout_reader: LayoutReaderRef,
        projection: BoundExpression,
        filter: Option<BoundExpression>,
        ordered: bool,
        row_range: Option<Range<u64>>,
        selection: Selection,
        splits: Splits,
        concurrency: usize,
        map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
        limit: Option<u64>,
        dtype: DType,
    ) -> Self {
        Self {
            session,
            layout_reader,
            projection,
            filter,
            ordered,
            row_range,
            selection,
            splits,
            concurrency,
            map_fn,
            limit,
            dtype,
        }
    }

    pub fn execute(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        let row_range = self.effective_row_range(row_range);

        let ranges = match &self.splits {
            Splits::Natural(boundaries) => {
                Either::Left(natural_ranges(boundaries, row_range.as_ref()).into_iter())
            }
            // `execute_stream` uses the staged path. Keep `execute` correct for callers that ask
            // for the task list directly, using projection boundaries as coupled splits.
            Splits::FilterProjection { projection, .. } => {
                Either::Left(natural_ranges(projection, row_range.as_ref()).into_iter())
            }
            Splits::Ranges(ranges) => Either::Right(match row_range.as_ref() {
                None => Either::Left(ranges.iter().cloned()),
                Some(range) => {
                    if range.is_empty() {
                        return Ok(Vec::new());
                    }
                    Either::Right(ranges.iter().filter_map(move |r| {
                        let start = cmp::max(r.start, range.start);
                        let end = cmp::min(r.end, range.end);
                        (start < end).then_some(start..end)
                    }))
                }
            }),
        };

        let mut limit = self.limit;
        let mut tasks = Vec::new();
        let ctx = Arc::new(TaskContext {
            filter: self.filter.clone().map(|f| Arc::new(FilterExpr::new(f))),
            reader: Arc::clone(&self.layout_reader),
            projection: self.projection.clone(),
            mapper: Arc::clone(&self.map_fn),
        });

        for range in ranges {
            let row_mask = self.selection.row_mask(&range);
            if row_mask.mask().all_false() {
                continue;
            }

            tasks.push(split_exec(Arc::clone(&ctx), row_mask, limit.as_mut())?);
            if limit.is_some_and(|l| l == 0) {
                break;
            }
        }

        Ok(tasks)
    }

    pub(crate) fn has_separate_filter_projection_splits(&self) -> bool {
        matches!(self.splits, Splits::FilterProjection { .. })
    }

    pub fn execute_stream(
        &self,
        row_range: Option<Range<u64>>,
    ) -> VortexResult<BoxStream<'static, VortexResult<A>>> {
        if let Splits::FilterProjection { filter, projection } = &self.splits {
            return self.execute_filter_projection_stream(row_range, filter, projection);
        }

        let num_workers = get_available_parallelism().unwrap_or(1);
        let concurrency = self.concurrency * num_workers;
        let handle = self.session.handle();

        let stream =
            futures::stream::iter(self.execute(row_range)?).map(move |task| handle.spawn(task));

        let stream = if self.ordered {
            stream.buffered(concurrency).boxed()
        } else {
            stream.buffer_unordered(concurrency).boxed()
        };

        Ok(stream
            .filter_map(|chunk| async move { chunk.transpose() })
            .boxed())
    }

    fn execute_filter_projection_stream(
        &self,
        row_range: Option<Range<u64>>,
        filter_boundaries: &[u64],
        projection_boundaries: &[u64],
    ) -> VortexResult<BoxStream<'static, VortexResult<A>>> {
        let row_range = self.effective_row_range(row_range);
        let filter_ranges = natural_ranges(filter_boundaries, row_range.as_ref());
        let projection_ranges = natural_ranges(projection_boundaries, row_range.as_ref());

        if filter_ranges.is_empty() || projection_ranges.is_empty() {
            return Ok(futures::stream::empty().boxed());
        }

        let ctx = Arc::new(TaskContext {
            filter: self.filter.clone().map(|f| Arc::new(FilterExpr::new(f))),
            reader: Arc::clone(&self.layout_reader),
            projection: self.projection.clone(),
            mapper: Arc::clone(&self.map_fn),
        });

        // Build filter evaluations eagerly so the readers can register all filter I/O before the
        // tasks begin making progress. Buffering remains ordered because projection masks cannot
        // be assembled until all preceding filter ranges are known.
        let mut filter_tasks = Vec::with_capacity(filter_ranges.len());
        for range in filter_ranges {
            let row_mask = self.selection.row_mask(&range);
            filter_tasks.push(filter_exec(Arc::clone(&ctx), row_mask)?);
        }

        let num_workers = get_available_parallelism().unwrap_or(1);
        let concurrency = self.concurrency * num_workers;
        let handle = self.session.handle();
        let filter_handle = handle.clone();
        let filtered_masks = futures::stream::iter(filter_tasks)
            .map(move |task: BoxFuture<'static, VortexResult<RowMask>>| filter_handle.spawn(task))
            .buffered(concurrency)
            .boxed();

        let projection_tasks = try_stream! {
            let mut repartitioner = ProjectionMaskRepartitioner::new(projection_ranges);
            futures::pin_mut!(filtered_masks);

            while let Some(filtered_mask) = filtered_masks.try_next().await? {
                for projection_mask in repartitioner.push(filtered_mask)? {
                    if !projection_mask.mask().all_false() {
                        yield project_exec(Arc::clone(&ctx), projection_mask)?;
                    }
                }
            }

            repartitioner.finish()?;
        };

        let projection_tasks = projection_tasks
            .map_ok(move |task: BoxFuture<'static, VortexResult<Option<A>>>| handle.spawn(task));
        let projected = if self.ordered {
            projection_tasks.try_buffered(concurrency).boxed()
        } else {
            projection_tasks.try_buffer_unordered(concurrency).boxed()
        };

        Ok(projected
            .filter_map(|chunk| async move { chunk.transpose() })
            .boxed())
    }

    fn effective_row_range(&self, row_range: Option<Range<u64>>) -> Option<Range<u64>> {
        let selection_range: Option<Range<u64>> = match &self.selection {
            Selection::IncludeByIndex(buf) if !buf.is_empty() => {
                Some(buf[0]..buf[buf.len() - 1] + 1)
            }
            Selection::IncludeRoaring(roaring) if !roaring.is_empty() => {
                Some(roaring.min().vortex_expect("empty")..roaring.max().vortex_expect("empty") + 1)
            }
            _ => None,
        };
        let row_range = intersect_ranges(self.row_range.as_ref(), row_range);
        intersect_ranges(row_range.as_ref(), selection_range)
    }
}

struct ProjectionMaskRepartitioner {
    ranges: std::vec::IntoIter<Range<u64>>,
    current: Option<Range<u64>>,
    next_row: Option<u64>,
    fragments: Vec<Mask>,
}

impl ProjectionMaskRepartitioner {
    fn new(ranges: Vec<Range<u64>>) -> Self {
        let mut ranges = ranges.into_iter();
        let current = ranges.next();
        let next_row = current.as_ref().map(|range| range.start);
        Self {
            ranges,
            current,
            next_row,
            fragments: Vec::new(),
        }
    }

    fn push(&mut self, filtered: RowMask) -> VortexResult<Vec<RowMask>> {
        let filtered_range = filtered.row_range();
        vortex_ensure!(
            self.next_row == Some(filtered_range.start),
            "non-contiguous filter mask: expected row {:?}, got {}",
            self.next_row,
            filtered_range.start
        );

        let mut completed = Vec::new();
        let mut source_start = 0;
        while source_start < filtered.mask().len() {
            let projection_range = self
                .current
                .as_ref()
                .vortex_expect("filter masks exceed projection ranges");
            let next_row = self.next_row.vortex_expect("current projection range");
            vortex_ensure!(
                next_row >= projection_range.start && next_row < projection_range.end,
                "invalid projection mask cursor {next_row} for range {projection_range:?}"
            );

            let projection_remaining = usize::try_from(projection_range.end - next_row)?;
            let fragment_len = projection_remaining.min(filtered.mask().len() - source_start);
            self.fragments.push(
                filtered
                    .mask()
                    .slice(source_start..source_start + fragment_len),
            );
            source_start += fragment_len;
            let next_row = next_row + fragment_len as u64;
            self.next_row = Some(next_row);

            if next_row == projection_range.end {
                let projection_start = projection_range.start;
                let mask = if self.fragments.len() == 1 {
                    self.fragments.pop().vortex_expect("one mask fragment")
                } else {
                    Mask::from_iter(std::mem::take(&mut self.fragments))
                };
                completed.push(RowMask::new(projection_start, mask));

                self.current = self.ranges.next();
                self.next_row = self.current.as_ref().map(|range| range.start);
                if let Some(next_projection) = &self.current {
                    vortex_ensure!(
                        next_projection.start == next_row,
                        "non-contiguous projection ranges: expected {next_row}, got {}",
                        next_projection.start
                    );
                }
            }
        }

        Ok(completed)
    }

    fn finish(self) -> VortexResult<()> {
        vortex_ensure!(
            self.current.is_none() && self.fragments.is_empty(),
            "filter masks ended before projection ranges"
        );
        Ok(())
    }
}

fn natural_ranges(boundaries: &[u64], row_range: Option<&Range<u64>>) -> Vec<Range<u64>> {
    debug_assert!(boundaries.is_sorted());
    let splits_iter = match row_range {
        None => Either::Left(boundaries.iter().copied()),
        Some(range) => {
            if range.is_empty() {
                return Vec::new();
            }
            let lo = boundaries.partition_point(|&x| x <= range.start);
            let hi = boundaries.partition_point(|&x| x < range.end);
            Either::Right(
                iter::once(range.start)
                    .chain(boundaries[lo..hi].iter().copied())
                    .chain(iter::once(range.end)),
            )
        }
    };

    splits_iter
        .tuple_windows()
        .map(|(start, end)| start..end)
        .collect()
}

fn intersect_ranges(left: Option<&Range<u64>>, right: Option<Range<u64>>) -> Option<Range<u64>> {
    match (left, right) {
        (None, None) => None,
        (None, Some(r)) => Some(r),
        (Some(l), None) => Some(l.clone()),
        (Some(l), Some(r)) => Some(cmp::max(l.start, r.start)..cmp::min(l.end, r.end)),
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_scan::row_mask::RowMask;

    use super::ProjectionMaskRepartitioner;
    use super::natural_ranges;

    #[test]
    fn splits_one_filter_mask_across_projection_ranges() -> VortexResult<()> {
        let input = Mask::from_iter([
            true, false, true, true, false, false, true, false, false, true,
        ]);
        let mut repartitioner = ProjectionMaskRepartitioner::new(vec![2..5, 5..9, 9..12]);

        let output = repartitioner.push(RowMask::new(2, input))?;
        repartitioner.finish()?;

        assert_eq!(output.len(), 3);
        assert_eq!(output[0].row_range(), 2..5);
        assert_eq!(output[0].mask(), &Mask::from_iter([true, false, true]));
        assert_eq!(output[1].row_range(), 5..9);
        assert_eq!(
            output[1].mask(),
            &Mask::from_iter([true, false, false, true])
        );
        assert_eq!(output[2].row_range(), 9..12);
        assert_eq!(output[2].mask(), &Mask::from_iter([false, false, true]));
        Ok(())
    }

    #[test]
    fn combines_filter_masks_into_projection_ranges() -> VortexResult<()> {
        let mut repartitioner = ProjectionMaskRepartitioner::new(vec![2..8, 8..12]);
        let mut output = Vec::new();

        output.extend(repartitioner.push(RowMask::new(2, Mask::from_iter([true, false])))?);
        output.extend(repartitioner.push(RowMask::new(4, Mask::from_iter([true, true, false])))?);
        output.extend(repartitioner.push(RowMask::new(
            7,
            Mask::from_iter([false, true, false, false, true]),
        ))?);
        repartitioner.finish()?;

        assert_eq!(output.len(), 2);
        assert_eq!(output[0].row_range(), 2..8);
        assert_eq!(
            output[0].mask(),
            &Mask::from_iter([true, false, true, true, false, false])
        );
        assert_eq!(output[1].row_range(), 8..12);
        assert_eq!(
            output[1].mask(),
            &Mask::from_iter([true, false, false, true])
        );
        Ok(())
    }

    #[test]
    fn natural_ranges_are_clipped_without_gaps() {
        assert_eq!(
            natural_ranges(&[0, 4, 9, 12], Some(&(2..10))),
            [2..4, 4..9, 9..10]
        );
    }
}
