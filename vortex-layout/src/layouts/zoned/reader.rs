// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use itertools::Itertools;
use tracing::trace;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::pruning::PruningState;
use crate::segments::SegmentSource;

pub struct ZonedReader {
    layout: ZonedLayout,
    name: Arc<str>,
    lazy_children: Arc<LazyReaderChildren>,
    pruning: PruningState,
}

impl ZonedReader {
    pub(super) fn try_new(
        layout: ZonedLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: crate::LayoutReaderContext,
    ) -> VortexResult<Self> {
        let aggregate_fns = layout.aggregate_fns(&session)?;
        let dtypes = vec![
            layout.dtype.clone(),
            layout.stats_table_dtype_for(&aggregate_fns),
        ];
        let names = vec![Arc::clone(&name), format!("{}.zones", name).into()];
        let lazy_children = Arc::new(LazyReaderChildren::new(
            Arc::clone(&layout.children),
            dtypes,
            names,
            Arc::clone(&segment_source),
            session.clone(),
            ctx,
        ));

        Ok(Self {
            pruning: PruningState::new(&layout, aggregate_fns, Arc::clone(&lazy_children), session),
            layout,
            name,
            lazy_children,
        })
    }

    fn data_child(&self) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(0)
    }

    /// Get the range of zone IDs containing a row range.
    pub(crate) fn zone_range(&self, row_range: &Range<u64>) -> Range<u64> {
        // Caller must ensure zone_len > 0. Legacy files may deserialize with zone_len == 0, but
        // pruning_evaluation disables zoned pruning for those layouts before calling this helper.
        debug_assert!(self.layout.zone_len > 0, "zone_len must be > 0");

        let zone_len_u64 = self.layout.zone_len as u64;
        let zone_start = row_range.start / zone_len_u64;
        let zone_end = row_range.end.div_ceil(zone_len_u64);
        zone_start..zone_end
    }

    /// Get the row index for the first row in a zone with the given `zone_index`.
    pub(crate) fn first_row_offset(&self, zone_idx: u64) -> u64 {
        zone_idx
            .saturating_mul(self.layout.zone_len as u64)
            .min(self.layout.row_count())
    }

    fn zone_row_lengths(&self, row_range: &Range<u64>) -> VortexResult<(Range<u64>, Vec<usize>)> {
        let row_count = row_range.end - row_range.start;
        let zone_range = self.zone_range(row_range);
        let zone_lengths = zone_range
            .clone()
            .map(|zone_idx| {
                let start = usize::try_from(
                    self.first_row_offset(zone_idx)
                        .saturating_sub(row_range.start),
                )?;
                let end = usize::try_from(
                    self.first_row_offset(zone_idx + 1)
                        .saturating_sub(row_range.start)
                        .min(row_count),
                )?;
                Ok::<_, VortexError>(end - start)
            })
            .try_collect()?;

        Ok((zone_range, zone_lengths))
    }

    fn expand_zone_mask(&self, row_range: &Range<u64>, zone_mask: &Mask) -> VortexResult<Mask> {
        let (zone_range, zone_lengths) = self.zone_row_lengths(row_range)?;
        let row_mask_len = usize::try_from(row_range.end - row_range.start)?;
        let mut builder = BitBufferMut::with_capacity(row_mask_len);
        for (zone_idx, &zone_length) in zone_range.zip_eq(&zone_lengths) {
            builder.append_n(zone_mask.value(usize::try_from(zone_idx)?), zone_length);
        }

        let row_mask = Mask::from(builder.freeze());
        assert_eq!(row_mask.len(), row_mask_len, "Mask length mismatch");
        Ok(row_mask)
    }
}

impl LayoutReader for ZonedReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        self.data_child()?
            .register_splits(field_mask, split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        trace!("Stats pruning evaluation: {} - {}", &self.name, expr);
        let data_eval = self
            .data_child()?
            .pruning_evaluation(row_range, expr, mask.clone())?;

        if self.layout.zone_len == 0 {
            trace!("Stats pruning evaluation: skipping zoned pruning for legacy zero-length zones");
            return Ok(data_eval);
        }

        let Some(pruning_mask_future) = self.pruning.pruning_mask_future(expr.clone()) else {
            trace!("Stats pruning evaluation: not prune-able {expr}");
            return Ok(data_eval);
        };

        let (zone_range, zone_lengths) = self.zone_row_lengths(row_range)?;

        let name = Arc::clone(&self.name);
        let expr = expr.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            trace!("Invoking stats pruning evaluation {}: {}", name, expr);

            let pruning_mask = pruning_mask_future.await?.mask()?;

            let mut builder = BitBufferMut::with_capacity(mask.len());
            for (zone_idx, &zone_length) in zone_range.clone().zip_eq(&zone_lengths) {
                builder.append_n(!pruning_mask.value(usize::try_from(zone_idx)?), zone_length);
            }

            let stats_mask = Mask::from(builder.freeze());
            assert_eq!(stats_mask.len(), mask.len(), "Mask length mismatch");

            // Intersect the masks.
            let mask_density = mask.density();
            let mut stats_mask = mask.bitand(&stats_mask);

            // Forward to data child for further pruning.
            if !stats_mask.all_false() {
                let data_mask = data_eval.await?;
                stats_mask = stats_mask.bitand(&data_mask);
            }

            trace!(
                "Stats evaluation approx {} - {} (mask = {}) => {}",
                name,
                expr,
                mask_density,
                stats_mask.density(),
            );

            Ok(stats_mask)
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        if self.layout.zone_len == 0 {
            return self.data_child()?.filter_evaluation(row_range, expr, mask);
        }

        let Some(satisfy_mask_future) = self.pruning.satisfy_mask_future(expr.clone()) else {
            return self.data_child()?.filter_evaluation(row_range, expr, mask);
        };

        // The zone map is normally resolved by pruning before filter evaluation. If it or the
        // input mask is not ready, delegate to preserve eager child prefetching.
        let Some(Ok(satisfied_zones)) = satisfy_mask_future.clone().now_or_never() else {
            return self.data_child()?.filter_evaluation(row_range, expr, mask);
        };
        let Some(Ok(input)) = mask.clone().now_or_never() else {
            return self.data_child()?.filter_evaluation(row_range, expr, mask);
        };

        let satisfied_rows = self.expand_zone_mask(row_range, &satisfied_zones)?;
        let child_input = input.clone().bitand_not(&satisfied_rows);
        if child_input.all_false() {
            return Ok(MaskFuture::ready(input));
        }

        let passthrough = input.bitand(&satisfied_rows);
        let child_result = self.data_child()?.filter_evaluation(
            row_range,
            expr,
            MaskFuture::ready(child_input),
        )?;

        Ok(MaskFuture::new(passthrough.len(), async move {
            Ok(child_result.await?.bitor(&passthrough))
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        // TODO(ngates): there are some projection expressions that we may also be able to
        //  short-circuit with statistics.
        self.data_child()?
            .projection_evaluation(row_range, expr, mask)
    }
}

#[cfg(test)]
mod test {
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_io::runtime::Handle;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::IntoLayout;
    use crate::LayoutBuildContext;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::VTable;
    use crate::children::OwnedLayoutChildren;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::zoned::LegacyStats;
    use crate::layouts::zoned::LegacyStatsLayoutEncoding;
    use crate::layouts::zoned::LegacyStatsMetadata;
    use crate::layouts::zoned::Zoned;
    use crate::layouts::zoned::writer::ZonedLayoutOptions;
    use crate::layouts::zoned::writer::ZonedStrategy;
    use crate::segments::SegmentFuture;
    use crate::segments::SegmentId;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;

    struct CountingSegmentSource {
        inner: Arc<dyn SegmentSource>,
        request_count: Arc<AtomicUsize>,
    }

    impl SegmentSource for CountingSegmentSource {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            self.request_count.fetch_add(1, Ordering::Relaxed);
            self.inner.request(id)
        }
    }

    fn counting_source(
        inner: Arc<dyn SegmentSource>,
    ) -> (Arc<dyn SegmentSource>, Arc<AtomicUsize>) {
        let request_count = Arc::new(AtomicUsize::new(0));
        let source: Arc<dyn SegmentSource> = Arc::new(CountingSegmentSource {
            inner,
            request_count: Arc::clone(&request_count),
        });
        (source, request_count)
    }

    fn session_with_handle(handle: Handle) -> VortexSession {
        array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle)
    }

    #[fixture]
    /// Create a stats layout with three chunks of primitive arrays.
    fn stats_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
                ..Default::default()
            },
        );
        let array_stream = ChunkedArray::from_iter([
            buffer![1, 2, 3].into_array(),
            buffer![4, 5, 6].into_array(),
            buffer![7, 8, 9].into_array(),
        ])
        .into_array()
        .to_array_stream()
        .sequenced(ptr);
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = session_with_handle(handle);
            strategy
                .write_stream(ctx, segments2, array_stream, eof, &session)
                .await
        })
        .unwrap();
        (segments, layout)
    }

    #[fixture]
    fn nullable_stats_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
                ..Default::default()
            },
        );
        let array_stream = PrimitiveArray::new(
            buffer![0i32, 4, 5],
            Validity::from_iter([false, true, true]),
        )
        .into_array()
        .to_array_stream()
        .sequenced(ptr);
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = session_with_handle(handle);
            strategy
                .write_stream(ctx, segments2, array_stream, eof, &session)
                .await
        })
        .unwrap();
        (segments, layout)
    }

    #[rstest]
    fn test_stats_evaluator(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        block_on(|handle| async {
            let mut ctx = array_session().create_execution_ctx();
            let session = session_with_handle(handle);
            let result = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &root(),
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            let expected = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
            assert_arrays_eq!(result, expected, &mut ctx);
        })
    }

    #[rstest]
    fn test_stats_pruning_mask(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        block_on(|handle| async {
            let row_count = layout.row_count();
            let session = session_with_handle(handle);
            let reader = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap();

            // Choose a prune-able expression
            let expr = gt(root(), lit(7));

            let result = reader
                .pruning_evaluation(
                    &(0..row_count),
                    &expr,
                    Mask::new_true(row_count.try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            assert_eq!(
                result,
                Mask::from_iter([false, false, false, false, false, false, true, true, true])
            );
        })
    }

    #[rstest]
    fn satisfied_zones_skip_data_child(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        block_on(|handle| async {
            let row_count = layout.row_count();
            let (source, request_count) = counting_source(segments);
            let session = session_with_handle(handle);
            let reader = layout.new_reader("".into(), source, &session, &Default::default())?;
            let input = Mask::new_true(usize::try_from(row_count)?);

            reader
                .pruning_evaluation(&(0..row_count), &gt(root(), lit(10)), input.clone())?
                .await?;
            let requests_before_filter = request_count.load(Ordering::Relaxed);

            let result = reader
                .filter_evaluation(
                    &(0..row_count),
                    &gt(root(), lit(0)),
                    MaskFuture::ready(input.clone()),
                )?
                .await?;

            assert_eq!(result, input);
            assert_eq!(
                request_count.load(Ordering::Relaxed),
                requests_before_filter,
                "a fully satisfied filter must not read the data child"
            );
            Ok(())
        })
    }

    #[rstest]
    fn partial_satisfaction_stitches(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        block_on(|handle| async {
            let row_count = layout.row_count();
            let (source, request_count) = counting_source(segments);
            let session = session_with_handle(handle);
            let reader = layout.new_reader("".into(), source, &session, &Default::default())?;
            let input = Mask::new_true(usize::try_from(row_count)?);

            reader
                .pruning_evaluation(&(0..row_count), &gt(root(), lit(10)), input.clone())?
                .await?;
            let requests_before_filter = request_count.load(Ordering::Relaxed);

            let result = reader
                .filter_evaluation(
                    &(0..row_count),
                    &gt(root(), lit(3)),
                    MaskFuture::ready(input),
                )?
                .await?;

            assert_eq!(
                result,
                Mask::from_iter([false, false, false, true, true, true, true, true, true])
            );
            assert_eq!(
                request_count.load(Ordering::Relaxed),
                requests_before_filter + 1,
                "only the unsatisfied zone should read a data segment"
            );
            Ok(())
        })
    }

    #[rstest]
    fn nullable_zone_with_nulls_not_skipped(
        #[from(nullable_stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        block_on(|handle| async {
            let row_count = layout.row_count();
            let session = session_with_handle(handle);
            let reader = layout.new_reader("".into(), segments, &session, &Default::default())?;
            let input = Mask::new_true(usize::try_from(row_count)?);

            reader
                .pruning_evaluation(&(0..row_count), &gt(root(), lit(10)), input.clone())?
                .await?;

            let result = reader
                .filter_evaluation(
                    &(0..row_count),
                    &gt(root(), lit(3)),
                    MaskFuture::ready(input),
                )?
                .await?;

            assert_eq!(result, Mask::from_iter([false, true, true]));
            Ok(())
        })
    }

    #[rstest]
    fn unwarmed_zone_map_keeps_filter_correct(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        block_on(|handle| async {
            let row_count = layout.row_count();
            let session = session_with_handle(handle);
            let reader = layout.new_reader("".into(), segments, &session, &Default::default())?;

            let result = reader
                .filter_evaluation(
                    &(0..row_count),
                    &gt(root(), lit(3)),
                    MaskFuture::new_true(usize::try_from(row_count)?),
                )?
                .await?;

            assert_eq!(
                result,
                Mask::from_iter([false, false, false, true, true, true, true, true, true])
            );
            Ok(())
        })
    }

    #[test]
    fn test_default_zoned_null_count_pruning_mask() {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
                ..Default::default()
            },
        );
        let array_stream = ChunkedArray::from_iter([
            PrimitiveArray::new(
                buffer![0i32, 0, 0],
                Validity::from_iter([false, false, false]),
            )
            .into_array(),
            PrimitiveArray::new(buffer![1i32, 2, 3], Validity::from_iter([true, true, true]))
                .into_array(),
            PrimitiveArray::new(
                buffer![0i32, 0, 0],
                Validity::from_iter([false, false, false]),
            )
            .into_array(),
        ])
        .into_array()
        .to_array_stream()
        .sequenced(ptr);
        let segments2 = Arc::<TestSegments>::clone(&segments);

        let layout = block_on(|handle| async move {
            let session = session_with_handle(handle);
            strategy
                .write_stream(ctx, segments2, array_stream, eof, &session)
                .await
        })
        .unwrap();

        block_on(|handle| async {
            let row_count = layout.row_count();
            let session = session_with_handle(handle);
            let reader = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap();

            let result = reader
                .pruning_evaluation(
                    &(0..row_count),
                    &is_not_null(root()),
                    Mask::new_true(row_count.try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            assert_eq!(
                result,
                Mask::from_iter([false, false, false, true, true, true, false, false, false])
            );
        })
    }

    #[rstest]
    fn legacy_zero_zone_len_delegates_pruning_and_filter(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        let zoned_layout = layout.as_::<Zoned>();
        let children =
            OwnedLayoutChildren::layout_children(vec![layout.child(0)?, layout.child(1)?]);
        let session = array_session();
        let read_ctx = ReadContext::new([]);
        let build_ctx = LayoutBuildContext {
            session: &session,
            array_read_ctx: &read_ctx,
        };
        let legacy_layout = <LegacyStats as VTable>::build(
            &LegacyStatsLayoutEncoding,
            layout.dtype(),
            layout.row_count(),
            &LegacyStatsMetadata {
                zone_len: 0,
                zone_map_schema: zoned_layout.zone_map_schema.clone(),
            },
            vec![],
            children.as_ref(),
            &build_ctx,
        )?
        .into_layout();

        block_on(|handle| async {
            let row_count = legacy_layout.row_count();
            let session = session_with_handle(handle);
            let reader =
                legacy_layout.new_reader("".into(), segments, &session, &Default::default())?;

            let result = reader
                .pruning_evaluation(
                    &(0..row_count),
                    &gt(root(), lit(7)),
                    Mask::new_true(row_count.try_into().unwrap()),
                )?
                .await?;

            assert!(result.all_true());

            let result = reader
                .filter_evaluation(
                    &(0..row_count),
                    &gt(root(), lit(7)),
                    MaskFuture::new_true(usize::try_from(row_count)?),
                )?
                .await?;
            assert_eq!(
                result,
                Mask::from_iter([false, false, false, false, false, false, false, true, true])
            );
            Ok(())
        })
    }
}
