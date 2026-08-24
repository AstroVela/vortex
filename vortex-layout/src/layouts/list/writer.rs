// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroU64;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::try_join;
use futures::future::try_join_all;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::List;
use vortex_array::arrays::ListView;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::list::ListArrayExt;
use vortex_array::arrays::list::ListDataParts;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::matcher::Matcher;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use super::repartition::repartition_list_elements;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::ListChunkBoundary;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStream;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Item carried on each child sub-stream: a sequenced, materialized chunk.
type ChildChunk = VortexResult<(SequenceId, ArrayRef)>;

struct ListChildSenders {
    elements: kanal::AsyncSender<ChildChunk>,
    offsets: kanal::AsyncSender<ChildChunk>,
    validity: Option<kanal::AsyncSender<ChildChunk>>,
}

#[derive(Default)]
struct ListTransposeState {
    element_base: u64,
    outer_base: u64,
    chunk_boundaries: Vec<ListChunkBoundary>,
    emitted_first_chunk: bool,
}

/// Strategy for writing list-typed arrays, with a fallback for non-list dtypes.
///
/// This is a *structural* writer that decomposes a list column into independent `elements`,
/// `offsets`, and (when nullable) `validity` sub-columns, each written through its own downstream
/// strategy, producing a single [`ListLayout`].
///
/// For list-typed input the strategy transposes the whole column stream into three sub-streams:
///  1. Each chunk is canonicalized to a [`ListArray`] (rebuilding a [`ListView`] via
///     [`list_from_list_view`] when necessary).
///  2. `offsets` are rebased to global `u64` positions (cumulative across chunks) so the single
///     `offsets` child indexes into the concatenated `elements` child.
///  3. `elements`, `offsets`, and `validity` are streamed to their child strategies concurrently.
///
/// For input whose dtype is not [`DType::List`], the stream is forwarded unchanged to the
/// configured `fallback` strategy.
///
/// [`ListArray`]: vortex_array::arrays::ListArray
#[derive(Clone)]
pub struct ListLayoutStrategy {
    elements: Arc<dyn LayoutStrategy>,
    offsets: Arc<dyn LayoutStrategy>,
    validity: Arc<dyn LayoutStrategy>,
    fallback: Arc<dyn LayoutStrategy>,
    element_repartition_target: Option<NonZeroU64>,
}

impl Default for ListLayoutStrategy {
    /// Routes every child (elements, offsets, validity) and the non-list fallback through
    /// [`FlatLayoutStrategy`]. Override individual children with the `with_*` builder methods.
    fn default() -> Self {
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        Self {
            elements: Arc::clone(&flat),
            offsets: Arc::clone(&flat),
            validity: Arc::clone(&flat),
            fallback: flat,
            element_repartition_target: None,
        }
    }
}

impl ListLayoutStrategy {
    /// Strategy for the `elements` child.
    pub fn with_elements(mut self, elements: Arc<dyn LayoutStrategy>) -> Self {
        self.elements = elements;
        self
    }

    /// Repartition the elements stream toward the requested byte size without splitting a sublist
    /// across chunks.
    pub fn with_list_aware_repartition(mut self, target_element_bytes: NonZeroU64) -> Self {
        self.element_repartition_target = Some(target_element_bytes);
        self
    }

    /// Strategy for the `offsets` child.
    pub fn with_offsets(mut self, offsets: Arc<dyn LayoutStrategy>) -> Self {
        self.offsets = offsets;
        self
    }

    /// Strategy for the `validity` child (written only when the list is nullable).
    pub fn with_validity(mut self, validity: Arc<dyn LayoutStrategy>) -> Self {
        self.validity = validity;
        self
    }

    /// Strategy for non-list input, which is forwarded through this strategy unchanged.
    pub fn with_fallback(mut self, fallback: Arc<dyn LayoutStrategy>) -> Self {
        self.fallback = fallback;
        self
    }
}

#[async_trait]
impl LayoutStrategy for ListLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        if !dtype.is_list() {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        let is_nullable = dtype.is_nullable();
        let element_dtype = dtype
            .as_list_element_opt()
            .vortex_expect("DType is List")
            .as_ref()
            .clone();
        // Global offsets are cumulative and may exceed the input offset width.
        let offsets_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);

        let (elements_tx, elements_rx) = kanal::bounded_async(1);
        let (offsets_tx, offsets_rx) = kanal::bounded_async(1);
        let (validity_tx, validity_rx) = if is_nullable {
            let (tx, rx) = kanal::bounded_async(1);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        let child_senders = ListChildSenders {
            elements: elements_tx,
            offsets: offsets_tx,
            validity: validity_tx,
        };

        // Transpose the list column into its child sub-streams and rebase offsets to global
        // positions. Kept joined with the child writers below so producer errors surface rather
        // than being hidden as an early channel close.
        let transpose_session = session.clone();
        let element_repartition_target = self.element_repartition_target;
        let fanout_fut = async move {
            if let Some(target) = element_repartition_target {
                transpose_repartitioned_list_column(
                    stream,
                    transpose_session,
                    child_senders,
                    target,
                )
                .await
            } else {
                transpose_list_column(stream, transpose_session, child_senders).await
            }
        };

        let (elements_ctx, elements_info) = ctx.child_context();
        let (offsets_ctx, _) = ctx.child_context();
        // Repartitioning creates a parent-owned fence schedule in element-row space. Make each
        // selected fence a physical layout boundary before handing it to an arbitrary structural
        // elements writer: otherwise a nested List/Struct writer could merge the stream back
        // together and the enclosing list could not safely persist the fence.
        let elements_strategy: Arc<dyn LayoutStrategy> = if element_repartition_target.is_some() {
            Arc::new(ChunkedLayoutStrategy::new(Arc::clone(&self.elements)))
        } else {
            Arc::clone(&self.elements)
        };
        let mut child_specs: Vec<(
            DType,
            Arc<dyn LayoutStrategy>,
            kanal::AsyncReceiver<ChildChunk>,
            LayoutWriterContext,
        )> = vec![
            (element_dtype, elements_strategy, elements_rx, elements_ctx),
            (
                offsets_dtype,
                Arc::clone(&self.offsets),
                offsets_rx,
                offsets_ctx,
            ),
        ];
        if let Some(validity_rx) = validity_rx {
            let (validity_ctx, _) = ctx.child_context();
            child_specs.push((
                DType::Bool(Nullability::NonNullable),
                Arc::clone(&self.validity),
                validity_rx,
                validity_ctx,
            ));
        }

        let handle = session.handle();
        let layout_futures: Vec<_> = child_specs
            .into_iter()
            .map(|(child_dtype, strategy, rx, child_ctx)| {
                let child_stream =
                    SequentialStreamAdapter::new(child_dtype, rx.into_stream().boxed()).sendable();
                let child_eof = eof.split_off();
                let segment_sink = Arc::clone(&segment_sink);
                let session = session.clone();
                handle.spawn_nested(move |h| async move {
                    let session = session.with_handle(h);
                    strategy
                        .write_stream(child_ctx, segment_sink, child_stream, child_eof, &session)
                        .await
                })
            })
            .collect();

        let (planned_chunk_boundaries, layouts) =
            try_join(fanout_fut, try_join_all(layout_futures)).await?;
        let mut layouts = layouts.into_iter();
        let elements_layout = layouts.next().vortex_expect("elements layout present");
        let offsets_layout = layouts.next().vortex_expect("offsets layout present");
        let validity_layout =
            is_nullable.then(|| layouts.next().vortex_expect("validity layout present"));

        let element_chunk_boundaries = elements_info.chunk_boundaries();
        let mut chunk_boundaries = planned_chunk_boundaries
            .into_iter()
            .filter(|boundary| {
                element_chunk_boundaries
                    .binary_search(&boundary.element_row_end())
                    .is_ok()
            })
            .filter(|boundary| {
                boundary.outer_row_end() != 0
                    && boundary.outer_row_end() < offsets_layout.row_count().saturating_sub(1)
                    && boundary.element_row_end() != 0
                    && boundary.element_row_end() < elements_layout.row_count()
            })
            .collect::<Vec<_>>();
        chunk_boundaries.sort_unstable_by_key(|boundary| boundary.outer_row_end());
        chunk_boundaries.dedup_by_key(|boundary| boundary.outer_row_end());
        ctx.report_chunk_boundaries(
            chunk_boundaries
                .iter()
                .map(|boundary| boundary.outer_row_end()),
        );

        Ok(ListLayout::new_with_chunk_boundaries(
            dtype,
            elements_layout,
            offsets_layout,
            validity_layout,
            chunk_boundaries,
        )
        .into_layout())
    }
}

/// Transpose a list column into its `elements`, `offsets`, and (when present) `validity` child
/// sub-streams. Rebases each chunk's local `offsets` to global `u64` positions so the single
/// `offsets` child indexes into the concatenated `elements` child.
///
/// The validity sender is present only when the list is nullable. Errors surface to the caller,
/// which joins this against the child writers, rather than being hidden as an early channel close.
async fn transpose_list_column(
    mut stream: SendableSequentialStream,
    session: VortexSession,
    child_senders: ListChildSenders,
) -> VortexResult<Vec<ListChunkBoundary>> {
    let mut exec_ctx = session.create_execution_ctx();
    let mut state = ListTransposeState::default();
    let mut saw_chunk = false;

    while let Some(chunk) = stream.next().await {
        let (sequence_id, array) = chunk?;
        saw_chunk = true;

        let mut sp = sequence_id.descend();
        let (elements, offsets, validity) = canonicalize_list_chunk(array, &mut exec_ctx)?;
        // An input list chunk end is an exact outer-list offset boundary. Keep it as a candidate
        // and persist it only if the elements writer confirms that it remained physical.
        state.chunk_boundaries.push(ListChunkBoundary::new(
            state.outer_base + offsets.len().saturating_sub(1) as u64,
            state.element_base + elements.len() as u64,
        ));
        emit_list_parts(
            vec![elements],
            offsets.into_array(),
            validity,
            &mut sp,
            &child_senders,
            &mut state,
            &mut exec_ctx,
        )
        .await?;
    }

    if !saw_chunk {
        vortex_bail!("ListLayoutStrategy needs at least one chunk");
    }

    Ok(state.chunk_boundaries)
}

/// Transpose a list column while snapping independently sized element chunks to list boundaries.
async fn transpose_repartitioned_list_column(
    mut stream: SendableSequentialStream,
    session: VortexSession,
    child_senders: ListChildSenders,
    element_repartition_target: NonZeroU64,
) -> VortexResult<Vec<ListChunkBoundary>> {
    let mut exec_ctx = session.create_execution_ctx();
    let mut state = ListTransposeState::default();
    let mut saw_chunk = false;

    while let Some(chunk) = stream.next().await {
        let (sequence_id, array) = chunk?;
        saw_chunk = true;

        let mut sp = sequence_id.descend();
        let (elements, offsets, validity) = canonicalize_list_chunk(array, &mut exec_ctx)?;
        let element_chunks = repartition_list_elements(
            elements,
            offsets.as_slice::<u64>(),
            element_repartition_target,
            &mut exec_ctx,
        )?;
        state
            .chunk_boundaries
            .extend(map_element_boundaries_to_list_rows(
                &element_chunks.boundaries,
                offsets.as_slice::<u64>(),
                state.outer_base,
                state.element_base,
            ));
        emit_list_parts(
            element_chunks.arrays,
            offsets.into_array(),
            validity,
            &mut sp,
            &child_senders,
            &mut state,
            &mut exec_ctx,
        )
        .await?;
    }

    if !saw_chunk {
        vortex_bail!("ListLayoutStrategy needs at least one chunk");
    }

    Ok(state.chunk_boundaries)
}

/// Translate producer-selected element ends into the enclosing list's row space. A boundary that
/// falls inside a list value is intentionally dropped: it cannot be a list scan split.
fn map_element_boundaries_to_list_rows(
    element_boundaries: &[u64],
    offsets: &[u64],
    outer_base: u64,
    element_base: u64,
) -> Vec<ListChunkBoundary> {
    let offset_base = offsets.first().copied().unwrap_or_default();
    element_boundaries
        .iter()
        .filter_map(|&element_row_end| {
            let offset = offset_base.checked_add(element_row_end)?;
            let offset_index = offsets.partition_point(|&candidate| candidate <= offset);
            (offset_index != 0 && offsets[offset_index - 1] == offset).then_some(())?;
            Some(ListChunkBoundary::new(
                outer_base + (offset_index - 1) as u64,
                element_base + element_row_end,
            ))
        })
        .collect()
}

/// Canonicalize a list chunk into elements, `u64` offsets, and validity.
fn canonicalize_list_chunk(
    array: ArrayRef,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, PrimitiveArray, Validity)> {
    let canonical = array.execute_until::<AnyList>(exec_ctx)?;
    let ListDataParts {
        elements,
        offsets,
        validity,
        ..
    } = if let Some(list) = canonical.as_opt::<List>() {
        list.reset_offsets(false, exec_ctx)?.into_data_parts()
    } else if let Some(view) = canonical.as_opt::<ListView>() {
        list_from_list_view(view.into_owned(), exec_ctx)?.into_data_parts()
    } else {
        unreachable!("AnyList matcher guarantees List or ListView")
    };
    let offsets = offsets
        .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
        .execute::<PrimitiveArray>(exec_ctx)?;
    Ok((elements, offsets, validity))
}

/// Emit one set of list parts to the child writers, rebasing its offsets to the global element
/// stream.
async fn emit_list_parts(
    elements: Vec<ArrayRef>,
    offsets: ArrayRef,
    validity: Validity,
    sp: &mut SequencePointer,
    child_senders: &ListChildSenders,
    state: &mut ListTransposeState,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let n_elements: u64 = elements.iter().map(|elements| elements.len() as u64).sum();
    let row_count = offsets.len().saturating_sub(1);

    for elements in elements {
        if child_senders
            .elements
            .send(Ok((sp.advance(), elements)))
            .await
            .is_err()
        {
            vortex_bail!("list elements writer finished before all chunks were sent");
        }
    }

    let offsets = global_offsets(
        offsets,
        state.element_base,
        !state.emitted_first_chunk,
        exec_ctx,
    )?;
    state.element_base += n_elements;
    state.outer_base += row_count as u64;
    state.emitted_first_chunk = true;

    if child_senders
        .offsets
        .send(Ok((sp.advance(), offsets)))
        .await
        .is_err()
    {
        vortex_bail!("list offsets writer finished before all chunks were sent");
    }
    if let Some(validity_tx) = &child_senders.validity {
        let validity = validity.execute_mask(row_count, exec_ctx)?.into_array();
        if validity_tx
            .send(Ok((sp.advance(), validity)))
            .await
            .is_err()
        {
            vortex_bail!("list validity writer finished before all chunks were sent");
        }
    }
    Ok(())
}

/// Matcher for `Array<List>` or `Array<ListView>`.
struct AnyList;

impl Matcher for AnyList {
    type Match<'a> = ();

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.as_opt::<List>().is_some() || array.as_opt::<ListView>().is_some()).then_some(())
    }
}

/// Rebase a chunk's local `offsets` into global `u64` positions for the whole-column `offsets`
/// child. Each chunk's offsets are shifted by `element_base` (the number of elements already
/// emitted) so they index into the concatenated `elements`. The duplicated boundary offset is
/// dropped on every chunk after the first, so the concatenation of all chunks' contributions is a
/// single monotonic `[0, .., total_elements]` array of length `row_count + 1`.
fn global_offsets(
    offsets: ArrayRef,
    element_base: u64,
    first: bool,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let based = if element_base == 0 {
        offsets
    } else {
        let base = ConstantArray::new(element_base, offsets.len()).into_array();
        offsets.binary(base, Operator::Add)?
    };
    let based = if first {
        based
    } else {
        based.slice(1..based.len())?
    };
    // Materialize so the child sub-stream carries a concrete array rather than a lazy expression.
    Ok(based.execute::<PrimitiveArray>(exec_ctx)?.into_array())
}

#[cfg(test)]
mod tests {
    use futures::stream;
    use vortex_array::ArrayContext;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_io::session::RuntimeSession;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::list::List;
    use crate::layouts::struct_::StructStrategy;
    use crate::layouts::table::TableStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;

    fn layout_test_session() -> VortexSession {
        vortex_array::array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_tokio()
    }

    fn flat_list_strategy() -> ListLayoutStrategy {
        ListLayoutStrategy::default()
    }

    async fn write<S: LayoutStrategy>(strategy: &S, array: ArrayRef) -> VortexResult<LayoutRef> {
        let session = layout_test_session();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        strategy
            .write_stream(
                ArrayContext::empty().into(),
                segments,
                stream,
                eof,
                &session,
            )
            .await
    }

    fn i32_list_dtype(nullable: bool) -> DType {
        DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            if nullable {
                Nullability::Nullable
            } else {
                Nullability::NonNullable
            },
        )
    }

    fn create_basic_list(validity: Validity) -> ArrayRef {
        ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 5, 5].into_array(),
            validity,
        )
        .unwrap()
        .into_array()
    }

    #[tokio::test]
    async fn basic_non_nullable_input() -> VortexResult<()> {
        let list = create_basic_list(Validity::NonNullable);

        let layout = write(&flat_list_strategy(), list).await?;
        assert_eq!(layout.row_count(), 3);

        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list(i32), children: 2
        ├── elements: vortex.flat, dtype: i32, segment: 0
        └── offsets: vortex.flat, dtype: u64, segment: 1
        ");
        Ok(())
    }

    #[tokio::test]
    async fn basic_nullable_input() -> VortexResult<()> {
        let list = create_basic_list(Validity::Array(
            BoolArray::from_iter([true, false, true]).into_array(),
        ));

        let layout = write(&flat_list_strategy(), list).await?;
        assert_eq!(layout.row_count(), 3);

        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list(i32)?, children: 3
        ├── elements: vortex.flat, dtype: i32, segment: 0
        ├── offsets: vortex.flat, dtype: u64, segment: 1
        └── validity: vortex.flat, dtype: bool, segment: 2
        ");
        Ok(())
    }

    #[tokio::test]
    async fn maps_independent_element_chunks_to_outer_row_boundaries() -> VortexResult<()> {
        let list = ListArray::try_new(
            buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array(),
            buffer![0u32, 2, 4, 9, 10].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let writer = ListLayoutStrategy::default()
            .with_list_aware_repartition(NonZeroU64::new(16).vortex_expect("non-zero target"))
            .with_elements(Arc::new(ChunkedLayoutStrategy::new(
                FlatLayoutStrategy::default(),
            )));
        let session = layout_test_session();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let (ctx, info) = LayoutWriterContext::new(ArrayContext::empty()).child_context();
        let layout = writer
            .write_stream(
                ctx,
                segments,
                list.to_array_stream().sequenced(ptr),
                eof,
                &session,
            )
            .await?;

        assert_eq!(
            layout.as_::<List>().chunk_boundaries(),
            [ListChunkBoundary::new(2, 4), ListChunkBoundary::new(3, 9),]
        );
        assert_eq!(info.chunk_boundaries(), [2, 3]);
        Ok(())
    }

    #[tokio::test]
    async fn nested_list_boundaries_are_mapped_compositionally() -> VortexResult<()> {
        let inner = ListArray::try_new(
            buffer![0i32, 1, 2, 3, 4, 5, 6, 7].into_array(),
            buffer![0u32, 2, 4, 6, 8].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let outer = ListArray::try_new(
            inner,
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let inner_strategy = ListLayoutStrategy::default()
            .with_list_aware_repartition(NonZeroU64::new(8).vortex_expect("non-zero target"))
            .with_elements(Arc::new(ChunkedLayoutStrategy::new(
                FlatLayoutStrategy::default(),
            )));
        let strategy = ListLayoutStrategy::default()
            .with_list_aware_repartition(NonZeroU64::new(8).vortex_expect("non-zero target"))
            .with_elements(Arc::new(inner_strategy));
        let session = layout_test_session();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let (ctx, info) = LayoutWriterContext::new(ArrayContext::empty()).child_context();
        let layout = strategy
            .write_stream(
                ctx,
                segments,
                outer.to_array_stream().sequenced(ptr),
                eof,
                &session,
            )
            .await?;

        assert_eq!(
            layout.as_::<List>().chunk_boundaries(),
            [ListChunkBoundary::new(1, 2)]
        );
        assert_eq!(info.chunk_boundaries(), [1]);
        Ok(())
    }

    #[tokio::test]
    async fn list_of_struct_of_list_maps_only_outer_aligned_boundaries() -> VortexResult<()> {
        let nested = ListArray::try_new(
            buffer![0i32, 1, 2, 3, 4, 5, 6, 7].into_array(),
            buffer![0u32, 2, 4, 6, 8].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let elements = StructArray::from_fields([("nested", nested)].as_slice())?.into_array();
        let outer = ListArray::try_new(
            elements,
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let nested_strategy = ListLayoutStrategy::default()
            .with_list_aware_repartition(NonZeroU64::new(8).vortex_expect("non-zero target"))
            .with_elements(Arc::new(ChunkedLayoutStrategy::new(
                FlatLayoutStrategy::default(),
            )));
        let struct_strategy = StructStrategy::new(Arc::clone(&flat), flat)
            .with_field_writer("nested", Arc::new(nested_strategy));
        let strategy = ListLayoutStrategy::default()
            .with_list_aware_repartition(NonZeroU64::new(8).vortex_expect("non-zero target"))
            .with_elements(Arc::new(struct_strategy));

        let session = layout_test_session();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let (ctx, info) = LayoutWriterContext::new(ArrayContext::empty()).child_context();
        let layout = strategy
            .write_stream(
                ctx,
                segments,
                outer.to_array_stream().sequenced(ptr),
                eof,
                &session,
            )
            .await?;

        assert_eq!(
            layout.as_::<List>().chunk_boundaries(),
            [ListChunkBoundary::new(1, 2)]
        );
        assert_eq!(info.chunk_boundaries(), [1]);
        Ok(())
    }

    /// Non-list input dispatches to the fallback strategy unchanged.
    #[tokio::test]
    async fn non_list_input_routes_to_fallback() -> VortexResult<()> {
        let primitive = buffer![1i32, 2, 3].into_array();
        let layout = write(&flat_list_strategy(), primitive).await?;
        insta::assert_snapshot!(layout.display_tree(), @"vortex.flat, dtype: i32, segment: 0");
        Ok(())
    }

    #[tokio::test]
    async fn empty_stream_errors() {
        let segments = Arc::new(TestSegments::default());
        let (_, eof) = SequenceId::root().split();
        let empty = stream::empty::<VortexResult<(SequenceId, ArrayRef)>>().boxed();
        let stream = SequentialStreamAdapter::new(i32_list_dtype(false), empty).sendable();
        let session = layout_test_session();

        let res = flat_list_strategy()
            .write_stream(
                ArrayContext::empty().into(),
                segments,
                stream,
                eof,
                &session,
            )
            .await;
        assert!(res.is_err())
    }

    #[tokio::test]
    async fn list_of_struct_tree() -> VortexResult<()> {
        let struct_array = StructArray::from_fields(
            [
                ("a", buffer![1i32, 2, 3, 4, 5].into_array()),
                ("b", buffer![10i32, 20, 30, 40, 50].into_array()),
            ]
            .as_slice(),
        )?
        .into_array();
        let list = ListArray::try_new(
            struct_array,
            buffer![0u32, 2, 5, 5].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let table_strategy: Arc<dyn LayoutStrategy> =
            Arc::new(TableStrategy::new(Arc::clone(&flat), Arc::clone(&flat)));
        let writer = ListLayoutStrategy::default().with_elements(table_strategy);

        let layout = write(&writer, list).await?;
        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list({a=i32, b=i32}), children: 2
        ├── elements: vortex.struct, dtype: {a=i32, b=i32}, children: 2
        │   ├── a: vortex.flat, dtype: i32, segment: 1
        │   └── b: vortex.flat, dtype: i32, segment: 2
        └── offsets: vortex.flat, dtype: u64, segment: 0
        ");
        Ok(())
    }

    #[tokio::test]
    async fn list_of_list_tree() -> VortexResult<()> {
        let inner_list = ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 2, 5, 5, 6].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let list = ListArray::try_new(
            inner_list,
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let writer =
            ListLayoutStrategy::default().with_elements(Arc::new(ListLayoutStrategy::default()));
        let layout = write(&writer, list).await?;
        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list(list(i32)), children: 2
        ├── elements: vortex.list, dtype: list(i32), children: 2
        │   ├── elements: vortex.flat, dtype: i32, segment: 1
        │   └── offsets: vortex.flat, dtype: u64, segment: 2
        └── offsets: vortex.flat, dtype: u64, segment: 0
        ");
        Ok(())
    }

    #[tokio::test]
    async fn list_of_list_of_list_tree() -> VortexResult<()> {
        let innermost = ListArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let middle = ListArray::try_new(
            innermost,
            buffer![0u32, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let outer =
            ListArray::try_new(middle, buffer![0u32, 1].into_array(), Validity::NonNullable)?
                .into_array();

        let writer = ListLayoutStrategy::default().with_elements(Arc::new(
            ListLayoutStrategy::default().with_elements(Arc::new(ListLayoutStrategy::default())),
        ));
        let layout = write(&writer, outer).await?;
        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list(list(list(i32))), children: 2
        ├── elements: vortex.list, dtype: list(list(i32)), children: 2
        │   ├── elements: vortex.list, dtype: list(i32), children: 2
        │   │   ├── elements: vortex.flat, dtype: i32, segment: 2
        │   │   └── offsets: vortex.flat, dtype: u64, segment: 3
        │   └── offsets: vortex.flat, dtype: u64, segment: 1
        └── offsets: vortex.flat, dtype: u64, segment: 0
        ");
        Ok(())
    }

    #[tokio::test]
    async fn chunked_list_input_with_chunked_strategy_succeeds() -> VortexResult<()> {
        let chunk0 = ListArray::try_new(
            buffer![1i32, 2, 3].into_array(),
            buffer![0u32, 2, 3].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();
        let chunk1 = ListArray::try_new(
            buffer![4i32, 5, 6, 7].into_array(),
            buffer![0u32, 1, 4].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let chunked =
            ChunkedArray::try_new(vec![chunk0, chunk1], i32_list_dtype(false))?.into_array();

        let layout = write(&ChunkedLayoutStrategy::new(flat_list_strategy()), chunked).await?;

        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.chunked, dtype: list(i32), children: 2
        ├── [0]: vortex.list, dtype: list(i32), children: 2
        │   ├── elements: vortex.flat, dtype: i32, segment: 0
        │   └── offsets: vortex.flat, dtype: u64, segment: 1
        └── [1]: vortex.list, dtype: list(i32), children: 2
            ├── elements: vortex.flat, dtype: i32, segment: 2
            └── offsets: vortex.flat, dtype: u64, segment: 3
        ");
        Ok(())
    }
}
