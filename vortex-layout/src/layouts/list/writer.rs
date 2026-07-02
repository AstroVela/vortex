// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::List;
use vortex_array::arrays::ListView;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::list::ListDataParts;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::dtype::DType;
use vortex_array::match_each_integer_ptype;
use vortex_array::matcher::Matcher;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStream;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Strategy for writing list-typed arrays, with a fallback for non-list dtypes.
///
/// Single-chunk only. For list-typed input the strategy:
///  1. Canonicalizes the input chunk into a [`ListView`].
///  2. Calls [`list_from_list_view`] to rebuild it into zero-copy-to-list form
///     (sorted, gapless, non-overlapping offsets) and produce a [`ListArray`].
///  3. Writes the `elements`, `offsets`, and (when nullable) `validity` columns into
///     separately configurable downstream strategies, producing a single [`ListLayout`].
///
/// For input whose dtype is not [`DType::List`], the stream is forwarded unchanged to the
/// configured `fallback` strategy. This lets `ListLayoutStrategy` slot in as a leaf strategy in
/// a heterogeneous column writer where some columns are lists and others are not.
///
/// # Chunking
///
/// `ListLayoutStrategy` bails on empty or multi-chunk input, matching the convention used by
/// [`FlatLayoutStrategy`].
///
/// [`ListArray`]: vortex_array::arrays::ListArray
#[derive(Clone)]
pub struct ListLayoutStrategy {
    elements: Arc<dyn LayoutStrategy>,
    offsets: Arc<dyn LayoutStrategy>,
    validity: Arc<dyn LayoutStrategy>,
    fallback: Arc<dyn LayoutStrategy>,
    /// When set, the flattened `elements` are split into chunks of approximately this many element
    /// rows — cutting only on list boundaries, so no single list straddles two chunks — and written
    /// through a [`ChunkedLayoutStrategy`]. A selective read then fetches only the element chunks
    /// its rows reference instead of the whole elements buffer. `None` writes the elements as a
    /// single (unchunked) layout.
    element_chunk_len: Option<usize>,
}

impl Default for ListLayoutStrategy {
    /// Routes every child (elements, offsets, validity) and the non-list fallback through
    /// [`FlatLayoutStrategy`], and does not chunk the elements. Override individual children with
    /// the `with_*` builder methods.
    fn default() -> Self {
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        Self {
            elements: Arc::clone(&flat),
            offsets: Arc::clone(&flat),
            validity: Arc::clone(&flat),
            fallback: flat,
            element_chunk_len: None,
        }
    }
}

impl ListLayoutStrategy {
    /// Strategy for the `elements` child.
    pub fn with_elements(mut self, elements: Arc<dyn LayoutStrategy>) -> Self {
        self.elements = elements;
        self
    }

    /// Chunk the flattened `elements` into blocks of approximately `len` element rows, cutting only
    /// on list boundaries so no single list straddles two chunks, and write them through a
    /// [`ChunkedLayoutStrategy`] wrapping the configured elements strategy. This makes selective and
    /// range reads fetch only the element chunks they reference. Pass a large `len` (or leave unset)
    /// to keep the elements as a single layout.
    pub fn with_element_chunk_len(mut self, len: usize) -> Self {
        self.element_chunk_len = Some(len);
        self
    }

    /// Split the flattened `elements` into list-aligned chunks (see [`list_aligned_boundaries`]),
    /// returning `None` when chunking is disabled or would produce a single chunk (in which case
    /// the elements are written as one layout).
    fn split_elements_into_chunks(
        &self,
        elements: &ArrayRef,
        offsets: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Vec<ArrayRef>>> {
        let Some(target) = self.element_chunk_len else {
            return Ok(None);
        };
        if target == 0 || offsets.len() <= 1 {
            return Ok(None);
        }
        let offsets_primitive = offsets.clone().execute::<PrimitiveArray>(exec_ctx)?;
        let boundaries = list_aligned_boundaries(&offsets_primitive, target);
        if boundaries.len() <= 2 {
            return Ok(None);
        }
        Ok(Some(slice_elements(elements, &boundaries)?))
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
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        if !dtype.is_list() {
            // Non-list input: route to the configured fallback strategy unchanged.
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        // Writer wants exactly one chunk
        let Some(chunk) = stream.next().await else {
            vortex_bail!("ListLayoutStrategy needs a single chunk");
        };
        let (sequence_id, array) = chunk?;

        let mut exec_ctx = session.create_execution_ctx();
        let ListDataParts {
            elements,
            offsets,
            validity,
            ..
        } = canonicalize_to_list_parts(array, &mut exec_ctx)?;

        // There is one extra element in `offsets`
        let row_count = offsets.len().saturating_sub(1);
        let validity_array = dtype
            .is_nullable()
            .then(|| {
                validity
                    .execute_mask(row_count, &mut exec_ctx)
                    .map(|m| m.into_array())
            })
            .transpose()?;

        // Split the flattened elements into list-aligned chunks when chunking is enabled, so a
        // selective read touches only the element chunks its rows reference and no list straddles a
        // chunk boundary. `None` keeps the elements as a single layout.
        let element_chunks = self.split_elements_into_chunks(&elements, &offsets, &mut exec_ctx)?;

        // Spawn each child write onto the runtime so they run concurrently.
        let handle = session.handle();
        let mut sp = sequence_id.descend();
        let elements_dtype = elements.dtype().clone();
        let offsets_dtype = offsets.dtype().clone();

        let elements_seq = sp.advance();
        let offsets_seq = sp.advance();
        let validity_seq = validity_array.as_ref().map(|_| sp.advance());

        let spawn = |strategy: Arc<dyn LayoutStrategy>,
                     stream: SendableSequentialStream,
                     child_eof: SequencePointer| {
            let ctx = ctx.clone();
            let segment_sink = Arc::clone(&segment_sink);
            let session = session.clone();
            handle.spawn_nested(move |h| async move {
                let session = session.with_handle(h);
                strategy
                    .write_stream(ctx, segment_sink, stream, child_eof, &session)
                    .await
            })
        };

        let (elements_strategy, elements_stream): (
            Arc<dyn LayoutStrategy>,
            SendableSequentialStream,
        ) = match element_chunks {
            Some(chunks) => (
                Arc::new(ChunkedLayoutStrategy::new(Arc::clone(&self.elements))),
                multi_chunk_stream(elements_dtype, elements_seq, chunks),
            ),
            None => (
                Arc::clone(&self.elements),
                single_chunk_stream(elements_dtype, elements_seq, elements),
            ),
        };

        let elements_task = spawn(elements_strategy, elements_stream, eof.split_off());
        let offsets_task = spawn(
            Arc::clone(&self.offsets),
            single_chunk_stream(offsets_dtype, offsets_seq, offsets),
            eof.split_off(),
        );
        let validity_task = match (validity_array, validity_seq) {
            (Some(arr), Some(seq)) => Some(spawn(
                Arc::clone(&self.validity),
                single_chunk_stream(arr.dtype().clone(), seq, arr),
                eof.split_off(),
            )),
            _ => None,
        };

        // Should not have more than one chunk
        if stream.next().await.is_some() {
            vortex_bail!("ListLayoutStrategy received more than a single chunk");
        }

        let (elements_layout, offsets_layout, validity_layout) =
            futures::try_join!(elements_task, offsets_task, async move {
                match validity_task {
                    Some(t) => t.await.map(Some),
                    None => Ok(None),
                }
            },)?;

        Ok(ListLayout::new(dtype, elements_layout, offsets_layout, validity_layout).into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        let list_bytes = self.elements.buffered_bytes()
            + self.offsets.buffered_bytes()
            + self.validity.buffered_bytes();
        list_bytes.max(self.fallback.buffered_bytes())
    }
}

/// Canonicalize a list-dtype array into [`ListDataParts`]. Short-circuits when the input is
/// already a `List` or `ListView` array — otherwise drives the execution loop until one of
/// those forms appears. `ListView` is rebuilt into zero-copy-to-list form via
/// [`list_from_list_view`] before its parts are extracted.
fn canonicalize_to_list_parts(
    array: ArrayRef,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ListDataParts> {
    let canonical = array.execute_until::<AnyList>(exec_ctx)?;
    if let Some(list) = canonical.as_opt::<List>() {
        Ok(list.into_owned().into_data_parts())
    } else if let Some(view) = canonical.as_opt::<ListView>() {
        Ok(list_from_list_view(view.into_owned(), exec_ctx)?.into_data_parts())
    } else {
        unreachable!("AnyList matcher guarantees List or ListView")
    }
}

/// Wrap a single array as a one-shot [`SendableSequentialStream`] for handoff to a child writer.
fn single_chunk_stream(
    dtype: DType,
    sequence_id: SequenceId,
    array: ArrayRef,
) -> SendableSequentialStream {
    SequentialStreamAdapter::new(
        dtype,
        stream::once(async move { Ok((sequence_id, array)) }).boxed(),
    )
    .sendable()
}

/// Wrap a sequence of element chunks as a multi-chunk [`SendableSequentialStream`], assigning each
/// chunk a sequence id descended from `base` (the same pattern the repartition writer uses).
fn multi_chunk_stream(
    dtype: DType,
    base: SequenceId,
    chunks: Vec<ArrayRef>,
) -> SendableSequentialStream {
    let mut sp = base.descend();
    let items: Vec<VortexResult<(SequenceId, ArrayRef)>> =
        chunks.into_iter().map(|c| Ok((sp.advance(), c))).collect();
    SequentialStreamAdapter::new(dtype, stream::iter(items).boxed()).sendable()
}

/// Pick element-buffer boundaries at which to cut the flattened elements into chunks of roughly
/// `target` element rows, cutting only on list boundaries so no list straddles two chunks.
///
/// `offsets` is the canonical (gapless, `offsets[0] == 0`) list offset array with `rows + 1`
/// entries. The returned boundaries are element positions beginning with `0` and ending with the
/// total element count; a list longer than `target` becomes its own chunk (it is never split).
// The `match_each_integer_ptype!` expansion duplicates the loop body across every integer ptype,
// which inflates clippy's cognitive-complexity score; the logic itself is a single linear scan.
#[allow(clippy::cognitive_complexity)]
fn list_aligned_boundaries(offsets: &PrimitiveArray, target: usize) -> Vec<u64> {
    let n = offsets.len();
    if n <= 1 {
        return vec![0];
    }
    let target = target as u64;
    let mut boundaries = vec![0u64];
    let mut last_cut = 0u64;
    // Offsets are validated non-negative upstream, so `as u64` is safe for signed ptypes; it is a
    // no-op when `T == u64`.
    match_each_integer_ptype!(offsets.ptype(), |T| {
        let slice = offsets.as_slice::<T>();
        for i in 0..(n - 1) {
            // Row `i` ends at `offsets[i + 1]`. Cut after it once the chunk reaches `target`.
            #[allow(clippy::unnecessary_cast)]
            let end = slice[i + 1] as u64;
            if end - last_cut >= target {
                boundaries.push(end);
                last_cut = end;
            }
        }
        #[allow(clippy::unnecessary_cast)]
        let total = slice[n - 1] as u64;
        if *boundaries.last().vortex_expect("boundaries starts with 0") != total {
            boundaries.push(total);
        }
    });
    boundaries
}

/// Slice `elements` into the sub-arrays delimited by `boundaries` (element positions).
fn slice_elements(elements: &ArrayRef, boundaries: &[u64]) -> VortexResult<Vec<ArrayRef>> {
    boundaries
        .windows(2)
        .map(|w| {
            let start = usize::try_from(w[0])?;
            let end = usize::try_from(w[1])?;
            elements.slice(start..end)
        })
        .collect()
}

/// Matcher for `Array<List>` or `Array<ListView>`. Used to short-circuit the execution loop
/// when the input is already in (or directly produces) a list form, avoiding a redundant
/// `ListView` round-trip when the writer already has the parts it needs.
struct AnyList;

impl Matcher for AnyList {
    type Match<'a> = ();

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.as_opt::<List>().is_some() || array.as_opt::<ListView>().is_some()).then_some(())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_io::session::RuntimeSession;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::list::ELEMENTS_CHILD_INDEX;
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
            .write_stream(ArrayContext::empty(), segments, stream, eof, &session)
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
        └── offsets: vortex.flat, dtype: u32, segment: 1
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
        ├── offsets: vortex.flat, dtype: u32, segment: 1
        └── validity: vortex.flat, dtype: bool, segment: 2
        ");
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
            .write_stream(ArrayContext::empty(), segments, stream, eof, &session)
            .await;
        assert!(res.is_err())
    }

    #[tokio::test]
    async fn chunked_list_input_without_chunked_strategy_fails() -> VortexResult<()> {
        let chunk0 = ListArray::try_new(
            buffer![1i32, 2].into_array(),
            buffer![0u32, 2].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();
        let chunk1 = ListArray::try_new(
            buffer![3i32, 4, 5].into_array(),
            buffer![0u32, 3].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();
        let chunked =
            ChunkedArray::try_new(vec![chunk0, chunk1], i32_list_dtype(false))?.into_array();

        let res = write(&flat_list_strategy(), chunked).await;
        assert!(res.is_err());
        Ok(())
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
        └── offsets: vortex.flat, dtype: u32, segment: 0
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
        │   └── offsets: vortex.flat, dtype: u32, segment: 2
        └── offsets: vortex.flat, dtype: u32, segment: 0
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
        │   │   └── offsets: vortex.flat, dtype: u32, segment: 3
        │   └── offsets: vortex.flat, dtype: u32, segment: 1
        └── offsets: vortex.flat, dtype: u32, segment: 0
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
        │   └── offsets: vortex.flat, dtype: u32, segment: 1
        └── [1]: vortex.list, dtype: list(i32), children: 2
            ├── elements: vortex.flat, dtype: i32, segment: 2
            └── offsets: vortex.flat, dtype: u32, segment: 3
        ");
        Ok(())
    }

    /// Element-chunk boundaries must land on list boundaries: no single list may straddle two
    /// element chunks.
    #[tokio::test]
    async fn element_chunks_do_not_straddle_lists() -> VortexResult<()> {
        // 9 lists, 20 elements. The offset values are the only legal chunk-boundary positions.
        let offsets = [0u64, 2, 5, 5, 8, 10, 13, 14, 18, 20];
        let list = ListArray::try_new(
            Buffer::from((0i32..20).collect::<Vec<_>>()).into_array(),
            buffer![0u32, 2, 5, 5, 8, 10, 13, 14, 18, 20].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        // Target 4 elements/chunk => cut at the first list boundary at/after each 4-element run.
        let strategy = ListLayoutStrategy::default().with_element_chunk_len(4);
        let layout = write(&strategy, list).await?;

        let elements = layout.child(ELEMENTS_CHILD_INDEX)?;
        assert!(
            elements.nchildren() > 1,
            "expected the elements to be split into multiple chunks"
        );

        // Chunk start offsets (skipping the leading 0) are the interior boundaries; each must
        // coincide with a list boundary.
        let boundaries: Vec<u64> = elements.child_row_offsets().flatten().collect();
        for &boundary in boundaries.iter().skip(1) {
            assert!(
                offsets.contains(&boundary),
                "element chunk boundary {boundary} is not a list boundary; offsets={offsets:?}"
            );
        }
        Ok(())
    }
}
