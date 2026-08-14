// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests for the OnPair layout: what the writer shreds a column into, and what the
//! reader reassembles from it.

use std::ops::Range;
use std::sync::Arc;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::byte_length;
use vortex_array::expr::eq;
use vortex_array::expr::is_not_null;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::VTable as _;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::compressed::CompressorPlugin;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::onpair::CODES_CHILD_INDEX;
use crate::layouts::onpair::CODES_OFFSETS_CHILD_INDEX;
use crate::layouts::onpair::DICT_BYTES_CHILD_INDEX;
use crate::layouts::onpair::DICT_OFFSETS_CHILD_INDEX;
use crate::layouts::onpair::OnPair;
use crate::layouts::onpair::writer::OnPairLayoutOptions;
use crate::layouts::onpair::writer::OnPairStrategy;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialArrayStreamExt;
use crate::session::LayoutSession;

fn layout_test_session() -> VortexSession {
    vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_tokio()
}

/// `OnPairScheme` lives behind btrblocks' `unstable_encodings` feature, so a default compressor
/// would never pick OnPair here and the probe would always fall back. Probe with OnPair directly to
/// test the layout itself rather than scheme selection.
fn onpair_probe() -> Arc<dyn CompressorPlugin> {
    Arc::new(|chunk: &ArrayRef, ctx: &mut ExecutionCtx| onpair_compress(chunk, DEFAULT_CONFIG, ctx))
}

/// The dictionary children see exactly one chunk, so they go through a flat leaf. The four per-chunk
/// children see one chunk per input chunk, so they need a chunked leaf — as in the real pipeline,
/// where `coalescing` ends in [`ChunkedLayoutStrategy`].
fn onpair_strategy() -> OnPairStrategy {
    OnPairStrategy::new(
        FlatLayoutStrategy::default(),
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        FlatLayoutStrategy::default(),
        OnPairLayoutOptions::default(),
        onpair_probe(),
    )
}

async fn write_layout<S: LayoutStrategy>(
    strategy: &S,
    array: ArrayRef,
) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef, VortexSession)> {
    let session = layout_test_session();
    let segments = Arc::new(TestSegments::default());
    let segments_ref: Arc<dyn SegmentSource> = Arc::<TestSegments>::clone(&segments);
    let (ptr, eof) = SequenceId::root().split();
    let stream = array.to_array_stream().sequenced(ptr);
    let layout = strategy
        .write_stream(
            ArrayContext::empty().into(),
            segments,
            stream,
            eof,
            &session,
        )
        .await?;
    Ok((segments_ref, layout, session))
}

fn urls(range: Range<usize>, nullable: bool) -> ArrayRef {
    let strings: Vec<String> = range
        .map(|i| format!("https://www.example.com/items/{i:06}"))
        .collect();
    let dtype = if nullable {
        DType::Utf8(Nullability::Nullable)
    } else {
        DType::Utf8(Nullability::NonNullable)
    };
    VarBinArray::from_iter(
        strings.iter().enumerate().map(|(i, s)| {
            // Every 5th row is null, but only when the column allows it.
            (!nullable || i % 5 != 0).then_some(s.as_str())
        }),
        dtype,
    )
    .into_array()
}

/// A 100-row column of URLs, either as one chunk or as three unevenly sized ones.
fn column(nullable: bool, chunked: bool) -> VortexResult<ArrayRef> {
    if !chunked {
        return Ok(urls(0..100, nullable));
    }
    let dtype = if nullable {
        DType::Utf8(Nullability::Nullable)
    } else {
        DType::Utf8(Nullability::NonNullable)
    };
    let chunks = vec![
        urls(0..40, nullable),
        urls(40..90, nullable),
        urls(90..100, nullable),
    ];
    Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
}

async fn onpair_reader(
    array: ArrayRef,
) -> VortexResult<(LayoutReaderRef, LayoutRef, VortexSession)> {
    let (segments, layout, session) = write_layout(&onpair_strategy(), array).await?;
    assert_eq!(
        layout.encoding_id(),
        OnPair.id(),
        "test column must actually take the OnPair layout"
    );
    let reader = layout.new_reader("".into(), segments, &session, &LayoutReaderContext::new())?;
    Ok((reader, layout, session))
}

// ---- writer ----

#[tokio::test]
async fn non_nullable_tree() -> VortexResult<()> {
    let (_, layout, _) = write_layout(&onpair_strategy(), urls(0..64, false)).await?;
    assert_eq!(layout.row_count(), 64);
    insta::assert_snapshot!(layout.display_tree(), @"
    vortex.onpair, dtype: utf8, children: 5
    ├── dict_bytes: vortex.flat, dtype: u8, segment: 0
    ├── dict_offsets: vortex.flat, dtype: u32, segment: 1
    ├── codes: vortex.flat, dtype: u16, segment: 2
    ├── codes_offsets: vortex.flat, dtype: u64, segment: 3
    └── uncompressed_lengths: vortex.flat, dtype: i32, segment: 4
    ");
    Ok(())
}

#[tokio::test]
async fn nullable_tree() -> VortexResult<()> {
    let (_, layout, _) = write_layout(&onpair_strategy(), urls(0..64, true)).await?;
    assert_eq!(layout.row_count(), 64);
    insta::assert_snapshot!(layout.display_tree(), @"
    vortex.onpair, dtype: utf8?, children: 6
    ├── dict_bytes: vortex.flat, dtype: u8, segment: 0
    ├── dict_offsets: vortex.flat, dtype: u32, segment: 1
    ├── codes: vortex.flat, dtype: u16, segment: 2
    ├── codes_offsets: vortex.flat, dtype: u64, segment: 3
    ├── uncompressed_lengths: vortex.flat, dtype: i32, segment: 4
    └── validity: vortex.flat, dtype: bool, segment: 5
    ");
    Ok(())
}

/// Non-string input is forwarded to the fallback untouched.
#[tokio::test]
async fn non_string_routes_to_fallback() -> VortexResult<()> {
    let (_, layout, _) = write_layout(&onpair_strategy(), buffer![1i32, 2, 3].into_array()).await?;
    insta::assert_snapshot!(layout.display_tree(), @"vortex.flat, dtype: i32, segment: 0");
    Ok(())
}

/// An all-null column has nothing to train on: the probe sees a constant array rather than OnPair
/// and the column takes the fallback.
#[tokio::test]
async fn all_null_routes_to_fallback() -> VortexResult<()> {
    let array = VarBinArray::from_iter(
        [None::<&str>, None, None],
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();
    let (_, layout, _) = write_layout(&onpair_strategy(), array).await?;
    assert_ne!(layout.encoding_id(), OnPair.id());
    Ok(())
}

/// `codes_offsets` is cumulative over the whole layout node, so a multi-chunk column contributes
/// exactly one extra entry in total — not one per chunk.
#[tokio::test]
async fn multi_chunk_codes_offsets_are_global() -> VortexResult<()> {
    let (_, layout, _) = write_layout(&onpair_strategy(), column(false, true)?).await?;
    assert_eq!(layout.encoding_id(), OnPair.id());
    assert_eq!(layout.row_count(), 100);
    let children = layout.children()?;
    assert_eq!(
        children[CODES_OFFSETS_CHILD_INDEX].row_count(),
        101,
        "codes_offsets must have row_count + 1 entries across all chunks"
    );
    // Every row of these URLs takes at least one token.
    assert!(children[CODES_CHILD_INDEX].row_count() >= 100);
    Ok(())
}

/// An empty stream has nothing to train on, so it takes the fallback rather than writing a
/// dictionary-less OnPair node.
#[tokio::test]
async fn empty_column_routes_to_fallback() -> VortexResult<()> {
    let array =
        VarBinArray::from_iter(Vec::<Option<&str>>::new(), DType::Utf8(false.into())).into_array();
    let (_, layout, _) = write_layout(&onpair_strategy(), array).await?;
    assert_ne!(layout.encoding_id(), OnPair.id());
    assert_eq!(layout.row_count(), 0);
    Ok(())
}

/// The shape that makes the layout worth having: however many chunks the column has, the two
/// dictionary children stay flat and single-segment while the four per-chunk children become chunked.
#[tokio::test]
async fn multi_chunk_tree() -> VortexResult<()> {
    let (_, layout, _) = write_layout(&onpair_strategy(), column(true, true)?).await?;
    insta::assert_snapshot!(layout.display_tree(), @r"
    vortex.onpair, dtype: utf8?, children: 6
    ├── dict_bytes: vortex.flat, dtype: u8, segment: 0
    ├── dict_offsets: vortex.flat, dtype: u32, segment: 1
    ├── codes: vortex.chunked, dtype: u16, children: 3
    │   ├── [0]: vortex.flat, dtype: u16, segment: 2
    │   ├── [1]: vortex.flat, dtype: u16, segment: 6
    │   └── [2]: vortex.flat, dtype: u16, segment: 10
    ├── codes_offsets: vortex.chunked, dtype: u64, children: 3
    │   ├── [0]: vortex.flat, dtype: u64, segment: 3
    │   ├── [1]: vortex.flat, dtype: u64, segment: 7
    │   └── [2]: vortex.flat, dtype: u64, segment: 11
    ├── uncompressed_lengths: vortex.chunked, dtype: i32, children: 3
    │   ├── [0]: vortex.flat, dtype: i32, segment: 4
    │   ├── [1]: vortex.flat, dtype: i32, segment: 8
    │   └── [2]: vortex.flat, dtype: i32, segment: 12
    └── validity: vortex.chunked, dtype: bool, children: 3
        ├── [0]: vortex.flat, dtype: bool, segment: 5
        ├── [1]: vortex.flat, dtype: bool, segment: 9
        └── [2]: vortex.flat, dtype: bool, segment: 13
    ");
    for child in [DICT_BYTES_CHILD_INDEX, DICT_OFFSETS_CHILD_INDEX] {
        assert_eq!(layout.children()?[child].segment_ids().len(), 1);
    }
    Ok(())
}

// ---- reader ----

/// A full-range, all-true projection must reproduce the column exactly, for every combination of
/// nullability and chunking.
#[rstest]
#[case::single_chunk_non_nullable(false, false)]
#[case::single_chunk_nullable(true, false)]
#[case::multi_chunk_non_nullable(false, true)]
#[case::multi_chunk_nullable(true, true)]
#[tokio::test]
async fn round_trip(#[case] nullable: bool, #[case] chunked: bool) -> VortexResult<()> {
    let array = column(nullable, chunked)?;
    let (reader, layout, session) = onpair_reader(array.clone()).await?;

    let actual = reader
        .projection_evaluation(
            &(0..layout.row_count()),
            &root(),
            MaskFuture::new_true(usize::try_from(layout.row_count())?),
        )?
        .await?;

    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(actual, array, &mut ctx);
    Ok(())
}

/// The layout claims `Binary` as well as `Utf8`, so it has to round-trip both.
#[rstest]
#[case::nullable(true)]
#[case::non_nullable(false)]
#[tokio::test]
async fn binary_round_trip(#[case] nullable: bool) -> VortexResult<()> {
    let nullability = Nullability::from(nullable);
    let values: Vec<Vec<u8>> = (0..100u32)
        .map(|i| format!("payload-{i:04}").into_bytes())
        .collect();
    let array = VarBinArray::from_iter(
        values
            .iter()
            .enumerate()
            .map(|(i, v)| (!nullable || i % 5 != 0).then_some(v.as_slice())),
        DType::Binary(nullability),
    )
    .into_array();

    let (reader, layout, session) = onpair_reader(array.clone()).await?;
    let actual = reader
        .projection_evaluation(
            &(0..layout.row_count()),
            &root(),
            MaskFuture::new_true(usize::try_from(layout.row_count())?),
        )?
        .await?;

    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(actual, array, &mut ctx);
    Ok(())
}

/// Every sub-range and mask combination must match the same projection over the ground-truth array
/// (`array.slice(range).filter(mask)`). These exercise the bounded read path, which crops the token
/// window to the selected rows and rebases the code boundaries onto it.
#[rstest]
#[case::full_all_true(0..100, Mask::new_true(100))]
#[case::subrange_all_true(10..60, Mask::new_true(50))]
#[case::subrange_sparse(10..60, Mask::from_iter((0..50).map(|i| i % 7 == 0)))]
#[case::across_chunk_boundary(38..42, Mask::new_true(4))]
#[case::prefix(0..1, Mask::new_true(1))]
#[case::suffix(99..100, Mask::new_true(1))]
#[case::full_range_sparse(0..100, Mask::from_iter((0..100).map(|i| i == 3 || i == 97)))]
#[case::empty_range(50..50, Mask::new_true(0))]
#[case::all_false(0..100, Mask::new_false(100))]
#[tokio::test]
async fn sub_range_and_mask(#[case] range: Range<u64>, #[case] mask: Mask) -> VortexResult<()> {
    let array = column(true, true)?;
    let (reader, _, session) = onpair_reader(array.clone()).await?;

    let actual = reader
        .projection_evaluation(&range, &root(), MaskFuture::ready(mask.clone()))?
        .await?;

    let expected = array
        .slice(usize::try_from(range.start)?..usize::try_from(range.end)?)?
        .filter(mask)?;
    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

/// Validity-class projections go through the validity-only path and must still agree with the same
/// expression evaluated over the whole column.
#[rstest]
#[case::is_null_nullable(true, is_null(root()))]
#[case::is_not_null_nullable(true, is_not_null(root()))]
#[case::is_null_non_nullable(false, is_null(root()))]
#[case::is_not_null_non_nullable(false, is_not_null(root()))]
#[tokio::test]
async fn validity_projection(#[case] nullable: bool, #[case] expr: Expression) -> VortexResult<()> {
    let array = column(nullable, true)?;
    let (reader, layout, session) = onpair_reader(array.clone()).await?;

    let actual = reader
        .projection_evaluation(
            &(0..layout.row_count()),
            &expr,
            MaskFuture::new_true(usize::try_from(layout.row_count())?),
        )?
        .await?;

    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(actual, array.apply(&expr)?, &mut ctx);
    Ok(())
}

/// `byte_length` is served from `uncompressed_lengths`, so it must produce exactly what evaluating
/// it over the decoded strings would — including nulls and the `u64` result dtype.
#[rstest]
#[case::nullable(true)]
#[case::non_nullable(false)]
#[tokio::test]
async fn byte_length_projection(#[case] nullable: bool) -> VortexResult<()> {
    let array = column(nullable, true)?;
    let (reader, layout, session) = onpair_reader(array.clone()).await?;

    let actual = reader
        .projection_evaluation(
            &(0..layout.row_count()),
            &byte_length(root()),
            MaskFuture::new_true(usize::try_from(layout.row_count())?),
        )?
        .await?;

    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(actual, array.apply(&byte_length(root()))?, &mut ctx);
    Ok(())
}

/// A sparse mask on the lengths path must select the same rows the strings path would.
#[tokio::test]
async fn byte_length_applies_sparse_mask() -> VortexResult<()> {
    let array = column(true, true)?;
    let (reader, _, session) = onpair_reader(array.clone()).await?;

    let mask = Mask::from_iter((0..100).map(|i| i % 11 == 0));
    let actual = reader
        .projection_evaluation(
            &(0..100),
            &byte_length(root()),
            MaskFuture::ready(mask.clone()),
        )?
        .await?;

    let expected = array.filter(mask)?.apply(&byte_length(root()))?;
    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

/// An equality filter reaches OnPair's compressed-domain compare through the reassembled array.
#[tokio::test]
async fn filter_evaluation_compare() -> VortexResult<()> {
    let array = column(false, true)?;
    let (reader, ..) = onpair_reader(array).await?;

    let expr = eq(root(), lit("https://www.example.com/items/000042"));
    let actual = reader
        .filter_evaluation(&(0..100), &expr, MaskFuture::new_true(100))?
        .await?;

    assert_eq!(actual, Mask::from_iter((0..100).map(|i| i == 42)));
    Ok(())
}

/// `filter_evaluation` must intersect with the mask it was given, not replace it.
#[tokio::test]
async fn filter_evaluation_intersects_input_mask() -> VortexResult<()> {
    let array = column(true, true)?;
    let (reader, ..) = onpair_reader(array).await?;

    // Rows 0, 40 and 90 are the null rows at the start of each chunk.
    let input = Mask::from_iter((0..100).map(|i| i < 45));
    let actual = reader
        .filter_evaluation(&(0..100), &is_null(root()), MaskFuture::ready(input))?
        .await?;

    assert_eq!(
        actual,
        Mask::from_iter((0..100).map(|i| i < 45 && i % 5 == 0))
    );
    Ok(())
}

// ---- read amplification ----

/// A [`SegmentSource`] that records how many times each segment is requested.
struct RecordingSegmentSource {
    inner: Arc<dyn SegmentSource>,
    requests: Arc<DashMap<SegmentId, usize>>,
}

impl SegmentSource for RecordingSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        *self.requests.entry(id).or_insert(0) += 1;
        self.inner.request(id)
    }
}

/// Wrap `segments` in a recorder and build a reader over `layout`.
fn recording_reader(
    layout: &LayoutRef,
    segments: Arc<dyn SegmentSource>,
    session: &VortexSession,
) -> VortexResult<(LayoutReaderRef, Arc<DashMap<SegmentId, usize>>)> {
    let requests = Arc::new(DashMap::default());
    let source = Arc::new(RecordingSegmentSource {
        inner: segments,
        requests: Arc::clone(&requests),
    });
    let reader = layout.new_reader("".into(), source, session, &LayoutReaderContext::new())?;
    Ok((reader, requests))
}

/// Every segment id in `layout`'s subtree. `LayoutRef::segment_ids` reports only directly referenced
/// segments, and a chunked child keeps its data in its own children, so the whole subtree has to be
/// walked — otherwise a chunked child looks like it holds no segments at all.
fn subtree_segments(layout: &LayoutRef) -> VortexResult<Vec<SegmentId>> {
    let mut ids = layout.segment_ids();
    for child in layout.children()? {
        ids.extend(subtree_segments(&child)?);
    }
    Ok(ids)
}

/// Every segment id belonging to one child of an OnPair layout.
fn child_segments(layout: &LayoutRef, child: usize) -> VortexResult<Vec<SegmentId>> {
    subtree_segments(&layout.children()?[child])
}

fn total_requests(requests: &DashMap<SegmentId, usize>, segments: &[SegmentId]) -> usize {
    segments
        .iter()
        .map(|id| requests.get(id).map(|n| *n).unwrap_or(0))
        .sum()
}

/// The whole point of the layout is that the dictionary is read once for the column, not once per
/// row range. Two disjoint projections must request each dictionary segment exactly once.
#[tokio::test]
async fn dictionary_is_read_once_across_row_ranges() -> VortexResult<()> {
    let array = column(false, true)?;
    let (segments, layout, session) = write_layout(&onpair_strategy(), array).await?;
    let (reader, requests) = recording_reader(&layout, segments, &session)?;

    for range in [0u64..30, 60..100] {
        let len = usize::try_from(range.end - range.start)?;
        reader
            .projection_evaluation(&range, &root(), MaskFuture::new_true(len))?
            .await?;
    }

    let dict = [
        child_segments(&layout, DICT_BYTES_CHILD_INDEX)?,
        child_segments(&layout, DICT_OFFSETS_CHILD_INDEX)?,
    ]
    .concat();
    assert!(!dict.is_empty(), "the dictionary must occupy some segments");
    for id in &dict {
        assert_eq!(
            requests.get(id).map(|n| *n),
            Some(1),
            "dictionary segment {id:?} was read more than once"
        );
    }
    // The codes were read for both ranges, so the reads really were separate.
    assert!(total_requests(&requests, &child_segments(&layout, CODES_CHILD_INDEX)?) >= 2);
    Ok(())
}

/// Validity-class and `byte_length` expressions must not touch the dictionary or the codes at all.
#[rstest]
#[case::is_not_null(is_not_null(root()))]
#[case::byte_length(byte_length(root()))]
#[tokio::test]
async fn cheap_expressions_skip_dictionary_and_codes(#[case] expr: Expression) -> VortexResult<()> {
    let array = column(true, true)?;
    let (segments, layout, session) = write_layout(&onpair_strategy(), array).await?;
    let (reader, requests) = recording_reader(&layout, segments, &session)?;

    reader
        .projection_evaluation(&(0..100), &expr, MaskFuture::new_true(100))?
        .await?;

    for child in [
        DICT_BYTES_CHILD_INDEX,
        DICT_OFFSETS_CHILD_INDEX,
        CODES_CHILD_INDEX,
        CODES_OFFSETS_CHILD_INDEX,
    ] {
        assert_eq!(
            total_requests(&requests, &child_segments(&layout, child)?),
            0,
            "child {child} should not be read for {expr}"
        );
    }
    assert!(!requests.is_empty(), "something must have been read");
    Ok(())
}
