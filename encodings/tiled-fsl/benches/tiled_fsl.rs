// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::fmt;
use std::num::NonZeroU32;
use std::ops::Range;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::assert_arrays_eq;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_session::VortexSession;
use vortex_tiled_fsl::TileGeometry;
use vortex_tiled_fsl::TiledFixedSizeList;
use vortex_tiled_fsl::TiledFixedSizeListArray;
use vortex_tiled_fsl::TiledFixedSizeListArrayExt;
use vortex_tiled_fsl::TiledFixedSizeListArraySlotsExt;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    assert_fixture_matrix();
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_tiled_fsl::initialize(&session);
    vortex_fastlanes::initialize(&session);
    session
});

#[derive(Clone, Copy)]
struct Args {
    rows: usize,
    dimensions: usize,
    tile_rows: u32,
    tile_dimensions: TileDimensions,
}

#[derive(Clone, Copy)]
enum TileDimensions {
    Full,
    Fixed(u32),
}

impl fmt::Display for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dimension_tile = match self.tile_dimensions {
            TileDimensions::Full => "full".to_owned(),
            TileDimensions::Fixed(dimensions) => dimensions.to_string(),
        };
        write!(
            f,
            "rows{}_dims{}_tile{}x{}",
            self.rows, self.dimensions, self.tile_rows, dimension_tile,
        )
    }
}

fn args() -> Vec<Args> {
    let mut args = Vec::new();
    let geometries = [
        (32, TileDimensions::Full),
        (64, TileDimensions::Full),
        (32, TileDimensions::Fixed(64)),
        (64, TileDimensions::Fixed(64)),
        (16, TileDimensions::Fixed(4)),
    ];
    for rows in [1_024, 16_384] {
        for dimensions in [128, 768, 1_536] {
            for (tile_rows, tile_dimensions) in geometries {
                args.push(Args {
                    rows,
                    dimensions,
                    tile_rows,
                    tile_dimensions,
                });
            }
        }
    }
    for rows in [31, 33, 63, 65] {
        for dimensions in [31, 33, 63, 65] {
            for tile_rows in [32, 64] {
                args.push(Args {
                    rows,
                    dimensions,
                    tile_rows,
                    tile_dimensions: TileDimensions::Fixed(64),
                });
            }
        }
    }
    args
}

#[derive(Clone, Copy)]
enum SliceKind {
    Small,
    Half,
    CrossTileBoundary,
}

impl fmt::Display for SliceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Small => "small",
            Self::Half => "half",
            Self::CrossTileBoundary => "cross_tile_boundary",
        })
    }
}

#[derive(Clone, Copy)]
struct SliceArgs {
    args: Args,
    kind: SliceKind,
}

impl fmt::Display for SliceArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.kind, self.args)
    }
}

fn slice_args() -> Vec<SliceArgs> {
    args()
        .into_iter()
        .flat_map(|args| {
            [SliceKind::Small, SliceKind::Half]
                .into_iter()
                .chain(
                    (args.rows > args.tile_rows as usize).then_some(SliceKind::CrossTileBoundary),
                )
                .map(move |kind| SliceArgs { args, kind })
        })
        .collect()
}

#[derive(Clone, Copy)]
enum TakeKind {
    SortedSparse,
    UnsortedDuplicated,
}

impl fmt::Display for TakeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SortedSparse => "sorted_sparse",
            Self::UnsortedDuplicated => "unsorted_duplicated",
        })
    }
}

#[derive(Clone, Copy)]
struct TakeArgs {
    args: Args,
    kind: TakeKind,
}

impl fmt::Display for TakeArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.kind, self.args)
    }
}

fn take_args() -> Vec<TakeArgs> {
    args()
        .into_iter()
        .flat_map(|args| {
            [TakeKind::SortedSparse, TakeKind::UnsortedDuplicated]
                .into_iter()
                .map(move |kind| TakeArgs { args, kind })
        })
        .collect()
}

#[derive(Clone, Copy)]
enum PhysicalEncoding {
    Raw,
    Bitpacked,
}

impl fmt::Display for PhysicalEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Raw => "raw",
            Self::Bitpacked => "bitpacked_4bit",
        })
    }
}

#[derive(Clone, Copy)]
struct ScoreArgs {
    args: Args,
    encoding: PhysicalEncoding,
}

impl fmt::Display for ScoreArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.encoding, self.args)
    }
}

fn score_args() -> Vec<ScoreArgs> {
    args()
        .into_iter()
        .flat_map(|args| {
            [PhysicalEncoding::Raw, PhysicalEncoding::Bitpacked]
                .into_iter()
                .map(move |encoding| ScoreArgs { args, encoding })
        })
        .collect()
}

fn tile_geometry(args: Args) -> TileGeometry {
    let dimensions = match args.tile_dimensions {
        TileDimensions::Full => u32::try_from(args.dimensions).unwrap(),
        TileDimensions::Fixed(dimensions) => dimensions,
    };
    TileGeometry::new(
        NonZeroU32::new(args.tile_rows).unwrap(),
        NonZeroU32::new(dimensions).unwrap(),
    )
}

fn fixture_value(row: usize, dimension: usize, _dimensions: usize) -> u8 {
    // Advancing one row changes every coordinate by 5 modulo 16, regardless of row width.
    ((row * 5 + dimension * 3) & 0x0f) as u8
}

fn representative_rows(args: Args) -> (Vec<u8>, Vec<u8>) {
    (
        (0..args.dimensions)
            .map(|dimension| fixture_value(0, dimension, args.dimensions))
            .collect(),
        (0..args.dimensions)
            .map(|dimension| fixture_value(1, dimension, args.dimensions))
            .collect(),
    )
}

fn assert_fixture_matrix() {
    let mut indistinct_widths = args()
        .into_iter()
        .filter_map(|args| {
            let (first, second) = representative_rows(args);
            (first == second).then_some(args.dimensions)
        })
        .collect::<Vec<_>>();
    indistinct_widths.sort_unstable();
    indistinct_widths.dedup();
    assert!(
        indistinct_widths.is_empty(),
        "adjacent fixture rows are identical for dimensions {indistinct_widths:?}",
    );
}

fn assert_adjacent_payloads_distinct(array: &FixedSizeListArray, args: Args) {
    if args.rows < 2 {
        return;
    }
    let elements = array.elements().as_::<Primitive>();
    let values = elements.as_slice::<u8>();
    assert_ne!(
        &values[..args.dimensions],
        &values[args.dimensions..args.dimensions * 2],
        "adjacent canonical rows are identical for {args}",
    );
}

fn assert_adjacent_scores_distinct(values: &[u8], args: Args, query: &[u8]) {
    if args.rows < 2 {
        return;
    }
    let scores =
        scoring::score_canonical(&values[..args.dimensions * 2], 2, args.dimensions, query);
    assert_ne!(
        scores[0], scores[1],
        "adjacent canonical row scores are identical for {args}",
    );
}

fn canonical_u8(args: Args) -> FixedSizeListArray {
    let array = FixedSizeListArray::new(
        PrimitiveArray::from_iter((0..args.rows).flat_map(|row| {
            (0..args.dimensions)
                .map(move |dimension| fixture_value(row, dimension, args.dimensions))
        }))
        .into_array(),
        u32::try_from(args.dimensions).unwrap(),
        Validity::NonNullable,
        args.rows,
    );
    assert_adjacent_payloads_distinct(&array, args);
    array
}

fn query(args: Args) -> Vec<u8> {
    (0..args.dimensions)
        .map(|dimension| ((dimension * 13 + 7) & 0x0f) as u8)
        .collect()
}

fn raw_tiled(args: Args, ctx: &mut ExecutionCtx) -> VortexResult<TiledFixedSizeListArray> {
    TiledFixedSizeList::encode(canonical_u8(args).as_view(), tile_geometry(args), ctx)
}

fn bitpacked_tiled(args: Args, ctx: &mut ExecutionCtx) -> VortexResult<TiledFixedSizeListArray> {
    let raw = raw_tiled(args, ctx)?;
    let physical = raw.elements().clone().execute::<PrimitiveArray>(ctx)?;
    let bitpacked = bitpack_encode(&physical, 4, None, ctx)?.into_array();
    TiledFixedSizeList::try_new(
        bitpacked,
        u32::try_from(args.dimensions)?,
        raw.array_validity(),
        args.rows,
        tile_geometry(args),
    )
}

fn canonical_f32(args: Args) -> FixedSizeListArray {
    FixedSizeListArray::new(
        PrimitiveArray::from_iter(
            (0..args.rows * args.dimensions).map(|index| ((index * 17) % 1_009) as f32 / 1_009.0),
        )
        .into_array(),
        u32::try_from(args.dimensions).unwrap(),
        Validity::NonNullable,
        args.rows,
    )
}

fn canonical_nullable_f32(args: Args) -> FixedSizeListArray {
    let element_count = args.rows * args.dimensions;
    FixedSizeListArray::new(
        PrimitiveArray::new(
            Buffer::from_iter(
                (0..element_count).map(|index| ((index * 17) % 1_009) as f32 / 1_009.0),
            ),
            Validity::from_iter((0..element_count).map(|index| index % 11 != 0)),
        )
        .into_array(),
        u32::try_from(args.dimensions).unwrap(),
        Validity::NonNullable,
        args.rows,
    )
}

fn assert_tiled_matches(
    canonical: &FixedSizeListArray,
    tiled: &TiledFixedSizeListArray,
    ctx: &mut ExecutionCtx,
) {
    assert_arrays_eq!(canonical, tiled, ctx);
}

fn tiled_score_fixture(
    args: ScoreArgs,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TiledFixedSizeListArray> {
    let tiled = match args.encoding {
        PhysicalEncoding::Raw => raw_tiled(args.args, ctx),
        PhysicalEncoding::Bitpacked => bitpacked_tiled(args.args, ctx),
    }?;
    assert_eq!(tiled.geometry(), tile_geometry(args.args));
    Ok(tiled)
}

fn slice_range(args: SliceArgs) -> Range<usize> {
    match args.kind {
        SliceKind::Small => {
            let start = args.args.rows / 3;
            start..start + 8
        }
        SliceKind::Half => {
            let start = args.args.rows / 4;
            start..start + args.args.rows / 2
        }
        SliceKind::CrossTileBoundary => {
            let boundary = args.args.tile_rows as usize;
            boundary - 1..boundary + 1
        }
    }
}

fn take_indices(args: TakeArgs) -> PrimitiveArray {
    let rows = u32::try_from(args.args.rows).unwrap();
    match args.kind {
        TakeKind::SortedSparse => {
            PrimitiveArray::from_iter([0, rows / 4, rows / 2, rows * 3 / 4, rows - 1])
        }
        TakeKind::UnsortedDuplicated => {
            PrimitiveArray::from_iter([rows - 1, 0, rows / 2, rows / 2, 1, rows - 1])
        }
    }
}

fn assert_scores_equal(
    args: Args,
    tiled: &TiledFixedSizeListArray,
    tiled_values: &[u8],
    query: &[u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let canonical = canonical_u8(args);
    let canonical_values = canonical
        .elements()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    assert_adjacent_scores_distinct(canonical_values.as_slice::<u8>(), args, query);
    assert_eq!(
        scoring::score_canonical(
            canonical_values.as_slice::<u8>(),
            args.rows,
            args.dimensions,
            query,
        ),
        scoring::score_tiled(tiled.as_view(), tiled_values, query),
        "canonical and tiled scores differ for {args}",
    );
    Ok(())
}

mod encode {
    use super::*;

    #[divan::bench(args = args())]
    fn non_nullable(bencher: Bencher, args: Args) {
        bench_encode(bencher, args, canonical_f32(args));
    }

    #[divan::bench(args = args())]
    fn nullable_bitmap(bencher: Bencher, args: Args) {
        bench_encode(bencher, args, canonical_nullable_f32(args));
    }

    fn bench_encode(bencher: Bencher, args: Args, canonical: FixedSizeListArray) {
        let mut ctx = SESSION.create_execution_ctx();
        let oracle =
            TiledFixedSizeList::encode(canonical.as_view(), tile_geometry(args), &mut ctx).unwrap();
        assert_tiled_matches(&canonical, &oracle, &mut ctx);

        bencher
            .with_inputs(|| SESSION.create_execution_ctx())
            .bench_values(|mut ctx| {
                divan::black_box(
                    TiledFixedSizeList::encode(canonical.as_view(), tile_geometry(args), &mut ctx)
                        .unwrap(),
                )
            });
    }
}

mod execute {
    use super::*;

    #[divan::bench(args = args())]
    fn non_nullable(bencher: Bencher, args: Args) {
        bench_execute(bencher, args, canonical_f32(args));
    }

    #[divan::bench(args = args())]
    fn nullable_bitmap(bencher: Bencher, args: Args) {
        bench_execute(bencher, args, canonical_nullable_f32(args));
    }

    fn bench_execute(bencher: Bencher, args: Args, canonical: FixedSizeListArray) {
        let mut ctx = SESSION.create_execution_ctx();
        let tiled =
            TiledFixedSizeList::encode(canonical.as_view(), tile_geometry(args), &mut ctx).unwrap();
        assert_tiled_matches(&canonical, &tiled, &mut ctx);

        bencher
            .with_inputs(|| (tiled.clone().into_array(), SESSION.create_execution_ctx()))
            .bench_values(|(array, mut ctx)| {
                divan::black_box(array.execute::<FixedSizeListArray>(&mut ctx).unwrap())
            });
    }
}

mod slice_reduce {
    use super::*;

    #[divan::bench(args = [1_024usize, 1_000_000])]
    fn full_width_unaligned(bencher: Bencher, rows: usize) {
        let args = Args {
            rows,
            dimensions: 1,
            tile_rows: 64,
            tile_dimensions: TileDimensions::Full,
        };
        let canonical = canonical_u8(args);
        let mut ctx = SESSION.create_execution_ctx();
        let tiled = raw_tiled(args, &mut ctx).unwrap();
        let range = 1..rows - 1;
        let expected = canonical.into_array().slice(range.clone()).unwrap();
        let reduced = <TiledFixedSizeList as SliceReduce>::slice(tiled.as_view(), range.clone())
            .unwrap()
            .unwrap();
        assert_arrays_eq!(expected, reduced, &mut ctx);

        bencher
            .with_inputs(|| (tiled.clone(), range.clone()))
            .bench_values(|(array, range)| {
                divan::black_box(
                    <TiledFixedSizeList as SliceReduce>::slice(array.as_view(), range)
                        .unwrap()
                        .unwrap(),
                )
            });
    }
}

mod slice_execute {
    use super::*;

    #[divan::bench(args = [SliceKind::Small, SliceKind::Half])]
    fn multi_slab_unaligned(bencher: Bencher, kind: SliceKind) {
        let args = Args {
            rows: 16_384,
            dimensions: 128,
            tile_rows: 64,
            tile_dimensions: TileDimensions::Fixed(64),
        };
        let slice_args = SliceArgs { args, kind };
        let range = slice_range(slice_args);
        let canonical = canonical_u8(args);
        let mut ctx = SESSION.create_execution_ctx();
        let tiled = raw_tiled(args, &mut ctx).unwrap();
        let expected = canonical.into_array().slice(range.clone()).unwrap();
        let lazy_slice = tiled.clone().into_array().slice(range).unwrap();
        let executed = lazy_slice
            .clone()
            .execute::<FixedSizeListArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(expected, executed, &mut ctx);

        bencher
            .with_inputs(|| (lazy_slice.clone(), SESSION.create_execution_ctx()))
            .bench_values(|(array, mut ctx)| {
                divan::black_box(array.execute::<FixedSizeListArray>(&mut ctx).unwrap())
            });
    }
}

mod tile_iteration {
    use super::*;

    #[divan::bench]
    fn full_view(bencher: Bencher) {
        bench_view(bencher, None);
    }

    #[divan::bench]
    fn prefix_boundary(bencher: Bencher) {
        bench_view(bencher, Some(1..64));
    }

    #[divan::bench]
    fn two_boundaries(bencher: Bencher) {
        bench_view(bencher, Some(1..127));
    }

    fn bench_view(bencher: Bencher, range: Option<Range<usize>>) {
        let args = Args {
            rows: 1_024,
            dimensions: 128,
            tile_rows: 64,
            tile_dimensions: TileDimensions::Full,
        };
        let canonical = canonical_u8(args).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let tiled = raw_tiled(args, &mut ctx).unwrap().into_array();
        let (expected, tiled) = match range {
            Some(range) => (
                canonical.slice(range.clone()).unwrap(),
                tiled.slice(range).unwrap(),
            ),
            None => (canonical, tiled),
        };
        assert_arrays_eq!(expected, tiled, &mut ctx);
        let tiled = tiled.as_::<TiledFixedSizeList>().clone();

        bencher.with_inputs(|| tiled.clone()).bench_values(|array| {
            divan::black_box(
                array
                    .tiles()
                    .map(|tile| tile.physical_range.len())
                    .sum::<usize>(),
            )
        });
    }
}

#[divan::bench(args = slice_args())]
fn slice(bencher: Bencher, args: SliceArgs) {
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = raw_tiled(args.args, &mut ctx).unwrap();
    let range = slice_range(args);
    bencher
        .with_inputs(|| (tiled.clone(), range.clone()))
        .bench_values(|(array, range)| divan::black_box(array.slice(range).unwrap()));
}

#[divan::bench(args = take_args())]
fn take(bencher: Bencher, args: TakeArgs) {
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = raw_tiled(args.args, &mut ctx).unwrap();
    let indices = take_indices(args).into_array();
    bencher
        .with_inputs(|| {
            (
                tiled.clone().into_array(),
                indices.clone(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, indices, mut ctx)| {
            divan::black_box(
                array
                    .take(indices)
                    .unwrap()
                    .execute_until::<FixedSizeList>(&mut ctx)
                    .unwrap(),
            )
        });
}

#[divan::bench(args = args())]
fn score_canonical(bencher: Bencher, args: Args) {
    let canonical = canonical_u8(args);
    let mut ctx = SESSION.create_execution_ctx();
    let values = canonical
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let query = query(args);
    assert_adjacent_scores_distinct(values.as_slice::<u8>(), args, &query);
    bencher.bench(|| {
        divan::black_box(scoring::score_canonical(
            values.as_slice::<u8>(),
            args.rows,
            args.dimensions,
            &query,
        ))
    });
}

#[divan::bench(args = score_args())]
fn score_prepared(bencher: Bencher, args: ScoreArgs) {
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = tiled_score_fixture(args, &mut ctx).unwrap();
    let query = query(args.args);
    let physical = tiled
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    assert_scores_equal(
        args.args,
        &tiled,
        physical.as_slice::<u8>(),
        &query,
        &mut ctx,
    )
    .unwrap();

    bencher.bench(|| {
        divan::black_box(scoring::score_tiled(
            tiled.as_view(),
            physical.as_slice::<u8>(),
            &query,
        ))
    });
}

#[divan::bench(args = score_args())]
fn score_end_to_end(bencher: Bencher, args: ScoreArgs) {
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = tiled_score_fixture(args, &mut ctx).unwrap();
    let query = query(args.args);
    let physical = tiled
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    assert_scores_equal(
        args.args,
        &tiled,
        physical.as_slice::<u8>(),
        &query,
        &mut ctx,
    )
    .unwrap();

    bencher
        .with_inputs(|| (tiled.elements().clone(), SESSION.create_execution_ctx()))
        .bench_values(|(physical, mut ctx)| {
            let physical = physical.execute::<PrimitiveArray>(&mut ctx).unwrap();
            divan::black_box(scoring::score_tiled(
                tiled.as_view(),
                physical.as_slice::<u8>(),
                &query,
            ))
        });
}

mod scoring {
    use vortex_array::ArrayView;
    use vortex_tiled_fsl::TiledFixedSizeList;
    use vortex_tiled_fsl::TiledFixedSizeListArrayExt;

    pub(super) fn score_canonical(
        values: &[u8],
        rows: usize,
        dimensions: usize,
        query: &[u8],
    ) -> Vec<u64> {
        let mut scores = vec![0; rows];
        for (row, score) in scores.iter_mut().enumerate() {
            let row_values = &values[row * dimensions..(row + 1) * dimensions];
            *score = row_values
                .iter()
                .zip(query)
                .map(|(&value, &weight)| u64::from(value) * u64::from(weight))
                .sum();
        }
        scores
    }

    pub(super) fn score_tiled(
        array: ArrayView<'_, TiledFixedSizeList>,
        values: &[u8],
        query: &[u8],
    ) -> Vec<u64> {
        let mut scores = vec![0; array.len()];
        for bounds in array.tiles() {
            let tile_rows = bounds.row_range.len();
            for (dimension_offset, dimension) in bounds.dimension_range.clone().enumerate() {
                let physical_start = bounds.physical_range.start + dimension_offset * tile_rows;
                let weight = u64::from(query[dimension]);
                for (row_offset, row) in bounds.row_range.clone().enumerate() {
                    scores[row] += u64::from(values[physical_start + row_offset]) * weight;
                }
            }
        }
        scores
    }
}
