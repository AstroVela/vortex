use std::num::NonZeroU32;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::compute::conformance::filter::test_filter_conformance;
use vortex_array::compute::conformance::take::test_take_conformance;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_native_ptype;
use vortex_array::test_harness::check_metadata;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::TileBounds;
use crate::TileGeometry;
use crate::TiledFixedSizeList;
use crate::TiledFixedSizeListArray;
use crate::TiledFixedSizeListArrayExt;
use crate::TiledFixedSizeListArraySlotsExt;
use crate::TiledFixedSizeListMetadata;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

fn geometry(rows: u32, dimensions: u32) -> TileGeometry {
    TileGeometry::new(
        NonZeroU32::new(rows).unwrap(),
        NonZeroU32::new(dimensions).unwrap(),
    )
}

fn physical_fixture(
    rows: usize,
    dimensions: u32,
    geometry: TileGeometry,
) -> VortexResult<TiledFixedSizeListArray> {
    let _ = &*SESSION;
    TiledFixedSizeList::try_new(
        PrimitiveArray::from_iter(
            (0..rows * dimensions as usize).map(|index| u16::try_from(index).unwrap_or(u16::MAX)),
        )
        .into_array(),
        dimensions,
        Validity::NonNullable,
        rows,
        geometry,
    )
}

fn fixture(
    rows: usize,
    dimensions: u32,
    geometry: TileGeometry,
) -> VortexResult<(FixedSizeListArray, TiledFixedSizeListArray, ExecutionCtx)> {
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::from_iter(
            (0..rows * dimensions as usize)
                .map(|index| u8::try_from(index % 16).unwrap_or_default()),
        )
        .into_array(),
        dimensions,
        Validity::NonNullable,
        rows,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry, &mut ctx)?;
    Ok((canonical, tiled, ctx))
}

fn assert_fsl_equivalent(
    canonical: &ArrayRef,
    candidate: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    assert_eq!(candidate.dtype(), canonical.dtype());
    assert_eq!(candidate.len(), canonical.len());
    assert_arrays_eq!(canonical, candidate, ctx);
    Ok(())
}

fn encode_fixture<T: NativePType>(
    values: Buffer<T>,
    element_validity: Validity,
    list_size: u32,
    outer_validity: Validity,
    len: usize,
    geometry: TileGeometry,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::new(values, element_validity).into_array(),
        list_size,
        outer_validity,
        len,
    );
    Ok(TiledFixedSizeList::encode(canonical.as_view(), geometry, ctx)?.into_array())
}

fn expected_tile_bounds(
    rows: usize,
    dimensions: usize,
    tile_geometry: TileGeometry,
) -> VortexResult<Vec<TileBounds>> {
    let tile_rows = usize::try_from(tile_geometry.rows().get())?;
    let tile_dimensions = usize::try_from(tile_geometry.dimensions().get())?;
    let mut physical_cursor = 0;
    let mut bounds = Vec::new();

    for dimension_start in (0..dimensions).step_by(tile_dimensions) {
        let dimension_end = dimension_start
            .saturating_add(tile_dimensions)
            .min(dimensions);
        for row_start in (0..rows).step_by(tile_rows) {
            let row_end = row_start.saturating_add(tile_rows).min(rows);
            let tile_len = (row_end - row_start) * (dimension_end - dimension_start);
            bounds.push(TileBounds {
                row_range: row_start..row_end,
                dimension_range: dimension_start..dimension_end,
                physical_range: physical_cursor..physical_cursor + tile_len,
            });
            physical_cursor += tile_len;
        }
    }

    Ok(bounds)
}

fn boundary_slice_ranges(rows: usize, tile_rows: usize) -> Vec<Range<usize>> {
    let mut ranges = vec![0..0, 0..rows];
    for boundary in (tile_rows..rows).step_by(tile_rows) {
        ranges.extend([
            0..boundary - 1,
            0..boundary,
            0..boundary + 1,
            boundary - 1..boundary,
            boundary..boundary + 1,
            boundary - 1..boundary + 1,
        ]);
    }
    ranges.sort_by_key(|range| (range.start, range.end));
    ranges.dedup();
    ranges
}

fn conformance_take_indices(rows: usize) -> VortexResult<Vec<ArrayRef>> {
    let row_count = u32::try_from(rows)?;
    let mut cases = vec![PrimitiveArray::from_iter::<[u32; 0]>([]).into_array()];
    if rows == 0 {
        cases.push(PrimitiveArray::from_option_iter([None::<u32>]).into_array());
        return Ok(cases);
    }

    cases.extend([
        PrimitiveArray::from_iter(0..row_count).into_array(),
        PrimitiveArray::from_iter((0..row_count).rev()).into_array(),
        PrimitiveArray::from_iter([0, row_count - 1, 0]).into_array(),
        PrimitiveArray::from_iter([row_count - 1, 0, row_count / 2]).into_array(),
        PrimitiveArray::from_option_iter([Some(row_count - 1), None, Some(0)]).into_array(),
    ]);
    Ok(cases)
}

fn assert_scalar_conformance(
    canonical: &FixedSizeListArray,
    tiled: &TiledFixedSizeListArray,
    tile_geometry: TileGeometry,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if canonical.is_empty() {
        return Ok(());
    }

    let rows = canonical.len();
    let tile_rows = usize::try_from(tile_geometry.rows().get())?;
    let mut scalar_rows = vec![0, rows - 1];
    for boundary in (tile_rows..rows).step_by(tile_rows) {
        scalar_rows.extend([boundary - 1, boundary]);
    }
    scalar_rows.sort_unstable();
    scalar_rows.dedup();
    for row in scalar_rows {
        assert_eq!(
            canonical.execute_scalar(row, ctx)?,
            tiled.execute_scalar(row, ctx)?,
        );
    }
    Ok(())
}

fn assert_tile_conformance(
    canonical: &FixedSizeListArray,
    tiled: &TiledFixedSizeListArray,
    tile_geometry: TileGeometry,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let rows = canonical.len();
    let dimensions = usize::try_from(canonical.list_size())?;
    let tile_rows = usize::try_from(tile_geometry.rows().get())?;
    let tile_dimensions = usize::try_from(tile_geometry.dimensions().get())?;
    let expected_row_tile_count = rows.div_ceil(tile_rows);
    let expected_dimension_tile_count = dimensions.div_ceil(tile_dimensions);
    assert_eq!(tiled.row_tile_count(), expected_row_tile_count);
    assert_eq!(tiled.dimension_tile_count(), expected_dimension_tile_count);

    let expected_bounds = expected_tile_bounds(rows, dimensions, tile_geometry)?;
    let actual_bounds: Vec<TileBounds> = tiled.tiles().collect();
    assert_eq!(actual_bounds, expected_bounds);
    for (dimension_tile, dimension_bounds) in (0..dimensions).step_by(tile_dimensions).enumerate() {
        for (row_tile, row_bounds) in (0..rows).step_by(tile_rows).enumerate() {
            let expected = &expected_bounds[dimension_tile * expected_row_tile_count + row_tile];
            assert_eq!(tiled.tile(row_tile, dimension_tile)?, *expected);

            let indices = expected
                .dimension_range
                .clone()
                .flat_map(|dimension| {
                    expected
                        .row_range
                        .clone()
                        .map(move |row| row * dimensions + dimension)
                })
                .map(u64::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            let expected_elements = canonical
                .elements()
                .clone()
                .take(PrimitiveArray::from_iter(indices).into_array())?;
            let actual_elements = tiled.tile_elements(expected)?;
            assert_arrays_eq!(expected_elements, actual_elements, ctx);

            assert_eq!(dimension_bounds, expected.dimension_range.start);
            assert_eq!(row_bounds, expected.row_range.start);
        }
    }
    Ok(())
}

fn assert_slice_conformance(
    canonical: &FixedSizeListArray,
    tiled: &TiledFixedSizeListArray,
    tile_geometry: TileGeometry,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let tile_rows = usize::try_from(tile_geometry.rows().get())?;
    for range in boundary_slice_ranges(canonical.len(), tile_rows) {
        let expected = canonical.clone().into_array().slice(range.clone())?;
        let actual = tiled.clone().into_array().slice(range)?;
        if canonical.is_empty() {
            assert!(actual.is::<TiledFixedSizeList>());
            assert_eq!(actual.as_::<TiledFixedSizeList>().geometry(), tile_geometry);
        } else if actual.is_empty() {
            assert!(actual.is::<FixedSizeList>());
        } else {
            assert!(actual.is::<TiledFixedSizeList>());
            assert_eq!(actual.as_::<TiledFixedSizeList>().geometry(), tile_geometry);
        }
        assert_fsl_equivalent(&expected, &actual, ctx)?;
    }
    Ok(())
}

fn assert_take_oracle_conformance(
    canonical: &FixedSizeListArray,
    tiled: &TiledFixedSizeListArray,
    tile_geometry: TileGeometry,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    for indices in conformance_take_indices(canonical.len())? {
        let expected = canonical.clone().into_array().take(indices.clone())?;
        let actual = tiled.clone().into_array().take(indices)?;
        let actual = if !canonical.is_empty() && !actual.is_empty() {
            let actual = actual.execute_until::<TiledFixedSizeList>(ctx)?;
            assert_eq!(actual.as_::<TiledFixedSizeList>().geometry(), tile_geometry);
            actual
        } else if actual.is_empty() {
            let actual = actual.execute_until::<FixedSizeList>(ctx)?;
            assert!(actual.is::<FixedSizeList>());
            actual
        } else {
            let actual = actual.execute_until::<Constant>(ctx)?;
            assert!(actual.is::<Constant>());
            actual
        };
        assert_fsl_equivalent(&expected, &actual, ctx)?;
    }
    Ok(())
}

fn assert_tiled_conformance_case(
    canonical: &FixedSizeListArray,
    tile_geometry: TileGeometry,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = TiledFixedSizeList::encode(canonical.as_view(), tile_geometry, &mut ctx)?;
    assert_eq!(tiled.geometry(), tile_geometry);

    let encoded = tiled
        .clone()
        .into_array()
        .execute::<FixedSizeListArray>(&mut ctx)?
        .into_array();
    assert_fsl_equivalent(&canonical.clone().into_array(), &encoded, &mut ctx)?;

    let reconstructed = TiledFixedSizeList::try_new(
        tiled.elements().clone(),
        canonical.list_size(),
        canonical.fixed_size_list_validity(),
        canonical.len(),
        tile_geometry,
    )?;
    assert_fsl_equivalent(
        &canonical.clone().into_array(),
        &reconstructed.into_array(),
        &mut ctx,
    )?;

    assert_scalar_conformance(canonical, &tiled, tile_geometry, &mut ctx)?;
    assert_tile_conformance(canonical, &tiled, tile_geometry, &mut ctx)?;
    assert_slice_conformance(canonical, &tiled, tile_geometry, &mut ctx)?;
    assert_take_oracle_conformance(canonical, &tiled, tile_geometry, &mut ctx)
}

#[test]
fn standard_harness_conformance() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let fixtures = vec![
        encode_fixture(
            buffer![0u8, 1, 2, 3, 4, 5],
            Validity::NonNullable,
            2,
            Validity::NonNullable,
            3,
            geometry(2, 2),
            &mut ctx,
        )?,
        encode_fixture(
            buffer![0i32, 1, 2, 3, 4, 5],
            Validity::NonNullable,
            2,
            Validity::from_iter([true, false, true]),
            3,
            geometry(2, 2),
            &mut ctx,
        )?,
        encode_fixture(
            buffer![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
            Validity::from_iter([true, false, true, true, false, true]),
            2,
            Validity::NonNullable,
            3,
            geometry(2, 2),
            &mut ctx,
        )?,
        encode_fixture(
            buffer![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0],
            Validity::from_iter([true, false, true, true, false, true]),
            2,
            Validity::from_iter([true, false, true]),
            3,
            geometry(2, 2),
            &mut ctx,
        )?,
        encode_fixture(
            buffer![0u8; 0],
            Validity::NonNullable,
            5,
            Validity::NonNullable,
            0,
            geometry(32, 64),
            &mut ctx,
        )?,
        encode_fixture(
            buffer![0u8; 0],
            Validity::NonNullable,
            0,
            Validity::NonNullable,
            65,
            geometry(32, 64),
            &mut ctx,
        )?,
        encode_fixture(
            buffer![7u8; 65 * 129],
            Validity::NonNullable,
            129,
            Validity::NonNullable,
            65,
            geometry(32, 64),
            &mut ctx,
        )?,
        encode_fixture(
            buffer![7u8; 3 * 5],
            Validity::NonNullable,
            5,
            Validity::NonNullable,
            3,
            geometry(32, 64),
            &mut ctx,
        )?,
    ];

    for tiled in fixtures {
        test_array_consistency(&tiled, &mut ctx);
        test_filter_conformance(&tiled, &mut ctx);
        test_take_conformance(&tiled, &mut ctx);
    }

    let (_, raw_tiled, _) = fixture(65, 128, geometry(32, 64))?;
    vortex_fastlanes::initialize(ctx.session());
    let physical = raw_tiled
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let bitpacked = bitpack_encode(&physical, 4, None, &mut ctx)?.into_array();
    let bitpacked_tiled = TiledFixedSizeList::try_new(
        bitpacked,
        128,
        raw_tiled.array_validity(),
        65,
        geometry(32, 64),
    )?
    .into_array();
    test_array_consistency(&bitpacked_tiled, &mut ctx);
    test_filter_conformance(&bitpacked_tiled, &mut ctx);
    test_take_conformance(&bitpacked_tiled, &mut ctx);
    Ok(())
}

#[test]
fn canonical_oracle_conformance_matrix() -> VortexResult<()> {
    const ROW_COUNTS: &[usize] = &[0, 1, 15, 16, 31, 32, 33, 63, 64, 65];
    const DIMENSION_COUNTS: &[u32] = &[0, 1, 3, 4, 63, 64, 65, 129];

    for &rows in ROW_COUNTS {
        for &dimensions in DIMENSION_COUNTS {
            let dimension_count = usize::try_from(dimensions)?;
            let values = (0..rows * dimension_count)
                .map(u16::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            let canonical = FixedSizeListArray::new(
                PrimitiveArray::new(Buffer::from(values), Validity::NonNullable).into_array(),
                dimensions,
                Validity::NonNullable,
                rows,
            );
            let full_width = dimensions.max(1);
            for tile_geometry in [
                geometry(16, 4),
                geometry(32, 64),
                geometry(64, 64),
                geometry(64, full_width),
            ] {
                assert_tiled_conformance_case(&canonical, tile_geometry)?;
            }
        }
    }
    Ok(())
}

#[test]
fn physical_indices_reject_overflowing_output_cardinality() {
    assert!(
        crate::gather::physical_indices_for_rows(
            1,
            usize::MAX,
            geometry(1, 1),
            &[Some(0), Some(0)],
        )
        .is_err()
    );
}

#[test]
fn collect_checked_rows_rejects_mismatched_mask_length() {
    assert!(crate::take::collect_checked_rows(&[0u32, 1], &Mask::new_true(1), 2).is_err());
}

#[test]
fn take_conformance() {
    let (_, tiled, mut ctx) = fixture(65, 129, geometry(32, 64)).unwrap();
    test_take_conformance(&tiled.into_array(), &mut ctx);
}

#[test]
fn take_preserves_encoding_geometry_order_duplicates_and_nulls() -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(65, 129, geometry(32, 64))?;
    let indices = PrimitiveArray::new(
        buffer![64u32, 1, 1, 32, 0],
        Validity::from_iter([true, true, false, true, true]),
    )
    .into_array();
    let actual = tiled
        .into_array()
        .take(indices.clone())?
        .execute_until::<TiledFixedSizeList>(&mut ctx)?;
    assert!(actual.is::<TiledFixedSizeList>());
    assert_eq!(
        actual.as_::<TiledFixedSizeList>().geometry(),
        geometry(32, 64)
    );
    assert_arrays_eq!(canonical.into_array().take(indices)?, actual, &mut ctx);
    Ok(())
}

fn assert_take_indices(indices: ArrayRef) -> VortexResult<()> {
    let index_count = indices.len();
    let (canonical, tiled, mut ctx) = fixture(3, 5, geometry(2, 3))?;
    let expected = canonical.into_array().take(indices.clone())?;
    let actual = tiled
        .into_array()
        .take(indices)?
        .execute_until::<TiledFixedSizeList>(&mut ctx)?;
    assert_eq!(
        actual.as_::<TiledFixedSizeList>().elements().len(),
        index_count * 5,
    );
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}

#[test]
fn take_accepts_all_integer_index_ptypes() -> VortexResult<()> {
    assert_take_indices(PrimitiveArray::from_iter([2u8, 0, 1]).into_array())?;
    assert_take_indices(PrimitiveArray::from_iter([2u16, 0, 1]).into_array())?;
    assert_take_indices(PrimitiveArray::from_iter([2u32, 0, 1]).into_array())?;
    assert_take_indices(PrimitiveArray::from_iter([2u64, 0, 1]).into_array())?;
    assert_take_indices(PrimitiveArray::from_iter([2i8, 0, 1]).into_array())?;
    assert_take_indices(PrimitiveArray::from_iter([2i16, 0, 1]).into_array())?;
    assert_take_indices(PrimitiveArray::from_iter([2i32, 0, 1]).into_array())?;
    assert_take_indices(PrimitiveArray::from_iter([2i64, 0, 1]).into_array())?;
    Ok(())
}

#[test]
fn take_preserves_outer_and_element_validity() -> VortexResult<()> {
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::new(
            buffer![0i32, 1, 10, 11, 20, 21],
            Validity::from_iter([true, false, true, true, false, true]),
        )
        .into_array(),
        2,
        Validity::from_iter([true, false, true]),
        3,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry(2, 2), &mut ctx)?;
    let indices =
        PrimitiveArray::from_option_iter([Some(2u32), Some(0), None, Some(1)]).into_array();
    let expected = canonical.into_array().take(indices.clone())?;
    let actual = tiled
        .into_array()
        .take(indices)?
        .execute_until::<TiledFixedSizeList>(&mut ctx)?;
    assert_eq!(
        actual.as_::<TiledFixedSizeList>().geometry(),
        geometry(2, 2),
    );
    assert_eq!(actual.as_::<TiledFixedSizeList>().elements().len(), 8);
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}

#[test]
fn take_all_null_indices_from_empty_source() -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(0, 5, geometry(2, 3))?;
    let indices = PrimitiveArray::from_option_iter([None::<u32>, None]).into_array();
    let expected = canonical.into_array().take(indices.clone())?;
    let actual = tiled.into_array().take(indices)?;
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}

#[test]
fn take_empty_indices_is_canonical_empty() -> VortexResult<()> {
    let (_, tiled, mut ctx) = fixture(3, 5, geometry(2, 3))?;
    let indices = PrimitiveArray::from_iter::<[u32; 0]>([]).into_array();
    let actual = tiled
        .into_array()
        .take(indices)?
        .execute_until::<FixedSizeList>(&mut ctx)?;
    assert!(actual.is::<FixedSizeList>());
    assert!(actual.is_empty());
    Ok(())
}

#[test]
fn take_zero_width_lists() -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(3, 0, geometry(2, 3))?;
    let indices = PrimitiveArray::from_iter([2u32, 0]).into_array();
    let expected = canonical.into_array().take(indices.clone())?;
    let actual = tiled
        .into_array()
        .take(indices)?
        .execute_until::<TiledFixedSizeList>(&mut ctx)?;
    assert_eq!(actual.as_::<TiledFixedSizeList>().elements().len(), 0);
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}

#[test]
fn collect_checked_rows_rejects_out_of_bounds_valid_index() {
    assert!(crate::take::collect_checked_rows(&[3u32], &Mask::new_true(1), 3).is_err());
}

#[test]
fn collect_checked_rows_rejects_negative_valid_index() {
    assert!(crate::take::collect_checked_rows(&[-1i64], &Mask::new_true(1), 3).is_err());
}

#[test]
fn collect_checked_rows_rejects_u64_max() {
    assert!(crate::take::collect_checked_rows(&[u64::MAX], &Mask::new_true(1), 3).is_err());
}

#[test]
fn nullable_take_makes_outer_dtype_nullable() -> VortexResult<()> {
    let (_, tiled, mut ctx) = fixture(3, 5, geometry(2, 3))?;
    let indices = PrimitiveArray::from_option_iter([Some(2u32), None]).into_array();
    let actual = tiled
        .into_array()
        .take(indices)?
        .execute_until::<TiledFixedSizeList>(&mut ctx)?;
    assert!(actual.dtype().is_nullable());
    assert_eq!(actual.as_::<TiledFixedSizeList>().elements().len(), 10);
    Ok(())
}

#[rstest]
#[case(0..0)]
#[case(0..1)]
#[case(0..31)]
#[case(0..32)]
#[case(0..33)]
#[case(1..64)]
#[case(31..65)]
#[case(64..65)]
fn slice_preserves_encoding_and_values(#[case] range: Range<usize>) -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(65, 129, geometry(32, 64))?;
    let expected = canonical.into_array().slice(range.clone())?;
    let actual = tiled.into_array().slice(range)?;
    if actual.is_empty() {
        assert!(actual.is::<FixedSizeList>());
    } else {
        assert!(actual.is::<TiledFixedSizeList>());
        assert_eq!(
            actual.as_::<TiledFixedSizeList>().geometry(),
            geometry(32, 64)
        );
    }
    assert_arrays_eq!(expected, actual, &mut ctx);
    Ok(())
}

#[test]
fn slice_preserves_special_cases() -> VortexResult<()> {
    let cases = [
        (3, 0, geometry(2, 3), 1..3),
        (3, 5, geometry(32, 64), 1..3),
        (4_096, 128, geometry(32, 64), 2_048..2_049),
    ];
    for (rows, dimensions, tile_geometry, range) in cases {
        let (canonical, tiled, mut ctx) = fixture(rows, dimensions, tile_geometry)?;
        let expected = canonical.into_array().slice(range.clone())?;
        let actual = tiled.into_array().slice(range.clone())?;
        assert_fsl_equivalent(&expected, &actual, &mut ctx)?;
        if !actual.is_empty() {
            let actual = actual.as_::<TiledFixedSizeList>();
            assert_eq!(actual.geometry(), tile_geometry);
            assert_eq!(actual.elements().len(), range.len() * dimensions as usize);
        }
    }
    Ok(())
}

#[test]
fn slice_preserves_mixed_validity() -> VortexResult<()> {
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::new(
            buffer![0i32, 1, 10, 11, 20, 21],
            Validity::from_iter([true, false, true, true, false, true]),
        )
        .into_array(),
        2,
        Validity::from_iter([true, false, true]),
        3,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry(2, 2), &mut ctx)?;
    let expected = canonical.into_array().slice(1..3)?;
    let actual = tiled.into_array().slice(1..3)?;
    assert_eq!(
        actual.as_::<TiledFixedSizeList>().geometry(),
        geometry(2, 2),
    );
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}

#[test]
fn bitpacked_child_roundtrips_and_remains_tiled_after_row_ops() -> VortexResult<()> {
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::from_iter((0..65).flat_map(|row| {
            let row_value = u8::try_from(row / 32).unwrap();
            (0..128).map(move |dimension| row_value + u8::try_from(dimension % 14).unwrap())
        }))
        .into_array(),
        128,
        Validity::NonNullable,
        65,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let raw_tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry(32, 64), &mut ctx)?;
    let first_row = canonical.execute_scalar(0, &mut ctx)?;
    let last_row = canonical.execute_scalar(64, &mut ctx)?;
    assert_ne!(first_row, last_row);
    let expected_sliced = canonical.clone().into_array().slice(1..64)?;
    let indices = PrimitiveArray::from_iter([64u32, 0, 32]).into_array();
    let expected_taken = canonical.clone().into_array().take(indices.clone())?;
    vortex_fastlanes::initialize(ctx.session());
    let physical = raw_tiled
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let bitpacked = bitpack_encode(&physical, 4, None, &mut ctx)?.into_array();
    let tiled = TiledFixedSizeList::try_new(
        bitpacked,
        128,
        raw_tiled.array_validity(),
        65,
        geometry(32, 64),
    )?;

    assert_arrays_eq!(canonical, tiled, &mut ctx);

    let sliced = tiled.clone().into_array().slice(1..64)?;
    assert!(sliced.is::<TiledFixedSizeList>());
    assert_arrays_eq!(expected_sliced, sliced, &mut ctx);

    let taken = tiled
        .into_array()
        .take(indices)?
        .execute_until::<TiledFixedSizeList>(&mut ctx)?;
    assert!(taken.is::<TiledFixedSizeList>());
    assert_arrays_eq!(expected_taken, taken, &mut ctx);
    Ok(())
}

#[test]
fn tiles_are_range_only_and_in_physical_order() -> VortexResult<()> {
    let (_, tiled, _) = fixture(3, 5, geometry(2, 3))?;
    let bounds: Vec<TileBounds> = tiled.tiles().collect();
    assert_eq!(
        bounds
            .iter()
            .map(|bounds| bounds.physical_range.clone())
            .collect::<Vec<_>>(),
        vec![0..6, 6..9, 9..13, 13..15],
    );
    assert_eq!(tiled.tile(1, 1)?, bounds[3]);
    assert_eq!(tiled.tile_elements(&bounds[3])?.len(), 2);
    assert!(tiled.tile(2, 0).is_err());
    assert!(tiled.tile(0, 2).is_err());
    Ok(())
}

#[test]
fn degenerate_arrays_have_no_tiles() -> VortexResult<()> {
    assert_eq!(physical_fixture(0, 5, geometry(2, 3))?.tiles().count(), 0);
    assert_eq!(physical_fixture(3, 0, geometry(2, 3))?.tiles().count(), 0);
    Ok(())
}

#[test]
fn golden_physical_child_and_round_trip() -> VortexResult<()> {
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::from_iter([0u16, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24])
            .into_array(),
        5,
        Validity::NonNullable,
        3,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry(2, 3), &mut ctx)?;
    let physical = tiled
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(
        physical.as_slice::<u16>(),
        &[0, 10, 1, 11, 2, 12, 20, 21, 22, 3, 13, 4, 14, 23, 24]
    );
    assert_fsl_equivalent(&canonical.into_array(), &tiled.into_array(), &mut ctx)
}

#[test]
fn all_native_ptypes_round_trip() -> VortexResult<()> {
    for ptype in [
        PType::U8,
        PType::U16,
        PType::U32,
        PType::U64,
        PType::I8,
        PType::I16,
        PType::I32,
        PType::I64,
        PType::F16,
        PType::F32,
        PType::F64,
    ] {
        match_each_native_ptype!(ptype, |T| {
            let canonical = FixedSizeListArray::new(
                PrimitiveArray::new(Buffer::<T>::zeroed(15), Validity::NonNullable).into_array(),
                5,
                Validity::NonNullable,
                3,
            );
            let mut ctx = SESSION.create_execution_ctx();
            let tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry(2, 3), &mut ctx)?;
            assert_fsl_equivalent(&canonical.into_array(), &tiled.into_array(), &mut ctx)
        })?;
    }
    Ok(())
}

#[test]
fn independent_outer_and_element_validity_round_trip() -> VortexResult<()> {
    let cases = [
        (
            Validity::NonNullable,
            Validity::NonNullable,
            Validity::NonNullable,
        ),
        (Validity::AllValid, Validity::AllValid, Validity::AllValid),
        (
            Validity::AllInvalid,
            Validity::AllInvalid,
            Validity::AllInvalid,
        ),
        (
            Validity::from_iter([true, false, true, true, false, true]),
            Validity::from_iter([true, false, true]),
            Validity::from_iter([true, true, false, false, true, true]),
        ),
    ];

    for (element_validity, outer_validity, expected_physical_validity) in cases {
        let canonical = FixedSizeListArray::new(
            PrimitiveArray::new(
                Buffer::copy_from([0u16, 1, 10, 11, 20, 21]),
                element_validity,
            )
            .into_array(),
            2,
            outer_validity,
            3,
        );
        let mut ctx = SESSION.create_execution_ctx();
        let tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry(2, 1), &mut ctx)?;

        assert!(
            tiled
                .elements()
                .validity()?
                .mask_eq(&expected_physical_validity, 6, &mut ctx,)?
        );
        assert!(tiled.array_validity().mask_eq(
            &canonical.fixed_size_list_validity(),
            3,
            &mut ctx,
        )?);

        let executed = tiled.into_array().execute::<FixedSizeListArray>(&mut ctx)?;
        assert_fsl_equivalent(&canonical.into_array(), &executed.into_array(), &mut ctx)?;
    }
    Ok(())
}

#[test]
fn mixed_validity_round_trip_with_partial_row_and_dimension_tiles() -> VortexResult<()> {
    let canonical_validity = Validity::from_iter([
        true, false, false, true, true, false, true, false, true, false, true, true, false, false,
        true,
    ]);
    let expected_physical_validity = Validity::from_iter([
        true, false, false, true, false, false, true, true, false, true, true, true, false, false,
        true,
    ]);
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::new(Buffer::copy_from([0u16; 15]), canonical_validity.clone()).into_array(),
        5,
        Validity::NonNullable,
        3,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry(2, 3), &mut ctx)?;

    assert!(
        tiled
            .elements()
            .validity()?
            .mask_eq(&expected_physical_validity, 15, &mut ctx,)?
    );
    let decoded = tiled.into_array().execute::<FixedSizeListArray>(&mut ctx)?;
    assert!(
        decoded
            .elements()
            .validity()?
            .mask_eq(&canonical_validity, 15, &mut ctx,)?
    );
    assert_fsl_equivalent(&canonical.into_array(), &decoded.into_array(), &mut ctx)
}

#[test]
fn degenerate_arrays_encode_execute_and_zero_width_scalars() -> VortexResult<()> {
    let zero_rows = FixedSizeListArray::new(
        PrimitiveArray::from_iter(std::iter::empty::<u8>()).into_array(),
        5,
        Validity::NonNullable,
        0,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let tiled_zero_rows =
        TiledFixedSizeList::encode(zero_rows.as_view(), geometry(32, 64), &mut ctx)?;
    let decoded_zero_rows = tiled_zero_rows
        .into_array()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    assert_fsl_equivalent(
        &zero_rows.into_array(),
        &decoded_zero_rows.into_array(),
        &mut ctx,
    )?;

    let zero_width = FixedSizeListArray::new(
        PrimitiveArray::from_iter(std::iter::empty::<u8>()).into_array(),
        0,
        Validity::NonNullable,
        3,
    );
    let tiled_zero_width =
        TiledFixedSizeList::encode(zero_width.as_view(), geometry(32, 64), &mut ctx)?;
    let decoded_zero_width = tiled_zero_width
        .clone()
        .into_array()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    assert_fsl_equivalent(
        &zero_width.clone().into_array(),
        &decoded_zero_width.into_array(),
        &mut ctx,
    )?;
    for row in 0..zero_width.len() {
        assert_eq!(
            zero_width.execute_scalar(row, &mut ctx)?,
            tiled_zero_width.execute_scalar(row, &mut ctx)?,
        );
    }
    Ok(())
}

#[test]
fn scalar_at_matches_canonical_boundaries() -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(65, 129, geometry(32, 64))?;
    for row in [0, 31, 32, 63, 64] {
        assert_eq!(
            canonical.execute_scalar(row, &mut ctx)?,
            tiled.execute_scalar(row, &mut ctx)?,
        );
    }
    Ok(())
}

#[test]
fn try_new_derives_dtype_and_accessors() -> VortexResult<()> {
    let tiled = physical_fixture(3, 5, geometry(2, 3))?;
    assert_eq!(tiled.len(), 3);
    assert_eq!(tiled.list_size(), 5);
    assert_eq!(tiled.geometry(), geometry(2, 3));
    assert_eq!(tiled.row_tile_count(), 2);
    assert_eq!(tiled.dimension_tile_count(), 2);
    assert_eq!(
        tiled.dtype(),
        &DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U16, Nullability::NonNullable)),
            5,
            Nullability::NonNullable,
        )
    );
    Ok(())
}

#[rstest]
#[case(3, 5, 14)]
#[case(3, 5, 16)]
fn rejects_wrong_child_length(
    #[case] rows: usize,
    #[case] dimensions: u32,
    #[case] physical_len: usize,
) {
    let elements = PrimitiveArray::from_iter(
        (0..physical_len).map(|index| u16::try_from(index).unwrap_or(u16::MAX)),
    )
    .into_array();
    assert!(
        TiledFixedSizeList::try_new(
            elements,
            dimensions,
            Validity::NonNullable,
            rows,
            geometry(2, 3),
        )
        .is_err()
    );
}

#[test]
fn tiled_fsl_metadata() {
    check_metadata(
        "tiled_fsl.metadata",
        &TiledFixedSizeListMetadata {
            tile_rows: 32,
            tile_dimensions: 64,
        }
        .encode_to_vec(),
    );
}

#[test]
fn rejects_non_primitive_child() {
    assert!(
        TiledFixedSizeList::try_new(
            VarBinViewArray::from_iter_str(["x"]).into_array(),
            1,
            Validity::NonNullable,
            1,
            geometry(1, 1),
        )
        .is_err()
    );
}

#[test]
fn rejects_wrong_outer_validity_length() {
    assert!(
        TiledFixedSizeList::try_new(
            PrimitiveArray::from_iter([1u8, 2, 3]).into_array(),
            1,
            Validity::from_iter([true, false]),
            3,
            geometry(2, 1),
        )
        .is_err()
    );
}

#[test]
fn rejects_len_times_list_size_overflow() {
    assert!(
        TiledFixedSizeList::try_new(
            PrimitiveArray::from_iter(std::iter::empty::<u8>()).into_array(),
            2,
            Validity::NonNullable,
            usize::MAX,
            geometry(1, 1),
        )
        .is_err()
    );
}

#[test]
fn rejects_zero_geometry_metadata() {
    let metadata = TiledFixedSizeListMetadata {
        tile_rows: 0,
        tile_dimensions: 64,
    };
    assert!(TileGeometry::try_from(&metadata).is_err());
}

#[test]
fn accepts_degenerate_and_oversized_geometry() -> VortexResult<()> {
    physical_fixture(0, 5, geometry(32, 64))?;
    physical_fixture(3, 0, geometry(32, 64))?;
    physical_fixture(3, 5, geometry(32, 64))?;
    Ok(())
}

#[test]
fn max_usize_representable_geometry_constructs_and_traverses() -> VortexResult<()> {
    let tiled = physical_fixture(1, 1, geometry(u32::MAX, u32::MAX))?;
    let bounds: Vec<TileBounds> = tiled.tiles().collect();
    assert_eq!(bounds.len(), 1);
    assert_eq!(bounds[0].row_range, 0..1);
    assert_eq!(bounds[0].dimension_range, 0..1);
    assert_eq!(bounds[0].physical_range, 0..1);
    Ok(())
}
