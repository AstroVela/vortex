// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroU32;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PiecewiseSequence;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::piecewise_sequence::array::PiecewiseSequenceArraySlotsExt;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::compute::conformance::filter::test_filter_conformance;
use vortex_array::compute::conformance::take::test_take_conformance;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_native_ptype;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::serde::ArrayChildren;
use vortex_array::test_harness::check_metadata;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_session::VortexSession;

use crate::TileBounds;
use crate::TileGeometry;
use crate::TiledFixedSizeList;
use crate::TiledFixedSizeListArray;
use crate::TiledFixedSizeListArrayExt;
use crate::TiledFixedSizeListArraySlotsExt;
use crate::TiledFixedSizeListMetadata;
use crate::transpose::decode_elements;
use crate::transpose::encode_elements;

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

#[test]
fn bitmap_validity_follows_value_permutation() -> VortexResult<()> {
    let canonical_validity = Validity::from_iter([true, false, true, true]);
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::new(buffer![10u16, 11, 20, 21], canonical_validity.clone()).into_array(),
        2,
        Validity::NonNullable,
        2,
    );
    let mut ctx = SESSION.create_execution_ctx();

    let tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry(2, 2), &mut ctx)?;

    let tiled_elements = tiled
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(tiled_elements.as_slice::<u16>(), &[10, 20, 11, 21]);
    assert!(tiled.elements().validity()?.mask_eq(
        &Validity::from_iter([true, true, false, true]),
        4,
        &mut ctx,
    )?);

    let decoded = tiled.into_array().execute::<FixedSizeListArray>(&mut ctx)?;
    assert!(
        decoded
            .elements()
            .validity()?
            .mask_eq(&canonical_validity, 4, &mut ctx,)?
    );
    assert_fsl_equivalent(&canonical.into_array(), &decoded.into_array(), &mut ctx)?;
    Ok(())
}

#[test]
fn constant_validity_forms_round_trip() -> VortexResult<()> {
    for validity in [
        Validity::NonNullable,
        Validity::AllValid,
        Validity::AllInvalid,
    ] {
        let canonical = PrimitiveArray::new(buffer![10u16, 11, 20, 21], validity.clone());
        let mut ctx = SESSION.create_execution_ctx();

        let tiled = encode_elements(canonical.as_view(), 2, 2, geometry(2, 2), &mut ctx)?;
        assert!(same_validity_form(tiled.validity()?, &validity));

        let decoded = decode_elements(tiled.as_view(), 2, 2, geometry(2, 2), &mut ctx)?;
        assert_eq!(decoded.as_slice::<u16>(), canonical.as_slice::<u16>());
        assert!(same_validity_form(
            decoded.validity()?,
            &canonical.validity()?
        ));
    }
    Ok(())
}

fn same_validity_form(actual: Validity, expected: &Validity) -> bool {
    matches!(
        (actual, expected),
        (Validity::NonNullable, Validity::NonNullable)
            | (Validity::AllValid, Validity::AllValid)
            | (Validity::AllInvalid, Validity::AllInvalid)
    )
}

#[test]
fn encode_elements_rejects_mismatched_extent() {
    let elements = PrimitiveArray::from_iter([10u16, 11, 20]);
    let mut ctx = SESSION.create_execution_ctx();

    let error = encode_elements(elements.as_view(), 2, 2, geometry(2, 2), &mut ctx).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("physical child length 3 does not match logical extent (2, 2)")
    );
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

fn offset_view_fixture() -> VortexResult<(ArrayRef, TiledFixedSizeListArray, ExecutionCtx)> {
    offset_view_fixture_with_backing_rows(192)
}

fn offset_view_fixture_with_backing_rows(
    backing_rows: usize,
) -> VortexResult<(ArrayRef, TiledFixedSizeListArray, ExecutionCtx)> {
    let dimensions = 8u32;
    let element_count = backing_rows * dimensions as usize;
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::new(
            Buffer::from_iter((0..element_count).map(|index| u16::try_from(index).unwrap())),
            Validity::from_iter((0..element_count).map(|index| index % 11 != 0)),
        )
        .into_array(),
        dimensions,
        Validity::AllValid,
        backing_rows,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry(64, 8), &mut ctx)?;
    let oracle = canonical.clone().into_array().slice(10..138)?;
    let view = TiledFixedSizeList::try_new_view(
        tiled.elements().clone(),
        dimensions,
        canonical.fixed_size_list_validity().slice(10..138)?,
        128,
        geometry(64, 8),
        10,
        backing_rows,
    )?;
    Ok((oracle, view, ctx))
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

fn assert_array_tree_validity(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    for descendant in array.depth_first_traversal() {
        descendant.validity()?.execute_mask(descendant.len(), ctx)?;
    }
    Ok(())
}

fn mixed_validity_fixture(
    rows: usize,
    dimensions: u32,
    tile_geometry: TileGeometry,
) -> VortexResult<(FixedSizeListArray, TiledFixedSizeListArray, ExecutionCtx)> {
    let element_count = rows
        .checked_mul(usize::try_from(dimensions)?)
        .ok_or_else(|| vortex_err!("mixed-validity fixture extent overflows usize"))?;
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::new(
            Buffer::from_iter((0..element_count).map(|index| index as u32)),
            Validity::from_iter((0..element_count).map(|index| index % 11 != 0)),
        )
        .into_array(),
        dimensions,
        Validity::from_iter((0..rows).map(|row| row % 7 != 0)),
        rows,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = TiledFixedSizeList::encode(canonical.as_view(), tile_geometry, &mut ctx)?;
    Ok((canonical, tiled, ctx))
}

#[rstest]
#[case(128, 128, 10, 150, 60, 70, true)]
#[case(129, 64, 64, 192, 1, 65, true)]
#[case(129, 64, 1, 130, 1, 64, false)]
fn row_view_path_conformance(
    #[case] dimensions: u32,
    #[case] tile_dimensions: u32,
    #[case] start: usize,
    #[case] stop: usize,
    #[case] nested_start: usize,
    #[case] nested_stop: usize,
    #[case] expect_tiled: bool,
) -> VortexResult<()> {
    let tile_geometry = geometry(64, tile_dimensions);
    let (canonical, tiled, mut ctx) = mixed_validity_fixture(200, dimensions, tile_geometry)?;
    let expected = canonical.clone().into_array().slice(start..stop)?;
    let actual = tiled.into_array().slice(start..stop)?;

    assert_eq!(actual.is::<TiledFixedSizeList>(), expect_tiled);
    assert_eq!(actual.is::<Slice>(), !expect_tiled);

    let tile_boundary = 64 - start % 64;
    let mut probes = vec![0, actual.len() - 1];
    if tile_boundary < actual.len() {
        probes.extend([tile_boundary - 1, tile_boundary]);
    }
    for row in probes {
        assert_eq!(
            expected.execute_scalar(row, &mut ctx)?,
            actual.execute_scalar(row, &mut ctx)?,
        );
    }

    let nested_expected = expected.clone().slice(nested_start..nested_stop)?;
    let nested_actual = actual.clone().slice(nested_start..nested_stop)?;
    if dimensions == 128 {
        assert!(nested_actual.is::<TiledFixedSizeList>());
        assert_eq!(start + nested_start..start + nested_stop, 70..80);
    } else {
        assert!(nested_actual.is::<Slice>());
    }
    assert_fsl_equivalent(&nested_expected, &nested_actual, &mut ctx)?;

    let indices =
        PrimitiveArray::from_option_iter([Some(u64::try_from(actual.len() - 1)?), None, Some(0)])
            .into_array();
    assert_arrays_eq!(
        expected.clone().take(indices.clone())?,
        actual.clone().take(indices)?,
        &mut ctx
    );

    assert!(
        actual
            .validity()?
            .mask_eq(&expected.validity()?, actual.len(), &mut ctx,)?
    );
    let expected_fsl = expected.execute::<FixedSizeListArray>(&mut ctx)?;
    let actual_fsl = actual.clone().execute::<FixedSizeListArray>(&mut ctx)?;
    assert!(actual_fsl.elements().validity()?.mask_eq(
        &expected_fsl.elements().validity()?,
        actual_fsl.elements().len(),
        &mut ctx,
    )?);
    assert_array_tree_validity(&actual, &mut ctx)?;
    assert_fsl_equivalent(
        &expected_fsl.into_array(),
        &actual_fsl.into_array(),
        &mut ctx,
    )
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
            bounds.push(TileBounds::new(
                row_start..row_end,
                dimension_start..dimension_end,
                physical_cursor..physical_cursor + tile_len,
                0..row_end - row_start,
                row_end - row_start == tile_rows,
            ));
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
        let actual = tiled.clone().into_array().slice(range.clone())?;
        if canonical.is_empty() {
            assert!(actual.is::<TiledFixedSizeList>());
            assert_eq!(actual.as_::<TiledFixedSizeList>().geometry(), tile_geometry);
        } else if actual.is_empty() {
            assert!(actual.is::<FixedSizeList>());
        } else if !tiled.is_full_width()
            && (range.start % tile_rows != 0
                || (range.end % tile_rows != 0 && range.end != tiled.len()))
        {
            assert!(actual.is::<Slice>());
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
    _tile_geometry: TileGeometry,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    for indices in conformance_take_indices(canonical.len())? {
        let expected = canonical.clone().into_array().take(indices.clone())?;
        let actual = tiled.clone().into_array().take(indices)?;
        let actual = if !canonical.is_empty() && !actual.is_empty() {
            actual.execute_until::<FixedSizeList>(ctx)?
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
fn take_conformance() {
    let (_, tiled, mut ctx) = fixture(65, 129, geometry(32, 64)).unwrap();
    test_take_conformance(&tiled.into_array(), &mut ctx);
}

#[test]
fn arbitrary_take_does_not_force_tiled_preservation() {
    assert!(
        !SESSION
            .kernels()
            .has_execute_parent(Dict.id(), TiledFixedSizeList.id())
    );
}

#[test]
fn take_falls_back_and_preserves_order_duplicates_and_nulls() -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(65, 129, geometry(32, 64))?;
    let indices = PrimitiveArray::new(
        buffer![64u32, 1, 1, 32, 0],
        Validity::from_iter([true, true, false, true, true]),
    )
    .into_array();
    let actual = tiled
        .into_array()
        .take(indices.clone())?
        .execute_until::<FixedSizeList>(&mut ctx)?;
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
        .execute_until::<FixedSizeList>(&mut ctx)?;
    assert_eq!(
        actual.as_::<FixedSizeList>().elements().len(),
        index_count * 5
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
fn take_fallback_preserves_outer_and_element_validity() -> VortexResult<()> {
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
        .execute_until::<FixedSizeList>(&mut ctx)?;
    assert_eq!(actual.as_::<FixedSizeList>().elements().len(), 8);
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
        .execute_until::<FixedSizeList>(&mut ctx)?;
    assert_eq!(actual.as_::<FixedSizeList>().elements().len(), 0);
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}

#[test]
fn nullable_take_makes_outer_dtype_nullable() -> VortexResult<()> {
    let (_, tiled, mut ctx) = fixture(3, 5, geometry(2, 3))?;
    let indices = PrimitiveArray::from_option_iter([Some(2u32), None]).into_array();
    let actual = tiled
        .into_array()
        .take(indices)?
        .execute_until::<FixedSizeList>(&mut ctx)?;
    assert!(actual.dtype().is_nullable());
    assert_eq!(actual.as_::<FixedSizeList>().elements().len(), 10);
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
fn slice_chooses_tiled_only_for_aligned_multi_slab_ranges(
    #[case] range: Range<usize>,
) -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(65, 129, geometry(32, 64))?;
    let expected = canonical.into_array().slice(range.clone())?;
    let actual = tiled.into_array().slice(range.clone())?;
    if actual.is_empty() {
        assert!(actual.is::<FixedSizeList>());
    } else if range.start % 32 == 0 && (range.end % 32 == 0 || range.end == 65) {
        assert!(actual.is::<TiledFixedSizeList>());
        assert_eq!(
            actual.as_::<TiledFixedSizeList>().geometry(),
            geometry(32, 64)
        );
    } else {
        assert!(actual.is::<Slice>());
    }
    assert_arrays_eq!(expected, actual, &mut ctx);
    Ok(())
}

#[test]
fn slice_uses_compact_piecewise_indices() -> VortexResult<()> {
    let (_, tiled, _) = fixture(65, 129, geometry(32, 64))?;
    let sliced = tiled.into_array().slice(0..64)?;
    let sliced_tiled = sliced.as_::<TiledFixedSizeList>();
    let elements = sliced_tiled.elements();
    let dict = elements.as_::<Dict>();

    assert!(dict.codes().is::<PiecewiseSequence>());
    let codes = dict.codes().as_::<PiecewiseSequence>();
    assert!(
        codes.starts().len() <= 3 * 129,
        "slice indices must scale with physical runs rather than scalar count"
    );
    Ok(())
}

#[test]
fn slice_preserves_special_cases() -> VortexResult<()> {
    let cases = [
        (3, 0, geometry(2, 3), 1..3, 0),
        (3, 5, geometry(32, 64), 1..3, 15),
        (4_096, 128, geometry(32, 64), 2_048..2_049, 128),
    ];
    for (rows, dimensions, tile_geometry, range, expected_elements) in cases {
        let (canonical, tiled, mut ctx) = fixture(rows, dimensions, tile_geometry)?;
        let expected = canonical.into_array().slice(range.clone())?;
        let actual = tiled.into_array().slice(range.clone())?;
        assert_fsl_equivalent(&expected, &actual, &mut ctx)?;
        if rows == 4_096 {
            assert!(actual.is::<Slice>());
        } else if !actual.is_empty() {
            let actual = actual.as_::<TiledFixedSizeList>();
            assert_eq!(actual.geometry(), tile_geometry);
            assert_eq!(actual.elements().len(), expected_elements);
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
fn aligned_multi_slab_slice_stays_tiled() -> VortexResult<()> {
    let (_, tiled, _) = fixture(256, 1_536, geometry(64, 64))?;

    let aligned = tiled.into_array().slice(64..192)?;

    assert!(aligned.is::<TiledFixedSizeList>());
    let aligned_tiled = aligned.as_::<TiledFixedSizeList>();
    let elements = aligned_tiled.elements();
    assert!(elements.is::<Dict>());
    assert!(elements.as_::<Dict>().codes().is::<PiecewiseSequence>());
    Ok(())
}

#[test]
fn unaligned_multi_slab_slice_stays_lazy_until_execution() -> VortexResult<()> {
    let million_rows = TiledFixedSizeList::try_new(
        ConstantArray::new(0u8, 1_000_000 * 1_536).into_array(),
        1_536,
        Validity::NonNullable,
        1_000_000,
        geometry(64, 64),
    )?;
    assert!(
        <TiledFixedSizeList as SliceReduce>::slice(million_rows.as_view(), 1..130)?.is_none(),
        "unaligned reduction must make an O(1) decision without scalar-run metadata"
    );

    let (_, tiled, _) = fixture(256, 1_536, geometry(64, 64))?;
    let unaligned = tiled.into_array().slice(1..130)?;
    assert!(unaligned.is::<Slice>());
    Ok(())
}

#[test]
fn multi_slab_slice_kernel_matches_canonical() -> VortexResult<()> {
    let rows = 256;
    let list_size = 1_536;
    let element_count = rows * list_size;
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::new(
            Buffer::from_iter((0..element_count).map(|index| index as u32)),
            Validity::from_iter((0..element_count).map(|index| index % 11 != 0)),
        )
        .into_array(),
        list_size as u32,
        Validity::from_iter((0..rows).map(|row| row % 7 != 0)),
        rows,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry(64, 64), &mut ctx)?;
    let expected = canonical.into_array().slice(1..130)?;

    let unaligned = tiled.into_array().slice(1..130)?;
    assert!(unaligned.is::<Slice>());
    let executed = unaligned.execute::<FixedSizeListArray>(&mut ctx)?;

    assert_eq!(executed.len(), 129);
    assert_eq!(executed.elements().len(), 129 * 1_536);
    assert_arrays_eq!(expected, executed, &mut ctx);
    Ok(())
}

#[test]
fn multi_slab_window_has_one_run_per_slab() -> VortexResult<()> {
    let (_, tiled, _) = fixture(256, 1_536, geometry(64, 64))?;

    let retained = tiled.into_array().slice(0..192)?;
    let retained_tiled = retained.as_::<TiledFixedSizeList>();
    let elements = retained_tiled.elements();
    let dict = elements.as_::<Dict>();
    let codes = dict.codes().as_::<PiecewiseSequence>();

    assert_eq!(codes.len(), 192 * 1_536);
    assert_eq!(codes.starts().len(), 24);
    assert_eq!(codes.lengths().len(), 24);
    Ok(())
}

#[test]
fn physical_slab_span_plan_bounds_partial_dimension_and_row_tails() -> VortexResult<()> {
    let (_, tiled, _) = fixture(130, 130, geometry(64, 64))?;

    let spans = crate::gather::plan_physical_row_tile_spans(tiled.as_view(), 64..130)?;

    assert_eq!(spans, vec![4_096..8_320, 12_416..16_640, 16_768..16_900]);
    Ok(())
}

#[test]
fn full_width_unaligned_slice_is_offset_view() -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(200, 128, geometry(64, 128))?;
    let expected = canonical.into_array().slice(10..150)?;
    let actual = tiled.into_array().slice(10..150)?;
    let sliced = actual.as_::<TiledFixedSizeList>();

    assert_eq!(sliced.row_offset(), 10);
    assert_eq!(sliced.len(), 140);
    assert_eq!(sliced.backing_rows(), 192);
    assert_eq!(sliced.elements().len(), 192 * 128);
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}

#[test]
fn nested_full_width_slices_rebase_and_trim() -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(200, 128, geometry(64, 128))?;
    let expected = canonical.into_array().slice(70..80)?;
    let actual = tiled.into_array().slice(10..150)?.slice(60..70)?;
    let sliced = actual.as_::<TiledFixedSizeList>();

    assert_eq!(sliced.row_offset(), 6);
    assert_eq!(sliced.len(), 10);
    assert_eq!(sliced.backing_rows(), 64);
    assert_eq!(sliced.elements().len(), 64 * 128);
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}

#[test]
fn full_width_slice_preserves_nullable_bitpacked_child() -> VortexResult<()> {
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::new(
            Buffer::from_iter(
                (0..200 * 128).map(|index| u16::try_from(index % 16).unwrap_or_default()),
            ),
            Validity::from_iter((0..200 * 128).map(|index| index % 11 != 0)),
        )
        .into_array(),
        128,
        Validity::from_iter((0..200).map(|row| row % 7 != 0)),
        200,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let raw_tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry(64, 128), &mut ctx)?;
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
        200,
        geometry(64, 128),
    )?;

    let expected = canonical.into_array().slice(10..150)?;
    let actual = tiled.into_array().slice(10..150)?;
    let sliced = actual.as_::<TiledFixedSizeList>();

    assert!(sliced.elements().is::<BitPacked>());
    assert!(sliced.elements().validity()?.mask_eq(
        &raw_tiled.elements().validity()?.slice(0..192 * 128)?,
        192 * 128,
        &mut ctx,
    )?);
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}

#[test]
fn full_width_slice_keeps_complete_interior_tiles() -> VortexResult<()> {
    let (_, tiled, _) = fixture(200, 128, geometry(64, 128))?;
    let actual = tiled.into_array().slice(10..150)?;
    let sliced = actual.as_::<TiledFixedSizeList>();
    let tiles: Vec<TileBounds> = sliced.tiles().collect();

    assert_eq!(tiles[0].row_range, 0..54);
    assert_eq!(tiles[1].row_range, 54..118);
    assert_eq!(tiles[2].row_range, 118..140);
    assert!(!tiles[0].is_full_tile());
    assert!(tiles[1].is_full_tile());
    assert!(!tiles[2].is_full_tile());
    Ok(())
}

#[test]
fn zero_width_nonempty_slice_preserves_tiled() -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(3, 0, geometry(64, 128))?;
    let expected = canonical.into_array().slice(1..3)?;
    let actual = tiled.into_array().slice(1..3)?;
    let sliced = actual.as_::<TiledFixedSizeList>();

    assert_eq!(sliced.row_offset(), 0);
    assert_eq!(sliced.backing_rows(), 2);
    assert_eq!(sliced.elements().len(), 0);
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}

#[test]
fn bitpacked_child_roundtrips_through_row_ops() -> VortexResult<()> {
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
    assert!(sliced.is::<Slice>());
    assert_arrays_eq!(expected_sliced, sliced, &mut ctx);

    let taken = tiled
        .into_array()
        .take(indices)?
        .execute_until::<FixedSizeList>(&mut ctx)?;
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
fn offset_view_tiles_expose_boundary_fragments() -> VortexResult<()> {
    let (_, view, _) = offset_view_fixture()?;
    assert_eq!(view.row_offset(), 10);
    assert_eq!(view.backing_rows(), 192);
    assert!(view.is_full_width());

    let tiles = view.tiles().collect::<Vec<_>>();
    assert_eq!(tiles[0].row_range, 0..54);
    assert_eq!(tiles[0].rows_within_tile, 10..64);
    assert!(!tiles[0].is_full_tile());
    assert_eq!(tiles[1].row_range, 54..118);
    assert_eq!(tiles[1].rows_within_tile, 0..64);
    assert!(tiles[1].is_full_tile());
    assert_eq!(tiles[2].row_range, 118..128);
    assert_eq!(tiles[2].rows_within_tile, 0..10);
    Ok(())
}

#[test]
fn offset_view_has_complete_interior_tiles() -> VortexResult<()> {
    let (_, view, _) = offset_view_fixture_with_backing_rows(138)?;
    let tiles = view.tiles().collect::<Vec<_>>();

    assert_eq!(tiles.len(), 3);
    assert_eq!(tiles[0].physical_range, 0..512);
    assert_eq!(tiles[1].physical_range, 512..1024);
    assert_eq!(tiles[2].physical_range, 1024..1104);
    assert_eq!(view.tile_elements(&tiles[0])?.len(), 512);
    assert_eq!(view.tile_elements(&tiles[1])?.len(), 512);
    assert_eq!(view.tile_elements(&tiles[2])?.len(), 80);
    assert!(!tiles[2].is_full_tile());
    Ok(())
}

#[test]
fn scalar_at_uses_row_offset() -> VortexResult<()> {
    let (oracle, view, mut ctx) = offset_view_fixture()?;

    for row in [0, 53, 54, 127] {
        assert_eq!(
            oracle.execute_scalar(row, &mut ctx)?,
            view.execute_scalar(row, &mut ctx)?,
        );
    }
    Ok(())
}

#[test]
fn malformed_view_metadata_is_rejected() {
    let elements = |rows: usize, dimensions: usize| {
        PrimitiveArray::from_iter(
            (0..rows * dimensions).map(|index| u16::try_from(index).unwrap_or(u16::MAX)),
        )
        .into_array()
    };

    assert!(
        TiledFixedSizeList::try_new_view(
            elements(2, 1),
            1,
            Validity::NonNullable,
            1,
            geometry(64, 1),
            2,
            2,
        )
        .is_err()
    );
    assert!(
        TiledFixedSizeList::try_new_view(
            elements(3, 5),
            5,
            Validity::NonNullable,
            2,
            geometry(64, 3),
            1,
            3,
        )
        .is_err()
    );
    assert!(
        TiledFixedSizeList::try_new_view(
            elements(6, 2),
            2,
            Validity::NonNullable,
            5,
            geometry(64, 2),
            1,
            7,
        )
        .is_err()
    );
    assert!(
        TiledFixedSizeList::try_new_view(
            elements(0, 2),
            2,
            Validity::NonNullable,
            0,
            geometry(64, 2),
            0,
            usize::MAX,
        )
        .is_err()
    );
    assert!(
        TiledFixedSizeList::try_new_view(
            elements(64, 1),
            1,
            Validity::NonNullable,
            0,
            geometry(64, 1),
            64,
            64,
        )
        .is_err()
    );
    assert!(
        TiledFixedSizeList::try_new_view(
            elements(3, 0),
            0,
            Validity::NonNullable,
            2,
            geometry(64, 3),
            1,
            3,
        )
        .is_err()
    );
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
            row_offset: u32::MAX,
            backing_rows: u64::MAX,
        }
        .encode_to_vec(),
    );
}

struct TestArrayChildren(Vec<ArrayRef>);

impl ArrayChildren for TestArrayChildren {
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        let child = self
            .0
            .as_slice()
            .get(index)
            .ok_or_else(|| vortex_err!(InvalidArgument: "missing test child {index}"))?;
        if child.dtype() != dtype || child.len() != len {
            return Err(vortex_err!(InvalidArgument:
                "test child {index} has dtype {} and length {}, expected {dtype} and {len}",
                child.dtype(), child.len()
            ));
        }
        Ok(child.clone())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

fn deserialize_test_metadata(
    metadata: TiledFixedSizeListMetadata,
    len: usize,
    list_size: u32,
    elements: ArrayRef,
) -> VortexResult<()> {
    let dtype = DType::FixedSizeList(
        Arc::new(elements.dtype().clone()),
        list_size,
        Nullability::NonNullable,
    );
    <TiledFixedSizeList as vortex_array::VTable>::deserialize(
        &TiledFixedSizeList,
        &dtype,
        len,
        &metadata.encode_to_vec(),
        &[] as &[BufferHandle],
        &TestArrayChildren(vec![elements]),
        &SESSION,
    )?;
    Ok(())
}

#[test]
fn deserialize_rejects_backing_extent_overflow() {
    let error = deserialize_test_metadata(
        TiledFixedSizeListMetadata {
            tile_rows: 64,
            tile_dimensions: 2,
            row_offset: 0,
            backing_rows: u64::MAX,
        },
        1,
        2,
        PrimitiveArray::from_iter(std::iter::empty::<u16>()).into_array(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("backing"));
}

#[test]
fn deserialize_rejects_window_outside_backing_rows() {
    let error = deserialize_test_metadata(
        TiledFixedSizeListMetadata {
            tile_rows: 64,
            tile_dimensions: 1,
            row_offset: 1,
            backing_rows: 2,
        },
        2,
        1,
        PrimitiveArray::from_iter([10u16, 20]).into_array(),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("row window 1..3 exceeds 2 backing rows")
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
        row_offset: 0,
        backing_rows: 0,
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
