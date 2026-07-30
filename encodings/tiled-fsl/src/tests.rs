use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_native_ptype;
use vortex_array::test_harness::check_metadata;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

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
