use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::test_harness::check_metadata;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::TileGeometry;
use crate::TiledFixedSizeList;
use crate::TiledFixedSizeListArray;
use crate::TiledFixedSizeListArrayExt;
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
        PrimitiveArray::from_iter((0..rows * dimensions as usize).map(|index| index as u16))
            .into_array(),
        dimensions,
        Validity::NonNullable,
        rows,
        geometry,
    )
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
    let elements =
        PrimitiveArray::from_iter((0..physical_len).map(|index| index as u16)).into_array();
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
