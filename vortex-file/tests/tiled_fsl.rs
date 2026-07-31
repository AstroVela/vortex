#![cfg(feature = "unstable_encodings")]
#![expect(clippy::tests_outside_test_module)]

mod common;

use std::num::NonZeroU32;
use std::sync::Arc;

use common::enable_all_registered_array_encodings;
use futures::StreamExt;
use futures::pin_mut;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::session::LayoutSession;
use vortex_tiled_fsl::TileGeometry;
use vortex_tiled_fsl::TiledFixedSizeList;
use vortex_tiled_fsl::TiledFixedSizeListArrayExt;
use vortex_tiled_fsl::TiledFixedSizeListArraySlotsExt;

const ROWS: usize = 65;
const DIMENSIONS: u32 = 128;

fn geometry(rows: u32, dimensions: u32) -> VortexResult<TileGeometry> {
    let rows = NonZeroU32::new(rows)
        .ok_or_else(|| vortex_err!(InvalidArgument: "tile rows must be nonzero"))?;
    let dimensions = NonZeroU32::new(dimensions)
        .ok_or_else(|| vortex_err!(InvalidArgument: "tile dimensions must be nonzero"))?;
    Ok(TileGeometry::new(rows, dimensions))
}

fn row_distinguishing_input() -> VortexResult<FixedSizeListArray> {
    let values = (0..ROWS)
        .flat_map(|row| {
            (0..DIMENSIONS as usize).map(move |dimension| {
                let value = match dimension {
                    // The first two dimensions retain the complete row number while remaining
                    // 4-bit-safe. This makes a transpose or row-order regression observable.
                    0 => row & 0x0f,
                    1 => (row >> 4) & 0x0f,
                    _ => (row * 17 + dimension * 5) & 0x0f,
                };
                u8::try_from(value).map_err(|error| {
                    vortex_err!(InvalidArgument: "four-bit fixture value is out of range: {error}")
                })
            })
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(FixedSizeListArray::new(
        PrimitiveArray::from_iter(values).into_array(),
        DIMENSIONS,
        Validity::NonNullable,
        ROWS,
    ))
}

#[tokio::test]
async fn unstable_tiled_fixed_size_lists_roundtrip_through_files() -> VortexResult<()> {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);

    let mut ctx = session.create_execution_ctx();
    let canonical = row_distinguishing_input()?;
    let expected_geometry = geometry(32, 64)?;
    let raw = TiledFixedSizeList::encode(canonical.as_view(), expected_geometry, &mut ctx)?;
    let physical = raw.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let bitpacked = bitpack_encode(&physical, 4, None, &mut ctx)?.into_array();
    let bitpacked = TiledFixedSizeList::try_new(
        bitpacked,
        DIMENSIONS,
        raw.array_validity(),
        ROWS,
        expected_geometry,
    )?;
    let input = StructArray::new(
        FieldNames::from(["raw", "bitpacked"]),
        vec![raw.into_array(), bitpacked.into_array()],
        ROWS,
        Validity::NonNullable,
    )
    .into_array();

    let mut bytes = Vec::new();
    session
        .write_options()
        .with_strategy(Arc::new(FlatLayoutStrategy::default()))
        .write(&mut bytes, input.clone().to_array_stream())
        .await?;

    let file = session
        .open_options()
        .open_buffer(ByteBuffer::from(bytes))?;
    let stream = file.scan()?.into_stream()?;
    pin_mut!(stream);
    let chunk = stream
        .next()
        .await
        .ok_or_else(|| vortex_err!(InvalidArgument: "written file has no chunks"))??;
    assert!(stream.next().await.is_none(), "one written chunk");

    let result = chunk.execute::<StructArray>(&mut ctx)?;
    let raw = result.unmasked_field(0).clone();
    let bitpacked = result.unmasked_field(1).clone();

    assert!(raw.is::<TiledFixedSizeList>());
    assert!(bitpacked.is::<TiledFixedSizeList>());
    assert_eq!(
        raw.as_::<TiledFixedSizeList>().geometry(),
        expected_geometry
    );
    assert_eq!(
        bitpacked.as_::<TiledFixedSizeList>().geometry(),
        expected_geometry
    );
    assert!(
        bitpacked
            .as_::<TiledFixedSizeList>()
            .elements()
            .is::<vortex_fastlanes::BitPacked>()
    );
    assert_arrays_eq!(input, result, &mut ctx);
    Ok(())
}
