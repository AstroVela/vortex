// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::tests_outside_test_module)]

use std::sync::Arc;

use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::layouts::self_paced::SelfPacedLayoutStrategy;
use vortex_layout::plan::exec::SourcePlan;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

const SELF_PACED_TEST_EDITION: EditionId = EditionId::new("self-paced-test", 2026, 8, 0);

fn session() -> VortexResult<VortexSession> {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: SELF_PACED_TEST_EDITION,
            min_vortex_version: None,
        })
        .map_err(|error| vortex_err!("{error}"))?;
    editions
        .declare_inclusion(EditionInclusion::new(
            "vortex.primitive",
            SELF_PACED_TEST_EDITION,
        ))
        .map_err(|error| vortex_err!("{error}"))?;
    session
        .enable_edition(SELF_PACED_TEST_EDITION)
        .map_err(|error| vortex_err!("{error}"))?;
    Ok(session)
}

fn chunk(values: impl IntoIterator<Item = i64>) -> VortexResult<vortex_array::ArrayRef> {
    StructArray::try_new(
        FieldNames::from(["value"]),
        vec![PrimitiveArray::from_iter(values).into_array()],
        3,
        Validity::NonNullable,
    )
    .map(IntoArray::into_array)
}

#[tokio::test]
async fn restricted_file_roundtrip_preserves_chunked_flat_layout_and_bytes() -> VortexResult<()> {
    let session = session()?;
    let chunks = vec![chunk(0..3)?, chunk(3..6)?];
    let dtype = chunks[0].dtype().clone();
    let array = ChunkedArray::try_new(chunks, dtype)?.into_array();
    let mut serialized = ByteBufferMut::empty();

    session
        .write_options()
        .with_strategy(Arc::new(SelfPacedLayoutStrategy::default()))
        .with_file_statistics(Vec::new())
        .write(&mut serialized, array.to_array_stream())
        .await?;

    let serialized = serialized.freeze();
    let expected_bytes = serialized.to_vec();
    let file = session.open_options().open_buffer(serialized.clone())?;
    assert_eq!(serialized.as_ref(), expected_bytes.as_slice());

    let plan = SourcePlan::try_from_layout(file.footer().layout())?;
    assert_eq!(plan.field_names, ["value"]);
    assert_eq!(plan.row_count, 6);
    assert_eq!(
        plan.chunks
            .iter()
            .map(|chunk| chunk.root_coverage.clone())
            .collect::<Vec<_>>(),
        [0..3, 3..6]
    );
    Ok(())
}

#[tokio::test]
async fn restricted_file_keeps_single_chunk_wrapper() -> VortexResult<()> {
    let session = session()?;
    let chunks = vec![chunk(0..3)?];
    let dtype = chunks[0].dtype().clone();
    let array = ChunkedArray::try_new(chunks, dtype)?.into_array();
    let mut serialized = ByteBufferMut::empty();

    session
        .write_options()
        .with_strategy(Arc::new(SelfPacedLayoutStrategy::default()))
        .with_file_statistics(Vec::new())
        .write(&mut serialized, array.to_array_stream())
        .await?;

    let file = session.open_options().open_buffer(serialized.freeze())?;
    let plan = SourcePlan::try_from_layout(file.footer().layout())?;
    assert_eq!(plan.chunks.len(), 1);
    assert_eq!(plan.chunks[0].root_coverage, 0..3);
    Ok(())
}

#[tokio::test]
async fn restricted_file_rejects_unimplemented_field_types() -> VortexResult<()> {
    let session = session()?;
    let array = StructArray::try_new(
        FieldNames::from(["value"]),
        vec![BoolArray::from_iter([true, false, true]).into_array()],
        3,
        Validity::NonNullable,
    )?
    .into_array();
    let mut serialized = ByteBufferMut::empty();

    let result = session
        .write_options()
        .with_strategy(Arc::new(SelfPacedLayoutStrategy::default()))
        .with_file_statistics(Vec::new())
        .write(&mut serialized, array.to_array_stream())
        .await;

    assert!(result.is_err_and(|error| error.to_string().contains("must be non-nullable i64")));
    Ok(())
}
