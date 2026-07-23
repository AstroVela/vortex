// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A session's enabled editions may include encodings the session has no registration for,
//! e.g. an optional encoding crate that is not compiled in (vortex-jni enables the unstable
//! edition for `vortex.parquet.variant` without registering its sibling encodings). Files
//! written by such a session must stay readable.

#![expect(clippy::tests_outside_test_module)]

use futures::TryStreamExt;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::assert_arrays_eq;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

const TEST_EDITION: EditionId = EditionId::new("test", 2026, 7, 0);

fn session_with_unregistered_edition_member() -> VortexResult<VortexSession> {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);

    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: TEST_EDITION,
            min_vortex_version: None,
        })
        .map_err(|error| vortex_err!("{error}"))?;
    for id in session.arrays().registry().ids() {
        editions
            .declare_inclusion(EditionInclusion::new(&id, TEST_EDITION))
            .map_err(|error| vortex_err!("{error}"))?;
    }
    editions
        .declare_inclusion(EditionInclusion::new(&"test.not_registered", TEST_EDITION))
        .map_err(|error| vortex_err!("{error}"))?;
    session
        .enable_edition(TEST_EDITION)
        .map_err(|error| vortex_err!("{error}"))?;
    Ok(session)
}

#[tokio::test]
async fn roundtrip_with_enabled_but_unregistered_encoding() -> VortexResult<()> {
    let session = session_with_unregistered_edition_member()?;

    let array = buffer![1u32, 2, 3, 4].into_array();
    let mut output = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut output, array.clone().to_array_stream())
        .await?;

    let file = session
        .open_options()
        .open_buffer(ByteBuffer::from(output))?;
    let chunks: Vec<_> = file.scan()?.into_stream()?.try_collect().await?;
    let actual = ChunkedArray::from_iter(chunks).into_array();
    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(array, actual, &mut ctx);
    Ok(())
}
