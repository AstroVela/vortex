// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::tests_outside_test_module)]

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_error::VortexResult;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

mod common;

use common::enable_all_registered_array_encodings;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);
    session
});

async fn write_micros(value: i64) -> VortexResult<()> {
    let ext_dtype = Timestamp::new(TimeUnit::Microseconds, NonNullable).erased();
    let storage = PrimitiveArray::from_iter([value]).into_array();
    let ts = ExtensionArray::try_new(ext_dtype, storage)?.into_array();
    let data = StructArray::from_fields(&[("ts", ts)])?.into_array();

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .write(&mut bytes, data.to_array_stream())
        .await?;
    Ok(())
}

#[tokio::test]
async fn write_timestamp_last_day_of_9999_does_not_panic() -> VortexResult<()> {
    // 9999-12-31T00:00:00Z is past jiff's max of 9999-12-30T22:00:00Z
    write_micros(253_402_214_400_000_000).await
}

#[tokio::test]
async fn write_timestamp_in_jiff_range() -> VortexResult<()> {
    // 9999-12-30T22:00:00Z jiff's maximum
    write_micros(253_402_207_200_000_000).await
}
