// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_buffer::ByteBufferMut;
use vortex_edition::EditionError;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::test_harness::validate_edition;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use super::CORE_2025_05_0;
use super::CORE_2026_07_0;
use super::DEFAULT_CORE_EDITION;
use super::DEFAULT_UNSTABLE_EDITION;
use super::EDITION_DECLARATIONS;
use super::UNSTABLE_2026_06_0;

fn session() -> Result<EditionSession, EditionError> {
    let session = EditionSession::empty();
    for declaration in EDITION_DECLARATIONS {
        session.declare(declaration)?;
    }
    Ok(session)
}

#[test]
fn every_declared_edition_validates() -> Result<(), EditionError> {
    let session = session()?;
    for declaration in EDITION_DECLARATIONS {
        validate_edition(&session, &declaration.edition.id)?;
    }
    Ok(())
}

/// The full encoding set of the newest frozen `core` edition. This set is frozen: the only
/// way it may change is by declaring a *new* edition, so a failure here means a frozen
/// declaration was edited.
#[test]
fn core_2026_07_encoding_set_is_pinned() {
    let session = session().unwrap_or_else(|e| panic!("registering editions: {e}"));
    let encodings = session.encodings_in(&CORE_2026_07_0);
    let ids: Vec<&str> = encodings
        .iter()
        .map(|inclusion| inclusion.encoding_id.as_str())
        .collect();
    assert_eq!(
        ids,
        [
            "fastlanes.bitpacked",
            "fastlanes.for",
            "fastlanes.rle",
            "vortex.alp",
            "vortex.alprd",
            "vortex.bool",
            "vortex.bytebool",
            "vortex.chunked",
            "vortex.constant",
            "vortex.datetimeparts",
            "vortex.decimal",
            "vortex.decimal_byte_parts",
            "vortex.dict",
            "vortex.ext",
            "vortex.fixed_size_list",
            "vortex.fsst",
            "vortex.list",
            "vortex.listview",
            "vortex.masked",
            "vortex.null",
            "vortex.pco",
            "vortex.primitive",
            "vortex.runend",
            "vortex.sequence",
            "vortex.sparse",
            "vortex.struct",
            "vortex.varbin",
            "vortex.varbinview",
            "vortex.variant",
            "vortex.zigzag",
            "vortex.zstd",
        ]
    );
}

#[test]
fn encodings_in_editions_unions_families() {
    let session = session().unwrap_or_else(|e| panic!("registering editions: {e}"));
    let core_only: Vec<_> = session
        .encodings_in(&CORE_2026_07_0)
        .into_iter()
        .map(|inclusion| inclusion.encoding_id)
        .collect();
    let mut both = core_only.clone();
    both.extend(
        session
            .encodings_in(&UNSTABLE_2026_06_0)
            .into_iter()
            .map(|inclusion| inclusion.encoding_id),
    );
    both.sort_unstable();
    both.dedup();

    assert!(both.len() > core_only.len());
    assert!(both.iter().any(|id| id.as_str() == "fastlanes.delta"));
    assert!(both.iter().any(|id| id.as_str() == "vortex.onpair"));
    assert!(core_only.iter().all(|id| both.contains(id)));
}

#[test]
fn earlier_editions_are_subsets() {
    let session = session().unwrap_or_else(|e| panic!("registering editions: {e}"));
    let first = session.encodings_in(&CORE_2025_05_0);
    let latest = session.encodings_in(&CORE_2026_07_0);
    assert!(first.iter().all(|inclusion| {
        latest
            .iter()
            .any(|latest| latest.encoding_id == inclusion.encoding_id)
    }));
    assert!(first.len() < latest.len());
}

#[test]
fn default_session_enables_the_write_editions() {
    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    let enabled = session.enabled_editions().editions();
    assert!(enabled.contains(&DEFAULT_CORE_EDITION));

    #[cfg(feature = "unstable_encodings")]
    assert!(enabled.contains(&DEFAULT_UNSTABLE_EDITION));
    #[cfg(not(feature = "unstable_encodings"))]
    assert!(!enabled.contains(&DEFAULT_UNSTABLE_EDITION));
}

#[test]
fn core_edition_ids_are_registered_array_encodings() {
    use vortex_array::session::ArraySessionExt;

    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    let registry = session.arrays().registry().clone();
    for inclusion in session.editions().encodings_in(&CORE_2026_07_0) {
        assert!(
            registry.find(&inclusion.encoding_id).is_some(),
            "{} is declared in core but not registered as an array encoding",
            inclusion.encoding_id
        );
    }
}

fn baseline_core_session() -> VortexResult<VortexSession> {
    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    session
        .enable_edition(CORE_2025_05_0)
        .map_err(|error| vortex_err!("{error}"))?;
    Ok(session)
}

fn sequential_integers() -> PrimitiveArray {
    PrimitiveArray::from_iter(0..65_536i32)
}

#[tokio::test]
async fn default_writer_filters_compressor_to_enabled_editions() -> VortexResult<()> {
    let session = baseline_core_session()?;
    let mut buffer = ByteBufferMut::empty();

    session
        .write_options()
        .write(
            &mut buffer,
            sequential_integers().into_array().to_array_stream(),
        )
        .await?;

    Ok(())
}

#[tokio::test]
async fn configured_btrblocks_builder_uses_enabled_editions_in_either_order() -> VortexResult<()> {
    let session = baseline_core_session()?;
    let allowed: HashSet<_> = session.enabled_encoding_ids().into_iter().collect();
    let strategies = [
        WriteStrategyBuilder::default()
            .with_btrblocks_builder(BtrBlocksCompressorBuilder::default())
            .with_allow_encodings(allowed.clone())
            .build(),
        WriteStrategyBuilder::default()
            .with_allow_encodings(allowed)
            .with_btrblocks_builder(BtrBlocksCompressorBuilder::default())
            .build(),
    ];

    for strategy in strategies {
        let mut buffer = ByteBufferMut::empty();
        session
            .write_options()
            .with_strategy(strategy)
            .write(
                &mut buffer,
                sequential_integers().into_array().to_array_stream(),
            )
            .await?;
    }

    Ok(())
}

#[tokio::test]
async fn opaque_compressor_cannot_write_outside_enabled_editions() -> VortexResult<()> {
    let session = baseline_core_session()?;
    let allowed = session.enabled_encoding_ids().into_iter().collect();
    let strategy = WriteStrategyBuilder::default()
        .with_compressor(BtrBlocksCompressorBuilder::default().build())
        .with_allow_encodings(allowed)
        .build();
    let mut buffer = ByteBufferMut::empty();

    let result = session
        .write_options()
        .with_strategy(strategy)
        .write(
            &mut buffer,
            sequential_integers().into_array().to_array_stream(),
        )
        .await;
    let error = match result {
        Ok(_) => {
            return Err(vortex_err!(
                "the unrestricted opaque compressor wrote an encoding outside core@2025.05"
            ));
        }
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("normalize forbids encoding (vortex.sequence)"),
        "unexpected error: {message}"
    );

    Ok(())
}
