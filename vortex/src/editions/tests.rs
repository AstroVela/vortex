// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::patched::use_experimental_patches;
use vortex_edition::EditionError;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::test_harness::validate_edition;
use vortex_session::VortexSession;

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

    // Experimental patched arrays make the compressor emit `vortex.patched`, which only the
    // `unstable` family declares, so they enable that edition too.
    let unstable = cfg!(feature = "unstable_encodings") || use_experimental_patches();
    assert_eq!(enabled.contains(&DEFAULT_UNSTABLE_EDITION), unstable);
}

/// The writer's static allow-list is gated by the enabled editions, so a default session must
/// not be able to write an encoding that only an `unstable` edition declares.
#[cfg(all(feature = "files", not(feature = "unstable_encodings")))]
#[test]
fn the_default_writer_cannot_emit_unstable_encodings() {
    use vortex_file::writable_encodings;

    use crate::VortexSessionDefault;

    if use_experimental_patches() {
        // The experimental flag opts the default session into the unstable edition as well.
        return;
    }

    let session = VortexSession::default();
    let writable = writable_encodings(&session);
    assert!(!writable.is_empty());

    let core: Vec<_> = session
        .editions()
        .encodings_in(&DEFAULT_CORE_EDITION)
        .into_iter()
        .map(|inclusion| inclusion.encoding_id)
        .collect();

    for id in &writable {
        assert!(
            core.contains(id),
            "{id} is writable but the enabled core edition does not include it"
        );
    }
    // `fastlanes.delta` is in the writer's static allow-list but only the `unstable` family
    // declares it, so the gate must have removed it.
    assert!(
        writable
            .iter()
            .all(|id| id.as_str() != "fastlanes.delta" && id.as_str() != "vortex.patched")
    );
}

#[test]
fn core_edition_ids_are_registered_array_encodings() {
    use vortex_array::session::ArraySessionExt;

    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    let registry = session.arrays().registry().clone();
    for inclusion in session.editions().encodings_in(&CORE_2026_07_0) {
        assert!(
            registry.contains_key(&inclusion.encoding_id),
            "{} is declared in core but not registered as an array encoding",
            inclusion.encoding_id
        );
    }
}

/// Under the feature set the benchmarks build with, the gate must be a no-op: every encoding in
/// the writer's allow-list is either covered by an enabled edition or declared by no edition at
/// all. If this holds, gating cannot change a single byte the benchmarks write, so any file-size
/// movement they report is pre-existing noise rather than a lost compression scheme.
#[cfg(all(feature = "files", feature = "unstable_encodings"))]
#[test]
fn the_benchmark_feature_set_gates_nothing() {
    use vortex_file::ALLOWED_ENCODINGS;
    use vortex_file::writable_encodings;

    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    assert_eq!(
        writable_encodings(&session),
        *ALLOWED_ENCODINGS,
        "the benchmarks enable both default editions, so nothing may be gated out"
    );
}

/// Without `unstable_encodings`, the default session enables only the `core` family, and
/// `fastlanes.delta` is the sole allow-list entry no core edition declares. Pinning the whole
/// difference documents exactly what a default writer stops emitting — and shows no compression
/// scheme is affected, since the delta scheme is itself compiled out in this configuration.
#[cfg(all(feature = "files", not(feature = "unstable_encodings")))]
#[test]
fn a_default_session_gates_out_only_fastlanes_delta() {
    use vortex_file::ALLOWED_ENCODINGS;
    use vortex_file::writable_encodings;

    use crate::VortexSessionDefault;

    if use_experimental_patches() {
        // The experimental flag opts into the unstable edition, which declares `fastlanes.delta`.
        return;
    }

    let session = VortexSession::default();
    let writable = writable_encodings(&session);

    let mut gated: Vec<&str> = ALLOWED_ENCODINGS
        .iter()
        .filter(|id| !writable.contains(*id))
        .map(|id| id.as_str())
        .collect();
    gated.sort_unstable();
    assert_eq!(gated, ["fastlanes.delta"]);
}

/// A session narrowed to the *baseline* core edition. `CORE_2025_05_0` predates Zstd, Pco,
/// FastLanes RLE, `vortex.masked`, `vortex.listview` and more, so a large part of the compressor
/// has to be filtered out — far more than any shipping configuration gates. Enabling an edition
/// replaces the enabled edition from the same family, so this narrows the default session.
#[cfg(feature = "files")]
fn baseline_core_session() -> Result<VortexSession, EditionError> {
    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    session.enable_edition(CORE_2025_05_0)?;
    Ok(session)
}

/// Every non-canonical encoding in an array tree, including the root. Canonical encodings are
/// always writable, so only the compressed ones need checking against the allow-list.
#[cfg(feature = "files")]
fn compressed_encodings(
    array: &vortex_array::ArrayRef,
    into: &mut Vec<vortex_session::registry::Id>,
) {
    if !array.is_canonical() {
        into.push(array.encoding_id());
    }
    for child in array.children() {
        compressed_encodings(&child, into);
    }
}

/// The compressor must never produce an encoding the writer would then reject.
///
/// This is the property the whole gate rests on: [`retain_allowed_encodings`] filters schemes by
/// their *declared* [`produced_encodings`], so a scheme that under-declares would survive the
/// filter, compress into a gated encoding, and make the writer fail the file rather than fall
/// back. Compressing real data of every canonical kind against a deliberately narrow allow-list
/// checks those declarations against what the schemes actually emit.
///
/// [`retain_allowed_encodings`]: vortex_btrblocks::BtrBlocksCompressorBuilder::retain_allowed_encodings
/// [`produced_encodings`]: vortex_compressor::Scheme::produced_encodings
#[cfg(feature = "files")]
#[test]
fn a_gated_compressor_never_emits_a_gated_encoding() -> Result<(), Box<dyn std::error::Error>> {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;

    use crate::compressor::BtrBlocksCompressorBuilder;

    let session = baseline_core_session()?;
    let allowed = vortex_file::writable_encodings(&session);

    // Sanity: the narrowed edition really does gate a broad set, or this test proves nothing.
    for gated in [
        "vortex.zstd",
        "vortex.pco",
        "fastlanes.rle",
        "vortex.masked",
    ] {
        assert!(
            !allowed.iter().any(|id| id.as_str() == gated),
            "{gated} must be gated out by the baseline edition"
        );
    }

    let arrays: Vec<vortex_array::ArrayRef> = vec![
        // Tightly clustered ints: FoR + BitPacking territory.
        PrimitiveArray::new(
            Buffer::from_iter((0..8192u32).map(|i| 1_000_000 + (i % 16))),
            Validity::NonNullable,
        )
        .into_array(),
        // Long runs: RunEnd and the RLE schemes.
        PrimitiveArray::new(
            Buffer::from_iter((0..8192u64).map(|i| i / 512)),
            Validity::NonNullable,
        )
        .into_array(),
        // Floats: ALP / ALPRD / Pco territory.
        PrimitiveArray::new(
            Buffer::from_iter((0..8192).map(|i| f64::from(i) * 0.125)),
            Validity::NonNullable,
        )
        .into_array(),
        // Low-cardinality strings: dictionary and FSST territory.
        VarBinArray::from_iter(
            (0..8192).map(|i| Some(format!("payload-{}", i % 32))),
            vortex_array::dtype::DType::Utf8(vortex_array::dtype::Nullability::Nullable),
        )
        .into_array(),
        // Highly compressible bytes: what the compact preset would reach for Zstd on.
        VarBinArray::from_iter(
            (0..4096).map(|i| Some(format!("{}{i}", "repetitive-".repeat(16)))),
            vortex_array::dtype::DType::Utf8(vortex_array::dtype::Nullability::Nullable),
        )
        .into_array(),
        BoolArray::from_iter((0..8192).map(|i| i % 7 == 0)).into_array(),
    ];

    // `with_compact` is the widest scheme set the writer ever uses, so it is the strongest case.
    let compressor = BtrBlocksCompressorBuilder::default()
        .with_compact()
        .retain_allowed_encodings(&allowed)
        .build();

    let mut ctx = session.create_execution_ctx();
    for array in arrays {
        let compressed = compressor.compress(&array, &mut ctx)?;

        let mut encodings = Vec::new();
        compressed_encodings(&compressed, &mut encodings);
        for encoding in encodings {
            assert!(
                allowed.contains(&encoding),
                "the gated compressor produced {encoding}, which the writer would reject; \
                 a scheme's produced_encodings() is under-declared"
            );
        }
    }
    Ok(())
}
