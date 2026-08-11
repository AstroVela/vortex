// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Golden-file serialization tests for every object reachable from an edition.
//!
//! Every edition member — array encodings, layouts, aggregate functions, and extension
//! dtypes, `core` and `unstable` alike — has a directory of golden files under
//! `vortex/goldenfiles/editions/<kind>/<id>/`. Each file `vNNN.bin` pins one historical
//! serialized form of that object:
//!
//! - the **newest** golden must be byte-identical to what the current code serializes, and
//! - **every** golden, however old, must still deserialize to the fixture's logical value.
//!
//! Together these enforce the editions evolution policy (`docs/specs/editions.md`): the
//! serialized form may only change by *adding* a new version, and every version that has
//! ever existed stays readable forever. See `goldenfiles/editions/AGENTS.md` for the rules
//! on adding golden files; in short, run the suite with `UPDATE_GOLDENFILES=1` to add a new
//! version and never edit or delete an existing one.

mod arrays;
mod kinds;

use std::fs;
use std::path::PathBuf;

use vortex_edition::ObjectKind;
use vortex_error::VortexResult;
use vortex_session::registry::Id;

use super::EDITION_DECLARATIONS;

/// Environment variable that permits *adding* a new golden version. Existing files are
/// never modified or removed, not even when it is set.
const UPDATE_ENV: &str = "UPDATE_GOLDENFILES";

/// Objects that are edition members but have no golden fixture, with the reason why.
///
/// Every entry here is a hole in the suite's coverage; add entries only when the object
/// genuinely cannot be produced by current writers, and say where its serialized form is
/// pinned instead.
fn exemptions() -> Vec<(ObjectKind, &'static str, &'static str)> {
    vec![(
        ObjectKind::Layout,
        "vortex.stats",
        "legacy read-only layout: writers no longer produce it; its serialized form is \
         pinned by the LegacyStats deserialization tests in vortex-layout",
    )]
}

fn goldens_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldenfiles/editions")
}

fn kind_dir_name(kind: ObjectKind) -> &'static str {
    match kind {
        ObjectKind::Array => "arrays",
        ObjectKind::Layout => "layouts",
        ObjectKind::Aggregation => "aggregations",
        ObjectKind::ExtensionDType => "extension_dtypes",
    }
}

fn golden_dir(kind: ObjectKind, id: &str) -> PathBuf {
    goldens_root().join(kind_dir_name(kind)).join(id)
}

/// Read all golden versions for an object, sorted by version number.
fn read_versions(dir: &PathBuf) -> Vec<(u32, PathBuf, Vec<u8>)> {
    let Ok(entries) = fs::read_dir(dir) else {
        return vec![];
    };
    let mut versions: Vec<(u32, PathBuf, Vec<u8>)> = entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            let version: u32 = name.strip_prefix('v')?.split('.').next()?.parse().ok()?;
            let bytes = fs::read(&path).ok()?;
            Some((version, path, bytes))
        })
        .collect();
    versions.sort_by_key(|(version, ..)| *version);
    versions
}

/// Check one object's golden files against its current serialized form.
///
/// - With `UPDATE_GOLDENFILES=1`, a **new** version file is added when the current bytes
///   differ from the newest golden (or none exists yet). Existing files are never touched.
/// - The newest golden must equal the current bytes.
/// - Every golden version, old and new, must pass `decode`: deserialize and match the
///   fixture's logical value.
fn check_golden(
    kind: ObjectKind,
    id: &str,
    current: &[u8],
    mut decode: impl FnMut(&[u8]) -> VortexResult<()>,
) {
    let dir = golden_dir(kind, id);
    let mut versions = read_versions(&dir);

    let newest_matches = versions
        .last()
        .is_some_and(|(_, _, bytes)| bytes == current);
    if std::env::var(UPDATE_ENV).is_ok() && !newest_matches {
        let next = versions.last().map(|(v, ..)| v + 1).unwrap_or(1);
        fs::create_dir_all(&dir)
            .unwrap_or_else(|e| panic!("creating golden dir {}: {e}", dir.display()));
        let path = dir.join(format!("v{next:03}.bin"));
        fs::write(&path, current)
            .unwrap_or_else(|e| panic!("writing golden {}: {e}", path.display()));
        versions = read_versions(&dir);
    }

    let Some((_, newest_path, newest)) = versions.last() else {
        panic!(
            "{kind} {id} has no golden files under {}. Run this test with {UPDATE_ENV}=1 to \
             add v001.bin, then commit it. See goldenfiles/editions/AGENTS.md.",
            dir.display()
        );
    };
    assert_eq!(
        newest.as_slice(),
        current,
        "{kind} {id}: current serialization no longer matches the newest golden {}.\n\
         If the serialized format changed intentionally (e.g. a new field), run this test \
         with {UPDATE_ENV}=1 to ADD a new golden version. NEVER edit or delete an existing \
         golden: every version that has ever existed must stay readable forever. See \
         goldenfiles/editions/AGENTS.md.",
        newest_path.display(),
    );

    for (version, path, bytes) in &versions {
        decode(bytes).unwrap_or_else(|e| {
            panic!(
                "{kind} {id}: golden v{version:03} ({}) no longer deserializes: {e}\n\
                 Historical serialized forms must stay readable forever; fix the \
                 deserializer instead of changing or removing the golden. See \
                 goldenfiles/editions/AGENTS.md.",
                path.display()
            )
        });
    }
}

/// The `(family, id)` pairs of every declared edition member of `kind`, across all
/// editions of all families, drafts included.
fn declared_members(kind: ObjectKind) -> Vec<(&'static str, Id)> {
    let mut members: Vec<(&'static str, Id)> = EDITION_DECLARATIONS
        .iter()
        .flat_map(|declaration| {
            let added = match kind {
                ObjectKind::Array => declaration.added_arrays,
                ObjectKind::Layout => declaration.added_layouts,
                ObjectKind::Aggregation => declaration.added_aggregations,
                ObjectKind::ExtensionDType => declaration.added_extension_dtypes,
            };
            added
                .iter()
                .map(|object| (declaration.edition.id.family, object.object_id()))
        })
        .collect();
    members.sort();
    members.dedup();
    members
}

/// Members of `kind` the suite must have fixtures for with the current feature set:
/// everything declared, minus `unstable`-family members when the `unstable_encodings`
/// feature is off, minus documented [`exemptions`].
fn required_members(kind: ObjectKind) -> Vec<Id> {
    declared_members(kind)
        .into_iter()
        .filter(|(family, _)| cfg!(feature = "unstable_encodings") || *family != "unstable")
        .map(|(_, id)| id)
        .filter(|id| {
            !exemptions()
                .iter()
                .any(|(k, exempt, _)| *k == kind && *exempt == id.as_str())
        })
        .collect()
}

/// Assert that `fixture_ids` covers every required member of `kind` and contains no ids
/// outside the declarations.
fn assert_fixture_completeness(kind: ObjectKind, fixture_ids: &[&str]) {
    let declared = declared_members(kind);
    for id in required_members(kind) {
        assert!(
            fixture_ids.contains(&id.as_str()),
            "{kind} {id} is an edition member but has no golden fixture; add one to the \
             golden suite (see goldenfiles/editions/AGENTS.md) or record a documented \
             exemption"
        );
    }
    for id in fixture_ids {
        assert!(
            declared.iter().any(|(_, member)| member.as_str() == *id),
            "{kind} {id} has a golden fixture but is not declared in any edition"
        );
    }
}

/// Every golden directory on disk must correspond to a declared member, so goldens cannot
/// silently outlive a (forbidden) removal or a rename of the object they pin.
#[cfg_attr(miri, ignore)]
#[test]
fn golden_dirs_match_declared_members() {
    for kind in ObjectKind::ALL {
        let declared = declared_members(kind);
        let kind_dir = goldens_root().join(kind_dir_name(kind));
        let Ok(entries) = fs::read_dir(&kind_dir) else {
            continue;
        };
        for entry in entries {
            let entry = entry.unwrap_or_else(|e| panic!("reading {}: {e}", kind_dir.display()));
            let id = entry.file_name();
            let id = id.to_str().expect("golden directory names are utf-8");
            assert!(
                declared.iter().any(|(_, member)| member.as_str() == id),
                "goldenfiles/editions/{}/{id} exists but {id} is not declared in any \
                 edition; edition members are never removed, so goldens must always match \
                 a declaration",
                kind_dir_name(kind),
            );
        }
    }
}
