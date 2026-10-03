// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;

use rstest::rstest;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use super::Fixture;
use super::MAX_RETAINED_HANDLES;
use super::OBSERVED_BACKEND;
use super::Prepared;
use super::build_sql;
use super::literal;
use super::nearest_id;
use super::source;
use crate::index::tests::BACKEND;

fn query(fixture: &Fixture, vector: &str) -> String {
    format!(
        "SELECT \"row\".id FROM vortex_index_search({}, [{vector}::FLOAT, 0::FLOAT], 1, validation_mode := 'snapshot')",
        literal(&fixture.reference)
    )
}

#[rstest]
#[case::source("source")]
#[case::artifact("artifact")]
#[case::manifest("manifest")]
fn test_snapshot_retains_verified_rows_and_index_after_disk_changes(
    #[case] changed: &str,
    #[values(false, true)] removed: bool,
) -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let handle = Prepared::new(&fixture.conn, &query(&fixture, "$1"))?;
    assert_eq!(nearest_id(handle.execute(7.0)?)?, 7);
    let path = match changed {
        "source" => fixture.file.clone(),
        "artifact" => fixture
            .root
            .path()
            .join(&fixture.generation)
            .join("artifacts/vectors.json"),
        _ => fixture
            .root
            .path()
            .join(&fixture.generation)
            .join("manifest.json"),
    };
    let original = fs::read(&path)?;
    if removed {
        fs::remove_file(&path)?;
    } else {
        fs::write(&path, b"corrupt")?;
    }
    assert_eq!(nearest_id(handle.execute(14.0)?)?, 14);
    fixture
        .conn
        .query("CREATE TABLE snapshot_rebind_marker(id INTEGER)")?;
    assert_eq!(nearest_id(handle.execute(21.0)?)?, 21);
    let strict = Prepared::new(&fixture.conn, &fixture.query("$1"))?;
    assert!(strict.execute(7.0).is_err());
    let fresh = Prepared::new(&fixture.conn, &query(&fixture, "$1"))?;
    assert!(fresh.execute(7.0).is_err());
    assert_eq!(fixture.counts()?.opens, 1);
    fs::write(path, original)?;
    assert_eq!(nearest_id(fresh.execute(7.0)?)?, 7);
    assert_eq!(fixture.counts()?.opens, 2);
    drop(handle);
    drop(fresh);
    drop(strict);
    assert_eq!(fixture.counts()?.live, 0);
    assert_eq!(fixture.scratch_count()?, 0);
    Ok(())
}

#[rstest]
fn test_snapshot_first_execution_still_verifies_full_contents(
    #[values(false, true)] sql_owner: bool,
) -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let sql = query(&fixture, "$1");
    let handle = if sql_owner {
        fixture
            .conn
            .query(&format!("PREPARE indexed_snapshot AS {sql}"))?;
        None
    } else {
        Some(Prepared::new(&fixture.conn, &sql)?)
    };
    let execute = |value| match &handle {
        Some(handle) => handle.execute(value),
        None => fixture
            .conn
            .query(&format!("EXECUTE indexed_snapshot({value})")),
    };
    let original = fs::read(&fixture.file)?;
    fs::write(&fixture.file, b"corrupt")?;
    assert!(execute(7.0).is_err());
    assert_eq!(fixture.counts()?.opens, 0);
    assert_eq!(fixture.scratch_count()?, 0);
    fs::write(&fixture.file, original)?;
    assert_eq!(nearest_id(execute(7.0)?)?, 7);
    assert_eq!(nearest_id(execute(14.0)?)?, 14);
    assert_eq!(fixture.counts()?.opens, 1);
    drop(handle);
    if sql_owner {
        fixture.conn.query("DEALLOCATE indexed_snapshot")?;
    }
    assert_eq!(fixture.counts()?.live, 0);
    assert_eq!(fixture.scratch_count()?, 0);
    Ok(())
}

#[rstest]
#[case::external_access("SET enable_external_access=false", "external access")]
#[case::local_filesystem("SET disabled_filesystems='LocalFileSystem'", "disabled")]
fn test_snapshot_does_not_bypass_execution_access_policy(
    #[case] setting: &str,
    #[case] message: &str,
) -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let handle = Prepared::new(&fixture.conn, &query(&fixture, "$1"))?;
    assert_eq!(nearest_id(handle.execute(7.0)?)?, 7);
    fixture.conn.query(setting)?;
    let error = handle
        .execute(14.0)
        .err()
        .ok_or_else(|| vortex_err!("Snapshot bypassed access policy"))?;
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(fixture.counts()?.opens, 1);
    Ok(())
}

#[test]
fn test_snapshot_rejects_reference_replacement_and_fresh_owner_accepts_it() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let handle = Prepared::new(&fixture.conn, &query(&fixture, "$1"))?;
    assert_eq!(nearest_id(handle.execute(7.0)?)?, 7);
    let second = source(
        &fixture.conn,
        fixture.root.path(),
        "second.vortex",
        128,
        256,
    )?;
    let replacement = fixture.root.path().join("replacement.json");
    fixture.conn.query(
        &build_sql(fixture.root.path(), &[second])
            .replace(BACKEND, OBSERVED_BACKEND)
            .replace(&literal(&fixture.reference), &literal(&replacement)),
    )?;
    fs::copy(replacement, &fixture.reference)?;
    let error = handle
        .execute(7.0)
        .err()
        .ok_or_else(|| vortex_err!("Snapshot switched reference identity"))?;
    assert!(error.to_string().contains("reference changed"), "{error}");
    let fresh = Prepared::new(&fixture.conn, &query(&fixture, "$1"))?;
    assert_eq!(nearest_id(fresh.execute(140.0)?)?, 140);
    Ok(())
}

#[test]
fn test_snapshot_budget_failure_is_explicit_and_capacity_is_released() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let sql = query(&fixture, "$1");
    let mut handles = Vec::new();
    for _ in 0..MAX_RETAINED_HANDLES {
        let handle = Prepared::new(&fixture.conn, &sql)?;
        assert_eq!(nearest_id(handle.execute(7.0)?)?, 7);
        handles.push(handle);
    }
    let excess = Prepared::new(&fixture.conn, &sql)?;
    let error = excess
        .execute(7.0)
        .err()
        .ok_or_else(|| vortex_err!("Snapshot silently fell back after budget exhaustion"))?;
    assert!(error.to_string().contains("budget"), "{error}");
    assert_eq!(fixture.counts()?.opens, MAX_RETAINED_HANDLES);
    drop(handles.pop());
    assert_eq!(nearest_id(excess.execute(14.0)?)?, 14);
    assert_eq!(nearest_id(excess.execute(21.0)?)?, 21);
    drop(excess);
    drop(handles);
    assert_eq!(fixture.counts()?.live, 0);
    assert_eq!(fixture.scratch_count()?, 0);
    Ok(())
}

#[test]
fn test_snapshot_does_not_change_strict_scans_in_the_same_owner() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let sql = format!(
        "{} UNION ALL {}",
        query(&fixture, "$1"),
        fixture.query("$1")
    );
    let handle = Prepared::new(&fixture.conn, &sql)?;
    assert_eq!(handle.execute(7.0)?.row_count(), 2);
    assert_eq!(fixture.counts()?.opens, 2);
    let original = fs::read(&fixture.file)?;
    fs::write(&fixture.file, b"corrupt")?;
    assert!(handle.execute(14.0).is_err());
    fs::write(&fixture.file, original)?;
    assert_eq!(handle.execute(21.0)?.row_count(), 2);
    assert_eq!(fixture.counts()?.opens, 2);
    drop(handle);
    assert_eq!(fixture.counts()?.live, 0);
    assert_eq!(fixture.scratch_count()?, 0);
    Ok(())
}

#[test]
fn test_failed_search_keeps_the_successfully_verified_snapshot() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    fixture.conn.query(&format!(
        "PREPARE indexed_snapshot AS SELECT \"row\".id FROM vortex_index_search({}, [$1::FLOAT, 0::FLOAT], 1, backend_options := $2, validation_mode := 'snapshot')",
        literal(&fixture.reference)
    ))?;
    assert!(
        fixture
            .conn
            .query("EXECUTE indexed_snapshot(7, 'unsupported')")
            .is_err()
    );
    assert_eq!(fixture.counts()?.opens, 1);
    assert_eq!(fixture.counts()?.live, 1);
    fs::write(&fixture.file, b"corrupt")?;
    assert_eq!(
        nearest_id(fixture.conn.query("EXECUTE indexed_snapshot(14, '')")?)?,
        14
    );
    assert_eq!(fixture.counts()?.opens, 1);
    fixture.conn.query("DEALLOCATE indexed_snapshot")?;
    assert_eq!(fixture.counts()?.live, 0);
    assert_eq!(fixture.scratch_count()?, 0);
    Ok(())
}

#[test]
fn test_ad_hoc_snapshot_owners_are_not_shared() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    for _ in 0..2 {
        assert_eq!(nearest_id(fixture.conn.query(&query(&fixture, "7"))?)?, 7);
    }
    assert_eq!(fixture.counts()?.opens, 2);
    assert_eq!(fixture.counts()?.live, 0);
    assert_eq!(fixture.scratch_count()?, 0);
    Ok(())
}

#[rstest]
fn test_invalid_validation_modes_are_rejected(
    #[values("NULL", "'unknown'", "'SNAPSHOT'", "'snapshot' || chr(0)")] mode: &str,
) -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let sql = format!(
        "SELECT * FROM vortex_index_search({}, [7::FLOAT, 0::FLOAT], 1, validation_mode := {mode})",
        literal(&fixture.reference)
    );
    assert!(fixture.conn.query(&sql).is_err());
    assert_eq!(fixture.counts()?.opens, 0);
    Ok(())
}
