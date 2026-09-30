// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use tempfile::TempDir;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use super::Prepared;
use super::build_sql;
use super::connection;
use super::literal;
use super::nearest_id;
use super::source;
use crate::duckdb::Connection;

struct Fixture {
    _root: TempDir,
    conn: Connection,
    reference: PathBuf,
    replacement: PathBuf,
}

fn fixture() -> VortexResult<Fixture> {
    let root = tempfile::tempdir()?;
    let conn = connection()?;
    let first = source(&conn, root.path(), "a.vortex", 0, 128)?;
    let second = source(&conn, root.path(), "b.vortex", 128, 256)?;
    let reference = root.path().join("index.json");
    let replacement = root.path().join("replacement.json");
    conn.query(&build_sql(root.path(), &[first]))?;
    conn.query(
        &build_sql(root.path(), &[second]).replace(&literal(&reference), &literal(&replacement)),
    )?;
    Ok(Fixture {
        _root: root,
        conn,
        reference,
        replacement,
    })
}

#[test]
fn test_empty_source_search() -> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let conn = connection()?;
    let file = source(&conn, root.path(), "empty.vortex", 0, 0)?;
    conn.query(&build_sql(root.path(), &[file]))?;
    let result = conn.query(&format!(
        "SELECT * FROM vortex_index_search({}, [7::FLOAT, 0::FLOAT], 1)",
        literal(&root.path().join("index.json"))
    ))?;
    assert_eq!(result.row_count(), 0);
    Ok(())
}

#[rstest::rstest]
#[case::execute("EXECUTE indexed(7)")]
#[case::explain_execute("EXPLAIN EXECUTE indexed(7)")]
fn test_prepare_execute_does_not_contaminate_new_query(
    #[case] sql: &str,
    #[values(false, true)] failed_prepare: bool,
    #[values(false, true)] new_prepared: bool,
) -> VortexResult<()> {
    let fixture = fixture()?;
    let conn = &fixture.conn;
    let query = format!(
        "SELECT \"row\".id FROM vortex_index_search({}, [7::FLOAT, 0::FLOAT], 1)",
        literal(&fixture.reference)
    );
    conn.query(&format!(
        "PREPARE indexed AS {}",
        query.replace("7::FLOAT", "$1::FLOAT")
    ))?;
    assert_eq!(nearest_id(conn.query("EXECUTE indexed(7)")?)?, 7);
    if failed_prepare {
        std::fs::copy(&fixture.replacement, &fixture.reference)?;
        let error = Prepared::new(conn, sql)
            .err()
            .ok_or_else(|| vortex_err!("Changed SQL owner was accepted during prepare"))?;
        assert!(error.to_string().contains("reference changed"), "{error}");
    } else {
        drop(Prepared::new(conn, sql)?);
        std::fs::copy(&fixture.replacement, &fixture.reference)?;
    }
    let result = if new_prepared {
        Prepared::new(conn, &format!("{query} WHERE rank >= $1"))?.execute(1.0)?
    } else {
        conn.query(&query)?
    };
    assert_eq!(nearest_id(result)?, 128);
    Ok(())
}

#[rstest::rstest]
#[case::execute("EXECUTE indexed(7)")]
#[case::explain_execute("EXPLAIN EXECUTE indexed(7)")]
fn test_prepare_execute_preserves_independent_first_bind(#[case] sql: &str) -> VortexResult<()> {
    let fixture = fixture()?;
    let conn = &fixture.conn;
    let query = format!(
        "SELECT \"row\".id FROM vortex_index_search({}, [7::FLOAT, 0::FLOAT], 1)",
        literal(&fixture.reference)
    );
    conn.query(&format!(
        "PREPARE indexed AS {}",
        query.replace("7::FLOAT", "$1::FLOAT")
    ))?;
    assert_eq!(nearest_id(conn.query("EXECUTE indexed(7)")?)?, 7);
    drop(Prepared::new(conn, sql)?);
    let fresh = Prepared::new(conn, &format!("{query} WHERE rank >= $1"))?;
    conn.query("SELECT 1")?;
    std::fs::copy(&fixture.replacement, &fixture.reference)?;
    match fresh.execute(1.0) {
        Ok(result) => vortex_bail!(
            "Independent handle lost its first-bind identity and returned id={}",
            nearest_id(result)?
        ),
        Err(error) => assert!(error.to_string().contains("reference changed"), "{error}"),
    }
    Ok(())
}

#[rstest::rstest]
#[case::select("SELECT nonexistent FROM vortex_index_search({reference}, [7::FLOAT, 0::FLOAT], 1)")]
#[case::cte(
    "WITH hits AS (SELECT * FROM vortex_index_search({reference}, [7::FLOAT, 0::FLOAT], 1)) SELECT nonexistent FROM hits"
)]
fn test_failed_prepare_does_not_contaminate_new_query(#[case] sql: &str) -> VortexResult<()> {
    let fixture = fixture()?;
    let conn = &fixture.conn;
    let error = Prepared::new(
        conn,
        &sql.replace("{reference}", &literal(&fixture.reference)),
    )
    .err()
    .ok_or_else(|| vortex_err!("Invalid projection was accepted"))?;
    assert!(error.to_string().contains("nonexistent"), "{error}");
    std::fs::copy(&fixture.replacement, &fixture.reference)?;
    let query = format!(
        "SELECT \"row\".id FROM vortex_index_search({}, [7::FLOAT, 0::FLOAT], 1) WHERE rank >= $1",
        literal(&fixture.reference)
    );
    assert_eq!(nearest_id(Prepared::new(conn, &query)?.execute(1.0)?)?, 128);
    Ok(())
}

#[rstest::rstest]
#[case::execute("EXECUTE indexed{arguments}", false)]
#[case::explain_execute("EXPLAIN EXECUTE indexed{arguments}", true)]
fn test_execute_wrapper_keeps_its_own_identity(
    #[case] sql: &str,
    #[case] explain: bool,
    #[values(false, true)] parameterized: bool,
) -> VortexResult<()> {
    let fixture = fixture()?;
    let conn = &fixture.conn;
    conn.query("CREATE TABLE rebind_marker AS SELECT 1 AS marker")?;
    let prepare = format!(
        "PREPARE indexed AS SELECT \"row\".id FROM vortex_index_search({}, [{}::FLOAT, 0::FLOAT], 1) CROSS JOIN rebind_marker",
        literal(&fixture.reference),
        if parameterized { "$1" } else { "7" }
    );
    let sql = sql.replace("{arguments}", if parameterized { "(7)" } else { "" });
    conn.query(&prepare)?;
    let wrapper = Prepared::new(conn, &sql)?;
    let result = wrapper.run()?;
    if explain {
        assert_eq!(result.row_count(), 1);
    } else {
        assert_eq!(nearest_id(result)?, 7);
    }
    let original = std::fs::read(&fixture.reference)?;
    std::fs::copy(&fixture.replacement, &fixture.reference)?;
    // The SQL name now denotes a new handle. The outer C API handle must keep
    // its own first-bind identity instead of borrowing only the new SQL owner.
    conn.query("DEALLOCATE indexed")?;
    conn.query(&prepare)?;
    conn.query("CREATE TABLE trigger_rebind(id INTEGER)")?;
    let error = wrapper
        .run()
        .err()
        .ok_or_else(|| vortex_err!("EXECUTE wrapper silently adopted a new SQL owner"))?;
    assert!(error.to_string().contains("reference changed"), "{error}");
    let fresh = Prepared::new(conn, &sql)?;
    let result = fresh.run()?;
    if explain {
        assert_eq!(result.row_count(), 1);
    } else {
        assert_eq!(nearest_id(result)?, 128);
    }
    std::fs::write(&fixture.reference, original)?;
    conn.query("DEALLOCATE indexed")?;
    conn.query(&prepare)?;
    conn.query("DROP TABLE trigger_rebind")?;
    let result = wrapper.run()?;
    if explain {
        assert_eq!(result.row_count(), 1);
    } else {
        assert_eq!(nearest_id(result)?, 7);
    }
    let error = fresh
        .run()
        .err()
        .ok_or_else(|| vortex_err!("Fresh EXECUTE wrapper lost its own reference identity"))?;
    assert!(error.to_string().contains("reference changed"), "{error}");
    Ok(())
}

#[rstest::rstest]
#[case::execute("EXECUTE indexed(7)", false)]
#[case::explain_execute("EXPLAIN EXECUTE indexed(7)", true)]
fn test_failed_rebind_does_not_pin_rejected_identity(
    #[case] sql: &str,
    #[case] explain: bool,
) -> VortexResult<()> {
    let fixture = fixture()?;
    let conn = &fixture.conn;
    conn.query("CREATE TABLE rebind_marker AS SELECT 1 AS marker")?;
    conn.query(&format!(
        "PREPARE indexed AS SELECT \"row\".id FROM vortex_index_search({}, [$1::FLOAT, 0::FLOAT], 1) CROSS JOIN rebind_marker",
        literal(&fixture.reference)
    ))?;
    assert_eq!(nearest_id(conn.query("EXECUTE indexed(7)")?)?, 7);
    let wrapper = Prepared::new(conn, sql)?;
    let original = std::fs::read(&fixture.reference)?;
    std::fs::copy(&fixture.replacement, &fixture.reference)?;
    conn.query("CREATE TABLE trigger_rebind(id INTEGER)")?;
    let error = wrapper
        .run()
        .err()
        .ok_or_else(|| vortex_err!("Changed inner SQL owner was accepted"))?;
    assert!(error.to_string().contains("reference changed"), "{error}");
    std::fs::write(&fixture.reference, original)?;
    let result = wrapper.run()?;
    if explain {
        assert_eq!(result.row_count(), 1);
    } else {
        assert_eq!(nearest_id(result)?, 7);
    }
    Ok(())
}
