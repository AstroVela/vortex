// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests for the vortex COPY function's WRITTEN_FILE_STATISTICS support, driven through
//! DuckDB with `COPY … (FORMAT vortex, RETURN_STATS)`.

use num_traits::AsPrimitive;
use tempfile::NamedTempFile;

use crate::duckdb::Connection;
use crate::duckdb::Database;

fn database_connection() -> Connection {
    let db = Database::open_in_memory().unwrap();
    crate::initialize(&db).unwrap();
    db.connect().unwrap()
}

/// `RETURN_STATS` binds only because the vortex copy function now implements
/// `copy_to_get_written_statistics`; running it exercises the whole path through DuckDB
/// (bind, the C++ fill trampoline, and the Rust getters) and returns the six-column
/// WRITTEN_FILE_STATISTICS schema. The nested per-column statistics map is validated against
/// DuckLake in the duckdb-vortex integration tests; here we assert the file-level statistics.
#[test]
fn copy_return_stats_reports_file_statistics() {
    let conn = database_connection();
    let file = NamedTempFile::with_suffix(".vortex").unwrap();
    let path = file.path().to_string_lossy();

    let result = conn
        .query(&format!(
            "COPY (SELECT * FROM (VALUES (1, 'a', 1.5), (2, 'b', NULL)) t(i, s, d)) \
             TO '{path}' (FORMAT vortex, RETURN_STATS)"
        ))
        .unwrap();

    // filename, count, file_size_bytes, footer_size, column_statistics, partition_keys
    assert_eq!(result.column_count(), 6);

    let chunk = result.into_iter().next().unwrap();
    let len = chunk.len().as_();
    // count and file_size_bytes are UBIGINT.
    let count = chunk.get_vector(1).as_slice_with_len::<u64>(len)[0];
    let file_size = chunk.get_vector(2).as_slice_with_len::<u64>(len)[0];

    assert_eq!(count, 2);
    assert!(file_size > 0);
}

/// A table with nested (struct/list) columns must not crash the statistics hook. Vortex reports
/// statistics for top-level fields only, so this asserts the COPY succeeds and returns rather than
/// checking per-leaf statistics.
#[test]
fn copy_return_stats_handles_nested_columns() {
    let conn = database_connection();
    let file = NamedTempFile::with_suffix(".vortex").unwrap();
    let path = file.path().to_string_lossy();

    let result = conn
        .query(&format!(
            "COPY (SELECT {{'a': 1, 'b': 2}} AS st, [1, 2, 3] AS lst, 7 AS i) \
             TO '{path}' (FORMAT vortex, RETURN_STATS)"
        ))
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let count = chunk
        .get_vector(1)
        .as_slice_with_len::<u64>(chunk.len().as_())[0];
    assert_eq!(count, 1);
}

/// Without `RETURN_STATS` the statistics hook is never invoked; a plain vortex COPY must still
/// succeed unchanged.
#[test]
fn copy_without_return_stats_still_works() {
    let conn = database_connection();
    let file = NamedTempFile::with_suffix(".vortex").unwrap();
    let path = file.path().to_string_lossy();

    conn.query(&format!(
        "COPY (SELECT 1 AS i, 'a' AS s) TO '{path}' (FORMAT vortex)"
    ))
    .unwrap();
}
