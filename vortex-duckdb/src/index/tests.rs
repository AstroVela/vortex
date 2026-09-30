// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::CString;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use serde::Deserialize;
use serde::Serialize;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_index::DistanceMetric;
use vortex_index::FlatIndex;
use vortex_index::Index;
use vortex_index::IndexBuildRequest;
use vortex_index::IndexBuilder;
use vortex_index::IndexMetadata;
use vortex_index::IndexProvider;
use vortex_index::IndexSource;
use vortex_index::IndexStore;
use vortex_index::RowAddress;
use vortex_index::RowFilter;
use vortex_index::SearchHit;
use vortex_index::VectorIndex;
use vortex_index::VectorSearchOptions;
use vortex_index::VectorSpec;

use super::register_index_provider_factory;
use crate::SESSION;
use crate::cpp;
use crate::duckdb::Connection;
use crate::duckdb::Database;
use crate::duckdb::QueryResult;

const BACKEND: &str = "sql.flat.fixture";

#[derive(Serialize, Deserialize)]
struct Vectors {
    rows: Vec<RowAddress>,
    values: Vec<f32>,
}

struct Provider;

fn factory(_session: VortexSession, _scratch: &Path) -> VortexResult<Arc<dyn IndexProvider>> {
    Ok(Arc::new(Provider))
}

#[async_trait]
impl IndexBuilder for Provider {
    async fn build(
        &self,
        request: IndexBuildRequest,
        source: Arc<dyn IndexSource>,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<IndexMetadata> {
        let change_identity = request.backend_options.as_ref() == b"changed-identity";
        if request.backend_options.as_ref() != b"{}" && !change_identity {
            vortex_bail!("Invalid test backend build options");
        }
        let mut metadata = request.metadata;
        let mut stream = source.scan(&metadata.covered_files, &metadata.fields)?;
        let mut ctx = SESSION.create_execution_ctx();
        let mut data = Vectors {
            rows: vec![],
            values: vec![],
        };
        while let Some(batch) = stream.try_next().await? {
            let array = batch.data.execute::<StructArray>(&mut ctx)?;
            let vectors = array
                .unmasked_field(0)
                .clone()
                .execute::<FixedSizeListArray>(&mut ctx)?;
            if vectors.list_size() != 2 {
                vortex_bail!("Expected two-dimensional vectors");
            }
            let values = vectors
                .elements()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?;
            data.rows.extend(batch.rows);
            data.values.extend_from_slice(values.as_slice::<f32>());
        }
        let artifact = store
            .write(
                "vectors.json",
                Bytes::from(serde_json::to_vec(&data).map_err(|error| vortex_err!("{error}"))?),
            )
            .await?;
        metadata.artifacts.push(artifact);
        if change_identity {
            metadata.fields = vec!["other".into()];
        }
        Ok(metadata)
    }
}

#[async_trait]
impl IndexProvider for Provider {
    fn id(&self) -> &str {
        BACKEND
    }
    fn supports_version(&self, version: u32) -> bool {
        version == 1
    }
    fn builder(&self) -> Option<&dyn IndexBuilder> {
        Some(self)
    }
    async fn open(
        &self,
        metadata: &IndexMetadata,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<Arc<dyn Index>> {
        let data: Vectors = serde_json::from_slice(&store.read(&metadata.artifacts[0]).await?)
            .map_err(|error| vortex_err!("{error}"))?;
        let mut flat_metadata = metadata.clone();
        flat_metadata.backend = FlatIndex::ID.into();
        flat_metadata.artifacts.clear();
        let flat = FlatIndex::try_new(
            flat_metadata,
            VectorSpec {
                dimension: std::num::NonZeroUsize::new(2)
                    .ok_or_else(|| vortex_err!("Dimension"))?,
                metric: DistanceMetric::SquaredL2,
            },
            data.rows,
            Buffer::from(data.values),
        )?;
        Ok(Arc::new(Reader {
            metadata: metadata.clone(),
            flat,
        }))
    }
}

#[derive(Debug)]
struct Reader {
    metadata: IndexMetadata,
    flat: FlatIndex,
}

impl Index for Reader {
    fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }
    fn as_vector(&self) -> Option<&dyn VectorIndex> {
        Some(self)
    }
}

#[async_trait]
impl VectorIndex for Reader {
    fn spec(&self) -> VectorSpec {
        self.flat.spec()
    }
    fn supports_exact(&self) -> bool {
        true
    }
    fn supports_filter(&self) -> bool {
        true
    }
    async fn search(
        &self,
        query: &[f32],
        options: &VectorSearchOptions,
        filter: &RowFilter,
    ) -> VortexResult<Vec<SearchHit>> {
        self.flat.search(query, options, filter).await
    }
}

fn connection() -> VortexResult<Connection> {
    register_index_provider_factory(BACKEND, factory)?;
    let database = Database::open_in_memory()?;
    crate::initialize(&database)?;
    database.connect()
}

fn literal(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "''"))
}

fn source(
    connection: &Connection,
    root: &Path,
    name: &str,
    begin: u64,
    end: u64,
) -> VortexResult<std::path::PathBuf> {
    let path = root.join(name);
    connection.query(&format!("COPY (SELECT i::UBIGINT AS id, [i::FLOAT, (i % 7)::FLOAT]::FLOAT[2] AS embedding, 'row-' || i AS label FROM range({begin}, {end}) AS t(i)) TO {} (FORMAT vortex)", literal(&path)))?;
    Ok(path)
}

fn build_sql(root: &Path, files: &[std::path::PathBuf]) -> String {
    format!(
        "SELECT * FROM vortex_index_build([{}], {}, 'embedding', '{BACKEND}', '{{}}')",
        files
            .iter()
            .map(|file| literal(file))
            .collect::<Vec<_>>()
            .join(","),
        literal(&root.join("index.json"))
    )
}

struct Prepared<'a> {
    statement: cpp::duckdb_prepared_statement,
    _connection: &'a Connection,
}

impl<'a> Prepared<'a> {
    fn new(connection: &'a Connection, sql: &str) -> VortexResult<Self> {
        let sql = CString::new(sql).map_err(|error| vortex_err!("{error}"))?;
        let mut prepared = Self {
            statement: std::ptr::null_mut(),
            _connection: connection,
        };
        // The live connection outlives this owned C API handle, including errors.
        let status = unsafe {
            cpp::duckdb_prepare(
                connection.as_ptr(),
                sql.as_ptr(),
                &raw mut prepared.statement,
            )
        };
        if status != cpp::duckdb_state::DuckDBSuccess {
            let error = unsafe { cpp::duckdb_prepare_error(prepared.statement) };
            if error.is_null() {
                vortex_bail!("Preparing index query failed");
            }
            vortex_bail!("{}", unsafe { CStr::from_ptr(error) }.to_string_lossy());
        }
        Ok(prepared)
    }

    fn execute(&self, value: f64) -> VortexResult<QueryResult> {
        let status = unsafe { cpp::duckdb_bind_double(self.statement, 1, value) };
        if status != cpp::duckdb_state::DuckDBSuccess {
            vortex_bail!("Binding index query parameter failed");
        }
        let mut result: cpp::duckdb_result = unsafe { std::mem::zeroed() };
        let status = unsafe { cpp::duckdb_execute_prepared(self.statement, &raw mut result) };
        // QueryResult owns cleanup on both successful and failed execution.
        let result = unsafe { QueryResult::new(result) };
        if status != cpp::duckdb_state::DuckDBSuccess {
            let error = unsafe { cpp::duckdb_result_error(result.as_ptr()) };
            if error.is_null() {
                vortex_bail!("Executing prepared index query failed");
            }
            vortex_bail!("{}", unsafe { CStr::from_ptr(error) }.to_string_lossy());
        }
        Ok(result)
    }
}

impl Drop for Prepared<'_> {
    fn drop(&mut self) {
        unsafe { cpp::duckdb_destroy_prepare(&raw mut self.statement) };
    }
}

#[test]
fn test_sql_build_reopen_ranked_take_and_no_bind_side_effects() -> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let conn = connection()?;
    let files = [
        source(&conn, root.path(), "a.vortex", 0, 128)?,
        source(&conn, root.path(), "b.vortex", 128, 256)?,
    ];
    let build = build_sql(root.path(), &files);
    conn.query(&format!("EXPLAIN {build}"))?;
    assert!(!root.path().join("index.json").exists());
    let built = conn.query(&build)?;
    assert_eq!(
        built
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("Build result"))?
            .len(),
        1
    );
    assert!(conn.query(&build).is_err());
    drop(conn);
    let conn = connection()?;
    let query = format!(
        "SELECT rank, file_id, row_offset, distance, \"row\".id, \"row\".label FROM vortex_index_search({}, [133::FLOAT, 0::FLOAT], 10) ORDER BY rank",
        literal(&root.path().join("index.json"))
    );
    let mut chunk = conn
        .query(&query)?
        .into_iter()
        .next()
        .ok_or_else(|| vortex_err!("Search result"))?;
    assert_eq!(chunk.len(), 10);
    // The expected nearest vector lives in the second source file, whose
    // physical row offset is independent of its application-level id.
    assert_eq!(
        unsafe { chunk.get_vector_mut(0).as_slice_mut::<u64>(10) }[0],
        1
    );
    assert_eq!(
        unsafe { chunk.get_vector_mut(1).as_slice_mut::<u64>(10) }[0],
        2
    );
    assert_eq!(
        unsafe { chunk.get_vector_mut(2).as_slice_mut::<u64>(10) }[0],
        5
    );
    assert_eq!(
        unsafe { chunk.get_vector_mut(3).as_slice_mut::<f32>(10) }[0],
        0.0
    );
    assert_eq!(
        unsafe { chunk.get_vector_mut(4).as_slice_mut::<u64>(10) }[0],
        133
    );
    Ok(())
}

#[test]
fn test_sql_rejects_invalid_queries_stale_sources_and_changed_prepared_reference()
-> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let conn = connection()?;
    let file = source(&conn, root.path(), "a.vortex", 0, 128)?;
    conn.query(&build_sql(root.path(), std::slice::from_ref(&file)))?;
    let path = root.path().join("index.json");
    for arguments in [
        "NULL, 10",
        "[NULL::FLOAT, 0::FLOAT], 10",
        "[1::FLOAT], 10",
        "[1::FLOAT, 0::FLOAT], 0",
        "[1::FLOAT, 0::FLOAT], 10001",
        "[1::FLOAT, 0::FLOAT], 10, backend_options := '{\"unknown\":1}'",
    ] {
        assert!(
            conn.query(&format!(
                "SELECT * FROM vortex_index_search({}, {arguments})",
                literal(&path)
            ))
            .is_err(),
            "{arguments}"
        );
    }
    conn.query(&format!(
        "PREPARE indexed AS SELECT * FROM vortex_index_search({}, [1::FLOAT, 0::FLOAT], 2)",
        literal(&path)
    ))?;
    let mut bytes = std::fs::read(&path)?;
    bytes.push(b' ');
    std::fs::write(&path, bytes)?;
    assert!(conn.query("EXECUTE indexed").is_err());
    let query = format!(
        "SELECT * FROM vortex_index_search({}, [1::FLOAT, 0::FLOAT], 2)",
        literal(&path)
    );
    conn.query(&query)?;
    std::fs::remove_file(file)?;
    assert!(conn.query(&query).is_err());
    Ok(())
}

#[test]
fn test_sql_null_vectors_fail_without_publication_and_external_access_is_checked()
-> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let conn = connection()?;
    let path = root.path().join("null.vortex");
    conn.query(&format!("COPY (SELECT [NULL::FLOAT, 0::FLOAT]::FLOAT[2] AS embedding FROM range(128)) TO {} (FORMAT vortex)", literal(&path)))?;
    let build = build_sql(root.path(), &[path]);
    assert!(conn.query(&build).is_err());
    assert!(!root.path().join("index.json").exists());
    assert!(
        std::fs::read_dir(root.path())?.all(|entry| entry.is_ok_and(|entry| !entry
            .file_name()
            .to_string_lossy()
            .starts_with(".index-scratch-")))
    );
    conn.query("SET enable_external_access = false")?;
    assert!(conn.query(&build).is_err());
    assert!(
        conn.query(&format!(
            "SELECT * FROM vortex_index_search({}, [1::FLOAT, 0::FLOAT], 2)",
            literal(&root.path().join("missing.json"))
        ))
        .is_err()
    );
    Ok(())
}

#[test]
fn test_sql_invalid_vector_types_return_errors_without_aborting() -> VortexResult<()> {
    const CHILD_TYPE: &str = "VORTEX_SQL_INDEX_TEST_VECTOR_TYPE";
    const CHILD_ROWS: &str = "VORTEX_SQL_INDEX_TEST_VECTOR_ROWS";
    if let Some(expression) = std::env::var_os(CHILD_TYPE) {
        let root = tempfile::tempdir()?;
        let conn = connection()?;
        let file = root.path().join("invalid.vortex");
        let rows = std::env::var(CHILD_ROWS).unwrap_or_else(|_| "128".into());
        conn.query(&format!(
            "COPY (SELECT {} AS embedding FROM range({rows})) TO {} (FORMAT vortex)",
            expression.to_string_lossy(),
            literal(&file)
        ))?;
        let entries = std::fs::read_dir(root.path())?.count();
        let error = conn
            .query(&build_sql(root.path(), &[file]))
            .err()
            .ok_or_else(|| vortex_err!("Invalid vector type accepted"))?;
        assert!(error.to_string().contains("Float32"), "{error}");
        assert!(!root.path().join("index.json").exists());
        assert_eq!(std::fs::read_dir(root.path())?.count(), entries);
        conn.query("SELECT 1")?;
        return Ok(());
    }
    for expression in [
        "[1::FLOAT, 2::FLOAT]",
        "1::FLOAT",
        "['a', 'b']::VARCHAR[2]",
        "[true, false]::BOOLEAN[2]",
        "[1::DOUBLE, 2::DOUBLE]::DOUBLE[2]",
    ] {
        for rows in [0, 128] {
            let child = Command::new(std::env::current_exe()?)
                .args([
                    "--exact",
                    "index::tests::test_sql_invalid_vector_types_return_errors_without_aborting",
                    "--nocapture",
                ])
                .env(CHILD_TYPE, expression)
                .env(CHILD_ROWS, rows.to_string())
                .output()?;
            assert!(
                child.status.success(),
                "Vector type {expression} with {rows} rows failed with {}: {}{}",
                child.status,
                String::from_utf8_lossy(&child.stdout),
                String::from_utf8_lossy(&child.stderr)
            );
        }
    }
    Ok(())
}

#[rstest::rstest]
#[case::direct("SELECT \"row\".id FROM vortex_index_search({reference}, $1, 1)")]
#[case::cte(
    "WITH hits AS (SELECT * FROM vortex_index_search({reference}, $1, 1)) SELECT \"row\".id FROM hits"
)]
#[case::scalar_subquery(
    "SELECT (SELECT \"row\".id FROM vortex_index_search({reference}, $1, 1)) AS id"
)]
#[case::union(
    "SELECT \"row\".id FROM vortex_index_search({reference}, $1, 1) UNION ALL SELECT 999::UBIGINT WHERE false"
)]
fn test_sql_parameterized_search_retains_reference_identity(#[case] sql: &str) -> VortexResult<()> {
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
    let query = sql.replace("{reference}", &literal(&reference));
    conn.query(&format!("PREPARE indexed AS {query}"))?;
    for _ in 0..2 {
        let result = conn
            .query("EXECUTE indexed([7::FLOAT, 0::FLOAT])")?
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("Missing parameterized search result"))?;
        assert_eq!(
            String::try_from(&*result)?,
            "Chunk - [1 Columns]\n- FLAT UBIGINT: 1 = [ 7]\n"
        );
    }
    conn.query("EXECUTE indexed([8::FLOAT, 1::FLOAT])")?;
    let original = std::fs::read(&reference)?;
    std::fs::write(&reference, std::fs::read(&replacement)?)?;
    let error = conn
        .query("EXECUTE indexed([7::FLOAT, 0::FLOAT])")
        .err()
        .ok_or_else(|| {
            vortex_err!("Parameterized search silently accepted a replaced reference")
        })?;
    assert!(error.to_string().contains("reference changed"), "{error}");
    // A separately prepared statement is allowed to accept the new reference.
    conn.query(&format!("PREPARE refreshed AS {query}"))?;
    let result = conn
        .query("EXECUTE refreshed([7::FLOAT, 0::FLOAT])")?
        .into_iter()
        .next()
        .ok_or_else(|| vortex_err!("Missing refreshed search result"))?;
    assert_eq!(
        String::try_from(&*result)?,
        "Chunk - [1 Columns]\n- FLAT UBIGINT: 1 = [ 128]\n"
    );
    std::fs::write(&reference, original)?;
    conn.query("EXECUTE indexed([7::FLOAT, 0::FLOAT])")?;
    Ok(())
}

#[rstest::rstest]
#[case::literal(false)]
#[case::parameterized(true)]
fn test_sql_prepared_search_keeps_identity_across_catalog_rebind(
    #[case] parameterized: bool,
) -> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let conn = connection()?;
    let file = source(&conn, root.path(), "a.vortex", 0, 128)?;
    conn.query(&build_sql(root.path(), &[file]))?;
    let path = root.path().join("index.json");
    let prepare = if parameterized {
        format!(
            "SELECT * FROM vortex_index_search({}, $1, 2)",
            literal(&path)
        )
    } else {
        format!(
            "SELECT * FROM vortex_index_search({}, [7::FLOAT, 0::FLOAT], 2)",
            literal(&path)
        )
    };
    let execute = if parameterized {
        "EXECUTE indexed([7::FLOAT, 0::FLOAT])"
    } else {
        "EXECUTE indexed"
    };
    conn.query(&format!("PREPARE indexed AS {prepare}"))?;
    conn.query(execute)?;
    let original = std::fs::read(&path)?;
    let mut changed = original.clone();
    changed.push(b' ');
    std::fs::write(&path, changed)?;
    conn.query("CREATE TABLE trigger_rebind(id INTEGER)")?;
    let error = conn
        .query(execute)
        .err()
        .ok_or_else(|| vortex_err!("Catalog rebind accepted a changed index reference"))?;
    assert!(error.to_string().contains("reference changed"), "{error}");
    std::fs::write(&path, original)?;
    conn.query(execute)?;
    conn.query("DEALLOCATE indexed")?;
    conn.query(&format!("PREPARE indexed AS {prepare}"))?;
    conn.query(execute)?;
    Ok(())
}

#[rstest::rstest]
#[case::select("SELECT * FROM vortex_index_search({reference}, [$1::FLOAT, 0::FLOAT], 1)")]
#[case::call("CALL vortex_index_search({reference}, [$1::FLOAT, 0::FLOAT], 1)")]
fn test_c_api_prepared_search_retains_reference_identity(#[case] sql: &str) -> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let conn = connection()?;
    let file = source(&conn, root.path(), "a.vortex", 0, 128)?;
    conn.query(&build_sql(root.path(), &[file]))?;
    let reference = root.path().join("index.json");
    let query = sql.replace("{reference}", &literal(&reference));
    let prepared = Prepared::new(&conn, &query)?;
    assert_eq!(prepared.execute(7.0)?.row_count(), 1);
    assert_eq!(prepared.execute(8.0)?.row_count(), 1);
    let original = std::fs::read(&reference)?;
    let mut changed = original.clone();
    changed.push(b' ');
    std::fs::write(&reference, changed)?;
    let error = prepared
        .execute(7.0)
        .err()
        .ok_or_else(|| vortex_err!("C API prepared search accepted a changed reference"))?;
    assert!(error.to_string().contains("reference changed"), "{error}");
    let fresh = Prepared::new(&conn, &query)?;
    assert_eq!(fresh.execute(7.0)?.row_count(), 1);
    std::fs::write(&reference, original)?;
    assert_eq!(prepared.execute(7.0)?.row_count(), 1);
    Ok(())
}

#[test]
fn test_sql_respects_disabled_local_filesystem_at_bind_and_execution() -> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let conn = connection()?;
    let file = source(&conn, root.path(), "a.vortex", 0, 128)?;
    let build = build_sql(root.path(), std::slice::from_ref(&file));
    conn.query(&build)?;
    let reference = root.path().join("index.json");
    let blocked_reference = root.path().join("blocked.json");
    let blocked_build = build.replace(&literal(&reference), &literal(&blocked_reference));
    let search = format!(
        "SELECT * FROM vortex_index_search({}, [1::FLOAT, 0::FLOAT], 2)",
        literal(&reference)
    );
    conn.query(&format!("PREPARE blocked_search AS {search}"))?;
    conn.query(&format!("PREPARE blocked_build AS {blocked_build}"))?;
    let entries = std::fs::read_dir(root.path())?.count();
    conn.query("SET disabled_filesystems = 'LocalFileSystem'")?;
    for sql in [
        search.as_str(),
        blocked_build.as_str(),
        "EXECUTE blocked_search",
        "EXECUTE blocked_build",
    ] {
        let error = conn
            .query(sql)
            .err()
            .ok_or_else(|| vortex_err!("Disabled local filesystem accepted: {sql}"))?;
        assert!(error.to_string().contains("LocalFileSystem"), "{error}");
        assert!(!blocked_reference.exists());
        assert_eq!(std::fs::read_dir(root.path())?.count(), entries);
    }
    let conn = connection()?;
    conn.query("SET disabled_filesystems = 'PipeFileSystem'")?;
    conn.query(&search)?;
    conn.query(&blocked_build)?;
    Ok(())
}

#[rstest::rstest]
#[case::search_reference(0)]
#[case::search_options(1)]
#[case::build_source(2)]
#[case::build_reference(3)]
#[case::build_field(4)]
#[case::build_backend(5)]
#[case::build_options(6)]
fn test_sql_rejects_nul_in_every_string_argument(#[case] argument: usize) -> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let conn = connection()?;
    let file = source(&conn, root.path(), "a.vortex", 0, 128)?;
    conn.query(&build_sql(root.path(), std::slice::from_ref(&file)))?;
    let reference = root.path().join("index.json");
    let new_reference = root.path().join("new-index.json");
    let nul = |value: String| format!("{value} || chr(0) || 'missing'");
    let sql = match argument {
        0 => format!(
            "SELECT * FROM vortex_index_search({}, [1::FLOAT, 0::FLOAT], 2)",
            nul(literal(&reference))
        ),
        1 => format!(
            "SELECT * FROM vortex_index_search({}, [1::FLOAT, 0::FLOAT], 2, backend_options := chr(0) || '{{\"unknown\":1}}')",
            literal(&reference)
        ),
        _ => {
            let mut inputs = [
                format!("[{}]", literal(&file)),
                literal(&new_reference),
                "'embedding'".into(),
                format!("'{BACKEND}'"),
                "'{}'".into(),
            ];
            if argument == 2 {
                inputs[0] = format!("[{}]", nul(literal(&file)));
            } else {
                inputs[argument - 2] = nul(inputs[argument - 2].clone());
            }
            format!("SELECT * FROM vortex_index_build({})", inputs.join(","))
        }
    };
    let entries = std::fs::read_dir(root.path())?.count();
    let error = conn
        .query(&sql)
        .err()
        .ok_or_else(|| vortex_err!("NUL argument was accepted: {sql}"))?;
    assert!(error.to_string().contains("NUL"), "{error}");
    assert!(!new_reference.exists());
    assert_eq!(std::fs::read_dir(root.path())?.count(), entries);
    Ok(())
}

#[test]
fn test_sql_cross_process_reopen_interleaved_files_and_multiple_output_chunks() -> VortexResult<()>
{
    const CHILD_REFERENCE: &str = "VORTEX_SQL_INDEX_TEST_REFERENCE";
    if let Some(path) = std::env::var_os(CHILD_REFERENCE) {
        let conn = connection()?;
        let output = conn.query(&format!(
            "SELECT count(*), min(rank), max(rank), count(DISTINCT \"row\".id), bool_and(\"row\".id = row_offset * 2 + file_id - 1) FROM vortex_index_search({}, [3072::FLOAT, 0::FLOAT], 4096)",
            literal(Path::new(&path))
        ))?;
        let chunk = output
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("Missing aggregate result"))?;
        assert_eq!(
            String::try_from(&*chunk)?,
            "Chunk - [5 Columns]\n- FLAT BIGINT: 1 = [ 4096]\n- FLAT UBIGINT: 1 = [ 1]\n- FLAT UBIGINT: 1 = [ 4096]\n- FLAT BIGINT: 1 = [ 4096]\n- FLAT BOOLEAN: 1 = [ true]\n"
        );
        return Ok(());
    }
    let root = tempfile::tempdir()?;
    let conn = connection()?;
    let mut files = Vec::new();
    for file in 0..2 {
        let path = root.path().join(format!("{file}.vortex"));
        conn.query(&format!(
            "COPY (SELECT (i * 2 + {file})::UBIGINT AS id, [(i * 2 + {file})::FLOAT, 0::FLOAT]::FLOAT[2] AS embedding, 'row-' || (i * 2 + {file}) AS label FROM range(3072) t(i)) TO {} (FORMAT vortex)",
            literal(&path)
        ))?;
        files.push(path);
    }
    conn.query(&build_sql(root.path(), &files))?;
    drop(conn);
    let child = Command::new(std::env::current_exe()?)
        .args([
            "--exact",
            "index::tests::test_sql_cross_process_reopen_interleaved_files_and_multiple_output_chunks",
            "--nocapture",
        ])
        .env(CHILD_REFERENCE, root.path().join("index.json"))
        .output()?;
    assert!(
        child.status.success(),
        "Cross-process query failed: {}{}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
    Ok(())
}

#[test]
fn test_sql_preserves_build_identity_and_rejects_manifest_artifact_changes() -> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let conn = connection()?;
    let file = source(&conn, root.path(), "a.vortex", 0, 128)?;
    let build = build_sql(root.path(), &[file]);
    assert!(
        conn.query(&build.replace("'{}'", "'changed-identity'"))
            .is_err()
    );
    assert!(!root.path().join("index.json").exists());
    conn.query(&build)?;
    let query = format!(
        "SELECT * FROM vortex_index_search({}, [1::FLOAT, 0::FLOAT], 2)",
        literal(&root.path().join("index.json"))
    );
    let (reference, _) = super::read_reference(&root.path().join("index.json"))?;
    let generation = root.path().join(reference.generation.generation);
    for path in [
        generation.join("manifest.json"),
        generation.join("artifacts/vectors.json"),
    ] {
        let original = std::fs::read(&path)?;
        let mut changed = original.clone();
        changed.push(b' ');
        std::fs::write(&path, changed)?;
        assert!(conn.query(&query).is_err());
        std::fs::write(&path, original)?;
        conn.query(&query)?;
    }
    Ok(())
}

#[test]
fn test_index_inputs_reject_non_regular_files_and_symlinks() -> VortexResult<()> {
    let root = tempfile::tempdir()?;
    let fifo = root.path().join("fifo");
    rustix::fs::mkfifoat(
        rustix::fs::CWD,
        &fifo,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
    )
    .map_err(|error| vortex_err!("{error}"))?;
    assert!(super::read_regular(&fifo, 1024).is_err());
    let file = root.path().join("file");
    std::fs::write(&file, b"regular")?;
    let link = root.path().join("link");
    symlink(&file, &link)?;
    assert!(super::read_regular(&link, 1024).is_err());
    assert!(super::read_regular(root.path(), 1024).is_err());
    assert!(super::read_regular(&file, 2).is_err());
    Ok(())
}
