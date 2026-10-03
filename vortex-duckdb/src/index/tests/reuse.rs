// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod snapshot;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;

use async_trait::async_trait;
use parking_lot::Mutex;
use tempfile::TempDir;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_index::Index;
use vortex_index::IndexBuilder;
use vortex_index::IndexMetadata;
use vortex_index::IndexProvider;
use vortex_index::IndexStore;

use super::BACKEND;
use super::Prepared;
use super::Provider;
use super::build_sql;
use super::connection;
use super::literal;
use super::nearest_id;
use super::source;
use crate::duckdb::Connection;
use crate::index::cache::MAX_RETAINED_HANDLES;
use crate::index::read_reference;
use crate::index::register_index_provider_factory;

const OBSERVED_BACKEND: &str = "sql.reuse.fixture";

#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    opens: usize,
    live: usize,
    closed_with_scratch: usize,
}

static COUNTS: LazyLock<Mutex<BTreeMap<String, Counts>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

struct ObservedProvider(PathBuf);

fn factory(_session: VortexSession, scratch: &Path) -> VortexResult<Arc<dyn IndexProvider>> {
    Ok(Arc::new(ObservedProvider(scratch.to_owned())))
}

#[async_trait]
impl IndexProvider for ObservedProvider {
    fn id(&self) -> &str {
        OBSERVED_BACKEND
    }

    fn supports_version(&self, version: u32) -> bool {
        version == 1
    }

    fn builder(&self) -> Option<&dyn IndexBuilder> {
        Some(&Provider)
    }

    async fn open(
        &self,
        metadata: &IndexMetadata,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<Arc<dyn Index>> {
        if self
            .0
            .parent()
            .is_some_and(|root| root.join("fail-open").exists())
        {
            vortex_bail!("Fixture provider open failed");
        }
        let inner = Provider.open(metadata, store).await?;
        let mut counts = COUNTS.lock();
        let counts = counts.entry(metadata.generation.clone()).or_default();
        counts.opens += 1;
        counts.live += 1;
        Ok(Arc::new(ObservedIndex {
            inner,
            scratch: self.0.clone(),
        }))
    }
}

#[derive(Debug)]
struct ObservedIndex {
    inner: Arc<dyn Index>,
    scratch: PathBuf,
}

impl Index for ObservedIndex {
    fn metadata(&self) -> &IndexMetadata {
        self.inner.metadata()
    }

    fn as_vector(&self) -> Option<&dyn vortex_index::VectorIndex> {
        self.inner.as_vector()
    }
}

impl Drop for ObservedIndex {
    fn drop(&mut self) {
        let mut counts = COUNTS.lock();
        let counts = counts
            .entry(self.metadata().generation.clone())
            .or_default();
        counts.live -= 1;
        counts.closed_with_scratch += usize::from(self.scratch.is_dir());
    }
}

struct Fixture {
    conn: Connection,
    root: TempDir,
    reference: PathBuf,
    file: PathBuf,
    generation: String,
}

impl Fixture {
    fn new() -> VortexResult<Self> {
        register_index_provider_factory(OBSERVED_BACKEND, factory)?;
        let root = tempfile::tempdir()?;
        let conn = connection()?;
        let file = source(&conn, root.path(), "source.vortex", 0, 128)?;
        let reference = root.path().join("index.json");
        conn.query(
            &build_sql(root.path(), std::slice::from_ref(&file)).replace(BACKEND, OBSERVED_BACKEND),
        )?;
        let (descriptor, _) = read_reference(&reference)?;
        let generation = descriptor.generation.generation;
        COUNTS.lock().insert(generation.clone(), Counts::default());
        Ok(Self {
            conn,
            root,
            reference,
            file,
            generation,
        })
    }

    fn query(&self, vector: &str) -> String {
        format!(
            "SELECT \"row\".id FROM vortex_index_search({}, [{vector}::FLOAT, 0::FLOAT], 1)",
            literal(&self.reference)
        )
    }

    fn counts(&self) -> VortexResult<Counts> {
        COUNTS
            .lock()
            .get(&self.generation)
            .copied()
            .ok_or_else(|| vortex_err!("Missing provider observations"))
    }

    fn scratch_count(&self) -> VortexResult<usize> {
        Ok(fs::read_dir(self.root.path())?
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".index-scratch-")
            })
            .count())
    }
}

#[test]
fn test_sql_owner_reuses_provider_across_parameter_and_catalog_rebind() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    fixture
        .conn
        .query(&format!("PREPARE indexed AS {}", fixture.query("$1")))?;
    assert_eq!(fixture.counts()?.opens, 0);
    for value in [7, 14, 21] {
        assert_eq!(
            nearest_id(fixture.conn.query(&format!("EXECUTE indexed({value})"))?)?,
            value
        );
    }
    fixture
        .conn
        .query("CREATE TABLE rebind_marker(id INTEGER)")?;
    assert_eq!(nearest_id(fixture.conn.query("EXECUTE indexed(7)")?)?, 7);
    assert_eq!(fixture.counts()?.opens, 1);
    assert_eq!(fixture.counts()?.live, 1);
    assert_eq!(fixture.scratch_count()?, 1);
    fixture.conn.query("DEALLOCATE indexed")?;
    assert_eq!(fixture.counts()?.live, 0);
    assert_eq!(fixture.counts()?.closed_with_scratch, 1);
    assert_eq!(fixture.scratch_count()?, 0);
    Ok(())
}

#[rstest::rstest]
#[case::query_parameter("$1", "", 7.0)]
#[case::where_parameter("7", "WHERE rank >= $1", 1.0)]
#[case::limit_parameter("7", "LIMIT $1", 1.0)]
fn test_c_api_owner_reuses_provider_and_releases_on_destroy(
    #[case] vector: &str,
    #[case] suffix: &str,
    #[case] parameter: f64,
) -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let handle = Prepared::new(
        &fixture.conn,
        &format!("{} {suffix}", fixture.query(vector)),
    )?;
    for _ in 0..3 {
        assert_eq!(nearest_id(handle.execute(parameter)?)?, 7);
    }
    assert_eq!(fixture.counts()?.opens, 1);
    assert_eq!(fixture.counts()?.live, 1);
    drop(handle);
    assert_eq!(fixture.counts()?.live, 0);
    assert_eq!(fixture.counts()?.closed_with_scratch, 1);
    assert_eq!(fixture.scratch_count()?, 0);
    Ok(())
}

#[test]
fn test_independent_owners_and_ad_hoc_queries_do_not_share_provider() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let sql = fixture.query("$1");
    let first = Prepared::new(&fixture.conn, &sql)?;
    let second = Prepared::new(&fixture.conn, &sql)?;
    assert_eq!(nearest_id(first.execute(7.0)?)?, 7);
    assert_eq!(nearest_id(second.execute(7.0)?)?, 7);
    assert_eq!(fixture.counts()?.opens, 2);
    assert_eq!(fixture.counts()?.live, 2);
    drop(first);
    assert_eq!(fixture.counts()?.live, 1);
    assert_eq!(nearest_id(second.execute(14.0)?)?, 14);
    assert_eq!(fixture.counts()?.opens, 2);
    drop(second);
    for _ in 0..2 {
        assert_eq!(nearest_id(fixture.conn.query(&fixture.query("7"))?)?, 7);
    }
    assert_eq!(fixture.counts()?.opens, 4);
    assert_eq!(fixture.counts()?.live, 0);
    assert_eq!(fixture.counts()?.closed_with_scratch, 4);
    assert_eq!(fixture.scratch_count()?, 0);
    Ok(())
}

#[rstest::rstest]
#[case::source("source")]
#[case::artifact("artifact")]
#[case::manifest("manifest")]
#[case::reference("reference")]
fn test_reuse_keeps_execution_time_file_validation(
    #[case] changed: &str,
    #[values(false, true)] removed: bool,
) -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let handle = Prepared::new(&fixture.conn, &fixture.query("$1"))?;
    assert_eq!(nearest_id(handle.execute(7.0)?)?, 7);
    let path = match changed {
        "source" => fixture.file.clone(),
        "artifact" => fixture
            .root
            .path()
            .join(&fixture.generation)
            .join("artifacts/vectors.json"),
        "manifest" => fixture
            .root
            .path()
            .join(&fixture.generation)
            .join("manifest.json"),
        _ => fixture.reference.clone(),
    };
    let original = fs::read(&path)?;
    if removed {
        fs::remove_file(&path)?;
    } else {
        fs::write(&path, b"corrupt")?;
    }
    assert!(handle.execute(7.0).is_err());
    fs::write(&path, &original)?;
    assert_eq!(nearest_id(handle.execute(7.0)?)?, 7);
    assert_eq!(fixture.counts()?.opens, 1);
    Ok(())
}

#[rstest::rstest]
#[case::external_access("SET enable_external_access=false", "external access")]
#[case::local_filesystem("SET disabled_filesystems='LocalFileSystem'", "disabled")]
fn test_reuse_does_not_bypass_changed_access_policy(
    #[case] setting: &str,
    #[case] message: &str,
) -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let handle = Prepared::new(&fixture.conn, &fixture.query("$1"))?;
    assert_eq!(nearest_id(handle.execute(7.0)?)?, 7);
    fixture.conn.query(setting)?;
    let error = handle
        .execute(7.0)
        .err()
        .ok_or_else(|| vortex_err!("Cached provider bypassed access policy"))?;
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(fixture.counts()?.opens, 1);
    drop(handle);
    assert_eq!(fixture.counts()?.live, 0);
    Ok(())
}

#[test]
fn test_connections_do_not_share_cached_provider_handles() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let other = connection()?;
    let sql = fixture.query("$1");
    let first = Prepared::new(&fixture.conn, &sql)?;
    let second = Prepared::new(&other, &sql)?;
    for _ in 0..2 {
        assert_eq!(nearest_id(first.execute(7.0)?)?, 7);
        assert_eq!(nearest_id(second.execute(7.0)?)?, 7);
    }
    assert_eq!(fixture.counts()?.opens, 2);
    assert_eq!(fixture.counts()?.live, 2);
    drop(first);
    drop(second);
    assert_eq!(fixture.counts()?.live, 0);
    assert_eq!(fixture.scratch_count()?, 0);
    Ok(())
}

#[test]
fn test_connection_handle_limit_falls_back_and_releases_capacity() -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let sql = fixture.query("$1");
    let mut handles = Vec::new();
    for _ in 0..=MAX_RETAINED_HANDLES {
        let handle = Prepared::new(&fixture.conn, &sql)?;
        assert_eq!(nearest_id(handle.execute(7.0)?)?, 7);
        handles.push(handle);
    }
    assert_eq!(fixture.counts()?.opens, MAX_RETAINED_HANDLES + 1);
    assert_eq!(fixture.counts()?.live, MAX_RETAINED_HANDLES);
    let last = handles
        .pop()
        .ok_or_else(|| vortex_err!("Missing excess handle"))?;
    assert_eq!(nearest_id(last.execute(14.0)?)?, 14);
    assert_eq!(fixture.counts()?.opens, MAX_RETAINED_HANDLES + 2);
    drop(handles.pop());
    assert_eq!(nearest_id(last.execute(7.0)?)?, 7);
    assert_eq!(nearest_id(last.execute(14.0)?)?, 14);
    assert_eq!(fixture.counts()?.opens, MAX_RETAINED_HANDLES + 3);
    assert_eq!(fixture.counts()?.live, MAX_RETAINED_HANDLES);
    drop(last);
    drop(handles);
    assert_eq!(fixture.counts()?.live, 0);
    assert_eq!(fixture.scratch_count()?, 0);
    Ok(())
}

#[rstest::rstest]
fn test_failed_provider_open_releases_scratch_and_cache_reservation(
    #[values("strict", "snapshot")] mode: &str,
) -> VortexResult<()> {
    let fixture = Fixture::new()?;
    let query = format!(
        "SELECT \"row\".id FROM vortex_index_search({}, [$1::FLOAT, 0::FLOAT], 1, validation_mode := '{mode}')",
        literal(&fixture.reference)
    );
    let handle = Prepared::new(&fixture.conn, &query)?;
    let failure = fixture.root.path().join("fail-open");
    fs::write(&failure, b"fail")?;
    for _ in 0..MAX_RETAINED_HANDLES {
        let error = handle
            .execute(7.0)
            .err()
            .ok_or_else(|| vortex_err!("Provider failure was ignored"))?;
        assert!(
            error.to_string().contains("provider open failed"),
            "{error}"
        );
        assert_eq!(fixture.scratch_count()?, 0);
    }
    fs::remove_file(failure)?;
    assert_eq!(nearest_id(handle.execute(7.0)?)?, 7);
    assert_eq!(nearest_id(handle.execute(14.0)?)?, 14);
    assert_eq!(fixture.counts()?.opens, 1);
    assert_eq!(fixture.counts()?.live, 1);
    drop(handle);
    assert_eq!(fixture.scratch_count()?, 0);
    Ok(())
}
