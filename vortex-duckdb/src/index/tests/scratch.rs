// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::session::VortexSession;
use vortex_index::Index;
use vortex_index::IndexBuilder;
use vortex_index::IndexMetadata;
use vortex_index::IndexProvider;
use vortex_index::IndexStore;

use super::BACKEND;
use super::Provider;
use super::build_sql;
use super::connection;
use super::literal;
use super::nearest_id;
use super::register_index_provider_factory;
use super::source;

const SCRATCH_BACKEND: &str = "sql.scratch.fixture";
static SCRATCH_MODES: Mutex<Vec<u32>> = Mutex::new(Vec::new());

struct ScratchProvider;

fn factory(_session: VortexSession, scratch: &Path) -> VortexResult<Arc<dyn IndexProvider>> {
    SCRATCH_MODES
        .lock()
        .push(fs::metadata(scratch)?.permissions().mode() & 0o777);
    Ok(Arc::new(ScratchProvider))
}

#[async_trait]
impl IndexProvider for ScratchProvider {
    fn id(&self) -> &str {
        SCRATCH_BACKEND
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
        Provider.open(metadata, store).await
    }
}

#[test]
fn test_scratch_leases_are_private() -> VortexResult<()> {
    const CHILD: &str = "VORTEX_SQL_INDEX_TEST_SCRATCH_PERMISSIONS";
    if std::env::var_os(CHILD).is_some() {
        register_index_provider_factory(SCRATCH_BACKEND, factory)?;
        let root = tempfile::tempdir()?;
        fs::set_permissions(root.path(), fs::Permissions::from_mode(0o755))?;
        let conn = connection()?;
        let file = source(&conn, root.path(), "source.vortex", 0, 128)?;
        conn.query(&build_sql(root.path(), &[file]).replace(BACKEND, SCRATCH_BACKEND))?;
        assert_eq!(
            nearest_id(conn.query(&format!(
                "SELECT \"row\".id FROM vortex_index_search({}, [7::FLOAT, 0::FLOAT], 1)",
                literal(&root.path().join("index.json"))
            ))?)?,
            7
        );
        assert_eq!(
            *SCRATCH_MODES.lock(),
            vec![0o700, 0o700],
            "Build and search must both pass private scratch directories to the factory"
        );
        return Ok(());
    }

    // Set the process-wide umask before starting the child test harness, never
    // in this process where unrelated tests may be running concurrently.
    let mut failures = Vec::new();
    for mask in ["002", "022", "077"] {
        let child = Command::new("sh")
            .args([
                "-c",
                "umask \"$1\" || exit\nshift\nexec \"$@\"",
                "vortex-index-scratch",
                mask,
            ])
            .arg(std::env::current_exe()?)
            .args([
                "--exact",
                "index::tests::scratch::test_scratch_leases_are_private",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()?;
        if !child.status.success() {
            failures.push(format!(
                "umask {mask} failed with {}: {}{}",
                child.status,
                String::from_utf8_lossy(&child.stdout),
                String::from_utf8_lossy(&child.stderr)
            ));
        }
    }
    if !failures.is_empty() {
        vortex_bail!("Private scratch checks failed:\n{}", failures.join("\n"));
    }
    Ok(())
}
