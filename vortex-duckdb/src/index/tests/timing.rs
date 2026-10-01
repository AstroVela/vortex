// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::process::Command;

use serde::Deserialize;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;

use super::build_sql;
use super::connection;
use super::literal;
use super::nearest_id;
use super::source;

#[derive(Deserialize)]
struct Event {
    event: String,
    format_version: u32,
    total_ms: f64,
    phases: BTreeMap<String, f64>,
}

#[test]
fn test_search_timing_is_opt_in() -> VortexResult<()> {
    const CHILD: &str = "VORTEX_SQL_INDEX_TEST_TIMING_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let root = tempfile::tempdir()?;
        let conn = connection()?;
        let file = source(&conn, root.path(), "source.vortex", 0, 128)?;
        conn.query(&build_sql(root.path(), &[file]))?;
        assert_eq!(
            nearest_id(conn.query(&format!(
                "SELECT \"row\".id FROM vortex_index_search({}, [7::FLOAT, 0::FLOAT], 1)",
                literal(&root.path().join("index.json"))
            ))?)?,
            7
        );
        return Ok(());
    }

    for enabled in [false, true] {
        let output = Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "index::tests::timing::test_search_timing_is_opt_in",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .env("VORTEX_INDEX_TIMING", if enabled { "1" } else { "0" })
            .output()?;
        if !output.status.success() {
            vortex_bail!(
                "Timing child failed: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        let events = stderr
            .lines()
            .filter_map(|line| serde_json::from_str::<Event>(line).ok())
            .filter(|event| event.event == "vortex_index_search_timing")
            .collect::<Vec<_>>();
        assert_eq!(events.len(), usize::from(enabled));
        if let Some(event) = events.first() {
            assert_eq!(event.format_version, 1);
            assert_eq!(event.phases.len(), 7);
            assert!(event.total_ms.is_finite() && event.total_ms > 0.0);
            assert!(event.phases.values().all(|ms| ms.is_finite() && *ms >= 0.0));
            assert!((event.phases.values().sum::<f64>() - event.total_ms).abs() < 1e-6);
        }
    }
    Ok(())
}
