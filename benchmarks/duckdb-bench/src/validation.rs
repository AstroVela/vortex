// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Query result validation against golden reference data.

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::bail;
use duckdb_bench::DuckClient;
use similar::ChangeTag;
use similar::TextDiff;
use vortex_bench::Format;
use vortex_bench::tpch::benchmark::TpcHBenchmark;

/// Verify vortex benchmark query output
pub fn verify_vortex_output(
    ctx: &DuckClient,
    queries: &[(usize, String)],
    format: Format,
) -> anyhow::Result<()> {
    let golden_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vortex-bench/sql/vortex/results/duckdb");
    let tmp_dir = env::temp_dir().join("vortex-queries-verify");
    fs::create_dir_all(&tmp_dir)?;
    let mut mismatches = Vec::new();

    for (query_idx, query) in queries {
        let golden_path = golden_dir.join(format!("{query_idx}.csv"));
        if !golden_path.exists() {
            continue;
        }

        let stripped = query.trim_end().trim_end_matches(';');
        let actual_path = tmp_dir.join(format!("{query_idx}.csv"));
        let write_csv = format!(
            "COPY (\n{stripped}\n) TO '{}' (HEADER, DELIMITER '|');",
            actual_path.display()
        );
        ctx.execute_query(&write_csv)?;

        let expected = fs::read_to_string(&golden_path)?;
        let actual = fs::read_to_string(&actual_path)?;

        if expected != actual {
            let diff = TextDiff::from_lines(&expected, &actual);
            for change in diff.iter_all_changes() {
                let sign = match change.tag() {
                    ChangeTag::Delete => "-",
                    ChangeTag::Insert => "+",
                    ChangeTag::Equal => " ",
                };
                print!("{sign}{change}");
            }
            eprintln!("query {query_idx} output does not match reference for {format}");
            mismatches.push(*query_idx);
        }
    }

    if !mismatches.is_empty() {
        bail!("vortex output mismatch for {format} on queries {mismatches:?}");
    }

    Ok(())
}

/// Verify DuckDB TPC-H results against reference data.
///
/// This function runs TPC-H queries via DuckDB on Vortex files and compares
/// the results against known-correct reference outputs.
///
/// Only runs for scale factor 1.0 since reference data is only available for SF=1.
#[expect(dead_code)]
pub fn verify_duckdb_tpch_results(
    benchmark: &TpcHBenchmark,
    queries: Vec<usize>,
) -> anyhow::Result<()> {
    // omit validation for sf != 1.
    if benchmark.scale_factor != "1.0" {
        return Ok(());
    }

    let query_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../vortex-duckdb/duckdb/extension/tpch/dbgen/queries");

    let tmp_dir = format!(
        "{}/spiral-tpch",
        // $RUNNER_TEMP is defined by GitHub Actions.
        env::var("TMPDIR").or_else(|_| env::var("RUNNER_TEMP"))?
    );

    if PathBuf::from(&tmp_dir).exists() {
        fs::remove_dir_all(&tmp_dir)?;
    }
    fs::create_dir(&tmp_dir)?;

    let duckdb = DuckClient::new_in_memory()?;
    duckdb.register_tables(benchmark, Format::OnDiskVortex)?;

    let mut query_files = fs::read_dir(query_dir)?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "sql"))
        .collect::<Vec<_>>();
    query_files.sort_by_key(|entry| entry.file_name());

    let mut is_matching_ref_result = true;

    for query_file in query_files
        .iter()
        .enumerate()
        .filter(|entry| queries.contains(&(entry.0 + 1)))
        .map(|query_file| query_file.1)
    {
        let query_file_path = query_file.path();
        let query_name = query_file_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid query filename"))?;

        let create_table = format!(
            "CREATE OR REPLACE TABLE {query_name}_result AS {};",
            fs::read_to_string(&query_file_path)?
        );

        let csv_actual = format!("{tmp_dir}/{query_name}.csv");
        let write_csv =
            format!("COPY {query_name}_result TO '{csv_actual}' (HEADER, DELIMITER '|');",);

        duckdb.execute_query(&create_table)?;
        duckdb.execute_query(&write_csv)?;

        let csv_expected = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../vortex-bench/tpch/results/duckdb/{query_name}.csv"
        ));
        let expected = fs::read_to_string(csv_expected)?;
        let actual = fs::read_to_string(csv_actual)?;

        if expected != actual {
            let diff = TextDiff::from_lines(&expected, &actual);

            for change in diff.iter_all_changes() {
                let sign = match change.tag() {
                    ChangeTag::Delete => "-",
                    ChangeTag::Insert => "+",
                    ChangeTag::Equal => " ",
                };
                print!("{sign}{change}");
            }

            eprintln!("query output does not match the reference for {query_name}");
            is_matching_ref_result = false;
        }
    }

    if !is_matching_ref_result {
        return Err(anyhow::anyhow!("not all queries matched the reference"));
    }

    Ok(())
}
