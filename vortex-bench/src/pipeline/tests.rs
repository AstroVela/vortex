// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex::error::VortexExpect;

use super::*;
use crate::BenchmarkArg;
use crate::Opts;
use crate::tpch::benchmark::TpcHBenchmark;

fn tpch() -> anyhow::Result<TpcHBenchmark> {
    TpcHBenchmark::new("1.0".to_string(), None)
}

#[test]
fn resolves_pattern_from_benchmark() -> anyhow::Result<()> {
    let benchmark = tpch()?;
    let spec = TableSpec::new("lineitem", None);
    let source = TableSource::resolve(&benchmark, spec, Format::OnDiskVortex, TableObject::View)?;

    assert_eq!(source.glob(), "lineitem_*.vortex");
    assert!(source.base_dir().ends_with("/vortex-file-compressed"));
    Ok(())
}

#[test]
fn falls_back_to_extension_glob() -> anyhow::Result<()> {
    let source = TableSource {
        name: "test",
        schema: None,
        base_url: Url::parse("file:///data/parquet/")?,
        pattern: None,
        load_format: Format::Parquet,
        object: TableObject::View,
    };

    assert_eq!(source.glob(), "*.parquet");
    assert_eq!(source.base_dir(), "/data/parquet");
    Ok(())
}

#[test]
fn registration_sql_matches_object_kind() -> anyhow::Result<()> {
    let benchmark = tpch()?;
    let spec = TableSpec::new("nation", None);
    let view = TableSource::resolve(&benchmark, spec, Format::Parquet, TableObject::View)?;
    assert!(
        view.duckdb_registration_sql()
            .starts_with("CREATE VIEW IF NOT EXISTS nation AS SELECT * FROM read_parquet('/")
    );
    assert!(
        view.duckdb_registration_sql()
            .ends_with("/parquet/nation_*.parquet');\n")
    );

    let spec = TableSpec::new("nation", None);
    let table = TableSource::resolve(&benchmark, spec, Format::Parquet, TableObject::Table)?;
    assert!(
        table
            .duckdb_registration_sql()
            .starts_with("CREATE TABLE IF NOT EXISTS nation")
    );
    Ok(())
}

#[test]
fn remote_base_dir_keeps_scheme() -> anyhow::Result<()> {
    let source = TableSource {
        name: "test",
        schema: None,
        base_url: Url::parse("s3://bucket/tpch/parquet/")?,
        pattern: None,
        load_format: Format::Parquet,
        object: TableObject::View,
    };

    assert_eq!(source.base_dir(), "s3://bucket/tpch/parquet");
    Ok(())
}

/// Every checked-in `create.sql` must register exactly the tables the benchmark declares, in both
/// dialects — otherwise a table silently goes missing at query time.
#[rstest]
#[case::tpch(BenchmarkArg::TpcH)]
#[case::tpcds(BenchmarkArg::TpcDS)]
#[case::clickbench(BenchmarkArg::ClickBench)]
#[case::clickbench_sorted(BenchmarkArg::ClickBenchSorted)]
#[case::appian(BenchmarkArg::Appian)]
#[case::fineweb(BenchmarkArg::Fineweb)]
#[case::gharchive(BenchmarkArg::GhArchive)]
#[case::polarsignals(BenchmarkArg::PolarSignals)]
#[case::statpopgen(BenchmarkArg::StatPopGen)]
fn create_sql_covers_every_table(#[case] arg: BenchmarkArg) -> anyhow::Result<()> {
    let benchmark = crate::create_benchmark(arg, &Opts::from(Vec::new()))?;
    let script = CreateScript::load(&*benchmark)?
        .ok_or_else(|| anyhow::anyhow!("{} has no create.sql", benchmark.dataset_name()))?;

    let expected: Vec<String> = benchmark
        .table_specs()
        .iter()
        .map(|spec| spec.name.to_string())
        .collect();

    for dialect in [SqlDialect::DuckDb, SqlDialect::DataFusion] {
        assert_eq!(
            script.table_names(dialect),
            expected,
            "{} create.sql `{dialect}` section",
            benchmark.dataset_name()
        );
    }

    Ok(())
}

/// Rendering substitutes every placeholder, for both the format's own files and DuckDB's
/// load-from-Parquet case.
#[test]
fn rendered_statements_have_no_placeholders() -> anyhow::Result<()> {
    let benchmark = tpch()?;
    let script = CreateScript::load(&benchmark)?.vortex_expect("tpch has a create.sql");

    for (dialect, registration) in [
        (SqlDialect::DuckDb, Registration::views_of(Format::Parquet)),
        (
            SqlDialect::DuckDb,
            Registration {
                object: TableObject::Table,
                load_format: Format::Parquet,
            },
        ),
        (
            SqlDialect::DataFusion,
            Registration::views_of(Format::OnDiskVortex),
        ),
    ] {
        let statements = script
            .render(dialect, &benchmark, registration)?
            .vortex_expect("tpch create.sql has both dialects");

        assert_eq!(statements.len(), 8);
        for statement in &statements {
            assert!(!statement.contains('{'), "unsubstituted: {statement}");
            assert!(!statement.contains("--"), "comment leaked: {statement}");
        }
    }

    Ok(())
}

/// The checked-in script and the generated fallback must register the same tables from the same
/// files, so a benchmark without a `create.sql` behaves identically.
#[test]
fn create_sql_matches_generated_statements() -> anyhow::Result<()> {
    let benchmark = tpch()?;
    let registration = Registration::views_of(Format::Parquet);

    for dialect in [SqlDialect::DuckDb, SqlDialect::DataFusion] {
        let script = CreateScript::load(&benchmark)?
            .vortex_expect("tpch has a create.sql")
            .render(dialect, &benchmark, registration)?
            .vortex_expect("tpch create.sql has both dialects");
        let generated = create_sql::generated_statements(dialect, &benchmark, registration)?;

        let normalize = |s: &String| s.trim().trim_end_matches(';').to_string();
        assert_eq!(
            script.iter().map(normalize).collect::<Vec<_>>(),
            generated.iter().map(normalize).collect::<Vec<_>>(),
            "{dialect}"
        );
    }

    Ok(())
}

/// A benchmark whose tables are only known at runtime has no file, and falls back cleanly.
#[test]
fn missing_create_sql_falls_back_to_generated() -> anyhow::Result<()> {
    let benchmark = crate::create_benchmark(BenchmarkArg::SpatialBench, &Opts::from(Vec::new()))?;
    assert!(CreateScript::load(&*benchmark)?.is_none());

    let statements = registration_statements(
        SqlDialect::DuckDb,
        &*benchmark,
        Registration::views_of(Format::Parquet),
    )?;
    assert_eq!(statements.len(), benchmark.table_specs().len());

    Ok(())
}

/// Sections are keyed by their `-- @engine` header; text before the first header is preamble.
#[test]
fn parse_splits_sections_and_drops_preamble() {
    let script = CreateScript::parse(
        "preamble.sql".into(),
        "-- docs\n-- @engine duckdb\nCREATE VIEW a AS SELECT 1;\n-- @engine datafusion\nCREATE \
         EXTERNAL TABLE a STORED AS PARQUET LOCATION 'x';\n",
    );

    assert_eq!(script.table_names(SqlDialect::DuckDb).len(), 0);
    assert_eq!(
        script
            .sections
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>(),
        vec!["duckdb", "datafusion"]
    );
}
