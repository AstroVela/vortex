// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The stages every SQL benchmark runs through.
//!
//! 1. [`generate_data`] writes the on-disk data for each requested [`Format`].
//! 2. [`register_tables`] registers each of the benchmark's tables (or views) into an engine
//!    catalog, via that engine's [`TableRegistrar`].
//! 3. [`crate::runner::SqlBenchmarkRunner`] runs the queries.
//!
//! Stage 1 is idempotent, so a benchmark binary can call it even when a separate `data-gen` run
//! already produced the data. Stage 2 is engine-specific in *mechanism* — DuckDB creates SQL
//! views, DataFusion and Lance register table providers — but uniform in *shape*: every engine
//! resolves the same [`TableSource`] per table.

use std::path::Path;
use std::process::Command;

use arrow_schema::Schema;
use glob::Pattern;
use tracing::info;
use url::Url;

use crate::Benchmark;
use crate::CompactionStrategy;
use crate::Format;
use crate::TableSpec;
use crate::conversions::convert_parquet_directory_to_vortex;

/// Stage 1: generate the benchmark's data for every requested format.
///
/// Always (re)generates the canonical Parquet base data via [`Benchmark::generate_base_data`],
/// then derives each requested format from it. Every step is idempotent, so calling this when the
/// data already exists is cheap.
///
/// Formats that no benchmark-agnostic conversion exists for (`Format::Lance`, for example) are
/// left to the caller; the per-format [`Benchmark::prepare_format`] hook still runs for them.
///
/// Conversions only run against local data — a remote `data_url` is assumed to be pre-populated.
pub async fn generate_data(
    benchmark: &dyn Benchmark,
    formats: impl IntoIterator<Item = Format>,
) -> anyhow::Result<()> {
    benchmark.generate_base_data().await?;

    let data_url = benchmark.data_url();
    if data_url.scheme() != "file" {
        info!(url = %data_url, "Remote data URL, assuming data already exists");
        return Ok(());
    }

    let base_path = data_url
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("Invalid file URL: {data_url}"))?;

    let mut seen = Vec::new();
    for format in formats {
        if seen.contains(&format) {
            continue;
        }
        seen.push(format);

        match format {
            Format::OnDiskVortex => {
                convert_parquet_directory_to_vortex(&base_path, CompactionStrategy::Default).await?
            }
            Format::VortexCompact => {
                convert_parquet_directory_to_vortex(&base_path, CompactionStrategy::Compact).await?
            }
            _ => {}
        }

        benchmark.prepare_format(format, &base_path).await?;
    }

    Ok(())
}

/// Pre-run stage 2 for DuckDB's persistent catalog, using the `duckdb` CLI.
///
/// DuckDB is the one engine whose catalog outlives the process: `CREATE TABLE` loads the Parquet
/// data into `{base_path}/duckdb/duckdb.db`, which is then benchmarked (and, in CI, uploaded)
/// as a data artifact in its own right. Registering it here keeps that load out of the benchmark
/// run. The statements are the same ones [`DuckDbRegistrar`]-style in-process registration would
/// issue, and are `IF NOT EXISTS`, so a later stage-2 pass over the same database is a no-op.
///
/// [`DuckDbRegistrar`]: TableRegistrar
pub fn generate_duckdb_database(benchmark: &dyn Benchmark, base_path: &Path) -> anyhow::Result<()> {
    let duckdb_dir = base_path.join(Format::OnDiskDuckDB.name());
    std::fs::create_dir_all(&duckdb_dir)?;

    let db_path = duckdb_dir.join("duckdb.db");
    if db_path.exists() {
        info!("DuckDB database already exists at {}", db_path.display());
        return Ok(());
    }

    for spec in benchmark.table_specs() {
        let source = TableSource::resolve(benchmark, spec, Format::Parquet, TableObject::Table)?;
        let output = Command::new("duckdb")
            .arg(&db_path)
            .arg("-c")
            .arg(source.duckdb_registration_sql())
            .output()?;

        if !output.status.success() {
            anyhow::bail!(
                "DuckDB CLI failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    Ok(())
}

/// How an engine materializes a benchmark table in its catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableObject {
    /// Data is loaded into the engine's own storage.
    Table,
    /// A view over the data files, read at query time.
    View,
}

impl TableObject {
    /// The SQL keyword for this object kind.
    pub fn sql_keyword(&self) -> &'static str {
        match self {
            TableObject::Table => "TABLE",
            TableObject::View => "VIEW",
        }
    }
}

/// What an engine reads when registering a benchmark's tables for one requested format.
#[derive(Clone, Copy, Debug)]
pub struct Registration {
    /// Whether tables are registered as tables or as views.
    pub object: TableObject,
    /// The format the files on disk are read from. Differs from the requested format when an
    /// engine loads from another one — DuckDB's on-disk database is loaded from Parquet.
    pub load_format: Format,
}

impl Registration {
    /// Read `format`'s own files, registered as views. The common case.
    pub fn views_of(format: Format) -> Self {
        Self {
            object: TableObject::View,
            load_format: format,
        }
    }
}

/// A single benchmark table resolved to the files backing it, for one format.
///
/// This is the engine-agnostic half of stage 2: resolving a [`TableSpec`] against the benchmark's
/// [`Benchmark::format_path`] and [`Benchmark::pattern`] is identical for every engine, so it
/// happens once here rather than in each engine's registration code.
#[derive(Debug)]
pub struct TableSource {
    /// The table's name in the engine catalog.
    pub name: &'static str,
    /// The table's schema, when the benchmark pins one instead of letting the engine infer it.
    pub schema: Option<Schema>,
    /// Directory URL holding the table's files.
    pub base_url: Url,
    /// Glob matching the table's files within `base_url`, when the benchmark narrows it.
    pub pattern: Option<Pattern>,
    /// The format the files are stored in.
    pub load_format: Format,
    /// Whether to register a table or a view.
    pub object: TableObject,
}

impl TableSource {
    /// Resolve `spec` against the benchmark's data layout.
    pub fn resolve(
        benchmark: &dyn Benchmark,
        spec: TableSpec,
        load_format: Format,
        object: TableObject,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            name: spec.name,
            schema: spec.schema,
            base_url: benchmark.format_path(load_format, benchmark.data_url())?,
            pattern: benchmark.pattern(spec.name, load_format),
            load_format,
            object,
        })
    }

    /// Glob matching this table's files, falling back to every file of the format's extension.
    pub fn glob(&self) -> String {
        match &self.pattern {
            Some(pattern) => pattern.as_str().to_string(),
            None => format!("*.{}", self.load_format.ext()),
        }
    }

    /// `base_url` as a plain directory string without a trailing slash, for engines that take a
    /// path rather than a URL. Local paths keep no `file://` prefix.
    pub fn base_dir(&self) -> String {
        let base_url = self.base_url.as_str();
        base_url
            .strip_prefix("file://")
            .unwrap_or(base_url)
            .trim_end_matches('/')
            .to_string()
    }

    /// The DuckDB statement registering this table.
    pub fn duckdb_registration_sql(&self) -> String {
        format!(
            "CREATE {object} IF NOT EXISTS {name} AS SELECT * FROM read_{ext}('{dir}/{glob}');\n",
            object = self.object.sql_keyword(),
            name = self.name,
            ext = self.load_format.ext(),
            dir = self.base_dir(),
            glob = self.glob(),
        )
    }
}

/// Stage 2: an engine's catalog, into which a benchmark's tables (or views) are registered.
///
/// Implementors handle only the engine-specific registration of one already-resolved
/// [`TableSource`]; [`register_tables`] drives the loop.
#[async_trait::async_trait(?Send)]
pub trait TableRegistrar {
    /// How this engine reads `format`: as tables or views, and from which format's files.
    ///
    /// Returns an error for formats the engine cannot read.
    fn registration(&self, format: Format) -> anyhow::Result<Registration>;

    /// Register a single table into the catalog.
    async fn register(&mut self, source: &TableSource) -> anyhow::Result<()>;
}

/// Stage 2: register every table of `benchmark` for `format` into `registrar`'s catalog.
pub async fn register_tables<R: TableRegistrar + ?Sized>(
    registrar: &mut R,
    benchmark: &dyn Benchmark,
    format: Format,
) -> anyhow::Result<()> {
    let Registration {
        object,
        load_format,
    } = registrar.registration(format)?;

    for spec in benchmark.table_specs() {
        let source = TableSource::resolve(benchmark, spec, load_format, object)?;
        info!(
            name = source.name,
            base_dir = source.base_dir(),
            glob = source.glob(),
            format = load_format.name(),
            "Registering {}",
            object.sql_keyword().to_lowercase(),
        );
        registrar.register(&source).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tpch::benchmark::TpcHBenchmark;

    fn tpch() -> anyhow::Result<TpcHBenchmark> {
        TpcHBenchmark::new("1.0".to_string(), None)
    }

    #[test]
    fn resolves_pattern_from_benchmark() -> anyhow::Result<()> {
        let benchmark = tpch()?;
        let spec = TableSpec::new("lineitem", None);
        let source =
            TableSource::resolve(&benchmark, spec, Format::OnDiskVortex, TableObject::View)?;

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
}
