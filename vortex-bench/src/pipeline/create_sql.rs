// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Loading a benchmark's checked-in table-registration SQL.
//!
//! A benchmark's queries live in `vortex-bench/sql/`, and so does the DDL that registers the
//! tables they run against: `sql/{dataset_name}/create.sql`. The file is a template — the data
//! directory depends on the scale factor and the requested format — so the harness substitutes a
//! handful of placeholders before executing it.
//!
//! The file is split into per-engine sections because DuckDB and DataFusion do not share a DDL
//! dialect: DuckDB registers a view over a `read_parquet`/`read_vortex` call, DataFusion a
//! `CREATE EXTERNAL TABLE ... STORED AS`. Both describe the same tables over the same files.

use std::fmt::Display;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use super::Registration;
use crate::Benchmark;
use crate::Format;

/// The DDL dialect a `create.sql` section is written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqlDialect {
    /// `CREATE VIEW ... AS SELECT * FROM read_parquet(...)`.
    DuckDb,
    /// `CREATE EXTERNAL TABLE ... STORED AS PARQUET LOCATION ...`.
    DataFusion,
}

impl SqlDialect {
    /// The tag naming this dialect in a `-- @engine` section header.
    pub fn tag(&self) -> &'static str {
        match self {
            SqlDialect::DuckDb => "duckdb",
            SqlDialect::DataFusion => "datafusion",
        }
    }
}

impl Display for SqlDialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.tag())
    }
}

/// Marks the start of a dialect's section within a `create.sql`.
const SECTION_HEADER: &str = "-- @engine ";

/// A benchmark's `create.sql`, parsed into per-dialect sections.
///
/// Statements still contain their placeholders; [`CreateScript::render`] substitutes them for a
/// specific format and object kind.
#[derive(Debug)]
pub struct CreateScript {
    path: PathBuf,
    pub(super) sections: Vec<(String, String)>,
}

impl CreateScript {
    /// Load `benchmark`'s create script, or `None` when it has no checked-in file.
    ///
    /// A benchmark whose table list is only known at runtime — Public BI's per-dataset tables,
    /// SpatialBench's optional `zone` — legitimately has no file; the caller falls back to
    /// statements generated from [`Benchmark::table_specs`].
    pub fn load(benchmark: &dyn Benchmark) -> anyhow::Result<Option<Self>> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(benchmark.create_sql_path());
        if !path.exists() {
            return Ok(None);
        }

        let text = fs::read_to_string(&path)?;
        Ok(Some(Self::parse(path, &text)))
    }

    /// Split `text` into `-- @engine <tag>` sections. Text before the first header is preamble and
    /// belongs to no dialect.
    pub(super) fn parse(path: PathBuf, text: &str) -> Self {
        let mut sections: Vec<(String, String)> = Vec::new();

        for line in text.lines() {
            match line.trim().strip_prefix(SECTION_HEADER) {
                Some(tag) => sections.push((tag.trim().to_string(), String::new())),
                None => {
                    if let Some((_, body)) = sections.last_mut() {
                        body.push_str(line);
                        body.push('\n');
                    }
                }
            }
        }

        Self { path, sections }
    }

    /// The statements registering `benchmark`'s tables in `dialect`, ready to execute.
    ///
    /// Returns `None` when the script has no section for `dialect`, which means that engine
    /// registers the benchmark some other way.
    pub fn render(
        &self,
        dialect: SqlDialect,
        benchmark: &dyn Benchmark,
        registration: Registration,
    ) -> anyhow::Result<Option<Vec<String>>> {
        let Some((_, body)) = self.sections.iter().find(|(tag, _)| tag == dialect.tag()) else {
            return Ok(None);
        };

        let load_format = registration.load_format;
        let dir = format_dir(benchmark, load_format)?;
        let rendered = body
            .replace("{object}", registration.object.sql_keyword())
            .replace("{dir}", &dir)
            .replace("{ext}", load_format.ext())
            .replace("{read}", read_function(load_format))
            .replace("{format}", stored_as(load_format)?);

        let statements = split_statements(&rendered);
        if statements.is_empty() {
            anyhow::bail!(
                "{} has an empty `{SECTION_HEADER}{dialect}` section",
                self.path.display()
            );
        }

        Ok(Some(statements))
    }

    /// The table names this script registers, in the given dialect. Used by tests to check the
    /// script against [`Benchmark::table_specs`].
    #[cfg(test)]
    pub fn table_names(&self, dialect: SqlDialect) -> Vec<String> {
        let Some((_, body)) = self.sections.iter().find(|(tag, _)| tag == dialect.tag()) else {
            return Vec::new();
        };

        split_statements(body)
            .iter()
            .filter_map(|stmt| {
                // `CREATE {object} IF NOT EXISTS <name> ...` / `CREATE EXTERNAL TABLE IF NOT
                // EXISTS <name> ...`; the name is the token after `EXISTS`.
                let mut tokens = stmt.split_whitespace();
                while let Some(token) = tokens.next() {
                    if token.eq_ignore_ascii_case("EXISTS") {
                        return tokens.next().map(str::to_string);
                    }
                }
                None
            })
            .collect()
    }
}

/// Split a rendered section into executable statements, dropping comments and blank lines.
///
/// Statements are separated by `;`, so a comment must never contain one — the same constraint the
/// query files already carry.
fn split_statements(body: &str) -> Vec<String> {
    body.split(';')
        .map(|stmt| {
            stmt.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with("--"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|stmt| !stmt.is_empty())
        .collect()
}

/// The directory holding `format`'s files for this benchmark, as a plain path or URL without a
/// trailing slash.
fn format_dir(benchmark: &dyn Benchmark, format: Format) -> anyhow::Result<String> {
    let url = benchmark.format_path(format, benchmark.data_url())?;
    let url = url.as_str();
    Ok(url
        .strip_prefix("file://")
        .unwrap_or(url)
        .trim_end_matches('/')
        .to_string())
}

/// DuckDB's table function for reading `format`'s files.
fn read_function(format: Format) -> &'static str {
    match format {
        Format::OnDiskVortex | Format::VortexCompact | Format::VortexNative => "read_vortex",
        _ => "read_parquet",
    }
}

/// DataFusion's `STORED AS` name for `format`.
fn stored_as(format: Format) -> anyhow::Result<&'static str> {
    match format {
        Format::Parquet => Ok("PARQUET"),
        Format::Csv => Ok("CSV"),
        Format::OnDiskVortex | Format::VortexCompact | Format::VortexNative => Ok("VORTEX"),
        format => anyhow::bail!("{format} has no DataFusion `STORED AS` name"),
    }
}

/// Statements registering `benchmark`'s tables, generated from [`Benchmark::table_specs`].
///
/// The fallback for benchmarks with no checked-in `create.sql`.
pub fn generated_statements(
    dialect: SqlDialect,
    benchmark: &dyn Benchmark,
    registration: Registration,
) -> anyhow::Result<Vec<String>> {
    let Registration {
        object,
        load_format,
    } = registration;

    benchmark
        .table_specs()
        .into_iter()
        .map(|spec| {
            let source = super::TableSource::resolve(benchmark, spec, load_format, object)?;
            match dialect {
                SqlDialect::DuckDb => Ok(source.duckdb_registration_sql()),
                SqlDialect::DataFusion => Ok(format!(
                    "CREATE EXTERNAL TABLE IF NOT EXISTS {name} STORED AS {format} LOCATION \
                     '{dir}/{glob}'",
                    name = source.name,
                    format = stored_as(load_format)?,
                    dir = source.base_dir(),
                    glob = source.glob(),
                )),
            }
        })
        .collect()
}
