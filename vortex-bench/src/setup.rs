// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dataset materialization: the [`SetupCtx`] handed to [`Benchmark::setup`] and the driver
//! that turns natively-produced formats into every requested one.
//!
//! # Model
//!
//! A benchmark declares the formats it can produce without help via
//! [`Benchmark::native_formats`], and materializes one of them in
//! [`Benchmark::setup`]. Everything else is derived by [`prepare_data`].
//!
//! Parquet is the pivot format because it is the only one that can be derived *from*:
//! [`crate::conversions`] converts Parquet to Vortex, but nothing converts Vortex back to
//! Parquet. A benchmark that produces only Vortex therefore cannot serve a Parquet run.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use parking_lot::Mutex;
use reqwest::Client;
use tracing::info;

use crate::Benchmark;
use crate::CompactionStrategy;
use crate::Format;
use crate::conversions::convert_parquet_directory_to_vortex;
use crate::datasets::data_downloads::download_many;
use crate::datasets::data_downloads::http_client;

/// A parquet (or other native-format) file produced by [`Benchmark::setup`], tagged with the
/// table it belongs to.
///
/// Tables are many-to-one with files: ClickBench emits 100 files for the single `hits` table,
/// while Appian emits nine files for nine tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Emitted {
    pub table: String,
    pub path: PathBuf,
}

/// Context handed to [`Benchmark::setup`].
///
/// Owns the staging directory the benchmark writes into, the shared download pool, and the
/// list of files the benchmark has produced so far.
pub struct SetupCtx {
    staging: PathBuf,
    emitted: Mutex<Vec<Emitted>>,
}

impl SetupCtx {
    /// Create a context staging into `staging`, creating the directory if needed.
    ///
    /// `staging` is deliberately stable across runs rather than a fresh temp dir, so a failure
    /// partway through a multi-file dataset leaves the already-fetched files in place for the
    /// next attempt.
    pub fn new(staging: impl Into<PathBuf>) -> Result<Self> {
        let staging = staging.into();
        std::fs::create_dir_all(&staging)?;
        Ok(Self {
            staging,
            emitted: Mutex::new(Vec::new()),
        })
    }

    /// Directory this benchmark should write its output into.
    pub fn staging(&self) -> &Path {
        &self.staging
    }

    /// Idempotently fetch `(url, path)` pairs through the shared download pool.
    ///
    /// Pass every file the benchmark needs in one call: the pool ramps concurrency across the
    /// whole batch and renders a single progress block. Files already on disk are skipped.
    pub async fn download<I, S, P>(&self, files: I) -> Result<Vec<PathBuf>>
    where
        I: IntoIterator<Item = (S, P)>,
        S: Into<String>,
        P: Into<PathBuf>,
    {
        // `download_many` takes `(path, url)`; flip so call sites read `(url, path)`.
        let downloads: Vec<(PathBuf, String)> = files
            .into_iter()
            .map(|(url, path)| (path.into(), url.into()))
            .collect();
        download_many(downloads).await
    }

    /// The shared HTTP client, for datasets that must consume a response stream rather than
    /// land a file — see [`crate::statpopgen`], which decodes a multi-hundred-gigabyte VCF on
    /// the fly and stops after a fixed row count.
    pub fn http(&self) -> &'static Client {
        http_client()
    }

    /// Register a file this benchmark produced as (part of) `table`.
    pub fn emit(&self, table: impl Into<String>, path: impl Into<PathBuf>) {
        self.emitted.lock().push(Emitted {
            table: table.into(),
            path: path.into(),
        });
    }

    /// Every file registered via [`SetupCtx::emit`], in emission order.
    pub fn emitted(&self) -> Vec<Emitted> {
        self.emitted.lock().clone()
    }
}

/// Materialize `formats` for `benchmark`.
///
/// Formats the benchmark lists in [`Benchmark::native_formats`] are produced directly by
/// [`Benchmark::setup`]. The Vortex formats are derived from Parquet when the benchmark does
/// not produce them natively. Benchmarks whose data already lives remotely (a non-`file://`
/// [`Benchmark::data_url`]) are skipped entirely.
pub async fn prepare_data(benchmark: &dyn Benchmark, formats: &[Format]) -> Result<()> {
    if benchmark.data_url().scheme() != "file" {
        info!(
            "{}: data is remote ({}), nothing to materialize",
            benchmark.dataset_name(),
            benchmark.data_url(),
        );
        return Ok(());
    }

    let base_path = benchmark
        .data_url()
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", benchmark.data_url()))?;

    let native = benchmark.native_formats();

    // Parquet underpins every derived format, so produce it if anything needs it.
    let needs_parquet = formats
        .iter()
        .any(|f| !native.contains(f) || *f == Format::Parquet);
    if needs_parquet && native.contains(&Format::Parquet) {
        setup_format(benchmark, &base_path, Format::Parquet).await?;
    }

    for &format in formats {
        // Lance and DuckDB are built by their own bench binaries from the Parquet above.
        if matches!(format, Format::Lance | Format::OnDiskDuckDB | Format::Csv) {
            continue;
        }

        match plan(native, format)
            .with_context(|| format!("benchmark {}", benchmark.dataset_name()))?
        {
            Plan::Native(Format::Parquet) => {
                // Already produced above.
            }
            Plan::Native(format) => setup_format(benchmark, &base_path, format).await?,
            Plan::DeriveFromParquet(format, compaction) => {
                convert_parquet_directory_to_vortex(&base_path, compaction).await?;
                benchmark.prepare_format(format, &base_path).await?;
            }
        }
    }

    Ok(())
}

/// Run `benchmark`'s setup for one natively-produced format, staging into `<base>/<format>/`.
async fn setup_format(benchmark: &dyn Benchmark, base_path: &Path, format: Format) -> Result<()> {
    let staging = base_path.join(format.name());
    let ctx = SetupCtx::new(&staging)?;

    benchmark.setup(&ctx, format).await?;

    let emitted = ctx.emitted();
    info!(
        "{}: setup produced {} {format} file(s) in {}",
        benchmark.dataset_name(),
        emitted.len(),
        staging.display(),
    );

    benchmark.prepare_format(format, base_path).await?;
    Ok(())
}

/// Which formats a benchmark produces natively and which are derived from Parquet.
///
/// Split out so the routing is testable without running a download or a generator.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Plan {
    /// `setup` is called for this format directly.
    Native(Format),
    /// Parquet is produced, then converted with this strategy.
    DeriveFromParquet(Format, CompactionStrategy),
}

pub(crate) fn plan(native: &[Format], requested: Format) -> Result<Plan> {
    if native.contains(&requested) {
        return Ok(Plan::Native(requested));
    }
    if !native.contains(&Format::Parquet) {
        bail!("cannot produce {requested}: not native ({native:?}) and Parquet is unavailable");
    }
    match requested {
        Format::OnDiskVortex => Ok(Plan::DeriveFromParquet(
            requested,
            CompactionStrategy::Default,
        )),
        Format::VortexCompact => Ok(Plan::DeriveFromParquet(
            requested,
            CompactionStrategy::Compact,
        )),
        other => bail!("cannot derive {other} from Parquet"),
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    /// StatPopGen streams HTTP straight to Parquet, then Vortex comes off that Parquet — the
    /// case that a URL-to-path download descriptor could not express.
    #[test]
    fn streaming_only_suite_derives_vortex_from_its_parquet() -> Result<()> {
        let native = [Format::Parquet];
        assert_eq!(
            plan(&native, Format::Parquet)?,
            Plan::Native(Format::Parquet)
        );
        assert_eq!(
            plan(&native, Format::OnDiskVortex)?,
            Plan::DeriveFromParquet(Format::OnDiskVortex, CompactionStrategy::Default),
        );
        Ok(())
    }

    /// TPC-H writes Vortex from its row generator, so it must not round-trip through Parquet.
    #[test]
    fn generator_that_writes_vortex_natively_skips_the_parquet_round_trip() -> Result<()> {
        let native = [Format::Parquet, Format::OnDiskVortex];
        assert_eq!(
            plan(&native, Format::OnDiskVortex)?,
            Plan::Native(Format::OnDiskVortex),
        );
        Ok(())
    }

    /// Parquet is the only pivot: nothing converts Vortex back, so a Vortex-only suite
    /// asking for Parquet is an error rather than a silent no-op.
    #[test]
    fn vortex_only_suite_cannot_produce_parquet() {
        let native = [Format::OnDiskVortex];
        assert!(plan(&native, Format::Parquet).is_err());
        assert!(plan(&native, Format::Lance).is_err());
    }

    #[test]
    fn emit_records_table_and_path_in_order() -> Result<()> {
        let dir = tempdir()?;
        let ctx = SetupCtx::new(dir.path())?;

        ctx.emit("hits", dir.path().join("hits_0.parquet"));
        ctx.emit("hits", dir.path().join("hits_1.parquet"));

        let emitted = ctx.emitted();
        assert_eq!(emitted.len(), 2);
        assert!(emitted.iter().all(|e| e.table == "hits"));
        assert_eq!(emitted[0].path, dir.path().join("hits_0.parquet"));
        assert_eq!(emitted[1].path, dir.path().join("hits_1.parquet"));
        Ok(())
    }

    #[test]
    fn new_creates_the_staging_directory() -> Result<()> {
        let dir = tempdir()?;
        let staging = dir.path().join("nested").join("parquet");
        let ctx = SetupCtx::new(&staging)?;
        assert!(staging.is_dir());
        assert_eq!(ctx.staging(), staging);
        Ok(())
    }
}
