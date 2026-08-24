// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Convert Parquet files to Vortex format.
//!
//! The input is a local path or a URL. URLs resolve through [`vortex::cloud::Registry`], so
//! `s3://`, `gs://`, `az://`, `http(s)://` and `hf://` all read without downloading the file
//! first: Parquet's async reader issues ranged requests against the object store, and only the
//! Vortex output is written to disk.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use anyhow::anyhow;
use clap::Parser;
use clap::ValueEnum;
use futures::StreamExt;
use indicatif::ProgressBar;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::registry::ObjectStoreRegistry;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::async_reader::ParquetObjectReader;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use url::Url;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::cloud::Registry;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::error::VortexExpect;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;

/// Compression strategy to use when converting Parquet files to Vortex format.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum Strategy {
    /// Use the BtrBlocks compressor strategy (default)
    #[default]
    Btrblocks,
    /// Use the Compact compression strategy for more aggressive compression.
    Compact,
}

/// Command-line flags for the convert command.
#[derive(Debug, Clone, Parser)]
pub struct ConvertArgs {
    /// Parquet file to convert: a local path, or a URL such as
    /// `s3://bucket/key.parquet` or `hf://datasets/owner/name/data/train.parquet`.
    pub file: String,

    /// Path of the Vortex file to write.
    ///
    /// Defaults to the input with a `.vortex` extension, in the working directory for a URL.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Compression strategy.
    #[arg(short, long, default_value = "btrblocks")]
    pub strategy: Strategy,

    /// Execute quietly. No output will be printed.
    #[arg(short, long)]
    pub quiet: bool,
}

/// Where a conversion reads from.
#[derive(Debug, Clone)]
pub enum Source {
    /// A file on the local filesystem.
    Local(PathBuf),
    /// An object store URL.
    Remote(Url),
}

impl Source {
    /// Classifies `input` as a URL or a local path.
    ///
    /// A single-letter scheme is treated as a path so a Windows `C:\data\x.parquet` is not read
    /// as a URL with scheme `c`. `file://` URLs resolve back to a local path.
    pub fn parse(input: &str) -> Self {
        match Url::parse(input) {
            Ok(url) if url.scheme() == "file" => url
                .to_file_path()
                .map_or_else(|()| Source::Local(PathBuf::from(input)), Source::Local),
            Ok(url) if url.scheme().len() > 1 => Source::Remote(url),
            _ => Source::Local(PathBuf::from(input)),
        }
    }

    /// The local path this source reads, if it is local.
    pub fn local_path(&self) -> Option<&Path> {
        match self {
            Source::Local(path) => Some(path),
            Source::Remote(_) => None,
        }
    }

    /// The default output path: the input's file name with a `.vortex` extension.
    fn default_output(&self) -> anyhow::Result<PathBuf> {
        match self {
            Source::Local(path) => Ok(path.with_extension("vortex")),
            Source::Remote(url) => {
                let name = url
                    .path_segments()
                    .and_then(|mut segments| segments.next_back())
                    .filter(|segment| !segment.is_empty())
                    .ok_or_else(|| anyhow!("{url} has no file name; pass --output"))?;
                Ok(PathBuf::from(name).with_extension("vortex"))
            }
        }
    }
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Source::Local(path) => write!(f, "{}", path.display()),
            Source::Remote(url) => write!(f, "{url}"),
        }
    }
}

/// Whether `input` names an object store URL rather than a local path.
pub fn is_url(input: &str) -> bool {
    Source::parse(input).local_path().is_none()
}

/// The batch size of the record batches.
pub const BATCH_SIZE: usize = 8192;

/// Convert Parquet files to Vortex.
///
/// # Errors
///
/// Returns an error if the input cannot be read or the output file cannot be written.
pub async fn exec_convert(session: &VortexSession, flags: ConvertArgs) -> anyhow::Result<()> {
    let source = Source::parse(&flags.file);
    let output = match flags.output.clone() {
        Some(output) => output,
        None => source.default_output()?,
    };

    if !flags.quiet {
        eprintln!("Converting input Parquet file: {source}");
    }

    match &source {
        Source::Local(path) => {
            let file = File::open(path)
                .await
                .with_context(|| format!("opening {}", path.display()))?;
            convert(session, file, &output, &flags).await
        }
        Source::Remote(url) => {
            let (store, path) = Registry::new()
                .resolve(url)
                .with_context(|| format!("resolving {url}"))?;
            let reader = object_reader(store, path, url).await?;
            convert(session, reader, &output, &flags).await
        }
    }
}

/// Builds a Parquet reader over an object store path.
///
/// The object size is fetched up front: Parquet reads its footer from the end of the file, so the
/// reader needs the length before it can issue that range request.
async fn object_reader(
    store: Arc<dyn ObjectStore>,
    path: object_store::path::Path,
    url: &Url,
) -> anyhow::Result<ParquetObjectReader> {
    let meta = store
        .head(&path)
        .await
        .with_context(|| format!("reading metadata for {url}"))?;
    Ok(ParquetObjectReader::new(store, path).with_file_size(meta.size))
}

/// Streams `reader` into a Vortex file at `output`.
async fn convert<R>(
    session: &VortexSession,
    reader: R,
    output: &Path,
    flags: &ConvertArgs,
) -> anyhow::Result<()>
where
    R: AsyncFileReader + Unpin + Send + 'static,
{
    let parquet = ParquetRecordBatchStreamBuilder::new(reader)
        .await?
        .with_batch_size(BATCH_SIZE);
    let num_rows = parquet.metadata().file_metadata().num_rows();

    let dtype = session
        .arrow()
        .from_arrow_schema(parquet.schema().as_ref())?;
    let arrow_session = ArrowSession::clone(&session.arrow());
    let mut vortex_stream = parquet
        .build()?
        .map(move |record_batch| {
            record_batch
                .map_err(|e| vortex_err!(External: e))
                .and_then(|rb| {
                    let schema = rb.schema();
                    arrow_session.from_arrow_record_batch(rb, &schema)
                })
        })
        .boxed();

    if !flags.quiet {
        // Parquet reader returns batches, rather than row groups. So make sure we correctly
        // configure the progress bar.
        let nbatches = u64::try_from(num_rows)
            .vortex_expect("negative row count?")
            .div_ceil(BATCH_SIZE as u64);
        vortex_stream = ProgressBar::new(nbatches)
            .wrap_stream(vortex_stream)
            .boxed();
    }

    let mut strategy = WriteStrategyBuilder::default();
    if matches!(flags.strategy, Strategy::Compact) {
        strategy =
            strategy.with_btrblocks_builder(BtrBlocksCompressorBuilder::default().with_compact());
    }

    let mut file = File::create(output)
        .await
        .with_context(|| format!("creating {}", output.display()))?;
    session
        .write_options()
        .with_strategy(strategy.build())
        .write(&mut file, ArrayStreamAdapter::new(dtype, vortex_stream))
        .await?;
    file.shutdown().await?;

    if !flags.quiet {
        eprintln!("Wrote {}", output.display());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_local_paths() {
        for input in ["data.parquet", "./a/b.parquet", "/tmp/x.parquet"] {
            let source = Source::parse(input);
            assert_eq!(source.local_path(), Some(Path::new(input)), "{input}");
        }
    }

    #[test]
    fn parses_a_windows_path_as_local_not_a_url() {
        let source = Source::parse(r"C:\data\x.parquet");
        assert_eq!(source.local_path(), Some(Path::new(r"C:\data\x.parquet")));
    }

    #[test]
    fn is_url_agrees_with_source_parse() {
        assert!(is_url("s3://bucket/key.parquet"));
        assert!(!is_url("data.parquet"));
        assert!(!is_url("file:///tmp/x.parquet"));
    }

    #[test]
    fn parses_object_store_urls_as_remote() {
        for input in [
            "s3://bucket/key.parquet",
            "gs://bucket/key.parquet",
            "hf://datasets/owner/name/data/train.parquet",
            "https://example.com/x.parquet",
        ] {
            assert!(Source::parse(input).local_path().is_none(), "{input}");
        }
    }

    #[test]
    fn file_urls_resolve_to_local_paths() {
        let source = Source::parse("file:///tmp/x.parquet");
        assert_eq!(source.local_path(), Some(Path::new("/tmp/x.parquet")));
    }

    #[test]
    fn default_output_replaces_the_extension() -> anyhow::Result<()> {
        assert_eq!(
            Source::parse("/tmp/data.parquet").default_output()?,
            PathBuf::from("/tmp/data.vortex")
        );
        assert_eq!(
            Source::parse("s3://bucket/nested/data.parquet").default_output()?,
            PathBuf::from("data.vortex")
        );
        assert_eq!(
            Source::parse("hf://datasets/owner/name/data/train.parquet").default_output()?,
            PathBuf::from("train.vortex")
        );
        Ok(())
    }

    #[test]
    fn default_output_needs_a_file_name() {
        assert!(Source::parse("s3://bucket/").default_output().is_err());
    }
}
