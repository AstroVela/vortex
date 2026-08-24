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
use std::sync::LazyLock;

use anyhow::Context;
use clap::Parser;
use clap::ValueEnum;
use futures::StreamExt;
use indicatif::ProgressBar;
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
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::file::multi::parse_uri_or_path;
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

/// Stores resolved from URLs, cached per bucket for the process.
///
/// Every other binding holds the registry this way — see `vortex-python`'s `resolve_store` and
/// `vortex-duckdb`'s `resolve_filesystem`.
static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

/// Where a conversion reads from.
enum Source {
    /// A file on the local filesystem.
    Local(PathBuf),
    /// An object store URL.
    Remote(Url),
}

impl Source {
    /// Classifies `input` as a URL or a local path.
    ///
    /// Delegates to [`parse_uri_or_path`], the parser the language bindings share, so the CLI
    /// agrees with them on what counts as a URL — including that a Windows `C:\data\x.parquet`
    /// is a path and not a URL with scheme `c`.
    fn parse(input: &str) -> VortexResult<Self> {
        let url = parse_uri_or_path(input)?;
        if url.scheme() != "file" {
            return Ok(Source::Remote(url));
        }
        url.to_file_path()
            .map(Source::Local)
            .map_err(|()| vortex_err!("not a local file path: {url}"))
    }

    /// The default output path: the input's file name with a `.vortex` extension.
    fn default_output(&self) -> PathBuf {
        match self {
            Source::Local(path) => path.with_extension("vortex"),
            // A URL with no final segment is a directory, which the reader rejects moments
            // later with a better message than anything this could say.
            Source::Remote(url) => PathBuf::from(
                url.path()
                    .rsplit('/')
                    .find(|segment| !segment.is_empty())
                    .unwrap_or("output"),
            )
            .with_extension("vortex"),
        }
    }
}

/// The batch size of the record batches.
pub const BATCH_SIZE: usize = 8192;

/// How much of a remote file's tail to fetch when looking for the Parquet footer.
///
/// Large enough to carry the whole footer for a typical file, so the metadata arrives in one
/// request instead of a length fetch followed by a body fetch.
const FOOTER_HINT: usize = 64 * 1024;

/// Convert Parquet files to Vortex.
///
/// # Errors
///
/// Returns an error if the input cannot be read or the output file cannot be written.
pub async fn exec_convert(session: &VortexSession, flags: ConvertArgs) -> anyhow::Result<()> {
    let source = Source::parse(&flags.file)?;
    let output = flags
        .output
        .clone()
        .unwrap_or_else(|| source.default_output());

    if !flags.quiet {
        eprintln!("Converting input Parquet file: {}", flags.file);
    }

    match &source {
        Source::Local(path) => {
            let file = File::open(path)
                .await
                .with_context(|| format!("opening {}", path.display()))?;
            convert(session, file, &output, &flags).await
        }
        Source::Remote(url) => {
            let (store, path) = REGISTRY
                .resolve(url)
                .with_context(|| format!("resolving {url}"))?;
            // Two deliberate choices, each saving a round trip on a one-shot command:
            // no `head` first, because given no file size the reader locates the footer with a
            // suffix range request; and a footer hint large enough that the metadata usually
            // arrives with it, rather than in a second fetch after the 8-byte length.
            let reader = ParquetObjectReader::new(store, path).with_footer_size_hint(FOOTER_HINT);
            convert(session, reader, &output, &flags).await
        }
    }
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

    /// The local path `input` resolves to, or `None` if it is a URL.
    fn local(input: &str) -> Option<PathBuf> {
        match Source::parse(input).ok()? {
            Source::Local(path) => Some(path),
            Source::Remote(_) => None,
        }
    }

    #[test]
    fn relative_paths_resolve_against_the_working_directory() -> anyhow::Result<()> {
        let path = local("./a/b.parquet").ok_or_else(|| anyhow::anyhow!("expected a path"))?;
        assert!(path.is_absolute());
        assert!(path.ends_with("a/b.parquet"));
        Ok(())
    }

    #[test]
    fn absolute_and_file_url_inputs_agree() {
        assert_eq!(
            local("/tmp/x.parquet"),
            Some(PathBuf::from("/tmp/x.parquet"))
        );
        assert_eq!(
            local("file:///tmp/x.parquet"),
            Some(PathBuf::from("/tmp/x.parquet"))
        );
    }

    #[test]
    fn object_store_urls_are_remote() {
        for input in [
            "s3://bucket/key.parquet",
            "gs://bucket/key.parquet",
            "hf://datasets/owner/name/data/train.parquet",
            "https://example.com/x.parquet",
        ] {
            assert_eq!(local(input), None, "{input}");
        }
    }

    #[test]
    fn default_output_replaces_the_extension() -> anyhow::Result<()> {
        assert_eq!(
            Source::parse("/tmp/data.parquet")?.default_output(),
            PathBuf::from("/tmp/data.vortex")
        );
        assert_eq!(
            Source::parse("s3://bucket/nested/data.parquet")?.default_output(),
            PathBuf::from("data.vortex")
        );
        assert_eq!(
            Source::parse("hf://datasets/owner/name/data/train.parquet")?.default_output(),
            PathBuf::from("train.vortex")
        );
        Ok(())
    }

    #[test]
    fn default_output_falls_back_when_a_url_has_no_file_name() -> anyhow::Result<()> {
        assert_eq!(
            Source::parse("s3://bucket/")?.default_output(),
            PathBuf::from("output.vortex")
        );
        Ok(())
    }
}
