// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Re-encode one or more Parquet files into a single Vortex file and compare the two on
//! on-disk size and full-scan decompression throughput.
//!
//! Unlike `compress-bench`, which walks a fixed dataset list and round-trips each one through
//! memory, this binary takes arbitrary Parquet paths and streams both the conversion and the
//! scans, so it can be pointed at inputs far larger than RAM.
//!
//! Build it with `--features unstable_encodings` to write the file with the preview edition
//! encodings (OnPair strings, Zstd buffer compression, Delta integers).

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::bail;
use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_schema::DataType;
use arrow_schema::Schema;
use clap::Parser;
use futures::StreamExt;
use futures::pin_mut;
use futures::stream;
use parquet::arrow::AsyncArrowWriter;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::ProjectionMask;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::FieldNames;
use vortex::error::vortex_err;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_bench::SESSION;

/// Command-line flags.
#[derive(Debug, Parser)]
#[command(
    version,
    about = "Compare Parquet and Vortex size and decode throughput"
)]
struct Args {
    /// Parquet files to read, concatenated in the order given.
    #[arg(long, required = true, num_args = 1..)]
    parquet: Vec<PathBuf>,

    /// Path of the Vortex file to write, and to scan when benchmarking.
    #[arg(long)]
    vortex: PathBuf,

    /// Reuse an existing Vortex file instead of re-encoding.
    #[arg(long)]
    skip_convert: bool,

    /// Timed scans per format. The fastest run is reported.
    #[arg(long, default_value_t = 3)]
    iterations: usize,

    /// Record batch size used by both readers.
    #[arg(long, default_value_t = 8192)]
    batch_size: usize,

    /// Add the compact schemes (Zstd for strings and binary) to the compressor.
    #[arg(long)]
    compact: bool,

    /// Also time a separate single-column scan for every top-level column.
    #[arg(long)]
    per_column: bool,

    /// Check that both formats decode to identical values before timing anything.
    #[arg(long)]
    verify: bool,

    /// Also re-encode the input as Parquet at each of these Zstd levels, and measure it.
    ///
    /// The source files ship as Snappy, so comparing only against them conflates the format
    /// with the codec. Each level is written to a sibling of `--vortex`.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    parquet_zstd: Vec<i32>,
}

/// What one timed scan observed.
#[derive(Clone, Copy)]
struct ScanResult {
    elapsed: Duration,
    rows: usize,
    /// Size of the decoded Arrow batches, which is the volume decompression produced.
    decoded_bytes: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    for path in &args.parquet {
        if !path.exists() {
            bail!("parquet file does not exist: {}", path.display());
        }
    }

    let parquet_bytes = total_size(&args.parquet)?;

    if !args.skip_convert {
        let elapsed = convert(&args).await?;
        println!(
            "converted {} parquet file(s) in {:.1}s",
            args.parquet.len(),
            elapsed.as_secs_f64()
        );
    }

    let vortex_bytes = std::fs::metadata(&args.vortex)
        .with_context(|| format!("stat {}", args.vortex.display()))?
        .len();

    if args.verify {
        verify(&args).await?;
    }

    let parquet_scan = best_scan(args.iterations, || {
        scan_parquet(&args.parquet, args.batch_size, None)
    })
    .await?;
    let vortex_scan = best_scan(args.iterations, || scan_vortex(&args, None)).await?;

    if parquet_scan.rows != vortex_scan.rows {
        bail!(
            "row count mismatch: parquet {} vs vortex {}",
            parquet_scan.rows,
            vortex_scan.rows
        );
    }

    report(
        &args,
        parquet_bytes,
        vortex_bytes,
        parquet_scan,
        vortex_scan,
    );

    if args.per_column {
        report_per_column(&args).await?;
    }

    if !args.parquet_zstd.is_empty() {
        report_parquet_zstd(&args, parquet_bytes, vortex_bytes, vortex_scan).await?;
    }

    Ok(())
}

/// Re-encode the input as Zstd Parquet at each requested level and measure size and decode.
///
/// The rewrite also drops the source files' ~1k-row row groups for the writer default, so this
/// is a well-configured Parquet baseline rather than a codec swap alone.
async fn report_parquet_zstd(
    args: &Args,
    source_bytes: u64,
    vortex_bytes: u64,
    vortex_scan: ScanResult,
) -> anyhow::Result<()> {
    println!("\n=== parquet re-encoded with zstd ===");
    println!(
        "  {:<10} {:>12} {:>10} {:>12} {:>12} {:>10}",
        "level", "size", "vs source", "encode", "decode", "vs vortex"
    );

    for &level in &args.parquet_zstd {
        let out = args
            .vortex
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!("parquet-zstd{level}.parquet"));

        let encode = encode_parquet_zstd(args, level, &out).await?;
        let bytes = std::fs::metadata(&out)?.len();
        let scan = best_scan(args.iterations, || {
            scan_parquet(std::slice::from_ref(&out), args.batch_size, None)
        })
        .await?;

        println!(
            "  zstd-{:<5} {:>12} {:>9.1}% {:>11.1}s {:>11.2}s {:>9.2}x",
            level,
            human_bytes(bytes as f64),
            (bytes as f64 / source_bytes as f64 - 1.0) * 100.0,
            encode.as_secs_f64(),
            scan.elapsed.as_secs_f64(),
            scan.elapsed.as_secs_f64() / vortex_scan.elapsed.as_secs_f64(),
        );
    }

    println!(
        "  (vortex file for reference: {}, decode {:.2}s)",
        human_bytes(vortex_bytes as f64),
        vortex_scan.elapsed.as_secs_f64()
    );

    Ok(())
}

/// Stream every input Parquet file into one Zstd-compressed Parquet file, returning encode time.
async fn encode_parquet_zstd(args: &Args, level: i32, out: &Path) -> anyhow::Result<Duration> {
    let mut builders = Vec::with_capacity(args.parquet.len());
    for path in &args.parquet {
        let file = File::open(path).await?;
        builders.push(
            ParquetRecordBatchStreamBuilder::new(file)
                .await?
                .with_batch_size(args.batch_size),
        );
    }
    let schema = Arc::clone(builders[0].schema());

    let mut readers = Vec::with_capacity(builders.len());
    for builder in builders {
        readers.push(builder.build()?);
    }
    let batches = stream::iter(readers).flatten();
    pin_mut!(batches);

    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(level)?))
        .build();

    let start = Instant::now();
    let file = File::create(out).await?;
    let mut writer = AsyncArrowWriter::try_new(file, schema, Some(properties))?;
    while let Some(batch) = batches.next().await {
        writer.write(&batch?).await?;
    }
    writer.close().await?;

    Ok(start.elapsed())
}

/// Sum the on-disk size of every input file.
fn total_size(paths: &[PathBuf]) -> anyhow::Result<u64> {
    paths
        .iter()
        .map(|p| {
            std::fs::metadata(p)
                .map(|m| m.len())
                .with_context(|| format!("stat {}", p.display()))
        })
        .sum()
}

/// Stream every input Parquet file into a single Vortex file, returning the encode wall time.
async fn convert(args: &Args) -> anyhow::Result<Duration> {
    let mut builders = Vec::with_capacity(args.parquet.len());
    for path in &args.parquet {
        let file = File::open(path)
            .await
            .with_context(|| format!("open {}", path.display()))?;
        let builder = ParquetRecordBatchStreamBuilder::new(file)
            .await
            .with_context(|| format!("read parquet metadata from {}", path.display()))?
            .with_batch_size(args.batch_size);
        builders.push(builder);
    }

    let schema = Arc::clone(builders[0].schema());
    for (path, builder) in args.parquet.iter().zip(&builders).skip(1) {
        if builder.schema().fields() != schema.fields() {
            bail!(
                "{} has a different schema to the first input",
                path.display()
            );
        }
    }

    let dtype = SESSION.arrow().from_arrow_schema(schema.as_ref())?;
    let arrow_session = ArrowSession::clone(&SESSION.arrow());

    let mut readers = Vec::with_capacity(builders.len());
    for builder in builders {
        readers.push(builder.build()?);
    }

    let batches = stream::iter(readers).flatten().map(move |record_batch| {
        record_batch
            .map_err(|e| vortex_err!(External: e))
            .and_then(|rb| {
                let batch_schema = rb.schema();
                arrow_session.from_arrow_record_batch(rb, &batch_schema)
            })
    });

    let mut strategy = WriteStrategyBuilder::default();
    if args.compact {
        strategy =
            strategy.with_btrblocks_builder(BtrBlocksCompressorBuilder::default().with_compact());
    }

    let start = Instant::now();
    let mut out = File::create(&args.vortex)
        .await
        .with_context(|| format!("create {}", args.vortex.display()))?;
    SESSION
        .write_options()
        .with_strategy(strategy.build())
        .write(&mut out, ArrayStreamAdapter::new(dtype, batches.boxed()))
        .await?;
    out.shutdown().await?;

    Ok(start.elapsed())
}

/// Run a scan `iterations` times plus one warm-up, and keep the fastest run.
///
/// The warm-up leaves both files in the page cache, so the reported time reflects decode work
/// rather than storage bandwidth.
async fn best_scan<F, Fut>(iterations: usize, mut scan: F) -> anyhow::Result<ScanResult>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = anyhow::Result<ScanResult>>,
{
    let mut best = scan().await?;
    for _ in 0..iterations {
        let result = scan().await?;
        if result.elapsed < best.elapsed {
            best = result;
        }
    }
    Ok(best)
}

/// Decode every input Parquet file, optionally projecting a single top-level column.
async fn scan_parquet(
    paths: &[PathBuf],
    batch_size: usize,
    column: Option<usize>,
) -> anyhow::Result<ScanResult> {
    let mut rows = 0;
    let mut decoded_bytes = 0;
    let mut elapsed = Duration::ZERO;

    for path in paths {
        let file = File::open(path).await?;
        let mut builder = ParquetRecordBatchStreamBuilder::new(file)
            .await?
            .with_batch_size(batch_size);
        if let Some(column) = column {
            let mask = ProjectionMask::roots(builder.parquet_schema(), [column]);
            builder = builder.with_projection(mask);
        }
        let reader = builder.build()?;
        pin_mut!(reader);

        let start = Instant::now();
        while let Some(batch) = reader.next().await {
            let batch = batch?;
            rows += batch.num_rows();
            decoded_bytes += batch.get_array_memory_size();
        }
        elapsed += start.elapsed();
    }

    Ok(ScanResult {
        elapsed,
        rows,
        decoded_bytes,
    })
}

/// Decode the Vortex file, optionally projecting a single top-level column.
async fn scan_vortex(args: &Args, column: Option<usize>) -> anyhow::Result<ScanResult> {
    let file = SESSION.open_options().open_path(&args.vortex).await?;
    let mut scan = file.scan()?;
    let dtype = scan.dtype()?;

    if let Some(column) = column {
        let field = dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("vortex file root is not a struct"))?
            .field_name(column)
            .ok_or_else(|| vortex_err!("column {column} out of range"))?
            .to_string();
        let names: FieldNames = [field].into_iter().collect();
        let projection = select(names, root())
            .optimize_recursive(&dtype)?
            .bind(&dtype)?;
        scan = scan.with_projection(projection);
    }

    let schema: Arc<Schema> = Arc::new(SESSION.arrow().to_arrow_schema(&scan.dtype()?)?);
    let stream = scan.into_record_batch_stream(schema)?;
    pin_mut!(stream);

    let mut rows = 0;
    let mut decoded_bytes = 0;
    let start = Instant::now();
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        rows += batch.num_rows();
        decoded_bytes += batch.get_array_memory_size();
    }

    Ok(ScanResult {
        elapsed: start.elapsed(),
        rows,
        decoded_bytes,
    })
}

/// Print the size and whole-file scan comparison.
fn report(
    args: &Args,
    parquet_bytes: u64,
    vortex_bytes: u64,
    parquet_scan: ScanResult,
    vortex_scan: ScanResult,
) {
    println!("\n=== inputs ===");
    for path in &args.parquet {
        println!("  {}", path.display());
    }
    println!("  rows: {}", parquet_scan.rows);
    println!(
        "  decoded (arrow in-memory): {}",
        human_bytes(parquet_scan.decoded_bytes as f64)
    );

    println!("\n=== size ===");
    println!("  parquet: {:>12}", human_bytes(parquet_bytes as f64));
    println!("  vortex:  {:>12}", human_bytes(vortex_bytes as f64));
    println!(
        "  vortex/parquet: {:.4}  ({:+.1}%)",
        vortex_bytes as f64 / parquet_bytes as f64,
        (vortex_bytes as f64 / parquet_bytes as f64 - 1.0) * 100.0
    );

    println!(
        "\n=== full scan, all columns (fastest of {}) ===",
        args.iterations
    );
    print_scan("parquet", parquet_scan);
    print_scan("vortex ", vortex_scan);
    println!(
        "  speedup (parquet/vortex): {:.2}x",
        parquet_scan.elapsed.as_secs_f64() / vortex_scan.elapsed.as_secs_f64()
    );
}

/// Print one format's scan line.
fn print_scan(label: &str, result: ScanResult) {
    let secs = result.elapsed.as_secs_f64();
    println!(
        "  {label}: {:>8.2}s   {:>10} decoded   {:>9}/s   {:>10.1} Mrow/s",
        secs,
        human_bytes(result.decoded_bytes as f64),
        human_bytes(result.decoded_bytes as f64 / secs),
        result.rows as f64 / secs / 1e6,
    );
}

/// Time a single-column scan of each top-level column in both formats.
async fn report_per_column(args: &Args) -> anyhow::Result<()> {
    let file = File::open(&args.parquet[0]).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let names: Vec<String> = builder
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    println!("\n=== per-column scan (fastest of {}) ===", args.iterations);
    println!(
        "  {:<28} {:>10} {:>10} {:>9}",
        "column", "parquet", "vortex", "speedup"
    );

    for (index, name) in names.iter().enumerate() {
        let parquet = best_scan(args.iterations, || {
            scan_parquet(&args.parquet, args.batch_size, Some(index))
        })
        .await?;
        let vortex = best_scan(args.iterations, || scan_vortex(args, Some(index))).await?;
        println!(
            "  {:<28} {:>9.2}s {:>9.2}s {:>8.2}x",
            truncate(name, 28),
            parquet.elapsed.as_secs_f64(),
            vortex.elapsed.as_secs_f64(),
            parquet.elapsed.as_secs_f64() / vortex.elapsed.as_secs_f64(),
        );
    }

    Ok(())
}

/// Shorten `value` to `width` characters so the table columns stay aligned.
fn truncate(value: &str, width: usize) -> String {
    if value.len() <= width {
        value.to_string()
    } else {
        format!("{}…", &value[..width - 1])
    }
}

/// Render a byte count with a binary unit suffix.
///
/// Takes `f64` so callers can pass a computed rate without a lossy narrowing cast.
fn human_bytes(bytes: f64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value:.0} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// Decode both formats once and compare a value-level digest of every column.
///
/// The two readers emit different Arrow string layouts, so the digest hashes logical values
/// rather than buffers; comparing `get_array_memory_size` would only compare representations.
async fn verify(args: &Args) -> anyhow::Result<()> {
    let mut parquet_digest: Vec<u64> = Vec::new();
    let mut parquet_types = Vec::new();
    for path in &args.parquet {
        let file = File::open(path).await?;
        let reader = ParquetRecordBatchStreamBuilder::new(file)
            .await?
            .with_batch_size(args.batch_size)
            .build()?;
        pin_mut!(reader);
        while let Some(batch) = reader.next().await {
            let batch = batch?;
            if parquet_types.is_empty() {
                parquet_types = column_types(&batch);
            }
            hash_batch(&batch, &mut parquet_digest)?;
        }
    }

    let file = SESSION.open_options().open_path(&args.vortex).await?;
    let scan = file.scan()?;
    let schema: Arc<Schema> = Arc::new(SESSION.arrow().to_arrow_schema(&scan.dtype()?)?);
    let stream = scan.into_record_batch_stream(schema)?;
    pin_mut!(stream);
    let mut vortex_digest: Vec<u64> = Vec::new();
    let mut vortex_types = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        if vortex_types.is_empty() {
            vortex_types = column_types(&batch);
        }
        hash_batch(&batch, &mut vortex_digest)?;
    }

    println!("=== verify ===");
    println!("  parquet arrow types: {}", parquet_types.join(", "));
    println!("  vortex  arrow types: {}", vortex_types.join(", "));
    if parquet_digest == vortex_digest {
        let rendered = parquet_digest
            .iter()
            .map(|digest| format!("{digest:#018x}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  values identical per column: {rendered}");
    } else {
        bail!("value mismatch: parquet {parquet_digest:x?} vs vortex {vortex_digest:x?}");
    }

    Ok(())
}

/// Describe each column as "name: arrow_type".
fn column_types(batch: &RecordBatch) -> Vec<String> {
    batch
        .schema()
        .fields()
        .iter()
        .map(|f| format!("{}: {}", f.name(), f.data_type()))
        .collect()
}

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold `bytes` into an FNV-1a digest.
fn hash_bytes(bytes: &[u8], digest: &mut u64) {
    for byte in bytes {
        *digest ^= u64::from(*byte);
        *digest = digest.wrapping_mul(FNV_PRIME);
    }
}

/// Fold every logical value into a per-column digest.
///
/// Digests are kept per column, not per batch, because the two readers choose different batch
/// sizes: a single digest over the concatenated batches would depend on where those boundaries
/// fall. Utf8 and Utf8View hash identically, so matching values produce matching digests.
fn hash_batch(batch: &RecordBatch, digests: &mut Vec<u64>) -> anyhow::Result<()> {
    digests.resize(batch.num_columns(), FNV_OFFSET);
    for (column, digest) in batch.columns().iter().zip(digests.iter_mut()) {
        match column.data_type() {
            DataType::Utf8 => {
                let array = column.as_string::<i32>();
                for index in 0..array.len() {
                    hash_value(
                        array.is_null(index),
                        || array.value(index).as_bytes(),
                        digest,
                    );
                }
            }
            DataType::Utf8View => {
                let array = column.as_string_view();
                for index in 0..array.len() {
                    hash_value(
                        array.is_null(index),
                        || array.value(index).as_bytes(),
                        digest,
                    );
                }
            }
            DataType::Int64 => {
                let array = column.as_primitive::<Int64Type>();
                for index in 0..array.len() {
                    let bytes = array.value(index).to_le_bytes();
                    hash_value(array.is_null(index), || &bytes, digest);
                }
            }
            other => bail!("verify does not handle arrow type {other}"),
        }
    }
    Ok(())
}

/// Fold one possibly-null value into `digest`.
fn hash_value<'a>(is_null: bool, bytes: impl FnOnce() -> &'a [u8], digest: &mut u64) {
    if is_null {
        hash_bytes(b"\0null", digest);
    } else {
        hash_bytes(bytes(), digest);
    }
}
