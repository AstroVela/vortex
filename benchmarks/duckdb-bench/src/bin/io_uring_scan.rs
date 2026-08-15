// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use clap::ValueEnum;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::buffer::Buffer;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutStrategy;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::layout::layouts::table::TableStrategy;
use vortex::session::VortexSession;
use vortex_duckdb::duckdb::Database;
use vortex_duckdb::io_bench::IoBenchMetrics;
use vortex_duckdb::io_bench::io_bench_metrics;

const DEFAULT_FILE_ROWS: u64 = 1 << 26;
const DEFAULT_BLOCK_ROWS: usize = 1 << 16;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Engine {
    Pread,
    PerWorker,
    Threaded,
    V1Layout,
}

impl Engine {
    fn sql_name(self) -> &'static str {
        match self {
            Self::Pread => "pread",
            Self::PerWorker => "per-worker",
            Self::Threaded => "threaded",
            Self::V1Layout => "v1-layout",
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Workload {
    Light,
    Heavy,
}

impl Workload {
    fn name(self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Heavy => "heavy",
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "Compare DuckDB pread and io_uring scan progress models")]
struct Args {
    /// Columnar benchmark file. Missing files are generated automatically.
    #[arg(long, default_value = "/tmp/vortex-duckdb-io-bench.bin")]
    path: PathBuf,

    /// Vortex file used by the current V1 layout-reader comparison.
    #[arg(
        long,
        default_value = "/tmp/vortex-duckdb-io-bench-uncompressed.vortex"
    )]
    vortex_path: PathBuf,

    /// Rows stored in a newly generated file (16 bytes per row).
    #[arg(long, default_value_t = DEFAULT_FILE_ROWS)]
    file_rows: u64,

    /// Rows scanned by each query. Zero scans the whole generated file.
    #[arg(long, default_value_t = 0)]
    rows: u64,

    /// Rows per physical read block. The default is a 1 MiB block.
    #[arg(long, default_value_t = DEFAULT_BLOCK_ROWS)]
    block_rows: usize,

    /// Read-ahead blocks per DuckDB worker.
    #[arg(long, default_value_t = 4)]
    prefetch: usize,

    /// DuckDB external worker count.
    #[arg(long, default_value_t = 8)]
    threads: usize,

    #[arg(
        long,
        value_delimiter = ',',
        default_value = "pread,per-worker,threaded,v1-layout"
    )]
    engines: Vec<Engine>,

    #[arg(long, value_delimiter = ',', default_value = "light,heavy")]
    workloads: Vec<Workload>,

    #[arg(long, default_value_t = 1)]
    warmup: usize,

    #[arg(long, default_value_t = 5)]
    iterations: usize,

    /// Open the data file with O_DIRECT. Pass `--direct=false` to use the page cache.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    direct: bool,

    /// Replace an existing file whose length differs from `--file-rows`.
    #[arg(long)]
    regenerate: bool,

    /// Replace only the uncompressed V1 comparison file.
    #[arg(long)]
    regenerate_vortex: bool,

    /// Ask the kernel to evict this engine's input file before each query.
    #[arg(long)]
    evict_cache: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate(&args)?;
    prepare_file(&args.path, args.file_rows, args.block_rows, args.regenerate)?;

    let scan_rows = if args.rows == 0 {
        args.file_rows
    } else {
        args.rows
    };
    if scan_rows > args.file_rows {
        bail!(
            "--rows {scan_rows} exceeds the generated file's {} rows",
            args.file_rows
        );
    }

    let database = Database::open_in_memory()?;
    // Deliberately omit the optimizer extension so aggregate pushdown cannot move
    // the light query's DuckDB work into the Vortex scan implementation.
    database.register_table_functions()?;
    let connection = database.connect()?;
    connection.query(&format!("SET threads = {}", args.threads))?;
    connection.register_io_bench_scan()?;
    let uses_v1_layout = args
        .engines
        .iter()
        .any(|engine| matches!(engine, Engine::V1Layout));
    if uses_v1_layout {
        prepare_vortex_file(&args)?;
        if args.direct {
            eprintln!(
                "note: --direct applies to the raw engines; the current V1 layout reader uses buffered I/O"
            );
        }
    }
    let vortex_bytes = if uses_v1_layout {
        args.vortex_path.metadata()?.len().to_string()
    } else {
        "n/a".to_string()
    };

    println!(
        "# file={} vortex_file={} vortex_bytes={} file_rows={} scan_rows={} block_rows={} block_bytes={} prefetch={} threads={} direct={} evict_cache={}",
        args.path.display(),
        args.vortex_path.display(),
        vortex_bytes,
        args.file_rows,
        scan_rows,
        args.block_rows,
        args.block_rows * 16,
        args.prefetch,
        args.threads,
        args.direct,
        args.evict_cache,
    );
    println!(
        "engine,workload,iteration,elapsed_ms,result_rows,local_readers,reads,bytes,callbacks,submission_batches,completions,waits,ready_hits,max_callback_gap_ms,max_submit_gap_ms,handoff_fast_receives,handoff_fast_receive_ms,handoff_fast_receive_max_us,handoff_wait_ms,handoff_queue_ms,handoff_queue_max_ms,producer_send_ms,producer_send_max_ms"
    );

    let path = sql_string(&args.path)?;
    let vortex_path = sql_string(&args.vortex_path)?;
    for workload in &args.workloads {
        for engine in &args.engines {
            let query = query(*workload, &path, &vortex_path, *engine, scan_rows, &args);
            for _ in 0..args.warmup {
                evict_engine_file(*engine, &args)?;
                drop(connection.query(&query)?);
            }
            for iteration in 1..=args.iterations {
                evict_engine_file(*engine, &args)?;
                let started = Instant::now();
                let result = connection.query(&query)?;
                let elapsed = started.elapsed();
                let result_rows = result.row_count();
                drop(result);
                print_result(
                    *engine,
                    *workload,
                    iteration,
                    elapsed.as_secs_f64() * 1000.0,
                    result_rows,
                    if matches!(engine, Engine::V1Layout) {
                        IoBenchMetrics::default()
                    } else {
                        io_bench_metrics()
                    },
                );
            }
        }
    }
    Ok(())
}

fn validate(args: &Args) -> Result<()> {
    if args.file_rows == 0 {
        bail!("--file-rows must be non-zero");
    }
    if args.block_rows == 0 || !(args.block_rows * 16).is_multiple_of(4096) {
        bail!("--block-rows must produce a non-zero 4096-byte-aligned block");
    }
    if args.prefetch == 0 {
        bail!("--prefetch must be non-zero");
    }
    if args.threads == 0 {
        bail!("--threads must be non-zero");
    }
    if args.iterations == 0 {
        bail!("--iterations must be non-zero");
    }
    Ok(())
}

fn prepare_file(path: &Path, rows: u64, block_rows: usize, regenerate: bool) -> Result<()> {
    let blocks = rows.div_ceil(block_rows as u64);
    let expected_bytes = blocks * block_rows as u64 * 16;
    if path.exists() {
        let actual = path.metadata()?.len();
        if actual == expected_bytes && !regenerate {
            return Ok(());
        }
        if !regenerate {
            bail!(
                "{} is {actual} bytes, expected {expected_bytes}; pass --regenerate to replace it",
                path.display()
            );
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    eprintln!(
        "generating {} rows ({:.2} GiB) at {}",
        blocks * block_rows as u64,
        expected_bytes as f64 / (1_u64 << 30) as f64,
        path.display()
    );
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::with_capacity(8 << 20, file);
    let mut block = vec![0_i64; block_rows * 2];
    for block_index in 0..blocks {
        let base = block_index * block_rows as u64;
        let (values, payload) = block.split_at_mut(block_rows);
        for row in 0..block_rows {
            let value = base + row as u64;
            values[row] = value as i64;
            payload[row] = splitmix64(value) as i64;
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(block.as_ptr().cast::<u8>(), block.len() * size_of::<i64>())
        };
        writer.write_all(bytes)?;
    }
    writer.flush()?;
    Ok(())
}

fn prepare_vortex_file(args: &Args) -> Result<()> {
    if args.vortex_path.exists() {
        if !args.regenerate_vortex {
            return Ok(());
        }
        std::fs::remove_file(&args.vortex_path)?;
    }
    if let Some(parent) = args.vortex_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    eprintln!(
        "generating uncompressed V1 layout-reader input at {}",
        args.vortex_path.display()
    );

    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    let chunked: Arc<dyn LayoutStrategy> = Arc::new(ChunkedLayoutStrategy::new(Arc::clone(&flat)));
    let strategy: Arc<dyn LayoutStrategy> = Arc::new(TableStrategy::new(
        Arc::clone(&chunked),
        Arc::clone(&chunked),
    ));

    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&args.vortex_path)?;
    let first_rows = usize::try_from(args.file_rows.min(args.block_rows as u64))?;
    let first = vortex_block(0, first_rows)?;
    let mut writer = session
        .write_options()
        .with_strategy(strategy)
        .with_file_statistics(Vec::new())
        .blocking(&runtime)
        .writer(
            BufWriter::with_capacity(8 << 20, file),
            first.dtype().clone(),
        );
    writer.push(first)?;

    let blocks = args.file_rows.div_ceil(args.block_rows as u64);
    for block in 1..blocks {
        let base = block * args.block_rows as u64;
        let rows = usize::try_from((args.file_rows - base).min(args.block_rows as u64))?;
        writer.push(vortex_block(base, rows)?)?;
    }
    writer.finish()?;
    Ok(())
}

fn vortex_block(base: u64, rows: usize) -> Result<vortex::array::ArrayRef> {
    let mut values = Vec::with_capacity(rows);
    let mut payload = Vec::with_capacity(rows);
    for row in 0..rows {
        let value = base + row as u64;
        values.push(value as i64);
        payload.push(splitmix64(value) as i64);
    }
    Ok(StructArray::try_from_iter([
        ("value", Buffer::from(values).into_array()),
        ("payload", Buffer::from(payload).into_array()),
    ])?
    .into_array())
}

fn evict_engine_file(engine: Engine, args: &Args) -> Result<()> {
    if !args.evict_cache {
        return Ok(());
    }
    let path = if matches!(engine, Engine::V1Layout) {
        &args.vortex_path
    } else {
        &args.path
    };
    let file = OpenOptions::new().read(true).open(path)?;
    let status = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if status != 0 {
        bail!(
            "failed to evict {} from the page cache: {}",
            path.display(),
            std::io::Error::from_raw_os_error(status)
        );
    }
    Ok(())
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn sql_string(path: &Path) -> Result<String> {
    let path = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("benchmark path is not UTF-8"))?;
    Ok(path.replace('\'', "''"))
}

fn query(
    workload: Workload,
    path: &str,
    vortex_path: &str,
    engine: Engine,
    rows: u64,
    args: &Args,
) -> String {
    let scan = if matches!(engine, Engine::V1Layout) {
        if rows == args.file_rows {
            format!("SELECT value, payload FROM read_vortex('{vortex_path}')")
        } else {
            format!("SELECT value, payload FROM read_vortex('{vortex_path}') LIMIT {rows}")
        }
    } else {
        raw_scan(path, engine.sql_name(), rows, args, args.direct)
    };
    match workload {
        Workload::Light => {
            format!("SELECT sum(value::HUGEINT), sum(payload::HUGEINT) FROM ({scan}) input")
        }
        Workload::Heavy => format!(
            "SELECT value & 4095 AS bucket, count(*), \
             sum(hash(hash(value, payload), hash(value * 3, payload), \
                      hash(value * 5, payload), hash(value * 7, payload))::HUGEINT) \
             FROM ({scan}) input GROUP BY bucket"
        ),
    }
}

fn raw_scan(path: &str, engine: &str, rows: u64, args: &Args, direct: bool) -> String {
    let block_rows = args.block_rows;
    let prefetch = args.prefetch;
    let workers = args.threads;
    format!(
        "SELECT value, payload FROM io_bench_scan('{path}', '{engine}', {rows}::UBIGINT, \
         {block_rows}::UBIGINT, {prefetch}::UBIGINT, {direct}, {workers}::UBIGINT)",
    )
}

fn print_result(
    engine: Engine,
    workload: Workload,
    iteration: usize,
    elapsed_ms: f64,
    result_rows: u64,
    metrics: IoBenchMetrics,
) {
    println!(
        "{},{},{iteration},{elapsed_ms:.3},{result_rows},{},{},{},{},{},{},{},{},{:.3},{:.3},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        engine.sql_name(),
        workload.name(),
        metrics.local_readers,
        metrics.reads,
        metrics.bytes,
        metrics.callbacks,
        metrics.submission_batches,
        metrics.completions,
        metrics.waits,
        metrics.ready_hits,
        metrics.max_callback_gap_ns as f64 / 1_000_000.0,
        metrics.max_submit_gap_ns as f64 / 1_000_000.0,
        metrics.handoff_fast_receives,
        metrics.handoff_fast_receive_ns as f64 / 1_000_000.0,
        metrics.handoff_fast_receive_max_ns as f64 / 1_000.0,
        metrics.handoff_wait_ns as f64 / 1_000_000.0,
        metrics.handoff_queue_ns as f64 / 1_000_000.0,
        metrics.handoff_queue_max_ns as f64 / 1_000_000.0,
        metrics.producer_send_ns as f64 / 1_000_000.0,
        metrics.producer_send_max_ns as f64 / 1_000_000.0,
    );
}
