// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Matched v1 (`LayoutReader`) versus v2 (plan-native) scan execution benchmark.
//!
//! Both implementations run in the same process over the same files, session, expressions and
//! concurrency strategy, and their samples are interleaved so that machine noise affects both
//! equally. Files are generated once into `VORTEX_SYNTH_DIR` (defaults to
//! `target/synthetic-scan-bench`) and reused on later runs.
//!
//! ```bash
//! RUSTC_WRAPPER= cargo run --release -p vortex-scan-v2 --example synthetic_plan_perf
//! ```
//!
//! Environment variables:
//!
//! - `VORTEX_SYNTH_DIR`: dataset directory.
//! - `SYNTH_FILES`, `SYNTH_ROWS`, `SYNTH_CHUNK`: dataset shape (regenerate after changing).
//! - `EXEC_ITERS`: measured samples per implementation per workload.
//! - `WORKLOADS`: comma-separated workload names to run.

use std::env;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use futures::future::try_join_all;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tracing_subscriber::EnvFilter;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::not_eq;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::scan::repeated_scan::RepeatedScan as ReaderRepeatedScan;
use vortex_layout::session::LayoutSession;
use vortex_scan_v2::RepeatedScan as PlanRepeatedScan;
use vortex_scan_v2::ScanBuilder;
use vortex_session::VortexSession;

/// A named filter/projection pair exercised by both scan implementations.
struct Workload {
    name: &'static str,
    filter: Option<Expression>,
    projection: Expression,
}

fn workloads() -> Vec<Workload> {
    vec![
        // Clustered, highly selective: zone pruning removes most zones.
        Workload {
            name: "clustered-1pct",
            filter: Some(not_eq(get_item("adv_engine_id", root()), lit(0_i16))),
            projection: select(["adv_engine_id"], root()),
        },
        // Same predicate, but the projection reads a wider column than the filter.
        Workload {
            name: "clustered-1pct-wide-projection",
            filter: Some(not_eq(get_item("adv_engine_id", root()), lit(0_i16))),
            projection: select(["adv_engine_id", "region_id", "user_id"], root()),
        },
        // Scattered predicate that no zone can prune, matching roughly half the rows.
        Workload {
            name: "scattered-50pct",
            filter: Some(gt(get_item("user_id", root()), lit(1_i64 << 62))),
            projection: select(["region_id"], root()),
        },
        // Scattered and very selective: zone pruning cannot help, the mask is sparse.
        Workload {
            name: "scattered-0p1pct",
            filter: Some(lt(
                get_item("user_id", root()),
                lit(-9_214_364_837_600_000_000_i64),
            )),
            projection: select(["region_id", "user_id"], root()),
        },
        // Two conjuncts over different columns.
        Workload {
            name: "conjunction",
            filter: Some(and(
                not_eq(get_item("adv_engine_id", root()), lit(0_i16)),
                gt(get_item("region_id", root()), lit(100_i32)),
            )),
            projection: select(["region_id"], root()),
        },
        // String equality over a dictionary-friendly column.
        Workload {
            name: "string-equality",
            filter: Some(eq(
                get_item("url", root()),
                lit("https://example.com/path/17"),
            )),
            projection: select(["url"], root()),
        },
        // Dictionary filter with a cheap projection, isolating the filter side.
        Workload {
            name: "string-filter-only",
            filter: Some(eq(
                get_item("url", root()),
                lit("https://example.com/path/17"),
            )),
            projection: select(["region_id"], root()),
        },
        // Dictionary projection with no filter, isolating the projection side.
        Workload {
            name: "string-projection-only",
            filter: None,
            projection: select(["url"], root()),
        },
        // No filter at all: pure projection throughput.
        Workload {
            name: "projection-only",
            filter: None,
            projection: select(["region_id", "adv_engine_id"], root()),
        },
    ]
}

fn main() -> VortexResult<()> {
    if env::var_os("RUST_LOG").is_some() {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .without_time()
            .init();
    }

    let dir = PathBuf::from(
        env::var("VORTEX_SYNTH_DIR").unwrap_or_else(|_| "target/synthetic-scan-bench".to_owned()),
    );
    let file_count = env_usize("SYNTH_FILES", 8)?;
    let rows_per_file = env_usize("SYNTH_ROWS", 1_000_000)?;
    let chunk_rows = env_usize("SYNTH_CHUNK", 65_536)?;
    let execution_iterations = env_usize("EXEC_ITERS", 9)?;
    let selected = env::var("WORKLOADS").ok();

    block_on(|handle| async move {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);
        vortex_file::register_default_encodings(&session);
        vortex::editions::register_default_editions(&session);
        vortex::editions::enable_default_editions(&session);

        let paths = generate_dataset(&dir, file_count, rows_per_file, chunk_rows, &session).await?;
        let files = try_join_all(
            paths
                .iter()
                .map(|path| session.open_options().open_path(path)),
        )
        .await?;

        println!(
            "dataset\tfiles={}\trows_per_file={rows_per_file}\tchunk_rows={chunk_rows}",
            files.len()
        );
        println!("workload\tv1_ms\tv2_ms\tdelta\trows");

        for workload in workloads() {
            if let Some(selected) = &selected
                && !selected.split(',').any(|name| name == workload.name)
            {
                continue;
            }
            let (v1, v2, rows) =
                run_workload(&files, &session, &workload, execution_iterations).await?;
            let v1_ms = v1.as_secs_f64() * 1_000.0;
            let v2_ms = v2.as_secs_f64() * 1_000.0;
            println!(
                "{}\t{v1_ms:.3}\t{v2_ms:.3}\t{:+.1}%\t{rows}",
                workload.name,
                (v2_ms - v1_ms) / v1_ms * 100.0,
            );
        }
        Ok(())
    })
}

/// Runs both implementations with interleaved samples and returns their medians.
async fn run_workload(
    files: &[VortexFile],
    session: &VortexSession,
    workload: &Workload,
    iterations: usize,
) -> VortexResult<(Duration, Duration, usize)> {
    let v1_scans = prepare_reader_scans(files, workload)?;
    let v2_scans = prepare_plan_scans(files, session, workload)?;

    let v1_rows = execute_reader_scans(&v1_scans).await?;
    let v2_rows = execute_plan_scans(&v2_scans).await?;
    if v1_rows != v2_rows {
        return Err(vortex_err!(
            "Workload {} disagreed: v1 produced {v1_rows} rows, v2 produced {v2_rows}",
            workload.name
        ));
    }

    let mut v1_times = Vec::with_capacity(iterations);
    let mut v2_times = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        // Alternate which implementation runs first so neither systematically pays for
        // the other's cache footprint.
        if iteration % 2 == 0 {
            v1_times.push(time_reader(&v1_scans).await?);
            v2_times.push(time_plan(&v2_scans).await?);
        } else {
            v2_times.push(time_plan(&v2_scans).await?);
            v1_times.push(time_reader(&v1_scans).await?);
        }
    }

    Ok((median(&mut v1_times), median(&mut v2_times), v1_rows))
}

async fn time_reader(scans: &[ReaderRepeatedScan<ArrayRef>]) -> VortexResult<Duration> {
    let start = Instant::now();
    black_box(execute_reader_scans(scans).await?);
    Ok(start.elapsed())
}

async fn time_plan(scans: &[PlanRepeatedScan<ArrayRef>]) -> VortexResult<Duration> {
    let start = Instant::now();
    black_box(execute_plan_scans(scans).await?);
    Ok(start.elapsed())
}

fn prepare_plan_scans(
    files: &[VortexFile],
    session: &VortexSession,
    workload: &Workload,
) -> VortexResult<Vec<PlanRepeatedScan<ArrayRef>>> {
    files
        .iter()
        .map(|file| {
            let mut builder = ScanBuilder::try_new(
                file.footer().layout(),
                file.segment_source(),
                session.clone(),
            )?
            .with_projection(workload.projection.clone());
            if let Some(filter) = &workload.filter {
                builder = builder.with_filter(filter.clone());
            }
            builder.prepare()
        })
        .collect()
}

fn prepare_reader_scans(
    files: &[VortexFile],
    workload: &Workload,
) -> VortexResult<Vec<ReaderRepeatedScan<ArrayRef>>> {
    files
        .iter()
        .map(|file| {
            let mut builder = file.scan()?.with_projection(workload.projection.clone());
            if let Some(filter) = &workload.filter {
                builder = builder.with_filter(filter.clone());
            }
            builder.prepare()
        })
        .collect()
}

async fn execute_plan_scans(scans: &[PlanRepeatedScan<ArrayRef>]) -> VortexResult<usize> {
    let outputs = try_join_all(scans.iter().map(|scan| async move {
        let array = scan.execute_array_stream(None)?.read_all().await?;
        Ok::<_, vortex_error::VortexError>(array.len())
    }))
    .await?;
    Ok(outputs.into_iter().sum())
}

async fn execute_reader_scans(scans: &[ReaderRepeatedScan<ArrayRef>]) -> VortexResult<usize> {
    let outputs = try_join_all(scans.iter().map(|scan| async move {
        let array = scan.execute_array_stream(None)?.read_all().await?;
        Ok::<_, vortex_error::VortexError>(array.len())
    }))
    .await?;
    Ok(outputs.into_iter().sum())
}

/// Writes the dataset if it is absent, and returns its file paths.
async fn generate_dataset(
    dir: &Path,
    file_count: usize,
    rows_per_file: usize,
    chunk_rows: usize,
    session: &VortexSession,
) -> VortexResult<Vec<PathBuf>> {
    fs::create_dir_all(dir)?;
    let mut paths = Vec::with_capacity(file_count);
    for index in 0..file_count {
        let path = dir.join(format!("part-{index:03}.vortex"));
        if !path.exists() {
            write_file(&path, index, rows_per_file, chunk_rows, session).await?;
        }
        paths.push(path);
    }
    Ok(paths)
}

async fn write_file(
    path: &Path,
    file_index: usize,
    rows: usize,
    chunk_rows: usize,
    session: &VortexSession,
) -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(0x5EED_0000 + file_index as u64);
    let mut chunks = Vec::new();
    let mut written = 0;
    let mut chunk_index = 0_usize;
    while written < rows {
        let len = chunk_rows.min(rows - written);
        chunks.push(chunk(&mut rng, chunk_index, len)?);
        written += len;
        chunk_index += 1;
    }
    let dtype = chunks[0].dtype().clone();
    let array = ChunkedArray::try_new(chunks, dtype)?.into_array();

    let mut buffer = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut buffer, array.to_array_stream())
        .await?;

    let temp = path.with_extension("vortex.tmp");
    fs::write(&temp, buffer.freeze().as_slice())?;
    fs::rename(&temp, path)?;
    Ok(())
}

/// Builds one chunk of the synthetic dataset.
///
/// `adv_engine_id` is clustered: only every seventh chunk contains non-zero values, so zone maps
/// can prune the rest. `user_id` is uniformly random, so no zone can be pruned for predicates
/// over it.
fn chunk(rng: &mut StdRng, chunk_index: usize, len: usize) -> VortexResult<ArrayRef> {
    let clustered = chunk_index.is_multiple_of(7);
    let adv_engine_id = PrimitiveArray::from_iter((0..len).map(|row| {
        if clustered && row.is_multiple_of(14) {
            i16::try_from(1 + (row % 5)).unwrap_or(1)
        } else {
            0_i16
        }
    }))
    .into_array();

    let region_id =
        PrimitiveArray::from_iter((0..len).map(|_| rng.random_range(0_i32..1_000))).into_array();
    let user_id = PrimitiveArray::from_iter((0..len).map(|_| rng.random::<i64>())).into_array();
    let url = VarBinArray::from_iter_nonnull(
        (0..len).map(|row| format!("https://example.com/path/{}", row % 64)),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();

    Ok(StructArray::from_fields(
        [
            ("adv_engine_id", adv_engine_id),
            ("region_id", region_id),
            ("user_id", user_id),
            ("url", url),
        ]
        .as_slice(),
    )?
    .into_array())
}

fn env_usize(name: &str, default: usize) -> VortexResult<usize> {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .map_err(|error| vortex_err!("Invalid {name} value {value}: {error}"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn median(values: &mut [Duration]) -> Duration {
    if values.is_empty() {
        return Duration::ZERO;
    }
    values.sort_unstable();
    values[values.len() / 2]
}
