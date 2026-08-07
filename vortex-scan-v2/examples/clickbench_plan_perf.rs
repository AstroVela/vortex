// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use futures::future::try_join_all;
use tracing_subscriber::EnvFilter;
use vortex_array::array_session;
use vortex_array::expr::get_item;
use vortex_array::expr::lit;
use vortex_array::expr::not_eq;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::stream::ArrayStreamExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::scan::repeated_scan::RepeatedScan as ReaderRepeatedScan;
use vortex_layout::session::LayoutSession;
use vortex_scan_v2::RepeatedScan as PlanRepeatedScan;
use vortex_scan_v2::ScanBuilder;

const DATASET: &str = "vortex-bench/data/clickbench_partitioned/vortex-file-compressed";

fn main() -> VortexResult<()> {
    if env::var_os("RUST_LOG").is_some() {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .without_time()
            .init();
    }
    let plan_iterations = env_usize("PLAN_ITERS", 25)?;
    let execution_iterations = env_usize("EXEC_ITERS", 1)?;
    let plan_only = env::var_os("PLAN_ONLY").is_some();
    let scan_version = env::var("SCAN_VERSION").unwrap_or_else(|_| "v2".to_owned());

    block_on(|handle| async move {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);
        vortex_file::register_default_encodings(&session);

        let mut paths = fs::read_dir(DATASET)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<PathBuf>, _>>()?;
        paths.retain(|path| {
            path.extension()
                .is_some_and(|extension| extension == "vortex")
        });
        paths.sort();

        let files = try_join_all(
            paths
                .iter()
                .map(|path| session.open_options().open_path(path)),
        )
        .await?;

        let (mut planning_times, mut execution_times, rows) = match scan_version.as_str() {
            "v1" => {
                benchmark_reader_scans(&files, plan_iterations, execution_iterations, plan_only)
                    .await?
            }
            "v2" => {
                benchmark_plan_scans(
                    &files,
                    &session,
                    plan_iterations,
                    execution_iterations,
                    plan_only,
                )
                .await?
            }
            _ => return Err(vortex_err!("Unknown SCAN_VERSION {scan_version}")),
        };

        println!(
            "RESULT\tscan={scan_version}\tfiles={}\trows={rows}\texecution_ms={:.3}\tplanning_ms={:.3}\titerations={plan_iterations}\texecution_iterations={execution_iterations}",
            files.len(),
            median(&mut execution_times).as_secs_f64() * 1_000.0,
            median(&mut planning_times).as_secs_f64() * 1_000.0,
        );
        Ok(())
    })
}

async fn benchmark_plan_scans(
    files: &[VortexFile],
    session: &vortex_session::VortexSession,
    plan_iterations: usize,
    execution_iterations: usize,
    plan_only: bool,
) -> VortexResult<(Vec<Duration>, Vec<Duration>, usize)> {
    let mut planning_times = Vec::with_capacity(plan_iterations);
    for _ in 0..plan_iterations {
        let start = Instant::now();
        black_box(prepare_plan_scans(files, session)?);
        planning_times.push(start.elapsed());
    }

    let mut execution_times = Vec::with_capacity(execution_iterations);
    let mut rows = 0;
    if !plan_only {
        let scans = prepare_plan_scans(files, session)?;
        rows = execute_plan_scans(&scans).await?;
        for _ in 0..execution_iterations {
            let start = Instant::now();
            rows = black_box(execute_plan_scans(&scans).await?);
            execution_times.push(start.elapsed());
        }
    }
    Ok((planning_times, execution_times, rows))
}

async fn benchmark_reader_scans(
    files: &[VortexFile],
    plan_iterations: usize,
    execution_iterations: usize,
    plan_only: bool,
) -> VortexResult<(Vec<Duration>, Vec<Duration>, usize)> {
    let mut planning_times = Vec::with_capacity(plan_iterations);
    for _ in 0..plan_iterations {
        let start = Instant::now();
        black_box(prepare_reader_scans(files)?);
        planning_times.push(start.elapsed());
    }

    let mut execution_times = Vec::with_capacity(execution_iterations);
    let mut rows = 0;
    if !plan_only {
        let scans = prepare_reader_scans(files)?;
        rows = execute_reader_scans(&scans).await?;
        for _ in 0..execution_iterations {
            let start = Instant::now();
            rows = black_box(execute_reader_scans(&scans).await?);
            execution_times.push(start.elapsed());
        }
    }
    Ok((planning_times, execution_times, rows))
}

fn prepare_plan_scans(
    files: &[VortexFile],
    session: &vortex_session::VortexSession,
) -> VortexResult<Vec<PlanRepeatedScan<vortex_array::ArrayRef>>> {
    let filter = not_eq(get_item("AdvEngineID", root()), lit(0_i16));
    let projection = select(["AdvEngineID"], root());
    files
        .iter()
        .map(|file| {
            ScanBuilder::try_new(
                file.footer().layout(),
                file.segment_source(),
                session.clone(),
            )?
            .with_filter(filter.clone())
            .with_projection(projection.clone())
            .prepare()
        })
        .collect()
}

fn prepare_reader_scans(
    files: &[VortexFile],
) -> VortexResult<Vec<ReaderRepeatedScan<vortex_array::ArrayRef>>> {
    let filter = not_eq(get_item("AdvEngineID", root()), lit(0_i16));
    let projection = select(["AdvEngineID"], root());
    files
        .iter()
        .map(|file| {
            file.scan()?
                .with_filter(filter.clone())
                .with_projection(projection.clone())
                .prepare()
        })
        .collect()
}

async fn execute_plan_scans(
    scans: &[PlanRepeatedScan<vortex_array::ArrayRef>],
) -> VortexResult<usize> {
    let outputs = try_join_all(scans.iter().map(|scan| async move {
        let array = scan.execute_array_stream(None)?.read_all().await?;
        Ok::<_, vortex_error::VortexError>(array.len())
    }))
    .await?;
    Ok(outputs.into_iter().sum())
}

async fn execute_reader_scans(
    scans: &[ReaderRepeatedScan<vortex_array::ArrayRef>],
) -> VortexResult<usize> {
    let outputs = try_join_all(scans.iter().map(|scan| async move {
        let array = scan.execute_array_stream(None)?.read_all().await?;
        Ok::<_, vortex_error::VortexError>(array.len())
    }))
    .await?;
    Ok(outputs.into_iter().sum())
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
