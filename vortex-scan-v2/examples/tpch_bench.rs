// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares plan-native scanning against the layout-reader scan on TPC-H data.
//!
//! The example generates the TPC-H `lineitem` table at the requested scale factor, writes it to a
//! Vortex file, then times planning and execution separately for both scan implementations.
//!
//! ```text
//! cargo run --release -p vortex-scan-v2 --example tpch_bench -- --scale-factor 1
//! ```

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use arrow_schema::SchemaRef;
use tokio::runtime::Builder;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tpchgen::generators::LineItemGenerator;
use tpchgen_arrow::LineItemArrow;
use tpchgen_arrow::RecordBatchIterator;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::get_item;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::lt_eq;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_arrow::FromArrowArray;
use vortex_arrow::FromArrowType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan_v2::ScanBuilder as PlanScanBuilder;
use vortex_session::VortexSession;

/// One benchmarked scan shape.
struct Workload {
    name: &'static str,
    projection: Expression,
    filter: Option<Expression>,
}

/// A `vortex.date[days]` literal, matching the dtype `tpchgen-arrow` produces.
fn date(days: i32) -> Expression {
    lit(Scalar::extension::<Date>(
        TimeUnit::Days,
        Scalar::primitive(days, Nullability::NonNullable),
    ))
}

/// A `decimal(15, 2)` literal, matching the dtype `tpchgen-arrow` produces.
fn decimal(units: i128) -> Expression {
    lit(Scalar::decimal(
        DecimalValue::I128(units),
        DecimalDType::new(15, 2),
        Nullability::NonNullable,
    ))
}

/// TPC-H scan shapes exercised by both scan implementations.
fn workloads() -> Vec<Workload> {
    // TPC-H dates are stored as days since epoch by `tpchgen-arrow`.
    // 1998-09-02 -> 10471, 1994-01-01 -> 8766, 1995-01-01 -> 9131.
    let shipdate = get_item("l_shipdate", root());
    let discount = get_item("l_discount", root());
    let quantity = get_item("l_quantity", root());

    vec![
        Workload {
            name: "project-all/no-filter",
            projection: root(),
            filter: None,
        },
        Workload {
            name: "project-2/no-filter",
            projection: select(["l_orderkey", "l_extendedprice"], root()),
            filter: None,
        },
        Workload {
            // Shape of TPC-H Q1: wide projection, weakly selective filter.
            name: "q1-like",
            projection: select(
                [
                    "l_returnflag",
                    "l_linestatus",
                    "l_quantity",
                    "l_extendedprice",
                    "l_discount",
                    "l_tax",
                ],
                root(),
            ),
            filter: Some(lt_eq(shipdate.clone(), date(10471))),
        },
        Workload {
            // Shape of TPC-H Q6: narrow projection, strongly selective conjunction.
            name: "q6-like",
            projection: select(["l_extendedprice", "l_discount"], root()),
            filter: Some(and(
                and(
                    and(
                        gt_eq(shipdate.clone(), date(8766)),
                        lt(shipdate.clone(), date(9131)),
                    ),
                    and(
                        gt_eq(discount.clone(), decimal(5)),
                        lt_eq(discount, decimal(7)),
                    ),
                ),
                lt(quantity, decimal(2400)),
            )),
        },
        Workload {
            // A point-ish lookup that should prune nearly every zone.
            name: "point-filter",
            projection: select(["l_orderkey", "l_extendedprice"], root()),
            filter: Some(and(
                gt_eq(shipdate.clone(), date(10400)),
                lt(shipdate, date(10401)),
            )),
        },
    ]
}

fn main() -> VortexResult<()> {
    let mut scale_factor = "1.0".to_string();
    let mut iterations = 5_usize;
    let mut plan_iterations = 20_usize;
    let mut only: Option<String> = None;
    let mut skip_exec = false;
    let mut skip_plan = false;
    let mut which: Option<String> = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scale-factor" | "-s" => {
                scale_factor = args.next().ok_or_else(|| vortex_err!("missing value"))?;
            }
            "--iterations" | "-i" => {
                iterations = args
                    .next()
                    .ok_or_else(|| vortex_err!("missing value"))?
                    .parse()
                    .map_err(|_| vortex_err!("invalid iterations"))?;
            }
            "--plan-iterations" => {
                plan_iterations = args
                    .next()
                    .ok_or_else(|| vortex_err!("missing value"))?
                    .parse()
                    .map_err(|_| vortex_err!("invalid plan iterations"))?;
            }
            "--only" => only = args.next(),
            "--plan-only" => skip_exec = true,
            "--exec-only" => skip_plan = true,
            "--impl" => which = args.next(),
            other => return Err(vortex_err!("unknown argument '{other}'")),
        }
    }

    if env::var_os("RUST_LOG").is_some() {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_target(true)
            .without_time()
            .init();
    }

    let runtime = Builder::new_multi_thread().enable_all().build()?;
    let _guard = runtime.enter();

    let session = VortexSession::default().with_tokio();

    let path = runtime.block_on(ensure_lineitem(&scale_factor, &session))?;
    let file = runtime.block_on(session.open_options().open_path(&path))?;
    println!(
        "file {} rows={} bytes={}",
        path.display(),
        file.row_count(),
        fs::metadata(&path)?.len()
    );

    println!(
        "\n{:<22} {:>10} {:>10} {:>8}   {:>10} {:>10} {:>8}",
        "workload", "plan v1", "plan v2", "ratio", "exec v1", "exec v2", "ratio"
    );
    println!("{}", "-".repeat(88));

    for workload in workloads() {
        if let Some(only) = &only
            && workload.name != only
        {
            continue;
        }

        let plan_iterations = if skip_plan { 0 } else { plan_iterations };
        let plan_v1 = time(plan_iterations, || {
            let mut builder = file.scan()?.with_projection(workload.projection.clone());
            if let Some(filter) = &workload.filter {
                builder = builder.with_filter(filter.clone());
            }
            builder.prepare().map(|_| ())
        })?;
        let plan_v2 = time(plan_iterations, || {
            let mut builder =
                plan_scan(&file, &session)?.with_projection(workload.projection.clone());
            if let Some(filter) = &workload.filter {
                builder = builder.with_filter(filter.clone());
            }
            builder.prepare().map(|_| ())
        })?;

        let (exec_v1, exec_v2, rows_v1, rows_v2) = if skip_exec {
            (Duration::ZERO, Duration::ZERO, 0, 0)
        } else {
            let run_v1 = which.as_deref() != Some("v2");
            let run_v2 = which.as_deref() != Some("v1");
            let mut rows_v1 = 0;
            let exec_v1 = time(if run_v1 { iterations } else { 0 }, || {
                rows_v1 = runtime.block_on(async {
                    let mut builder = file.scan()?.with_projection(workload.projection.clone());
                    if let Some(filter) = &workload.filter {
                        builder = builder.with_filter(filter.clone());
                    }
                    let array = builder.into_array_stream()?.read_all().await?;
                    VortexResult::Ok(array.len())
                })?;
                Ok(())
            })?;
            let mut rows_v2 = 0;
            let exec_v2 = time(if run_v2 { iterations } else { 0 }, || {
                rows_v2 = runtime.block_on(async {
                    let mut builder =
                        plan_scan(&file, &session)?.with_projection(workload.projection.clone());
                    if let Some(filter) = &workload.filter {
                        builder = builder.with_filter(filter.clone());
                    }
                    let array = builder.into_array_stream()?.read_all().await?;
                    VortexResult::Ok(array.len())
                })?;
                Ok(())
            })?;
            (exec_v1, exec_v2, rows_v1, rows_v2)
        };

        if rows_v1 != rows_v2 && !skip_exec && which.is_none() {
            println!(
                "WARNING: {} produced {rows_v1} rows under v1 and {rows_v2} rows under v2",
                workload.name
            );
        }

        println!(
            "{:<22} {:>10} {:>10} {:>7.2}x   {:>10} {:>10} {:>7.2}x",
            workload.name,
            format_duration(plan_v1),
            format_duration(plan_v2),
            ratio(plan_v2, plan_v1),
            format_duration(exec_v1),
            format_duration(exec_v2),
            ratio(exec_v2, exec_v1),
        );
    }

    Ok(())
}

fn plan_scan(
    file: &VortexFile,
    session: &VortexSession,
) -> VortexResult<PlanScanBuilder<ArrayRef>> {
    PlanScanBuilder::try_new(
        file.footer().layout(),
        file.segment_source(),
        session.clone(),
    )
}

/// Runs `f` `iterations` times after one warm-up and returns the median duration.
fn time(iterations: usize, mut f: impl FnMut() -> VortexResult<()>) -> VortexResult<Duration> {
    if iterations == 0 {
        return Ok(Duration::ZERO);
    }
    f()?;
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f()?;
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    Ok(samples[samples.len() / 2])
}

fn ratio(candidate: Duration, baseline: Duration) -> f64 {
    if baseline.is_zero() {
        return f64::NAN;
    }
    candidate.as_secs_f64() / baseline.as_secs_f64()
}

fn format_duration(duration: Duration) -> String {
    let micros = duration.as_secs_f64() * 1e6;
    if micros < 1000.0 {
        format!("{micros:.1}us")
    } else {
        format!("{:.2}ms", micros / 1000.0)
    }
}

/// Generates and writes `lineitem` for `scale_factor` unless the file already exists.
async fn ensure_lineitem(scale_factor: &str, session: &VortexSession) -> VortexResult<PathBuf> {
    let dir = env::var_os("VORTEX_TPCH_DIR").map_or_else(
        || {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../target/tpch")
                .join(scale_factor)
        },
        PathBuf::from,
    );
    fs::create_dir_all(&dir)?;
    let path = dir.join("lineitem.vortex");
    if path.exists() {
        return Ok(path);
    }

    println!("generating TPC-H lineitem at scale factor {scale_factor}...");
    let scale_factor: f64 = scale_factor
        .parse()
        .map_err(|_| vortex_err!("invalid scale factor"))?;
    let iterator =
        LineItemArrow::new(LineItemGenerator::new(scale_factor, 1, 1)).with_batch_size(8192 * 8);
    let schema: SchemaRef = Arc::clone(iterator.schema());
    let dtype = DType::from_arrow(schema);

    let (sender, receiver) = mpsc::channel::<VortexResult<ArrayRef>>(2);
    let generator = tokio::task::spawn_blocking(move || {
        for batch in iterator {
            let array = ArrayRef::from_arrow(&batch, false)?;
            if sender.blocking_send(Ok(array)).is_err() {
                break;
            }
        }
        VortexResult::Ok(())
    });

    let stream = ArrayStreamAdapter::new(dtype, ReceiverStream::new(receiver));
    let temporary = path.with_extension("vortex.tmp");
    let mut output = tokio::fs::File::create(&temporary).await?;
    session.write_options().write(&mut output, stream).await?;
    generator
        .await
        .map_err(|error| vortex_err!("generator task failed: {error}"))??;
    fs::rename(&temporary, &path)?;
    Ok(path)
}
