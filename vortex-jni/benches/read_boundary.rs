// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native "floor" for the `VortexJniReadBenchmark` JMH lanes.
//!
//! Reads the same canonical file (see [`jni_bench_data`]) on the same [`RUNTIME`]/[`POOL`] statics,
//! in two variants:
//!
//! - `partitions` calls [`partition_record_batches`], the function the `NativePartition.scanArrow`
//!   entry point calls, so the gap to the JMH `ops/s` is the JNI crossing and the Arrow C Data
//!   export and nothing else.
//! - `scan_builder` reads the same file through `ScanBuilder` with the Arrow conversion mapped
//!   inside the split tasks, which is how native Rust callers scan. The gap between the two
//!   variants is what the JNI's partition-at-a-time consumption costs.
//!
//! `ItemsCount::new(ROWS)` matches `@OperationsPerInvocation(ROWS)`, and the `*`/`*_pooled` pairs
//! match the JMH `workerThreads` `0`/`-1` params.

#![expect(clippy::unwrap_used)]

mod jni_bench_data;

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::Field;
use divan::Bencher;
use divan::counter::ItemsCount;
use futures::StreamExt;
use vortex::array::VortexSessionExecute;
use vortex::arrow::ArrowSessionExt;
use vortex::dtype::FieldName;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::runtime::BlockingRuntime;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scan::DataSourceRef;
use vortex::scan::ScanRequest;
use vortex::scan::selection::Selection;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_jni::POOL;
use vortex_jni::RUNTIME;
use vortex_jni::new_session;
use vortex_jni::open_data_source;
use vortex_jni::partition_record_batches;

use crate::jni_bench_data::ROWS;

fn main() {
    divan::main();
}

/// The lanes, mirroring `VortexJniReadBenchmark`.
#[derive(Clone, Copy)]
enum Lane {
    /// Read all six columns.
    FullScan,
    /// Native projection of `id, y`.
    Projection,
    /// Native filter `cat = 'alpha'` (~1/16 selectivity).
    SelectiveFilter,
}

/// Opened once per Divan sample, outside the timed region. Both variants read the same file: the
/// data source is what Java holds, the [`VortexFile`] is what `ScanBuilder` needs.
struct Env {
    session: VortexSession,
    data_source: DataSourceRef,
    file: VortexFile,
}

impl Env {
    fn open() -> VortexResult<Self> {
        let path = jni_bench_data::default_path();
        jni_bench_data::ensure_canonical(&path)?;

        let session = *new_session();
        let uri = url::Url::from_file_path(&path)
            .unwrap_or_else(|()| unreachable!("canonical path is absolute"))
            .to_string();
        let data_source = open_data_source(&session, &[uri], &HashMap::new())?;
        let file = RUNTIME.block_on(session.open_options().open_path(&path))?;
        Ok(Self {
            session,
            data_source,
            file,
        })
    }

    /// The native half of the Java `countRows`: one partition at a time, each consumed to
    /// completion through the JNI's own conversion path.
    fn run_partitions(&self, lane: Lane) -> VortexResult<u64> {
        let scan = RUNTIME.block_on(self.data_source.scan(scan_request(lane)))?;
        let mut partitions = scan.partitions();

        let mut rows = 0u64;
        while let Some(partition) = RUNTIME.block_on(partitions.next()) {
            let (_schema, batches) = partition_record_batches(&self.session, partition?)?;
            for batch in batches {
                rows += batch?.num_rows() as u64;
            }
        }
        Ok(rows)
    }

    /// The same read as a native caller writes it: the Arrow conversion is the scan's `map`, so it
    /// runs inside the split tasks rather than behind a second buffered stage.
    fn run_scan_builder(&self, lane: Lane) -> VortexResult<u64> {
        let mut builder = self.file.scan()?.with_ordered(false);
        match lane {
            Lane::Projection => builder = builder.with_projection(projection_expr()),
            Lane::SelectiveFilter => builder = builder.with_filter(filter_expr()),
            Lane::FullScan => {}
        }

        let schema = self.session.arrow().to_arrow_schema(&builder.dtype()?)?;
        let target = Arc::new(Field::new_struct("", schema.fields().clone(), false));
        let session = self.session.clone();

        let mut rows = 0u64;
        for batch in builder
            .map(move |array| {
                let mut ctx = session.create_execution_ctx();
                let arrow =
                    session
                        .arrow()
                        .execute_arrow(array, Some(target.as_ref()), &mut ctx)?;
                Ok(RecordBatch::from(arrow.as_struct().clone()).num_rows() as u64)
            })
            .into_iter(&*RUNTIME)?
        {
            rows += batch?;
        }
        Ok(rows)
    }
}

fn projection_expr() -> Expression {
    select(vec![FieldName::from("id"), FieldName::from("y")], root())
}

fn filter_expr() -> Expression {
    Binary.new_expr(
        Operator::Eq,
        [get_item(FieldName::from("cat"), root()), lit("alpha")],
    )
}

/// What a Java `ScanOptions` produces: all defaults but the projection and filter.
fn scan_request(lane: Lane) -> ScanRequest {
    ScanRequest {
        projection: match lane {
            Lane::Projection => projection_expr(),
            _ => root(),
        },
        filter: matches!(lane, Lane::SelectiveFilter).then(filter_expr),
        row_range: None,
        selection: Selection::All,
        ordered: false,
        limit: None,
        partition_selection: Selection::All,
        partition_range: None,
    }
}

/// `pooled` maps to the JMH `workerThreads` param: `false` is `0`, `true` is `-1`.
fn run_lane
    bencher: Bencher<'_, '_>,
    lane: Lane,
    pooled: bool,
    run: fn(&Env, Lane) -> VortexResult<u64>,
) {
    if pooled {
        POOL.set_workers_to_available_parallelism();
    } else {
        POOL.set_workers(0);
    }
    bencher
        .with_inputs(|| Env::open().unwrap())
        .input_counter(|_| ItemsCount::new(ROWS))
        .bench_refs(move |env| run(env, lane).unwrap());
}

/// Through the JNI's own partition consumption — the floor for `VortexJniReadBenchmark`.
mod partitions {
    use super::*;

    #[divan::bench]
    fn full_scan(bencher: Bencher) {
        run_lane(bencher, Lane::FullScan, false, Env::run_partitions);
    }

    #[divan::bench]
    fn projection(bencher: Bencher) {
        run_lane(bencher, Lane::Projection, false, Env::run_partitions);
    }

    #[divan::bench]
    fn selective_filter(bencher: Bencher) {
        run_lane(bencher, Lane::SelectiveFilter, false, Env::run_partitions);
    }

    #[divan::bench]
    fn full_scan_pooled(bencher: Bencher) {
        run_lane(bencher, Lane::FullScan, true, Env::run_partitions);
    }

    #[divan::bench]
    fn projection_pooled(bencher: Bencher) {
        run_lane(bencher, Lane::Projection, true, Env::run_partitions);
    }

    #[divan::bench]
    fn selective_filter_pooled(bencher: Bencher) {
        run_lane(bencher, Lane::SelectiveFilter, true, Env::run_partitions);
    }
}

/// Through `ScanBuilder`, as a native Rust caller would scan.
mod scan_builder {
    use super::*;

    #[divan::bench]
    fn full_scan(bencher: Bencher) {
        run_lane(bencher, Lane::FullScan, false, Env::run_scan_builder);
    }

    #[divan::bench]
    fn projection(bencher: Bencher) {
        run_lane(bencher, Lane::Projection, false, Env::run_scan_builder);
    }

    #[divan::bench]
    fn selective_filter(bencher: Bencher) {
        run_lane(bencher, Lane::SelectiveFilter, false, Env::run_scan_builder);
    }

    #[divan::bench]
    fn full_scan_pooled(bencher: Bencher) {
        run_lane(bencher, Lane::FullScan, true, Env::run_scan_builder);
    }

    #[divan::bench]
    fn projection_pooled(bencher: Bencher) {
        run_lane(bencher, Lane::Projection, true, Env::run_scan_builder);
    }

    #[divan::bench]
    fn selective_filter_pooled(bencher: Bencher) {
        run_lane(bencher, Lane::SelectiveFilter, true, Env::run_scan_builder);
    }
}
