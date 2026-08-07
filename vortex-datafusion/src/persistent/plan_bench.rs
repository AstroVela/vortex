// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Env-gated scan-planning micro-benchmark.
//!
//! When `VORTEX_PLAN_BENCH=<iterations>` is set, every unique `(file, projection, filter)`
//! opened through the [`super::opener::VortexOpener`] is measured through both scan
//! implementations with the exact expressions DataFusion pushed down:
//!
//! - v1: the `LayoutReader`-based [`vortex::layout::scan::scan_builder::ScanBuilder`];
//! - v2: the plan-native [`vortex_scan_v2::ScanBuilder`].
//!
//! Each implementation is timed cold (including reader/plan-tree construction) and warm
//! (reusing a prebuilt reader/base plan). One tab-separated `VXPLANBENCH` line per unique
//! input is written to stderr for offline aggregation. This is a diagnostic for the
//! plan-native scan work and is inert unless the environment variable is set.

use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use std::time::Instant;

use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::file::VortexFile;
use vortex::layout::LayoutReaderRef;
use vortex::layout::plan::PlanExecutionContext;
use vortex::layout::plan::new_plan;
use vortex::layout::scan::scan_builder::ScanBuilder as V1ScanBuilder;
use vortex::session::VortexSession;
use vortex_scan_v2::ScanBuilder as V2ScanBuilder;
use vortex_utils::aliases::dash_map::DashMap;

/// Returns the configured iteration count, or `None` when the benchmark is disabled.
pub(crate) fn plan_bench_iterations() -> Option<usize> {
    std::env::var("VORTEX_PLAN_BENCH").ok()?.parse().ok()
}

fn seen_signatures() -> &'static DashMap<u64, ()> {
    static SEEN: OnceLock<DashMap<u64, ()>> = OnceLock::new();
    SEEN.get_or_init(DashMap::default)
}

/// Measures v1 and v2 scan planning over identical inputs and logs one summary line.
pub(crate) fn run_plan_bench(
    iterations: usize,
    session: &VortexSession,
    layout_reader: &LayoutReaderRef,
    vxf: &VortexFile,
    location: &str,
    projection: &Expression,
    filter: Option<&Expression>,
) {
    let mut hasher = DefaultHasher::new();
    location.hash(&mut hasher);
    projection.to_string().hash(&mut hasher);
    filter.map(ToString::to_string).hash(&mut hasher);
    let signature = hasher.finish();

    if seen_signatures().insert(signature, ()).is_some() {
        return;
    }

    match measure(iterations, session, layout_reader, vxf, projection, filter) {
        Ok(timings) => {
            eprintln!(
                "VXPLANBENCH\tfile={location}\trows={rows}\titers={iterations}\tsig={signature:016x}\t\
                 v1_cold_us={v1_cold:.1}\tv1_warm_us={v1_warm:.1}\tv2_cold_us={v2_cold:.1}\tv2_warm_us={v2_warm:.1}\t\
                 filter={filter}\tprojection={projection}",
                rows = vxf.row_count(),
                v1_cold = timings.v1_cold.as_secs_f64() * 1e6,
                v1_warm = timings.v1_warm.as_secs_f64() * 1e6,
                v2_cold = timings.v2_cold.as_secs_f64() * 1e6,
                v2_warm = timings.v2_warm.as_secs_f64() * 1e6,
                filter = filter.map(ToString::to_string).unwrap_or_default(),
            );
        }
        Err(error) => {
            eprintln!("VXPLANBENCH\tfile={location}\tsig={signature:016x}\terror={error}");
        }
    }
}

struct PlanBenchTimings {
    v1_cold: Duration,
    v1_warm: Duration,
    v2_cold: Duration,
    v2_warm: Duration,
}

fn measure(
    iterations: usize,
    session: &VortexSession,
    layout_reader: &LayoutReaderRef,
    vxf: &VortexFile,
    projection: &Expression,
    filter: Option<&Expression>,
) -> VortexResult<PlanBenchTimings> {
    let v1_cold = median_time(iterations, || {
        let reader = vxf.footer().layout().new_reader(
            "".into(),
            vxf.segment_source(),
            session,
            &Default::default(),
        )?;
        let scan = V1ScanBuilder::new(session.clone(), reader)
            .with_projection(projection.clone())
            .with_some_filter(filter.cloned())
            .prepare()?;
        black_box(&scan);
        Ok(())
    })?;

    let v1_warm = median_time(iterations, || {
        let scan = V1ScanBuilder::new(session.clone(), Arc::clone(layout_reader))
            .with_projection(projection.clone())
            .with_some_filter(filter.cloned())
            .prepare()?;
        black_box(&scan);
        Ok(())
    })?;

    let v2_cold = median_time(iterations, || {
        let scan = V2ScanBuilder::try_new(
            vxf.footer().layout(),
            vxf.segment_source(),
            session.clone(),
        )?
        .with_projection(projection.clone())
        .with_some_filter(filter.cloned())
        .prepare()?;
        black_box(&scan);
        Ok(())
    })?;

    let base_plan = new_plan(vxf.footer().layout())?;
    let v2_warm = median_time(iterations, || {
        let scan = V2ScanBuilder::from_plan(
            Arc::clone(&base_plan),
            PlanExecutionContext::new(vxf.segment_source(), session.clone()),
        )
        .with_projection(projection.clone())
        .with_some_filter(filter.cloned())
        .prepare()?;
        black_box(&scan);
        Ok(())
    })?;

    Ok(PlanBenchTimings {
        v1_cold,
        v1_warm,
        v2_cold,
        v2_warm,
    })
}

fn median_time(
    iterations: usize,
    mut run: impl FnMut() -> VortexResult<()>,
) -> VortexResult<Duration> {
    // One untimed warm-up amortizes lazy statics and allocator warm-up.
    run()?;
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        run()?;
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    Ok(samples[samples.len() / 2])
}
