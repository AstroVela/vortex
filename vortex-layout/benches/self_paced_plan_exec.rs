// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::plan::exec::Conjunct;
use vortex_layout::plan::exec::FieldId;
use vortex_layout::plan::exec::MemorySegments;
use vortex_layout::plan::exec::Predicate;
use vortex_layout::plan::exec::RetentionPolicy;
use vortex_layout::plan::exec::RunOptions;
use vortex_layout::plan::exec::ScanQuery;
use vortex_layout::plan::exec::SchedulePolicy;
use vortex_layout::plan::exec::SourcePlan;
use vortex_layout::plan::exec::SpeculativeIoConfig;
use vortex_layout::plan::exec::run_self_paced;
use vortex_layout::plan::exec::stable_output_hash;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
});

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let _guard = RUNTIME.enter();
    vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_tokio()
});

fn fixture() -> (SourcePlan, Arc<MemorySegments>, ScanQuery) {
    let chunks = (0..16)
        .map(|chunk| {
            let start = chunk * 4096;
            vec![
                (0..4096).map(|row| ((start + row) % 100) as i64).collect(),
                (0..4096)
                    .map(|row| (((start + row) * 17) % 100) as i64)
                    .collect(),
                (0..4096).map(|row| (start + row) as i64).collect(),
            ]
        })
        .collect();
    let (plan, source) =
        SourcePlan::from_i64_chunks(vec!["a".into(), "b".into(), "c".into()], chunks).unwrap();
    let query = ScanQuery {
        conjuncts: vec![
            Conjunct {
                field: FieldId(0),
                predicate: Predicate::LessThan(50),
            },
            Conjunct {
                field: FieldId(1),
                predicate: Predicate::LessThan(50),
            },
        ],
        projection: vec![FieldId(0), FieldId(2)],
    };
    (plan, source, query)
}

#[divan::bench(args = [1usize, 4, 16, 64])]
fn transition_budget(bencher: Bencher, budget: usize) {
    let (plan, source, query) = fixture();
    bencher.bench(|| {
        let result = RUNTIME
            .block_on(run_self_paced(
                &plan,
                query.clone(),
                4096,
                Arc::<MemorySegments>::clone(&source),
                &SESSION,
                RunOptions {
                    policy: SchedulePolicy::SmallFrontier(4),
                    transition_budget: budget,
                    retention: RetentionPolicy::RetainUntilDead,
                    concurrency: 1,
                    collect_trace: false,
                    speculative_io: SpeculativeIoConfig::disabled(),
                },
            ))
            .unwrap();
        divan::black_box(stable_output_hash(&result.batches, &SESSION).unwrap())
    });
}
