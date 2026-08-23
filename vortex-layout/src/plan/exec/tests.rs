// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Struct;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;

use super::ClaimResult;
use super::Completion;
use super::Conjunct;
use super::Execution;
use super::FieldId;
use super::Predicate;
use super::ResolvedArray;
use super::ResolvedValue;
use super::ResourceLifetime;
use super::RetentionPolicy;
use super::RunOptions;
use super::ScanQuery;
use super::SchedulePolicy;
use super::SourcePlan;
use super::SpeculativeReadPolicy;
use super::TaskUpdate;
use super::run_eager;
use super::run_self_paced;
use super::run_self_paced_ranges;
use super::stable_output_hash;
use super::write_execution_trace;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::test::new_session;

fn fixture() -> VortexResult<(SourcePlan, Arc<super::MemorySegments>, ScanQuery)> {
    let chunks = (0..2)
        .map(|chunk| {
            let base = chunk * 8;
            vec![
                (base..base + 8).map(i64::from).collect(),
                (0..8).map(|row| i64::from((row + chunk) % 7)).collect(),
                (base + 100..base + 108).map(i64::from).collect(),
            ]
        })
        .collect();
    let (plan, source) =
        SourcePlan::from_i64_chunks(vec!["a".into(), "b".into(), "c".into()], chunks)?;
    let query = ScanQuery {
        conjuncts: vec![
            Conjunct {
                field: FieldId(0),
                predicate: Predicate::GreaterThan(4),
            },
            Conjunct {
                field: FieldId(1),
                predicate: Predicate::LessThan(5),
            },
        ],
        projection: vec![FieldId(0), FieldId(2)],
    };
    Ok((plan, source, query))
}

struct UnknownSizeSource(Arc<super::MemorySegments>);

impl SegmentSource for UnknownSizeSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.0.request(id)
    }
}

struct TrackingSource {
    inner: Arc<super::MemorySegments>,
    requests: parking_lot::Mutex<Vec<SegmentId>>,
}

impl SegmentSource for TrackingSource {
    fn estimated_size(&self, id: SegmentId) -> Option<usize> {
        self.inner.estimated_size(id)
    }

    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.requests.lock().push(id);
        self.inner.request(id)
    }
}

#[test]
fn same_field_bounds_coalesce_into_one_range() {
    let mut query = ScanQuery {
        conjuncts: vec![
            Conjunct {
                field: FieldId(0),
                predicate: Predicate::GreaterThan(3),
            },
            Conjunct {
                field: FieldId(1),
                predicate: Predicate::Equal(7),
            },
            Conjunct {
                field: FieldId(0),
                predicate: Predicate::LessThan(9),
            },
        ],
        projection: vec![FieldId(0)],
    };
    query.coalesce_same_field_predicates();
    assert_eq!(query.conjuncts.len(), 2);
    assert_eq!(
        query.conjuncts[0].predicate,
        Predicate::RangeExclusive { lower: 3, upper: 9 }
    );
    for value in 0..12 {
        assert_eq!(
            query.conjuncts[0].predicate.matches(value),
            value > 3 && value < 9
        );
    }
}

#[tokio::test]
async fn explicit_morsel_ranges_preserve_output() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, query) = fixture()?;
    let expected = run_eager(
        &plan,
        &query,
        4,
        Arc::<super::MemorySegments>::clone(&source),
        &session,
    )
    .await?;
    let actual = run_self_paced_ranges(
        &plan,
        query,
        &[0..3, 3..11, 11..16],
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::AllReady,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 3,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;
    assert_eq!(
        stable_output_hash(&actual.batches, &session)?,
        stable_output_hash(&expected, &session)?
    );
    Ok(())
}

#[tokio::test]
async fn multi_field_projection_uses_one_struct_selection_per_morsel() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, query) = fixture()?;
    let eager = run_eager(
        &plan,
        &query,
        4,
        Arc::<super::MemorySegments>::clone(&source),
        &session,
    )
    .await?;
    let actual = run_self_paced(
        &plan,
        query,
        4,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::AdaptivePredicates { concurrency: 4 },
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 4,
            collect_trace: true,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;
    assert_eq!(
        stable_output_hash(&actual.batches, &session)?,
        stable_output_hash(&eager, &session)?
    );
    let struct_selections = actual
        .trace
        .iter()
        .filter(|event| {
            event.message.contains("event=claim")
                && event.message.contains("operation=select_struct")
        })
        .count();
    assert_eq!(
        struct_selections,
        actual
            .batches
            .iter()
            .filter(|batch| !batch.array.is_empty())
            .count()
    );
    assert!(
        actual
            .trace
            .iter()
            .all(|event| !event.message.contains("operation=select_flat"))
    );
    Ok(())
}

#[rstest]
#[case(SchedulePolicy::PredicateFirst)]
#[case(SchedulePolicy::AdaptivePredicates { concurrency: 4 })]
#[case(SchedulePolicy::AllReady)]
#[case(SchedulePolicy::ProjectionPrefetch)]
#[case(SchedulePolicy::SmallFrontier(2))]
#[case(SchedulePolicy::Reverse)]
#[case(SchedulePolicy::Random(7))]
#[tokio::test]
async fn schedules_match_eager(#[case] policy: SchedulePolicy) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, query) = fixture()?;
    let eager = run_eager(
        &plan,
        &query,
        4,
        Arc::<super::MemorySegments>::clone(&source),
        &session,
    )
    .await?;
    let actual = run_self_paced(
        &plan,
        query,
        4,
        source,
        &session,
        RunOptions {
            policy,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 1,
            collect_trace: true,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;
    assert_eq!(
        stable_output_hash(&actual.batches, &session)?,
        stable_output_hash(&eager, &session)?
    );
    assert!(actual.metrics.max_updates_per_advance <= 4);
    assert!(
        actual
            .trace
            .windows(2)
            .all(|events| events[0].elapsed_ns <= events[1].elapsed_ns)
    );
    let mut rendered = Vec::new();
    write_execution_trace(&mut rendered, &actual.trace)?;
    let rendered = String::from_utf8_lossy(&rendered);
    assert!(rendered.contains("event=advance"));
    assert!(rendered.contains("event=wait_start"));
    assert!(rendered.contains("event=wait_end"));
    assert!(rendered.contains("event=demand_adopt"));
    Ok(())
}

#[tokio::test]
async fn current_and_noop_predicates_do_not_offer_combine_tasks() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, mut query) = fixture()?;
    query.conjuncts[1].predicate = Predicate::LessThan(i64::MAX);
    let result = run_self_paced(
        &plan,
        query,
        4,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::AllReady,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 1,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;
    assert_eq!(result.metrics.demand_combinations, 0);
    assert!(result.metrics.demand_direct_adoptions > 0);
    assert!(result.metrics.demand_noop_adoptions > 0);
    Ok(())
}

#[tokio::test]
async fn shared_resources_move_from_reusable_to_dead() -> VortexResult<()> {
    let (plan, _source, query) = fixture()?;
    let mut execution = Execution::try_new(&plan, query, 4, RetentionPolicy::RetainUntilDead)?;
    let first = execution
        .morsels()
        .next()
        .ok_or_else(|| vortex_error::vortex_err!("fixture has no morsels"))?;
    let resource = execution
        .resources()
        .next()
        .ok_or_else(|| vortex_error::vortex_err!("fixture has no resources"))?;
    assert_eq!(
        execution.resource_lifetime(resource),
        ResourceLifetime::Reusable
    );
    let result = execution.advance(first, 1)?;
    assert!(result.work.is_empty());
    assert_eq!(
        execution.resource_lifetime(resource),
        ResourceLifetime::Pinned
    );
    Ok(())
}

#[tokio::test]
async fn wrong_duplicate_and_revoked_tasks_do_not_stay_running() -> VortexResult<()> {
    let (plan, _source, query) = fixture()?;
    let mut execution = Execution::try_new(&plan, query, 4, RetentionPolicy::RetainUntilDead)?;
    let morsel = execution
        .morsels()
        .next()
        .ok_or_else(|| vortex_error::vortex_err!("fixture has no morsels"))?;
    let offered = loop {
        let result = execution.advance(morsel, 1)?;
        if let Some(TaskUpdate::Offer(task)) = result.work.into_iter().next() {
            break task;
        }
    };
    let ClaimResult::Runnable(_) = execution.claim(offered.id)? else {
        vortex_error::vortex_bail!("newly offered task was revoked");
    };
    let wrong = Completion {
        task: offered.id,
        output: offered.output,
        elapsed_ns: 0,
        read_bytes: None,
        result: Ok(ResolvedValue::Segment(
            vortex_array::buffer::BufferHandle::new_host(vortex_buffer::ByteBuffer::from(vec![0])),
        )),
    };
    assert!(execution.complete(wrong).is_err());
    assert!(execution.claim(offered.id).is_err());
    assert!(
        execution
            .complete(Completion {
                task: offered.id,
                output: offered.output,
                elapsed_ns: 0,
                read_bytes: None,
                result: Ok(ResolvedValue::Array(ResolvedArray::plain(
                    vortex_array::arrays::BoolArray::from_iter([true]).into_array(),
                ))),
            })
            .is_err()
    );

    let next_offer = loop {
        let result = execution.advance(morsel, 1)?;
        if let Some(TaskUpdate::Offer(task)) = result.work.into_iter().next() {
            break task;
        }
    };
    let update = execution.revoke(next_offer.id)?;
    assert!(matches!(update, TaskUpdate::Revoke(id) if id == next_offer.id));
    assert!(matches!(
        execution.claim(next_offer.id)?,
        ClaimResult::Revoked
    ));
    Ok(())
}

#[tokio::test]
async fn logical_row_count_does_not_change_resource_graph_size() -> VortexResult<()> {
    let (small, _, query) = fixture()?;
    let (large, _) = SourcePlan::from_i64_chunks(
        vec!["a".into(), "b".into(), "c".into()],
        vec![vec![vec![1; 80], vec![2; 80], vec![3; 80]]],
    )?;
    let small = Execution::try_new(&small, query.clone(), 4, RetentionPolicy::RetainUntilDead)?;
    let large = Execution::try_new(&large, query, 4, RetentionPolicy::RetainUntilDead)?;
    assert_eq!(small.metrics().resource_nodes, 6);
    assert_eq!(large.metrics().resource_nodes, 3);
    assert_eq!(large.resources().count(), 3);
    Ok(())
}

#[tokio::test]
async fn predicate_first_pipelines_while_all_ready_combines() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, query) = fixture()?;
    let pipelined = run_self_paced(
        &plan,
        query.clone(),
        4,
        Arc::<super::MemorySegments>::clone(&source),
        &session,
        RunOptions {
            policy: SchedulePolicy::PredicateFirst,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 1,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;
    let parallel = run_self_paced(
        &plan,
        query.clone(),
        4,
        Arc::<super::MemorySegments>::clone(&source),
        &session,
        RunOptions {
            policy: SchedulePolicy::AllReady,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 1,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;
    let adaptive = run_self_paced(
        &plan,
        query,
        8,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::AdaptivePredicates { concurrency: 4 },
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 4,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;

    assert_eq!(pipelined.metrics.demand_combinations, 0);
    assert!(parallel.metrics.demand_combinations > 0);
    assert_eq!(
        adaptive.metrics.inline_demand_combinations,
        adaptive.metrics.demand_combinations
    );
    assert_eq!(
        stable_output_hash(&pipelined.batches, &session)?,
        stable_output_hash(&parallel.batches, &session)?
    );
    assert_eq!(
        stable_output_hash(&pipelined.batches, &session)?,
        stable_output_hash(&adaptive.batches, &session)?
    );
    Ok(())
}

#[tokio::test]
async fn empty_demand_does_not_read_projection_resources() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, mut query) = fixture()?;
    query.conjuncts.truncate(1);
    query.conjuncts[0].predicate = Predicate::LessThan(i64::MIN);
    query.projection = vec![FieldId(2)];
    let result = run_self_paced(
        &plan,
        query,
        4,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::PredicateFirst,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 1,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;

    assert_eq!(
        result
            .batches
            .iter()
            .map(|batch| batch.array.len())
            .sum::<usize>(),
        0
    );
    assert_eq!(result.metrics.io_offered, 4);
    assert_eq!(result.metrics.speculative_io_offered, 2);
    assert_eq!(result.metrics.speculative_io_admitted, 0);
    assert_eq!(result.metrics.speculative_io_completed_bytes, 0);
    Ok(())
}

#[tokio::test]
async fn partial_projection_selection_remains_lazy() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, query) = fixture()?;
    let result = run_self_paced(
        &plan,
        query,
        8,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::AdaptivePredicates { concurrency: 4 },
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 4,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;

    assert!(result.batches.iter().any(|batch| {
        batch.array.is::<Struct>()
            && batch
                .array
                .as_::<Struct>()
                .iter_unmasked_fields()
                .all(|field| field.is::<Filter>())
    }));
    Ok(())
}

#[tokio::test]
async fn empty_demand_revokes_parallel_predicate_offers() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, mut query) = fixture()?;
    query.conjuncts[0].predicate = Predicate::LessThan(i64::MIN);
    let result = run_self_paced(
        &plan,
        query,
        4,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::SmallFrontier(1),
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 2,
            collect_trace: true,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;

    assert!(result.metrics.tasks_revoked > 0);
    assert!(
        result
            .trace
            .iter()
            .any(|event| event.message == "event=executor_pool_start")
    );
    Ok(())
}

#[tokio::test]
async fn adaptive_empty_demand_does_not_offer_remaining_predicates() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, mut query) = fixture()?;
    for conjunct in &mut query.conjuncts {
        conjunct.predicate = Predicate::LessThan(i64::MIN);
    }
    let morsel_rows = 4;
    let morsel_count = usize::try_from(plan.row_count)?.div_ceil(morsel_rows);
    let result = run_self_paced(
        &plan,
        query,
        morsel_rows,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::AdaptivePredicates { concurrency: 4 },
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 4,
            collect_trace: true,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;

    let predicate_claims = result
        .trace
        .iter()
        .filter(|event| {
            event.message.contains("event=claim")
                && event.message.contains("operation=evaluate_predicate")
        })
        .count();
    assert_eq!(predicate_claims, morsel_count);
    Ok(())
}

#[tokio::test]
async fn adaptive_equal_scores_preserve_query_order() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, query) = fixture()?;
    let result = run_self_paced(
        &plan,
        query,
        4,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::AdaptivePredicates { concurrency: 1 },
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 1,
            collect_trace: true,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;

    let first_predicate = result
        .trace
        .iter()
        .find(|event| event.message.contains("event=predicate_offer"));
    assert!(
        first_predicate.is_some_and(|event| event.message.contains("conjunct=0")),
        "first predicate was {first_predicate:?}"
    );
    Ok(())
}

#[tokio::test]
async fn empty_projection_preserves_selected_row_count() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, mut query) = fixture()?;
    query.conjuncts.clear();
    query.projection.clear();
    let expected_rows = usize::try_from(plan.row_count)?;
    let result = run_self_paced(
        &plan,
        query,
        4,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::PredicateFirst,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 1,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;

    assert_eq!(
        result
            .batches
            .iter()
            .map(|batch| batch.array.len())
            .sum::<usize>(),
        expected_rows
    );
    Ok(())
}

#[tokio::test]
async fn morsels_can_cross_flat_resource_boundaries() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, query) = fixture()?;
    let eager = run_eager(
        &plan,
        &query,
        12,
        Arc::<super::MemorySegments>::clone(&source),
        &session,
    )
    .await?;
    let result = run_self_paced(
        &plan,
        query,
        12,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::AllReady,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 2,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;

    assert_eq!(result.batches[0].coverage, 0..12);
    assert_eq!(
        stable_output_hash(&result.batches, &session)?,
        stable_output_hash(&eager, &session)?
    );
    Ok(())
}

#[tokio::test]
async fn cross_chunk_morsel_skips_projection_slice_with_empty_demand() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, mut query) = fixture()?;
    query.conjuncts.truncate(1);
    query.conjuncts[0].predicate = Predicate::LessThan(8);
    query.projection = vec![FieldId(2)];
    let projected = plan
        .chunks
        .iter()
        .map(|chunk| {
            chunk
                .fields
                .iter()
                .find(|field| field.field == FieldId(2))
                .map(|field| field.segment)
                .ok_or_else(|| vortex_error::vortex_err!("fixture chunk has no projected field"))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let source = Arc::new(TrackingSource {
        inner: source,
        requests: parking_lot::Mutex::new(Vec::new()),
    });
    let result = run_self_paced(
        &plan,
        query,
        12,
        Arc::<TrackingSource>::clone(&source),
        &session,
        RunOptions {
            policy: SchedulePolicy::PredicateFirst,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 2,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig::disabled(),
        },
    )
    .await?;

    assert_eq!(
        result
            .batches
            .iter()
            .map(|batch| batch.array.len())
            .sum::<usize>(),
        8
    );
    let requests = source.requests.lock();
    assert!(requests.contains(&projected[0]));
    assert!(!requests.contains(&projected[1]));
    Ok(())
}

#[tokio::test]
async fn speculative_projection_io_reports_wasted_bytes() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, mut query) = fixture()?;
    query.conjuncts.truncate(1);
    query.conjuncts[0].predicate = Predicate::LessThan(i64::MIN);
    query.projection = vec![FieldId(2)];
    let result = run_self_paced(
        &plan,
        query,
        4,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::PredicateFirst,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 1,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig {
                predicate: SpeculativeReadPolicy::Disabled,
                projection: SpeculativeReadPolicy::Eager,
                max_in_flight_bytes: usize::MAX,
                unknown_read_bytes: 64,
            },
        },
    )
    .await?;

    assert!(result.metrics.speculative_projection_io_offered > 0);
    assert!(result.metrics.speculative_io_admitted > 0);
    assert!(result.metrics.speculative_io_completed_bytes > 0);
    assert_eq!(result.metrics.speculative_io_useful_bytes, 0);
    assert_eq!(
        result.metrics.speculative_io_wasted_bytes,
        result.metrics.speculative_io_completed_bytes
    );
    Ok(())
}

#[tokio::test]
async fn speculative_predicate_io_reports_useful_bytes() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, mut query) = fixture()?;
    query.projection.clear();
    let result = run_self_paced(
        &plan,
        query,
        4,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::PredicateFirst,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 1,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig {
                predicate: SpeculativeReadPolicy::Eager,
                projection: SpeculativeReadPolicy::Disabled,
                max_in_flight_bytes: usize::MAX,
                unknown_read_bytes: 64,
            },
        },
    )
    .await?;

    assert!(result.metrics.speculative_predicate_io_offered > 0);
    assert!(result.metrics.speculative_io_admitted > 0);
    assert!(result.metrics.speculative_io_useful_bytes > 0);
    Ok(())
}

#[rstest]
#[case(SpeculativeReadPolicy::Eager, 63)]
#[case(
    SpeculativeReadPolicy::Adaptive {
        minimum_expected_rows: usize::MAX,
    },
    usize::MAX
)]
#[tokio::test]
async fn speculative_io_respects_admission_limits(
    #[case] projection: SpeculativeReadPolicy,
    #[case] max_in_flight_bytes: usize,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, mut query) = fixture()?;
    query.conjuncts.truncate(1);
    query.conjuncts[0].predicate = Predicate::LessThan(i64::MIN);
    query.projection = vec![FieldId(2)];
    let result = run_self_paced(
        &plan,
        query,
        4,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::PredicateFirst,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 1,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig {
                predicate: SpeculativeReadPolicy::Disabled,
                projection,
                max_in_flight_bytes,
                unknown_read_bytes: 64,
            },
        },
    )
    .await?;

    assert!(result.metrics.speculative_projection_io_offered > 0);
    assert_eq!(result.metrics.speculative_io_admitted, 0);
    assert_eq!(result.metrics.speculative_io_completed_bytes, 0);
    Ok(())
}

#[tokio::test]
async fn speculative_io_charges_unknown_size_fallback() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (plan, source, mut query) = fixture()?;
    query.conjuncts.truncate(1);
    query.conjuncts[0].predicate = Predicate::LessThan(i64::MIN);
    query.projection = vec![FieldId(2)];
    let source: Arc<dyn SegmentSource> = Arc::new(UnknownSizeSource(source));
    let result = run_self_paced(
        &plan,
        query,
        4,
        source,
        &session,
        RunOptions {
            policy: SchedulePolicy::PredicateFirst,
            transition_budget: 4,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 2,
            collect_trace: false,
            speculative_io: super::SpeculativeIoConfig {
                predicate: SpeculativeReadPolicy::Disabled,
                projection: SpeculativeReadPolicy::Eager,
                max_in_flight_bytes: 63,
                unknown_read_bytes: 64,
            },
        },
    )
    .await?;

    assert!(result.metrics.speculative_io_unknown_size > 0);
    assert_eq!(result.metrics.speculative_io_admitted, 0);
    Ok(())
}
