// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::io::Write;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Instant;

use futures::StreamExt;
use futures::channel::mpsc;
use parking_lot::Mutex;
use smallvec::SmallVec;
use smallvec::smallvec;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use super::AdvanceResult;
use super::ArraySlotId;
use super::ClaimResult;
use super::Completion;
use super::ExecBatch;
use super::Execution;
use super::Metrics;
use super::MorselId;
use super::MorselState;
use super::Necessity;
use super::Operation;
use super::OutputSlot;
use super::ResolvedValue;
use super::RetentionPolicy;
use super::RunnableTask;
use super::ScanQuery;
use super::SchedulePolicy;
use super::SourcePlan;
use super::SpeculativeIoConfig;
use super::SpeculativeReadPolicy;
use super::TaskId;
use super::TaskUpdate;
use super::TraceEvent;
use super::evaluate;
use super::evaluate::primitive_values;
use crate::segments::SegmentSource;

static EXECUTOR_POOLS: LazyLock<Mutex<BTreeMap<usize, futures::executor::ThreadPool>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Coordinator phase timing is a diagnostic mode: it attributes coordinator wall time to drain,
/// advance, schedule, dispatch, inline completion, and channel-wait phases. It stays off unless
/// requested so clean comparison runs pay no `Instant` overhead.
static PHASE_TIMING: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("VORTEX_SELF_PACED_PHASE_TIMING").is_some());

/// Experimental shard count for the concurrent runner. Each shard owns a contiguous group of
/// morsels with its own coordinator; `1` preserves the single-coordinator behavior.
static SHARDS: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("VORTEX_SELF_PACED_SHARDS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|shards| *shards >= 1)
        .unwrap_or(1)
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShardMode {
    /// Shard coordinators dispatch evaluation work to one shared worker pool.
    Pooled,
    /// Every shard thread coordinates its own morsels and evaluates their tasks inline: there is
    /// no separate coordinator and no cross-thread task dispatch. Shard count defaults to the
    /// configured concurrency so the thread total matches the pooled worker budget.
    Owned,
    /// The extensible streaming pipeline: the scheduler only sees `dyn MorselPipeline`, demand
    /// compute is a pluggable `DemandPolicy`, and children may have arbitrary chunk boundaries.
    Pipeline,
}

static SHARD_MODE: LazyLock<ShardMode> =
    LazyLock::new(
        || match std::env::var("VORTEX_SELF_PACED_SHARD_MODE").as_deref() {
            Ok("owned") => ShardMode::Owned,
            Ok("pipeline") => ShardMode::Pipeline,
            _ => ShardMode::Pooled,
        },
    );

fn demand_policy_from_env() -> Arc<dyn super::DemandPolicy> {
    match std::env::var("VORTEX_SELF_PACED_DEMAND").as_deref() {
        Ok("eager") => Arc::new(super::EagerDemand),
        Ok("cascade") => Arc::new(super::CascadeDemand),
        _ => Arc::new(super::AdaptiveDemand::new()),
    }
}

fn phase_start(enabled: bool) -> Option<Instant> {
    enabled.then(Instant::now)
}

fn phase_add(slot: &mut u64, started: Option<Instant>) {
    if let Some(started) = started {
        *slot += u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    }
}

#[derive(Default)]
struct CoordinatorPhases {
    drain_ns: u64,
    advance_ns: u64,
    schedule_ns: u64,
    dispatch_ns: u64,
    inline_ns: u64,
    wait_ns: u64,
    complete_ns: u64,
    iterations: usize,
}
type OfferedTasks = SmallVec<[TaskId; 16]>;
type TaskStarts = BTreeMap<TaskId, (Instant, &'static str, Option<MorselId>)>;

#[derive(Clone, Debug)]
pub struct RunResult {
    pub batches: Vec<ExecBatch>,
    pub metrics: Metrics,
    pub trace: Vec<TraceEvent>,
}

#[derive(Clone, Copy, Debug)]
pub struct RunOptions {
    pub policy: SchedulePolicy,
    pub transition_budget: usize,
    pub retention: RetentionPolicy,
    pub concurrency: usize,
    pub collect_trace: bool,
    pub speculative_io: SpeculativeIoConfig,
}

pub async fn run_self_paced(
    plan: &SourcePlan,
    query: ScanQuery,
    morsel_rows: usize,
    source: Arc<dyn SegmentSource>,
    session: &VortexSession,
    options: RunOptions,
) -> VortexResult<RunResult> {
    if morsel_rows == 0 {
        vortex_bail!("morsel_rows must be non-zero");
    }
    let morsel_rows = u64::try_from(morsel_rows)?;
    let morsel_ranges = (0..plan.row_count)
        .step_by(usize::try_from(morsel_rows)?)
        .map(|start| start..(start + morsel_rows).min(plan.row_count))
        .collect::<Vec<_>>();
    run_self_paced_ranges(plan, query, &morsel_ranges, source, session, options).await
}

pub async fn run_self_paced_ranges(
    plan: &SourcePlan,
    query: ScanQuery,
    morsel_ranges: &[Range<u64>],
    source: Arc<dyn SegmentSource>,
    session: &VortexSession,
    options: RunOptions,
) -> VortexResult<RunResult> {
    if options.concurrency == 0 {
        vortex_bail!("execution concurrency must be non-zero");
    }
    if options.concurrency > 1 {
        if *SHARD_MODE == ShardMode::Pipeline {
            let threads = if *SHARDS > 1 {
                *SHARDS
            } else {
                options.concurrency
            }
            .min(morsel_ranges.len().max(1));
            let pipeline: Arc<dyn super::MorselPipeline> = Arc::new(
                super::StructScanPipeline::new(plan, query, demand_policy_from_env()),
            );
            return super::run_pipeline_sharded(pipeline, morsel_ranges, source, session, threads);
        }
        let owned = *SHARD_MODE == ShardMode::Owned;
        let requested_shards = if owned && *SHARDS <= 1 {
            options.concurrency
        } else {
            *SHARDS
        };
        let shards = requested_shards
            .min(options.concurrency)
            .min(morsel_ranges.len().max(1));
        if shards > 1 || (owned && shards == 1) {
            return run_self_paced_sharded(
                plan,
                query,
                morsel_ranges,
                source,
                session,
                options,
                shards,
                owned,
            );
        }
        let pool = executor_pool(options.concurrency)?;
        return run_self_paced_concurrent(
            plan,
            query,
            morsel_ranges,
            source,
            session,
            options,
            options.concurrency,
            pool,
        )
        .await;
    }

    run_self_paced_single(plan, query, morsel_ranges, source, session, options).await
}

/// The single-threaded runner: one thread both coordinates and evaluates every task inline.
/// Owned-mode sharding runs one of these per thread over that thread's morsel group, so each
/// morsel group's coordination happens on the thread that executes its work.
async fn run_self_paced_single(
    plan: &SourcePlan,
    query: ScanQuery,
    morsel_ranges: &[Range<u64>],
    source: Arc<dyn SegmentSource>,
    session: &VortexSession,
    options: RunOptions,
) -> VortexResult<RunResult> {
    let collect_trace = options.collect_trace;
    let init_started = collect_trace.then(Instant::now);
    let mut execution = Execution::try_new_with_policy_and_ranges(
        plan,
        query,
        morsel_ranges,
        options.retention,
        options.policy,
    )?;
    execution.set_speculative_io(options.speculative_io);
    execution.populate_segment_sizes(source.as_ref());
    execution.set_trace_enabled(collect_trace);
    if let Some(init_started) = init_started {
        let init_latency_ns = init_started.elapsed().as_nanos();
        execution.record_trace(
            None,
            format_args!("event=execution_init init_latency_ns={init_latency_ns}"),
        );
    }
    let morsels = execution.morsels().collect::<Vec<_>>();
    let mut batches = Vec::with_capacity(morsels.len());
    let mut offered = OfferedTasks::new();

    for morsel in morsels {
        loop {
            let transitions_before = collect_trace.then(|| execution.metrics().transitions);
            let result = execution.advance(morsel, options.transition_budget)?;
            apply_updates(&result, &mut offered, options.speculative_io);
            if let Some(transitions_before) = transitions_before {
                record_updates(&mut execution, &result, offered.len(), 0);
                let transitions = execution.metrics().transitions - transitions_before;
                let demand_rows = execution.metrics().demand_rows_current;
                execution.record_trace(
                    Some(morsel),
                    format_args!(
                        "event=advance state={:?} transitions={} updates={} offered={} running=0 demand_rows={}",
                        result.state,
                        transitions,
                        result.work.len(),
                        offered.len(),
                        demand_rows,
                    ),
                );
            }
            if let Some(batch) = result.output {
                if collect_trace {
                    execution.record_trace(
                        Some(morsel),
                        format_args!(
                            "event=output rows={} coverage={}..{} offered={} running=0",
                            batch.array.len(),
                            batch.coverage.start,
                            batch.coverage.end,
                            offered.len(),
                        ),
                    );
                }
                batches.push(batch);
                break;
            }
            if result.state == MorselState::Budgeted {
                continue;
            }

            let (admitted, considered) = choose_tasks(
                &execution,
                &offered,
                options.policy,
                options.speculative_io,
                options.speculative_io.max_in_flight_bytes,
                usize::MAX,
            )?;
            execution.record_scheduler_pass(considered, admitted.len());
            if collect_trace {
                execution.record_trace(
                    Some(morsel),
                    format_args!(
                        "event=schedule admitted={} offered={} running=0 policy={:?}",
                        admitted.len(),
                        offered.len(),
                        options.policy,
                    ),
                );
            }
            if admitted.is_empty() && result.state == MorselState::Quiescent {
                vortex_bail!("self-paced execution reached quiescence without runnable work");
            }
            for task_id in admitted {
                let speculative_charge =
                    speculative_read_charge(&execution, task_id, options.speculative_io)?;
                record_speculative_io_admission(
                    &mut execution,
                    task_id,
                    speculative_charge,
                    Some(morsel),
                    collect_trace,
                )?;
                remove_offered(&mut offered, task_id);
                match execution.claim(task_id)? {
                    ClaimResult::Runnable(task) => {
                        let trace_task = collect_trace.then(|| {
                            let operation = operation_name(&task.operation);
                            let task_morsel = output_morsel(task.output).or(Some(morsel));
                            execution.record_trace(
                                task_morsel,
                                format_args!(
                                    "event=claim task={} operation={operation} offered={} running=1",
                                    task_id.0,
                                    offered.len(),
                                ),
                            );
                            execution.record_trace(
                                task_morsel,
                                format_args!(
                                    "event=wait_start reason=task_completion task={} operation={operation} offered={} running=1",
                                    task_id.0,
                                    offered.len(),
                                ),
                            );
                            (Instant::now(), operation, task_morsel)
                        });
                        let completion = evaluate(task, source.as_ref(), session).await;
                        if let Some((started, operation, task_morsel)) = trace_task {
                            execution.record_trace(
                                task_morsel,
                                format_args!(
                                    "event=wait_end task={} operation={operation} task_latency_ns={} success={} offered={} running=0",
                                    task_id.0,
                                    started.elapsed().as_nanos(),
                                    completion.result.is_ok(),
                                    offered.len(),
                                ),
                            );
                        }
                        execution.complete(completion)?;
                    }
                    ClaimResult::Revoked => {
                        if collect_trace {
                            execution.record_trace(
                                Some(morsel),
                                format_args!(
                                    "event=claim_revoked task={} offered={}",
                                    task_id.0,
                                    offered.len()
                                ),
                            );
                        }
                    }
                }
            }
        }
    }

    execution.finalize_speculative_io_metrics();
    Ok(RunResult {
        batches,
        metrics: execution.metrics().clone(),
        trace: execution.trace().to_vec(),
    })
}

fn executor_pool(size: usize) -> VortexResult<futures::executor::ThreadPool> {
    let mut pools = EXECUTOR_POOLS.lock();
    if let Some(pool) = pools.get(&size) {
        return Ok(pool.clone());
    }
    let pool = futures::executor::ThreadPoolBuilder::new()
        .pool_size(size)
        .name_prefix(format!("self-paced-{size}-"))
        .create()
        .map_err(|error| vortex_error::vortex_err!("cannot create executor pool: {error}"))?;
    pools.insert(size, pool.clone());
    Ok(pool)
}

/// Run the scan as independent shards: each shard owns a contiguous group of morsels with its own
/// `Execution` and coordinator thread, while all shards share one worker pool and split the
/// admission budget. Resources are deduplicated within a shard only, so a segment straddling a
/// shard boundary may be read once per shard.
#[expect(
    clippy::too_many_arguments,
    reason = "experimental sharded entry mirrors the concurrent runner's signature"
)]
fn run_self_paced_sharded(
    plan: &SourcePlan,
    query: ScanQuery,
    morsel_ranges: &[Range<u64>],
    source: Arc<dyn SegmentSource>,
    session: &VortexSession,
    options: RunOptions,
    shards: usize,
    owned: bool,
) -> VortexResult<RunResult> {
    let pool = if owned {
        None
    } else {
        Some(executor_pool(options.concurrency)?)
    };
    let admission = (options.concurrency / shards).max(1);
    let group_len = morsel_ranges.len().div_ceil(shards);
    let results = std::thread::scope(|scope| {
        let handles = morsel_ranges
            .chunks(group_len)
            .map(|ranges| {
                let query = query.clone();
                let source = Arc::clone(&source);
                let session = session.clone();
                let pool = pool.clone();
                scope.spawn(move || {
                    futures::executor::block_on(async move {
                        match pool {
                            Some(pool) => {
                                run_self_paced_concurrent(
                                    plan, query, ranges, source, &session, options, admission, pool,
                                )
                                .await
                            }
                            None => {
                                run_self_paced_single(
                                    plan, query, ranges, source, &session, options,
                                )
                                .await
                            }
                        }
                    })
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| vortex_error::vortex_err!("self-paced shard panicked"))?
            })
            .collect::<VortexResult<Vec<_>>>()
    })?;
    let mut merged: Option<RunResult> = None;
    for result in results {
        match &mut merged {
            None => merged = Some(result),
            Some(merged) => {
                merged.batches.extend(result.batches);
                merged.metrics.absorb(&result.metrics);
                merged.trace.extend(result.trace);
            }
        }
    }
    merged.ok_or_else(|| vortex_error::vortex_err!("sharded execution produced no shards"))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the sharded runner supplies a shared pool and a per-shard admission budget"
)]
async fn run_self_paced_concurrent(
    plan: &SourcePlan,
    query: ScanQuery,
    morsel_ranges: &[Range<u64>],
    source: Arc<dyn SegmentSource>,
    session: &VortexSession,
    options: RunOptions,
    admission: usize,
    pool: futures::executor::ThreadPool,
) -> VortexResult<RunResult> {
    let collect_trace = options.collect_trace;
    let init_started = collect_trace.then(Instant::now);
    let mut execution = Execution::try_new_with_policy_and_ranges(
        plan,
        query,
        morsel_ranges,
        options.retention,
        options.policy,
    )?;
    execution.set_speculative_io(options.speculative_io);
    execution.populate_segment_sizes(source.as_ref());
    execution.set_trace_enabled(collect_trace);
    if let Some(init_started) = init_started {
        let init_latency_ns = init_started.elapsed().as_nanos();
        execution.record_trace(
            None,
            format_args!("event=execution_init init_latency_ns={init_latency_ns}"),
        );
    }
    let morsels = execution.morsels().collect::<Vec<_>>();
    let mut batches = vec![None; morsels.len()];
    let mut unfinished = morsels.len();
    let mut ready = morsels.iter().copied().collect::<VecDeque<_>>();
    let mut queued = vec![true; morsels.len()];
    let mut offered = OfferedTasks::new();
    if collect_trace {
        execution.record_trace(None, format_args!("event=executor_pool_ready"));
    }
    let (completion_tx, mut completion_rx) = mpsc::unbounded();
    let mut running = 0usize;
    let mut task_starts = BTreeMap::new();
    let mut speculative_in_flight = BTreeMap::<TaskId, usize>::new();
    let phase_timing = *PHASE_TIMING;
    let mut phases = CoordinatorPhases::default();
    let run_started = phase_start(phase_timing);

    while unfinished != 0 || running != 0 {
        phases.iterations += 1;
        let phase_started = phase_start(phase_timing);
        let drained = drain_ready_completions(
            &mut completion_rx,
            &mut execution,
            &mut running,
            &mut speculative_in_flight,
            &mut task_starts,
            offered.len(),
            &batches,
            &mut ready,
            &mut queued,
        )?;
        execution.record_completion_batch(drained);
        phase_add(&mut phases.drain_ns, phase_started);
        let phase_started = phase_start(phase_timing);
        while let Some(morsel) = ready.pop_front() {
            queued[morsel.0] = false;
            if batches[morsel.0].is_some() {
                continue;
            }
            let transitions_before = collect_trace.then(|| execution.metrics().transitions);
            let result = execution.advance(morsel, options.transition_budget)?;
            apply_updates(&result, &mut offered, options.speculative_io);
            if let Some(transitions_before) = transitions_before {
                record_updates(&mut execution, &result, offered.len(), running);
                let transitions = execution.metrics().transitions - transitions_before;
                let demand_rows = execution.metrics().demand_rows_current;
                execution.record_trace(
                    Some(morsel),
                    format_args!(
                        "event=advance state={:?} transitions={} updates={} offered={} running={} demand_rows={}",
                        result.state,
                        transitions,
                        result.work.len(),
                        offered.len(),
                        running,
                        demand_rows,
                    ),
                );
            }
            if let Some(batch) = result.output {
                if collect_trace {
                    execution.record_trace(
                        Some(morsel),
                        format_args!(
                            "event=output rows={} coverage={}..{} offered={} running={}",
                            batch.array.len(),
                            batch.coverage.start,
                            batch.coverage.end,
                            offered.len(),
                            running,
                        ),
                    );
                }
                batches[morsel.0] = Some(batch);
                unfinished -= 1;
            } else if result.state == MorselState::Budgeted {
                enqueue_morsel(&mut ready, &mut queued, morsel);
            }
        }
        phase_add(&mut phases.advance_ns, phase_started);

        if unfinished != 0 {
            let phase_started = phase_start(phase_timing);
            let capacity = admission.saturating_sub(running);
            let speculative_bytes = speculative_in_flight.values().sum::<usize>();
            let available_speculative_bytes = options
                .speculative_io
                .max_in_flight_bytes
                .saturating_sub(speculative_bytes);
            let (admitted, considered) = choose_tasks(
                &execution,
                &offered,
                options.policy,
                options.speculative_io,
                available_speculative_bytes,
                capacity,
            )?;
            execution.record_scheduler_pass(considered, admitted.len());
            phase_add(&mut phases.schedule_ns, phase_started);
            let phase_started = phase_start(phase_timing);
            for task_id in admitted {
                let speculative_charge =
                    speculative_read_charge(&execution, task_id, options.speculative_io)?;
                record_speculative_io_admission(
                    &mut execution,
                    task_id,
                    speculative_charge,
                    None,
                    collect_trace,
                )?;
                remove_offered(&mut offered, task_id);
                if let ClaimResult::Runnable(task) = execution.claim(task_id)? {
                    let inline = execute_inline(&task.operation, options.policy);
                    if collect_trace {
                        let operation = operation_name(&task.operation);
                        let morsel = output_morsel(task.output);
                        task_starts.insert(task_id, (Instant::now(), operation, morsel));
                        execution.record_trace(
                            morsel,
                            format_args!(
                                "event=claim task={} operation={operation} offered={} running={}",
                                task_id.0,
                                offered.len(),
                                running + 1,
                            ),
                        );
                    }
                    if inline {
                        let inline_started = phase_start(phase_timing);
                        let wake_morsels = complete_inline_task(
                            task,
                            source.as_ref(),
                            session,
                            &mut execution,
                            &mut task_starts,
                            offered.len(),
                            running,
                        )
                        .await?;
                        enqueue_woken_morsels(wake_morsels, &batches, &mut ready, &mut queued);
                        phase_add(&mut phases.inline_ns, inline_started);
                        continue;
                    }
                    let source = Arc::clone(&source);
                    let session = session.clone();
                    let completion_tx = completion_tx.clone();
                    pool.spawn_ok(async move {
                        let mut completion = evaluate(task, source.as_ref(), &session).await;
                        if phase_timing {
                            completion.sent_at = Some(Instant::now());
                        }
                        drop(completion_tx.unbounded_send(completion));
                    });
                    if let Some(charge) = speculative_charge {
                        speculative_in_flight.insert(task_id, charge);
                    }
                    running += 1;
                }
            }
            phase_add(&mut phases.dispatch_ns, phase_started);
        }

        if !ready.is_empty() {
            continue;
        }
        if running == 0 {
            if unfinished != 0 {
                vortex_bail!("self-paced execution reached quiescence without runnable work");
            }
            break;
        }

        let wait_started = if !collect_trace {
            None
        } else {
            execution.record_trace(
                None,
                format_args!(
                    "event=wait_start reason=task_completion offered={} running={} unfinished={unfinished}",
                    offered.len(),
                    running,
                ),
            );
            Some(Instant::now())
        };
        let phase_wait_started = phase_start(phase_timing);
        if let Some(completion) = completion_rx.next().await {
            phase_add(&mut phases.wait_ns, phase_wait_started);
            let phase_started = phase_start(phase_timing);
            let mut drained = 1;
            complete_concurrent_task(
                completion,
                &mut execution,
                &mut running,
                &mut speculative_in_flight,
                &mut task_starts,
                offered.len(),
                &batches,
                &mut ready,
                &mut queued,
            )?;
            drained += drain_ready_completions(
                &mut completion_rx,
                &mut execution,
                &mut running,
                &mut speculative_in_flight,
                &mut task_starts,
                offered.len(),
                &batches,
                &mut ready,
                &mut queued,
            )?;
            execution.record_completion_batch(drained);
            phase_add(&mut phases.complete_ns, phase_started);
            if let Some(wait_started) = wait_started {
                execution.record_trace(
                    None,
                    format_args!(
                        "event=wait_end reason=task_completion wait_latency_ns={} offered={} running={} unfinished={unfinished}",
                        wait_started.elapsed().as_nanos(),
                        offered.len(),
                        running,
                    ),
                );
            }
        }
    }

    if phase_timing {
        let metrics = execution.metrics_mut();
        metrics.coordinator_loop_iterations = phases.iterations;
        metrics.coordinator_drain_ns = phases.drain_ns;
        metrics.coordinator_advance_ns = phases.advance_ns;
        metrics.coordinator_schedule_ns = phases.schedule_ns;
        metrics.coordinator_dispatch_ns = phases.dispatch_ns;
        metrics.coordinator_inline_ns = phases.inline_ns;
        metrics.coordinator_wait_ns = phases.wait_ns;
        metrics.coordinator_complete_ns = phases.complete_ns;
        phase_add(&mut metrics.coordinator_total_ns, run_started);
    }
    execution.finalize_speculative_io_metrics();
    Ok(RunResult {
        batches: batches.into_iter().flatten().collect(),
        metrics: execution.metrics().clone(),
        trace: execution.trace().to_vec(),
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "completion bookkeeping belongs together at the executor boundary"
)]
fn drain_ready_completions(
    completion_rx: &mut mpsc::UnboundedReceiver<Completion>,
    execution: &mut Execution,
    running: &mut usize,
    speculative_in_flight: &mut BTreeMap<TaskId, usize>,
    task_starts: &mut TaskStarts,
    offered: usize,
    batches: &[Option<ExecBatch>],
    ready: &mut VecDeque<MorselId>,
    queued: &mut [bool],
) -> VortexResult<usize> {
    let mut drained = 0;
    while let Ok(completion) = completion_rx.try_recv() {
        complete_concurrent_task(
            completion,
            execution,
            running,
            speculative_in_flight,
            task_starts,
            offered,
            batches,
            ready,
            queued,
        )?;
        drained += 1;
    }
    Ok(drained)
}

#[expect(
    clippy::too_many_arguments,
    reason = "completion bookkeeping belongs together at the executor boundary"
)]
fn complete_concurrent_task(
    completion: Completion,
    execution: &mut Execution,
    running: &mut usize,
    speculative_in_flight: &mut BTreeMap<TaskId, usize>,
    task_starts: &mut TaskStarts,
    offered: usize,
    batches: &[Option<ExecBatch>],
    ready: &mut VecDeque<MorselId>,
    queued: &mut [bool],
) -> VortexResult<()> {
    *running -= 1;
    let task_id = completion.task;
    if let Some(sent_at) = completion.sent_at {
        execution.record_completion_dwell(
            u64::try_from(sent_at.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
    }
    speculative_in_flight.remove(&task_id);
    let wake_morsels = execution.completion_morsels(task_id)?;
    if let Some((started, operation, morsel)) = task_starts.remove(&task_id) {
        execution.record_trace(
            morsel,
            format_args!(
                "event=wait_end task={} operation={operation} task_latency_ns={} success={} offered={} running={}",
                task_id.0,
                started.elapsed().as_nanos(),
                completion.result.is_ok(),
                offered,
                running,
            ),
        );
    }
    execution.complete(completion)?;
    enqueue_woken_morsels(wake_morsels, batches, ready, queued);
    Ok(())
}

fn execute_inline(operation: &Operation, policy: SchedulePolicy) -> bool {
    matches!(
        operation,
        Operation::CombineDemand { .. } | Operation::MergeDemandFragments
    ) || (!matches!(policy, SchedulePolicy::LegacyAdaptivePredicates { .. })
        && matches!(
            operation,
            Operation::PackStruct { .. }
                | Operation::SelectFlat {
                    selection_all_true: true,
                    ..
                }
                | Operation::SelectStruct {
                    selection_all_true: true,
                    ..
                }
        ))
}

fn enqueue_woken_morsels(
    morsels: impl IntoIterator<Item = MorselId>,
    batches: &[Option<ExecBatch>],
    ready: &mut VecDeque<MorselId>,
    queued: &mut [bool],
) {
    for morsel in morsels {
        if batches[morsel.0].is_none() {
            enqueue_morsel(ready, queued, morsel);
        }
    }
}

async fn complete_inline_task(
    task: RunnableTask,
    source: &dyn SegmentSource,
    session: &VortexSession,
    execution: &mut Execution,
    task_starts: &mut TaskStarts,
    offered: usize,
    running: usize,
) -> VortexResult<SmallVec<[MorselId; 8]>> {
    let record_demand_combination = matches!(task.operation, Operation::CombineDemand { .. });
    let completion = evaluate(task, source, session).await;
    let wake_morsels = execution.completion_morsels(completion.task)?;
    if let Some((started, operation, morsel)) = task_starts.remove(&completion.task) {
        execution.record_trace(
            morsel,
            format_args!(
                "event=inline_complete task={} operation={operation} task_latency_ns={} success={} offered={} running={}",
                completion.task.0,
                started.elapsed().as_nanos(),
                completion.result.is_ok(),
                offered,
                running,
            ),
        );
    }
    if record_demand_combination {
        execution.record_inline_demand_combination();
    }
    execution.complete(completion)?;
    Ok(wake_morsels)
}

fn enqueue_morsel(ready: &mut VecDeque<MorselId>, queued: &mut [bool], morsel: MorselId) {
    if !queued[morsel.0] {
        queued[morsel.0] = true;
        ready.push_back(morsel);
    }
}

fn record_updates(
    execution: &mut Execution,
    result: &AdvanceResult,
    offered: usize,
    running: usize,
) {
    for update in &result.work {
        match update {
            TaskUpdate::Offer(task) => execution.record_trace(
                output_morsel(task.output),
                format_args!(
                    "event=offer task={} class={:?} necessity={:?} operation={} offered={offered} running={running}",
                    task.id.0,
                    task.class,
                    task.necessity,
                    operation_name(&task.operation),
                ),
            ),
            TaskUpdate::Promote(task) => execution.record_trace(
                None,
                format_args!(
                    "event=promote task={} offered={offered} running={running}",
                    task.0
                ),
            ),
            TaskUpdate::Revoke(task) => execution.record_trace(
                None,
                format_args!(
                    "event=revoke task={} offered={offered} running={running}",
                    task.0
                ),
            ),
        }
    }
}

fn operation_name(operation: &Operation) -> &'static str {
    match operation {
        Operation::Read { .. } => "read",
        Operation::ReadDecodeFlat { .. } => "read_decode_flat",
        Operation::DecodeFlat { .. } => "decode_flat",
        Operation::EvaluatePredicate { .. } => "evaluate_predicate",
        Operation::CombineDemand { .. } => "combine_demand",
        Operation::MergeDemandFragments => "merge_demand_fragments",
        Operation::SelectFlat { .. } => "select_flat",
        Operation::SelectStruct { .. } => "select_struct",
        Operation::PackStruct { .. } => "pack_struct",
    }
}

fn output_morsel(output: OutputSlot) -> Option<MorselId> {
    match output {
        OutputSlot::Array(ArraySlotId::Morsel(morsel, _)) => Some(morsel),
        OutputSlot::Array(ArraySlotId::Scan(_)) | OutputSlot::Segment(_) => None,
    }
}

pub fn write_execution_trace(mut writer: impl Write, events: &[TraceEvent]) -> std::io::Result<()> {
    for (step, event) in events.iter().enumerate() {
        let morsel = event
            .morsel
            .map_or_else(|| "-".to_string(), |morsel| morsel.0.to_string());
        writeln!(
            writer,
            "step={step} t_us={:.3} morsel={morsel} {}",
            event.elapsed_ns as f64 / 1_000.0,
            event.message,
        )?;
    }
    Ok(())
}

fn apply_updates(
    result: &AdvanceResult,
    offered: &mut OfferedTasks,
    speculative_io: SpeculativeIoConfig,
) {
    for update in &result.work {
        match update {
            TaskUpdate::Offer(task) => {
                let schedulable = task.necessity == Necessity::Required
                    || match task.operation {
                        Operation::Read { phase, .. } | Operation::ReadDecodeFlat { phase, .. } => {
                            speculative_io.max_in_flight_bytes != 0
                                && ((phase.includes_predicate()
                                    && speculative_io.predicate != SpeculativeReadPolicy::Disabled)
                                    || (phase.includes_projection()
                                        && speculative_io.projection
                                            != SpeculativeReadPolicy::Disabled))
                        }
                        _ => false,
                    };
                if schedulable {
                    add_offered(offered, task.id);
                }
            }
            TaskUpdate::Promote(task) => add_offered(offered, *task),
            TaskUpdate::Revoke(task) => {
                remove_offered(offered, *task);
            }
        }
    }
}

fn add_offered(offered: &mut OfferedTasks, task: TaskId) {
    if let Err(index) = offered.binary_search(&task) {
        offered.insert(index, task);
    }
}

fn remove_offered(offered: &mut OfferedTasks, task: TaskId) {
    if let Ok(index) = offered.binary_search(&task) {
        offered.remove(index);
    }
}

fn record_speculative_io_admission(
    execution: &mut Execution,
    task: TaskId,
    charge: Option<usize>,
    morsel: Option<MorselId>,
    collect_trace: bool,
) -> VortexResult<()> {
    if !collect_trace {
        return Ok(());
    }
    let Some(charge) = charge else {
        return Ok(());
    };
    let Some(estimate) = execution.read_estimate(task)? else {
        return Ok(());
    };
    execution.record_trace(
        morsel,
        format_args!(
            "event=speculative_io_admit task={} phase={:?} estimated_bytes={:?} charge={} current_rows={} expected_rows={:.3}",
            task.0,
            estimate.phase,
            estimate.estimated_bytes,
            charge,
            estimate.current_rows,
            estimate.expected_rows,
        ),
    );
    Ok(())
}

fn choose_tasks(
    execution: &Execution,
    offered: &OfferedTasks,
    policy: SchedulePolicy,
    speculative_io: SpeculativeIoConfig,
    available_speculative_bytes: usize,
    max_tasks: usize,
) -> VortexResult<(SmallVec<[TaskId; 16]>, usize)> {
    if offered.is_empty() || max_tasks == 0 {
        return Ok((SmallVec::new(), 0));
    }
    let (start, reverse) = match policy {
        SchedulePolicy::Reverse | SchedulePolicy::AdaptivePredicates { .. } => {
            (offered.len() - 1, true)
        }
        SchedulePolicy::Random(seed) => {
            let len = offered.len();
            let index = usize::try_from(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
                .unwrap_or(0)
                % len;
            (index, false)
        }
        SchedulePolicy::AllReady
        | SchedulePolicy::ProjectionPrefetch
        | SchedulePolicy::SmallFrontier(_)
        | SchedulePolicy::PredicateFirst
        | SchedulePolicy::LegacyAdaptivePredicates { .. } => (0, false),
    };

    let mut admitted = SmallVec::new();
    let mut remaining_bytes = available_speculative_bytes;
    let limit = match policy {
        SchedulePolicy::SmallFrontier(limit) => max_tasks.min(limit.max(1)),
        _ => max_tasks,
    };
    let mut considered = 0;
    // With speculation disabled, candidate tasks are never offered, so the second necessity pass
    // would rescan the whole offered list without admitting anything.
    let speculation_enabled = speculative_io.max_in_flight_bytes != 0
        && (speculative_io.predicate != SpeculativeReadPolicy::Disabled
            || speculative_io.projection != SpeculativeReadPolicy::Disabled);
    'necessities: for necessity in [Necessity::Required, Necessity::Candidate] {
        if necessity == Necessity::Candidate && !speculation_enabled {
            break;
        }
        for offset in 0..offered.len() {
            let index = if reverse {
                start - offset
            } else {
                (start + offset) % offered.len()
            };
            let task = offered[index];
            considered += 1;
            let stored = execution
                .task(task)
                .ok_or_else(|| vortex_error::vortex_err!("unknown offered task {}", task.0))?;
            if stored.necessity != necessity {
                continue;
            }
            if necessity == Necessity::Required {
                admitted.push(task);
                if admitted.len() == limit {
                    break 'necessities;
                }
                continue;
            }
            let Some(charge) = speculative_read_charge(execution, task, speculative_io)? else {
                continue;
            };
            if charge > remaining_bytes {
                continue;
            }
            remaining_bytes -= charge;
            admitted.push(task);
            if admitted.len() == limit {
                break 'necessities;
            }
        }
    }
    Ok((admitted, considered))
}

fn speculative_read_charge(
    execution: &Execution,
    task: TaskId,
    config: SpeculativeIoConfig,
) -> VortexResult<Option<usize>> {
    if !execution
        .task(task)
        .is_some_and(|task| task.necessity == Necessity::Candidate)
    {
        return Ok(None);
    }
    let Some(estimate) = execution.read_estimate(task)? else {
        return Ok(None);
    };
    let predicate = estimate
        .phase
        .includes_predicate()
        .then_some(config.predicate);
    let projection = estimate
        .phase
        .includes_projection()
        .then_some(config.projection);
    let mut policies = [predicate, projection].into_iter().flatten();
    let admitted = policies.any(|policy| match policy {
        SpeculativeReadPolicy::Disabled => false,
        SpeculativeReadPolicy::Eager => true,
        SpeculativeReadPolicy::Adaptive {
            minimum_expected_rows,
        } => estimate.expected_rows >= minimum_expected_rows as f64,
    });
    if !admitted {
        return Ok(None);
    }
    let charge = estimate
        .estimated_bytes
        .unwrap_or(config.unknown_read_bytes);
    if charge == 0 {
        return Ok(None);
    }
    Ok(Some(charge))
}

pub async fn run_eager(
    plan: &SourcePlan,
    query: &ScanQuery,
    morsel_rows: usize,
    source: Arc<dyn SegmentSource>,
    session: &VortexSession,
) -> VortexResult<Vec<ExecBatch>> {
    let mut decoded = BTreeMap::new();
    let mut batches = Vec::new();
    for chunk in &plan.chunks {
        for flat in &chunk.fields {
            if query
                .conjuncts
                .iter()
                .any(|conjunct| conjunct.field == flat.field)
                || query.projection.contains(&flat.field)
            {
                let read = RunnableTask {
                    id: TaskId(usize::MAX),
                    inputs: SmallVec::new(),
                    output: OutputSlot::Segment(super::SegmentSlotId::Scan(usize::MAX)),
                    operation: Operation::Read {
                        segment: flat.segment,
                        phase: super::ReadPhase::PredicateAndProjection,
                        estimated_bytes: flat.estimated_bytes,
                    },
                };
                let Completion { result, .. } = evaluate(read, source.as_ref(), session).await;
                let segment = result?;
                let decode = RunnableTask {
                    id: TaskId(usize::MAX),
                    inputs: smallvec![segment],
                    output: OutputSlot::Array(ArraySlotId::Scan(usize::MAX)),
                    operation: Operation::DecodeFlat {
                        encoding: flat.encoding.clone(),
                        row_count: flat.row_count,
                    },
                };
                let Completion { result, .. } = evaluate(decode, source.as_ref(), session).await;
                let ResolvedValue::Array(array) = result? else {
                    unreachable!()
                };
                decoded.insert(flat.field, array);
            }
        }

        let mut root_start = chunk.root_coverage.start;
        while root_start < chunk.root_coverage.end {
            let root_end = (root_start + u64::try_from(morsel_rows)?).min(chunk.root_coverage.end);
            let local_start = usize::try_from(root_start - chunk.root_coverage.start)?;
            let local_end = usize::try_from(root_end - chunk.root_coverage.start)?;
            let mut demand = vec![true; local_end - local_start];
            for conjunct in &query.conjuncts {
                let values = primitive_values(&decoded[&conjunct.field], session)?;
                for (selected, value) in demand.iter_mut().zip(&values[local_start..local_end]) {
                    *selected &= conjunct.predicate.matches(*value);
                }
            }
            let true_count = demand.iter().filter(|value| **value).count();
            let mut fields = Vec::with_capacity(query.projection.len());
            for field in &query.projection {
                let values = primitive_values(&decoded[field], session)?;
                fields.push(
                    PrimitiveArray::from_iter(
                        values[local_start..local_end]
                            .iter()
                            .zip(&demand)
                            .filter_map(|(value, selected)| selected.then_some(*value)),
                    )
                    .into_array(),
                );
            }
            let names = query
                .projection
                .iter()
                .map(|field| plan.field_names[field.0].clone())
                .collect::<Vec<_>>();
            let selection = vortex_array::arrays::BoolArray::from_iter(demand).into_array();
            let array = StructArray::try_new(
                FieldNames::from(
                    names
                        .iter()
                        .map(|name| Arc::<str>::from(name.as_str()))
                        .collect::<Vec<_>>(),
                ),
                fields,
                true_count,
                Validity::NonNullable,
            )?
            .into_array();
            batches.push(ExecBatch {
                coverage: root_start..root_end,
                selection,
                array,
            });
            root_start = root_end;
        }
    }
    Ok(batches)
}

pub fn stable_output_hash(
    batches: &[ExecBatch],
    session: &VortexSession,
) -> VortexResult<(usize, u64)> {
    let mut field_hashes = Vec::new();
    let mut rows = 0usize;
    for batch in batches {
        let mut ctx = session.create_execution_ctx();
        let array = if batch.array.is::<Struct>() {
            batch.array.as_::<Struct>().into_owned()
        } else {
            batch.array.clone().execute::<StructArray>(&mut ctx)?
        };
        rows += array.len();
        let fields = array.iter_unmasked_fields().collect::<Vec<_>>();
        if field_hashes.is_empty() {
            field_hashes.resize(fields.len(), 0xcbf29ce484222325u64);
        } else if field_hashes.len() != fields.len() {
            vortex_bail!(
                "output field count changed from {} to {} between batches",
                field_hashes.len(),
                fields.len()
            );
        }
        for (field_idx, field) in fields.into_iter().enumerate() {
            let values = if field.is::<Primitive>() {
                field.as_::<Primitive>().into_owned()
            } else {
                field.clone().execute::<PrimitiveArray>(&mut ctx)?
            };
            for value in values.as_slice::<i64>() {
                field_hashes[field_idx] ^= u64::from_le_bytes(value.to_le_bytes());
                field_hashes[field_idx] = field_hashes[field_idx].wrapping_mul(0x100000001b3);
            }
        }
    }
    let hash = field_hashes
        .into_iter()
        .fold(0xcbf29ce484222325u64, |hash, field_hash| {
            (hash ^ field_hash).wrapping_mul(0x100000001b3)
        });
    Ok((rows, hash))
}
