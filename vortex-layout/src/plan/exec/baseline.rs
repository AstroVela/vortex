// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use futures::channel::mpsc;
use once_cell::sync::OnceCell;
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
use super::TaskId;
use super::TaskUpdate;
use super::TraceEvent;
use super::evaluate;
use super::evaluate::primitive_values;
use crate::segments::SegmentSource;

static EXECUTOR_POOL: OnceCell<futures::executor::ThreadPool> = OnceCell::new();
type OfferedTasks = SmallVec<[TaskId; 16]>;
type TaskStarts = BTreeMap<TaskId, (Instant, &'static str, Option<super::MorselId>)>;

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
}

pub async fn run_self_paced(
    plan: SourcePlan,
    query: ScanQuery,
    morsel_rows: usize,
    source: Arc<dyn SegmentSource>,
    session: &VortexSession,
    options: RunOptions,
) -> VortexResult<RunResult> {
    let collect_trace = options.collect_trace;
    if options.concurrency == 0 {
        vortex_bail!("execution concurrency must be non-zero");
    }
    if options.concurrency > 1 {
        return run_self_paced_concurrent(plan, query, morsel_rows, source, session, options).await;
    }

    let init_started = collect_trace.then(Instant::now);
    let mut execution = Execution::try_new_with_policy(
        plan,
        query,
        morsel_rows,
        options.retention,
        options.policy,
    )?;
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
            apply_updates(&result, &mut offered);
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

            let admitted = choose_tasks(&execution, &offered, options.policy);
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

    Ok(RunResult {
        batches,
        metrics: execution.metrics().clone(),
        trace: execution.trace().to_vec(),
    })
}

async fn run_self_paced_concurrent(
    plan: SourcePlan,
    query: ScanQuery,
    morsel_rows: usize,
    source: Arc<dyn SegmentSource>,
    session: &VortexSession,
    options: RunOptions,
) -> VortexResult<RunResult> {
    let collect_trace = options.collect_trace;
    let init_started = collect_trace.then(Instant::now);
    let mut execution = Execution::try_new_with_policy(
        plan,
        query,
        morsel_rows,
        options.retention,
        options.policy,
    )?;
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
        execution.record_trace(None, format_args!("event=executor_pool_start"));
    }
    let pool = EXECUTOR_POOL.get_or_try_init(|| {
        futures::executor::ThreadPool::new()
            .map_err(|error| vortex_error::vortex_err!("cannot create executor pool: {error}"))
    })?;
    if collect_trace {
        execution.record_trace(None, format_args!("event=executor_pool_ready"));
    }
    let (completion_tx, mut completion_rx) = mpsc::unbounded();
    let mut running = 0usize;
    let mut task_starts = BTreeMap::new();

    while unfinished != 0 || running != 0 {
        while let Some(morsel) = ready.pop_front() {
            queued[morsel.0] = false;
            if batches[morsel.0].is_some() {
                continue;
            }
            let transitions_before = collect_trace.then(|| execution.metrics().transitions);
            let result = execution.advance(morsel, options.transition_budget)?;
            apply_updates(&result, &mut offered);
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

        if unfinished != 0 {
            let capacity = options.concurrency.saturating_sub(running);
            for task_id in choose_tasks(&execution, &offered, options.policy)
                .into_iter()
                .take(capacity)
            {
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
                        continue;
                    }
                    let source = Arc::clone(&source);
                    let session = session.clone();
                    let completion_tx = completion_tx.clone();
                    pool.spawn_ok(async move {
                        let completion = evaluate(task, source.as_ref(), &session).await;
                        drop(completion_tx.unbounded_send(completion));
                    });
                    running += 1;
                }
            }
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
        if let Some(completion) = completion_rx.next().await {
            running -= 1;
            let task_id = completion.task;
            let wake_morsels = execution.completion_morsels(task_id)?;
            if let Some((started, operation, morsel)) = task_starts.remove(&task_id) {
                execution.record_trace(
                    morsel,
                    format_args!(
                        "event=wait_end task={} operation={operation} task_latency_ns={} success={} offered={} running={}",
                        task_id.0,
                        started.elapsed().as_nanos(),
                        completion.result.is_ok(),
                        offered.len(),
                        running,
                    ),
                );
            }
            execution.complete(completion)?;
            enqueue_woken_morsels(wake_morsels, &batches, &mut ready, &mut queued);
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

    Ok(RunResult {
        batches: batches.into_iter().flatten().collect(),
        metrics: execution.metrics().clone(),
        trace: execution.trace().to_vec(),
    })
}

fn execute_inline(operation: &Operation, policy: SchedulePolicy) -> bool {
    matches!(operation, Operation::CombineDemand { .. })
        || (!matches!(policy, SchedulePolicy::LegacyAdaptivePredicates { .. })
            && matches!(operation, Operation::PackStruct { .. }))
}

fn enqueue_woken_morsels(
    morsels: impl IntoIterator<Item = super::MorselId>,
    batches: &[Option<ExecBatch>],
    ready: &mut VecDeque<super::MorselId>,
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
) -> VortexResult<SmallVec<[super::MorselId; 8]>> {
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

fn enqueue_morsel(
    ready: &mut VecDeque<super::MorselId>,
    queued: &mut [bool],
    morsel: super::MorselId,
) {
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
        Operation::DecodeFlat { .. } => "decode_flat",
        Operation::EvaluatePredicate { .. } => "evaluate_predicate",
        Operation::CombineDemand { .. } => "combine_demand",
        Operation::SelectFlat { .. } => "select_flat",
        Operation::PackStruct { .. } => "pack_struct",
    }
}

fn output_morsel(output: OutputSlot) -> Option<super::MorselId> {
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

fn apply_updates(result: &AdvanceResult, offered: &mut OfferedTasks) {
    for update in &result.work {
        match update {
            TaskUpdate::Offer(task) => {
                if let Err(index) = offered.binary_search(&task.id) {
                    offered.insert(index, task.id);
                }
            }
            TaskUpdate::Promote(_) => {}
            TaskUpdate::Revoke(task) => {
                remove_offered(offered, *task);
            }
        }
    }
}

fn remove_offered(offered: &mut OfferedTasks, task: TaskId) {
    if let Ok(index) = offered.binary_search(&task) {
        offered.remove(index);
    }
}

fn choose_tasks(
    execution: &Execution,
    offered: &OfferedTasks,
    policy: SchedulePolicy,
) -> SmallVec<[TaskId; 16]> {
    if offered.is_empty() {
        return SmallVec::new();
    }
    match policy {
        SchedulePolicy::AllReady | SchedulePolicy::ProjectionPrefetch => {
            offered.iter().copied().collect()
        }
        SchedulePolicy::SmallFrontier(limit) => {
            offered.iter().copied().take(limit.max(1)).collect()
        }
        SchedulePolicy::Reverse => offered.iter().rev().copied().collect(),
        SchedulePolicy::Random(seed) => {
            let len = offered.len();
            let index = usize::try_from(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
                .unwrap_or(0)
                % len;
            offered.get(index).copied().into_iter().collect()
        }
        SchedulePolicy::PredicateFirst
        | SchedulePolicy::AdaptivePredicates { .. }
        | SchedulePolicy::LegacyAdaptivePredicates { .. } => {
            let required = offered
                .iter()
                .copied()
                .filter(|task| {
                    execution
                        .task(*task)
                        .is_some_and(|task| task.necessity == Necessity::Required)
                })
                .collect::<SmallVec<_>>();
            if required.is_empty() {
                offered.iter().next().copied().into_iter().collect()
            } else {
                required
            }
        }
    }
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
