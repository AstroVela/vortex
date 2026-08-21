// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::time::Instant;

use smallvec::SmallVec;
use smallvec::smallvec;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::AdvanceResult;
use super::ArraySlotId;
use super::ClaimResult;
use super::Completion;
use super::DemandVersion;
use super::ExecBatch;
use super::InputSlot;
use super::MorselId;
use super::MorselState;
use super::Necessity;
use super::Operation;
use super::OutputSlot;
use super::ResolvedArray;
use super::ResolvedValue;
use super::ResourceId;
use super::ResourceLifetime;
use super::ResourceNode;
use super::RetentionPolicy;
use super::RunnableTask;
use super::ScanQuery;
use super::SchedulePolicy;
use super::SegmentSlotId;
use super::SourcePlan;
use super::Task;
use super::TaskId;
use super::TaskUpdate;
use super::WorkClass;
use super::slots::Slot;
use super::slots::SlotState;

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub advance_calls: usize,
    pub transitions: usize,
    pub nodes_inspected: usize,
    pub tasks_offered: usize,
    pub tasks_claimed: usize,
    pub tasks_promoted: usize,
    pub tasks_revoked: usize,
    pub tasks_completed: usize,
    pub io_offered: usize,
    pub cpu_offered: usize,
    pub demand_rows_initial: usize,
    pub demand_rows_current: usize,
    pub demand_combinations: usize,
    pub inline_demand_combinations: usize,
    pub demand_direct_adoptions: usize,
    pub demand_noop_adoptions: usize,
    pub adaptive_predicate_launches: usize,
    pub adaptive_predicate_waits: usize,
    pub predicate_reorders: usize,
    pub segment_reuse_hits: usize,
    pub decode_reuse_hits: usize,
    pub resource_nodes: usize,
    pub morsel_slots: usize,
    pub max_updates_per_advance: usize,
}

#[derive(Clone, Debug)]
pub struct TraceEvent {
    pub elapsed_ns: u64,
    pub morsel: Option<MorselId>,
    pub message: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TaskStatus {
    Offered,
    Running,
    Completed,
    Revoked,
}

#[derive(Clone, Copy, Debug)]
enum TaskOwner {
    Resource(ResourceId),
    Morsel(MorselId),
}

#[derive(Clone, Debug)]
struct StoredTask {
    task: Task,
    status: TaskStatus,
    owner: TaskOwner,
    leases: SmallVec<[ResourceId; 2]>,
    unwanted: bool,
}

#[derive(Clone, Debug)]
struct ResourceUse {
    resource: ResourceId,
    necessity: Necessity,
    joined: bool,
}

#[derive(Clone, Debug)]
struct ResourceSlice {
    resource: ResourceId,
    local_range: std::ops::Range<usize>,
}

#[derive(Clone, Debug)]
struct ConjunctState {
    slices: Vec<ResourceSlice>,
    result_slot: ArraySlotId,
    predicate_task: Option<TaskId>,
    combined: bool,
}

#[derive(Clone, Debug, Default)]
struct PredicateStats {
    samples: u64,
    input_rows: u64,
    output_rows: u64,
    elapsed_ns: u128,
}

impl PredicateStats {
    fn observe(&mut self, input_rows: usize, output_rows: usize, elapsed_ns: u64) {
        self.samples += 1;
        self.input_rows += input_rows as u64;
        self.output_rows += output_rows as u64;
        self.elapsed_ns += u128::from(elapsed_ns);
    }

    fn survival(&self, prior: f64) -> f64 {
        if self.input_rows == 0 {
            prior
        } else {
            self.output_rows as f64 / self.input_rows as f64
        }
    }

    fn duration_ns(&self, fallback: f64) -> f64 {
        if self.samples == 0 {
            fallback
        } else {
            self.elapsed_ns as f64 / self.samples as f64
        }
    }
}

#[derive(Clone, Debug)]
struct ProjectionState {
    slices: Vec<ResourceSlice>,
    output_slot: ArraySlotId,
    task: Option<TaskId>,
}

#[derive(Clone, Debug)]
struct PendingCombine {
    task: TaskId,
    output_slot: ArraySlotId,
    conjunct: usize,
    source_version: DemandVersion,
}

#[derive(Clone, Debug)]
struct Morsel {
    id: MorselId,
    root_range: std::ops::Range<u64>,
    uses: Vec<ResourceUse>,
    activation_cursor: usize,
    waiting_on: Option<ResourceId>,
    arrays: Vec<Slot<ResolvedArray>>,
    demand_slot: ArraySlotId,
    demand_version: DemandVersion,
    conjuncts: Vec<ConjunctState>,
    predicate_offer_cursor: usize,
    combine: Option<PendingCombine>,
    cancelling: bool,
    sealed: bool,
    projections: Vec<ProjectionState>,
    projection_offer_cursor: usize,
    pack_slot: ArraySlotId,
    pack_task: Option<TaskId>,
    tasks: SmallVec<[TaskId; 8]>,
    retired: bool,
}

pub struct Execution {
    query: ScanQuery,
    projection_names: FieldNames,
    resources: Vec<ResourceNode>,
    segment_slots: Vec<Slot<vortex_array::buffer::BufferHandle>>,
    scan_array_slots: Vec<Slot<ResolvedArray>>,
    morsels: Vec<Morsel>,
    predicate_stats: Vec<PredicateStats>,
    tasks: Vec<StoredTask>,
    retention: RetentionPolicy,
    policy: SchedulePolicy,
    metrics: Metrics,
    trace: Vec<TraceEvent>,
    trace_enabled: bool,
    trace_start: Instant,
}

impl Execution {
    pub fn try_new(
        plan: SourcePlan,
        query: ScanQuery,
        morsel_rows: usize,
        retention: RetentionPolicy,
    ) -> VortexResult<Self> {
        Self::try_new_with_policy(
            plan,
            query,
            morsel_rows,
            retention,
            SchedulePolicy::AllReady,
        )
    }

    pub(crate) fn try_new_with_policy(
        plan: SourcePlan,
        query: ScanQuery,
        morsel_rows: usize,
        retention: RetentionPolicy,
        policy: SchedulePolicy,
    ) -> VortexResult<Self> {
        if morsel_rows == 0 {
            vortex_bail!("morsel_rows must be non-zero");
        }
        for conjunct in &query.conjuncts {
            if conjunct.field.0 >= plan.field_names.len() {
                vortex_bail!("predicate field {} is out of range", conjunct.field.0);
            }
        }
        for field in &query.projection {
            if field.0 >= plan.field_names.len() {
                vortex_bail!("projection field {} is out of range", field.0);
            }
        }
        let projection_names = query
            .projection
            .iter()
            .map(|field| plan.field_names[field.0].clone())
            .collect();

        let required_fields = query
            .conjuncts
            .iter()
            .map(|conjunct| conjunct.field)
            .collect::<BTreeSet<_>>();
        let used_fields = required_fields
            .iter()
            .copied()
            .chain(query.projection.iter().copied())
            .collect::<BTreeSet<_>>();
        let resource_capacity = plan
            .chunks
            .iter()
            .flat_map(|chunk| &chunk.fields)
            .filter(|flat| used_fields.contains(&flat.field))
            .count();
        let morsel_capacity = usize::try_from(plan.row_count)?.div_ceil(morsel_rows);
        let mut resources = Vec::with_capacity(resource_capacity);
        let mut resource_by_segment = BTreeMap::new();
        let mut segment_slots = Vec::with_capacity(resource_capacity);
        let mut scan_array_slots = Vec::with_capacity(resource_capacity);

        for chunk in &plan.chunks {
            for flat in &chunk.fields {
                if !used_fields.contains(&flat.field) {
                    continue;
                }
                if resource_by_segment.contains_key(&flat.segment) {
                    continue;
                }
                let id = ResourceId(resources.len());
                let segment_slot = SegmentSlotId::Scan(segment_slots.len());
                segment_slots.push(Slot::default());
                let array_slot = ArraySlotId::Scan(scan_array_slots.len());
                scan_array_slots.push(Slot::default());
                resources.push(ResourceNode {
                    id,
                    field: flat.field,
                    segment: flat.segment,
                    root_coverage: flat.root_coverage.clone(),
                    row_count: flat.row_count,
                    encoding: flat.encoding.clone(),
                    segment_slot,
                    array_slot,
                    unresolved_users: 0,
                    joined_users: 0,
                    leases: 0,
                    read_task: None,
                    decode_task: None,
                });
                resource_by_segment.insert(flat.segment, id);
            }
        }

        let mut morsels = Vec::with_capacity(morsel_capacity);
        let mut initial_demand_by_len = BTreeMap::new();
        let mut start = 0u64;
        while start < plan.row_count {
            let end = (start + u64::try_from(morsel_rows)?).min(plan.row_count);
            let id = MorselId(morsels.len());
            let mut uses = Vec::new();
            let overlapping = resources
                .iter()
                .filter(|resource| {
                    resource.root_coverage.start < end && resource.root_coverage.end > start
                })
                .map(|resource| resource.id)
                .collect::<Vec<_>>();
            for resource in overlapping {
                let necessity = if required_fields.contains(&resources[resource.0].field) {
                    Necessity::Required
                } else {
                    Necessity::Candidate
                };
                uses.push(ResourceUse {
                    resource,
                    necessity,
                    joined: false,
                });
                resources[resource.0].unresolved_users += 1;
            }

            let len = usize::try_from(end - start)?;
            let mut arrays =
                Vec::with_capacity(2 + query.projection.len() + query.conjuncts.len() * 2);
            let demand_slot = ArraySlotId::Morsel(id, arrays.len());
            let initial_demand = initial_demand_by_len
                .entry(len)
                .or_insert_with(|| {
                    ResolvedArray::boolean(
                        BoolArray::new(BitBuffer::new_set(len), Validity::NonNullable).into_array(),
                        len,
                    )
                })
                .clone();
            arrays.push(Slot {
                state: SlotState::Ready(initial_demand),
            });
            let slices = |field: super::FieldId| -> VortexResult<Vec<ResourceSlice>> {
                let slices = uses
                    .iter()
                    .filter(|use_| resources[use_.resource.0].field == field)
                    .map(|use_| {
                        let coverage = &resources[use_.resource.0].root_coverage;
                        let slice_start = start.max(coverage.start);
                        let slice_end = end.min(coverage.end);
                        Ok(ResourceSlice {
                            resource: use_.resource,
                            local_range: usize::try_from(slice_start - coverage.start)?
                                ..usize::try_from(slice_end - coverage.start)?,
                        })
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                let covered = slices
                    .iter()
                    .map(|slice| slice.local_range.len())
                    .sum::<usize>();
                if covered != len {
                    vortex_bail!("field {} covers {covered} of {len} morsel rows", field.0);
                }
                Ok(slices)
            };
            let mut conjuncts = Vec::with_capacity(query.conjuncts.len());
            for conjunct in &query.conjuncts {
                let result_slot = ArraySlotId::Morsel(id, arrays.len());
                arrays.push(Slot::default());
                conjuncts.push(ConjunctState {
                    slices: slices(conjunct.field)?,
                    result_slot,
                    predicate_task: None,
                    combined: false,
                });
            }
            let mut projections = Vec::with_capacity(query.projection.len());
            for field in &query.projection {
                let output_slot = ArraySlotId::Morsel(id, arrays.len());
                arrays.push(Slot::default());
                projections.push(ProjectionState {
                    slices: slices(*field)?,
                    output_slot,
                    task: None,
                });
            }
            let pack_slot = ArraySlotId::Morsel(id, arrays.len());
            arrays.push(Slot::default());
            morsels.push(Morsel {
                id,
                root_range: start..end,
                uses,
                activation_cursor: 0,
                waiting_on: None,
                arrays,
                demand_slot,
                demand_version: DemandVersion(0),
                conjuncts,
                predicate_offer_cursor: 0,
                combine: None,
                cancelling: false,
                sealed: query.conjuncts.is_empty(),
                projections,
                projection_offer_cursor: 0,
                pack_slot,
                pack_task: None,
                tasks: SmallVec::new(),
                retired: false,
            });
            start = end;
        }

        let initial_rows = morsels.iter().try_fold(0usize, |total, morsel| {
            Ok::<_, std::num::TryFromIntError>(
                total + usize::try_from(morsel.root_range.end - morsel.root_range.start)?,
            )
        })?;
        let task_capacity = resources.len() * 2
            + morsels.len() * (query.conjuncts.len() * 2 + query.projection.len() + 1);
        let predicate_stats = vec![PredicateStats::default(); query.conjuncts.len()];
        Ok(Self {
            query,
            projection_names,
            metrics: Metrics {
                demand_rows_initial: initial_rows,
                demand_rows_current: initial_rows,
                resource_nodes: resources.len(),
                morsel_slots: morsels.iter().map(|morsel| morsel.arrays.len()).sum(),
                ..Metrics::default()
            },
            resources,
            segment_slots,
            scan_array_slots,
            morsels,
            predicate_stats,
            tasks: Vec::with_capacity(task_capacity),
            retention,
            policy,
            trace: Vec::new(),
            trace_enabled: true,
            trace_start: Instant::now(),
        })
    }

    pub fn morsels(&self) -> impl Iterator<Item = MorselId> + '_ {
        self.morsels.iter().map(|morsel| morsel.id)
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    pub(crate) fn record_inline_demand_combination(&mut self) {
        self.metrics.inline_demand_combinations += 1;
    }

    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace
    }

    pub fn set_trace_enabled(&mut self, enabled: bool) {
        self.trace_enabled = enabled;
    }

    pub fn record_trace(&mut self, morsel: Option<MorselId>, message: std::fmt::Arguments<'_>) {
        if !self.trace_enabled {
            return;
        }
        self.trace.push(TraceEvent {
            elapsed_ns: u64::try_from(self.trace_start.elapsed().as_nanos()).unwrap_or(u64::MAX),
            morsel,
            message: message.to_string(),
        });
    }

    pub fn resource_lifetime(&self, resource: ResourceId) -> ResourceLifetime {
        self.resources[resource.0].lifetime()
    }

    pub fn resources(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.resources.iter().map(|resource| resource.id)
    }

    pub fn task(&self, id: TaskId) -> Option<&Task> {
        self.tasks.get(id.0).map(|stored| &stored.task)
    }

    pub fn completion_morsels(&self, task: TaskId) -> VortexResult<SmallVec<[MorselId; 8]>> {
        let stored = self
            .tasks
            .get(task.0)
            .ok_or_else(|| vortex_error::vortex_err!("unknown task {}", task.0))?;
        Ok(match stored.owner {
            TaskOwner::Morsel(morsel) => smallvec![morsel],
            TaskOwner::Resource(resource) => self
                .morsels
                .iter()
                .filter(|morsel| {
                    !morsel.retired
                        && if matches!(self.policy, SchedulePolicy::LegacyAdaptivePredicates { .. })
                        {
                            morsel
                                .uses
                                .iter()
                                .any(|use_| use_.joined && use_.resource == resource)
                        } else {
                            morsel.waiting_on == Some(resource)
                        }
                })
                .map(|morsel| morsel.id)
                .collect(),
        })
    }

    pub fn advance(
        &mut self,
        morsel: MorselId,
        transition_budget: usize,
    ) -> VortexResult<AdvanceResult> {
        if transition_budget == 0 {
            vortex_bail!("transition budget must be non-zero");
        }
        self.metrics.advance_calls += 1;
        if self.morsels.get(morsel.0).is_none() {
            vortex_bail!("unknown morsel {}", morsel.0);
        }
        if self.morsels[morsel.0].retired {
            return Ok(AdvanceResult {
                work: Vec::new(),
                output: None,
                state: MorselState::Retired,
            });
        }
        self.morsels[morsel.0].waiting_on = None;

        let mut work = Vec::new();
        let mut output = None;
        let mut transitions = 0usize;
        while transitions < transition_budget {
            self.metrics.nodes_inspected += 1;
            if !self.transition(morsel, &mut work, &mut output)? {
                break;
            }
            transitions += 1;
            self.metrics.transitions += 1;
            if output.is_some() {
                break;
            }
        }
        self.metrics.max_updates_per_advance = self.metrics.max_updates_per_advance.max(work.len());
        debug_assert!(work.len() <= transition_budget);

        let state = if self.morsels[morsel.0].retired {
            MorselState::Retired
        } else if transitions == transition_budget {
            MorselState::Budgeted
        } else {
            MorselState::Quiescent
        };
        Ok(AdvanceResult {
            work,
            output,
            state,
        })
    }

    fn transition(
        &mut self,
        morsel_id: MorselId,
        updates: &mut Vec<TaskUpdate>,
        output: &mut Option<ExecBatch>,
    ) -> VortexResult<bool> {
        if self.cancel_one_morsel_offer(morsel_id, updates)? {
            return Ok(true);
        }
        if self.adopt_combine(morsel_id)? {
            return Ok(true);
        }

        let activation_cursor = self.morsels[morsel_id.0].activation_cursor;
        if activation_cursor < self.morsels[morsel_id.0].uses.len() {
            let resource = self.morsels[morsel_id.0].uses[activation_cursor].resource;
            self.morsels[morsel_id.0].uses[activation_cursor].joined = true;
            self.morsels[morsel_id.0].activation_cursor += 1;
            let segment_slot = self.resources[resource.0].segment_slot;
            let array_slot = self.resources[resource.0].array_slot;
            debug_assert_ne!(self.resources[resource.0].unresolved_users, 0);
            self.resources[resource.0].unresolved_users -= 1;
            self.resources[resource.0].joined_users += 1;
            if self.segment_slot(segment_slot).ready().is_some() {
                self.metrics.segment_reuse_hits += 1;
            }
            if self.array_slot(array_slot).ready().is_some() {
                self.metrics.decode_reuse_hits += 1;
            }
            if self.trace_enabled {
                self.push_trace(
                    morsel_id,
                    format!("event=resource_join resource={}", resource.0),
                );
            }
            return Ok(true);
        }

        let sealed = self.morsels[morsel_id.0].sealed;
        let demand_nonempty = self
            .current_demand(morsel_id)?
            .boolean_summary()?
            .true_count
            != 0;
        for use_idx in 0..self.morsels[morsel_id.0].uses.len() {
            let use_ = self.morsels[morsel_id.0].uses[use_idx].clone();
            let should_prepare = use_.necessity == Necessity::Required
                || (!sealed && self.policy == SchedulePolicy::ProjectionPrefetch)
                || (sealed && demand_nonempty);
            if !should_prepare {
                continue;
            }
            let necessity = if use_.necessity == Necessity::Required || sealed {
                Necessity::Required
            } else {
                Necessity::Candidate
            };
            if let Some(update) = self.ensure_resource(use_.resource, necessity)? {
                updates.push(update);
                return Ok(true);
            }
        }

        if !self.morsels[morsel_id.0].sealed {
            if let Some(conjunct_idx) = self.next_predicate(morsel_id) {
                let predicates_in_flight = self.morsels[morsel_id.0]
                    .conjuncts
                    .iter()
                    .filter(|conjunct| conjunct.predicate_task.is_some() && !conjunct.combined)
                    .count();
                let predicate_window = self.predicate_window(morsel_id, conjunct_idx);
                let pipeline_blocked = predicates_in_flight >= predicate_window;
                if pipeline_blocked {
                    // Adopt completed predicates before capturing more work from this demand.
                    if matches!(
                        self.policy,
                        SchedulePolicy::AdaptivePredicates { .. }
                            | SchedulePolicy::LegacyAdaptivePredicates { .. }
                    ) {
                        self.metrics.adaptive_predicate_waits += 1;
                    }
                } else {
                    let conjunct = &self.morsels[morsel_id.0].conjuncts[conjunct_idx];
                    if let Some(resource) = conjunct.slices.iter().find_map(|slice| {
                        let resource_array = self.resources[slice.resource.0].array_slot;
                        self.array_slot(resource_array)
                            .ready()
                            .is_none()
                            .then_some(slice.resource)
                    }) {
                        self.morsels[morsel_id.0].waiting_on = Some(resource);
                        return Ok(false);
                    }
                    let first_unoffered = self.morsels[morsel_id.0]
                        .conjuncts
                        .iter()
                        .position(|conjunct| conjunct.predicate_task.is_none());
                    if first_unoffered != Some(conjunct_idx)
                        && matches!(
                            self.policy,
                            SchedulePolicy::AdaptivePredicates { .. }
                                | SchedulePolicy::LegacyAdaptivePredicates { .. }
                        )
                    {
                        self.metrics.predicate_reorders += 1;
                    }
                    let demand = self.morsels[morsel_id.0].demand_slot;
                    let output_slot = conjunct.result_slot;
                    let local_ranges = conjunct
                        .slices
                        .iter()
                        .map(|slice| slice.local_range.clone())
                        .collect();
                    let mut inputs = conjunct
                        .slices
                        .iter()
                        .map(|slice| InputSlot::Array(self.resources[slice.resource.0].array_slot))
                        .collect::<SmallVec<_>>();
                    inputs.push(InputSlot::Array(demand));
                    let predicate = self.query.conjuncts[conjunct_idx].predicate;
                    let version = self.morsels[morsel_id.0].demand_version;
                    let input_true_count = self
                        .current_demand(morsel_id)?
                        .boolean_summary()?
                        .true_count;
                    let task = self.offer_task(
                        TaskOwner::Morsel(morsel_id),
                        WorkClass::Cpu,
                        Necessity::Required,
                        inputs,
                        OutputSlot::Array(output_slot),
                        Operation::EvaluatePredicate {
                            conjunct: conjunct_idx,
                            local_ranges,
                            predicate,
                            demand_version: version,
                            input_true_count,
                        },
                    )?;
                    self.morsels[morsel_id.0].conjuncts[conjunct_idx].predicate_task =
                        Some(task.id);
                    self.morsels[morsel_id.0].predicate_offer_cursor += 1;
                    if matches!(
                        self.policy,
                        SchedulePolicy::AdaptivePredicates { .. }
                            | SchedulePolicy::LegacyAdaptivePredicates { .. }
                    ) && predicates_in_flight != 0
                    {
                        self.metrics.adaptive_predicate_launches += 1;
                    }
                    updates.push(TaskUpdate::Offer(task));
                    if self.trace_enabled {
                        self.push_trace(
                            morsel_id,
                            format!("event=predicate_offer conjunct={conjunct_idx}"),
                        );
                    }
                    return Ok(true);
                }
            }

            if self.morsels[morsel_id.0].combine.is_none() {
                for conjunct_idx in 0..self.query.conjuncts.len() {
                    let state = &self.morsels[morsel_id.0].conjuncts[conjunct_idx];
                    let predicate = state.result_slot;
                    let predicate_task = state.predicate_task;
                    if state.combined || self.array_slot(predicate).ready().is_none() {
                        continue;
                    }
                    let predicate_task = predicate_task.ok_or_else(|| {
                        vortex_error::vortex_err!("resolved predicate has no producing task")
                    })?;
                    let (predicate_version, snapshot) = {
                        let task = &self.tasks[predicate_task.0].task;
                        let version = match task.operation {
                            Operation::EvaluatePredicate { demand_version, .. } => demand_version,
                            _ => {
                                vortex_bail!("predicate slot was produced by a non-predicate task")
                            }
                        };
                        let Some(InputSlot::Array(snapshot)) = task.inputs.last().copied() else {
                            vortex_bail!("predicate task has no demand input");
                        };
                        (version, snapshot)
                    };
                    let version = self.morsels[morsel_id.0].demand_version;
                    if predicate_version == version {
                        self.adopt_demand(
                            morsel_id,
                            predicate,
                            conjunct_idx,
                            predicate_task,
                            true,
                        )?;
                        return Ok(true);
                    }
                    let predicate_count = self
                        .array_slot(predicate)
                        .ready()
                        .ok_or_else(|| vortex_error::vortex_err!("predicate is not resolved"))?
                        .boolean_summary()?
                        .true_count;
                    let snapshot_count = self
                        .array_slot(snapshot)
                        .ready()
                        .ok_or_else(|| {
                            vortex_error::vortex_err!("predicate demand snapshot is not resolved")
                        })?
                        .boolean_summary()?
                        .true_count;
                    if predicate_count == snapshot_count {
                        self.morsels[morsel_id.0].conjuncts[conjunct_idx].combined = true;
                        self.metrics.demand_noop_adoptions += 1;
                        if self.trace_enabled {
                            let current_count = self
                                .current_demand(morsel_id)?
                                .boolean_summary()?
                                .true_count;
                            self.push_trace(
                                morsel_id,
                                format!(
                                    "event=demand_adopt method=noop version={} previous_rows={} rows={} task={}",
                                    version.0,
                                    current_count,
                                    current_count,
                                    predicate_task.0,
                                ),
                            );
                        }
                        return Ok(true);
                    }
                    let current = self.morsels[morsel_id.0].demand_slot;
                    let output_slot = self.alloc_morsel_array(morsel_id);
                    let task = self.offer_task(
                        TaskOwner::Morsel(morsel_id),
                        WorkClass::Cpu,
                        Necessity::Required,
                        smallvec![InputSlot::Array(current), InputSlot::Array(predicate)],
                        OutputSlot::Array(output_slot),
                        Operation::CombineDemand {
                            demand_version: version,
                        },
                    )?;
                    self.morsels[morsel_id.0].combine = Some(PendingCombine {
                        task: task.id,
                        output_slot,
                        conjunct: conjunct_idx,
                        source_version: version,
                    });
                    updates.push(TaskUpdate::Offer(task));
                    return Ok(true);
                }
            }

            if self.morsels[morsel_id.0].combine.is_none()
                && self.morsels[morsel_id.0]
                    .conjuncts
                    .iter()
                    .all(|state| state.combined)
            {
                self.morsels[morsel_id.0].sealed = true;
                if self.trace_enabled {
                    self.push_trace(morsel_id, "event=demand_seal".to_string());
                }
                return Ok(true);
            }
            return Ok(false);
        }

        let true_count = self
            .current_demand(morsel_id)?
            .boolean_summary()?
            .true_count;
        if true_count != 0 {
            let projection_idx = self.morsels[morsel_id.0].projection_offer_cursor;
            if projection_idx < self.morsels[morsel_id.0].projections.len() {
                let projection = self.morsels[morsel_id.0].projections[projection_idx].clone();
                if let Some(resource) = projection.slices.iter().find_map(|slice| {
                    let resource_array = self.resources[slice.resource.0].array_slot;
                    self.array_slot(resource_array)
                        .ready()
                        .is_none()
                        .then_some(slice.resource)
                }) {
                    self.morsels[morsel_id.0].waiting_on = Some(resource);
                    return Ok(false);
                }
                let demand = self.morsels[morsel_id.0].demand_slot;
                let local_ranges = projection
                    .slices
                    .iter()
                    .map(|slice| slice.local_range.clone())
                    .collect();
                let mut inputs = projection
                    .slices
                    .iter()
                    .map(|slice| InputSlot::Array(self.resources[slice.resource.0].array_slot))
                    .collect::<SmallVec<_>>();
                inputs.push(InputSlot::Array(demand));
                let task = self.offer_task(
                    TaskOwner::Morsel(morsel_id),
                    WorkClass::Cpu,
                    Necessity::Required,
                    inputs,
                    OutputSlot::Array(projection.output_slot),
                    Operation::SelectFlat { local_ranges },
                )?;
                self.morsels[morsel_id.0].projections[projection_idx].task = Some(task.id);
                self.morsels[morsel_id.0].projection_offer_cursor += 1;
                updates.push(TaskUpdate::Offer(task));
                return Ok(true);
            }
        }

        if self.morsels[morsel_id.0].pack_task.is_none() {
            let all_ready = true_count == 0
                || self.morsels[morsel_id.0]
                    .projections
                    .iter()
                    .all(|projection| self.array_slot(projection.output_slot).ready().is_some());
            if all_ready {
                let inputs = if true_count == 0 {
                    SmallVec::new()
                } else {
                    self.morsels[morsel_id.0]
                        .projections
                        .iter()
                        .map(|projection| InputSlot::Array(projection.output_slot))
                        .collect()
                };
                let output_slot = self.morsels[morsel_id.0].pack_slot;
                let task = self.offer_task(
                    TaskOwner::Morsel(morsel_id),
                    WorkClass::Cpu,
                    Necessity::Required,
                    inputs,
                    OutputSlot::Array(output_slot),
                    Operation::PackStruct {
                        names: self.projection_names.clone(),
                        len: true_count,
                    },
                )?;
                self.morsels[morsel_id.0].pack_task = Some(task.id);
                updates.push(TaskUpdate::Offer(task));
                return Ok(true);
            }
        }

        let pack_slot = self.morsels[morsel_id.0].pack_slot;
        if let Some(array) = self.array_slot_mut(pack_slot).take_ready() {
            let demand_slot = self.morsels[morsel_id.0].demand_slot;
            let demand = self
                .array_slot_mut(demand_slot)
                .take_ready()
                .ok_or_else(|| vortex_error::vortex_err!("current demand is not resolved"))?;
            *output = Some(ExecBatch {
                coverage: self.morsels[morsel_id.0].root_range.clone(),
                selection: demand.array,
                array: array.array,
            });
            self.retire(morsel_id)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn next_predicate(&self, morsel: MorselId) -> Option<usize> {
        let legacy = matches!(self.policy, SchedulePolicy::LegacyAdaptivePredicates { .. });
        let first_unoffered = self.morsels[morsel.0]
            .conjuncts
            .iter()
            .position(|conjunct| {
                (legacy || !conjunct.combined) && conjunct.predicate_task.is_none()
            })?;
        if !matches!(
            self.policy,
            SchedulePolicy::AdaptivePredicates { .. }
                | SchedulePolicy::LegacyAdaptivePredicates { .. }
        ) {
            return Some(first_unoffered);
        }
        self.morsels[morsel.0]
            .conjuncts
            .iter()
            .enumerate()
            .filter(|(_, conjunct)| {
                (legacy || !conjunct.combined) && conjunct.predicate_task.is_none()
            })
            .max_by(|(left, _), (right, _)| {
                self.predicate_score(*left)
                    .total_cmp(&self.predicate_score(*right))
                    .then_with(|| right.cmp(left))
            })
            .map(|(index, _)| index)
    }

    fn predicate_window(&self, morsel: MorselId, next: usize) -> usize {
        let concurrency = match self.policy {
            SchedulePolicy::AdaptivePredicates { concurrency }
            | SchedulePolicy::LegacyAdaptivePredicates { concurrency } => concurrency,
            SchedulePolicy::PredicateFirst => return 1,
            _ => return usize::MAX,
        };
        let supply_window = concurrency.max(1).div_ceil(self.morsels.len().max(1));
        let outstanding = self.morsels[morsel.0]
            .conjuncts
            .iter()
            .enumerate()
            .filter(|(_, conjunct)| conjunct.predicate_task.is_some() && !conjunct.combined)
            .map(|(index, _)| index)
            .collect::<SmallVec<[usize; 8]>>();
        if outstanding.is_empty() || outstanding.len() >= supply_window {
            return outstanding.len().max(1);
        }
        if self.predicate_stats[next].samples == 0
            || outstanding
                .iter()
                .any(|index| self.predicate_stats[*index].samples == 0)
        {
            return outstanding.len();
        }

        let fallback_cost = self
            .predicate_stats
            .iter()
            .filter(|stats| stats.samples != 0)
            .map(|stats| stats.duration_ns(0.0))
            .sum::<f64>()
            / self
                .predicate_stats
                .iter()
                .filter(|stats| stats.samples != 0)
                .count()
                .max(1) as f64;
        let fallback_cost = fallback_cost.max(10_000.0);
        let outstanding_survival = outstanding.iter().fold(1.0, |survival, index| {
            survival * self.expected_survival(*index)
        });
        let outstanding_latency = outstanding
            .iter()
            .map(|index| self.predicate_stats[*index].duration_ns(fallback_cost))
            .fold(0.0, f64::max);
        let next_cost = self.predicate_stats[next].duration_ns(fallback_cost);

        // Mask traversal remains even after pruning; comparisons scale with surviving candidates.
        let next_after_wait = next_cost * (0.5 + 0.5 * outstanding_survival);
        let serial_latency = outstanding_latency + next_after_wait;
        let parallel_latency = outstanding_latency.max(next_cost) + 3_000.0;
        if parallel_latency < serial_latency {
            supply_window
        } else {
            outstanding.len()
        }
    }

    fn predicate_score(&self, conjunct: usize) -> f64 {
        let stats = &self.predicate_stats[conjunct];
        let fallback_cost = 10_000.0;
        (1.0 - self.expected_survival(conjunct)) / stats.duration_ns(fallback_cost)
    }

    fn expected_survival(&self, conjunct: usize) -> f64 {
        let prior = match self.query.conjuncts[conjunct].predicate {
            super::Predicate::Equal(_) => 0.1,
            super::Predicate::LessThan(_) | super::Predicate::GreaterThan(_) => 0.5,
        };
        self.predicate_stats[conjunct].survival(prior)
    }

    fn adopt_combine(&mut self, morsel: MorselId) -> VortexResult<bool> {
        let Some(pending) = self.morsels[morsel.0].combine.clone() else {
            return Ok(false);
        };
        if self.array_slot(pending.output_slot).ready().is_none() {
            return Ok(false);
        }
        if pending.source_version != self.morsels[morsel.0].demand_version {
            vortex_bail!("demand combination completed against a non-current version");
        }
        self.adopt_demand(
            morsel,
            pending.output_slot,
            pending.conjunct,
            pending.task,
            false,
        )?;
        self.morsels[morsel.0].combine = None;
        Ok(true)
    }

    fn adopt_demand(
        &mut self,
        morsel: MorselId,
        output_slot: ArraySlotId,
        conjunct: usize,
        task: TaskId,
        direct: bool,
    ) -> VortexResult<()> {
        let previous = self.current_demand(morsel)?.boolean_summary()?.true_count;
        let current = self
            .array_slot(output_slot)
            .ready()
            .ok_or_else(|| vortex_error::vortex_err!("adopted demand is not resolved"))?
            .boolean_summary()?
            .true_count;
        if current > previous {
            vortex_bail!("demand grew from {previous} to {current} rows");
        }
        self.morsels[morsel.0].demand_slot = output_slot;
        self.morsels[morsel.0].demand_version.0 += 1;
        self.morsels[morsel.0].conjuncts[conjunct].combined = true;
        self.metrics.demand_rows_current -= previous - current;
        if direct {
            self.metrics.demand_direct_adoptions += 1;
        } else {
            self.metrics.demand_combinations += 1;
        }
        if current == 0 {
            for conjunct in &mut self.morsels[morsel.0].conjuncts {
                conjunct.combined = true;
            }
            self.morsels[morsel.0].predicate_offer_cursor = self.query.conjuncts.len();
            self.morsels[morsel.0].cancelling = true;
        }
        if self.trace_enabled {
            self.push_trace(
                morsel,
                format!(
                    "event=demand_adopt method={} version={} previous_rows={previous} rows={current} task={}",
                    if direct { "direct" } else { "combine" },
                    self.morsels[morsel.0].demand_version.0,
                    task.0,
                ),
            );
        }
        Ok(())
    }

    fn cancel_one_morsel_offer(
        &mut self,
        morsel: MorselId,
        updates: &mut Vec<TaskUpdate>,
    ) -> VortexResult<bool> {
        if !self.morsels[morsel.0].cancelling {
            return Ok(false);
        }
        let tasks = self.morsels[morsel.0].tasks.clone();
        for task in tasks.iter().copied() {
            if self.tasks[task.0].status == TaskStatus::Offered {
                updates.push(self.revoke(task)?);
                return Ok(true);
            }
        }
        for task in tasks {
            match self.tasks[task.0].status {
                TaskStatus::Running => self.tasks[task.0].unwanted = true,
                TaskStatus::Completed | TaskStatus::Revoked => {}
                TaskStatus::Offered => unreachable!("offered tasks were handled above"),
            }
        }
        self.morsels[morsel.0].cancelling = false;
        Ok(true)
    }

    fn ensure_resource(
        &mut self,
        resource: ResourceId,
        necessity: Necessity,
    ) -> VortexResult<Option<TaskUpdate>> {
        let (segment_slot, array_slot) = {
            let node = &self.resources[resource.0];
            (node.segment_slot, node.array_slot)
        };
        if self.segment_slot(segment_slot).ready().is_none() {
            if let Some(task_id) = self.resources[resource.0].read_task {
                return self.promote_if_needed(task_id, necessity);
            }
            let segment = self.resources[resource.0].segment;
            let task = self.offer_task(
                TaskOwner::Resource(resource),
                WorkClass::Io,
                necessity,
                SmallVec::new(),
                OutputSlot::Segment(segment_slot),
                Operation::Read { segment },
            )?;
            self.resources[resource.0].read_task = Some(task.id);
            return Ok(Some(TaskUpdate::Offer(task)));
        }
        if self.array_slot(array_slot).ready().is_none() {
            if let Some(task_id) = self.resources[resource.0].decode_task {
                return self.promote_if_needed(task_id, necessity);
            }
            let row_count = self.resources[resource.0].row_count;
            let encoding = self.resources[resource.0].encoding.clone();
            let task = self.offer_task(
                TaskOwner::Resource(resource),
                WorkClass::Cpu,
                necessity,
                smallvec![InputSlot::Segment(segment_slot)],
                OutputSlot::Array(array_slot),
                Operation::DecodeFlat {
                    encoding,
                    row_count,
                },
            )?;
            self.resources[resource.0].decode_task = Some(task.id);
            return Ok(Some(TaskUpdate::Offer(task)));
        }
        Ok(None)
    }

    fn promote_if_needed(
        &mut self,
        task: TaskId,
        necessity: Necessity,
    ) -> VortexResult<Option<TaskUpdate>> {
        let stored = &mut self.tasks[task.0];
        if necessity == Necessity::Required
            && stored.task.necessity == Necessity::Candidate
            && stored.status == TaskStatus::Offered
        {
            stored.task.necessity = Necessity::Required;
            self.metrics.tasks_promoted += 1;
            return Ok(Some(TaskUpdate::Promote(task)));
        }
        Ok(None)
    }

    fn offer_task(
        &mut self,
        owner: TaskOwner,
        class: WorkClass,
        necessity: Necessity,
        inputs: SmallVec<[InputSlot; 2]>,
        output: OutputSlot,
        operation: Operation,
    ) -> VortexResult<Task> {
        let id = TaskId(self.tasks.len());
        self.reserve_output(output, id)?;
        let task = Task {
            id,
            class,
            necessity,
            inputs,
            output,
            operation,
        };
        self.tasks.push(StoredTask {
            task: task.clone(),
            status: TaskStatus::Offered,
            owner,
            leases: SmallVec::new(),
            unwanted: false,
        });
        if let TaskOwner::Morsel(morsel) = owner {
            self.morsels[morsel.0].tasks.push(id);
        }
        self.metrics.tasks_offered += 1;
        match class {
            WorkClass::Io => self.metrics.io_offered += 1,
            WorkClass::Cpu => self.metrics.cpu_offered += 1,
        }
        Ok(task)
    }

    pub fn claim(&mut self, task: TaskId) -> VortexResult<ClaimResult> {
        let Some(stored) = self.tasks.get(task.0) else {
            vortex_bail!("unknown task {}", task.0);
        };
        if stored.status == TaskStatus::Revoked {
            return Ok(ClaimResult::Revoked);
        }
        if stored.status != TaskStatus::Offered {
            vortex_bail!("task {} cannot be claimed twice", task.0);
        }
        let inputs = stored
            .task
            .inputs
            .iter()
            .map(|slot| self.resolve_input(*slot))
            .collect::<VortexResult<SmallVec<_>>>()?;
        let output = stored.task.output;
        let operation = stored.task.operation.clone();
        let mut leases = SmallVec::<[ResourceId; 2]>::new();
        for resource in stored
            .task
            .inputs
            .iter()
            .filter_map(|slot| self.resource_for_input(*slot))
        {
            if !leases.contains(&resource) {
                leases.push(resource);
            }
        }
        self.claim_output(output, task)?;
        for resource in &leases {
            self.resources[resource.0].leases += 1;
        }
        let stored = &mut self.tasks[task.0];
        stored.status = TaskStatus::Running;
        stored.leases = leases;
        self.metrics.tasks_claimed += 1;
        Ok(ClaimResult::Runnable(RunnableTask {
            id: task,
            inputs,
            output,
            operation,
        }))
    }

    pub fn complete(&mut self, completion: Completion) -> VortexResult<()> {
        let Some(stored) = self.tasks.get(completion.task.0) else {
            vortex_bail!("completion references unknown task {}", completion.task.0);
        };
        if stored.status != TaskStatus::Running {
            vortex_bail!("task {} is not running", completion.task.0);
        }
        let expected = stored.task.output;
        if completion.output != expected {
            self.fail_task(completion.task);
            vortex_bail!("task {} completed into the wrong slot", completion.task.0);
        }
        let predicate_observation = match (
            &self.tasks[completion.task.0].task.operation,
            completion.result.as_ref(),
        ) {
            (
                Operation::EvaluatePredicate {
                    conjunct,
                    input_true_count,
                    ..
                },
                Ok(ResolvedValue::Array(array)),
            ) => array.boolean_summary().ok().map(|summary| {
                (
                    *conjunct,
                    *input_true_count,
                    summary.true_count,
                    completion.elapsed_ns,
                )
            }),
            _ => None,
        };
        let value = match completion.result {
            Ok(value) => value,
            Err(error) => {
                self.fail_task(completion.task);
                return Err(error);
            }
        };
        if !resolved_kind_matches(expected, &value) {
            self.fail_task(completion.task);
            vortex_bail!(
                "task {} returned the wrong resolved value kind",
                completion.task.0
            );
        }
        if matches!(
            self.tasks[completion.task.0].task.operation,
            Operation::EvaluatePredicate { .. } | Operation::CombineDemand { .. }
        ) {
            let ResolvedValue::Array(array) = &value else {
                unreachable!()
            };
            let Ok(summary) = array.boolean_summary() else {
                self.fail_task(completion.task);
                vortex_bail!("task {} omitted its mask summary", completion.task.0);
            };
            if summary.len != array.array.len() || summary.true_count > summary.len {
                self.fail_task(completion.task);
                vortex_bail!(
                    "task {} returned an invalid mask summary",
                    completion.task.0
                );
            }
        }
        if self.tasks[completion.task.0].unwanted {
            self.discard_output(expected);
        } else {
            self.install_output(expected, completion.task, value)?;
        }
        self.finish_task(completion.task);
        if let Some((conjunct, input_rows, output_rows, elapsed_ns)) = predicate_observation {
            self.predicate_stats[conjunct].observe(input_rows, output_rows, elapsed_ns);
        }
        self.metrics.tasks_completed += 1;
        Ok(())
    }

    pub fn revoke(&mut self, task: TaskId) -> VortexResult<TaskUpdate> {
        let Some(stored) = self.tasks.get(task.0) else {
            vortex_bail!("unknown task {}", task.0);
        };
        if stored.status != TaskStatus::Offered {
            vortex_bail!("only an offered task can be revoked");
        }
        let output = stored.task.output;
        self.revoke_output(output, task)?;
        self.tasks[task.0].status = TaskStatus::Revoked;
        self.metrics.tasks_revoked += 1;
        Ok(TaskUpdate::Revoke(task))
    }

    fn finish_task(&mut self, task: TaskId) {
        let leases = std::mem::take(&mut self.tasks[task.0].leases);
        for resource in leases {
            self.resources[resource.0].leases -= 1;
            self.cleanup_resource(resource);
        }
        self.tasks[task.0].status = TaskStatus::Completed;
    }

    fn fail_task(&mut self, task: TaskId) {
        let output = self.tasks[task.0].task.output;
        let owner = self.tasks[task.0].owner;
        self.fail_output(output, task);
        self.finish_task(task);
        self.discard_output(output);
        if let TaskOwner::Resource(resource) = owner {
            if self.resources[resource.0].read_task == Some(task) {
                self.resources[resource.0].read_task = None;
            }
            if self.resources[resource.0].decode_task == Some(task) {
                self.resources[resource.0].decode_task = None;
            }
        }
    }

    fn retire(&mut self, morsel: MorselId) -> VortexResult<()> {
        for use_idx in 0..self.morsels[morsel.0].uses.len() {
            let use_ = &self.morsels[morsel.0].uses[use_idx];
            if !use_.joined {
                continue;
            }
            let resource = use_.resource;
            debug_assert_ne!(self.resources[resource.0].joined_users, 0);
            self.resources[resource.0].joined_users -= 1;
            self.cleanup_resource(resource);
        }
        for task in self.morsels[morsel.0].tasks.iter().copied() {
            match self.tasks[task.0].status {
                TaskStatus::Offered => {
                    vortex_bail!("morsel {} retired with offered task {}", morsel.0, task.0)
                }
                TaskStatus::Running => self.tasks[task.0].unwanted = true,
                TaskStatus::Completed | TaskStatus::Revoked => {}
            }
        }
        for slot in &mut self.morsels[morsel.0].arrays {
            slot.discard();
        }
        self.morsels[morsel.0].retired = true;
        if self.trace_enabled {
            self.push_trace(morsel, "event=retire".to_string());
        }
        Ok(())
    }

    fn cleanup_resource(&mut self, resource: ResourceId) {
        let lifetime = self.resources[resource.0].lifetime();
        if lifetime == ResourceLifetime::Dead
            || (lifetime == ResourceLifetime::Reusable
                && self.retention == RetentionPolicy::EvictWhenUnpinned)
        {
            let segment = self.resources[resource.0].segment_slot;
            let array = self.resources[resource.0].array_slot;
            if !matches!(self.segment_slot(segment).state, SlotState::Running(_)) {
                self.segment_slot_mut(segment).discard();
                self.resources[resource.0].read_task = None;
            }
            if !matches!(self.array_slot(array).state, SlotState::Running(_)) {
                self.array_slot_mut(array).discard();
                self.resources[resource.0].decode_task = None;
            }
        }
    }

    fn current_demand(&self, morsel: MorselId) -> VortexResult<&ResolvedArray> {
        self.array_slot(self.morsels[morsel.0].demand_slot)
            .ready()
            .ok_or_else(|| vortex_error::vortex_err!("current demand is not resolved"))
    }

    fn alloc_morsel_array(&mut self, morsel: MorselId) -> ArraySlotId {
        let slot = ArraySlotId::Morsel(morsel, self.morsels[morsel.0].arrays.len());
        self.morsels[morsel.0].arrays.push(Slot::default());
        self.metrics.morsel_slots += 1;
        slot
    }

    fn reserve_output(&mut self, output: OutputSlot, task: TaskId) -> VortexResult<()> {
        match output {
            OutputSlot::Segment(slot) => self.segment_slot_mut(slot).reserve(task),
            OutputSlot::Array(slot) => self.array_slot_mut(slot).reserve(task),
        }
    }

    fn claim_output(&mut self, output: OutputSlot, task: TaskId) -> VortexResult<()> {
        match output {
            OutputSlot::Segment(slot) => self.segment_slot_mut(slot).claim(task),
            OutputSlot::Array(slot) => self.array_slot_mut(slot).claim(task),
        }
    }

    fn revoke_output(&mut self, output: OutputSlot, task: TaskId) -> VortexResult<()> {
        match output {
            OutputSlot::Segment(slot) => self.segment_slot_mut(slot).revoke(task),
            OutputSlot::Array(slot) => self.array_slot_mut(slot).revoke(task),
        }
    }

    fn fail_output(&mut self, output: OutputSlot, task: TaskId) {
        match output {
            OutputSlot::Segment(slot) => self.segment_slot_mut(slot).fail(task),
            OutputSlot::Array(slot) => self.array_slot_mut(slot).fail(task),
        }
    }

    fn discard_output(&mut self, output: OutputSlot) {
        match output {
            OutputSlot::Segment(slot) => self.segment_slot_mut(slot).discard(),
            OutputSlot::Array(slot) => self.array_slot_mut(slot).discard(),
        }
    }

    fn install_output(
        &mut self,
        output: OutputSlot,
        task: TaskId,
        value: ResolvedValue,
    ) -> VortexResult<()> {
        match (output, value) {
            (OutputSlot::Segment(slot), ResolvedValue::Segment(value)) => {
                self.segment_slot_mut(slot).install(task, value)
            }
            (OutputSlot::Array(slot), ResolvedValue::Array(value)) => {
                self.array_slot_mut(slot).install(task, value)
            }
            _ => unreachable!("resolved kind checked before installation"),
        }
    }

    fn resolve_input(&self, input: InputSlot) -> VortexResult<ResolvedValue> {
        match input {
            InputSlot::Segment(slot) => self
                .segment_slot(slot)
                .ready()
                .cloned()
                .map(ResolvedValue::Segment)
                .ok_or_else(|| vortex_error::vortex_err!("segment input is not ready")),
            InputSlot::Array(slot) => self
                .array_slot(slot)
                .ready()
                .cloned()
                .map(ResolvedValue::Array)
                .ok_or_else(|| vortex_error::vortex_err!("array input is not ready")),
        }
    }

    fn resource_for_input(&self, input: InputSlot) -> Option<ResourceId> {
        let resource = match input {
            InputSlot::Segment(SegmentSlotId::Scan(index))
            | InputSlot::Array(ArraySlotId::Scan(index)) => self.resources.get(index),
            InputSlot::Array(ArraySlotId::Morsel(..)) => None,
        }?;
        let belongs_to_resource = match input {
            InputSlot::Segment(slot) => resource.segment_slot == slot,
            InputSlot::Array(slot) => resource.array_slot == slot,
        };
        belongs_to_resource.then_some(resource.id)
    }

    fn segment_slot(&self, slot: SegmentSlotId) -> &Slot<vortex_array::buffer::BufferHandle> {
        match slot {
            SegmentSlotId::Scan(index) => &self.segment_slots[index],
        }
    }

    fn segment_slot_mut(
        &mut self,
        slot: SegmentSlotId,
    ) -> &mut Slot<vortex_array::buffer::BufferHandle> {
        match slot {
            SegmentSlotId::Scan(index) => &mut self.segment_slots[index],
        }
    }

    fn array_slot(&self, slot: ArraySlotId) -> &Slot<ResolvedArray> {
        match slot {
            ArraySlotId::Scan(index) => &self.scan_array_slots[index],
            ArraySlotId::Morsel(morsel, index) => &self.morsels[morsel.0].arrays[index],
        }
    }

    fn array_slot_mut(&mut self, slot: ArraySlotId) -> &mut Slot<ResolvedArray> {
        match slot {
            ArraySlotId::Scan(index) => &mut self.scan_array_slots[index],
            ArraySlotId::Morsel(morsel, index) => &mut self.morsels[morsel.0].arrays[index],
        }
    }

    fn push_trace(&mut self, morsel: MorselId, message: String) {
        self.record_trace(Some(morsel), format_args!("{message}"));
    }
}

fn resolved_kind_matches(output: OutputSlot, value: &ResolvedValue) -> bool {
    matches!(
        (output, value),
        (OutputSlot::Segment(_), ResolvedValue::Segment(_))
            | (OutputSlot::Array(_), ResolvedValue::Array(_))
    )
}
