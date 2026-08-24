// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An extensible streaming pipeline executor for the self-paced experiment.
//!
//! The scheduler knows exactly one thing: a [`MorselPipeline`] that turns a morsel row range into
//! an [`ExecBatch`]. Threads self-schedule morsels from one shared cursor and drive their
//! pipeline inline, so new execution nodes are added by implementing a trait, never by touching
//! the scheduler.
//!
//! Row-domain relationships are executor-vtable methods, not materialized mappings. Each node
//! contributes a **down demand transform** and **up result transforms**, modeled on the layout's
//! native metadata:
//!
//! - [`FieldDomain::push_demand`] cuts a parent-range demand into child segments with
//!   demanded-row counts, so empty children are skipped before any read;
//! - [`FieldDomain::pull_mask`] / [`FieldDomain::pull_array`] reassemble per-child masks or
//!   decoded arrays back into the parent domain.
//!
//! Planning does no compute: building the pipeline wires the topology (which domains exist, which
//! fields the query touches, the output names) and the scan computes its morsel splits once; the
//! struct node shares one refcounted demand handle with every child. All remaining work — the
//! offset arithmetic of cutting, demand pricing, reads, evaluation — happens at execution, in
//! parallel on the owning threads. (An eager plan-time materialization of the cutting was built
//! and measured slower, because it serialized work the threads do for ~free; see the findings.)
//!
//! [`ConcatDomain`] implements the chunked relationship with the chunk-offset prefix sums. The
//! struct relationship is the identity: one demand mask shared by refcount across children and a
//! zero-copy struct pack upward. The shared per-morsel demand is computed by a pluggable
//! [`DemandPolicy`] ([`CascadeDemand`] or [`EagerDemand`]); policies and projection speak only to
//! [`FieldSet`], so they are agnostic to the relationship kind behind each field.

use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;

use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use super::ExecBatch;
use super::FieldId;
use super::FlatPlan;
use super::Metrics;
use super::RunResult;
use super::ScanQuery;
use super::SourcePlan;
use super::evaluate::decode_flat;
use super::evaluate::evaluate_predicate_full;
use super::evaluate::evaluate_predicate_slice;
use super::evaluate::pack_struct_array;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Per-thread execution context: the segment source, the session, and this thread's decoded-chunk
/// cache. The cache is what lets a field consumed by both filter and projection decode once; its
/// entries are refcounted views over the source bytes and live for the thread's morsel group.
pub struct PipelineCtx<'a> {
    pub source: &'a dyn SegmentSource,
    pub session: &'a VortexSession,
    decoded: HashMap<SegmentId, ArrayRef>,
}

impl<'a> PipelineCtx<'a> {
    pub fn new(source: &'a dyn SegmentSource, session: &'a VortexSession) -> Self {
        Self {
            source,
            session,
            decoded: HashMap::new(),
        }
    }

    /// Read and decode one chunk, deduplicated per thread by segment identity.
    pub async fn decoded_chunk(&mut self, plan: &FlatPlan) -> VortexResult<ArrayRef> {
        if let Some(array) = self.decoded.get(&plan.segment) {
            return Ok(array.clone());
        }
        let segment = self.source.request(plan.segment).await?;
        let array = decode_flat(&segment, &plan.encoding, plan.row_count, self.session)?;
        self.decoded.insert(plan.segment, array.clone());
        Ok(array)
    }
}

/// One field's chunks at their native physical boundaries. Nothing here assumes alignment with
/// any other field.
#[derive(Clone, Debug)]
pub struct FieldChunks {
    pub field: FieldId,
    pub chunks: Vec<FlatPlan>,
}

/// Regroup a [`SourcePlan`] into per-field chunk lists. The plan's chunks happen to be aligned,
/// but every consumer below treats each field's list independently.
pub fn field_chunks(plan: &SourcePlan) -> Vec<FieldChunks> {
    let mut fields = (0..plan.field_names.len())
        .map(|field| FieldChunks {
            field: FieldId(field),
            chunks: Vec::new(),
        })
        .collect::<Vec<_>>();
    for chunk in &plan.chunks {
        for flat in &chunk.fields {
            fields[flat.field.0].chunks.push(flat.clone());
        }
    }
    fields
}

/// One child segment produced by a down demand transform: the physical leaf to read, the range it
/// covers in child-local and parent-local coordinates, how many of its rows are demanded, and the
/// demand restricted to it (`None` means every row is demanded).
pub struct ChildSegment<'p> {
    pub plan: &'p FlatPlan,
    pub chunk_local: Range<usize>,
    pub parent_local: Range<usize>,
    pub demanded: usize,
    pub demand: Option<BitBuffer>,
}

/// The executor-vtable seam for a row-domain relationship: demand flows down, results flow up.
/// Implementations model the relationship with the layout's own metadata rather than any shared
/// mapping structure.
pub trait FieldDomain: Send + Sync {
    /// Down demand transform: cut `range` (parent coordinates) into child segments, attaching to
    /// each its demanded-row count and demand slice so the caller can skip empty children.
    fn push_demand<'a>(
        &'a self,
        range: &Range<u64>,
        demand: Option<&BitBuffer>,
    ) -> VortexResult<Vec<ChildSegment<'a>>>;

    /// Up mask transform: reassemble per-segment masks, given in parent-local order and covering
    /// all of `range`, into one parent-domain mask.
    fn pull_mask(
        &self,
        range: &Range<u64>,
        parts: Vec<(Range<usize>, BitBuffer)>,
    ) -> VortexResult<BitBuffer>;

    /// Up array transform: reassemble the surviving segments' decoded arrays into one array in
    /// the parent domain, applying each segment's demand. `true_count` is the expected output
    /// row count under the demand that produced `segments`. `shared_mask` is the parent node's
    /// whole-range selection, built once per morsel; it applies whenever this field gathered the
    /// entire range, saving a per-field mask construction.
    fn pull_array(
        &self,
        segments: &[ChildSegment<'_>],
        arrays: Vec<ArrayRef>,
        true_count: usize,
        range_rows: usize,
        shared_mask: Option<&Mask>,
    ) -> VortexResult<ArrayRef>;
}

fn price_segment(
    parent_local: &Range<usize>,
    demand: Option<&BitBuffer>,
) -> (usize, Option<BitBuffer>) {
    match demand {
        None => (parent_local.len(), None),
        Some(demand) => (
            demand.count_range(parent_local.start, parent_local.end),
            Some(demand.slice(parent_local.clone())),
        ),
    }
}

/// The chunked (concatenation) relationship, modeled directly on the chunk-offset prefix sums:
/// binary search locates the first overlapping chunk, `count_range` prices each child under the
/// demand, and the up transforms are ordered appends / zero-copy chunk assembly.
pub struct ConcatDomain {
    chunks: FieldChunks,
}

impl ConcatDomain {
    pub fn new(chunks: FieldChunks) -> Self {
        Self { chunks }
    }
}

impl FieldDomain for ConcatDomain {
    fn push_demand<'a>(
        &'a self,
        range: &Range<u64>,
        demand: Option<&BitBuffer>,
    ) -> VortexResult<Vec<ChildSegment<'a>>> {
        let chunks = &self.chunks.chunks;
        let start = chunks.partition_point(|chunk| chunk.root_coverage.end <= range.start);
        let mut segments = Vec::new();
        let mut covered = range.start;
        for plan in &chunks[start..] {
            if plan.root_coverage.start >= range.end {
                break;
            }
            let overlap_start = plan.root_coverage.start.max(range.start);
            let overlap_end = plan.root_coverage.end.min(range.end);
            if overlap_start != covered {
                vortex_bail!(
                    "field {} chunks do not cover the range",
                    self.chunks.field.0
                );
            }
            covered = overlap_end;
            let parent_local = usize::try_from(overlap_start - range.start)?
                ..usize::try_from(overlap_end - range.start)?;
            let (demanded, segment_demand) = price_segment(&parent_local, demand);
            segments.push(ChildSegment {
                plan,
                chunk_local: usize::try_from(overlap_start - plan.root_coverage.start)?
                    ..usize::try_from(overlap_end - plan.root_coverage.start)?,
                parent_local,
                demanded,
                demand: segment_demand,
            });
        }
        if covered != range.end {
            vortex_bail!(
                "field {} chunks do not cover the range",
                self.chunks.field.0
            );
        }
        Ok(segments)
    }

    fn pull_mask(
        &self,
        range: &Range<u64>,
        parts: Vec<(Range<usize>, BitBuffer)>,
    ) -> VortexResult<BitBuffer> {
        let rows = usize::try_from(range.end - range.start)?;
        // A morsel covered by one segment needs no reassembly: the part is the mask.
        if let [(parent_local, part)] = parts.as_slice()
            && parent_local.start == 0
            && parent_local.end == rows
            && part.len() == rows
        {
            let (_, part) = parts.into_iter().next().vortex_expect("one part");
            return Ok(part);
        }
        let mut mask = BitBufferMut::with_capacity(rows);
        let mut covered = 0;
        for (parent_local, part) in parts {
            if parent_local.start != covered || part.len() != parent_local.len() {
                vortex_bail!("mask parts do not tile the parent range");
            }
            covered = parent_local.end;
            mask.append_buffer(&part);
        }
        if covered != rows {
            vortex_bail!("mask parts do not cover the parent range");
        }
        Ok(mask.freeze())
    }

    fn pull_array(
        &self,
        segments: &[ChildSegment<'_>],
        arrays: Vec<ArrayRef>,
        true_count: usize,
        range_rows: usize,
        shared_mask: Option<&Mask>,
    ) -> VortexResult<ArrayRef> {
        let mut parts = Vec::with_capacity(arrays.len());
        for (segment, array) in segments.iter().zip(arrays) {
            parts.push(array.slice(segment.chunk_local.clone())?);
        }
        let unfiltered = if parts.len() == 1 {
            parts.pop().vortex_expect("one part")
        } else {
            let dtype = parts
                .first()
                .ok_or_else(|| vortex_error::vortex_err!("no segments survived the demand"))?
                .dtype()
                .clone();
            ChunkedArray::try_new(parts, dtype)?.into_array()
        };
        let gathered_rows = segments
            .iter()
            .map(|segment| segment.parent_local.len())
            .sum::<usize>();
        if true_count == range_rows && gathered_rows == range_rows {
            return Ok(unfiltered);
        }
        // The segments were priced by the down transform, so the coverage check needs no bit
        // scan: their demanded counts must account for every selected row.
        let gathered_selected = segments
            .iter()
            .map(|segment| segment.demanded)
            .sum::<usize>();
        if gathered_selected != true_count {
            vortex_bail!("gathered segments do not cover every selected row");
        }
        if gathered_selected == gathered_rows {
            return Ok(unfiltered);
        }
        // A field that gathered the whole range selects by the parent's mask, built once.
        if gathered_rows == range_rows
            && let Some(mask) = shared_mask
        {
            return unfiltered.filter(mask.clone());
        }
        let included = if let [segment] = segments {
            segment
                .demand
                .clone()
                .vortex_expect("partial demand priced below its row count")
        } else {
            let mut included = BitBufferMut::with_capacity(gathered_rows);
            for segment in segments {
                match &segment.demand {
                    Some(demand) => included.append_buffer(demand),
                    None => included.append_buffer(&BitBuffer::new_set(segment.parent_local.len())),
                }
            }
            included.freeze()
        };
        unfiltered.filter(Mask::from_buffer(included))
    }
}

/// The per-morsel view every node and policy works through: the down demand transform for any
/// field, served from the plan-time compilation when the field has one and from the runtime
/// vtable path otherwise. Callers cannot tell the difference, which is the point.
pub struct FieldSet<'a> {
    domains: &'a [Box<dyn FieldDomain>],
    range: &'a Range<u64>,
}

impl<'a> FieldSet<'a> {
    /// Down demand transform for `field` over this morsel.
    pub fn push_demand(
        &self,
        field: FieldId,
        demand: Option<&BitBuffer>,
    ) -> VortexResult<Vec<ChildSegment<'a>>> {
        self.domains[field.0].push_demand(self.range, demand)
    }

    /// The field's vtable, for the up transforms.
    pub fn domain(&self, field: FieldId) -> &'a dyn FieldDomain {
        self.domains[field.0].as_ref()
    }

    pub fn range(&self) -> &'a Range<u64> {
        self.range
    }
}

fn primitive_i64(array: &ArrayRef, session: &VortexSession) -> VortexResult<Vec<i64>> {
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Primitive;
    use vortex_array::arrays::PrimitiveArray;
    let primitive = if array.is::<Primitive>() {
        array.as_::<Primitive>().into_owned()
    } else {
        let mut ctx = session.create_execution_ctx();
        array.clone().execute::<PrimitiveArray>(&mut ctx)?
    };
    Ok(primitive.as_slice::<i64>().to_vec())
}

/// Borrow the decoded chunk's i64 values without copying when it is already primitive.
fn with_i64_values<R>(
    array: &ArrayRef,
    session: &VortexSession,
    apply: impl FnOnce(&[i64]) -> R,
) -> VortexResult<R> {
    use vortex_array::arrays::Primitive;
    if array.is::<Primitive>() {
        let primitive = array.as_::<Primitive>();
        return Ok(apply(primitive.as_slice::<i64>()));
    }
    let values = primitive_i64(array, session)?;
    Ok(apply(&values))
}

/// The pluggable shared-demand computation: given the morsel's field set and the query, it
/// produces the morsel's demand mask (`None` means all rows survive). Implementations choose how
/// much work to avoid; the pipeline and scheduler are unaffected by the choice.
pub trait DemandPolicy: Send + Sync {
    fn name(&self) -> &'static str;

    fn morsel_demand<'a, 'c>(
        &'a self,
        ctx: &'a mut PipelineCtx<'c>,
        fields: &'a FieldSet<'a>,
        query: &'a ScanQuery,
    ) -> BoxFuture<'a, VortexResult<Option<BitBuffer>>>;
}

/// Evaluate conjuncts in query order against shrinking demand: rows already rejected are never
/// evaluated again, and a chunk whose demanded rows are zero is neither read nor decoded.
pub struct CascadeDemand;

impl DemandPolicy for CascadeDemand {
    fn name(&self) -> &'static str {
        "cascade"
    }

    fn morsel_demand<'a, 'c>(
        &'a self,
        ctx: &'a mut PipelineCtx<'c>,
        fields: &'a FieldSet<'a>,
        query: &'a ScanQuery,
    ) -> BoxFuture<'a, VortexResult<Option<BitBuffer>>> {
        Box::pin(async move {
            let range = fields.range();
            let mut demand: Option<BitBuffer> = None;
            for conjunct in &query.conjuncts {
                if demand.as_ref().is_some_and(|mask| mask.true_count() == 0) {
                    return Ok(demand);
                }
                let segments = fields.push_demand(conjunct.field, demand.as_ref())?;
                let mut parts = Vec::with_capacity(segments.len());
                for segment in segments {
                    if segment.demanded == 0 {
                        parts.push((
                            segment.parent_local.clone(),
                            BitBuffer::new_unset(segment.parent_local.len()),
                        ));
                        continue;
                    }
                    let array = ctx.decoded_chunk(segment.plan).await?;
                    let result = with_i64_values(&array, ctx.session, |values| {
                        let values = &values[segment.chunk_local.clone()];
                        match &segment.demand {
                            Some(demand) => evaluate_predicate_slice(
                                values,
                                demand,
                                segment.demanded,
                                conjunct.predicate,
                            ),
                            // Full demand needs no mask at all.
                            None => evaluate_predicate_full(values, conjunct.predicate),
                        }
                    })?;
                    parts.push((segment.parent_local.clone(), result));
                }
                // Every segment was evaluated against the current demand, so the pulled-up mask
                // is already a subset of it: adopt directly, no intersection needed.
                demand = Some(fields.domain(conjunct.field).pull_mask(range, parts)?);
            }
            Ok(demand)
        })
    }
}

/// The cascade with observed-selectivity ordering: each morsel evaluates conjuncts most-selective
/// first, using survival rates accumulated from earlier morsels (query order until observed).
/// Ordering is a pure execution choice — conjunction is commutative and every result is adopted
/// as a subset of the demand it was evaluated under — so output is identical to [`CascadeDemand`].
pub struct AdaptiveDemand {
    /// Per-conjunct (demanded rows, surviving rows) observations, shared across threads.
    stats: std::sync::OnceLock<Vec<(AtomicUsize, AtomicUsize)>>,
}

impl AdaptiveDemand {
    pub fn new() -> Self {
        Self {
            stats: std::sync::OnceLock::new(),
        }
    }

    fn order(&self, conjuncts: usize) -> Vec<usize> {
        let stats = self
            .stats
            .get_or_init(|| (0..conjuncts).map(|_| Default::default()).collect());
        let mut order = (0..conjuncts).collect::<Vec<_>>();
        // Unobserved conjuncts keep their query position via a neutral survival of 1.0 plus the
        // index epsilon, so the first morsels run in query order.
        order.sort_by(|&lhs, &rhs| {
            let survival = |idx: usize| {
                let (input, output) = &stats[idx];
                let input = input.load(Ordering::Relaxed);
                if input == 0 {
                    1.0 + idx as f64 * 1e-9
                } else {
                    output.load(Ordering::Relaxed) as f64 / input as f64
                }
            };
            survival(lhs).total_cmp(&survival(rhs))
        });
        order
    }

    fn observe(&self, conjunct: usize, demanded: usize, survived: usize) {
        if let Some(stats) = self.stats.get() {
            stats[conjunct].0.fetch_add(demanded, Ordering::Relaxed);
            stats[conjunct].1.fetch_add(survived, Ordering::Relaxed);
        }
    }
}

impl Default for AdaptiveDemand {
    fn default() -> Self {
        Self::new()
    }
}

impl DemandPolicy for AdaptiveDemand {
    fn name(&self) -> &'static str {
        "adaptive"
    }

    fn morsel_demand<'a, 'c>(
        &'a self,
        ctx: &'a mut PipelineCtx<'c>,
        fields: &'a FieldSet<'a>,
        query: &'a ScanQuery,
    ) -> BoxFuture<'a, VortexResult<Option<BitBuffer>>> {
        Box::pin(async move {
            let range = fields.range();
            let mut demand: Option<BitBuffer> = None;
            for conjunct_idx in self.order(query.conjuncts.len()) {
                let conjunct = &query.conjuncts[conjunct_idx];
                let demand_count = demand.as_ref().map(BitBuffer::true_count);
                if demand_count == Some(0) {
                    return Ok(demand);
                }
                // Dense demand makes gating cost more than it avoids (measured crossover):
                // evaluate the conjunct in full and intersect, exactly as the eager policy.
                if let (Some(current), Some(count)) = (&demand, demand_count)
                    && count * 2 >= current.len()
                {
                    let segments = fields.push_demand(conjunct.field, None)?;
                    let mut demanded_rows = 0;
                    let mut parts = Vec::with_capacity(segments.len());
                    for segment in segments {
                        demanded_rows += segment.parent_local.len();
                        let array = ctx.decoded_chunk(segment.plan).await?;
                        let result = with_i64_values(&array, ctx.session, |values| {
                            evaluate_predicate_full(
                                &values[segment.chunk_local.clone()],
                                conjunct.predicate,
                            )
                        })?;
                        parts.push((segment.parent_local.clone(), result));
                    }
                    let mask = fields.domain(conjunct.field).pull_mask(range, parts)?;
                    let mask = current & &mask;
                    self.observe(conjunct_idx, demanded_rows, mask.true_count());
                    demand = Some(mask);
                    continue;
                }
                let segments = fields.push_demand(conjunct.field, demand.as_ref())?;
                let mut demanded_rows = 0;
                let mut parts = Vec::with_capacity(segments.len());
                for segment in segments {
                    if segment.demanded == 0 {
                        parts.push((
                            segment.parent_local.clone(),
                            BitBuffer::new_unset(segment.parent_local.len()),
                        ));
                        continue;
                    }
                    demanded_rows += segment.demanded;
                    let array = ctx.decoded_chunk(segment.plan).await?;
                    let result = with_i64_values(&array, ctx.session, |values| {
                        let values = &values[segment.chunk_local.clone()];
                        match &segment.demand {
                            Some(demand) => evaluate_predicate_slice(
                                values,
                                demand,
                                segment.demanded,
                                conjunct.predicate,
                            ),
                            // Full demand needs no mask at all.
                            None => evaluate_predicate_full(values, conjunct.predicate),
                        }
                    })?;
                    parts.push((segment.parent_local.clone(), result));
                }
                let mask = fields.domain(conjunct.field).pull_mask(range, parts)?;
                self.observe(conjunct_idx, demanded_rows, mask.true_count());
                demand = Some(mask);
            }
            Ok(demand)
        })
    }
}

/// Evaluate every conjunct over every row, then intersect the masks. No work avoidance: this is
/// the baseline the cascade should beat on selective queries.
pub struct EagerDemand;

impl DemandPolicy for EagerDemand {
    fn name(&self) -> &'static str {
        "eager"
    }

    fn morsel_demand<'a, 'c>(
        &'a self,
        ctx: &'a mut PipelineCtx<'c>,
        fields: &'a FieldSet<'a>,
        query: &'a ScanQuery,
    ) -> BoxFuture<'a, VortexResult<Option<BitBuffer>>> {
        Box::pin(async move {
            let range = fields.range();
            let mut demand: Option<BitBuffer> = None;
            for conjunct in &query.conjuncts {
                let segments = fields.push_demand(conjunct.field, None)?;
                let mut parts = Vec::with_capacity(segments.len());
                for segment in segments {
                    let array = ctx.decoded_chunk(segment.plan).await?;
                    let result = with_i64_values(&array, ctx.session, |values| {
                        evaluate_predicate_full(
                            &values[segment.chunk_local.clone()],
                            conjunct.predicate,
                        )
                    })?;
                    parts.push((segment.parent_local.clone(), result));
                }
                let mask = fields.domain(conjunct.field).pull_mask(range, parts)?;
                demand = Some(match demand {
                    None => mask,
                    Some(demand) => &demand & &mask,
                });
            }
            Ok(demand)
        })
    }
}

/// The scheduler's entire knowledge of execution: a morsel goes in, a batch comes out. Any node
/// graph can sit behind this without scheduler changes.
pub trait MorselPipeline: Send + Sync {
    fn execute<'a, 'c>(
        &'a self,
        ctx: &'a mut PipelineCtx<'c>,
        range: Range<u64>,
    ) -> BoxFuture<'a, VortexResult<ExecBatch>>;
}

/// The restricted-layout scan pipeline. The struct node's own row relationship is the identity:
/// its down demand transform shares one mask by refcount with every child domain, and its up
/// array transform is the zero-copy struct pack.
pub struct StructScanPipeline {
    fields: Vec<Box<dyn FieldDomain>>,
    query: ScanQuery,
    names: FieldNames,
    demand: Arc<dyn DemandPolicy>,
}

impl StructScanPipeline {
    pub fn new(plan: &SourcePlan, query: ScanQuery, demand: Arc<dyn DemandPolicy>) -> Self {
        let names = FieldNames::from(
            query
                .projection
                .iter()
                .map(|field| Arc::<str>::from(plan.field_names[field.0].as_str()))
                .collect::<Vec<_>>(),
        );
        Self::from_parts(field_chunks(plan), query, names, demand)
    }

    /// Build from explicit per-field chunk lists, which may have arbitrary, mutually unaligned
    /// boundaries.
    pub fn from_parts(
        fields: Vec<FieldChunks>,
        query: ScanQuery,
        names: FieldNames,
        demand: Arc<dyn DemandPolicy>,
    ) -> Self {
        let fields = fields
            .into_iter()
            .map(|chunks| Box::new(ConcatDomain::new(chunks)) as Box<dyn FieldDomain>)
            .collect::<Vec<_>>();
        Self {
            fields,
            query,
            names,
            demand,
        }
    }

    fn field_set<'a>(&'a self, range: &'a Range<u64>) -> FieldSet<'a> {
        FieldSet {
            domains: &self.fields,
            range,
        }
    }
}

impl MorselPipeline for StructScanPipeline {
    fn execute<'a, 'c>(
        &'a self,
        ctx: &'a mut PipelineCtx<'c>,
        range: Range<u64>,
    ) -> BoxFuture<'a, VortexResult<ExecBatch>> {
        Box::pin(async move {
            let rows = usize::try_from(range.end - range.start)?;
            let fields = self.field_set(&range);
            let demand = self
                .demand
                .morsel_demand(&mut *ctx, &fields, &self.query)
                .await?;
            let (selection, true_count) = match demand {
                None => (BitBuffer::new_set(rows), rows),
                Some(demand) => {
                    let true_count = demand.true_count();
                    (demand, true_count)
                }
            };

            let mut field_arrays = Vec::with_capacity(self.query.projection.len());
            if true_count > 0 {
                // Identity down transform of the struct node: every child sees the same demand,
                // and partial demand builds one selection mask shared by every field.
                let shared_demand = (true_count != rows).then_some(&selection);
                let shared_mask =
                    (true_count != rows).then(|| Mask::from_buffer(selection.clone()));
                for field in &self.query.projection {
                    let segments = fields
                        .push_demand(*field, shared_demand)?
                        .into_iter()
                        .filter(|segment| segment.demanded > 0)
                        .collect::<Vec<_>>();
                    let mut arrays = Vec::with_capacity(segments.len());
                    for segment in &segments {
                        arrays.push(ctx.decoded_chunk(segment.plan).await?);
                    }
                    field_arrays.push(fields.domain(*field).pull_array(
                        &segments,
                        arrays,
                        true_count,
                        rows,
                        shared_mask.as_ref(),
                    )?);
                }
            }
            let array = pack_struct_array(self.names.clone(), field_arrays, true_count)?;
            Ok(ExecBatch {
                coverage: range,
                selection: BoolArray::new(selection, Validity::NonNullable).into_array(),
                array,
            })
        })
    }
}

/// The scheduler: worker tasks on a shared, reused thread pool self-schedule morsels from one
/// cursor, so a task that finishes early keeps pulling instead of idling behind a straggler, and
/// sub-millisecond scans do not pay per-run thread spawns. Output order is restored by morsel
/// index. It depends only on `dyn MorselPipeline`.
pub fn run_pipeline_sharded(
    pipeline: Arc<dyn MorselPipeline>,
    morsel_ranges: &[Range<u64>],
    source: Arc<dyn SegmentSource>,
    session: &VortexSession,
    threads: usize,
) -> VortexResult<RunResult> {
    static POOLS: LazyLock<parking_lot::Mutex<HashMap<usize, futures::executor::ThreadPool>>> =
        LazyLock::new(|| parking_lot::Mutex::new(HashMap::new()));

    let threads = threads.max(1).min(morsel_ranges.len().max(1));
    let pool = {
        let mut pools = POOLS.lock();
        match pools.get(&threads) {
            Some(pool) => pool.clone(),
            None => {
                let pool = futures::executor::ThreadPoolBuilder::new()
                    .pool_size(threads)
                    .name_prefix(format!("pipeline-{threads}-"))
                    .create()
                    .map_err(|error| {
                        vortex_error::vortex_err!("cannot create pipeline pool: {error}")
                    })?;
                pools.insert(threads, pool.clone());
                pool
            }
        }
    };

    let ranges: Arc<[Range<u64>]> = morsel_ranges.into();
    let next_morsel = Arc::new(AtomicUsize::new(0));
    let (result_tx, result_rx) = mpsc::channel();
    for _ in 0..threads {
        let pipeline = Arc::clone(&pipeline);
        let ranges = Arc::clone(&ranges);
        let next_morsel = Arc::clone(&next_morsel);
        let source = Arc::clone(&source);
        let session = session.clone();
        let result_tx = result_tx.clone();
        pool.spawn_ok(async move {
            let mut ctx = PipelineCtx::new(source.as_ref(), &session);
            let mut batches = Vec::new();
            let result = loop {
                let index = next_morsel.fetch_add(1, Ordering::Relaxed);
                let Some(range) = ranges.get(index) else {
                    break Ok(batches);
                };
                match pipeline.execute(&mut ctx, range.clone()).await {
                    Ok(batch) => batches.push((index, batch)),
                    Err(error) => break Err(error),
                }
            };
            drop(result_tx.send(result));
        });
    }
    drop(result_tx);

    let mut batches = Vec::with_capacity(ranges.len());
    for _ in 0..threads {
        let worker = result_rx
            .recv()
            .map_err(|_| vortex_error::vortex_err!("pipeline worker disappeared"))?;
        batches.extend(worker?);
    }
    batches.sort_unstable_by_key(|(index, _)| *index);
    Ok(RunResult {
        batches: batches.into_iter().map(|(_, batch)| batch).collect(),
        metrics: Metrics::default(),
        trace: Vec::new(),
    })
}
