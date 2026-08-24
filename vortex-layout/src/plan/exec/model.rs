// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use smallvec::SmallVec;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::ReadContext;

use crate::LayoutRef;
use crate::layouts::chunked::Chunked;
use crate::layouts::flat::Flat;
use crate::layouts::struct_::Struct;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSink;
use crate::segments::SegmentSource;
use crate::sequence::SequenceId;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub usize);
    };
}

id_type!(MorselId);
id_type!(ResourceId);
id_type!(TaskId);
id_type!(DemandVersion);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FieldId(pub usize);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ArraySlotId {
    Scan(usize),
    Morsel(MorselId, usize),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SegmentSlotId {
    Scan(usize),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum InputSlot {
    Segment(SegmentSlotId),
    Array(ArraySlotId),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OutputSlot {
    Segment(SegmentSlotId),
    Array(ArraySlotId),
}

#[derive(Clone, Debug)]
pub struct BooleanMaskSummary {
    pub len: usize,
    pub true_count: usize,
    pub(crate) values: BitBuffer,
}

impl BooleanMaskSummary {
    pub fn true_count_in(&self, range: Range<usize>) -> VortexResult<usize> {
        if range.start > range.end || range.end > self.len {
            vortex_bail!("mask range {range:?} is outside {} rows", self.len);
        }
        Ok(if self.true_count == 0 {
            0
        } else if self.true_count == self.len {
            range.len()
        } else if range.len() == self.len {
            self.true_count
        } else {
            self.values.slice(range).true_count()
        })
    }
}

#[derive(Clone, Debug)]
pub enum ArraySummary {
    None,
    BooleanMask(BooleanMaskSummary),
}

#[derive(Clone, Debug)]
pub struct ResolvedArray {
    pub array: ArrayRef,
    pub summary: ArraySummary,
    pub cached_predicates: Vec<CachedPredicate>,
}

#[derive(Clone, Debug)]
pub struct CachedPredicate {
    pub conjunct: usize,
    pub values: BitBuffer,
    pub evaluated: BitBuffer,
    pub input_true_count: usize,
    pub elapsed_ns: u64,
}

impl ResolvedArray {
    pub fn plain(array: ArrayRef) -> Self {
        Self {
            array,
            summary: ArraySummary::None,
            cached_predicates: Vec::new(),
        }
    }

    pub fn plain_with_predicates(array: ArrayRef, cached_predicates: Vec<CachedPredicate>) -> Self {
        Self {
            array,
            summary: ArraySummary::None,
            cached_predicates,
        }
    }

    pub fn boolean(array: ArrayRef, values: BitBuffer) -> Self {
        debug_assert_eq!(array.len(), values.len());
        Self {
            summary: ArraySummary::BooleanMask(BooleanMaskSummary {
                len: array.len(),
                true_count: values.true_count(),
                values,
            }),
            array,
            cached_predicates: Vec::new(),
        }
    }

    pub fn boolean_summary(&self) -> VortexResult<&BooleanMaskSummary> {
        match &self.summary {
            ArraySummary::BooleanMask(summary) => Ok(summary),
            ArraySummary::None => vortex_bail!("array is missing its boolean-mask summary"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum ResolvedValue {
    Segment(BufferHandle),
    Array(ResolvedArray),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Predicate {
    Equal(i64),
    LessThan(i64),
    GreaterThan(i64),
    RangeExclusive { lower: i64, upper: i64 },
}

impl Predicate {
    pub(crate) fn matches(self, value: i64) -> bool {
        match self {
            Self::Equal(rhs) => value == rhs,
            Self::LessThan(rhs) => value < rhs,
            Self::GreaterThan(rhs) => value > rhs,
            Self::RangeExclusive { lower, upper } => value > lower && value < upper,
        }
    }

    fn intersection(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::GreaterThan(left), Self::GreaterThan(right)) => {
                Some(Self::GreaterThan(left.max(right)))
            }
            (Self::LessThan(left), Self::LessThan(right)) => Some(Self::LessThan(left.min(right))),
            (Self::GreaterThan(lower), Self::LessThan(upper))
            | (Self::LessThan(upper), Self::GreaterThan(lower)) => {
                Some(Self::RangeExclusive { lower, upper })
            }
            (Self::RangeExclusive { lower, upper }, Self::GreaterThan(next_lower))
            | (Self::GreaterThan(next_lower), Self::RangeExclusive { lower, upper }) => {
                Some(Self::RangeExclusive {
                    lower: lower.max(next_lower),
                    upper,
                })
            }
            (Self::RangeExclusive { lower, upper }, Self::LessThan(next_upper))
            | (Self::LessThan(next_upper), Self::RangeExclusive { lower, upper }) => {
                Some(Self::RangeExclusive {
                    lower,
                    upper: upper.min(next_upper),
                })
            }
            (
                Self::RangeExclusive {
                    lower: left_lower,
                    upper: left_upper,
                },
                Self::RangeExclusive {
                    lower: right_lower,
                    upper: right_upper,
                },
            ) => Some(Self::RangeExclusive {
                lower: left_lower.max(right_lower),
                upper: left_upper.min(right_upper),
            }),
            (Self::Equal(left), Self::Equal(right)) if left == right => Some(Self::Equal(left)),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conjunct {
    pub field: FieldId,
    pub predicate: Predicate,
}

#[derive(Clone, Debug)]
pub struct ScanQuery {
    pub conjuncts: Vec<Conjunct>,
    pub projection: Vec<FieldId>,
}

impl ScanQuery {
    pub(crate) fn coalesce_same_field_predicates(&mut self) {
        let mut current = 0;
        while current < self.conjuncts.len() {
            let mut candidate = current + 1;
            while candidate < self.conjuncts.len() {
                if self.conjuncts[current].field == self.conjuncts[candidate].field
                    && let Some(predicate) = self.conjuncts[current]
                        .predicate
                        .intersection(self.conjuncts[candidate].predicate)
                {
                    self.conjuncts[current].predicate = predicate;
                    self.conjuncts.remove(candidate);
                } else {
                    candidate += 1;
                }
            }
            current += 1;
        }
    }
}

#[derive(Clone, Debug)]
pub enum FlatEncoding {
    RawI64,
    Serialized {
        dtype: DType,
        read_ctx: ReadContext,
        array_tree: Option<ByteBuffer>,
    },
}

#[derive(Clone, Debug)]
pub struct FlatPlan {
    pub field: FieldId,
    pub segment: SegmentId,
    pub root_coverage: Range<u64>,
    pub row_count: usize,
    pub estimated_bytes: Option<usize>,
    pub encoding: FlatEncoding,
}

#[derive(Clone, Debug)]
pub struct ChunkPlan {
    pub root_coverage: Range<u64>,
    pub fields: Vec<FlatPlan>,
}

#[derive(Clone, Debug)]
pub struct SourcePlan {
    pub field_names: Vec<String>,
    pub chunks: Vec<ChunkPlan>,
    pub row_count: u64,
}

impl SourcePlan {
    pub fn populate_segment_sizes(&mut self, source: &dyn SegmentSource) {
        for flat in self
            .chunks
            .iter_mut()
            .flat_map(|chunk| chunk.fields.iter_mut())
        {
            flat.estimated_bytes = source.estimated_size(flat.segment);
        }
    }

    pub fn try_from_layout(layout: &LayoutRef) -> VortexResult<Self> {
        let root = layout
            .as_opt::<Struct>()
            .ok_or_else(|| vortex_error::vortex_err!("experiment requires a Struct root layout"))?;
        if root.dtype().is_nullable() {
            vortex_bail!("nullable Struct layouts are outside the experiment scope");
        }
        let struct_fields = root
            .dtype()
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_error::vortex_err!("experiment requires a Struct root dtype"))?;
        let field_names = struct_fields
            .names()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let mut chunks = Vec::<ChunkPlan>::new();
        for field_idx in 0..field_names.len() {
            let field = root
                .slot(field_idx + 1)?
                .ok_or_else(|| vortex_error::vortex_err!("missing field {field_idx}"))?;
            let chunked = field.as_opt::<Chunked>().ok_or_else(|| {
                vortex_error::vortex_err!("field {field_idx} is not a Chunked layout")
            })?;
            let mut root_start = 0u64;
            for chunk_idx in 0..chunked.nchildren() {
                let flat = chunked.slot(chunk_idx)?.ok_or_else(|| {
                    vortex_error::vortex_err!("missing field {field_idx} chunk {chunk_idx}")
                })?;
                let flat = flat.as_opt::<Flat>().ok_or_else(|| {
                    vortex_error::vortex_err!(
                        "field {field_idx} chunk {chunk_idx} is not a Flat layout"
                    )
                })?;
                if flat.dtype()
                    != &DType::Primitive(
                        vortex_array::dtype::PType::I64,
                        vortex_array::dtype::Nullability::NonNullable,
                    )
                {
                    vortex_bail!("experiment supports only non-nullable i64 fields");
                }
                let root_end = root_start + flat.row_count();
                if field_idx == 0 {
                    chunks.push(ChunkPlan {
                        root_coverage: root_start..root_end,
                        fields: Vec::with_capacity(field_names.len()),
                    });
                } else if chunks
                    .get(chunk_idx)
                    .is_none_or(|chunk| chunk.root_coverage != (root_start..root_end))
                {
                    vortex_bail!("field chunk boundaries are not aligned");
                }
                chunks[chunk_idx].fields.push(FlatPlan {
                    field: FieldId(field_idx),
                    segment: flat.segment_id(),
                    root_coverage: root_start..root_end,
                    row_count: usize::try_from(flat.row_count())?,
                    estimated_bytes: None,
                    encoding: FlatEncoding::Serialized {
                        dtype: flat.dtype().clone(),
                        read_ctx: flat.array_ctx().clone(),
                        array_tree: flat.array_tree().cloned(),
                    },
                });
                root_start = root_end;
            }
            if root_start != root.row_count() {
                vortex_bail!("field {field_idx} chunks do not cover the Struct row count");
            }
        }

        Ok(Self {
            field_names,
            chunks,
            row_count: root.row_count(),
        })
    }

    pub fn from_i64_chunks(
        field_names: Vec<String>,
        chunks: Vec<Vec<Vec<i64>>>,
    ) -> VortexResult<(Self, Arc<MemorySegments>)> {
        let source = Arc::new(MemorySegments::default());
        let mut plans = Vec::with_capacity(chunks.len());
        let mut root_start = 0u64;
        for (chunk_idx, fields) in chunks.into_iter().enumerate() {
            if fields.len() != field_names.len() {
                vortex_bail!(
                    "chunk {chunk_idx} has {} fields, expected {}",
                    fields.len(),
                    field_names.len()
                );
            }
            let row_count = fields.first().map_or(0, Vec::len);
            if fields.iter().any(|field| field.len() != row_count) {
                vortex_bail!("chunk {chunk_idx} fields have different row counts");
            }
            let root_end = root_start + u64::try_from(row_count)?;
            let mut flat_plans = Vec::with_capacity(fields.len());
            for (field_idx, values) in fields.into_iter().enumerate() {
                let segment = source.insert(Buffer::from_iter(values).into_byte_buffer())?;
                flat_plans.push(FlatPlan {
                    field: FieldId(field_idx),
                    segment,
                    root_coverage: root_start..root_end,
                    row_count,
                    estimated_bytes: source.estimated_size(segment),
                    encoding: FlatEncoding::RawI64,
                });
            }
            plans.push(ChunkPlan {
                root_coverage: root_start..root_end,
                fields: flat_plans,
            });
            root_start = root_end;
        }
        Ok((
            Self {
                field_names,
                chunks: plans,
                row_count: root_start,
            },
            source,
        ))
    }
}

#[derive(Debug, Default)]
pub struct MemorySegments {
    buffers: parking_lot::Mutex<Vec<ByteBuffer>>,
}

impl MemorySegments {
    pub fn insert(&self, buffer: ByteBuffer) -> VortexResult<SegmentId> {
        let mut buffers = self.buffers.lock();
        let id = SegmentId::try_from(buffers.len())?;
        buffers.push(buffer);
        Ok(id)
    }
}

impl SegmentSource for MemorySegments {
    fn estimated_size(&self, id: SegmentId) -> Option<usize> {
        self.buffers.lock().get(*id as usize).map(ByteBuffer::len)
    }

    fn request(&self, id: SegmentId) -> SegmentFuture {
        use futures::FutureExt;

        let value = self.buffers.lock().get(*id as usize).cloned();
        async move {
            value
                .map(BufferHandle::new_host)
                .ok_or_else(|| vortex_error::vortex_err!("segment {id} not found"))
        }
        .boxed()
    }
}

#[async_trait::async_trait]
impl SegmentSink for MemorySegments {
    async fn write(
        &self,
        _sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId> {
        let total_len = buffers.iter().map(ByteBuffer::len).sum();
        let mut combined = vortex_buffer::ByteBufferMut::with_capacity(total_len);
        for buffer in buffers {
            combined.extend_from_slice(buffer.as_ref());
        }
        self.insert(combined.freeze())
    }
}

#[derive(Clone, Debug)]
pub struct ExecBatch {
    pub coverage: Range<u64>,
    pub selection: ArrayRef,
    pub array: ArrayRef,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Necessity {
    Required,
    Candidate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkClass {
    Io,
    Cpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadPhase {
    Predicate,
    Projection,
    PredicateAndProjection,
}

impl ReadPhase {
    pub fn includes_predicate(self) -> bool {
        matches!(self, Self::Predicate | Self::PredicateAndProjection)
    }

    pub fn includes_projection(self) -> bool {
        matches!(self, Self::Projection | Self::PredicateAndProjection)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeculativeReadPolicy {
    Disabled,
    Eager,
    Adaptive { minimum_expected_rows: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpeculativeIoConfig {
    pub predicate: SpeculativeReadPolicy,
    pub projection: SpeculativeReadPolicy,
    pub max_in_flight_bytes: usize,
    pub unknown_read_bytes: usize,
}

impl SpeculativeIoConfig {
    pub const fn disabled() -> Self {
        Self {
            predicate: SpeculativeReadPolicy::Disabled,
            projection: SpeculativeReadPolicy::Disabled,
            max_in_flight_bytes: 0,
            unknown_read_bytes: 0,
        }
    }

    pub const fn eager(max_in_flight_bytes: usize, unknown_read_bytes: usize) -> Self {
        Self {
            predicate: SpeculativeReadPolicy::Eager,
            projection: SpeculativeReadPolicy::Eager,
            max_in_flight_bytes,
            unknown_read_bytes,
        }
    }
}

impl Default for SpeculativeIoConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadEstimate {
    pub phase: ReadPhase,
    pub estimated_bytes: Option<usize>,
    pub current_rows: usize,
    pub expected_rows: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulePolicy {
    PredicateFirst,
    AdaptivePredicates { concurrency: usize },
    LegacyAdaptivePredicates { concurrency: usize },
    AllReady,
    ProjectionPrefetch,
    SmallFrontier(usize),
    Reverse,
    Random(u64),
}

#[derive(Clone, Debug)]
pub enum Operation {
    Read {
        segment: SegmentId,
        phase: ReadPhase,
        estimated_bytes: Option<usize>,
    },
    ReadDecodeFlat {
        segment: SegmentId,
        phase: ReadPhase,
        estimated_bytes: Option<usize>,
        encoding: FlatEncoding,
        row_count: usize,
        predicates: Vec<(usize, Predicate, BitBuffer, usize)>,
    },
    DecodeFlat {
        encoding: FlatEncoding,
        row_count: usize,
    },
    EvaluatePredicate {
        conjunct: usize,
        local_ranges: Vec<Range<usize>>,
        predicate: Predicate,
        demand_version: DemandVersion,
        input_true_count: usize,
    },
    CombineDemand {
        demand_version: DemandVersion,
    },
    MergeDemandFragments,
    SelectFlat {
        local_ranges: Vec<Range<usize>>,
        selection_ranges: Vec<Range<usize>>,
        selection_all_true: bool,
        pack_names: Option<FieldNames>,
    },
    SelectStruct {
        field_local_ranges: Vec<Vec<Range<usize>>>,
        selection_ranges: Vec<Range<usize>>,
        selection_all_true: bool,
        names: FieldNames,
    },
    PackStruct {
        names: FieldNames,
        len: usize,
    },
}

#[derive(Clone, Debug)]
pub struct Task {
    pub id: TaskId,
    pub class: WorkClass,
    pub necessity: Necessity,
    pub inputs: SmallVec<[InputSlot; 2]>,
    pub output: OutputSlot,
    pub operation: Operation,
}

#[derive(Clone, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "keeping offered tasks inline avoids the measured allocation/indirection regression"
)]
pub enum TaskUpdate {
    Offer(Task),
    Promote(TaskId),
    Revoke(TaskId),
}

#[derive(Clone, Debug)]
pub struct RunnableTask {
    pub id: TaskId,
    pub inputs: SmallVec<[ResolvedValue; 2]>,
    pub output: OutputSlot,
    pub operation: Operation,
}

#[derive(Clone, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "resolved task inputs stay inline to avoid a heap allocation per claimed task"
)]
pub enum ClaimResult {
    Runnable(RunnableTask),
    Revoked,
}

#[derive(Debug)]
pub struct Completion {
    pub task: TaskId,
    pub output: OutputSlot,
    pub elapsed_ns: u64,
    pub read_bytes: Option<usize>,
    /// When phase timing is enabled, the instant the worker queued this completion; the
    /// coordinator uses it to measure how long finished results waited to be adopted.
    pub sent_at: Option<std::time::Instant>,
    pub result: VortexResult<ResolvedValue>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MorselState {
    Budgeted,
    Quiescent,
    Retired,
}

#[derive(Clone, Debug)]
pub struct AdvanceResult {
    pub work: Vec<TaskUpdate>,
    pub output: Option<ExecBatch>,
    pub state: MorselState,
}
