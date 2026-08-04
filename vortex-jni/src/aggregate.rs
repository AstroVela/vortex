// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI bindings for aggregate pushdown.
//!
//! Engines such as Spark can push a group-by-free aggregation into the scan. The Java side
//! describes the aggregates it wants (`count(*)`, `count(col)`, `min(col)`, `max(col)`,
//! `sum(col)`), and [`AggregatePlan::try_new`] decides whether Vortex can compute all of them
//! with the semantics the engine expects. When it can, the plan is attached to a scan and each
//! partition produces a **single row** holding that partition's *partial* aggregates; the engine
//! combines partials across partitions (`sum` for counts and sums, `min`/`max` for extrema).
//!
//! Aggregates are evaluated with the same [`DynAccumulator`]s the DuckDB integration uses, so
//! compressed arrays keep their encoding-specific fast paths and statistics short-circuits. A
//! `count(*)`-only plan over an unfiltered scan is answered from partition metadata alone, without
//! reading any data.
//!
//! Not every aggregate is pushable — see [`AggregatePlan::try_new`] for the rules. When a plan
//! cannot be built the Java side falls back to a regular scan and aggregates the rows itself.

use std::ptr;
use std::sync::Arc;

use arrow_array::ffi::FFI_ArrowSchema;
use futures::StreamExt;
use jni::EnvUnowned;
use jni::objects::JClass;
use jni::objects::JIntArray;
use jni::objects::JObjectArray;
use jni::objects::JString;
use jni::sys::jint;
use jni::sys::jlong;
use vortex::aggregate_fn::Accumulator;
use vortex::aggregate_fn::AggregateFnVTable;
use vortex::aggregate_fn::DynAccumulator;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::aggregate_fn::fns::count::Count;
use vortex::aggregate_fn::fns::max::Max;
use vortex::aggregate_fn::fns::min::Min;
use vortex::aggregate_fn::fns::sum::Sum;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::ScalarFn;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::array::stream::SendableArrayStream;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::select;
use vortex::io::runtime::BlockingRuntime;
use vortex::scalar::Scalar;
use vortex::scalar_fn::fns::pack::Pack;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

use crate::RUNTIME;
use crate::data_source::NativeDataSource;
use crate::errors::try_or_throw;
use crate::session::session_ref;

/// Aggregate functions that can be pushed into a scan.
///
/// The discriminants are the wire codes shared with `dev.vortex.api.Aggregate.Kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AggregateKind {
    /// `count(*)`: the number of rows the partition returns.
    CountStar = 0,
    /// `count(col)`: the number of non-null values in a column.
    Count = 1,
    /// `min(col)`, ignoring nulls.
    Min = 2,
    /// `max(col)`, ignoring nulls.
    Max = 3,
    /// `sum(col)`, ignoring nulls.
    Sum = 4,
}

impl AggregateKind {
    fn try_from_code(code: jint) -> VortexResult<Self> {
        Ok(match code {
            0 => Self::CountStar,
            1 => Self::Count,
            2 => Self::Min,
            3 => Self::Max,
            4 => Self::Sum,
            other => vortex_bail!("unknown aggregate kind code: {other}"),
        })
    }

    /// Lower-case function label, used to name the plan's output fields.
    fn label(&self) -> &'static str {
        match self {
            Self::CountStar => "count_star",
            Self::Count => "count",
            Self::Min => "min",
            Self::Max => "max",
            Self::Sum => "sum",
        }
    }

    /// Whether this aggregate reads a column.
    fn is_column_aggregate(&self) -> bool {
        !matches!(self, Self::CountStar)
    }
}

/// An aggregate requested by the Java side: a function plus the top-level column it reads.
#[derive(Clone, Debug)]
pub(crate) struct AggregateSpec {
    kind: AggregateKind,
    /// Name of the aggregated column. `None` for `count(*)`.
    column: Option<String>,
}

/// NaN handling for pushed aggregates.
///
/// Engines that push aggregates down (Spark, DuckDB) treat NaN as an ordinary value: it is
/// counted by `count`, and it poisons `sum`/`max`. Vortex defaults to skipping NaNs, so pushed
/// aggregates always opt into the NaN-including behaviour. `min` is the one function whose
/// NaN-including semantics differ (Vortex poisons the result, engines return the smallest
/// non-NaN value), which is why [`AggregatePlan::try_new`] refuses `min` over floats.
const PUSHDOWN_OPTS: NumericalAggregateOpts = NumericalAggregateOpts::include_nans();

/// One aggregate of an [`AggregatePlan`], resolved against the source's type.
#[derive(Clone, Debug)]
struct PushedAggregate {
    kind: AggregateKind,
    /// Position of the input column in the plan's projection; `None` for `count(*)`.
    input: Option<usize>,
    /// Logical type of the input column; `None` for `count(*)`.
    input_dtype: Option<DType>,
}

/// A validated, group-by-free aggregation that a scan can compute.
///
/// The plan owns the scan projection (only the aggregated columns are read) and the output type
/// (one field per requested aggregate, in request order).
#[derive(Debug)]
pub(crate) struct AggregatePlan {
    aggregates: Vec<PushedAggregate>,
    projection: Expression,
    output_dtype: DType,
}

impl AggregatePlan {
    /// Resolve `specs` against a source of type `dtype`.
    ///
    /// Returns `Ok(None)` when at least one aggregate cannot be computed the way the calling
    /// engine would compute it, in which case the caller must not push the aggregation down:
    ///
    /// * the column is missing from the source, or is not a top-level field;
    /// * `min` over a floating-point column, because Vortex cannot reproduce "smallest non-NaN
    ///   value, unless every value is NaN" in a single pass;
    /// * `min`/`max` over a column whose type has no defined ordering (structs, lists, maps);
    /// * `sum` over a 64-bit integer column, whose partial can overflow the accumulator;
    /// * `sum` over an unsigned or boolean column, which has no signed Java/Spark counterpart;
    /// * `sum` over any other type Vortex cannot sum.
    pub(crate) fn try_new(dtype: &DType, specs: &[AggregateSpec]) -> VortexResult<Option<Self>> {
        if specs.is_empty() {
            vortex_bail!("aggregate plan requires at least one aggregate");
        }
        let fields = dtype.as_struct_fields_opt().ok_or_else(|| {
            vortex_err!("aggregate pushdown requires a struct source, got {dtype}")
        })?;

        let mut columns: Vec<FieldName> = Vec::new();
        let mut aggregates = Vec::with_capacity(specs.len());
        let mut output_names = Vec::with_capacity(specs.len());
        let mut output_dtypes = Vec::with_capacity(specs.len());

        for (ordinal, spec) in specs.iter().enumerate() {
            let input_dtype = match &spec.column {
                None => {
                    if spec.kind.is_column_aggregate() {
                        vortex_bail!("{} requires a column", spec.kind.label());
                    }
                    None
                }
                Some(column) => {
                    if !spec.kind.is_column_aggregate() {
                        vortex_bail!("count(*) must not name a column");
                    }
                    match fields.field(column) {
                        // An unknown column is a planning decision, not an error: engines may
                        // ask about columns that live outside the files (Spark partition values).
                        None => return Ok(None),
                        Some(field_dtype) => Some(field_dtype),
                    }
                }
            };

            let Some(output_dtype) = output_dtype(spec.kind, input_dtype.as_ref()) else {
                return Ok(None);
            };

            let input = input_dtype.as_ref().map(|_| {
                let column = spec
                    .column
                    .as_ref()
                    .vortex_expect("column aggregate has a column");
                let name = FieldName::from(column.as_str());
                columns.iter().position(|c| *c == name).unwrap_or_else(|| {
                    columns.push(name);
                    columns.len() - 1
                })
            });

            aggregates.push(PushedAggregate {
                kind: spec.kind,
                input,
                input_dtype,
            });
            // Output names must be unique, so they carry the request ordinal. Engines address
            // pushed aggregates by position, not by name.
            output_names.push(FieldName::from(format!("{}_{ordinal}", spec.kind.label())));
            output_dtypes.push(output_dtype);
        }

        let output_dtype = DType::Struct(
            StructFields::new(FieldNames::from(output_names), output_dtypes),
            Nullability::NonNullable,
        );

        Ok(Some(Self {
            aggregates,
            projection: select(FieldNames::from(columns), root()),
            output_dtype,
        }))
    }

    /// Projection the scan must apply: the distinct aggregated columns, in first-use order.
    pub(crate) fn projection(&self) -> &Expression {
        &self.projection
    }

    /// Type of the single row each partition produces: one field per requested aggregate.
    pub(crate) fn output_dtype(&self) -> &DType {
        &self.output_dtype
    }

    /// Whether every aggregate is `count(*)`, so the plan needs no column data at all.
    pub(crate) fn is_row_count_only(&self) -> bool {
        self.aggregates
            .iter()
            .all(|agg| agg.kind == AggregateKind::CountStar)
    }

    /// Build the single output row from a known row count, without reading any data.
    ///
    /// Only valid for a [`Self::is_row_count_only`] plan over a scan that returns every row.
    pub(crate) fn row_count_only_result(&self, row_count: u64) -> VortexResult<ArrayRef> {
        debug_assert!(self.is_row_count_only(), "plan reads column data");
        let scalars = self
            .aggregates
            .iter()
            .map(|_| count_scalar(row_count))
            .collect::<VortexResult<Vec<_>>>()?;
        self.finish(scalars)
    }

    /// Drain `stream`, returning this partition's partial aggregates as a single-row struct array.
    pub(crate) fn aggregate(
        &self,
        mut stream: SendableArrayStream,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let mut ctx = session.create_execution_ctx();
        let mut states = self.states()?;

        while let Some(chunk) = RUNTIME.block_on(stream.next()) {
            let chunk = chunk?;
            let rows = chunk.len() as u64;
            let chunk = as_struct(chunk, &mut ctx)?;
            for state in &mut states {
                state.accumulate(&chunk, rows, &mut ctx)?;
            }
        }

        let scalars = states
            .iter_mut()
            .map(|state| state.finish())
            .collect::<VortexResult<Vec<_>>>()?;
        self.finish(scalars)
    }

    fn states(&self) -> VortexResult<Vec<AggregateState>> {
        self.aggregates
            .iter()
            .map(|agg| {
                let Some(input_dtype) = agg.input_dtype.clone() else {
                    return Ok(AggregateState::RowCount { rows: 0 });
                };
                let input = agg.input.vortex_expect("column aggregate has an input");
                // Vortex sums an empty or all-null column to zero and an overflowing column to
                // null, whereas engines expect null for the former and cannot distinguish the
                // latter. Counting the summed values tells the two cases apart.
                let non_null: Option<Box<dyn DynAccumulator>> = match agg.kind {
                    AggregateKind::Sum => Some(Box::new(Accumulator::try_new(
                        Count,
                        PUSHDOWN_OPTS,
                        input_dtype.clone(),
                    )?)),
                    _ => None,
                };
                let accumulator: Box<dyn DynAccumulator> = match agg.kind {
                    AggregateKind::CountStar => {
                        vortex_bail!("count(*) does not read a column")
                    }
                    AggregateKind::Count => {
                        Box::new(Accumulator::try_new(Count, PUSHDOWN_OPTS, input_dtype)?)
                    }
                    AggregateKind::Min => {
                        Box::new(Accumulator::try_new(Min, PUSHDOWN_OPTS, input_dtype)?)
                    }
                    AggregateKind::Max => {
                        Box::new(Accumulator::try_new(Max, PUSHDOWN_OPTS, input_dtype)?)
                    }
                    AggregateKind::Sum => {
                        Box::new(Accumulator::try_new(Sum, PUSHDOWN_OPTS, input_dtype)?)
                    }
                };
                Ok(AggregateState::Column {
                    kind: agg.kind,
                    input,
                    accumulator,
                    non_null,
                })
            })
            .collect()
    }

    /// Wrap one scalar per aggregate into the plan's single-row output array.
    fn finish(&self, scalars: Vec<Scalar>) -> VortexResult<ArrayRef> {
        let fields = self.output_dtype.as_struct_fields();
        let arrays = scalars
            .into_iter()
            .map(|scalar| ConstantArray::new(scalar, 1).into_array());
        Ok(
            StructArray::try_new_with_dtype(arrays, fields.clone(), 1, Validity::NonNullable)?
                .into_array(),
        )
    }
}

/// Type of a pushed aggregate's output column.
///
/// Returns `None` when the aggregate is not pushable for this input type; see
/// [`AggregatePlan::try_new`] for the reasoning behind each rule.
fn output_dtype(kind: AggregateKind, input_dtype: Option<&DType>) -> Option<DType> {
    // Counts are returned as `i64` rather than Vortex's `u64` because unsigned integers have no
    // Java or Spark counterpart. A count cannot exceed `i64::MAX` rows.
    let count_dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    match kind {
        AggregateKind::CountStar => Some(count_dtype),
        AggregateKind::Count => Some(count_dtype),
        AggregateKind::Min | AggregateKind::Max => {
            let input_dtype = input_dtype?;
            if kind == AggregateKind::Min && input_dtype.is_float() {
                return None;
            }
            if !is_ordered_dtype(input_dtype) {
                return None;
            }
            Some(input_dtype.as_nullable())
        }
        AggregateKind::Sum => {
            let input_dtype = input_dtype?;
            match input_dtype {
                DType::Primitive(ptype, _) => match ptype {
                    // A partial sum of 64-bit integers can overflow the accumulator.
                    PType::I64 | PType::U64 => None,
                    // Unsigned sums have no signed counterpart to widen into.
                    PType::U8 | PType::U16 | PType::U32 => None,
                    PType::I8 | PType::I16 | PType::I32 => {
                        Some(DType::Primitive(PType::I64, Nullability::Nullable))
                    }
                    PType::F16 | PType::F32 | PType::F64 => {
                        Some(DType::Primitive(PType::F64, Nullability::Nullable))
                    }
                },
                // Decimal sums widen precision by 10, matching Spark and DataFusion.
                DType::Decimal(..) => Sum
                    .return_dtype(&PUSHDOWN_OPTS, input_dtype)
                    .map(|dtype| dtype.as_nullable()),
                // Includes booleans, which Vortex sums to `u64`.
                _ => None,
            }
        }
    }
}

/// Whether `min`/`max` are defined and computable for this type.
fn is_ordered_dtype(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Bool(_)
            | DType::Primitive(..)
            | DType::Decimal(..)
            | DType::Utf8(..)
            | DType::Binary(..)
            | DType::Extension(..)
    )
}

/// In-progress state of one pushed aggregate.
enum AggregateState {
    /// `count(*)`: rows seen so far.
    RowCount { rows: u64 },
    /// A column aggregate and, for `sum`, the count that tells "no values" from "overflow".
    Column {
        kind: AggregateKind,
        input: usize,
        accumulator: Box<dyn DynAccumulator>,
        non_null: Option<Box<dyn DynAccumulator>>,
    },
}

impl AggregateState {
    fn accumulate(
        &mut self,
        chunk: &StructArray,
        rows: u64,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match self {
            Self::RowCount { rows: seen } => {
                *seen += rows;
                Ok(())
            }
            Self::Column {
                input,
                accumulator,
                non_null,
                ..
            } => {
                let field = chunk.unmasked_field(*input);
                accumulator.accumulate(field, ctx)?;
                if let Some(non_null) = non_null {
                    non_null.accumulate(field, ctx)?;
                }
                Ok(())
            }
        }
    }

    fn finish(&mut self) -> VortexResult<Scalar> {
        match self {
            Self::RowCount { rows } => count_scalar(*rows),
            Self::Column {
                kind,
                accumulator,
                non_null,
                ..
            } => {
                let value = accumulator.finish()?;
                match kind {
                    AggregateKind::Count => {
                        let count = value
                            .as_primitive()
                            .typed_value::<u64>()
                            .ok_or_else(|| vortex_err!("count must not be null"))?;
                        count_scalar(count)
                    }
                    AggregateKind::Sum => {
                        let summed = non_null
                            .as_mut()
                            .vortex_expect("sum tracks a non-null count")
                            .finish()?
                            .as_primitive()
                            .typed_value::<u64>()
                            .ok_or_else(|| vortex_err!("count must not be null"))?;
                        if summed == 0 {
                            // No values to sum: null, rather than Vortex's zero.
                            return Ok(Scalar::null(value.dtype().as_nullable()));
                        }
                        if value.is_null() {
                            vortex_bail!(
                                "sum overflowed its {} accumulator; retry without aggregate pushdown",
                                value.dtype()
                            );
                        }
                        Ok(value)
                    }
                    _ => Ok(value),
                }
            }
        }
    }
}

/// A count as the non-nullable `i64` the plan's output type declares.
fn count_scalar(count: u64) -> VortexResult<Scalar> {
    let count = i64::try_from(count).map_err(|_| vortex_err!("count {count} exceeds i64::MAX"))?;
    Ok(Scalar::primitive(count, Nullability::NonNullable))
}

/// View a scan chunk as a struct array without materializing its fields.
///
/// Accumulators dispatch on encoding, so canonicalizing the columns here would throw away the
/// encoding-specific kernels and statistics short-circuits that make pushdown worthwhile.
fn as_struct(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<StructArray> {
    let array = array.optimize_recursive(ctx.session())?;
    Ok(if let Some(array) = array.as_opt::<Struct>() {
        array.into_owned()
    } else if let Some(scalar_fn) = array.as_opt::<ScalarFn>()
        && let Some(pack) = scalar_fn.scalar_fn().as_opt::<Pack>()
    {
        StructArray::new(
            pack.names.clone(),
            scalar_fn.children(),
            array.len(),
            pack.nullability.into(),
        )
    } else {
        array.execute::<Canonical>(ctx)?.into_struct()
    })
}

/// Wraps an [`AggregatePlan`] behind a single pointer.
pub(crate) struct NativeAggregate {
    inner: Arc<AggregatePlan>,
}

impl NativeAggregate {
    /// SAFETY: pointer must have been returned from [`Java_dev_vortex_jni_NativeAggregate_plan`].
    pub(crate) unsafe fn from_ptr<'a>(ptr: jlong) -> &'a Self {
        debug_assert!(ptr != 0, "null aggregate plan pointer");
        unsafe { &*(ptr as *const Self) }
    }

    pub(crate) fn inner(&self) -> &Arc<AggregatePlan> {
        &self.inner
    }
}

/// Plan an aggregate pushdown against a data source.
///
/// `kinds` holds one `dev.vortex.api.Aggregate.Kind` code per aggregate and `columns` the
/// matching column names, with a null entry for `count(*)`. Returns `0` when the aggregation
/// cannot be pushed down, in which case the caller must aggregate the scanned rows itself.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeAggregate_plan(
    mut env: EnvUnowned,
    _class: JClass,
    data_source_ptr: jlong,
    kinds: JIntArray,
    columns: JObjectArray,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let ds = unsafe { NativeDataSource::from_ptr(data_source_ptr) };

        let count = kinds.len(env)?;
        if count == 0 {
            throw_runtime!("no aggregates provided");
        }
        if columns.len(env)? != count {
            throw_runtime!("kinds and columns must have equal length");
        }

        let mut codes = vec![0 as jint; count];
        kinds.get_region(env, 0, &mut codes)?;

        let mut specs = Vec::with_capacity(count);
        for (idx, code) in codes.into_iter().enumerate() {
            let column = columns.get_element(env, idx)?;
            let column = if column.is_null() {
                None
            } else {
                Some(env.cast_local::<JString>(column)?.try_to_string(env)?)
            };
            specs.push(AggregateSpec {
                kind: AggregateKind::try_from_code(code)?,
                column,
            });
        }

        Ok(match AggregatePlan::try_new(ds.inner().dtype(), &specs)? {
            None => 0,
            Some(plan) => Box::into_raw(Box::new(NativeAggregate {
                inner: Arc::new(plan),
            })) as jlong,
        })
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeAggregate_free(
    _env: EnvUnowned,
    _class: JClass,
    pointer: jlong,
) {
    if pointer == 0 {
        return;
    }
    drop(unsafe { Box::from_raw(pointer as *mut NativeAggregate) });
}

/// Export the plan's output schema into the Arrow C Data Interface schema struct at
/// `schema_addr`: one field per requested aggregate, in request order.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeAggregate_arrowSchema(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    pointer: jlong,
    schema_addr: jlong,
) {
    try_or_throw(&mut env, |_| {
        if schema_addr == 0 {
            throw_runtime!("null arrow schema address");
        }
        let session = unsafe { session_ref(session_ptr) };
        let plan = unsafe { NativeAggregate::from_ptr(pointer) };
        let arrow_schema = session
            .arrow()
            .to_arrow_schema(plan.inner().output_dtype())?;
        let ffi_schema = FFI_ArrowSchema::try_from(&arrow_schema)?;
        unsafe {
            ptr::write(schema_addr as *mut FFI_ArrowSchema, ffi_schema);
        }
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use async_trait::async_trait;
    use futures::StreamExt as _;
    use futures::stream;
    use vortex::array::arrays::ExtensionArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::TemporalArray;
    use vortex::buffer::ByteBuffer;
    use vortex::buffer::ByteBufferMut;
    use vortex::error::VortexError;
    use vortex::expr::col;
    use vortex::expr::gt;
    use vortex::expr::lit;
    use vortex::extension::datetime::TimeUnit;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::file::multi::MultiFileDataSource;
    use vortex::io::VortexReadAt;
    use vortex::io::filesystem::FileListing;
    use vortex::io::filesystem::FileSystem;
    use vortex::io::filesystem::FileSystemRef;
    use vortex::scan::DataSourceRef;
    use vortex::scan::ScanRequest;

    use super::*;
    use crate::session::new_session;

    const PATH: &str = "test.vortex";

    /// Serves a single in-memory Vortex file, so tests need no temporary directory.
    struct MemoryFileSystem {
        bytes: ByteBuffer,
    }

    impl fmt::Debug for MemoryFileSystem {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MemoryFileSystem").finish()
        }
    }

    #[async_trait]
    impl FileSystem for MemoryFileSystem {
        fn list(&self, _prefix: &str) -> stream::BoxStream<'_, VortexResult<FileListing>> {
            stream::empty().boxed()
        }

        async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
            Ok((path == PATH).then_some(FileListing {
                path: path.to_string(),
                size: Some(self.bytes.len() as u64),
            }))
        }

        async fn open_read(&self, _path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
            Ok(Arc::new(self.bytes.clone()))
        }

        async fn delete(&self, _path: &str) -> VortexResult<()> {
            vortex_bail!("read-only filesystem")
        }
    }

    fn spec(kind: AggregateKind, column: Option<&str>) -> AggregateSpec {
        AggregateSpec {
            kind,
            column: column.map(str::to_owned),
        }
    }

    /// Write `array` to an in-memory file and open it as a single-file data source.
    fn file_source(session: &VortexSession, array: ArrayRef) -> VortexResult<DataSourceRef> {
        let mut output = ByteBufferMut::empty();
        RUNTIME.block_on(
            session
                .write_options()
                .write(&mut output, array.to_array_stream()),
        )?;
        let fs: FileSystemRef = Arc::new(MemoryFileSystem {
            bytes: ByteBuffer::from(output),
        });

        let ds = RUNTIME.block_on(
            MultiFileDataSource::new(session.clone())
                .with_glob(PATH, Some(fs))
                .build(),
        )?;
        Ok(Arc::new(ds))
    }

    /// A source of `{a: i32?, b: f64?}` over an in-memory file.
    fn source(session: &VortexSession) -> VortexResult<DataSourceRef> {
        let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(2)]).into_array();
        let b = PrimitiveArray::from_option_iter([Some(1.5f64), Some(f64::NAN), None, Some(0.5)])
            .into_array();
        file_source(
            session,
            StructArray::from_fields(&[("a", a), ("b", b)])?.into_array(),
        )
    }

    /// Scan `ds` with `plan` attached, returning one row of partial aggregates per partition.
    fn aggregate(
        ds: &DataSourceRef,
        plan: &AggregatePlan,
        session: &VortexSession,
        filter: Option<Expression>,
    ) -> VortexResult<Vec<Vec<Scalar>>> {
        let scan = RUNTIME.block_on(ds.scan(ScanRequest {
            projection: plan.projection().clone(),
            filter,
            ..Default::default()
        }))?;
        let mut partitions = scan.partitions();
        let mut rows = Vec::new();
        while let Some(partition) = RUNTIME.block_on(partitions.next()) {
            let row = plan.aggregate(partition?.execute()?, session)?;
            assert_eq!(row.len(), 1, "aggregates produce exactly one row");
            assert_eq!(row.dtype(), plan.output_dtype());
            let row = row.as_opt::<Struct>().vortex_expect("struct row");
            let mut ctx = session.create_execution_ctx();
            rows.push(
                (0..row.struct_fields().nfields())
                    .map(|idx| row.unmasked_field(idx).execute_scalar(0, &mut ctx))
                    .collect::<VortexResult<Vec<_>>>()?,
            );
        }
        Ok(rows)
    }

    fn as_i64(scalar: &Scalar) -> Option<i64> {
        scalar.as_primitive().typed_value::<i64>()
    }

    fn as_i32(scalar: &Scalar) -> Option<i32> {
        scalar.as_primitive().typed_value::<i32>()
    }

    fn as_f64(scalar: &Scalar) -> Option<f64> {
        scalar.as_primitive().typed_value::<f64>()
    }

    fn plan(
        session: &VortexSession,
        specs: &[AggregateSpec],
    ) -> VortexResult<Option<AggregatePlan>> {
        let ds = source(session)?;
        AggregatePlan::try_new(ds.dtype(), specs)
    }

    #[test]
    fn count_star_plan_reads_no_columns() -> VortexResult<()> {
        let session = *new_session();
        let plan = plan(&session, &[spec(AggregateKind::CountStar, None)])?
            .vortex_expect("count(*) is always pushable");

        assert!(plan.is_row_count_only());
        assert_eq!(
            plan.output_dtype(),
            &DType::Struct(
                StructFields::new(
                    FieldNames::from(vec![FieldName::from("count_star_0")]),
                    vec![DType::Primitive(PType::I64, Nullability::NonNullable)],
                ),
                Nullability::NonNullable,
            )
        );
        let row = plan.row_count_only_result(7)?;
        let mut ctx = session.create_execution_ctx();
        let count = row
            .as_opt::<Struct>()
            .vortex_expect("struct row")
            .unmasked_field(0)
            .execute_scalar(0, &mut ctx)?;
        assert_eq!(as_i64(&count), Some(7));
        Ok(())
    }

    #[test]
    fn repeated_column_is_projected_once() -> VortexResult<()> {
        let session = *new_session();
        let plan = plan(
            &session,
            &[
                spec(AggregateKind::Min, Some("a")),
                spec(AggregateKind::CountStar, None),
                spec(AggregateKind::Max, Some("a")),
            ],
        )?
        .vortex_expect("min/max over an integer column are pushable");

        assert!(!plan.is_row_count_only());
        assert_eq!(
            plan.projection().to_string(),
            select(FieldNames::from(vec![FieldName::from("a")]), root()).to_string()
        );
        assert_eq!(
            plan.output_dtype().as_struct_fields().names().as_ref(),
            &[
                FieldName::from("min_0"),
                FieldName::from("count_star_1"),
                FieldName::from("max_2"),
            ]
        );
        Ok(())
    }

    #[test]
    fn unpushable_aggregates_yield_no_plan() -> VortexResult<()> {
        let session = *new_session();

        // Vortex cannot reproduce the engine's "smallest non-NaN, unless all NaN" min.
        assert!(plan(&session, &[spec(AggregateKind::Min, Some("b"))])?.is_none());
        // A column that is not in the files at all.
        assert!(plan(&session, &[spec(AggregateKind::Max, Some("missing"))])?.is_none());
        // One unpushable aggregate rejects the whole aggregation.
        assert!(
            plan(
                &session,
                &[
                    spec(AggregateKind::CountStar, None),
                    spec(AggregateKind::Min, Some("b")),
                ],
            )?
            .is_none()
        );
        Ok(())
    }

    #[test]
    fn sum_output_types_follow_the_input() -> VortexResult<()> {
        let session = *new_session();

        let sum_dtype = |column: &str| -> VortexResult<Option<DType>> {
            Ok(plan(&session, &[spec(AggregateKind::Sum, Some(column))])?
                .and_then(|plan| plan.output_dtype().as_struct_fields().fields().next()))
        };

        // 32-bit integers widen into i64, floats into f64.
        assert_eq!(
            sum_dtype("a")?,
            Some(DType::Primitive(PType::I64, Nullability::Nullable))
        );
        assert_eq!(
            sum_dtype("b")?,
            Some(DType::Primitive(PType::F64, Nullability::Nullable))
        );
        // A partial sum of 64-bit integers can overflow the accumulator.
        assert_eq!(
            output_dtype(
                AggregateKind::Sum,
                Some(&DType::Primitive(PType::I64, Nullability::Nullable))
            ),
            None
        );
        // Unsigned sums have no signed counterpart to widen into.
        assert_eq!(
            output_dtype(
                AggregateKind::Sum,
                Some(&DType::Primitive(PType::U32, Nullability::Nullable))
            ),
            None
        );
        Ok(())
    }

    #[test]
    fn aggregates_a_scan() -> VortexResult<()> {
        let session = *new_session();
        let ds = source(&session)?;
        let specs = [
            spec(AggregateKind::CountStar, None),
            spec(AggregateKind::Count, Some("a")),
            spec(AggregateKind::Min, Some("a")),
            spec(AggregateKind::Max, Some("a")),
            spec(AggregateKind::Sum, Some("a")),
        ];
        let plan = AggregatePlan::try_new(ds.dtype(), &specs)?.vortex_expect("pushable");

        let rows = aggregate(&ds, &plan, &session, None)?;
        assert_eq!(rows.len(), 1, "one file, one partition");
        let row = &rows[0];

        assert_eq!(as_i64(&row[0]), Some(4), "count(*) counts nulls");
        assert_eq!(as_i64(&row[1]), Some(3), "count(a) skips nulls");
        assert_eq!(as_i32(&row[2]), Some(1), "min(a)");
        assert_eq!(as_i32(&row[3]), Some(3), "max(a)");
        assert_eq!(as_i64(&row[4]), Some(6), "sum(a) widens to i64");
        Ok(())
    }

    #[test]
    fn nan_participates_in_max_and_sum() -> VortexResult<()> {
        let session = *new_session();
        let ds = source(&session)?;
        let specs = [
            spec(AggregateKind::Count, Some("b")),
            spec(AggregateKind::Max, Some("b")),
            spec(AggregateKind::Sum, Some("b")),
        ];
        let plan = AggregatePlan::try_new(ds.dtype(), &specs)?.vortex_expect("pushable");

        let rows = aggregate(&ds, &plan, &session, None)?;
        let row = &rows[0];

        assert_eq!(as_i64(&row[0]), Some(3), "NaN is a value, null is not");
        assert!(
            as_f64(&row[1]).is_some_and(f64::is_nan),
            "max(b) is NaN, as engines order NaN highest: {:?}",
            row[1]
        );
        assert!(
            as_f64(&row[2]).is_some_and(f64::is_nan),
            "NaN poisons sum(b): {:?}",
            row[2]
        );
        Ok(())
    }

    #[test]
    fn filtered_count_star_needs_no_columns() -> VortexResult<()> {
        let session = *new_session();
        let ds = source(&session)?;
        let plan = AggregatePlan::try_new(ds.dtype(), &[spec(AggregateKind::CountStar, None)])?
            .vortex_expect("pushable");

        // The projection is empty, so the rows are counted without decoding any column.
        let rows = aggregate(
            &ds,
            &plan,
            &session,
            Some(gt(col("a"), lit(Scalar::from(1i32)))),
        )?;
        assert_eq!(as_i64(&rows[0][0]), Some(2), "a > 1 selects 3 and 2");
        Ok(())
    }

    #[test]
    fn sum_of_no_values_is_null() -> VortexResult<()> {
        let session = *new_session();
        let all_null =
            PrimitiveArray::from_option_iter([None, None, None as Option<i32>]).into_array();
        let ds = file_source(
            &session,
            StructArray::from_fields(&[("a", all_null)])?.into_array(),
        )?;

        let specs = [
            spec(AggregateKind::Sum, Some("a")),
            spec(AggregateKind::Min, Some("a")),
            spec(AggregateKind::Count, Some("a")),
        ];
        let plan = AggregatePlan::try_new(ds.dtype(), &specs)?.vortex_expect("pushable");
        let rows = aggregate(&ds, &plan, &session, None)?;
        let row = &rows[0];

        assert!(row[0].is_null(), "sum of no values is null, not zero");
        assert!(row[1].is_null(), "min of no values is null");
        assert_eq!(as_i64(&row[2]), Some(0), "count of no values is zero");
        Ok(())
    }

    #[test]
    fn min_max_over_a_timestamp_column() -> VortexResult<()> {
        let session = *new_session();
        let timestamps = TemporalArray::new_timestamp(
            PrimitiveArray::from_option_iter([Some(20i64), Some(10), None]).into_array(),
            TimeUnit::Milliseconds,
            None,
        );
        let array =
            StructArray::from_fields(&[("t", ExtensionArray::from(timestamps).into_array())])?
                .into_array();
        let ds = file_source(&session, array)?;

        let specs = [
            spec(AggregateKind::Min, Some("t")),
            spec(AggregateKind::Max, Some("t")),
        ];
        let plan = AggregatePlan::try_new(ds.dtype(), &specs)?
            .vortex_expect("timestamps are ordered, so min/max are pushable");
        assert_eq!(
            plan.output_dtype().as_struct_fields().field("min_0"),
            Some(ds.dtype().as_struct_fields().field("t").vortex_expect("t")),
            "min keeps the column's extension type, which is already nullable"
        );

        let rows = aggregate(&ds, &plan, &session, None)?;
        let extrema: Vec<_> = rows[0]
            .iter()
            .map(|scalar| {
                scalar
                    .as_extension()
                    .to_storage_scalar()
                    .as_primitive()
                    .typed_value::<i64>()
            })
            .collect();
        assert_eq!(extrema, vec![Some(10), Some(20)]);
        Ok(())
    }

    #[test]
    fn rejects_malformed_specs() {
        let session = *new_session();
        let err = |specs: &[AggregateSpec]| -> VortexError {
            plan(&session, specs).expect_err("malformed spec")
        };

        assert!(err(&[]).to_string().contains("at least one aggregate"));
        assert!(
            err(&[spec(AggregateKind::Min, None)])
                .to_string()
                .contains("requires a column")
        );
        assert!(
            err(&[spec(AggregateKind::CountStar, Some("a"))])
                .to_string()
                .contains("must not name a column")
        );
    }
}
