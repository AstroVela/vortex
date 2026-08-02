// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI entry point for pushed-down aggregate evaluation.
//!
//! `Java_dev_vortex_jni_NativeAggregate_compute` scans a data source (optionally filtered),
//! folds every chunk into streaming [`DynAccumulator`]s — mirroring the DuckDB integration's
//! aggregate pushdown — and exports the final values as a single-row Arrow record batch
//! through an `FFI_ArrowArrayStream`.

use std::ptr;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::ArrowError;
use arrow_schema::Field;
use futures::StreamExt;
use jni::EnvUnowned;
use jni::objects::JByteArray;
use jni::objects::JClass;
use jni::objects::JObjectArray;
use jni::objects::JString;
use jni::sys::jlong;
use vortex::aggregate_fn::Accumulator;
use vortex::aggregate_fn::DynAccumulator;
use vortex::aggregate_fn::EmptyOptions;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::aggregate_fn::fns::count::Count;
use vortex::aggregate_fn::fns::max::Max;
use vortex::aggregate_fn::fns::min::Min;
use vortex::aggregate_fn::fns::nan_count::NanCount;
use vortex::aggregate_fn::fns::sum::Sum;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute as _;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::ScalarFn;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::half::f16;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::select;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::arrow::RecordBatchIteratorAdapter;
use vortex::scalar::Scalar;
use vortex::scalar_fn::fns::pack::Pack;
use vortex::scan::ScanRequest;
use vortex::scan::selection::Selection;
use vortex_arrow::ArrowSessionExt;

use crate::RUNTIME;
use crate::data_source::NativeDataSource;
use crate::errors::try_or_throw;
use crate::session::session_ref;

/// Aggregate kinds; codes must match `dev.vortex.api.Aggregate.Kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AggKind {
    Min,
    Max,
    Sum,
    /// Count of non-null values in a column.
    Count,
    /// Count of rows.
    CountStar,
}

impl AggKind {
    fn from_code(code: u8) -> VortexResult<Self> {
        Ok(match code {
            0 => Self::Min,
            1 => Self::Max,
            2 => Self::Sum,
            3 => Self::Count,
            4 => Self::CountStar,
            other => vortex_bail!("unknown aggregate kind code: {other}"),
        })
    }

    fn state(self, dtype: DType) -> VortexResult<AggState> {
        let opts = NumericalAggregateOpts::default();
        Ok(match self {
            Self::Min if dtype.is_float() => AggState::MinMaxFloat {
                is_max: false,
                inner: Box::new(Accumulator::try_new(Min, opts, dtype.clone())?),
                nan_count: Box::new(Accumulator::try_new(NanCount, EmptyOptions, dtype)?),
            },
            Self::Max if dtype.is_float() => AggState::MinMaxFloat {
                is_max: true,
                inner: Box::new(Accumulator::try_new(Max, opts, dtype.clone())?),
                nan_count: Box::new(Accumulator::try_new(NanCount, EmptyOptions, dtype)?),
            },
            Self::Min => AggState::Column(Box::new(Accumulator::try_new(Min, opts, dtype)?)),
            Self::Max => AggState::Column(Box::new(Accumulator::try_new(Max, opts, dtype)?)),
            Self::Count => AggState::Column(Box::new(Accumulator::try_new(
                Count,
                opts_for(&dtype),
                dtype,
            )?)),
            Self::Sum => AggState::Sum {
                sum: Box::new(Accumulator::try_new(Sum, opts_for(&dtype), dtype.clone())?),
                non_null_count: Box::new(Accumulator::try_new(Count, opts_for(&dtype), dtype)?),
            },
            Self::CountStar => AggState::RowCount,
        })
    }
}

/// SUM and COUNT options: Spark propagates NaN into sums and counts NaN as an ordinary
/// non-null value, so NaNs must flow into these accumulators rather than being skipped like
/// the statistics default. Float MIN/MAX instead skip NaNs and recover Spark's NaN-is-largest
/// ordering from a [`NanCount`] (see [`AggState::MinMaxFloat`]).
fn opts_for(dtype: &DType) -> NumericalAggregateOpts {
    if dtype.is_float() {
        NumericalAggregateOpts::include_nans()
    } else {
        NumericalAggregateOpts::default()
    }
}

/// A NaN scalar of the given float dtype.
fn nan_scalar(dtype: &DType) -> VortexResult<Scalar> {
    let DType::Primitive(ptype, _) = dtype else {
        vortex_bail!("NaN scalar requires a float dtype, got {dtype}");
    };
    Ok(match ptype {
        PType::F16 => Scalar::primitive(f16::NAN, Nullability::Nullable),
        PType::F32 => Scalar::primitive(f32::NAN, Nullability::Nullable),
        PType::F64 => Scalar::primitive(f64::NAN, Nullability::Nullable),
        _ => vortex_bail!("NaN scalar requires a float dtype, got {dtype}"),
    })
}

/// One requested aggregate: its kind and, for column aggregates, the index of its column in
/// the deduplicated projection.
struct AggSpec {
    kind: AggKind,
    field_index: usize,
}

/// Accumulation state for one requested aggregate.
enum AggState {
    /// count(*): folded from chunk lengths, no accumulator needed.
    RowCount,
    Column(Box<dyn DynAccumulator>),
    /// Float MIN/MAX under Spark's ordering, where NaN is larger than every other value. The
    /// inner accumulator skips NaNs entirely; the NaN count recovers the cases whose result
    /// must be NaN: any NaN present for max, and NaN with no other non-null values for min.
    MinMaxFloat {
        is_max: bool,
        inner: Box<dyn DynAccumulator>,
        nan_count: Box<dyn DynAccumulator>,
    },
    /// SQL SUM over a column. Vortex's sum of zero non-null values is `0`, but SQL requires
    /// `NULL`, so a non-null count is tracked alongside the sum.
    Sum {
        sum: Box<dyn DynAccumulator>,
        non_null_count: Box<dyn DynAccumulator>,
    },
}

impl AggState {
    fn accumulate(&mut self, batch: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        match self {
            Self::RowCount => Ok(()),
            Self::Column(accumulator) => accumulator.accumulate(batch, ctx),
            Self::MinMaxFloat {
                inner, nan_count, ..
            } => {
                inner.accumulate(batch, ctx)?;
                nan_count.accumulate(batch, ctx)
            }
            Self::Sum {
                sum,
                non_null_count,
            } => {
                sum.accumulate(batch, ctx)?;
                non_null_count.accumulate(batch, ctx)
            }
        }
    }

    fn finish(self, row_count: u64) -> VortexResult<Scalar> {
        match self {
            Self::RowCount => Ok(Scalar::from(row_count as i64)),
            Self::Column(mut accumulator) => accumulator.finish(),
            Self::MinMaxFloat {
                is_max,
                mut inner,
                mut nan_count,
            } => {
                let nans = nan_count
                    .finish()?
                    .as_primitive()
                    .typed_value::<u64>()
                    .ok_or_else(|| vortex_err!("NaN count must not be null"))?;
                let value = inner.finish()?;
                if nans > 0 && (is_max || value.is_null()) {
                    nan_scalar(value.dtype())
                } else {
                    Ok(value)
                }
            }
            Self::Sum {
                mut sum,
                mut non_null_count,
            } => {
                let count = non_null_count
                    .finish()?
                    .as_primitive()
                    .typed_value::<u64>()
                    .ok_or_else(|| vortex_err!("non-null count must not be null"))?;
                let value = sum.finish()?;
                if count == 0 {
                    Ok(Scalar::null(value.dtype().clone()))
                } else {
                    Ok(value)
                }
            }
        }
    }
}

fn convert_chunk(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<StructArray> {
    let array_result = array.optimize_recursive(ctx.session())?;
    Ok(if let Some(array) = array_result.as_opt::<Struct>() {
        array.into_owned()
    } else if let Some(array) = array_result.as_opt::<ScalarFn>()
        && let Some(pack_options) = array.scalar_fn().as_opt::<Pack>()
    {
        StructArray::new(
            pack_options.names.clone(),
            array.children(),
            array.len(),
            pack_options.nullability.into(),
        )
    } else {
        array_result.execute::<Canonical>(ctx)?.into_struct()
    })
}

/// Evaluate the requested aggregates over the (optionally filtered) data source and write a
/// single-row Arrow record batch — one column per aggregate, in request order — to the
/// `FFI_ArrowArrayStream` at `stream_addr`.
///
/// `agg_kinds[i]` and `agg_columns[i]` describe aggregate `i`; the column entry is ignored
/// (and may be null) for count(*). The filter pointer is a borrowed `NativeExpression`
/// (0 for none).
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeAggregate_compute(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    data_source_ptr: jlong,
    agg_kinds: JByteArray,
    agg_columns: JObjectArray,
    filter_ptr: jlong,
    stream_addr: jlong,
) {
    try_or_throw(&mut env, |env| {
        if stream_addr == 0 {
            throw_runtime!("null arrow stream address");
        }
        let ds = unsafe { NativeDataSource::from_ptr(data_source_ptr) };
        let session = unsafe { session_ref(session_ptr) };

        let kinds: Vec<u8> = env.convert_byte_array(&agg_kinds)?;
        let column_count = agg_columns.len(env)?;
        if kinds.len() != column_count {
            throw_runtime!("aggregate kinds and columns must have the same length");
        }

        // Deduplicated projection columns, in first-appearance order.
        let mut columns: Vec<String> = Vec::with_capacity(kinds.len());
        let mut specs: Vec<AggSpec> = Vec::with_capacity(kinds.len());
        for (i, code) in kinds.iter().enumerate() {
            let kind = AggKind::from_code(*code)?;
            if kind == AggKind::CountStar {
                specs.push(AggSpec {
                    kind,
                    field_index: usize::MAX,
                });
                continue;
            }
            let obj = agg_columns.get_element(env, i)?;
            let s = env.cast_local::<JString>(obj)?;
            let name: String = s.try_to_string(env)?;
            let field_index = columns.iter().position(|c| c == &name).unwrap_or_else(|| {
                columns.push(name);
                columns.len() - 1
            });
            specs.push(AggSpec { kind, field_index });
        }

        let filter = if filter_ptr == 0 {
            None
        } else {
            Some(unsafe { &*(filter_ptr as *const Expression) }.clone())
        };

        // Column aggregates need the input dtypes to build accumulators.
        let source_dtype = ds.inner().dtype().clone();
        let field_dtype = |name: &str| -> VortexResult<DType> {
            source_dtype
                .as_struct_fields_opt()
                .and_then(|fields| fields.field(name))
                .ok_or_else(|| vortex_err!("aggregate column {name} not found in data source"))
        };

        let mut states: Vec<AggState> = Vec::with_capacity(specs.len());
        for spec in &specs {
            if spec.kind == AggKind::CountStar {
                states.push(AggState::RowCount);
            } else {
                let dtype = field_dtype(&columns[spec.field_index])?;
                states.push(spec.kind.state(dtype)?);
            }
        }

        let projection = if columns.is_empty() {
            root()
        } else {
            let fields: Vec<FieldName> = columns
                .iter()
                .map(|name| Arc::<str>::from(name.as_str()).into())
                .collect();
            select(fields, root())
        };
        let request = ScanRequest {
            projection,
            filter,
            selection: Selection::All,
            ..Default::default()
        };

        let mut row_count: u64 = 0;
        RUNTIME.block_on(async {
            let scan = ds.inner().scan(request).await?;
            let mut partitions = scan.partitions();
            let mut ctx = session.create_execution_ctx();
            while let Some(partition) = partitions.next().await {
                let mut stream = partition?.execute()?;
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk?;
                    row_count += chunk.len() as u64;
                    if columns.is_empty() {
                        continue;
                    }
                    let chunk = convert_chunk(chunk, &mut ctx)?;
                    for (spec, state) in specs.iter().zip(states.iter_mut()) {
                        if spec.kind != AggKind::CountStar {
                            state.accumulate(chunk.unmasked_field(spec.field_index), &mut ctx)?;
                        }
                    }
                }
            }
            VortexResult::Ok(())
        })?;

        // Assemble the single-row result: counts as non-nullable i64, everything else in the
        // accumulator's return dtype.
        let i64_dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut names: Vec<FieldName> = Vec::with_capacity(specs.len());
        let mut fields: Vec<ArrayRef> = Vec::with_capacity(specs.len());
        for (i, (spec, state)) in specs.iter().zip(states.into_iter()).enumerate() {
            let mut scalar = state.finish(row_count)?;
            if spec.kind == AggKind::Count {
                scalar = scalar.cast(&i64_dtype)?;
            }
            names.push(FieldName::from(format!("agg_{i}")));
            fields.push(ConstantArray::new(scalar, 1).into_array());
        }
        let result = StructArray::try_new(names.into(), fields, 1, Validity::NonNullable)?;

        let result_dtype = result.dtype().clone();
        let schema = Arc::new(session.arrow().to_arrow_schema(&result_dtype)?);
        let target = Arc::new(Field::new_struct("", schema.fields().clone(), false));
        let mut ctx = session.create_execution_ctx();
        let arrow =
            session
                .arrow()
                .execute_arrow(result.into_array(), Some(target.as_ref()), &mut ctx)?;
        let batch = RecordBatch::from(arrow.as_struct().clone());

        let iter = std::iter::once(Ok::<RecordBatch, ArrowError>(batch));
        let reader = RecordBatchIteratorAdapter::new(iter, schema);
        let arrow_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        unsafe {
            ptr::write(stream_addr as *mut FFI_ArrowArrayStream, arrow_stream);
        }
        Ok(())
    });
}
