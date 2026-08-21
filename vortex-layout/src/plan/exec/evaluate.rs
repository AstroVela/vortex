// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::time::Instant;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Bool;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::FieldNames;
use vortex_array::serde::SerializedArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use super::ArraySummary;
use super::Completion;
use super::FlatEncoding;
use super::Operation;
use super::ResolvedArray;
use super::ResolvedValue;
use super::RunnableTask;
use crate::segments::SegmentSource;

pub async fn evaluate(
    task: RunnableTask,
    source: &dyn SegmentSource,
    session: &VortexSession,
) -> Completion {
    let started = matches!(task.operation, Operation::EvaluatePredicate { .. }).then(Instant::now);
    let result = evaluate_inner(&task, source, session).await;
    Completion {
        task: task.id,
        output: task.output,
        elapsed_ns: started.map_or(0, |started| {
            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
        }),
        result,
    }
}

async fn evaluate_inner(
    task: &RunnableTask,
    source: &dyn SegmentSource,
    session: &VortexSession,
) -> VortexResult<ResolvedValue> {
    match &task.operation {
        Operation::Read { segment } => Ok(ResolvedValue::Segment(source.request(*segment).await?)),
        Operation::DecodeFlat {
            encoding,
            row_count,
        } => {
            let segment = segment_input(task, 0)?;
            let array = match encoding {
                FlatEncoding::RawI64 => {
                    let values = Buffer::<i64>::from_byte_buffer(segment.as_host().clone());
                    if values.len() != *row_count {
                        vortex_bail!(
                            "raw segment has {} rows, expected {row_count}",
                            values.len()
                        );
                    }
                    PrimitiveArray::new(values, Validity::NonNullable).into_array()
                }
                FlatEncoding::Serialized {
                    dtype,
                    read_ctx,
                    array_tree,
                } => {
                    let serialized = if let Some(array_tree) = array_tree {
                        SerializedArray::from_flatbuffer_and_segment(
                            array_tree.clone(),
                            segment.clone(),
                        )?
                    } else {
                        SerializedArray::try_from(segment.clone())?
                    };
                    serialized.decode(dtype, *row_count, read_ctx, session)?
                }
            };
            Ok(ResolvedValue::Array(ResolvedArray::plain(array)))
        }
        Operation::EvaluatePredicate {
            local_ranges,
            predicate,
            ..
        } => {
            let demand_array = array_input(task, local_ranges.len())?;
            let demand_summary = demand_array.boolean_summary()?;
            let demand = boolean_bits(demand_array, session)?;
            let input_len = local_ranges.iter().map(|range| range.len()).sum::<usize>();
            if demand.len() != input_len {
                vortex_bail!("predicate demand length does not match its row range");
            }
            let result = if let [local_range] = local_ranges.as_slice() {
                let array = primitive_array(array_input(task, 0)?, session)?;
                let values = array
                    .as_slice::<i64>()
                    .get(local_range.clone())
                    .ok_or_else(|| {
                        vortex_error::vortex_err!("local range is outside its Flat array")
                    })?;
                evaluate_predicate_slice(values, &demand, demand_summary.true_count, *predicate)
            } else {
                let mut result = BitBufferMut::with_capacity(demand.len());
                let mut offset = 0;
                for (input, local_range) in local_ranges.iter().enumerate() {
                    let array = primitive_array(array_input(task, input)?, session)?;
                    let values = array
                        .as_slice::<i64>()
                        .get(local_range.clone())
                        .ok_or_else(|| {
                            vortex_error::vortex_err!("local range is outside its Flat array")
                        })?;
                    let slice_demand = demand.slice(offset..offset + values.len());
                    result.append_buffer(&evaluate_predicate_slice(
                        values,
                        &slice_demand,
                        slice_demand.true_count(),
                        *predicate,
                    ));
                    offset += values.len();
                }
                result.freeze()
            };
            let true_count = result.true_count();
            Ok(ResolvedValue::Array(ResolvedArray::boolean(
                BoolArray::new(result, Validity::NonNullable).into_array(),
                true_count,
            )))
        }
        Operation::CombineDemand { .. } => {
            let lhs = boolean_bits(array_input(task, 0)?, session)?;
            let rhs = boolean_bits(array_input(task, 1)?, session)?;
            if lhs.len() != rhs.len() {
                vortex_bail!("cannot combine demand masks with different lengths");
            }
            let result = &lhs & &rhs;
            let true_count = result.true_count();
            Ok(ResolvedValue::Array(ResolvedArray::boolean(
                BoolArray::new(result, Validity::NonNullable).into_array(),
                true_count,
            )))
        }
        Operation::SelectFlat { local_ranges } => {
            let selection_array = array_input(task, local_ranges.len())?;
            let selection_summary = selection_array.boolean_summary()?;
            let input_len = local_ranges.iter().map(|range| range.len()).sum::<usize>();
            if selection_summary.len != input_len {
                vortex_bail!("selection length does not match its row range");
            }
            if selection_summary.true_count == 0 {
                return Ok(ResolvedValue::Array(ResolvedArray::plain(
                    PrimitiveArray::from_iter(std::iter::empty::<i64>()).into_array(),
                )));
            }
            if selection_summary.true_count == selection_summary.len
                && let [local_range] = local_ranges.as_slice()
            {
                let array = &array_input(task, 0)?.array;
                return Ok(ResolvedValue::Array(ResolvedArray::plain(
                    array.slice(local_range.clone())?,
                )));
            }
            let selection = boolean_bits(selection_array, session)?;
            let mut selected = BufferMut::with_capacity(selection_summary.true_count);
            let mut offset = 0;
            for (input, local_range) in local_ranges.iter().enumerate() {
                let array = primitive_array(array_input(task, input)?, session)?;
                let values = array
                    .as_slice::<i64>()
                    .get(local_range.clone())
                    .ok_or_else(|| {
                        vortex_error::vortex_err!("local range is outside its Flat array")
                    })?;
                selection
                    .slice(offset..offset + values.len())
                    .for_each_set_index(|index| selected.push(values[index]));
                offset += values.len();
            }
            Ok(ResolvedValue::Array(ResolvedArray::plain(
                PrimitiveArray::new(selected.freeze(), Validity::NonNullable).into_array(),
            )))
        }
        Operation::PackStruct { names, len } => {
            let arrays = if task.inputs.is_empty() {
                Vec::new()
            } else {
                task.inputs
                    .iter()
                    .map(|value| match value {
                        ResolvedValue::Array(array) => Ok(array.array.clone()),
                        ResolvedValue::Segment(_) => {
                            vortex_bail!("PackStruct received a segment input")
                        }
                    })
                    .collect::<VortexResult<Vec<ArrayRef>>>()?
            };
            let array = pack_struct_array(names.clone(), arrays, *len)?;
            Ok(ResolvedValue::Array(ResolvedArray::plain(array)))
        }
    }
}

fn evaluate_predicate_slice(
    values: &[i64],
    demand: &BitBuffer,
    true_count: usize,
    predicate: super::Predicate,
) -> BitBuffer {
    if true_count == demand.len() {
        BitBuffer::collect_bool_multiversioned(demand.len(), |index| {
            // The collector visits `0..demand.len()`, which equals `values.len()`.
            predicate.matches(unsafe { *values.get_unchecked(index) })
        })
    } else if true_count <= demand.len() / 5 {
        let mut result = BitBufferMut::new_unset(demand.len());
        demand.for_each_set_index(|index| {
            if predicate.matches(unsafe { *values.get_unchecked(index) }) {
                // SAFETY: `index` came from a bit buffer with the same length as `result`.
                unsafe { result.set_unchecked(index) };
            }
        });
        result.freeze()
    } else {
        demand.map_cmp(|index, selected| {
            // `map_cmp` visits `0..demand.len()`, which equals `values.len()`.
            selected && predicate.matches(unsafe { *values.get_unchecked(index) })
        })
    }
}

pub(crate) fn pack_struct_array(
    names: FieldNames,
    mut arrays: Vec<ArrayRef>,
    len: usize,
) -> VortexResult<ArrayRef> {
    if arrays.is_empty() {
        arrays = names
            .iter()
            .map(|_| PrimitiveArray::from_iter(std::iter::empty::<i64>()).into_array())
            .collect();
    }
    if arrays.iter().any(|array| array.len() != len) {
        vortex_bail!("PackStruct inputs are not aligned");
    }
    Ok(StructArray::try_new(names, arrays, len, Validity::NonNullable)?.into_array())
}

fn segment_input(
    task: &RunnableTask,
    index: usize,
) -> VortexResult<&vortex_array::buffer::BufferHandle> {
    match task.inputs.get(index) {
        Some(ResolvedValue::Segment(buffer)) => Ok(buffer),
        _ => vortex_bail!("task input {index} is not a segment"),
    }
}

fn array_input(task: &RunnableTask, index: usize) -> VortexResult<&ResolvedArray> {
    match task.inputs.get(index) {
        Some(ResolvedValue::Array(array)) => Ok(array),
        _ => vortex_bail!("task input {index} is not an array"),
    }
}

pub(crate) fn primitive_values(
    array: &ResolvedArray,
    session: &VortexSession,
) -> VortexResult<Vec<i64>> {
    if !matches!(array.summary, ArraySummary::None) {
        vortex_bail!("expected a value array, got a boolean mask");
    }
    let mut ctx = session.create_execution_ctx();
    Ok(array
        .array
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?
        .as_slice::<i64>()
        .to_vec())
}

fn primitive_array(array: &ResolvedArray, session: &VortexSession) -> VortexResult<PrimitiveArray> {
    if !matches!(array.summary, ArraySummary::None) {
        vortex_bail!("expected a value array, got a boolean mask");
    }
    if array.array.is::<Primitive>() {
        return Ok(array.array.as_::<Primitive>().into_owned());
    }
    let mut ctx = session.create_execution_ctx();
    array.array.clone().execute::<PrimitiveArray>(&mut ctx)
}

fn boolean_bits(array: &ResolvedArray, session: &VortexSession) -> VortexResult<BitBuffer> {
    let summary = array.boolean_summary()?;
    if summary.len != array.array.len() || summary.true_count > summary.len {
        vortex_bail!("invalid boolean-mask summary");
    }
    if array.array.is::<Bool>() {
        return Ok(array.array.as_::<Bool>().to_bit_buffer());
    }
    let mut ctx = session.create_execution_ctx();
    Ok(array
        .array
        .clone()
        .execute::<BoolArray>(&mut ctx)?
        .to_bit_buffer())
}

pub fn boolean_values(array: &ResolvedArray, session: &VortexSession) -> VortexResult<Vec<bool>> {
    let summary = array.boolean_summary()?;
    if summary.len != array.array.len() || summary.true_count > summary.len {
        vortex_bail!("invalid boolean-mask summary");
    }
    let mut ctx = session.create_execution_ctx();
    let values = array
        .array
        .clone()
        .execute::<BoolArray>(&mut ctx)?
        .to_bit_buffer()
        .iter()
        .collect::<Vec<_>>();
    if values.iter().filter(|value| **value).count() != summary.true_count {
        vortex_bail!("boolean-mask summary true count is incorrect");
    }
    Ok(values)
}
