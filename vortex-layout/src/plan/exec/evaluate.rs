// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::time::Instant;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
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
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::ArraySummary;
use super::CachedPredicate;
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
    let started = matches!(
        task.operation,
        Operation::EvaluatePredicate { .. } | Operation::MergeDemandFragments
    )
    .then(Instant::now);
    let mut read_bytes = None;
    let result = evaluate_inner(&task, source, session, &mut read_bytes).await;
    Completion {
        task: task.id,
        output: task.output,
        elapsed_ns: started.map_or(0, |started| {
            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
        }),
        read_bytes,
        result,
    }
}

async fn evaluate_inner(
    task: &RunnableTask,
    source: &dyn SegmentSource,
    session: &VortexSession,
    read_bytes: &mut Option<usize>,
) -> VortexResult<ResolvedValue> {
    match &task.operation {
        Operation::Read { segment, .. } => {
            let segment = source.request(*segment).await?;
            *read_bytes = Some(segment.len());
            Ok(ResolvedValue::Segment(segment))
        }
        Operation::ReadDecodeFlat {
            segment,
            encoding,
            row_count,
            predicates,
            ..
        } => {
            let segment = source.request(*segment).await?;
            *read_bytes = Some(segment.len());
            let resolved =
                ResolvedArray::plain(decode_flat(&segment, encoding, *row_count, session)?);
            let mut cached = Vec::with_capacity(predicates.len());
            if !predicates.is_empty() {
                let array = primitive_array(&resolved, session)?;
                for (conjunct, predicate, demand, input_true_count) in predicates {
                    let started = Instant::now();
                    let values = evaluate_predicate_slice(
                        array.as_slice::<i64>(),
                        demand,
                        *input_true_count,
                        *predicate,
                    );
                    cached.push(CachedPredicate {
                        conjunct: *conjunct,
                        values,
                        evaluated: demand.clone(),
                        input_true_count: *input_true_count,
                        elapsed_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    });
                }
            }
            Ok(ResolvedValue::Array(ResolvedArray::plain_with_predicates(
                resolved.array,
                cached,
            )))
        }
        Operation::DecodeFlat {
            encoding,
            row_count,
        } => {
            let segment = segment_input(task, 0)?;
            let array = decode_flat(segment, encoding, *row_count, session)?;
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
            Ok(ResolvedValue::Array(ResolvedArray::boolean(
                BoolArray::new(result.clone(), Validity::NonNullable).into_array(),
                result,
            )))
        }
        Operation::CombineDemand { .. } => {
            let lhs = boolean_bits(array_input(task, 0)?, session)?;
            let rhs = boolean_bits(array_input(task, 1)?, session)?;
            if lhs.len() != rhs.len() {
                vortex_bail!("cannot combine demand masks with different lengths");
            }
            let result = &lhs & &rhs;
            Ok(ResolvedValue::Array(ResolvedArray::boolean(
                BoolArray::new(result.clone(), Validity::NonNullable).into_array(),
                result,
            )))
        }
        Operation::MergeDemandFragments => {
            let mut len = 0;
            let mut fragments = Vec::with_capacity(task.inputs.len());
            for input in 0..task.inputs.len() {
                let fragment = boolean_bits(array_input(task, input)?, session)?;
                len += fragment.len();
                fragments.push(fragment);
            }
            let mut result = BitBufferMut::with_capacity(len);
            for fragment in fragments {
                result.append_buffer(&fragment);
            }
            let result = result.freeze();
            Ok(ResolvedValue::Array(ResolvedArray::boolean(
                BoolArray::new(result.clone(), Validity::NonNullable).into_array(),
                result,
            )))
        }
        Operation::SelectFlat {
            local_ranges,
            selection_ranges,
            selection_all_true: _,
            pack_names,
        } => {
            if local_ranges.len() != selection_ranges.len() {
                vortex_bail!("selection input and row-range counts differ");
            }
            let selection_array = array_input(task, local_ranges.len())?;
            let selection_summary = selection_array.boolean_summary()?;
            if selection_ranges.iter().any(|range| {
                range.start > range.end || range.end > selection_summary.len || range.is_empty()
            }) {
                vortex_bail!("selection range is empty or outside its demand mask");
            }
            if selection_summary.true_count == 0 {
                return selected_output(
                    PrimitiveArray::from_iter(std::iter::empty::<i64>()).into_array(),
                    pack_names.as_ref(),
                );
            }
            if selection_summary.true_count == selection_summary.len {
                let covers_selection = selection_ranges
                    .first()
                    .is_some_and(|range| range.start == 0)
                    && selection_ranges
                        .last()
                        .is_some_and(|range| range.end == selection_summary.len)
                    && selection_ranges
                        .windows(2)
                        .all(|ranges| ranges[0].end == ranges[1].start);
                if covers_selection {
                    let chunks = local_ranges
                        .iter()
                        .enumerate()
                        .map(|(input, range)| array_input(task, input)?.array.slice(range.clone()))
                        .collect::<VortexResult<Vec<_>>>()?;
                    let array = if let [chunk] = chunks.as_slice() {
                        chunk.clone()
                    } else {
                        let dtype = chunks[0].dtype().clone();
                        ChunkedArray::try_new(chunks, dtype)?.into_array()
                    };
                    return selected_output(array, pack_names.as_ref());
                }
            }
            let chunks = local_ranges
                .iter()
                .enumerate()
                .map(|(input, range)| array_input(task, input)?.array.slice(range.clone()))
                .collect::<VortexResult<Vec<_>>>()?;
            let array = if let [chunk] = chunks.as_slice() {
                chunk.clone()
            } else {
                let dtype = chunks[0].dtype().clone();
                ChunkedArray::try_new(chunks, dtype)?.into_array()
            };
            let mut included_selection = BitBufferMut::with_capacity(array.len());
            for range in selection_ranges {
                included_selection.append_buffer(&selection_summary.values.slice(range.clone()));
            }
            let included_selection = included_selection.freeze();
            if included_selection.len() != array.len()
                || included_selection.true_count() != selection_summary.true_count
            {
                vortex_bail!("selection ranges do not cover every selected row");
            }
            selected_output(
                array.filter(Mask::from_buffer(included_selection))?,
                pack_names.as_ref(),
            )
        }
        Operation::SelectStruct {
            field_local_ranges,
            selection_ranges,
            selection_all_true: _,
            names,
        } => {
            let value_input_count = field_local_ranges.iter().map(Vec::len).sum::<usize>();
            let selection_array = array_input(task, value_input_count)?;
            let selection_summary = selection_array.boolean_summary()?;
            let mut input = 0;
            let mut fields = Vec::with_capacity(field_local_ranges.len());
            for local_ranges in field_local_ranges {
                let chunks = local_ranges
                    .iter()
                    .map(|range| {
                        let array = array_input(task, input)?;
                        input += 1;
                        array.array.slice(range.clone())
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                let array = if let [chunk] = chunks.as_slice() {
                    chunk.clone()
                } else {
                    let dtype = chunks[0].dtype().clone();
                    ChunkedArray::try_new(chunks, dtype)?.into_array()
                };
                fields.push(array);
            }
            let unfiltered_len = fields.first().map_or(0, |field| field.len());
            let array = pack_struct_array(names.clone(), fields, unfiltered_len)?;
            let mut included_selection = BitBufferMut::with_capacity(unfiltered_len);
            for range in selection_ranges {
                included_selection.append_buffer(&selection_summary.values.slice(range.clone()));
            }
            let included_selection = included_selection.freeze();
            if included_selection.len() != unfiltered_len
                || included_selection.true_count() != selection_summary.true_count
            {
                vortex_bail!("struct selection ranges do not cover every selected row");
            }
            let array = if included_selection.true_count() == included_selection.len() {
                array
            } else {
                array.filter(Mask::from_buffer(included_selection))?
            };
            Ok(ResolvedValue::Array(ResolvedArray::plain(array)))
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

fn decode_flat(
    segment: &vortex_array::buffer::BufferHandle,
    encoding: &FlatEncoding,
    row_count: usize,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    match encoding {
        FlatEncoding::RawI64 => {
            let values = Buffer::<i64>::from_byte_buffer(segment.as_host().clone());
            if values.len() != row_count {
                vortex_bail!(
                    "raw segment has {} rows, expected {row_count}",
                    values.len()
                );
            }
            Ok(PrimitiveArray::new(values, Validity::NonNullable).into_array())
        }
        FlatEncoding::Serialized {
            dtype,
            read_ctx,
            array_tree,
        } => {
            let serialized = if let Some(array_tree) = array_tree {
                SerializedArray::from_flatbuffer_and_segment(array_tree.clone(), segment.clone())?
            } else {
                SerializedArray::try_from(segment.clone())?
            };
            serialized.decode(dtype, row_count, read_ctx, session)
        }
    }
}

fn selected_output(
    array: ArrayRef,
    pack_names: Option<&FieldNames>,
) -> VortexResult<ResolvedValue> {
    let array = if let Some(names) = pack_names {
        let len = array.len();
        pack_struct_array(names.clone(), vec![array], len)?
    } else {
        array
    };
    Ok(ResolvedValue::Array(ResolvedArray::plain(array)))
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

fn boolean_bits(array: &ResolvedArray, _session: &VortexSession) -> VortexResult<BitBuffer> {
    let summary = array.boolean_summary()?;
    if summary.len != array.array.len() || summary.true_count > summary.len {
        vortex_bail!("invalid boolean-mask summary");
    }
    Ok(summary.values.clone())
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
