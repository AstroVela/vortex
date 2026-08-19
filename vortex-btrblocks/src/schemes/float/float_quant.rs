// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless float quantization with a fixed frame-of-reference child.

use std::ops::Range;

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::PType;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar::Scalar;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_primitive_map;
use vortex_float_quant::FloatQuant;
use vortex_float_quant::FloatQuantAnalysis;
use vortex_float_quant::analyze_float_quant;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::normalize_null_values;

const SAMPLE_BLOCK_LEN: usize = 64;
const MIN_SAMPLE_BLOCKS: usize = 16;
const SAMPLE_BLOCK_MULTIPLE: usize = 16;

/// FloatQuant split with a fixed frame-of-reference primary child.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatQuantScheme;

impl Scheme for FloatQuantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.float_quant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![FloatQuant.id(), FoR.id(), BitPacked.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, _best_so_far, _compress_ctx, exec_ctx| {
                let sample = float_quant_sample(data.array_as_primitive(), exec_ctx)?;
                let Some(analysis) = analyze_float_quant(sample.as_view()) else {
                    return Ok(EstimateVerdict::Skip);
                };
                if !analysis.secondary_is_constant || analysis.primary_bit_width == 0 {
                    return Ok(EstimateVerdict::Skip);
                }

                let before_nbytes = sample.nbytes();
                let compressed = encode_constant_secondary(sample.as_view(), analysis)?;
                let after_nbytes = compressed.nbytes();
                if after_nbytes == 0 || after_nbytes >= before_nbytes {
                    return Ok(EstimateVerdict::Skip);
                }

                Ok(EstimateVerdict::Ratio(
                    before_nbytes as f64 / after_nbytes as f64,
                ))
            },
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let source = data.array_as_primitive();
        let primitive = normalize_null_values(source, exec_ctx)?;
        let Some(analysis) = analyze_float_quant(primitive.as_view()) else {
            return Ok(source.array().clone());
        };
        if !analysis.secondary_is_constant || analysis.primary_bit_width == 0 {
            return Ok(source.array().clone());
        }

        encode_constant_secondary(primitive.as_view(), analysis)
    }
}

fn encode_constant_secondary(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    analysis: FloatQuantAnalysis,
) -> VortexResult<ArrayRef> {
    let (packed, latent_ptype, reference) = match primitive.ptype() {
        PType::F32 => {
            let primary_min = u32::try_from(analysis.primary_min)?;
            let packed = bitpack_primitive_map(
                primitive.as_slice::<f32>(),
                analysis.primary_bit_width,
                |value| (ordered_u32(value.to_bits()) >> analysis.k) - primary_min,
            )
            .into_byte_buffer();
            (packed, PType::U32, Scalar::from(primary_min))
        }
        PType::F64 => {
            let packed = bitpack_primitive_map(
                primitive.as_slice::<f64>(),
                analysis.primary_bit_width,
                |value| (ordered_u64(value.to_bits()) >> analysis.k) - analysis.primary_min,
            )
            .into_byte_buffer();
            (packed, PType::U64, Scalar::from(analysis.primary_min))
        }
        _ => unreachable!(),
    };
    let compressed_primary = BitPacked::try_new(
        BufferHandle::new_host(packed),
        latent_ptype,
        primitive.validity()?,
        None,
        analysis.primary_bit_width,
        primitive.len(),
        0,
    )?
    .into_array();
    let compressed_primary = FoR::try_new(compressed_primary, reference)?.into_array();
    Ok(FloatQuant::try_new(compressed_primary, None, primitive.ptype(), analysis.k)?.into_array())
}

#[inline]
fn ordered_u32(bits: u32) -> u32 {
    if bits & (1_u32 << 31) == 0 {
        bits ^ (1_u32 << 31)
    } else {
        !bits
    }
}

#[inline]
fn ordered_u64(bits: u64) -> u64 {
    if bits & (1_u64 << 63) == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

fn float_quant_sample(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let sample_blocks = (primitive.len() / 100 / SAMPLE_BLOCK_LEN)
        .next_multiple_of(SAMPLE_BLOCK_MULTIPLE)
        .max(MIN_SAMPLE_BLOCKS);
    let sample_len = sample_blocks * SAMPLE_BLOCK_LEN;
    if primitive.len() <= sample_len {
        return normalize_null_values(primitive, exec_ctx);
    }

    let ranges = float_quant_sample_ranges(primitive.len(), sample_blocks);
    let validity = primitive.validity()?;
    if validity.definitely_no_nulls() {
        return Ok(match_each_float_ptype!(primitive.ptype(), |T| {
            let values = primitive.as_slice::<T>();
            let mut sample = Vec::with_capacity(sample_len);
            for range in ranges {
                sample.extend_from_slice(&values[range]);
            }
            PrimitiveArray::from_iter(sample)
        }));
    }

    let validity = validity.execute_mask(primitive.len(), exec_ctx)?;
    Ok(match_each_float_ptype!(primitive.ptype(), |T| {
        let values = primitive.as_slice::<T>();
        PrimitiveArray::from_option_iter(
            ranges
                .into_iter()
                .flatten()
                .map(|index| validity.value(index).then_some(values[index])),
        )
    }))
}

fn float_quant_sample_ranges(len: usize, sample_blocks: usize) -> Vec<Range<usize>> {
    let partition_len = len / sample_blocks;
    let long_partitions = len % sample_blocks;
    let mut partition_start = 0;
    (0..sample_blocks)
        .map(|partition_index| {
            let current_partition_len =
                partition_len + usize::from(partition_index < long_partitions);
            let start = partition_start + (current_partition_len - SAMPLE_BLOCK_LEN) / 2;
            partition_start += current_partition_len;
            start..start + SAMPLE_BLOCK_LEN
        })
        .collect()
}
