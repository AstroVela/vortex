// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless float quantization with a fixed frame-of-reference child.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_primitive_map;
use vortex_fastlanes::bitpack_compress::bitpack_primitive_map_pair;
use vortex_float_quant::FloatQuant;
use vortex_float_quant::FloatQuantAnalysis;
use vortex_float_quant::analyze_float_quant;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::normalize_null_values;
use crate::schemes::sample_primitive_one_percent;

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
                let sample = sample_primitive_one_percent(data.array_as_primitive(), exec_ctx)?;
                let Some(analysis) = analyze_float_quant(sample.as_view()) else {
                    return Ok(EstimateVerdict::Skip);
                };
                let before_nbytes = sample.nbytes();
                let compressed = encode_float_quant(sample.as_view(), analysis)?;
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
        encode_float_quant(primitive.as_view(), analysis)
    }
}

fn encode_float_quant(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    analysis: FloatQuantAnalysis,
) -> VortexResult<ArrayRef> {
    let (primary_packed, secondary_packed, latent_ptype, reference) = match primitive.ptype() {
        PType::F16 => {
            let primary_min = u16::try_from(analysis.primary_min)?;
            let values = primitive.as_slice::<f16>();
            let (primary, secondary) = if analysis.secondary_bit_width == 0 {
                (
                    bitpack_primitive_map(values, analysis.primary_bit_width, |value| {
                        (ordered_u16(value.to_bits()) >> analysis.k) - primary_min
                    }),
                    None,
                )
            } else {
                let low_mask = (1_u16 << analysis.k) - 1;
                let (primary, secondary) = bitpack_primitive_map_pair(
                    values,
                    analysis.primary_bit_width,
                    analysis.secondary_bit_width,
                    |value| {
                        let bits = value.to_bits();
                        (
                            (ordered_u16(bits) >> analysis.k) - primary_min,
                            bits & low_mask,
                        )
                    },
                );
                (primary, Some(secondary))
            };
            (
                primary.into_byte_buffer(),
                secondary.map(|packed| packed.into_byte_buffer()),
                PType::U16,
                Scalar::from(primary_min),
            )
        }
        PType::F32 => {
            let primary_min = u32::try_from(analysis.primary_min)?;
            let values = primitive.as_slice::<f32>();
            let (primary, secondary) = if analysis.secondary_bit_width == 0 {
                (
                    bitpack_primitive_map(values, analysis.primary_bit_width, |value| {
                        (ordered_u32(value.to_bits()) >> analysis.k) - primary_min
                    }),
                    None,
                )
            } else {
                let low_mask = (1_u32 << analysis.k) - 1;
                let (primary, secondary) = bitpack_primitive_map_pair(
                    values,
                    analysis.primary_bit_width,
                    analysis.secondary_bit_width,
                    |value| {
                        let bits = value.to_bits();
                        (
                            (ordered_u32(bits) >> analysis.k) - primary_min,
                            bits & low_mask,
                        )
                    },
                );
                (primary, Some(secondary))
            };
            (
                primary.into_byte_buffer(),
                secondary.map(|packed| packed.into_byte_buffer()),
                PType::U32,
                Scalar::from(primary_min),
            )
        }
        PType::F64 => {
            let values = primitive.as_slice::<f64>();
            let (primary, secondary) = if analysis.secondary_bit_width == 0 {
                (
                    bitpack_primitive_map(values, analysis.primary_bit_width, |value| {
                        (ordered_u64(value.to_bits()) >> analysis.k) - analysis.primary_min
                    }),
                    None,
                )
            } else {
                let low_mask = (1_u64 << analysis.k) - 1;
                let (primary, secondary) = bitpack_primitive_map_pair(
                    values,
                    analysis.primary_bit_width,
                    analysis.secondary_bit_width,
                    |value| {
                        let bits = value.to_bits();
                        (
                            (ordered_u64(bits) >> analysis.k) - analysis.primary_min,
                            bits & low_mask,
                        )
                    },
                );
                (primary, Some(secondary))
            };
            (
                primary.into_byte_buffer(),
                secondary.map(|packed| packed.into_byte_buffer()),
                PType::U64,
                Scalar::from(analysis.primary_min),
            )
        }
        _ => unreachable!(),
    };
    let compressed_primary = BitPacked::try_new(
        BufferHandle::new_host(primary_packed),
        latent_ptype,
        primitive.validity()?,
        None,
        analysis.primary_bit_width,
        primitive.len(),
        0,
    )?
    .into_array();
    let compressed_primary = FoR::try_new(compressed_primary, reference)?.into_array();
    let compressed_secondary = secondary_packed
        .map(|packed| {
            BitPacked::try_new(
                BufferHandle::new_host(packed),
                latent_ptype,
                Validity::NonNullable,
                None,
                analysis.secondary_bit_width,
                primitive.len(),
                0,
            )
            .map(IntoArray::into_array)
        })
        .transpose()?;
    Ok(FloatQuant::try_new(
        compressed_primary,
        compressed_secondary,
        primitive.ptype(),
        analysis.k,
    )?
    .into_array())
}

#[inline]
fn ordered_u16(bits: u16) -> u16 {
    if bits & (1_u16 << 15) == 0 {
        bits ^ (1_u16 << 15)
    } else {
        !bits
    }
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
