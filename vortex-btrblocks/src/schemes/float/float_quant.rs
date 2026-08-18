// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless float quantization with recursively compressed integer children.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked;
use vortex_float_quant::FloatQuant;
use vortex_float_quant::FloatQuantAnalysis;
use vortex_float_quant::FloatQuantArraySlotsExt;
use vortex_float_quant::analyze_float_quant;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

const ALWAYS_USE_MIN_RATIO: f64 = 2.0;

fn is_strong_constant_split(analysis: FloatQuantAnalysis, ptype: PType) -> bool {
    analysis.secondary_is_constant
        && analysis.primary_bit_width > 0
        && ptype.bit_width() as f64 / f64::from(analysis.primary_bit_width) >= ALWAYS_USE_MIN_RATIO
}

/// FloatQuant split with normal BtrBlocks compression for both latent children.
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
        vec![FloatQuant.id()]
    }

    fn num_children(&self) -> usize {
        2
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let primitive = data.array_as_primitive();
        if compress_ctx.finished_cascading() || primitive.ptype() != PType::F64 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = data.array_as_primitive();
        let Some(analysis) = analyze_float_quant(primitive) else {
            return Ok(primitive.array().clone());
        };
        if is_strong_constant_split(analysis, primitive.ptype()) {
            let biased =
                FloatQuant::primary_for_primitive(primitive, analysis.k, analysis.primary_min)?;
            // SAFETY: The analysis computes this width from the exact primary minimum and maximum.
            let compressed_primary =
                unsafe { bitpack_encode_unchecked(biased, analysis.primary_bit_width)? }
                    .into_array();
            let reference = match primitive.ptype() {
                PType::F32 => Scalar::from(u32::try_from(analysis.primary_min)?),
                PType::F64 => Scalar::from(analysis.primary_min),
                _ => unreachable!(),
            };
            let compressed_primary = FoR::try_new(compressed_primary, reference)?.into_array();
            let compressed_secondary = match primitive.ptype() {
                PType::F32 => ConstantArray::new(Scalar::from(0u32), primitive.len()).into_array(),
                PType::F64 => ConstantArray::new(Scalar::from(0u64), primitive.len()).into_array(),
                _ => unreachable!(),
            };
            return Ok(FloatQuant::try_new(
                compressed_primary,
                compressed_secondary,
                primitive.ptype(),
                analysis.k,
            )?
            .into_array());
        }

        let encoded = FloatQuant::from_primitive(primitive, analysis.k)?;
        let primary = encoded
            .primary()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?;
        let secondary = encoded
            .secondary()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?;
        let compressed_primary = compressor.compress_child(
            &primary.into_array(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;
        let compressed_secondary = compressor.compress_child(
            &secondary.into_array(),
            &compress_ctx,
            self.id(),
            1,
            exec_ctx,
        )?;
        Ok(FloatQuant::try_new(
            compressed_primary,
            compressed_secondary,
            primitive.ptype(),
            analysis.k,
        )?
        .into_array())
    }
}
