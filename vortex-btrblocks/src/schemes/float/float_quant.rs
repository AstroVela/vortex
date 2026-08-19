// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless float quantization with a fixed frame-of-reference child.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FoR;
use vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked;
use vortex_float_quant::FloatQuant;
use vortex_float_quant::analyze_float_quant;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::normalize_null_values;

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

        let biased = FloatQuant::primary_for_primitive(
            primitive.as_view(),
            analysis.k,
            analysis.primary_min,
        )?;
        // SAFETY: The analysis computes this width from the exact primary minimum and maximum.
        let compressed_primary =
            unsafe { bitpack_encode_unchecked(biased, analysis.primary_bit_width)? }.into_array();
        let reference = match primitive.ptype() {
            PType::F32 => Scalar::from(u32::try_from(analysis.primary_min)?),
            PType::F64 => Scalar::from(analysis.primary_min),
            _ => unreachable!(),
        };
        let compressed_primary = FoR::try_new(compressed_primary, reference)?.into_array();
        Ok(
            FloatQuant::try_new(compressed_primary, None, primitive.ptype(), analysis.k)?
                .into_array(),
        )
    }
}
