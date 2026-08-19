// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compact float compression with ordered IEEE bits and fixed range packing.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Primitive;
use vortex_block_residual::OrderedFloat;
use vortex_block_residual::OrderedFloatArraySlotsExt;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_range_packed::RangePacked;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::schemes::range_packed_compact_verdict;
use crate::schemes::sample_primitive_one_percent;

/// Compress floats as fixed-bin packed ordered IEEE bits.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct OrderedRangePackedScheme;

impl Scheme for OrderedRangePackedScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.ordered_range_packed"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![OrderedFloat.id(), RangePacked.id()]
    }

    fn num_children(&self) -> usize {
        1
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
            |_compressor, data, best_native, _compress_ctx, exec_ctx| {
                let sample = sample_primitive_one_percent(data.array_as_primitive(), exec_ctx)?;
                let before_nbytes = sample.nbytes();
                let ordered = OrderedFloat::from_primitive(sample.as_view())?;
                let encoded =
                    RangePacked::from_primitive(ordered.encoded().as_::<Primitive>(), exec_ctx)?;
                Ok(range_packed_compact_verdict(
                    before_nbytes,
                    encoded.nbytes(),
                    best_native,
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
        let primitive = data.array_as_primitive();
        let ordered = OrderedFloat::from_primitive(primitive)?;
        let encoded = RangePacked::from_primitive(ordered.encoded().as_::<Primitive>(), exec_ctx)?;
        Ok(OrderedFloat::try_new(encoded.into_array(), primitive.ptype())?.into_array())
    }
}
