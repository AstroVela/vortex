// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compact integer compression with fixed bin identifiers and packed offsets.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_error::VortexResult;
use vortex_range_packed::RangePacked;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::schemes::range_packed_compact_verdict;
use crate::schemes::sample_primitive_one_percent;

/// Compress integers with one fixed range table and packed offsets.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RangePackedScheme;

impl Scheme for RangePackedScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.range_packed"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![RangePacked.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, best_native, _compress_ctx, exec_ctx| {
                let sample = sample_primitive_one_percent(data.array_as_primitive(), exec_ctx)?;
                let before_nbytes = sample.nbytes();
                let encoded = RangePacked::from_primitive(sample.as_view(), exec_ctx)?;
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
        Ok(RangePacked::from_primitive(data.array_as_primitive(), exec_ctx)?.into_array())
    }
}
