// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native range entropy compression for primitive numeric arrays.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_error::VortexResult;
use vortex_range_entropy::RangeEntropy;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

const RESTART_BLOCK_LEN: usize = 8192;

/// Native range bins plus tANS for primitive numeric arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RangeEntropyScheme;

impl Scheme for RangeEntropyScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.numeric.range_entropy"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int() || canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![RangeEntropy.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(
            RangeEntropy::from_primitive(data.array_as_primitive(), RESTART_BLOCK_LEN)?
                .into_array(),
        )
    }
}
