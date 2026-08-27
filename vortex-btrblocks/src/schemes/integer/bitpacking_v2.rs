// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunk-wise BitPacking integer encoding.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPackedV2;
use vortex_fastlanes::bitpack_v2_compress::bitpack_v2_encode;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::compress_patches_v2;

/// Chunk-wise BitPacking encoding for non-negative integers.
///
/// Unlike [`BitPackingScheme`](super::BitPackingScheme), which packs a column at a single width,
/// this scheme sizes every 1024-element FastLanes chunk on its own, so a locally narrow run of
/// values is not charged for the widest value elsewhere in the column.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BitPackingV2Scheme;

impl Scheme for BitPackingV2Scheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.bitpacking_v2"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![BitPackedV2.id()]
    }

    /// Children: patch indices=0, patch values=1, patch chunk offsets=2.
    fn num_children(&self) -> usize {
        3
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let stats = data.integer_stats(exec_ctx);

        // BitPacking only works for non-negative values.
        if stats.erased().min_is_negative() {
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
        let primitive_array = data.array_as_primitive();
        let ptype = primitive_array.ptype();
        let packed = bitpack_v2_encode(&primitive_array.into_owned(), exec_ctx)?;

        // If every chunk needs the full width, packing bought us nothing.
        if packed
            .bit_widths()
            .iter()
            .all(|&width| width as usize == ptype.bit_width())
        {
            return Ok(data.array().clone());
        }

        let packed_stats = packed.statistics().to_owned();
        let mut parts = BitPackedV2::into_parts(packed);
        parts.patches = parts
            .patches
            .take()
            .map(|p| compress_patches_v2(compressor, p, &compress_ctx, self.id(), exec_ctx))
            .transpose()?;

        Ok(BitPackedV2::try_new(
            parts.packed,
            parts.bit_widths,
            ptype,
            parts.validity,
            parts.patches,
            parts.len,
            parts.offset,
        )?
        .with_stats_set(packed_stats)
        .into_array())
    }
}
