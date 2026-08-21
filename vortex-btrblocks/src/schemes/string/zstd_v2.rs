// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zstd string compression storing value lengths apart from value bytes.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

/// Zstd compression for string arrays, with a separate lengths stream.
///
/// Unlike [`super::ZstdScheme`], the frames hold only value bytes, so a reader learns where every
/// value lives from the lengths alone. That makes a slice or filter decompress only the frames
/// holding the values it asks for, which is what the framing here is sized for.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ZstdV2Scheme {
    level: i32,
    values_per_frame: usize,
}

impl ZstdV2Scheme {
    /// Compression level used by [`ZstdV2Scheme::DEFAULT`].
    pub const DEFAULT_LEVEL: i32 = 6;

    /// Values per frame used by [`ZstdV2Scheme::DEFAULT`].
    ///
    /// Frames are the unit a read can skip, so unlike `vortex.zstd` — where a smaller frame only
    /// costs ratio — there is something to buy here by cutting the array into several.
    pub const DEFAULT_VALUES_PER_FRAME: usize = 8192;

    /// The configuration registered by `BtrBlocksCompressorBuilder::with_zstd_v2`.
    pub const DEFAULT: Self = Self::new(Self::DEFAULT_LEVEL, Self::DEFAULT_VALUES_PER_FRAME);

    /// A scheme compressing at `level`, cutting a frame every `values_per_frame` values.
    pub const fn new(level: i32, values_per_frame: usize) -> Self {
        Self {
            level,
            values_per_frame,
        }
    }
}

impl Scheme for ZstdV2Scheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.zstd.v2"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![vortex_zstd_v2::ZstdV2.id()]
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
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let compacted = data
            .array_as_varbinview()
            .into_owned()
            .compact_buffers(exec_ctx)?;
        Ok(vortex_zstd_v2::ZstdV2::from_var_bin_view(
            &compacted,
            self.level,
            self.values_per_frame,
            exec_ctx,
        )?
        .into_array())
    }
}
