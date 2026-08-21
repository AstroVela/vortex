// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zstd string compression without dictionaries (nvCOMP compatible).

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
use crate::schemes::DEFAULT_ZSTD_LEVEL;

/// Zstd compression without dictionaries (nvCOMP compatible).
///
/// `LEVEL` is the zstd compression level the frames are written at. It is a parameter of the type
/// rather than a field because the compressor holds `&'static dyn Scheme`, so a scheme configured
/// at runtime could not be registered. Zstd frames carry their own parameters, so a file written
/// at any level is read back by the same decoder.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ZstdScheme<const LEVEL: i32 = DEFAULT_ZSTD_LEVEL>;

impl<const LEVEL: i32> ZstdScheme<LEVEL> {
    /// A `'static` handle to this scheme, for [`with_new_scheme`].
    ///
    /// [`with_new_scheme`]: crate::BtrBlocksCompressorBuilder::with_new_scheme
    pub const INSTANCE: &'static Self = &Self;
}

impl<const LEVEL: i32> Scheme for ZstdScheme<LEVEL> {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.zstd"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![vortex_zstd::Zstd.id()]
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
        Ok(
            vortex_zstd::Zstd::from_var_bin_view_without_dict(&compacted, LEVEL, 8192, exec_ctx)?
                .into_array(),
        )
    }
}
