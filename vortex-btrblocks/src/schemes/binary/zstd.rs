// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zstd compression for binary arrays.

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

/// Zstd compression without dictionaries for binary arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ZstdScheme {
    level: i32,
}

impl ZstdScheme {
    /// Creates a scheme that compresses with the given zstd level.
    pub const fn new(level: i32) -> Self {
        Self { level }
    }
}

impl Default for ZstdScheme {
    fn default() -> Self {
        Self::new(DEFAULT_ZSTD_LEVEL)
    }
}

impl Scheme for ZstdScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.binary.zstd"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_binary()
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
        Ok(vortex_zstd::Zstd::from_var_bin_view_without_dict(
            &compacted, self.level, 8192, exec_ctx,
        )?
        .into_array())
    }
}
