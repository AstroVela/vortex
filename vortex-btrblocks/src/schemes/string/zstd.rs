// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zstd string compression.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_error::VortexResult;
use vortex_zstd::ZstdOptions;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

/// Zstd compression for string arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ZstdScheme {
    options: ZstdOptions,
}

impl ZstdScheme {
    /// Compression level used by [`ZstdScheme::DEFAULT`].
    ///
    /// Level 6 buys 15-20% over level 3 on string columns of the size the file writer produces,
    /// and decompression gets faster because there are fewer bytes to parse. The cost is write
    /// throughput, which is why it is a `with_compact` scheme rather than a default one.
    pub const DEFAULT_LEVEL: i32 = 6;

    /// The configuration registered by `BtrBlocksCompressorBuilder::with_compact`.
    ///
    /// One frame per array: the write strategy already hands the compressor row blocks of around
    /// a megabyte, so splitting them further loses redundancy without making reads any narrower.
    pub const DEFAULT: Self = Self::new(ZstdOptions::new(Self::DEFAULT_LEVEL));

    /// A scheme compressing with `options`.
    pub const fn new(options: ZstdOptions) -> Self {
        Self { options }
    }
}

impl Scheme for ZstdScheme {
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
            vortex_zstd::Zstd::from_var_bin_view_with_options(&compacted, self.options, exec_ctx)?
                .into_array(),
        )
    }
}
