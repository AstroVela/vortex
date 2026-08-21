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
use vortex_zstd::DictionaryMode;
use vortex_zstd::ZstdOptions;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

/// Zstd compression without dictionaries for binary arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ZstdScheme {
    options: ZstdOptions,
}

impl ZstdScheme {
    /// Compression level used by [`ZstdScheme::DEFAULT`].
    pub const DEFAULT_LEVEL: i32 = 6;

    /// Values per frame used by [`ZstdScheme::DEFAULT`].
    ///
    /// Unlike the string scheme, binary arrays are decoded on the GPU, where nvCOMP parallelizes
    /// across frames. One frame per array would leave that batch with a single item.
    pub const DEFAULT_VALUES_PER_FRAME: usize = 8192;

    /// The configuration registered by `BtrBlocksCompressorBuilder::with_compact` and
    /// `only_cuda_compatible`.
    ///
    /// Dictionaries stay off: nvCOMP decodes binary arrays on the GPU and does not support them.
    pub const DEFAULT: Self = Self::new(
        ZstdOptions::new(Self::DEFAULT_LEVEL)
            .with_values_per_frame(Self::DEFAULT_VALUES_PER_FRAME)
            .with_dictionary(DictionaryMode::Never),
    );

    /// A scheme compressing with `options`.
    pub const fn new(options: ZstdOptions) -> Self {
        Self { options }
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
        Ok(
            vortex_zstd::Zstd::from_var_bin_view_with_options(&compacted, self.options, exec_ctx)?
                .into_array(),
        )
    }
}
