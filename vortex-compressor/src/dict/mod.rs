// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in dictionary compression.
//!
//! Dictionary encoding is not a pluggable [`Scheme`]: the compressor always considers it for
//! integer, float, string, and binary leaves, unless a type class is disabled via [`DictTypes`].
//! In cascade history and exclusion rules it is identified by the single well-known
//! [`DICT_SCHEME_ID`], regardless of the value type being encoded.

mod float;
mod integer;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::builders::dict::dict_encode;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::estimate::DeferredEstimate;
use crate::estimate::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

/// Well-known [`SchemeId`] for the compressor's built-in dictionary compression.
///
/// This ID appears in the cascade history when the compressor dictionary-encodes an array, so
/// external schemes can declare exclusion rules against it (e.g. "skip me on dict codes"). Its
/// children are `values = 0` and `codes = 1` (see [`dict_children`]).
pub const DICT_SCHEME_ID: SchemeId = SchemeId {
    name: "vortex.dict",
};

/// Child indices for the compressor's built-in dictionary compression.
pub mod dict_children {
    /// The deduplicated values child.
    pub const VALUES: usize = 0;
    /// The codes child (compact unsigned integers indexing into values).
    pub const CODES: usize = 1;
}

/// Per-type-class toggles for the compressor's built-in dictionary compression.
///
/// All type classes are enabled by default. Disabling a class prevents the compressor from
/// considering dictionary encoding for leaves of that class, e.g. for decode environments that
/// cannot expand string dictionaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DictTypes {
    /// Consider dictionary encoding for integer arrays.
    pub integer: bool,
    /// Consider dictionary encoding for float arrays.
    pub float: bool,
    /// Consider dictionary encoding for string arrays.
    pub string: bool,
    /// Consider dictionary encoding for binary arrays.
    pub binary: bool,
}

impl Default for DictTypes {
    fn default() -> Self {
        Self::all()
    }
}

impl DictTypes {
    /// Enables dictionary compression for all type classes.
    pub const fn all() -> Self {
        Self {
            integer: true,
            float: true,
            string: true,
            binary: true,
        }
    }

    /// Disables dictionary compression for all type classes.
    pub const fn none() -> Self {
        Self {
            integer: false,
            float: false,
            string: false,
            binary: false,
        }
    }

    /// Whether dictionary compression is enabled for the given canonical array.
    pub(crate) fn enabled_for(&self, canonical: &Canonical) -> bool {
        let dtype = canonical.dtype();
        (self.integer && dtype.is_int())
            || (self.float && dtype.is_float())
            || (self.string && dtype.is_utf8())
            || (self.binary && dtype.is_binary())
    }
}

/// The compressor's built-in dictionary compression.
///
/// This implements [`Scheme`] so that it can ride the existing estimation and selection
/// machinery, but it is not registrable: the compressor injects it as a candidate itself. All
/// value types share the single [`DICT_SCHEME_ID`] identity, so "no dict anywhere under dict"
/// follows from ordinary self-exclusion.
#[derive(Debug)]
pub(crate) struct BuiltinDictScheme;

/// The singleton instance the compressor injects into scheme selection.
pub(crate) static DICT_SCHEME: BuiltinDictScheme = BuiltinDictScheme;

impl Scheme for BuiltinDictScheme {
    fn scheme_name(&self) -> &'static str {
        DICT_SCHEME_ID.name
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        let dtype = canonical.dtype();
        dtype.is_int() || dtype.is_float() || dtype.is_utf8() || dtype.is_binary()
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        GenerateStatsOptions {
            count_distinct_values: true,
        }
    }

    /// Children: values=0, codes=1 (see [`dict_children`]).
    fn num_children(&self) -> usize {
        2
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let dtype = data.array().dtype();

        if dtype.is_int() {
            return integer::estimate(data, exec_ctx);
        }

        if dtype.is_float() {
            return float::estimate(data, exec_ctx);
        }

        varbinview_estimate(data, exec_ctx)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let dtype = data.array().dtype();

        let dict = if dtype.is_int() {
            let stats = data.integer_stats(exec_ctx);
            integer::dictionary_encode(data.array_as_primitive(), &stats)?
        } else if dtype.is_float() {
            let stats = data.float_stats(exec_ctx);
            float::dictionary_encode(data.array_as_primitive(), &stats)?
        } else {
            dict_encode(data.array(), exec_ctx)?
        };

        compress_dict_children(compressor, dict, &compress_ctx, exec_ctx)
    }
}

/// Estimates dictionary compression for string and binary arrays.
fn varbinview_estimate(data: &ArrayAndStats, exec_ctx: &mut ExecutionCtx) -> CompressionEstimate {
    let stats = data.varbinview_stats(exec_ctx);

    if stats.value_count() == 0 {
        return CompressionEstimate::Verdict(EstimateVerdict::Skip);
    }

    // The estimated distinct count is a lower bound, so it can only prove non-viability.
    // If > 50% of the values are distinct, skip dictionary compression.
    if stats
        .estimated_distinct_count()
        .is_some_and(|c| c > stats.value_count() / 2)
    {
        return CompressionEstimate::Verdict(EstimateVerdict::Skip);
    }

    // Let sampling determine the expected ratio.
    CompressionEstimate::Deferred(DeferredEstimate::Sample)
}

/// Recursively compresses a freshly-encoded [`DictArray`]'s values and codes children and
/// reassembles the result.
fn compress_dict_children(
    compressor: &CascadingCompressor,
    dict: DictArray,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let has_all_values_referenced = dict.has_all_values_referenced();

    let compressed_values = compressor.compress_child(
        dict.values(),
        compress_ctx,
        DICT_SCHEME_ID,
        dict_children::VALUES,
        exec_ctx,
    )?;

    let narrowed_codes = dict
        .codes()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?
        .narrow(exec_ctx)?
        .into_array();
    let compressed_codes = compressor.compress_child(
        &narrowed_codes,
        compress_ctx,
        DICT_SCHEME_ID,
        dict_children::CODES,
        exec_ctx,
    )?;

    // SAFETY: compressing codes or values does not alter the dict invariants.
    unsafe {
        Ok(
            DictArray::new_unchecked(compressed_codes, compressed_values)
                .set_all_values_referenced(has_all_values_referenced)
                .into_array(),
        )
    }
}
