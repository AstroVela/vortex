// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use vortex_compressor::DictTypes;
use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::CascadingCompressor;
use crate::Scheme;
use crate::SchemeExt;
use crate::SchemeId;
#[cfg(feature = "zstd")]
use crate::schemes::binary;
use crate::schemes::decimal;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::string;
use crate::schemes::temporal;

/// All available compression schemes.
///
/// Constant and dictionary compression are built into the compressor itself and do not appear in
/// this list (see [`DictTypes`] for disabling dictionary compression per type class).
///
/// This list is order-sensitive: the builder preserves this order when constructing
/// the final scheme list, so that tie-breaking is deterministic.
pub const ALL_SCHEMES: &[&dyn Scheme] = &[
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Integer schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // NOTE: FoR must precede BitPacking to avoid unnecessary patches.
    &integer::FoRScheme,
    // NOTE: ZigZag should precede BitPacking because we don't want negative numbers.
    &integer::ZigZagScheme,
    &integer::BitPackingScheme,
    &integer::SparseScheme,
    &integer::RunEndScheme,
    &integer::SequenceScheme,
    &integer::IntRLEScheme,
    // Prefer all other schemes above delta, for now (since its slower to decompress).
    #[cfg(feature = "unstable_encodings")]
    &integer::DeltaScheme::new(1.25),
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Float schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &float::ALPScheme,
    &float::ALPRDScheme,
    &float::NullDominatedSparseScheme,
    &float::FloatRLEScheme,
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // String schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Both string-fragmentation schemes are registered; the sample-based
    // selector keeps whichever is smaller per column.
    &string::FSSTScheme,
    #[cfg(feature = "unstable_encodings")]
    &string::OnPairScheme,
    &string::NullDominatedSparseScheme,
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Decimal schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &decimal::DecimalScheme,
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Temporal schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &temporal::TemporalScheme,
];

/// Builder for creating configured [`BtrBlocksCompressor`] instances.
///
/// By default, all schemes in [`ALL_SCHEMES`] are enabled in a deterministic order, and the
/// compressor's built-in dictionary compression is enabled for every type class. Feature-gated
/// schemes (Pco, Zstd) are not in `ALL_SCHEMES` and must be added explicitly via
/// [`with_new_scheme`](BtrBlocksCompressorBuilder::with_new_scheme) or `with_compact` when the
/// `zstd` feature is enabled.
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressorBuilder, DictTypes, Scheme, SchemeExt};
/// use vortex_btrblocks::schemes::integer::FoRScheme;
///
/// // Default compressor with all schemes in ALL_SCHEMES.
/// let compressor = BtrBlocksCompressorBuilder::default().build();
///
/// // Remove specific schemes.
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .exclude_schemes([FoRScheme.id()])
///     .build();
///
/// // Disable the built-in dictionary compression for strings and binary.
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .with_dict_types(DictTypes {
///         string: false,
///         binary: false,
///         ..DictTypes::all()
///     })
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BtrBlocksCompressorBuilder {
    schemes: Vec<&'static dyn Scheme>,
    dict_types: DictTypes,
}

impl Default for BtrBlocksCompressorBuilder {
    fn default() -> Self {
        Self {
            schemes: ALL_SCHEMES.to_vec(),
            dict_types: DictTypes::default(),
        }
    }
}

impl BtrBlocksCompressorBuilder {
    /// Creates a builder with no schemes registered.
    ///
    /// Useful when the caller wants explicit, scheme-by-scheme control over the compressor. Note
    /// that the compressor's built-in constant and dictionary compression still apply; use
    /// [`with_dict_types`](Self::with_dict_types) to disable dictionary compression.
    pub fn empty() -> Self {
        Self {
            schemes: Vec::new(),
            dict_types: DictTypes::default(),
        }
    }

    /// Adds an external compression scheme not in [`ALL_SCHEMES`].
    ///
    /// This allows encoding crates outside of `vortex-btrblocks` to register their own schemes
    /// with the compressor.
    ///
    /// # Panics
    ///
    /// Panics if a scheme with the same [`SchemeId`] is already present.
    pub fn with_new_scheme(mut self, scheme: &'static dyn Scheme) -> Self {
        assert!(
            !self.schemes.iter().any(|s| s.id() == scheme.id()),
            "scheme {:?} is already present in the builder",
            scheme.id(),
        );

        self.schemes.push(scheme);
        self
    }

    /// Configures which type classes the compressor's built-in dictionary compression applies
    /// to. All type classes are enabled by default.
    pub fn with_dict_types(mut self, dict_types: DictTypes) -> Self {
        self.dict_types = dict_types;
        self
    }

    /// Returns the currently configured dictionary compression type classes.
    pub fn dict_types(&self) -> DictTypes {
        self.dict_types
    }

    /// Adds compact encoding schemes (Zstd for strings and binary, Pco for numerics).
    ///
    /// This provides better compression ratios than the default, especially for floating-point
    /// heavy datasets. Requires the `zstd` feature. When the `pco` feature is also enabled,
    /// Pco schemes for integers and floats are included.
    ///
    /// # Panics
    ///
    /// Panics if any of the compact schemes are already present.
    #[cfg(feature = "zstd")]
    pub fn with_compact(self) -> Self {
        let builder = self
            .with_new_scheme(&string::ZstdScheme)
            .with_new_scheme(&binary::ZstdScheme);

        #[cfg(feature = "pco")]
        let builder = builder
            .with_new_scheme(&integer::PcoScheme)
            .with_new_scheme(&float::PcoScheme);

        builder
    }

    /// Excludes schemes without CUDA kernel support and adds Zstd for string and binary compression.
    ///
    /// With the `unstable_encodings` feature, buffer-level Zstd compression is used which
    /// preserves the array buffer layout for zero-conversion GPU decompression. Without it,
    /// interleaved Zstd compression is used.
    ///
    /// This preset is intended for files that will be decoded by CUDA kernels. It may choose a
    /// larger encoded representation than the default compressor.
    pub fn only_cuda_compatible(self) -> Self {
        // String fragmentation schemes (OnPair, FSST) require host-side
        // dictionary expansion at decode time, which is incompatible with
        // pure-GPU decompression paths. Strip whichever string-fragment
        // scheme is enabled by feature.
        #[cfg_attr(not(feature = "unstable_encodings"), allow(unused_mut))]
        let mut excluded: Vec<SchemeId> = vec![
            integer::SparseScheme.id(),
            integer::IntRLEScheme.id(),
            float::FloatRLEScheme.id(),
            float::NullDominatedSparseScheme.id(),
            string::FSSTScheme.id(),
        ];
        #[cfg(feature = "unstable_encodings")]
        excluded.push(string::OnPairScheme.id());
        // Delta has no GPU decode kernel and its prefix-sum decode is inherently sequential, so it
        // is incompatible with pure-GPU decompression paths.
        #[cfg(feature = "unstable_encodings")]
        excluded.push(integer::DeltaScheme::default().id());
        // String and binary dictionaries require host-side expansion at decode time, so disable
        // the built-in dictionary compression for those type classes.
        let builder = self.exclude_schemes(excluded).with_dict_types(DictTypes {
            string: false,
            binary: false,
            ..DictTypes::all()
        });

        #[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
        let builder = builder
            .with_new_scheme(&string::ZstdBuffersScheme)
            .with_new_scheme(&binary::ZstdBuffersScheme);
        #[cfg(all(feature = "zstd", not(feature = "unstable_encodings")))]
        let builder = builder
            .with_new_scheme(&string::ZstdScheme)
            .with_new_scheme(&binary::ZstdScheme);

        builder
    }

    /// Removes the specified compression schemes by their [`SchemeId`].
    pub fn exclude_schemes(mut self, ids: impl IntoIterator<Item = SchemeId>) -> Self {
        let ids: HashSet<_> = ids.into_iter().collect();
        self.schemes.retain(|s| !ids.contains(&s.id()));
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`].
    pub fn build(self) -> BtrBlocksCompressor {
        BtrBlocksCompressor(CascadingCompressor::new(self.schemes).with_dict_types(self.dict_types))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_starts_with_no_schemes() {
        let builder = BtrBlocksCompressorBuilder::empty();
        assert!(builder.schemes.is_empty());
    }

    #[test]
    fn default_includes_all_schemes() {
        let builder = BtrBlocksCompressorBuilder::default();
        assert_eq!(builder.schemes.len(), ALL_SCHEMES.len());
    }
}
