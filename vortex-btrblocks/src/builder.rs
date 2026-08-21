// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

#[cfg(feature = "zstd")]
use std::any::Any;
#[cfg(feature = "zstd")]
use std::any::TypeId;
#[cfg(feature = "zstd")]
use std::sync::LazyLock;

#[cfg(feature = "zstd")]
use parking_lot::Mutex;
use vortex_array::ArrayId;
#[cfg(feature = "zstd")]
use vortex_error::VortexExpect;
#[cfg(feature = "zstd")]
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::CascadingCompressor;
use crate::Scheme;
use crate::SchemeExt;
use crate::SchemeId;
#[cfg(feature = "zstd")]
use crate::schemes::DEFAULT_ZSTD_LEVEL;
use crate::schemes::binary;
use crate::schemes::decimal;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::string;
use crate::schemes::temporal;

/// All available compression schemes.
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
    &integer::IntDictScheme,
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
    &float::FloatDictScheme,
    &float::NullDominatedSparseScheme,
    &float::FloatRLEScheme,
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // String schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &string::StringDictScheme,
    // Both string-fragmentation schemes are registered; the sample-based
    // selector keeps whichever is smaller per column.
    &string::FSSTScheme,
    #[cfg(feature = "unstable_encodings")]
    &string::OnPairScheme,
    &string::NullDominatedSparseScheme,
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Binary schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &binary::BinaryDictScheme,
    // Decimal schemes.
    &decimal::DecimalScheme,
    // Temporal schemes.
    &temporal::TemporalScheme,
];

/// Builder for creating configured [`BtrBlocksCompressor`] instances.
///
/// By default, all schemes in [`ALL_SCHEMES`] are enabled in a deterministic order. Feature-gated
/// schemes (Pco, Zstd) are not in `ALL_SCHEMES` and must be added explicitly via
/// [`with_new_scheme`](BtrBlocksCompressorBuilder::with_new_scheme) or `with_compact` when the
/// `zstd` feature is enabled.
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressorBuilder, Scheme, SchemeExt};
/// use vortex_btrblocks::schemes::integer::IntDictScheme;
///
/// // Default compressor with all schemes in ALL_SCHEMES.
/// let compressor = BtrBlocksCompressorBuilder::default().build();
///
/// // Remove specific schemes.
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .exclude_schemes([IntDictScheme.id()])
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BtrBlocksCompressorBuilder {
    schemes: Vec<&'static dyn Scheme>,

    /// Zstd compression level handed to the zstd schemes registered by
    /// [`with_compact`](BtrBlocksCompressorBuilder::with_compact) and
    /// [`only_cuda_compatible`](BtrBlocksCompressorBuilder::only_cuda_compatible).
    #[cfg(feature = "zstd")]
    zstd_level: i32,
}

impl Default for BtrBlocksCompressorBuilder {
    fn default() -> Self {
        Self {
            schemes: ALL_SCHEMES.to_vec(),
            #[cfg(feature = "zstd")]
            zstd_level: DEFAULT_ZSTD_LEVEL,
        }
    }
}

impl BtrBlocksCompressorBuilder {
    /// Creates a builder with no schemes registered.
    ///
    /// Useful when the caller wants explicit, scheme-by-scheme control over the compressor.
    pub fn empty() -> Self {
        Self {
            schemes: Vec::new(),
            #[cfg(feature = "zstd")]
            zstd_level: DEFAULT_ZSTD_LEVEL,
        }
    }

    /// Sets the zstd compression level used by the zstd schemes, defaulting to
    /// [`DEFAULT_ZSTD_LEVEL`].
    ///
    /// Higher levels compress harder and more slowly; negative levels are zstd's fast modes. Only
    /// zstd schemes registered after this call observe the new level, so set it before
    /// [`with_compact`](Self::with_compact) or [`only_cuda_compatible`](Self::only_cuda_compatible).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vortex_btrblocks::BtrBlocksCompressorBuilder;
    ///
    /// // Compress zstd-eligible columns harder than the default level.
    /// let compressor = BtrBlocksCompressorBuilder::default()
    ///     .with_zstd_level(9)
    ///     .with_compact()
    ///     .build();
    /// ```
    #[cfg(feature = "zstd")]
    pub fn with_zstd_level(mut self, level: i32) -> Self {
        self.zstd_level = level;
        self
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
        let level = self.zstd_level;
        let builder = self
            .with_new_scheme(interned_scheme::<string::ZstdScheme>(
                level,
                string::ZstdScheme::new,
            ))
            .with_new_scheme(interned_scheme::<binary::ZstdScheme>(
                level,
                binary::ZstdScheme::new,
            ));

        #[cfg(feature = "pco")]
        let builder = builder
            .with_new_scheme(&integer::PcoScheme)
            .with_new_scheme(&float::PcoScheme);

        builder
    }

    /// Excludes schemes without CUDA kernel support, keeps FSST for string compression,
    /// and adds Zstd for binary compression.
    ///
    /// With the `unstable_encodings` feature, buffer-level Zstd compression is used for binary
    /// arrays, preserving their buffer layout for zero-conversion GPU decompression. Without it,
    /// interleaved binary Zstd compression is used.
    ///
    /// This preset is intended for files that will be decoded by CUDA kernels. It may choose a
    /// larger encoded representation than the default compressor.
    pub fn only_cuda_compatible(self) -> Self {
        // Keep FSST, which has a CUDA decoder and direct Arrow offset-based export. Other
        // string fragmentation and dictionary schemes still require unsupported decode paths.
        #[cfg_attr(
            not(any(feature = "pco", feature = "unstable_encodings")),
            allow(unused_mut)
        )]
        let mut excluded: Vec<SchemeId> = vec![
            integer::SparseScheme.id(),
            integer::IntRLEScheme.id(),
            float::ALPRDScheme.id(),
            float::FloatRLEScheme.id(),
            float::NullDominatedSparseScheme.id(),
            string::StringDictScheme.id(),
            binary::BinaryDictScheme.id(),
        ];
        // Delta has no GPU decode kernel and its prefix-sum decode is inherently sequential, so it
        // is incompatible with pure-GPU decompression paths.
        #[cfg(feature = "unstable_encodings")]
        excluded.push(integer::DeltaScheme::default().id());
        #[cfg(feature = "pco")]
        excluded.extend([integer::PcoScheme.id(), float::PcoScheme.id()]);
        let builder = self.exclude_schemes(excluded);

        #[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
        let builder = {
            let level = builder.zstd_level;
            builder.with_new_scheme(interned_scheme::<binary::ZstdBuffersScheme>(
                level,
                binary::ZstdBuffersScheme::new,
            ))
        };
        #[cfg(all(feature = "zstd", not(feature = "unstable_encodings")))]
        let builder = {
            let level = builder.zstd_level;
            builder.with_new_scheme(interned_scheme::<binary::ZstdScheme>(
                level,
                binary::ZstdScheme::new,
            ))
        };

        builder
    }

    /// Removes the specified compression schemes by their [`SchemeId`].
    pub fn exclude_schemes(mut self, ids: impl IntoIterator<Item = SchemeId>) -> Self {
        let ids: HashSet<_> = ids.into_iter().collect();
        self.schemes.retain(|s| !ids.contains(&s.id()));
        self
    }

    /// Retains only schemes whose produced encodings all belong to `allowed`.
    ///
    /// The file writer uses this to restrict compression to the encodings of its configured
    /// editions.
    pub fn retain_allowed_encodings(mut self, allowed: &HashSet<ArrayId>) -> Self {
        self.schemes
            .retain(|s| s.produced_encodings().iter().all(|id| allowed.contains(id)));
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`].
    pub fn build(self) -> BtrBlocksCompressor {
        BtrBlocksCompressor(CascadingCompressor::new(self.schemes))
    }
}

/// Returns a `'static` scheme value for a runtime-chosen zstd level.
///
/// Schemes are registered as `&'static dyn Scheme`, so a level picked at runtime cannot be held in
/// a scheme value on the stack. Each distinct (scheme type, level) pair is leaked once and reused
/// afterwards, so repeatedly building compressors does not leak repeatedly.
#[cfg(feature = "zstd")]
fn interned_scheme<S: Scheme + Any>(level: i32, make: fn(i32) -> S) -> &'static S {
    /// Leaked schemes, keyed by the scheme type they were built from and their zstd level.
    type SchemeCache = Mutex<HashMap<(TypeId, i32), &'static (dyn Any + Send + Sync)>>;
    static CACHE: LazyLock<SchemeCache> = LazyLock::new(|| Mutex::new(HashMap::new()));

    let mut cache = CACHE.lock();
    let scheme = *cache
        .entry((TypeId::of::<S>(), level))
        .or_insert_with(|| Box::leak(Box::new(make(level))) as &'static (dyn Any + Send + Sync));
    scheme
        .downcast_ref::<S>()
        .vortex_expect("interned scheme has the type it was cached under")
}

#[cfg(test)]
mod tests {
    use vortex_array::VTable;
    use vortex_fastlanes::FoR;

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

    #[test]
    fn retain_allowed_encodings_filters_schemes() {
        let allowed: HashSet<ArrayId> = [FoR.id()].into_iter().collect();
        let builder = BtrBlocksCompressorBuilder::default().retain_allowed_encodings(&allowed);
        assert_eq!(builder.schemes.len(), 1);
        assert_eq!(builder.schemes[0].id(), integer::FoRScheme.id());

        let none = BtrBlocksCompressorBuilder::default().retain_allowed_encodings(&HashSet::new());
        assert!(none.schemes.is_empty());
    }

    #[test]
    fn retaining_all_declared_outputs_keeps_every_scheme() {
        let allowed: HashSet<ArrayId> = ALL_SCHEMES
            .iter()
            .flat_map(|scheme| scheme.produced_encodings())
            .collect();
        let builder = BtrBlocksCompressorBuilder::default().retain_allowed_encodings(&allowed);
        assert_eq!(builder.schemes.len(), ALL_SCHEMES.len());
    }

    #[test]
    fn cuda_compatible_excludes_alprd() {
        let builder = BtrBlocksCompressorBuilder::default().only_cuda_compatible();
        assert!(
            !builder
                .schemes
                .iter()
                .any(|s| s.id() == float::ALPRDScheme.id())
        );
    }

    #[test]
    fn cuda_compatible_uses_fsst_for_strings() {
        let builder = BtrBlocksCompressorBuilder::default().only_cuda_compatible();
        assert!(
            builder
                .schemes
                .iter()
                .any(|scheme| scheme.id() == string::FSSTScheme.id())
        );
        #[cfg(feature = "zstd")]
        assert!(
            !builder
                .schemes
                .iter()
                .any(|scheme| scheme.id() == string::ZstdScheme::default().id())
        );
    }

    #[test]
    #[cfg(feature = "pco")]
    fn cuda_compatible_excludes_pco() {
        let builder = BtrBlocksCompressorBuilder::default()
            .with_new_scheme(&integer::PcoScheme)
            .with_new_scheme(&float::PcoScheme)
            .only_cuda_compatible();
        for scheme in [integer::PcoScheme.id(), float::PcoScheme.id()] {
            assert!(!builder.schemes.iter().any(|s| s.id() == scheme));
        }
    }
}
