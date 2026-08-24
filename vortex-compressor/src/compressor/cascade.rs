// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core cascading compression flow.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::CanonicalValidity;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::Masked;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::Variant;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::builders::VarBinBuilder;
use vortex_array::dtype::OffsetBuilderPType;
use vortex_array::ArrayView;
use vortex_mask::Mask;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::arrays::masked::MaskedArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::union::UnionArrayExt;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use super::CascadingCompressor;
use super::constant;
use crate::scheme::CompressorContext;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;
use crate::trace;

/// The array to store when no scheme's output is worth keeping.
///
/// Canonical string data is `VarBinView`, which spends 16 bytes of view per value so filters and
/// takes can address values without walking offsets. That is the right in-memory layout and the
/// wrong on-disk one: `VarBinArray` describes the same values with a single offset each and
/// decodes for the price of an offsets scan rather than any per-value work. Declining a codec
/// should not mean writing the widest layout available, so demote views to offsets on the way out.
fn store_uncompressed(array: ArrayRef, exec_ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    if !(array.dtype().is_utf8() || array.dtype().is_binary()) {
        return Ok(array);
    }
    let Some(view) = array.as_opt::<VarBinView>() else {
        return Ok(array);
    };

    // Materialize validity once: a per-element `is_valid` inside the loop would re-resolve the
    // validity array for every value.
    let validity = view.validity()?.execute_mask(view.len(), exec_ctx)?;

    // Narrow offsets whenever the data buffer fits, so the layout that replaces 16-byte views
    // does not spend 8 bytes per value describing a chunk addressable in 4.
    let total_bytes: u64 = view.views().iter().map(|v| u64::from(v.len())).sum();
    let varbin = if total_bytes <= i32::MAX as u64 {
        build_varbin::<i32>(&view, &validity)
    } else {
        build_varbin::<i64>(&view, &validity)
    };

    // Only worth it when offsets really are cheaper than views, which is false for arrays of
    // very short values where the data buffer dominates.
    Ok(if varbin.nbytes() < array.nbytes() {
        varbin
    } else {
        array
    })
}

/// Rebuild a view array as offset-addressed `VarBin` with `O`-typed offsets.
fn build_varbin<O>(view: &ArrayView<'_, VarBinView>, validity: &Mask) -> ArrayRef
where
    O: OffsetBuilderPType,
    usize: num_traits::AsPrimitive<O>,
{
    let mut builder = VarBinBuilder::<O>::with_capacity(view.dtype().clone(), view.len());
    if validity.all_true() {
        for idx in 0..view.len() {
            builder.append_value(view.bytes_at(idx).as_slice());
        }
    } else {
        for idx in 0..view.len() {
            if validity.value(idx) {
                builder.append_value(view.bytes_at(idx).as_slice());
            } else {
                builder.push_null();
            }
        }
    }
    builder.finish_into_varbin().into_array()
}

impl CascadingCompressor {
    /// Compresses an array using cascading adaptive compression.
    ///
    /// First canonicalizes and compacts the array, then applies optimal compression schemes.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn compress(
        &self,
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let before_nbytes = array.nbytes();
        let span = trace::compress_span(array.len(), array.dtype(), before_nbytes);
        let _enter = span.enter();

        let canonical = array.clone().execute::<CanonicalValidity>(exec_ctx)?.0;
        let compact = canonical.compact(exec_ctx)?;
        let compressed = self.compress_canonical(compact, CompressorContext::new(), exec_ctx)?;

        trace::record_compress_outcome(&span, before_nbytes, compressed.nbytes());

        Ok(compressed)
    }

    /// Compresses a child array produced by a cascading scheme.
    ///
    /// If the cascade budget is exhausted, the canonical array is returned as-is. Otherwise, the
    /// child context is created by descending and recording the parent scheme + child index, and
    /// compression proceeds normally.
    ///
    /// # Errors
    ///
    /// Returns an error if compression fails.
    pub fn compress_child(
        &self,
        child: &ArrayRef,
        parent_ctx: &CompressorContext,
        parent_id: SchemeId,
        child_index: usize,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if parent_ctx.finished_cascading() {
            trace::cascade_exhausted(parent_id, child_index);
            return Ok(child.clone());
        }

        let canonical = child.clone().execute::<CanonicalValidity>(exec_ctx)?.0;
        let compact = canonical.compact(exec_ctx)?;

        let child_ctx = parent_ctx
            .clone()
            .descend_with_scheme(parent_id, child_index);
        self.compress_canonical(compact, child_ctx, exec_ctx)
    }

    /// Compresses a canonical array by dispatching to type-specific logic.
    ///
    /// # Errors
    ///
    /// Returns an error if compression of any sub-array fails.
    pub(super) fn compress_canonical(
        &self,
        array: Canonical,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        match array {
            Canonical::Null(null_array) => Ok(null_array.into_array()),
            Canonical::Bool(bool_array) => {
                self.choose_and_compress(Canonical::Bool(bool_array), compress_ctx, exec_ctx)
            }
            Canonical::Primitive(primitive) => {
                self.choose_and_compress(Canonical::Primitive(primitive), compress_ctx, exec_ctx)
            }
            Canonical::Decimal(decimal) => {
                self.choose_and_compress(Canonical::Decimal(decimal), compress_ctx, exec_ctx)
            }
            Canonical::Struct(struct_array) => {
                let fields = struct_array
                    .iter_unmasked_fields()
                    .map(|field| self.compress(field, exec_ctx))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(StructArray::try_new(
                    struct_array.names().clone(),
                    fields,
                    struct_array.len(),
                    struct_array.validity()?,
                )?
                .into_array())
            }
            Canonical::Union(union_array) => {
                let type_ids = self.compress(union_array.type_ids(), exec_ctx)?;
                let children = union_array
                    .iter_children()
                    .map(|child| self.compress(child, exec_ctx))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(
                    UnionArray::try_new(type_ids, union_array.variants().clone(), children)?
                        .into_array(),
                )
            }
            Canonical::List(list_view_array) => {
                if list_view_array.is_zero_copy_to_list() || list_view_array.elements().is_empty() {
                    let list_array = list_from_list_view(list_view_array, exec_ctx)?;
                    self.compress_list_array(list_array, compress_ctx, exec_ctx)
                } else {
                    self.compress_list_view_array(list_view_array, compress_ctx, exec_ctx)
                }
            }
            Canonical::Map(map_array) => self.compress_map_array(map_array, compress_ctx, exec_ctx),
            Canonical::FixedSizeList(fsl_array) => {
                let compressed_elems = self.compress(fsl_array.elements(), exec_ctx)?;

                Ok(FixedSizeListArray::try_new(
                    compressed_elems,
                    fsl_array.list_size(),
                    fsl_array.validity()?,
                    fsl_array.len(),
                )?
                .into_array())
            }
            Canonical::VarBinView(varbinview) => {
                self.choose_and_compress(Canonical::VarBinView(varbinview), compress_ctx, exec_ctx)
            }
            Canonical::Extension(ext_array) => {
                // Try scheme-based compression first.
                let scheme_compressed = self.choose_and_compress(
                    Canonical::Extension(ext_array.clone()),
                    compress_ctx,
                    exec_ctx,
                )?;

                // A constant extension array (that might be masked) is already in its terminal
                // representation, and compressing the storage separately cannot do better.
                if scheme_compressed.is::<Constant>() {
                    return Ok(scheme_compressed);
                }
                if let Some(masked) = scheme_compressed.as_opt::<Masked>()
                    && masked.child().is::<Constant>()
                {
                    return Ok(scheme_compressed);
                }

                // Also compress the underlying storage array. Some extension schemes can beat the
                // extension storage but still lose to ordinary storage compression.
                let compressed_storage = self.compress(ext_array.storage_array(), exec_ctx)?;
                let storage_compressed =
                    ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_storage)
                        .into_array();

                if scheme_compressed.nbytes() < storage_compressed.nbytes() {
                    Ok(scheme_compressed)
                } else {
                    Ok(storage_compressed)
                }
            }
            Canonical::Variant(variant_array) => {
                let core_storage =
                    self.compress_physical_slots(variant_array.core_storage(), exec_ctx)?;
                let shredded = variant_array
                    .shredded()
                    .map(|arr| {
                        // Avoid stack-overflow for variant shredded values
                        if arr.is::<Variant>() {
                            self.compress_physical_slots(arr, exec_ctx)
                        } else {
                            self.compress(arr, exec_ctx)
                        }
                    })
                    .transpose()?;

                Ok(VariantArray::try_new(core_storage, shredded)?.into_array())
            }
        }
    }

    /// The main scheme-selection entry point for a single leaf array.
    ///
    /// Filters allowed schemes by [`matches`] and exclusion rules, merges their [`stats_options`]
    /// into a single [`GenerateStatsOptions`], and picks the winner by estimated compression
    /// ratio.
    ///
    /// If a winner is found and its compressed output is actually smaller, that output is
    /// returned. Otherwise, the original array is returned unchanged.
    ///
    /// Empty, all-null, and constant arrays are handled by the compressor itself before any
    /// scheme evaluation (constant detection is skipped while compressing samples).
    ///
    /// [`matches`]: Scheme::matches
    /// [`stats_options`]: Scheme::stats_options
    fn choose_and_compress(
        &self,
        canonical: Canonical,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let eligible_schemes: Vec<&'static dyn Scheme> = self
            .schemes
            .iter()
            .copied()
            .filter(|s| s.matches(&canonical) && !self.is_excluded(*s, &compress_ctx))
            .collect();

        let array: ArrayRef = canonical.into();

        if array.is_empty() {
            return Ok(array);
        }

        if array.all_invalid(exec_ctx)? {
            return Ok(
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            );
        }

        let before_nbytes = array.nbytes();

        let merged_opts = eligible_schemes
            .iter()
            .fold(GenerateStatsOptions::default(), |acc, s| {
                acc.merge(s.stats_options())
            });
        let compress_ctx = compress_ctx.with_merged_stats_options(merged_opts);

        let data = ArrayAndStats::new(array, merged_opts);

        // Constant detection is built into the compressor: a constant leaf always short-circuits
        // scheme selection. Samples are exempt because a constant sample does not imply that the
        // full array is constant.
        if !compress_ctx.is_sample() && constant::is_constant_for_compression(&data, exec_ctx)? {
            let _winner_span =
                trace::winner_compress_span(constant::CONSTANT_SCHEME_ID, before_nbytes).entered();
            let compressed = constant::compress_constant(data.array(), exec_ctx)?;

            let after_nbytes = compressed.nbytes();
            let actual_ratio =
                (after_nbytes != 0).then(|| before_nbytes as f64 / after_nbytes as f64);
            let accepted = after_nbytes < before_nbytes;
            trace::record_winner_compress_result(after_nbytes, None, actual_ratio, accepted);

            return if accepted {
                Ok(compressed)
            } else {
                store_uncompressed(data.into_array(), exec_ctx)
            };
        }

        if eligible_schemes.is_empty() {
            return Ok(data.into_array());
        }

        let Some((winner, winner_estimate)) =
            self.choose_best_scheme(&eligible_schemes, &data, compress_ctx.clone(), exec_ctx)?
        else {
            return store_uncompressed(data.into_array(), exec_ctx);
        };

        // Run the winning scheme's `compress`. On failure, emit an ERROR event carrying the
        // scheme name and cascade history before propagating.
        let error_ctx = trace::enabled_error_context(&compress_ctx);
        let _winner_span = trace::winner_compress_span(winner.id(), before_nbytes).entered();
        let compressed = winner
            .compress(self, &data, compress_ctx, exec_ctx)
            .inspect_err(|err| {
                // NB: this is the only way we can tell which scheme panicked / bailed on their
                // data, especially for third-party schemes where the error site may not carry any
                // compressor context.
                trace::scheme_compress_failed(winner.id(), before_nbytes, error_ctx.as_ref(), err);
            })?;

        let after_nbytes = compressed.nbytes();
        let actual_ratio = (after_nbytes != 0).then(|| before_nbytes as f64 / after_nbytes as f64);

        // Accept only when the winner clears its own gain floor. A scheme that barely shrinks the
        // array still charges full decode price on every read, so `min_gain` lets expensive schemes
        // (FSST, zstd) decline work that is not worth materializing later, while cheap ones keep
        // the default floor of zero.
        let accepted = (after_nbytes as f64) <= (before_nbytes as f64) * (1.0 - winner.min_gain());

        trace::record_winner_compress_result(
            after_nbytes,
            winner_estimate.trace_ratio(),
            actual_ratio,
            accepted,
        );

        if accepted {
            Ok(compressed)
        } else {
            store_uncompressed(data.into_array(), exec_ctx)
        }
    }
}
