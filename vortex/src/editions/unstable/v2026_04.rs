// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The April 2026 `unstable` encoding cohort.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;

/// The April 2026 draft edition of the `unstable` family.
pub const UNSTABLE_2026_04_0: EditionId = EditionId::new("unstable", 2026, 4, 0);

/// The declaration of [`UNSTABLE_2026_04_0`] and the objects that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: UNSTABLE_2026_04_0,
        min_vortex_version: None,
    },
    added_arrays: &[
        &"vortex.parquet.variant",
        &"vortex.patched",
        // The tensor scalar functions persist as scalar-fn array encodings whose ids are
        // the function ids (registration is gated behind
        // `vortex_tensor::SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV`).
        &"vortex.tensor.cosine_similarity",
        &"vortex.tensor.inner_product",
        &"vortex.tensor.l2_norm",
        &"vortex.tensor.normalized",
    ],
    added_layouts: &[],
    added_aggregations: &[],
    added_extension_dtypes: &[&"vortex.tensor.fixed_shape_tensor", &"vortex.tensor.vector"],
};
