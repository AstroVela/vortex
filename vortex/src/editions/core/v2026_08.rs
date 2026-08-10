// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 core edition adding the canonical Map encoding, and the first edition
//! to guarantee layouts and aggregate functions alongside array encodings.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;

/// The August 2026 core edition containing canonical Map arrays, the modern layouts, and
/// the serializable aggregate functions stored by zone maps.
pub const CORE_2026_08: EditionId = EditionId::new("core", 2026, 8, 0);

/// The declaration of [`CORE_2026_08`] and the objects that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_08,
        min_vortex_version: Some("0.84.0"),
    },
    added_arrays: &[&"vortex.map"],
    // `vortex.dict` and `vortex.zoned` shipped long before this edition, but zoned
    // metadata moved to serialized aggregate descriptors in mid-2026, so this is the first
    // frozen edition whose min_vortex_version can read what today's writers emit. The
    // membership floor is recorded conservatively here rather than back-dated; move a
    // layout to an earlier edition only with compat-fixture evidence that the earlier
    // edition's min_vortex_version reads its current serialized form.
    added_layouts: &[&"vortex.dict", &"vortex.list", &"vortex.zoned"],
    // Every aggregate function registered by the default session whose options serialize:
    // the set a zoned layout's zone maps may store. `vortex.min_max` is deliberately
    // absent (it is not serializable) and stays a purely in-memory aggregate.
    added_aggregations: &[
        &"vortex.all_nan",
        &"vortex.all_non_distinct",
        &"vortex.all_non_nan",
        &"vortex.all_non_null",
        &"vortex.all_null",
        &"vortex.bounded_max",
        &"vortex.bounded_min",
        &"vortex.first",
        &"vortex.is_constant",
        &"vortex.is_sorted",
        &"vortex.last",
        &"vortex.max",
        &"vortex.min",
        &"vortex.nan_count",
        &"vortex.null_count",
        &"vortex.sum",
        &"vortex.uncompressed_size_in_bytes",
    ],
    added_expressions: &[],
    added_extension_dtypes: &[],
};
