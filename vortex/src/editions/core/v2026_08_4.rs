// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 draft core edition with numeric array encodings.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The fifth August 2026 edition of the `core` family.
pub const CORE_2026_08_4: EditionId = EditionId::new("core", 2026, 8, 4);

/// The declaration of [`CORE_2026_08_4`] and its new components.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_08_4,
        min_vortex_version: None,
    },
    added: &[
        EditionMember::array(&"vortex.block_residual"),
        EditionMember::array(&"vortex.float_quant"),
        EditionMember::array(&"vortex.ordered_float"),
    ],
};
