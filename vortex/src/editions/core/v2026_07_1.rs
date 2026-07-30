// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The July 2026 core edition revision adding the canonical Map encoding.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;

/// The July 2026 core edition revision containing canonical Map arrays.
pub const CORE_2026_07_1: EditionId = EditionId::new("core", 2026, 7, 1);

/// The declaration of [`CORE_2026_07_1`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_07_1,
        min_vortex_version: Some("0.66.0"),
    },
    added: &[&"vortex.map"],
};
