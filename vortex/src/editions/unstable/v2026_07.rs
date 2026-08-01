// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;

pub const UNSTABLE_2026_07_0: EditionId = EditionId::new("unstable", 2026, 7, 0);

pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: UNSTABLE_2026_07_0,
        min_vortex_version: None,
    },
    added: &[&"vortex.tiled_fsl"],
};
