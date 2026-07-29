// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared setup for the `vortex-file` integration tests.
#![expect(
    clippy::expect_used,
    reason = "test setup runs inside a LazyLock, where there is no error to propagate"
)]

use vortex_array::session::ArraySessionExt;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_session::VortexSession;

/// The edition these tests enable.
const TEST_EDITION: EditionId = EditionId::new("filetest", 2026, 1, 0);

/// Declare and enable an edition covering every encoding `session` has registered.
///
/// The writer admits only encodings an enabled edition includes, and the first-party
/// declarations live in the `vortex` facade, which `vortex-file` cannot depend on. A session
/// assembled here therefore has to declare its own edition to write anything.
pub fn enable_all_registered_encodings(session: &VortexSession) {
    session
        .editions()
        .declare_edition(Edition {
            id: TEST_EDITION,
            min_vortex_version: Some("0.1.0"),
        })
        .expect("test edition is undeclared");

    let registered = session
        .arrays()
        .registry()
        .read(|map| map.keys().copied().collect::<Vec<_>>());
    for id in registered {
        session
            .editions()
            .declare_inclusion(EditionInclusion::new(&id, TEST_EDITION))
            .expect("each encoding is registered once");
    }

    session
        .enable_edition(TEST_EDITION)
        .expect("test edition was just declared");
}
