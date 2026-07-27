// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helpers for tests that write files from a session built below the `vortex` facade.

use vortex_array::session::ArraySessionExt;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

/// A private edition family, so it can never collide with a first-party declaration.
const TEST_EDITION: EditionId = EditionId::new("test", 2026, 7, 0);

/// Let the writer emit every array encoding registered with `session`.
///
/// [`crate::register_default_encodings`] only makes encodings readable; the writer's policy comes
/// from the session's enabled editions, which the `vortex` facade seeds. Sessions assembled
/// directly from [`vortex_array::array_session`] have no editions enabled and so can write
/// nothing. This declares one throwaway edition covering everything currently registered and
/// enables it.
///
/// Call this after all encodings are registered: later registrations are not picked up.
///
/// # Panics
///
/// Panics if called twice on the same session, or if an encoding already belongs to an edition.
pub fn enable_all_registered_array_encodings(session: &VortexSession) {
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: TEST_EDITION,
            min_vortex_version: None,
        })
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("test edition is valid");
    for id in session.arrays().registry().ids() {
        editions
            .declare_inclusion(EditionInclusion::new(&id, TEST_EDITION))
            .map_err(|error| vortex_err!("{error}"))
            .vortex_expect("registered array encoding has one test-edition inclusion");
    }
    session
        .enable_edition(TEST_EDITION)
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("test edition is registered");
}
