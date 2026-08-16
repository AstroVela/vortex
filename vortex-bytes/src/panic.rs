// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The crate's single panic site.
//!
//! `clippy::panic` is denied across the workspace, and `vortex-error`'s `vortex_panic!` is the
//! usual way to opt out of it. This crate deliberately has no Vortex dependencies, so it carries
//! its own: one allowance here rather than one at every call site.
//!
//! Routing panics through an out-of-line `#[cold]` function is also what keeps the panic paths
//! out of the hot code that guards against them.

use std::fmt::Arguments;

/// Panic with a formatted message.
macro_rules! bytes_panic {
    ($($arg:tt)*) => {
        $crate::panic::panic_fmt(format_args!($($arg)*))
    };
}

pub(crate) use bytes_panic;

/// Panic with an already-formatted message. See [`bytes_panic`].
#[cold]
#[inline(never)]
#[expect(
    clippy::panic,
    reason = "the crate's sanctioned panic site, in place of vortex-error's vortex_panic!"
)]
pub(crate) fn panic_fmt(args: Arguments<'_>) -> ! {
    panic!("{args}")
}
