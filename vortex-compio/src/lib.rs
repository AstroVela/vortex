// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compio runtime and filesystem I/O integration for Vortex.
//!
//! [`CompioRuntime`] drives Vortex work on Compio's thread-local completion-based runtime. Create
//! one runtime per worker thread to use a thread-per-core execution model. [`CompioFileReadAt`]
//! performs positioned reads directly into aligned Vortex buffers, using io_uring on Linux when
//! available and Compio's polling driver as a fallback.

mod read_at;
mod runtime;

pub use read_at::*;
pub use runtime::*;
