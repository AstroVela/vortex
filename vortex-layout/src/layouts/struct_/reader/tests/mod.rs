// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for expressions over nullable structs.
//!
//! The same scenarios are exercised twice, by two deliberately independent groups:
//!
//! * [`in_memory`] evaluates each expression against an in-memory struct array. This is the
//!   reference semantics — no layouts, no partitioning.
//! * [`layout`] evaluates the same expression against a [`StructLayout`] read back through
//!   [`StructReader`], which partitions it over the layout's children.
//!
//! Each group owns its own fixtures and helpers. They are intentionally *not* shared between the
//! groups: if the layout group reused the in-memory group's expectations, a change in the
//! reference semantics would silently move both sides at once and the cross-check would be
//! worthless.
//!
//! [`StructLayout`]: crate::layouts::struct_::StructLayout
//! [`StructReader`]: super::StructReader

mod in_memory;
mod layout;
