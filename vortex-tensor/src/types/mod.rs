// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Internal homes for tensor extension types.
//!
//! Each child module owns one extension dtype, its validation, and its interchange support. The
//! crate root re-exports these modules as [`fixed_shape_tensor`], [`unit_vector`], and [`vector`].

pub mod fixed_shape_tensor;
pub mod unit_vector;
pub mod vector;
