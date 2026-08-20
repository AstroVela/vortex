// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unit-vector extension type for fixed-length float vectors.
//!
//! A [`UnitVector`] has the same storage layout as [`Vector`], but each non-null row is finite,
//! nonzero, and has an L2 norm within [`unit_norm_tolerance`] of one. The refinement is approximate
//! because its coordinates use finite-precision floating-point values.
//!
//! Use [`UnitVector::try_new_unit_vector_array`] when the storage values are not already trusted.
//! [`UnitVector::new_unchecked`] preserves claims from trusted producers and interchange metadata.
//! Norm-based operations only replace the physical norm with one when the caller selects
//! [`NormMode::AssumeNormalized`].
//!
//! [`NormMode::AssumeNormalized`]: crate::scalar_fns::NormMode::AssumeNormalized
//! [`Vector`]: crate::vector::Vector
//! [`unit_norm_tolerance`]: crate::unit_norm_tolerance

use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_error::VortexResult;

mod arrow;
pub use arrow::ARROW_UNIT_VECTOR_EXTENSION_NAME;

mod matcher;
pub use matcher::AnyUnitVector;

mod validate;

mod vtable;

/// A fixed-length float vector whose non-null rows are approximately unit length.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct UnitVector;

impl UnitVector {
    /// Constructs a [`UnitVector`] array after validating every non-null row.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage dtype is incompatible or a non-null row is zero,
    /// non-finite, or outside [`unit_norm_tolerance`](crate::unit_norm_tolerance) of unit length.
    pub fn try_new_unit_vector_array(
        storage: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // SAFETY: `validate_unit_vector_rows` validates every non-null row before this array is
        // returned.
        let array = unsafe { Self::new_unchecked(storage)? };
        validate::validate_unit_vector_rows(&array, ctx)?;

        Ok(array)
    }

    /// Constructs a [`UnitVector`] array without validating its row values.
    ///
    /// # Safety
    ///
    /// Every non-null row must be finite, nonzero, and have an L2 norm within
    /// [`unit_norm_tolerance`](crate::unit_norm_tolerance) of one. Violating this contract can
    /// produce incorrect results from operations using [`NormMode::AssumeNormalized`], but it
    /// cannot cause memory unsafety.
    ///
    /// [`NormMode::AssumeNormalized`]: crate::scalar_fns::NormMode::AssumeNormalized
    pub unsafe fn new_unchecked(storage: ArrayRef) -> VortexResult<ArrayRef> {
        ExtensionArray::try_new_from_vtable(UnitVector, EmptyMetadata, storage)
            .map(|array| array.into_array())
    }
}

#[cfg(test)]
mod tests;
