// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What a sink-writing row closure may return.

use std::ops::BitOrAssign;

use vortex_error::VortexResult;

/// A value-dependent failure bit reduced across the whole row loop and handed to the output sink.
///
/// Unlike [`VortexResult`], this never exits the loop. It is for kernels such as checked addition
/// that can safely write a provisional value for every row and report any failure once at the end.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeferredError(i64);

impl DeferredError {
    /// Record whether this row encountered an error.
    pub const fn new(failed: bool) -> Self {
        Self(if failed { i64::MIN } else { 0 })
    }

    /// Record an error represented by the sign bit of `word`.
    ///
    /// This avoids materializing a per-row boolean when a checked operation already computes a
    /// word whose sign bit means failure. Non-sign bits are ignored by [`occurred`](Self::occurred)
    /// and may hold arbitrary values.
    pub const fn from_sign_bit(word: i64) -> Self {
        Self(word)
    }

    /// Whether any row accumulated into this value failed.
    pub const fn occurred(self) -> bool {
        self.0 < 0
    }
}

impl BitOrAssign for DeferredError {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// What a row computation that _writes_ into an [`OutputSink`](crate::scalar_fn::OutputSink) may
/// produce: nothing, an early [`VortexResult`] error, or a non-branching [`DeferredError`] bit.
///
/// The value is already in the sink by the time the closure returns, so the only thing left to
/// report is failure.
pub trait SinkResult: 'static {
    /// Whether this return type can carry an error.
    const FALLIBLE: bool;

    /// Whether this result carries a non-branching failure bit for the sink.
    const DEFERRED: bool;

    /// Merge this row's outcome into the batch-wide deferred error.
    fn accumulate(self, deferred_error: &mut DeferredError) -> VortexResult<()>;
}

impl SinkResult for () {
    const FALLIBLE: bool = false;
    const DEFERRED: bool = false;

    fn accumulate(self, _deferred_error: &mut DeferredError) -> VortexResult<()> {
        Ok(())
    }
}

impl SinkResult for VortexResult<()> {
    const FALLIBLE: bool = true;
    const DEFERRED: bool = false;

    fn accumulate(self, _deferred_error: &mut DeferredError) -> VortexResult<()> {
        self
    }
}

impl SinkResult for DeferredError {
    const FALLIBLE: bool = false;
    const DEFERRED: bool = true;

    fn accumulate(self, deferred_error: &mut DeferredError) -> VortexResult<()> {
        *deferred_error |= self;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::DeferredError;

    #[test]
    fn sign_bit_is_the_only_error_bit() {
        let mut error = DeferredError::from_sign_bit(i64::MAX);
        assert!(!error.occurred());

        error |= DeferredError::from_sign_bit(i64::MIN);
        assert!(error.occurred());
    }
}
