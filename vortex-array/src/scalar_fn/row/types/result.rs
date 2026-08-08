// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What a sink-writing row closure may return.

use std::ops::BitOrAssign;

use vortex_error::VortexResult;

mod private {
    pub trait Sealed {}
}

/// A value-dependent failure bit reduced across the row loop and handed to the output sink.
///
/// Unlike [`VortexResult`], this never exits the loop. Use it when every row can write a safe
/// provisional value and report failure once at the end.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeferredError(
    /// Whether this value records a deferred error.
    bool,
);

impl DeferredError {
    /// Record whether this row encountered an error.
    pub const fn new(failed: bool) -> Self {
        Self(failed)
    }

    /// Whether any row accumulated into this value failed.
    pub const fn occurred(self) -> bool {
        self.0
    }
}

impl BitOrAssign for DeferredError {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// The result of writing one row: success, an immediate error, or deferred error evidence.
///
/// The executor OR-reduces [`Accumulated`](Self::Accumulated) in a loop-local. The accumulated word
/// should be no wider than the computed element so error tracking does not constrain vector width.
/// This trait is sealed; row functions choose one of its supplied implementations.
pub trait SinkResult: 'static + private::Sealed {
    /// The word this result reduces into, kept in a loop-local by the executor.
    type Accumulated: 'static + Copy + Default;

    /// Whether this return type can carry an error.
    const FALLIBLE: bool;

    /// Whether this result defers failure reporting until the sink finishes.
    const DEFERRED: bool;

    /// Merge this row's outcome into the batch-wide reduction.
    fn accumulate(self, accumulated: &mut Self::Accumulated) -> VortexResult<()>;

    /// Whether the finished reduction means some row failed.
    fn occurred(accumulated: Self::Accumulated) -> bool;
}

impl private::Sealed for () {}

impl SinkResult for () {
    type Accumulated = ();

    const FALLIBLE: bool = false;
    const DEFERRED: bool = false;

    fn accumulate(self, _accumulated: &mut ()) -> VortexResult<()> {
        Ok(())
    }

    fn occurred(_accumulated: ()) -> bool {
        false
    }
}

impl private::Sealed for VortexResult<()> {}

impl SinkResult for VortexResult<()> {
    type Accumulated = ();

    const FALLIBLE: bool = true;
    const DEFERRED: bool = false;

    fn accumulate(self, _accumulated: &mut ()) -> VortexResult<()> {
        self
    }

    fn occurred(_accumulated: ()) -> bool {
        false
    }
}

/// The evidence widths a row closure may reduce. `bool` is the ordinary answer; the unsigned
/// integers exist for a kernel whose per-row comparison would cost it its vectorization.
macro_rules! impl_sink_result_word {
    ($($word:ty),+ $(,)?) => {
        $(
            impl private::Sealed for $word {}

            impl SinkResult for $word {
                type Accumulated = $word;

                const FALLIBLE: bool = false;
                const DEFERRED: bool = true;

                fn accumulate(self, accumulated: &mut $word) -> VortexResult<()> {
                    *accumulated |= self;
                    Ok(())
                }

                fn occurred(accumulated: $word) -> bool {
                    accumulated != <$word>::default()
                }
            }
        )+
    };
}

impl_sink_result_word!(bool, u8, u16, u32, u64);
