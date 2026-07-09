// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use smallvec::smallvec;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::TakeSlices;
use crate::arrays::take_slices::selector_output_len;
use crate::arrays::take_slices::selector_output_len_with_ctx;

/// The child array selected by the run sequence.
pub(super) const CHILD_SLOT: usize = 0;
/// The selector naming the start of each child run.
pub(super) const STARTS_SLOT: usize = 1;
/// The selector naming the length of each child run.
pub(super) const LENGTHS_SLOT: usize = 2;
pub(super) const NUM_SLOTS: usize = 3;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child", "starts", "lengths"];

/// Metadata for a [`TakeSlices`] array.
#[derive(Clone, Debug)]
pub struct TakeSlicesData {
    len: usize,
}

impl Display for TakeSlicesData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "len: {}", self.len())
    }
}

/// Extension methods for [`TakeSlices`] arrays.
pub trait TakeSlicesArrayExt: TypedArrayRef<TakeSlices> {
    /// The child array selected by this run sequence.
    fn child(&self) -> &ArrayRef {
        self.as_ref().slots()[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated take-slices child slot")
    }

    /// The selector naming each child run's start offset.
    fn starts(&self) -> &ArrayRef {
        self.as_ref().slots()[STARTS_SLOT]
            .as_ref()
            .vortex_expect("validated take-slices starts slot")
    }

    /// The selector naming each child run's length.
    fn lengths(&self) -> &ArrayRef {
        self.as_ref().slots()[LENGTHS_SLOT]
            .as_ref()
            .vortex_expect("validated take-slices lengths slot")
    }
}
impl<T: TypedArrayRef<TakeSlices>> TakeSlicesArrayExt for T {}

impl TakeSlicesData {
    fn new(len: usize) -> Self {
        Self { len }
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Array<TakeSlices> {
    /// Constructs a new validated `TakeSlicesArray`.
    ///
    /// Prefer [`Self::try_new_with_ctx`] when the caller already has an execution context.
    pub fn try_new(child: ArrayRef, starts: ArrayRef, lengths: ArrayRef) -> VortexResult<Self> {
        let len = selector_output_len(child.len(), &starts, &lengths)?;
        Ok(Self::new_validated(child, starts, lengths, len))
    }

    /// Constructs a new validated `TakeSlicesArray` using the caller's execution context to
    /// validate selector arrays and compute the output length.
    pub fn try_new_with_ctx(
        child: ArrayRef,
        starts: ArrayRef,
        lengths: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let len = selector_output_len_with_ctx(child.len(), &starts, &lengths, ctx)?;
        Ok(Self::new_validated(child, starts, lengths, len))
    }

    fn new_validated(child: ArrayRef, starts: ArrayRef, lengths: ArrayRef, len: usize) -> Self {
        let dtype = child.dtype().clone();
        let data = TakeSlicesData::new(len);

        // SAFETY: `selector_output_len` validates selector dtypes, run bounds, and computes `len`;
        // the outer dtype is copied from the child, and all required slots are populated.
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(TakeSlices, dtype, len, data).with_slots(smallvec![
                    Some(child),
                    Some(starts),
                    Some(lengths)
                ]),
            )
        }
    }
}
