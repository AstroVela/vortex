// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::smallvec;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::EmptyArrayData;
use crate::array::TypedArrayRef;
use crate::arrays::TakeSlices;

/// The child array selected by the run sequence.
pub(super) const CHILD_SLOT: usize = 0;
/// The start index for each child run.
pub(super) const STARTS_SLOT: usize = 1;
/// The length for each child run.
pub(super) const LENGTHS_SLOT: usize = 2;
pub(super) const NUM_SLOTS: usize = 3;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child", "starts", "lengths"];

/// Extension methods for [`TakeSlices`] arrays.
pub trait TakeSlicesArrayExt: TypedArrayRef<TakeSlices> {
    /// The child array selected by this run sequence.
    fn child(&self) -> &ArrayRef {
        self.as_ref().slots()[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated take-slices child slot")
    }

    /// The start index for each child run.
    fn starts(&self) -> &ArrayRef {
        self.as_ref().slots()[STARTS_SLOT]
            .as_ref()
            .vortex_expect("validated take-slices starts slot")
    }

    /// The length for each child run.
    fn lengths(&self) -> &ArrayRef {
        self.as_ref().slots()[LENGTHS_SLOT]
            .as_ref()
            .vortex_expect("validated take-slices lengths slot")
    }
}
impl<T: TypedArrayRef<TakeSlices>> TakeSlicesArrayExt for T {}

impl Array<TakeSlices> {
    /// Constructs a new `TakeSlicesArray` from start/length arrays and caller-provided output length.
    ///
    /// Construction validates only the structural array invariants. Index values are interpreted
    /// when the lazy gather is executed.
    pub fn try_new(
        child: ArrayRef,
        starts: ArrayRef,
        lengths: ArrayRef,
        len: usize,
    ) -> VortexResult<Self> {
        let dtype = child.dtype().clone();
        Array::try_from_parts(
            ArrayParts::new(TakeSlices, dtype, len, EmptyArrayData).with_slots(smallvec![
                Some(child),
                Some(starts),
                Some(lengths)
            ]),
        )
    }

    /// Constructs a new `TakeSlicesArray` without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure the child dtype is the output dtype, start/length arrays are
    /// non-nullable unsigned integers of equal length, and `len` is the sum of selected lengths.
    pub unsafe fn new_unchecked(
        child: ArrayRef,
        starts: ArrayRef,
        lengths: ArrayRef,
        len: usize,
    ) -> Self {
        let dtype = child.dtype().clone();
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(TakeSlices, dtype, len, EmptyArrayData).with_slots(smallvec![
                    Some(child),
                    Some(starts),
                    Some(lengths)
                ]),
            )
        }
    }
}
