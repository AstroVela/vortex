// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::smallvec;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::EmptyArrayData;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::TakeSlices;

#[array_slots(TakeSlices)]
pub struct TakeSlicesSlots {
    /// The child array selected by the run sequence.
    pub child: ArrayRef,
    /// The start index for each child run.
    pub starts: ArrayRef,
    /// The length for each child run.
    pub lengths: ArrayRef,
}

/// Extension methods for [`TakeSlices`] arrays.
pub trait TakeSlicesArrayExt: TypedArrayRef<TakeSlices> + TakeSlicesArraySlotsExt {
    /// The child array selected by this run sequence.
    fn child(&self) -> &ArrayRef {
        TakeSlicesArraySlotsExt::child(self)
    }

    /// The start index for each child run.
    fn starts(&self) -> &ArrayRef {
        TakeSlicesArraySlotsExt::starts(self)
    }

    /// The length for each child run.
    fn lengths(&self) -> &ArrayRef {
        TakeSlicesArraySlotsExt::lengths(self)
    }
}
impl<T: TypedArrayRef<TakeSlices> + TakeSlicesArraySlotsExt> TakeSlicesArrayExt for T {}

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
