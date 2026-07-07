// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Range;
use std::sync::Arc;

use smallvec::smallvec;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::TakeSlices;

/// The child array being selected by ordered slices.
pub(super) const CHILD_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child"];

/// Metadata for a [`TakeSlices`] array.
#[derive(Clone, Debug)]
pub struct TakeSlicesData {
    pub(super) slices: Arc<[(usize, usize)]>,
    len: usize,
}

impl Display for TakeSlicesData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "nslices: {}, len: {}", self.slices.len(), self.len())
    }
}

/// Extension methods for [`TakeSlices`] arrays.
pub trait TakeSlicesArrayExt: TypedArrayRef<TakeSlices> {
    /// The child array selected by this ordered range list.
    fn child(&self) -> &ArrayRef {
        self.as_ref().slots()[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated take-slices child slot")
    }

    /// The ordered, non-empty child ranges represented by this array.
    fn slices(&self) -> &[(usize, usize)] {
        &self.slices
    }
}
impl<T: TypedArrayRef<TakeSlices>> TakeSlicesArrayExt for T {}

impl TakeSlicesData {
    fn try_new(child_len: usize, slices: Vec<(usize, usize)>) -> VortexResult<Self> {
        let mut len = 0usize;
        for &(start, end) in &slices {
            vortex_ensure!(
                start < end,
                "TakeSlicesArray range must be non-empty: {start}..{end}"
            );
            vortex_ensure!(
                end <= child_len,
                "TakeSlicesArray range {start}..{end} exceeds child array length {child_len}"
            );
            len = len
                .checked_add(end - start)
                .ok_or_else(|| vortex_err!("TakeSlicesArray length overflow"))?;
        }

        Ok(Self {
            slices: Arc::from(slices.into_boxed_slice()),
            len,
        })
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// The ordered ranges used to select child values.
    pub fn slices(&self) -> &[(usize, usize)] {
        &self.slices
    }

    /// Returns the ordered ranges as `Range<usize>` values.
    pub fn slice_ranges(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        self.slices.iter().map(|&(start, end)| start..end)
    }
}

impl Array<TakeSlices> {
    /// Constructs a new validated `TakeSlicesArray`.
    pub fn try_new(child: ArrayRef, slices: Vec<(usize, usize)>) -> VortexResult<Self> {
        let dtype = child.dtype().clone();
        let data = TakeSlicesData::try_new(child.len(), slices)?;
        let len = data.len();

        // SAFETY: `TakeSlicesData::try_new` validates range bounds and computes `len`; the outer
        // dtype is copied from the child, and the required child slot is populated.
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(TakeSlices, dtype, len, data).with_slots(smallvec![Some(child)]),
            )
        })
    }
}
