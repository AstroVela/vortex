// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Safe owned output values for infallible row kernels.
//!
//! A [`RowKernelOutput`] owns a completely initialized batch. The executor chooses which rows run
//! and validates the resulting array. A dense kernel can use [`PackedBoolOutput`] to write native
//! mask words directly without exposing uninitialized storage.

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::validity::Validity;

/// A completely initialized output batch produced by an infallible [`RowKernel`].
///
/// [`from_values`](Self::from_values) is the portable fallback used by the default dense
/// collector. Specialized kernels can construct their associated output directly.
///
/// [`RowKernel`]: crate::scalar_fn::unstable::row::RowKernel
pub trait RowKernelOutput: Sized {
    /// The logical value produced for one row.
    type Element: OutputElement;

    /// Construct the output from one initialized value per row.
    ///
    /// Valid-only execution supplies [`Default::default`] placeholders at invalid rows. Batch
    /// execution masks those rows before returning the array.
    /// The output must preserve the input length and row order.
    fn from_values(values: Vec<Self::Element>) -> VortexResult<Self>;

    /// Construct the all-valid output array.
    ///
    /// The array must preserve the output's row count and match
    /// [`OutputElement::element_dtype`] except for outer nullability. Per-row evaluation is
    /// infallible, so errors from this method must not report value-dependent semantic failures.
    fn finish(self) -> VortexResult<ArrayRef>;
}

/// A kernel output backed by one native Rust value per row.
pub struct VecOutput<T> {
    values: Vec<T>,
}

impl<T: OutputElement> RowKernelOutput for VecOutput<T> {
    type Element = T;

    fn from_values(values: Vec<Self::Element>) -> VortexResult<Self> {
        Ok(Self { values })
    }

    fn finish(self) -> VortexResult<ArrayRef> {
        Ok(T::build(self.values))
    }
}

/// A boolean kernel output backed by initialized native mask words.
///
/// Dense SIMD kernels can write AVX-512 mask registers directly into [`words_mut`](Self::words_mut).
/// Bit `i % 64` of word `i / 64` stores row `i`, with the least-significant bit storing the first
/// row in each word.
/// Unused tail bits can contain any value; [`finish`](RowKernelOutput::finish) clears them before
/// constructing the boolean array.
pub struct PackedBoolOutput {
    words: BufferMut<u64>,
    row_count: usize,
}

impl PackedBoolOutput {
    /// Allocate an all-false output with `row_count` initialized bits.
    pub fn zeroed(row_count: usize) -> Self {
        Self {
            words: BufferMut::zeroed(row_count.div_ceil(64)),
            row_count,
        }
    }

    /// Return the logical number of output rows.
    pub fn len(&self) -> usize {
        self.row_count
    }

    /// Return whether the output contains no rows.
    pub fn is_empty(&self) -> bool {
        self.row_count == 0
    }

    /// Return the initialized native words that store the output bits in row order.
    pub fn words_mut(&mut self) -> &mut [u64] {
        self.words.as_mut_slice()
    }
}

impl RowKernelOutput for PackedBoolOutput {
    type Element = bool;

    fn from_values(values: Vec<Self::Element>) -> VortexResult<Self> {
        let mut output = Self::zeroed(values.len());

        for (index, value) in values.into_iter().enumerate() {
            if value {
                output.words[index / 64] |= 1_u64 << (index % 64);
            }
        }

        Ok(output)
    }

    fn finish(mut self) -> VortexResult<ArrayRef> {
        if let Some(last_word) = self.words.last_mut()
            && !self.row_count.is_multiple_of(64)
        {
            *last_word &= (1_u64 << (self.row_count % 64)) - 1;
        }

        for word in self.words.iter_mut() {
            *word = word.to_le();
        }

        let mut bytes = self.words.into_byte_buffer();
        bytes.truncate(self.row_count.div_ceil(8));
        let values = BitBuffer::new(bytes.freeze(), self.row_count);

        Ok(BoolArray::new(values, Validity::NonNullable).into_array())
    }
}
