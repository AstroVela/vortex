// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Explicit SIMD primitive comparison kernels that write bitmap words directly.

use std::marker::PhantomData;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::dtype::NativePType;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::unstable::row::ArgView;
use crate::scalar_fn::unstable::row::DenseRows;
use crate::scalar_fn::unstable::row::PackedBoolOutput;
use crate::scalar_fn::unstable::row::RowKernel;

#[derive(Clone, Copy)]
pub(super) struct PrimitiveComparisonKernel<T> {
    op: CompareOperator,
    marker: PhantomData<T>,
}

impl<T> PrimitiveComparisonKernel<T> {
    pub(super) fn new(op: CompareOperator) -> Self {
        Self {
            op,
            marker: PhantomData,
        }
    }
}

impl<T: SimdCompare> RowKernel<(T, T)> for PrimitiveComparisonKernel<T> {
    type Element = bool;
    type Output = PackedBoolOutput;

    fn eval(&self, (lhs, rhs): (T, T)) -> bool {
        apply_op(lhs, rhs, self.op)
    }

    fn collect_dense(&self, rows: DenseRows<'_, (T, T)>) -> VortexResult<Self::Output> {
        let row_count = rows.len();
        let lhs = PrimitiveInput::from_arg_view(rows.inputs().0.view(), row_count)?;
        let rhs = PrimitiveInput::from_arg_view(rows.inputs().1.view(), row_count)?;
        let mut output = PackedBoolOutput::zeroed(row_count);

        compare_into_words(lhs, rhs, self.op, row_count, output.words_mut());
        Ok(output)
    }
}

#[derive(Clone, Copy)]
pub(super) enum PrimitiveInput<'a, T> {
    Slice(&'a [T]),
    Constant(T),
}

impl<'a, T: NativePType> PrimitiveInput<'a, T> {
    fn from_arg_view(view: ArgView<'a, T>, row_count: usize) -> VortexResult<Self> {
        match view {
            ArgView::Column(values) => {
                vortex_ensure!(
                    values.len() == row_count,
                    "a decoded row input does not address exactly {row_count} rows",
                );

                Ok(Self::Slice(values))
            }
            ArgView::Constant(value) => {
                vortex_ensure!(
                    value.len() == 1,
                    "a decoded batch constant does not contain exactly one row",
                );

                Ok(Self::Constant(value[0]))
            }
        }
    }

    /// Read one logical input row without checking its index.
    ///
    /// # Safety
    ///
    /// For [`Self::Slice`], `index` must be less than the slice length. A constant accepts every
    /// logical row index.
    unsafe fn get_unchecked(self, index: usize) -> T {
        match self {
            // SAFETY: forwarded from this method's contract.
            Self::Slice(values) => unsafe { *values.get_unchecked(index) },
            Self::Constant(value) => value,
        }
    }
}

fn compare_into_words<T: SimdCompare>(
    lhs: PrimitiveInput<'_, T>,
    rhs: PrimitiveInput<'_, T>,
    op: CompareOperator,
    row_count: usize,
    words: &mut [u64],
) {
    if let (PrimitiveInput::Constant(lhs), PrimitiveInput::Constant(rhs)) = (lhs, rhs) {
        let value = apply_op(lhs, rhs, op);
        words.fill(if value { u64::MAX } else { 0 });
        if value
            && let Some(last) = words.last_mut()
            && !row_count.is_multiple_of(64)
        {
            *last = (1u64 << (row_count % 64)) - 1;
        }
        return;
    }

    let full_words = row_count / 64;

    #[cfg(target_arch = "x86_64")]
    if avx512_available::<T>() {
        let comparison = DenseComparison::new(lhs, rhs, op);
        // SAFETY: `avx512_available` proves the required features. Each full word addresses exactly
        // 64 input rows, or broadcasts a validated constant.
        unsafe { T::compare_words_avx512(comparison, &mut words[..full_words]) };
    } else if avx2_available() {
        let comparison = DenseComparison::new(lhs, rhs, op);
        // SAFETY: `avx2_available` proves AVX2 support. Each full word addresses exactly 64 input
        // rows, or broadcasts a validated constant.
        unsafe { T::compare_words_avx2(comparison, &mut words[..full_words]) };
    } else {
        compare_words_scalar(lhs, rhs, op, &mut words[..full_words]);
    }

    #[cfg(not(target_arch = "x86_64"))]
    compare_words_scalar(lhs, rhs, op, &mut words[..full_words]);

    let remainder = row_count % 64;
    if remainder != 0 {
        let base = full_words * 64;
        words[full_words] = (0..remainder).fold(0, |packed, bit| {
            // SAFETY: `base + bit < row_count` for every tail lane.
            let lhs = unsafe { lhs.get_unchecked(base + bit) };
            // SAFETY: see above.
            let rhs = unsafe { rhs.get_unchecked(base + bit) };
            packed | ((apply_op(lhs, rhs, op) as u64) << bit)
        });
    }
}

#[cfg(target_arch = "x86_64")]
fn avx512_available<T: SimdCompare>() -> bool {
    #[cfg(feature = "_test-harness")]
    {
        cfg!(target_feature = "avx512f")
            && (!T::REQUIRES_AVX512BW || cfg!(target_feature = "avx512bw"))
    }

    #[cfg(not(feature = "_test-harness"))]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && (!T::REQUIRES_AVX512BW || std::arch::is_x86_feature_detected!("avx512bw"))
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    #[cfg(feature = "_test-harness")]
    {
        cfg!(target_feature = "avx2")
    }

    #[cfg(not(feature = "_test-harness"))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(super) enum DenseComparison<'a, T> {
    EqualArrays {
        lhs: &'a [T],
        rhs: &'a [T],
        invert: u64,
    },
    GreaterArrays {
        lhs: &'a [T],
        rhs: &'a [T],
        invert: u64,
    },
    EqualArrayConstant {
        values: &'a [T],
        constant: T,
        invert: u64,
    },
    GreaterArrayConstant {
        values: &'a [T],
        constant: T,
        invert: u64,
    },
    GreaterConstantArray {
        constant: T,
        values: &'a [T],
        invert: u64,
    },
}

#[cfg(target_arch = "x86_64")]
impl<'a, T: NativePType> DenseComparison<'a, T> {
    fn new(lhs: PrimitiveInput<'a, T>, rhs: PrimitiveInput<'a, T>, op: CompareOperator) -> Self {
        if let (PrimitiveInput::Constant(lhs), PrimitiveInput::Slice(rhs)) = (lhs, rhs) {
            return Self::new(
                PrimitiveInput::Slice(rhs),
                PrimitiveInput::Constant(lhs),
                op.swap(),
            );
        }

        match (lhs, rhs, op) {
            (PrimitiveInput::Slice(lhs), PrimitiveInput::Slice(rhs), CompareOperator::Eq) => {
                Self::EqualArrays {
                    lhs,
                    rhs,
                    invert: 0,
                }
            }
            (PrimitiveInput::Slice(lhs), PrimitiveInput::Slice(rhs), CompareOperator::NotEq) => {
                Self::EqualArrays {
                    lhs,
                    rhs,
                    invert: u64::MAX,
                }
            }
            (PrimitiveInput::Slice(lhs), PrimitiveInput::Slice(rhs), CompareOperator::Gt) => {
                Self::GreaterArrays {
                    lhs,
                    rhs,
                    invert: 0,
                }
            }
            (PrimitiveInput::Slice(lhs), PrimitiveInput::Slice(rhs), CompareOperator::Gte) => {
                Self::GreaterArrays {
                    lhs: rhs,
                    rhs: lhs,
                    invert: u64::MAX,
                }
            }
            (PrimitiveInput::Slice(lhs), PrimitiveInput::Slice(rhs), CompareOperator::Lt) => {
                Self::GreaterArrays {
                    lhs: rhs,
                    rhs: lhs,
                    invert: 0,
                }
            }
            (PrimitiveInput::Slice(lhs), PrimitiveInput::Slice(rhs), CompareOperator::Lte) => {
                Self::GreaterArrays {
                    lhs,
                    rhs,
                    invert: u64::MAX,
                }
            }
            (
                PrimitiveInput::Slice(values),
                PrimitiveInput::Constant(constant),
                CompareOperator::Eq,
            ) => Self::EqualArrayConstant {
                values,
                constant,
                invert: 0,
            },
            (
                PrimitiveInput::Slice(values),
                PrimitiveInput::Constant(constant),
                CompareOperator::NotEq,
            ) => Self::EqualArrayConstant {
                values,
                constant,
                invert: u64::MAX,
            },
            (
                PrimitiveInput::Slice(values),
                PrimitiveInput::Constant(constant),
                CompareOperator::Gt,
            ) => Self::GreaterArrayConstant {
                values,
                constant,
                invert: 0,
            },
            (
                PrimitiveInput::Slice(values),
                PrimitiveInput::Constant(constant),
                CompareOperator::Gte,
            ) => Self::GreaterConstantArray {
                constant,
                values,
                invert: u64::MAX,
            },
            (
                PrimitiveInput::Slice(values),
                PrimitiveInput::Constant(constant),
                CompareOperator::Lt,
            ) => Self::GreaterConstantArray {
                constant,
                values,
                invert: 0,
            },
            (
                PrimitiveInput::Slice(values),
                PrimitiveInput::Constant(constant),
                CompareOperator::Lte,
            ) => Self::GreaterArrayConstant {
                values,
                constant,
                invert: u64::MAX,
            },
            (PrimitiveInput::Constant(_), PrimitiveInput::Constant(_), _) => unreachable!(),
            (PrimitiveInput::Constant(_), PrimitiveInput::Slice(_), _) => unreachable!(),
        }
    }
}

#[cold]
fn compare_words_scalar<T: NativePType>(
    lhs: PrimitiveInput<'_, T>,
    rhs: PrimitiveInput<'_, T>,
    op: CompareOperator,
    words: &mut [u64],
) {
    for (word_index, word) in words.iter_mut().enumerate() {
        let base = word_index * 64;
        *word = (0..64).fold(0, |packed, bit| {
            // SAFETY: every complete word addresses 64 validated rows.
            let lhs = unsafe { lhs.get_unchecked(base + bit) };
            // SAFETY: see above.
            let rhs = unsafe { rhs.get_unchecked(base + bit) };
            packed | ((apply_op(lhs, rhs, op) as u64) << bit)
        });
    }
}

fn apply_op<T: NativePType>(lhs: T, rhs: T, op: CompareOperator) -> bool {
    match op {
        CompareOperator::Eq => lhs.is_eq(rhs),
        CompareOperator::NotEq => !lhs.is_eq(rhs),
        CompareOperator::Gt => lhs.is_gt(rhs),
        CompareOperator::Gte => lhs.is_ge(rhs),
        CompareOperator::Lt => lhs.is_lt(rhs),
        CompareOperator::Lte => lhs.is_le(rhs),
    }
}

pub(super) trait SimdCompare: NativePType {
    /// Whether this type's AVX-512 implementation also requires AVX-512BW.
    #[cfg(target_arch = "x86_64")]
    const REQUIRES_AVX512BW: bool;

    /// Compare complete 64-row chunks with AVX2 and write one bitmap word per chunk.
    ///
    /// # Safety
    ///
    /// The current CPU must support AVX2. Every slice in `comparison` must contain at least
    /// `words.len() * 64` elements. Violating these requirements can execute an unsupported
    /// instruction or read outside an input allocation.
    #[cfg(target_arch = "x86_64")]
    unsafe fn compare_words_avx2(comparison: DenseComparison<'_, Self>, words: &mut [u64]);

    /// Compare complete 64-row chunks with AVX-512 and write one bitmap word per chunk.
    ///
    /// # Safety
    ///
    /// The current CPU must support AVX-512F and, when [`Self::REQUIRES_AVX512BW`] is true,
    /// AVX-512BW. Every slice in `comparison` must contain at least `words.len() * 64` elements.
    /// Violating these requirements can execute an unsupported instruction or read outside an
    /// input allocation.
    #[cfg(target_arch = "x86_64")]
    unsafe fn compare_words_avx512(comparison: DenseComparison<'_, Self>, words: &mut [u64]);
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use half::f16;

    use super::DenseComparison;
    use super::SimdCompare;

    #[inline(always)]
    fn compact_even_bits(mut mask: u32) -> u64 {
        mask &= 0x5555_5555;
        mask = (mask | (mask >> 1)) & 0x3333_3333;
        mask = (mask | (mask >> 2)) & 0x0f0f_0f0f;
        mask = (mask | (mask >> 4)) & 0x00ff_00ff;
        ((mask | (mask >> 8)) & 0x0000_ffff) as u64
    }

    macro_rules! load_or_broadcast {
        ($input:expr, $base:expr, $constant:literal, $load:ident, $set:ident, $vec:ty) => {{
            if $constant {
                $set(unsafe { *$input } as _)
            } else {
                // SAFETY: the enclosing word kernel proves this vector load is in bounds.
                unsafe { $load($input.add($base).cast::<$vec>()) }
            }
        }};
    }

    trait ToI16Bits {
        fn to_i16_bits(self) -> i16;
    }

    impl ToI16Bits for i16 {
        fn to_i16_bits(self) -> i16 {
            self
        }
    }

    impl ToI16Bits for u16 {
        fn to_i16_bits(self) -> i16 {
            self as i16
        }
    }

    impl ToI16Bits for f16 {
        fn to_i16_bits(self) -> i16 {
            self.to_bits() as i16
        }
    }

    trait ToI32Bits {
        fn to_i32_bits(self) -> i32;
    }

    impl ToI32Bits for i32 {
        fn to_i32_bits(self) -> i32 {
            self
        }
    }

    impl ToI32Bits for u32 {
        fn to_i32_bits(self) -> i32 {
            self as i32
        }
    }

    impl ToI32Bits for f32 {
        fn to_i32_bits(self) -> i32 {
            self.to_bits() as i32
        }
    }

    macro_rules! load_16_or_broadcast {
        ($input:expr, $base:expr, $constant:literal, $load:ident, $set:ident, $vec:ty) => {{
            if $constant {
                $set(unsafe { *$input }.to_i16_bits())
            } else {
                // SAFETY: the enclosing word kernel proves this vector load is in bounds.
                unsafe { $load($input.add($base).cast::<$vec>()) }
            }
        }};
    }

    macro_rules! load_32_or_broadcast {
        ($input:expr, $base:expr, $constant:literal, $load:ident, $set:ident, $vec:ty) => {{
            if $constant {
                $set(unsafe { *$input }.to_i32_bits())
            } else {
                // SAFETY: the enclosing word kernel proves this vector load is in bounds.
                unsafe { $load($input.add($base).cast::<$vec>()) }
            }
        }};
    }

    trait ToI64Bits {
        fn to_i64_bits(self) -> i64;
    }

    impl ToI64Bits for u8 {
        fn to_i64_bits(self) -> i64 {
            i64::from(self)
        }
    }

    impl ToI64Bits for i64 {
        fn to_i64_bits(self) -> i64 {
            self
        }
    }

    impl ToI64Bits for u64 {
        fn to_i64_bits(self) -> i64 {
            self as i64
        }
    }

    impl ToI64Bits for f64 {
        fn to_i64_bits(self) -> i64 {
            self.to_bits() as i64
        }
    }

    macro_rules! load_64_or_broadcast {
        ($input:expr, $base:expr, $constant:literal, $load:ident, $set:ident, $vec:ty) => {{
            if $constant {
                $set(unsafe { *$input }.to_i64_bits())
            } else {
                // SAFETY: the enclosing word kernel proves this vector load is in bounds.
                unsafe { $load($input.add($base).cast::<$vec>()) }
            }
        }};
    }

    macro_rules! dispatch_comparison {
        ($comparison:expr, $words:expr, $compare:ident) => {
            match $comparison {
                DenseComparison::EqualArrays { lhs, rhs, invert } => {
                    $compare!(
                        lhs.as_ptr(),
                        rhs.as_ptr(),
                        false,
                        false,
                        true,
                        invert,
                        $words
                    )
                }
                DenseComparison::GreaterArrays { lhs, rhs, invert } => {
                    $compare!(
                        lhs.as_ptr(),
                        rhs.as_ptr(),
                        false,
                        false,
                        false,
                        invert,
                        $words
                    )
                }
                DenseComparison::EqualArrayConstant {
                    values,
                    constant,
                    invert,
                } => $compare!(
                    values.as_ptr(),
                    std::ptr::from_ref(&constant),
                    false,
                    true,
                    true,
                    invert,
                    $words
                ),
                DenseComparison::GreaterArrayConstant {
                    values,
                    constant,
                    invert,
                } => $compare!(
                    values.as_ptr(),
                    std::ptr::from_ref(&constant),
                    false,
                    true,
                    false,
                    invert,
                    $words
                ),
                DenseComparison::GreaterConstantArray {
                    constant,
                    values,
                    invert,
                } => $compare!(
                    std::ptr::from_ref(&constant),
                    values.as_ptr(),
                    true,
                    false,
                    false,
                    invert,
                    $words
                ),
            }
        };
    }

    // The kernels below store each natural mask chunk directly into the output allocation. This
    // avoids assembling a word with shifts and ORs, which LLVM can turn into a slower vectorized
    // reduction. The subword order is the bitmap byte order because x86-64 is little-endian.
    macro_rules! impl_8 {
        ($t:ty, $unsigned:expr) => {
            #[allow(clippy::cast_possible_truncation)]
            impl SimdCompare for $t {
                const REQUIRES_AVX512BW: bool = true;

                #[target_feature(enable = "avx2")]
                unsafe fn compare_words_avx2(
                    comparison: DenseComparison<'_, Self>,
                    words: &mut [u64],
                ) {
                    macro_rules! compare {
                        ($lhs:expr, $rhs:expr, $lhs_constant:literal, $rhs_constant:literal, $eq:literal, $invert:expr, $words:expr) => {{
                            let output = $words.as_mut_ptr().cast::<u32>();
                            for chunk in 0..($words.len() * 2) {
                                let offset = chunk * 32;
                                let lhs = load_or_broadcast!(
                                    $lhs,
                                    offset,
                                    $lhs_constant,
                                    _mm256_loadu_si256,
                                    _mm256_set1_epi8,
                                    __m256i
                                );
                                let rhs = load_or_broadcast!(
                                    $rhs,
                                    offset,
                                    $rhs_constant,
                                    _mm256_loadu_si256,
                                    _mm256_set1_epi8,
                                    __m256i
                                );
                                let mask = if $eq {
                                    _mm256_movemask_epi8(_mm256_cmpeq_epi8(lhs, rhs)) as u32
                                } else {
                                    let (lhs, rhs) = if $unsigned {
                                        let sign = _mm256_set1_epi8(i8::MIN);
                                        (
                                            _mm256_xor_si256(lhs, sign),
                                            _mm256_xor_si256(rhs, sign),
                                        )
                                    } else {
                                        (lhs, rhs)
                                    };
                                    _mm256_movemask_epi8(_mm256_cmpgt_epi8(lhs, rhs)) as u32
                                };

                                // SAFETY: each output value represents 32 rows. The caller
                                // provides two values for each complete 64-row output word.
                                unsafe { output.add(chunk).write(mask ^ ($invert as u32)) };
                            }
                        }};
                    }

                    dispatch_comparison!(comparison, words, compare);
                }

                #[target_feature(enable = "avx512f,avx512bw")]
                unsafe fn compare_words_avx512(
                    comparison: DenseComparison<'_, Self>,
                    words: &mut [u64],
                ) {
                    macro_rules! compare {
                        ($lhs:expr, $rhs:expr, $lhs_constant:literal, $rhs_constant:literal, $eq:literal, $invert:expr, $words:expr) => {{
                            for (word_index, word) in $words.iter_mut().enumerate() {
                                let base = word_index * 64;
                                let lhs = load_or_broadcast!(
                                    $lhs,
                                    base,
                                    $lhs_constant,
                                    _mm512_loadu_si512,
                                    _mm512_set1_epi8,
                                    __m512i
                                );
                                let rhs = load_or_broadcast!(
                                    $rhs,
                                    base,
                                    $rhs_constant,
                                    _mm512_loadu_si512,
                                    _mm512_set1_epi8,
                                    __m512i
                                );
                                let mask = if $eq {
                                    _mm512_cmpeq_epi8_mask(lhs, rhs)
                                } else if $unsigned {
                                    _mm512_cmp_epu8_mask::<6>(lhs, rhs)
                                } else {
                                    _mm512_cmp_epi8_mask::<6>(lhs, rhs)
                                };
                                *word = mask ^ $invert;
                            }
                        }};
                    }

                    dispatch_comparison!(comparison, words, compare);
                }
            }
        };
    }

    macro_rules! impl_16 {
        ($t:ty, $unsigned:expr, $float:expr) => {
            #[allow(clippy::cast_possible_truncation)]
            impl SimdCompare for $t {
                const REQUIRES_AVX512BW: bool = true;

                #[target_feature(enable = "avx2")]
                unsafe fn compare_words_avx2(
                    comparison: DenseComparison<'_, Self>,
                    words: &mut [u64],
                ) {
                    macro_rules! compare {
                        ($lhs:expr, $rhs:expr, $lhs_constant:literal, $rhs_constant:literal, $eq:literal, $invert:expr, $words:expr) => {{
                            let output = $words.as_mut_ptr().cast::<u16>();
                            for chunk in 0..($words.len() * 4) {
                                let offset = chunk * 16;
                                let lhs = load_16_or_broadcast!(
                                    $lhs,
                                    offset,
                                    $lhs_constant,
                                    _mm256_loadu_si256,
                                    _mm256_set1_epi16,
                                    __m256i
                                );
                                let rhs = load_16_or_broadcast!(
                                    $rhs,
                                    offset,
                                    $rhs_constant,
                                    _mm256_loadu_si256,
                                    _mm256_set1_epi16,
                                    __m256i
                                );
                                let mask = if $eq {
                                    compact_even_bits(
                                        _mm256_movemask_epi8(_mm256_cmpeq_epi16(lhs, rhs)) as u32,
                                    ) as u16
                                } else {
                                    let (lhs, rhs) = if $float {
                                        // SAFETY: this method's target-feature contract includes AVX2.
                                        unsafe { (float_key_16(lhs), float_key_16(rhs)) }
                                    } else if $unsigned {
                                        let sign = _mm256_set1_epi16(i16::MIN);
                                        (
                                            _mm256_xor_si256(lhs, sign),
                                            _mm256_xor_si256(rhs, sign),
                                        )
                                    } else {
                                        (lhs, rhs)
                                    };
                                    compact_even_bits(
                                        _mm256_movemask_epi8(_mm256_cmpgt_epi16(lhs, rhs)) as u32,
                                    ) as u16
                                };

                                // SAFETY: each output value represents 16 rows. The caller
                                // provides four values for each complete 64-row output word.
                                unsafe { output.add(chunk).write(mask ^ ($invert as u16)) };
                            }
                        }};
                    }

                    dispatch_comparison!(comparison, words, compare);
                }

                #[target_feature(enable = "avx512f,avx512bw")]
                unsafe fn compare_words_avx512(
                    comparison: DenseComparison<'_, Self>,
                    words: &mut [u64],
                ) {
                    macro_rules! compare {
                        ($lhs:expr, $rhs:expr, $lhs_constant:literal, $rhs_constant:literal, $eq:literal, $invert:expr, $words:expr) => {{
                            let output = $words.as_mut_ptr().cast::<u32>();
                            for chunk in 0..($words.len() * 2) {
                                let offset = chunk * 32;
                                let lhs = load_16_or_broadcast!(
                                    $lhs,
                                    offset,
                                    $lhs_constant,
                                    _mm512_loadu_si512,
                                    _mm512_set1_epi16,
                                    __m512i
                                );
                                let rhs = load_16_or_broadcast!(
                                    $rhs,
                                    offset,
                                    $rhs_constant,
                                    _mm512_loadu_si512,
                                    _mm512_set1_epi16,
                                    __m512i
                                );
                                let mask = if $eq {
                                    _mm512_cmpeq_epi16_mask(lhs, rhs)
                                } else {
                                    let (lhs, rhs) = if $float {
                                        // SAFETY: this method's target-feature contract includes AVX-512BW.
                                        unsafe {
                                            (
                                                float_key_16_avx512(lhs),
                                                float_key_16_avx512(rhs),
                                            )
                                        }
                                    } else {
                                        (lhs, rhs)
                                    };
                                    if $unsigned {
                                        _mm512_cmp_epu16_mask::<6>(lhs, rhs)
                                    } else {
                                        _mm512_cmp_epi16_mask::<6>(lhs, rhs)
                                    }
                                };

                                // SAFETY: each output value represents 32 rows. The caller
                                // provides two values for each complete 64-row output word.
                                unsafe { output.add(chunk).write(mask ^ ($invert as u32)) };
                            }
                        }};
                    }

                    dispatch_comparison!(comparison, words, compare);
                }
            }
        };
    }

    macro_rules! impl_32 {
        ($t:ty, $unsigned:expr, $float:expr) => {
            #[allow(clippy::cast_possible_truncation)]
            impl SimdCompare for $t {
                const REQUIRES_AVX512BW: bool = false;

                #[target_feature(enable = "avx2")]
                unsafe fn compare_words_avx2(
                    comparison: DenseComparison<'_, Self>,
                    words: &mut [u64],
                ) {
                    macro_rules! compare {
                        ($lhs:expr, $rhs:expr, $lhs_constant:literal, $rhs_constant:literal, $eq:literal, $invert:expr, $words:expr) => {{
                            let output = $words.as_mut_ptr().cast::<u8>();
                            for chunk in 0..($words.len() * 8) {
                                let offset = chunk * 8;
                                let lhs = load_32_or_broadcast!(
                                    $lhs,
                                    offset,
                                    $lhs_constant,
                                    _mm256_loadu_si256,
                                    _mm256_set1_epi32,
                                    __m256i
                                );
                                let rhs = load_32_or_broadcast!(
                                    $rhs,
                                    offset,
                                    $rhs_constant,
                                    _mm256_loadu_si256,
                                    _mm256_set1_epi32,
                                    __m256i
                                );
                                let mask = if $eq {
                                    _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(
                                        lhs, rhs,
                                    ))) as u8
                                } else {
                                    let (lhs, rhs) = if $float {
                                        // SAFETY: this method's target-feature contract includes AVX2.
                                        unsafe { (float_key_32(lhs), float_key_32(rhs)) }
                                    } else if $unsigned {
                                        let sign = _mm256_set1_epi32(i32::MIN);
                                        (
                                            _mm256_xor_si256(lhs, sign),
                                            _mm256_xor_si256(rhs, sign),
                                        )
                                    } else {
                                        (lhs, rhs)
                                    };
                                    _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(
                                        lhs, rhs,
                                    ))) as u8
                                };

                                // SAFETY: each output byte represents eight rows. The caller
                                // provides eight bytes for each complete 64-row output word.
                                unsafe { output.add(chunk).write(mask ^ ($invert as u8)) };
                            }
                        }};
                    }

                    dispatch_comparison!(comparison, words, compare);
                }

                #[target_feature(enable = "avx512f")]
                unsafe fn compare_words_avx512(
                    comparison: DenseComparison<'_, Self>,
                    words: &mut [u64],
                ) {
                    macro_rules! compare {
                        ($lhs:expr, $rhs:expr, $lhs_constant:literal, $rhs_constant:literal, $eq:literal, $invert:expr, $words:expr) => {{
                            let output = $words.as_mut_ptr().cast::<u16>();
                            for chunk in 0..($words.len() * 4) {
                                let offset = chunk * 16;
                                let lhs = load_32_or_broadcast!(
                                    $lhs,
                                    offset,
                                    $lhs_constant,
                                    _mm512_loadu_si512,
                                    _mm512_set1_epi32,
                                    __m512i
                                );
                                let rhs = load_32_or_broadcast!(
                                    $rhs,
                                    offset,
                                    $rhs_constant,
                                    _mm512_loadu_si512,
                                    _mm512_set1_epi32,
                                    __m512i
                                );
                                let mask = if $eq {
                                    _mm512_cmpeq_epi32_mask(lhs, rhs)
                                } else {
                                    let (lhs, rhs) = if $float {
                                        // SAFETY: this method's target-feature contract includes AVX-512F.
                                        unsafe {
                                            (
                                                float_key_32_avx512(lhs),
                                                float_key_32_avx512(rhs),
                                            )
                                        }
                                    } else {
                                        (lhs, rhs)
                                    };
                                    if $unsigned {
                                        _mm512_cmp_epu32_mask::<6>(lhs, rhs)
                                    } else {
                                        _mm512_cmp_epi32_mask::<6>(lhs, rhs)
                                    }
                                };

                                // SAFETY: each output value represents 16 rows. The caller
                                // provides four values for each complete 64-row output word.
                                unsafe { output.add(chunk).write(mask ^ ($invert as u16)) };
                            }
                        }};
                    }

                    dispatch_comparison!(comparison, words, compare);
                }
            }
        };
    }

    macro_rules! impl_64 {
        ($t:ty, $unsigned:expr, $float:expr) => {
            #[allow(clippy::cast_possible_truncation)]
            impl SimdCompare for $t {
                const REQUIRES_AVX512BW: bool = false;

                #[target_feature(enable = "avx2")]
                unsafe fn compare_words_avx2(
                    comparison: DenseComparison<'_, Self>,
                    words: &mut [u64],
                ) {
                    macro_rules! compare {
                        ($lhs:expr, $rhs:expr, $lhs_constant:literal, $rhs_constant:literal, $eq:literal, $invert:expr, $words:expr) => {{
                            macro_rules! compare_chunk {
                                ($offset:expr) => {{
                                    let lhs = load_64_or_broadcast!(
                                        $lhs,
                                        $offset,
                                        $lhs_constant,
                                        _mm256_loadu_si256,
                                        _mm256_set1_epi64x,
                                        __m256i
                                    );
                                    let rhs = load_64_or_broadcast!(
                                        $rhs,
                                        $offset,
                                        $rhs_constant,
                                        _mm256_loadu_si256,
                                        _mm256_set1_epi64x,
                                        __m256i
                                    );
                                    let mask = if $eq {
                                        (_mm256_movemask_pd(_mm256_castsi256_pd(
                                            _mm256_cmpeq_epi64(lhs, rhs),
                                        )) as u8)
                                            & 0xf
                                    } else {
                                        let (lhs, rhs) = if $float {
                                            // SAFETY: this method's target-feature contract includes AVX2.
                                            unsafe { (float_key_64(lhs), float_key_64(rhs)) }
                                        } else if $unsigned {
                                            let sign = _mm256_set1_epi64x(i64::MIN);
                                            (
                                                _mm256_xor_si256(lhs, sign),
                                                _mm256_xor_si256(rhs, sign),
                                            )
                                        } else {
                                            (lhs, rhs)
                                        };
                                        (_mm256_movemask_pd(_mm256_castsi256_pd(
                                            _mm256_cmpgt_epi64(lhs, rhs),
                                        )) as u8)
                                            & 0xf
                                    };

                                    mask
                                }};
                            }

                            let output = $words.as_mut_ptr().cast::<u8>();
                            for byte in 0..($words.len() * 8) {
                                let offset = byte * 8;
                                let low = compare_chunk!(offset);
                                let high = compare_chunk!(offset + 4);
                                let mask = (low | (high << 4)) ^ ($invert as u8);

                                // SAFETY: each output byte represents eight rows. The caller
                                // provides eight bytes for each complete 64-row output word.
                                unsafe { output.add(byte).write(mask) };
                            }
                        }};
                    }

                    dispatch_comparison!(comparison, words, compare);
                }

                #[target_feature(enable = "avx512f")]
                unsafe fn compare_words_avx512(
                    comparison: DenseComparison<'_, Self>,
                    words: &mut [u64],
                ) {
                    macro_rules! compare {
                        ($lhs:expr, $rhs:expr, $lhs_constant:literal, $rhs_constant:literal, $eq:literal, $invert:expr, $words:expr) => {{
                            let output = $words.as_mut_ptr().cast::<u8>();
                            for chunk in 0..($words.len() * 8) {
                                let offset = chunk * 8;
                                let lhs = load_64_or_broadcast!(
                                    $lhs,
                                    offset,
                                    $lhs_constant,
                                    _mm512_loadu_si512,
                                    _mm512_set1_epi64,
                                    __m512i
                                );
                                let rhs = load_64_or_broadcast!(
                                    $rhs,
                                    offset,
                                    $rhs_constant,
                                    _mm512_loadu_si512,
                                    _mm512_set1_epi64,
                                    __m512i
                                );
                                let mask = if $eq {
                                    _mm512_cmpeq_epi64_mask(lhs, rhs)
                                } else {
                                    let (lhs, rhs) = if $float {
                                        // SAFETY: this method's target-feature contract includes AVX-512F.
                                        unsafe {
                                            (
                                                float_key_64_avx512(lhs),
                                                float_key_64_avx512(rhs),
                                            )
                                        }
                                    } else {
                                        (lhs, rhs)
                                    };
                                    if $unsigned {
                                        _mm512_cmp_epu64_mask::<6>(lhs, rhs)
                                    } else {
                                        _mm512_cmp_epi64_mask::<6>(lhs, rhs)
                                    }
                                };

                                // SAFETY: each output byte represents eight rows. The caller
                                // provides one complete output word for each 64 rows.
                                unsafe { output.add(chunk).write(mask ^ ($invert as u8)) };
                            }
                        }};
                    }

                    dispatch_comparison!(comparison, words, compare);
                }
            }
        };
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn float_key_16(bits: __m256i) -> __m256i {
        let negative = _mm256_srai_epi16::<15>(bits);
        _mm256_xor_si256(bits, _mm256_srli_epi16::<1>(negative))
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    #[inline]
    unsafe fn float_key_16_avx512(bits: __m512i) -> __m512i {
        let negative = _mm512_srai_epi16::<15>(bits);
        _mm512_xor_si512(bits, _mm512_srli_epi16::<1>(negative))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn float_key_32(bits: __m256i) -> __m256i {
        let negative = _mm256_srai_epi32::<31>(bits);
        _mm256_xor_si256(bits, _mm256_srli_epi32::<1>(negative))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn float_key_32_avx512(bits: __m512i) -> __m512i {
        let negative = _mm512_srai_epi32::<31>(bits);
        _mm512_xor_si512(bits, _mm512_srli_epi32::<1>(negative))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn float_key_64(bits: __m256i) -> __m256i {
        let negative = _mm256_cmpgt_epi64(_mm256_setzero_si256(), bits);
        _mm256_xor_si256(bits, _mm256_srli_epi64::<1>(negative))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn float_key_64_avx512(bits: __m512i) -> __m512i {
        let negative = _mm512_srai_epi64::<63>(bits);
        _mm512_xor_si512(bits, _mm512_srli_epi64::<1>(negative))
    }

    impl_8!(i8, false);
    impl_8!(u8, true);
    impl_16!(i16, false, false);
    impl_16!(u16, true, false);
    impl_16!(f16, false, true);
    impl_32!(i32, false, false);
    impl_32!(u32, true, false);
    impl_32!(f32, false, true);
    impl_64!(i64, false, false);
    impl_64!(u64, true, false);
    impl_64!(f64, false, true);
}

#[cfg(not(target_arch = "x86_64"))]
mod portable {
    use half::f16;

    use super::SimdCompare;

    macro_rules! impl_simd_compare {
        ($($t:ty),+ $(,)?) => {
            $(impl SimdCompare for $t {})+
        };
    }

    impl_simd_compare!(i8, u8, i16, u16, f16, i32, u32, f32, i64, u64, f64);
}

#[cfg(all(test, target_arch = "x86_64"))]
#[allow(clippy::cast_possible_truncation, clippy::tests_outside_test_module)]
mod tests {
    use std::fmt::Debug;

    use half::f16;

    use super::DenseComparison;
    use super::PrimitiveInput;
    use super::SimdCompare;
    use super::apply_op;
    #[cfg(feature = "_test-harness")]
    use super::avx2_available;
    #[cfg(feature = "_test-harness")]
    use super::avx512_available;
    use super::compare_into_words;
    use crate::dtype::NativePType;
    use crate::scalar_fn::fns::operators::CompareOperator;

    const OPS: [CompareOperator; 6] = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Gt,
        CompareOperator::Gte,
        CompareOperator::Lt,
        CompareOperator::Lte,
    ];

    #[cfg(feature = "_test-harness")]
    #[test]
    fn benchmark_dispatch_uses_compiled_features() {
        assert_eq!(avx2_available(), cfg!(target_feature = "avx2"));
        assert_eq!(avx512_available::<i64>(), cfg!(target_feature = "avx512f"));
        assert_eq!(
            avx512_available::<i8>(),
            cfg!(target_feature = "avx512f") && cfg!(target_feature = "avx512bw")
        );
    }

    fn expected<T: NativePType>(
        lhs: PrimitiveInput<'_, T>,
        rhs: PrimitiveInput<'_, T>,
        op: CompareOperator,
    ) -> u64 {
        (0..64).fold(0, |word, bit| {
            // SAFETY: every test slice has exactly 64 elements.
            let lhs = unsafe { lhs.get_unchecked(bit) };
            // SAFETY: see above.
            let rhs = unsafe { rhs.get_unchecked(bit) };
            word | ((apply_op(lhs, rhs, op) as u64) << bit)
        })
    }

    fn assert_simd_matches_scalar<T>(lhs: &[T; 64], rhs: &[T; 64])
    where
        T: SimdCompare + Debug,
    {
        let shapes = [
            (PrimitiveInput::Slice(lhs), PrimitiveInput::Slice(rhs)),
            (PrimitiveInput::Slice(lhs), PrimitiveInput::Constant(rhs[7])),
            (
                PrimitiveInput::Constant(lhs[11]),
                PrimitiveInput::Slice(rhs),
            ),
        ];

        for op in OPS {
            for (lhs, rhs) in shapes {
                let expected = expected(lhs, rhs, op);
                let comparison = DenseComparison::new(lhs, rhs, op);
                if std::arch::is_x86_feature_detected!("avx2") {
                    // SAFETY: runtime detection proves AVX2 support, and the inputs cover 64 rows.
                    let mut actual = [0];
                    unsafe { T::compare_words_avx2(comparison, &mut actual) };
                    assert_eq!(actual[0], expected, "AVX2 {op:?}");
                }
                if std::arch::is_x86_feature_detected!("avx512f")
                    && (!T::REQUIRES_AVX512BW || std::arch::is_x86_feature_detected!("avx512bw"))
                {
                    // SAFETY: runtime detection proves the type's required AVX-512 features, and
                    // the inputs cover 64 rows.
                    let mut actual = [0];
                    unsafe { T::compare_words_avx512(comparison, &mut actual) };
                    assert_eq!(actual[0], expected, "AVX-512 {op:?}");
                }
            }
        }

        assert_word_loop_matches_scalar(lhs, rhs);
    }

    fn assert_word_loop_matches_scalar<T>(lhs_seed: &[T; 64], rhs_seed: &[T; 64])
    where
        T: SimdCompare + Debug,
    {
        for len in [0, 1, 7, 8, 9, 63, 64, 65, 129] {
            let lhs_values = lhs_seed
                .iter()
                .copied()
                .cycle()
                .take(len)
                .collect::<Vec<_>>();
            let rhs_values = rhs_seed
                .iter()
                .copied()
                .cycle()
                .take(len)
                .collect::<Vec<_>>();
            let shapes = [
                (
                    PrimitiveInput::Slice(lhs_values.as_slice()),
                    PrimitiveInput::Slice(rhs_values.as_slice()),
                ),
                (
                    PrimitiveInput::Slice(lhs_values.as_slice()),
                    PrimitiveInput::Constant(rhs_seed[7]),
                ),
                (
                    PrimitiveInput::Constant(lhs_seed[11]),
                    PrimitiveInput::Slice(rhs_values.as_slice()),
                ),
                (
                    PrimitiveInput::Constant(lhs_seed[11]),
                    PrimitiveInput::Constant(rhs_seed[7]),
                ),
            ];

            for op in OPS {
                for (lhs, rhs) in shapes {
                    let mut actual = vec![u64::MAX; len.div_ceil(64)];
                    compare_into_words(lhs, rhs, op, len, &mut actual);

                    let mut expected = vec![0; len.div_ceil(64)];
                    for index in 0..len {
                        // SAFETY: each slice input contains exactly `len` values.
                        let lhs = unsafe { lhs.get_unchecked(index) };
                        // SAFETY: see above.
                        let rhs = unsafe { rhs.get_unchecked(index) };
                        expected[index / 64] |= (apply_op(lhs, rhs, op) as u64) << (index % 64);
                    }

                    assert_eq!(actual, expected, "word loop len={len} op={op:?}");
                }
            }
        }
    }

    #[test]
    fn u8_masks_match_scalar() {
        let lhs = std::array::from_fn(|index| (index as u8).wrapping_mul(37));
        let rhs = std::array::from_fn(|index| (index as u8).wrapping_mul(19).wrapping_add(127));
        assert_simd_matches_scalar(&lhs, &rhs);
    }

    #[test]
    fn i8_masks_match_scalar() {
        let lhs = std::array::from_fn(|index| (index as i8).wrapping_mul(37));
        let rhs = std::array::from_fn(|index| (index as i8).wrapping_mul(-19).wrapping_add(63));
        assert_simd_matches_scalar(&lhs, &rhs);
    }

    #[test]
    fn signed_16_masks_match_scalar() {
        let lhs = std::array::from_fn(|index| (index as i16).wrapping_mul(i16::MAX / 31));
        let rhs = std::array::from_fn(|index| (index as i16).wrapping_mul(i16::MIN / 29));
        assert_simd_matches_scalar(&lhs, &rhs);
    }

    #[test]
    fn unsigned_16_masks_match_scalar() {
        let lhs = std::array::from_fn(|index| (index as u16).wrapping_mul(u16::MAX / 31));
        let rhs = std::array::from_fn(|index| (index as u16).wrapping_mul(u16::MAX / 29));
        assert_simd_matches_scalar(&lhs, &rhs);
    }

    #[test]
    fn signed_32_masks_match_scalar() {
        let lhs = std::array::from_fn(|index| (index as i32).wrapping_mul(i32::MAX / 31));
        let rhs = std::array::from_fn(|index| (index as i32).wrapping_mul(i32::MIN / 29));
        assert_simd_matches_scalar(&lhs, &rhs);
    }

    #[test]
    fn unsigned_32_masks_match_scalar() {
        let lhs = std::array::from_fn(|index| (index as u32).wrapping_mul(u32::MAX / 31));
        let rhs = std::array::from_fn(|index| (index as u32).wrapping_mul(u32::MAX / 29));
        assert_simd_matches_scalar(&lhs, &rhs);
    }

    #[test]
    fn signed_64_masks_match_scalar() {
        let lhs = std::array::from_fn(|index| (index as i64).wrapping_mul(i64::MAX / 31));
        let rhs = std::array::from_fn(|index| (index as i64).wrapping_mul(i64::MIN / 29));
        assert_simd_matches_scalar(&lhs, &rhs);
    }

    #[test]
    fn unsigned_64_masks_match_scalar() {
        let lhs = std::array::from_fn(|index| (index as u64).wrapping_mul(u64::MAX / 31));
        let rhs = std::array::from_fn(|index| (index as u64).wrapping_mul(u64::MAX / 29));
        assert_simd_matches_scalar(&lhs, &rhs);
    }

    #[test]
    fn float_64_total_order_masks_match_scalar() {
        const BITS: [u64; 16] = [
            0xfff8_0000_0000_0001,
            0xfff0_0000_0000_0000,
            0xbff0_0000_0000_0000,
            0x8000_0000_0000_0001,
            0x8000_0000_0000_0000,
            0x0000_0000_0000_0000,
            0x0000_0000_0000_0001,
            0x3ff0_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            0x7ff8_0000_0000_0001,
            0x7fff_ffff_ffff_ffff,
            0xfff8_0000_0000_0000,
            0x4000_0000_0000_0000,
            0xc000_0000_0000_0000,
            0x3fe0_0000_0000_0000,
        ];
        let lhs = std::array::from_fn(|index| f64::from_bits(BITS[index % BITS.len()]));
        let rhs = std::array::from_fn(|index| f64::from_bits(BITS[(index * 7 + 3) % BITS.len()]));
        assert_simd_matches_scalar(&lhs, &rhs);
    }

    #[test]
    fn float_16_total_order_masks_match_scalar() {
        const BITS: [u16; 16] = [
            0xfe01, 0xfc00, 0xbc00, 0x8001, 0x8000, 0x0000, 0x0001, 0x3c00, 0x7c00, 0x7e00, 0x7e01,
            0x7fff, 0xfe00, 0x4000, 0xc000, 0x3800,
        ];
        let lhs = std::array::from_fn(|index| f16::from_bits(BITS[index % BITS.len()]));
        let rhs = std::array::from_fn(|index| f16::from_bits(BITS[(index * 7 + 3) % BITS.len()]));
        assert_simd_matches_scalar(&lhs, &rhs);
    }

    #[test]
    fn float_32_total_order_masks_match_scalar() {
        const BITS: [u32; 16] = [
            0xffc0_0001,
            0xff80_0000,
            0xbf80_0000,
            0x8000_0001,
            0x8000_0000,
            0x0000_0000,
            0x0000_0001,
            0x3f80_0000,
            0x7f80_0000,
            0x7fc0_0000,
            0x7fc0_0001,
            0x7fff_ffff,
            0xffc0_0000,
            0x4000_0000,
            0xc000_0000,
            0x3f00_0000,
        ];
        let lhs = std::array::from_fn(|index| f32::from_bits(BITS[index % BITS.len()]));
        let rhs = std::array::from_fn(|index| f32::from_bits(BITS[(index * 7 + 3) % BITS.len()]));
        assert_simd_matches_scalar(&lhs, &rhs);
    }
}
