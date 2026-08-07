// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The checked arithmetic one row of a primitive column is computed with.
//!
//! Each operator is a type implementing [`CheckedPrimitiveOp`] at every native width, and each
//! width implements [`CheckedArithmetic`] with the value and failure evidence written separately.
//! Keeping them apart is what lets [`row`](super::row) write a value for every row and reduce the
//! evidence without a branch, so the loop vectorizes.

use std::ops::BitOrAssign;

use crate::dtype::NativePType;
use crate::dtype::half::f16;

/// Checked addition, failing on integer overflow.
pub(super) struct CheckedAdd;

/// Checked subtraction, failing on integer overflow.
pub(super) struct CheckedSub;

/// Checked multiplication, failing on integer overflow.
pub(super) struct CheckedMul;

/// Checked division, failing on integer division by zero and on `MIN / -1`.
pub(super) struct CheckedDiv;

/// Evidence that some row failed, in a form that OR-reduces across the batch.
///
/// A plain `bool` is the obvious choice and the right one for most operations. Unsigned
/// multiplication is the exception: deriving a `bool` from the widened product costs a comparison,
/// and LLVM rewrites that comparison plus the product into `llvm.umul.with.overflow`, which has no
/// vector form and scalarizes the whole loop. Carrying the discarded high half instead means the row
/// never compares, so the multiply stays a widening vector multiply and the reduction stays a
/// vector OR. **The width must not exceed the element's**, or the reduction becomes the loop's
/// bottleneck instead of the arithmetic.
pub(super) trait Failure: Copy + Default + PartialEq + BitOrAssign {}

impl<T: Copy + Default + PartialEq + BitOrAssign> Failure for T {}

/// One arithmetic operator at one width, as a value and its failure evidence.
///
/// The pair rather than an `Option<T>` is what a row can write unconditionally: the value is stored
/// whatever the evidence says, and a failing row is either masked away as null or turned into a
/// batch error before anything reads it.
pub(super) trait CheckedPrimitiveOp<T: NativePType>: 'static + Sized {
    /// The error reported for a batch in which some valid row failed.
    const ERROR: &'static str;

    /// How this operation reports a failing row. See [`Failure`].
    type Failure: Failure;

    /// The result of this operation, paired with evidence of whether the row failed.
    fn apply(lhs: T, rhs: T) -> (T, Self::Failure);
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedAdd {
    const ERROR: &'static str = "integer overflow in checked add";

    type Failure = bool;

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.add_value(rhs), lhs.add_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedSub {
    const ERROR: &'static str = "integer overflow in checked sub";

    type Failure = bool;

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.sub_value(rhs), lhs.sub_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedMul {
    const ERROR: &'static str = "integer overflow in checked mul";

    type Failure = T::MulFailure;

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, T::MulFailure) {
        (lhs.mul_value(rhs), lhs.mul_failure(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedDiv {
    const ERROR: &'static str = "integer division by zero or overflow in checked div";

    type Failure = bool;

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        let failed = lhs.div_error(rhs);
        let value = if failed {
            T::default()
        } else {
            lhs.div_value(rhs)
        };
        (value, failed)
    }
}

/// The per-width arithmetic behind [`CheckedPrimitiveOp`], with each operation split into the value
/// it produces and whether producing it failed.
///
/// Every `_value` method **must** be total: it is called for rows behind nulls, whose operands are
/// arbitrary, so it may not panic or trap. Integer division is the one that needs care, and
/// [`CheckedDiv`] supplies the default instead of dividing when the divisor is rejected.
pub(super) trait CheckedArithmetic: NativePType {
    /// How multiplication reports a failing row.
    ///
    /// `Self` for the unsigned widths that have a widening multiply, so the row can hand back the
    /// discarded high half rather than comparing. `bool` everywhere else: the narrow signed widths
    /// already vectorize through a two-sided range check, floats never overflow, and the 64-bit
    /// widths use a full-width evidence word.
    type MulFailure: Failure;

    fn add_value(self, rhs: Self) -> Self;
    fn add_error(self, rhs: Self) -> bool;
    fn sub_value(self, rhs: Self) -> Self;
    fn sub_error(self, rhs: Self) -> bool;
    fn mul_value(self, rhs: Self) -> Self;
    fn mul_failure(self, rhs: Self) -> Self::MulFailure;
    fn div_value(self, rhs: Self) -> Self;
    fn div_error(self, rhs: Self) -> bool;
}

/// The integer arithmetic every width shares, given the two things that actually differ between
/// them: how multiplication reports a failing row, and how add/sub/div detect one.
macro_rules! impl_checked_integer {
    (
        $ty:ty,
        add_error: |$add_lhs:ident, $add_rhs:ident| $add_error:expr,
        sub_error: |$sub_lhs:ident, $sub_rhs:ident| $sub_error:expr,
        div_error: |$div_lhs:ident, $div_rhs:ident| $div_error:expr,
        mul_failure: $(#[$mul_failure_attr:meta])* $mul_failure_ty:ty
            = |$mf_lhs:ident, $mf_rhs:ident| $mul_failure:expr,
    ) => {
        impl CheckedArithmetic for $ty {
            type MulFailure = $mul_failure_ty;

            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                let ($add_lhs, $add_rhs) = (self, rhs);
                $add_error
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                let ($sub_lhs, $sub_rhs) = (self, rhs);
                $sub_error
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            $(#[$mul_failure_attr])*
            fn mul_failure(self, rhs: Self) -> $mul_failure_ty {
                let ($mf_lhs, $mf_rhs) = (self, rhs);
                $mul_failure
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                let ($div_lhs, $div_rhs) = (self, rhs);
                $div_error
            }
        }
    };
}

/// The unsigned widths. The discarded high half of the widened product is the failure evidence,
/// and costs none of the comparison LLVM folds into `umul.with.overflow`.
macro_rules! impl_checked_unsigned {
    ($ty:ty, widening_mul: $wide:ty) => {
        impl_checked_integer!(
            $ty,
            add_error: |lhs, rhs| lhs > <$ty>::MAX - rhs,
            sub_error: |lhs, rhs| lhs < rhs,
            div_error: |_lhs, rhs| rhs == 0,
            mul_failure: $ty = |lhs, rhs| (((lhs as $wide) * (rhs as $wide)) >> <$ty>::BITS) as $ty,
        );
    };
}

/// The signed widths. The narrow widths report a two-sided range check as `bool`; the 64-bit width
/// reports the discarded high half as a word so deriving the evidence does not scalarize the loop.
macro_rules! impl_checked_signed {
    ($ty:ty, widening_mul: $wide:ty) => {
        impl_checked_signed!($ty, mul_failure: bool = |lhs, rhs| {
            let product = (lhs as $wide) * (rhs as $wide);
            product < <$ty>::MIN as $wide || product > <$ty>::MAX as $wide
        });
    };
    ($ty:ty, high_half_mul: $wide:ty => $failure:ty) => {
        impl_checked_signed!($ty, mul_failure: #[expect(
            clippy::cast_possible_truncation,
            reason = "the truncated half is the result, and the discarded half is the evidence"
        )] $failure = |lhs, rhs| {
            let wide = (lhs as $wide) * (rhs as $wide);
            let kept = wide as $ty;
            let discarded = (wide >> <$ty>::BITS) as $ty;

            (discarded ^ (kept >> (<$ty>::BITS - 1))) as $failure
        });
    };
    (
        $ty:ty,
        mul_failure: $(#[$mul_failure_attr:meta])* $mul_failure_ty:ty
            = |$lhs:ident, $rhs:ident| $mul_failure:expr
    ) => {
        impl_checked_integer!(
            $ty,
            add_error: |lhs, rhs| {
                let value = lhs.wrapping_add(rhs);
                ((lhs ^ value) & (rhs ^ value)) < 0
            },
            sub_error: |lhs, rhs| {
                let value = lhs.wrapping_sub(rhs);
                ((lhs ^ rhs) & (lhs ^ value)) < 0
            },
            div_error: |lhs, rhs| rhs == 0 || (lhs == <$ty>::MIN && rhs == -1),
            mul_failure: $(#[$mul_failure_attr])* $mul_failure_ty = |$lhs, $rhs| $mul_failure,
        );
    };
}

macro_rules! impl_checked_float {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CheckedArithmetic for $ty {
                type MulFailure = bool;

                #[inline(always)]
                fn add_value(self, rhs: Self) -> Self {
                    self + rhs
                }

                #[inline(always)]
                fn add_error(self, _rhs: Self) -> bool {
                    false
                }

                #[inline(always)]
                fn sub_value(self, rhs: Self) -> Self {
                    self - rhs
                }

                #[inline(always)]
                fn sub_error(self, _rhs: Self) -> bool {
                    false
                }

                #[inline(always)]
                fn mul_value(self, rhs: Self) -> Self {
                    self * rhs
                }

                #[inline(always)]
                fn mul_failure(self, _rhs: Self) -> bool {
                    false
                }

                #[inline(always)]
                fn div_value(self, rhs: Self) -> Self {
                    self / rhs
                }

                #[inline(always)]
                fn div_error(self, _rhs: Self) -> bool {
                    false
                }
            }
        )+
    };
}

impl_checked_unsigned!(u8, widening_mul: u16);
impl_checked_unsigned!(u16, widening_mul: u32);
impl_checked_unsigned!(u32, widening_mul: u64);
impl_checked_unsigned!(u64, widening_mul: u128);
impl_checked_signed!(i8, widening_mul: i16);
impl_checked_signed!(i16, widening_mul: i32);
impl_checked_signed!(i32, widening_mul: i64);
impl_checked_signed!(i64, high_half_mul: i128 => u64);
impl_checked_float!(f16, f32, f64);

#[cfg(test)]
mod tests {
    use super::CheckedArithmetic;

    /// Values whose pairwise products are worth probing: the saturating boundaries, the sign-change
    /// pivots, and a spread of magnitudes that straddles the 64-bit split.
    const PROBES: &[i64] = &[
        0,
        1,
        -1,
        2,
        -2,
        3,
        i64::MIN,
        i64::MIN + 1,
        i64::MAX,
        i64::MAX - 1,
        1 << 31,
        1 << 32,
        1 << 62,
        -(1 << 62),
        0x7FFF_FFFF,
        -0x8000_0000,
    ];

    /// Every `mul_failure` implementation is either a bit trick or a two-sided range check, so
    /// hold each against `checked_mul`, whose `None` is the definition of overflow.
    #[track_caller]
    fn assert_agrees_with_checked_mul<T: CheckedArithmetic>(lhs: T, rhs: T, reference: Option<T>) {
        let failed = lhs.mul_failure(rhs) != <T::MulFailure as Default>::default();

        assert_eq!(failed, reference.is_none(), "{lhs:?} * {rhs:?}");
    }

    #[test]
    fn mul_failure_agrees_with_checked_mul_at_64_bits() {
        for &lhs in PROBES {
            for &rhs in PROBES {
                assert_agrees_with_checked_mul(lhs, rhs, lhs.checked_mul(rhs));

                let (lhs, rhs) = (lhs as u64, rhs as u64);
                assert_agrees_with_checked_mul(lhs, rhs, lhs.checked_mul(rhs));
            }
        }
    }

    /// The 8-bit widths are cheap enough to check exhaustively, pinning the unsigned shift and the
    /// signed range check against every product that exists.
    #[test]
    fn mul_failure_agrees_with_checked_mul_exhaustively_at_8_bits() {
        for lhs in u8::MIN..=u8::MAX {
            for rhs in u8::MIN..=u8::MAX {
                assert_agrees_with_checked_mul(lhs, rhs, lhs.checked_mul(rhs));
            }
        }

        for lhs in i8::MIN..=i8::MAX {
            for rhs in i8::MIN..=i8::MAX {
                assert_agrees_with_checked_mul(lhs, rhs, lhs.checked_mul(rhs));
            }
        }
    }
}
