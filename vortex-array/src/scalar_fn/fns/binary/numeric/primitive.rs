// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The checked arithmetic one row of a primitive column is computed with.
//!
//! Each operator is a type implementing [`CheckedPrimitiveOp`] at every native width, and each
//! width implements [`CheckedArithmetic`] with the value and the failure test written separately.
//! Keeping them apart is what lets [`row`](super::row) write a value for every row and reduce
//! failure as one bit, so the loop holds no branch and vectorizes.

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

/// One arithmetic operator at one width, as a value and a failure bit.
///
/// The pair rather than an `Option<T>` is what a row can write unconditionally: the value is stored
/// whatever the failure bit says, and a failing row is either masked away as null or turned into a
/// batch error before anything reads it.
pub(super) trait CheckedPrimitiveOp<T: NativePType>: 'static + Sized {
    /// The error reported for a batch in which some valid row failed.
    const ERROR: &'static str;

    /// The result of this operation, paired with whether the row failed.
    fn apply(lhs: T, rhs: T) -> (T, bool);
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedAdd {
    const ERROR: &'static str = "integer overflow in checked add";

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.add_value(rhs), lhs.add_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedSub {
    const ERROR: &'static str = "integer overflow in checked sub";

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.sub_value(rhs), lhs.sub_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedMul {
    const ERROR: &'static str = "integer overflow in checked mul";

    #[inline(always)]
    fn apply(lhs: T, rhs: T) -> (T, bool) {
        (lhs.mul_value(rhs), lhs.mul_error(rhs))
    }
}

impl<T: CheckedArithmetic> CheckedPrimitiveOp<T> for CheckedDiv {
    const ERROR: &'static str = "integer division by zero or overflow in checked div";

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
trait CheckedArithmetic: NativePType {
    fn add_value(self, rhs: Self) -> Self;
    fn add_error(self, rhs: Self) -> bool;
    fn sub_value(self, rhs: Self) -> Self;
    fn sub_error(self, rhs: Self) -> bool;
    fn mul_value(self, rhs: Self) -> Self;
    fn mul_error(self, rhs: Self) -> bool;
    fn div_value(self, rhs: Self) -> Self;
    fn div_error(self, rhs: Self) -> bool;
}

macro_rules! impl_checked_unsigned {
    ($ty:ty,widening_mul: $wide:ty) => {
        impl CheckedArithmetic for $ty {
            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                self > <$ty>::MAX - rhs
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                self < rhs
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            fn mul_error(self, rhs: Self) -> bool {
                (self as $wide) * (rhs as $wide) > <$ty>::MAX as $wide
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                rhs == 0
            }
        }
    };
    ($ty:ty,overflowing_mul) => {
        impl CheckedArithmetic for $ty {
            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                self > <$ty>::MAX - rhs
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                self < rhs
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            fn mul_error(self, rhs: Self) -> bool {
                self.overflowing_mul(rhs).1
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                rhs == 0
            }
        }
    };
}

macro_rules! impl_checked_signed {
    ($ty:ty,widening_mul: $wide:ty) => {
        impl CheckedArithmetic for $ty {
            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                let value = self.wrapping_add(rhs);
                ((self ^ value) & (rhs ^ value)) < 0
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                let value = self.wrapping_sub(rhs);
                ((self ^ rhs) & (self ^ value)) < 0
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            fn mul_error(self, rhs: Self) -> bool {
                let product = (self as $wide) * (rhs as $wide);
                product < <$ty>::MIN as $wide || product > <$ty>::MAX as $wide
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                rhs == 0 || (self == <$ty>::MIN && rhs == -1)
            }
        }
    };
    ($ty:ty,overflowing_mul) => {
        impl CheckedArithmetic for $ty {
            #[inline(always)]
            fn add_value(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn add_error(self, rhs: Self) -> bool {
                let value = self.wrapping_add(rhs);
                ((self ^ value) & (rhs ^ value)) < 0
            }

            #[inline(always)]
            fn sub_value(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn sub_error(self, rhs: Self) -> bool {
                let value = self.wrapping_sub(rhs);
                ((self ^ rhs) & (self ^ value)) < 0
            }

            #[inline(always)]
            fn mul_value(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            fn mul_error(self, rhs: Self) -> bool {
                self.overflowing_mul(rhs).1
            }

            #[inline(always)]
            fn div_value(self, rhs: Self) -> Self {
                self / rhs
            }

            #[inline(always)]
            fn div_error(self, rhs: Self) -> bool {
                rhs == 0 || (self == <$ty>::MIN && rhs == -1)
            }
        }
    };
}

macro_rules! impl_checked_float {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CheckedArithmetic for $ty {
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
                fn mul_error(self, _rhs: Self) -> bool {
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
impl_checked_unsigned!(u64, overflowing_mul);
impl_checked_signed!(i8, widening_mul: i16);
impl_checked_signed!(i16, widening_mul: i32);
impl_checked_signed!(i32, widening_mul: i64);
impl_checked_signed!(i64, overflowing_mul);
impl_checked_float!(f16, f32, f64);
