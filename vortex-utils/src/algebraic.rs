// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reassociable floating-point arithmetic.
//!
//! IEEE 754 addition and multiplication are not associative, so LLVM must evaluate a float
//! reduction strictly left to right. That serializes the loop on the latency of a single FMA
//! port — a float `sum` runs at roughly one element per 3-4 cycles no matter how wide the
//! machine is.
//!
//! Rust 1.98 stabilized `algebraic_*` operations, which grant LLVM the `reassoc`, `nsz`,
//! `contract`, and `afn` fast-math flags. That is enough for the vectorizer to split a reduction
//! across several accumulators and fold them at the end. Crucially it is *not* `nnan`/`ninf`:
//! NaN and infinity still propagate exactly as IEEE requires, so kernels that define NaN
//! behaviour (`sum` poisoning, `nan_count`) keep their semantics.
//!
//! What does change is the *order* of summation, and therefore the rounding error, so results can
//! differ in the low bits from a strict left-to-right reduction and can vary with the vector
//! width the compiler picked. Only use [`AlgebraicFloat`] where an order-independent result is
//! acceptable — reductions such as `sum`, dot products, and norms. Do not use it where a caller
//! depends on exact IEEE reproducibility.
//!
//! The workspace MSRV is 1.95, below the 1.98 stabilization, so the operations are selected by
//! `build.rs`: it sets `cfg(vortex_float_algebraic)` when the compiling toolchain is new enough
//! and the trait falls back to ordinary IEEE operators otherwise. Both paths compile and produce
//! valid results; only performance and low-bit rounding differ.

use half::f16;

/// Floating-point arithmetic that the compiler may reassociate, contract, and vectorize.
///
/// Implemented for `f16`, `f32`, and `f64`. Generic kernels bound on this trait get the fast
/// reduction on toolchains that support it without spelling out a per-type `match`; see the
/// [module docs](self) for the semantics and the accuracy caveat.
///
/// ```
/// use vortex_utils::algebraic::AlgebraicFloat;
///
/// /// Sums a slice without imposing a summation order.
/// fn sum<T: AlgebraicFloat + Default>(values: &[T]) -> T {
///     values.iter().fold(T::default(), |acc, &v| acc.alg_add(v))
/// }
///
/// assert_eq!(sum(&[1.0f64, 2.0, 3.0]), 6.0);
/// // NaN and infinity still propagate: these are not `nnan`/`ninf` fast-math ops.
/// assert!(sum(&[1.0f64, f64::NAN]).is_nan());
/// ```
pub trait AlgebraicFloat: Copy + sealed::Sealed {
    /// Adds `rhs` to `self`, permitting reassociation with neighbouring operations.
    #[must_use]
    fn alg_add(self, rhs: Self) -> Self;

    /// Subtracts `rhs` from `self`, permitting reassociation with neighbouring operations.
    #[must_use]
    fn alg_sub(self, rhs: Self) -> Self;

    /// Multiplies `self` by `rhs`, permitting reassociation and contraction into an FMA.
    #[must_use]
    fn alg_mul(self, rhs: Self) -> Self;

    /// Divides `self` by `rhs`, permitting the compiler to substitute a reciprocal multiply.
    #[must_use]
    fn alg_div(self, rhs: Self) -> Self;
}

/// Implements [`AlgebraicFloat`] for a primitive float via its inherent `algebraic_*` methods.
#[cfg(vortex_float_algebraic)]
macro_rules! impl_algebraic_native {
    ($($T:ty),*) => {
        $(
            // Clippy reads the crate's declared 1.95 MSRV and cannot see that `build.rs` only
            // enables this arm on 1.98 and later, where the operations are stable.
            #[expect(
                clippy::incompatible_msrv,
                reason = "gated on cfg(vortex_float_algebraic), set only for rustc >= 1.98"
            )]
            impl AlgebraicFloat for $T {
                #[inline(always)]
                fn alg_add(self, rhs: Self) -> Self { self.algebraic_add(rhs) }

                #[inline(always)]
                fn alg_sub(self, rhs: Self) -> Self { self.algebraic_sub(rhs) }

                #[inline(always)]
                fn alg_mul(self, rhs: Self) -> Self { self.algebraic_mul(rhs) }

                #[inline(always)]
                fn alg_div(self, rhs: Self) -> Self { self.algebraic_div(rhs) }
            }
        )*
    };
}

/// MSRV fallback: ordinary IEEE operators, evaluated in the order written.
#[cfg(not(vortex_float_algebraic))]
macro_rules! impl_algebraic_native {
    ($($T:ty),*) => {
        $(
            impl AlgebraicFloat for $T {
                #[inline(always)]
                fn alg_add(self, rhs: Self) -> Self { self + rhs }

                #[inline(always)]
                fn alg_sub(self, rhs: Self) -> Self { self - rhs }

                #[inline(always)]
                fn alg_mul(self, rhs: Self) -> Self { self * rhs }

                #[inline(always)]
                fn alg_div(self, rhs: Self) -> Self { self / rhs }
            }
        )*
    };
}

impl_algebraic_native!(f32, f64);

/// `f16` has no hardware arithmetic on the targets Vortex builds for, so `half` evaluates every
/// operation by widening to `f32`. Mirroring that here keeps `alg_*` bit-identical to the
/// corresponding `half` operator apart from the reassociation itself.
impl AlgebraicFloat for f16 {
    #[inline(always)]
    fn alg_add(self, rhs: Self) -> Self {
        f16::from_f32(self.to_f32().alg_add(rhs.to_f32()))
    }

    #[inline(always)]
    fn alg_sub(self, rhs: Self) -> Self {
        f16::from_f32(self.to_f32().alg_sub(rhs.to_f32()))
    }

    #[inline(always)]
    fn alg_mul(self, rhs: Self) -> Self {
        f16::from_f32(self.to_f32().alg_mul(rhs.to_f32()))
    }

    #[inline(always)]
    fn alg_div(self, rhs: Self) -> Self {
        f16::from_f32(self.to_f32().alg_div(rhs.to_f32()))
    }
}

mod sealed {
    use half::f16;

    /// Prevents downstream implementations of [`AlgebraicFloat`](super::AlgebraicFloat).
    pub trait Sealed {}

    impl Sealed for f16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

#[cfg(test)]
mod tests {
    use half::f16;
    use rstest::rstest;

    use super::AlgebraicFloat;

    /// Reassociation must not change the result of a sum that is exact in binary floating point.
    #[rstest]
    #[case(&[1.0f64, 2.0, 3.0, 4.0], 10.0)]
    #[case(&[0.5f64, 0.25, 0.125], 0.875)]
    #[case(&[], 0.0)]
    fn exact_sums_are_unchanged(#[case] values: &[f64], #[case] expected: f64) {
        let sum = values.iter().fold(0.0f64, |acc, &v| acc.alg_add(v));
        assert_eq!(sum, expected);
    }

    /// The algebraic operations are not `nnan`/`ninf`, so both must still propagate.
    #[test]
    fn nan_and_infinity_propagate() {
        assert!(f64::NAN.alg_add(1.0).is_nan());
        assert!(1.0f64.alg_mul(f64::NAN).is_nan());
        assert!(f32::NAN.alg_sub(1.0).is_nan());
        assert_eq!(1.0f64.alg_div(0.0), f64::INFINITY);
        assert_eq!(f64::INFINITY.alg_add(1.0), f64::INFINITY);
        assert!(f64::INFINITY.alg_sub(f64::INFINITY).is_nan());
    }

    /// Each operation agrees with the corresponding IEEE operator on a single application, where
    /// there is nothing to reassociate against.
    #[test]
    fn single_ops_match_ieee() {
        assert_eq!(3.5f64.alg_add(1.25), 3.5 + 1.25);
        assert_eq!(3.5f64.alg_sub(1.25), 3.5 - 1.25);
        assert_eq!(3.5f64.alg_mul(1.25), 3.5 * 1.25);
        assert_eq!(3.5f64.alg_div(1.25), 3.5 / 1.25);
        assert_eq!(3.5f32.alg_add(1.25), 3.5 + 1.25);
    }

    #[test]
    fn f16_matches_half_operators() {
        let a = f16::from_f32(1.5);
        let b = f16::from_f32(0.25);
        assert_eq!(a.alg_add(b), a + b);
        assert_eq!(a.alg_sub(b), a - b);
        assert_eq!(a.alg_mul(b), a * b);
        assert_eq!(a.alg_div(b), a / b);
    }
}
