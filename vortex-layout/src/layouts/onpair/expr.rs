// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::not;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::byte_length::ByteLength;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_error::VortexResult;

/// The minimal set of OnPair children an expression needs for evaluation.
///
/// For example:
///     - `is_null(root())` only needs the validity child.
///     - `byte_length(root())` only needs the uncompressed_lengths and validity children.
///     - `root()` needs the dictionary, the codes, and their offsets as well.
///
/// Declaration order is significant: [`get_necessary_onpair_children`] takes the
/// max over an expression's operands, so variants run cheapest to most expensive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum OnPairChildrenNeeded {
    /// Only the validity child is needed (`is_null` / `is_not_null`).
    Validity,
    /// Only the uncompressed_lengths and validity children are needed (`byte_length`).
    LengthsAndValidity,
    /// All children are needed.
    All,
}

/// The minimal set of OnPair children needed to evaluate `expr`, where `root()` is a field with a
/// string dtype.
pub(super) fn get_necessary_onpair_children(expr: &Expression) -> OnPairChildrenNeeded {
    if is_null_root(expr) {
        return OnPairChildrenNeeded::Validity;
    }

    if is_byte_length_root(expr) {
        return OnPairChildrenNeeded::LengthsAndValidity;
    }

    if is_root(expr) {
        return OnPairChildrenNeeded::All;
    }

    // Otherwise the requirement is the max over the operands. Childless expressions that never
    // touch the column, such as literals, fall back to the cheapest usable child.
    expr.children()
        .iter()
        .map(get_necessary_onpair_children)
        .max()
        .unwrap_or(OnPairChildrenNeeded::Validity)
}

fn is_null_root(expr: &Expression) -> bool {
    (expr.is::<IsNull>() || expr.is::<IsNotNull>())
        && expr.children().len() == 1
        && is_root(expr.child(0))
}

fn is_byte_length_root(expr: &Expression) -> bool {
    expr.is::<ByteLength>() && expr.children().len() == 1 && is_root(expr.child(0))
}

/// Rewrite a validity-class expression so it can be evaluated against the OnPair column's validity
/// bool array (`true` == valid row): `is_not_null(root())` becomes `root()` and `is_null(root())`
/// becomes `not(root())`. All other nodes are rebuilt with rewritten children.
pub(super) fn rewrite_validity_expr(expr: &Expression) -> VortexResult<Expression> {
    if expr.is::<IsNotNull>() && expr.children().len() == 1 && is_root(expr.child(0)) {
        return Ok(root());
    }
    if expr.is::<IsNull>() && expr.children().len() == 1 && is_root(expr.child(0)) {
        return Ok(not(root()));
    }
    let children = expr
        .children()
        .iter()
        .map(rewrite_validity_expr)
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}

/// Rewrite a lengths-class expression so it can be evaluated against the `uncompressed_lengths`
/// child. `byte_length(root())` becomes `root()`. Other references to `root()` are left intact: for
/// lengths-class expressions they can only be validity checks, and the caller gives the lengths
/// array the same validity as the original column.
///
/// The caller must also give the lengths array the dtype [`ByteLength`] would have returned — `u64`
/// carrying the column's nullability — since the rewritten expression's operands are typed against
/// that, not against the `i32` the child is stored as.
pub(super) fn rewrite_lengths_expr(expr: &Expression) -> VortexResult<Expression> {
    if is_byte_length_root(expr) {
        return Ok(root());
    }

    let children = expr
        .children()
        .iter()
        .map(rewrite_lengths_expr)
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::byte_length;
    use vortex_array::expr::cast;
    use vortex_array::expr::eq;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::like;
    use vortex_array::expr::lit;
    use vortex_array::expr::not;
    use vortex_array::expr::root;

    use super::*;

    /// `get_necessary_onpair_children` keys off the deepest child an expression touches; `All` is
    /// the always-correct default for anything not specifically recognized.
    #[rstest]
    // `is_null` / `is_not_null` of the column itself need only validity.
    #[case::is_null(is_null(root()), OnPairChildrenNeeded::Validity)]
    #[case::is_not_null(is_not_null(root()), OnPairChildrenNeeded::Validity)]
    // Compound over validity-only operands stays validity.
    #[case::not_is_null(not(is_null(root())), OnPairChildrenNeeded::Validity)]
    // A column-independent (constant) expression falls to the cheapest usable child.
    #[case::constant(lit(5), OnPairChildrenNeeded::Validity)]
    // `byte_length(root())` needs lengths and validity, but not the dictionary or the codes.
    #[case::byte_length(byte_length(root()), OnPairChildrenNeeded::LengthsAndValidity)]
    // Compound over lengths-only operands stays lengths.
    #[case::byte_length_filter(
        gt(byte_length(root()), lit(1u64)),
        OnPairChildrenNeeded::LengthsAndValidity
    )]
    #[case::cast_byte_length(
        cast(
            byte_length(root()),
            DType::Primitive(PType::I64, Nullability::Nullable),
        ),
        OnPairChildrenNeeded::LengthsAndValidity
    )]
    // A bare column reference needs the dictionary and the codes.
    #[case::bare_root(root(), OnPairChildrenNeeded::All)]
    // Any other fn over the column needs the strings themselves.
    #[case::like_root(like(root(), lit("a%")), OnPairChildrenNeeded::All)]
    #[case::eq_literal(eq(root(), lit("abc")), OnPairChildrenNeeded::All)]
    // `is_null` only short-circuits to validity when its argument is the column itself.
    #[case::is_null_of_derived(is_null(like(root(), lit("a%"))), OnPairChildrenNeeded::All)]
    // `byte_length` likewise only short-circuits on the column itself.
    #[case::byte_length_of_derived(byte_length(like(root(), lit("a%"))), OnPairChildrenNeeded::All)]
    // Max over operands: validity + strings => strings.
    #[case::validity_and_strings(eq(is_null(root()), root()), OnPairChildrenNeeded::All)]
    // Max over operands: lengths + strings => strings.
    #[case::lengths_and_strings(eq(byte_length(root()), root()), OnPairChildrenNeeded::All)]
    fn classify_expr_class(#[case] expr: Expression, #[case] expected: OnPairChildrenNeeded) {
        assert_eq!(get_necessary_onpair_children(&expr), expected);
    }

    /// The validity rewrite maps the two null checks onto a bool array where `true` == valid, and
    /// rebuilds everything else around them.
    #[rstest]
    #[case::is_not_null(is_not_null(root()), root())]
    #[case::is_null(is_null(root()), not(root()))]
    #[case::nested(not(is_null(root())), not(not(root())))]
    #[case::compound(eq(is_null(root()), is_not_null(root())), eq(not(root()), root()))]
    #[case::untouched(lit(5), lit(5))]
    fn rewrite_validity(
        #[case] expr: Expression,
        #[case] expected: Expression,
    ) -> VortexResult<()> {
        assert_eq!(rewrite_validity_expr(&expr)?, expected);
        Ok(())
    }

    /// The lengths rewrite strips `byte_length` off the root and leaves the rest intact.
    #[rstest]
    #[case::bare(byte_length(root()), root())]
    #[case::filter(gt(byte_length(root()), lit(1u64)), gt(root(), lit(1u64)))]
    // A validity check alongside the lengths stays as-is: the lengths array carries the column's
    // validity, so `is_null` evaluates identically against it.
    #[case::with_validity(
        eq(is_null(root()), gt(byte_length(root()), lit(1u64))),
        eq(is_null(root()), gt(root(), lit(1u64)))
    )]
    fn rewrite_lengths(#[case] expr: Expression, #[case] expected: Expression) -> VortexResult<()> {
        assert_eq!(rewrite_lengths_expr(&expr)?, expected);
        Ok(())
    }
}
