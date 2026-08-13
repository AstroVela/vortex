// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Constructors for expressions over this crate's scalar functions.
//!
//! These mirror the constructors in [`vortex_array::expr`] for the functions that live in
//! `vortex-array` itself.

use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::fns::byte_length::ByteLength;
use crate::fns::case_when::CaseWhen;
use crate::fns::case_when::CaseWhenOptions;
use crate::fns::ext_storage::ExtStorage;
use crate::fns::list_length::ListLength;
use crate::fns::list_sum::ListSum;

// ---- CaseWhen ----

/// Creates a CASE WHEN expression with one WHEN/THEN pair and an ELSE value.
pub fn case_when(
    condition: Expression,
    then_value: Expression,
    else_value: Expression,
) -> Expression {
    let options = CaseWhenOptions {
        num_when_then_pairs: 1,
        has_else: true,
    };
    CaseWhen.new_expr(options, [condition, then_value, else_value])
}

/// Creates a bound CASE WHEN expression with one WHEN/THEN pair and an ELSE value.
pub fn bound_case_when(
    condition: BoundExpression,
    then_value: BoundExpression,
    else_value: BoundExpression,
) -> BoundExpression {
    let options = CaseWhenOptions {
        num_when_then_pairs: 1,
        has_else: true,
    };
    CaseWhen
        .try_new_bound_expr(options, [condition, then_value, else_value])
        .vortex_expect("case expressions must have boolean conditions and matching branch dtypes")
}

/// Creates a CASE WHEN expression with one WHEN/THEN pair and no ELSE value.
pub fn case_when_no_else(condition: Expression, then_value: Expression) -> Expression {
    let options = CaseWhenOptions {
        num_when_then_pairs: 1,
        has_else: false,
    };
    CaseWhen.new_expr(options, [condition, then_value])
}

/// Creates a bound CASE WHEN expression with one WHEN/THEN pair and no ELSE value.
pub fn bound_case_when_no_else(
    condition: BoundExpression,
    then_value: BoundExpression,
) -> BoundExpression {
    let options = CaseWhenOptions {
        num_when_then_pairs: 1,
        has_else: false,
    };
    CaseWhen
        .try_new_bound_expr(options, [condition, then_value])
        .vortex_expect("case expressions must have boolean conditions")
}

/// Creates an n-ary CASE WHEN expression from WHEN/THEN pairs and an optional ELSE value.
pub fn nested_case_when(
    when_then_pairs: Vec<(Expression, Expression)>,
    else_value: Option<Expression>,
) -> Expression {
    assert!(
        !when_then_pairs.is_empty(),
        "nested_case_when requires at least one when/then pair"
    );

    let has_else = else_value.is_some();
    let mut children = Vec::with_capacity(when_then_pairs.len() * 2 + usize::from(has_else));
    for (condition, then_value) in &when_then_pairs {
        children.push(condition.clone());
        children.push(then_value.clone());
    }
    if let Some(else_expr) = else_value {
        children.push(else_expr);
    }

    let Ok(num_when_then_pairs) = u32::try_from(when_then_pairs.len()) else {
        vortex_panic!("nested_case_when has too many when/then pairs");
    };
    let options = CaseWhenOptions {
        num_when_then_pairs,
        has_else,
    };
    CaseWhen.new_expr(options, children)
}

/// Creates a bound n-ary CASE WHEN expression from WHEN/THEN pairs and an optional ELSE value.
pub fn bound_nested_case_when(
    when_then_pairs: Vec<(BoundExpression, BoundExpression)>,
    else_value: Option<BoundExpression>,
) -> BoundExpression {
    assert!(
        !when_then_pairs.is_empty(),
        "nested_case_when requires at least one when/then pair"
    );

    let Ok(num_when_then_pairs) = u32::try_from(when_then_pairs.len()) else {
        vortex_panic!("nested_case_when has too many when/then pairs");
    };
    let has_else = else_value.is_some();
    let mut children = Vec::with_capacity(when_then_pairs.len() * 2 + usize::from(has_else));
    for (condition, then_value) in when_then_pairs {
        children.push(condition);
        children.push(then_value);
    }
    if let Some(else_expr) = else_value {
        children.push(else_expr);
    }

    let options = CaseWhenOptions {
        num_when_then_pairs,
        has_else,
    };
    CaseWhen
        .try_new_bound_expr(options, children)
        .vortex_expect("case expressions must have boolean conditions and matching branch dtypes")
}

// ---- ByteLength ----

/// Creates an expression that computes the byte length of each element.
/// This is akin to ANSI SQL OCTET_LENGTH(), or DuckDB's strlen().
///
/// ```rust
/// # use vortex_array::expr::root;
/// # use vortex_scalar_fn_core::exprs::byte_length;
/// let expr = byte_length(root());
/// ```
pub fn byte_length(input: Expression) -> Expression {
    ByteLength.new_expr(EmptyOptions, [input])
}

/// Creates a bound expression that computes each element's byte length.
pub fn bound_byte_length(input: BoundExpression) -> BoundExpression {
    ByteLength
        .try_new_bound_expr(EmptyOptions, [input])
        .vortex_expect("byte-length expressions require a variable-length binary child")
}

// ---- ExtStorage ----

/// Creates an expression that extracts the storage values from an extension array.
///
/// ```rust
/// # use vortex_array::expr::root;
/// # use vortex_scalar_fn_core::exprs::ext_storage;
/// let expr = ext_storage(root());
/// ```
pub fn ext_storage(input: Expression) -> Expression {
    ExtStorage.new_expr(EmptyOptions, [input])
}

/// Creates a bound expression that extracts an extension array's storage values.
pub fn bound_ext_storage(input: BoundExpression) -> BoundExpression {
    ExtStorage
        .try_new_bound_expr(EmptyOptions, [input])
        .vortex_expect("extension-storage expressions require an extension child")
}

// ---- ListLength ----

/// Creates an expression that computes the number of elements in each list
/// for `List` and `FixedSizeList` inputs. This is akin to ANSI SQL `CARDINALITY()`,
/// or DuckDB's `len()`/`array_length()`.
///
/// ```rust
/// # use vortex_array::expr::root;
/// # use vortex_scalar_fn_core::exprs::list_length;
/// let expr = list_length(root());
/// ```
pub fn list_length(input: Expression) -> Expression {
    ListLength.new_expr(EmptyOptions, [input])
}

/// Creates a bound expression that computes the number of elements in each list.
pub fn bound_list_length(input: BoundExpression) -> BoundExpression {
    ListLength
        .try_new_bound_expr(EmptyOptions, [input])
        .vortex_expect("list-length expressions require a list child")
}

// ---- ListSum ----

/// Creates an expression that sums the elements of each list for `List` and
/// `FixedSizeList` inputs, akin to DuckDB's `list_sum()`.
///
/// Follows SQL `SUM` semantics per list: null lists, empty lists, and lists whose elements are
/// all null yield null; null elements are skipped; integer and decimal overflow yields a null
/// value. The result dtype follows `sum`'s widening rules and is always nullable. NaN float
/// elements are skipped by default; see [`list_sum_opts`] for the NaN-including variant.
///
/// ```rust
/// # use vortex_array::expr::root;
/// # use vortex_scalar_fn_core::exprs::list_sum;
/// let expr = list_sum(root());
/// ```
pub fn list_sum(input: Expression) -> Expression {
    ListSum.new_expr(NumericalAggregateOpts::default(), [input])
}

/// Creates a bound expression that sums the elements of each list.
pub fn bound_list_sum(input: BoundExpression) -> BoundExpression {
    ListSum
        .try_new_bound_expr(NumericalAggregateOpts::default(), [input])
        .vortex_expect("list-sum expressions require a numeric list child")
}

/// Creates a [`list_sum`] expression with explicit [`NumericalAggregateOpts`], controlling
/// whether NaN float elements are skipped (the default) or poison the list's sum to NaN.
pub fn list_sum_opts(input: Expression, options: NumericalAggregateOpts) -> Expression {
    ListSum.new_expr(options, [input])
}

/// Creates a bound list-sum expression with explicit aggregate options.
pub fn bound_list_sum_opts(
    input: BoundExpression,
    options: NumericalAggregateOpts,
) -> BoundExpression {
    ListSum
        .try_new_bound_expr(options, [input])
        .vortex_expect("list-sum expressions require a numeric list child")
}

/// Constructors for expressions whose children have already been bound and type-checked.
///
/// These mirror the constructors in [`crate::exprs`] and panic when the supplied children do not
/// form a well-typed expression. Use [`BoundExpression::try_new`] when construction must be
/// fallible.
pub mod bound {
    pub use super::bound_byte_length as byte_length;
    pub use super::bound_case_when as case_when;
    pub use super::bound_case_when_no_else as case_when_no_else;
    pub use super::bound_ext_storage as ext_storage;
    pub use super::bound_list_length as list_length;
    pub use super::bound_list_sum as list_sum;
    pub use super::bound_list_sum_opts as list_sum_opts;
    pub use super::bound_nested_case_when as nested_case_when;
}
