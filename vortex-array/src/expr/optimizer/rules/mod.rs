// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::expr::BoundExpression;
use crate::expr::ExpressionId;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::cast::Cast;

mod binary;
mod cast;
mod conditional;
mod nulls;
mod structural;

pub(crate) use binary::BinaryBoolean;
pub(crate) use binary::BinaryNullComparison;
pub(crate) use binary::FindBetween;
pub(crate) use cast::CastLiteralOrIdentity;
pub(crate) use conditional::ConstantMask;
pub(crate) use conditional::ConstantZip;
pub(crate) use nulls::CaseWhenToFillNull;
pub(crate) use nulls::RemoveRedundantFillNull;
pub(crate) use structural::GetItemFromPack;
pub(crate) use structural::MergeToPack;
pub(crate) use structural::SelectFromPack;

/// Shared reference to a bound-expression rewrite rule.
pub type BoundExpressionRewriteRuleRef = Arc<dyn BoundExpressionRewriteRule>;

/// An equivalence rewrite for bound expressions with a particular root node implementation.
///
/// The optimizer invokes a rule only when the expression's root ID equals
/// [`Self::expression_id`]. Returning `None` means the rule does not match. A replacement must be
/// semantically equivalent to the input and have exactly the same dtype, including nullability.
/// The optimizer verifies the dtype and rejects unchanged replacements.
pub trait BoundExpressionRewriteRule: Debug + Send + Sync + 'static {
    /// Returns a diagnostic name for this rule.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Returns the expression node ID handled by this rule.
    fn expression_id(&self) -> ExpressionId;

    /// Try to rewrite `expr` to a semantically equivalent bound expression.
    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>>;
}

fn preserve_dtype(replacement: BoundExpression, dtype: &DType) -> VortexResult<BoundExpression> {
    if replacement.dtype() == dtype {
        return Ok(replacement);
    }
    Cast.try_new_bound_expr(dtype.clone(), [replacement])
}
