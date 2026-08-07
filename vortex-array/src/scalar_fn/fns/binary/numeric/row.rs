// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The primitive arithmetic operators as a [`RowFn`].
//!
//! [`Binary`] keeps its ID, options serialization, and semantic contracts. It delegates only the
//! execution of primitive `Add`, `Sub`, `Mul`, and `Div` to [`NumericBinary`]. The helper is not
//! registered and appears in no serialized expression.
//!
//! Shared lifting owns decoding, constant handling, output allocation, nullability, validity, and
//! nullable retry. The declaration below contains only type dispatch and the per-row operation.
//!
//! [`Binary`]: crate::scalar_fn::fns::binary::Binary

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use super::primitive::CheckedAdd;
use super::primitive::CheckedDiv;
use super::primitive::CheckedMul;
use super::primitive::CheckedPrimitiveOp;
use super::primitive::CheckedSub;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::scalar::NumericOperator;
use crate::scalar_fn::RowFn;
use crate::scalar_fn::RowVisitor;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::VecExecutionArgs;

/// Execute a numeric operation between two primitive-typed arrays.
pub(super) fn execute_numeric_primitive(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let args = VecExecutionArgs::new(vec![lhs.clone(), rhs.clone()], lhs.len());

    ScalarFnVTable::execute(&NumericBinary, &op, &args, ctx)
}

/// The primitive arithmetic operators as a row function.
#[derive(Clone)]
struct NumericBinary;

impl RowFn for NumericBinary {
    type Options = NumericOperator;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    // Fallibility is declared before dispatch knows the primitive width. The float widths inherit
    // this conservative declaration at no execution cost.
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.numeric_binary");
        *ID
    }

    fn dispatch<Visitor: RowVisitor>(
        &self,
        op: &Self::Options,
        args: &[DType],
        visitor: Visitor,
    ) -> VortexResult<Visitor::Out> {
        let ptype = operand_ptype(args)?;

        match_each_native_ptype!(ptype, |Primitive| {
            match op {
                NumericOperator::Add => visit_checked::<Primitive, CheckedAdd, Visitor>(visitor),
                NumericOperator::Sub => visit_checked::<Primitive, CheckedSub, Visitor>(visitor),
                NumericOperator::Mul => visit_checked::<Primitive, CheckedMul, Visitor>(visitor),
                NumericOperator::Div => visit_checked::<Primitive, CheckedDiv, Visitor>(visitor),
            }
        })
    }
}

/// Return the primitive width selected by the left operand.
///
/// The visited `(Primitive, Primitive)` tuple validates both operands against this width.
fn operand_ptype(args: &[DType]) -> VortexResult<PType> {
    let lhs = args
        .first()
        .ok_or_else(|| vortex_err!("a numeric operator takes two operands, got none"))?;

    PType::try_from(lhs)
}

/// Visit two primitive columns and defer one OR-reducible failure word per row.
fn visit_checked<Primitive, Operator, Visitor>(visitor: Visitor) -> VortexResult<Visitor::Out>
where
    Primitive: NativePType,
    Operator: CheckedPrimitiveOp<Primitive>,
    Visitor: RowVisitor,
{
    visitor.visit_prepared_deferred::<(Primitive, Primitive), Primitive, _, Operator::Failure>(
        |_| (),
        |&(), (lhs, rhs)| Operator::apply(lhs, rhs),
        |failure| {
            if failure != <Operator::Failure as Default>::default() {
                return Err(vortex_err!(InvalidArgument: "{}", Operator::ERROR));
            }

            Ok(())
        },
    )
}
