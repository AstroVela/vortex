// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

pub use kernel::*;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::NullHandling;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::StrictScalarFnVTable;
use crate::validity::Validity;

/// Expression that logically inverts boolean values.
///
/// This is a [`StrictScalarFnVTable`] rather than a row function: the kernel is one `!` per
/// *word* of the packed bit buffer, not one per row. Null propagation, constant folding, validity
/// and options serde come from the strict lifting; `execute_strict` only has to negate bits.
#[derive(Clone)]
pub struct Not;

impl StrictScalarFnVTable for Not {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.not");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for Not expression"),
        }
    }

    fn return_element_dtype(
        &self,
        _options: &Self::Options,
        arg_dtypes: &[DType],
    ) -> VortexResult<DType> {
        vortex_ensure!(
            matches!(arg_dtypes[0], DType::Bool(_)),
            "Not expression expects a boolean child, got: {}",
            arg_dtypes[0],
        );
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn null_handling(&self, _options: &Self::Options) -> NullHandling {
        NullHandling::Dense
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }

    fn execute_strict(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        // The strict lifting applies the input's validity to whatever we return, so keeping the
        // input's declared nullability here saves it a cast node.
        let nullability = input.dtype().nullability();
        // One `!` per `u64` word, and in place when the bits are not shared. Executing into
        // `BoolArray` is a downcast when the input is already canonical.
        let bits = input.execute::<BoolArray>(ctx)?.into_bit_buffer();
        Ok(BoolArray::new(!bits, Validity::from(nullability)).into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::BoolArray;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::bool::BoolArrayExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::not;
    use crate::expr::root;
    use crate::expr::test_harness;

    #[test]
    fn is_strict() {
        assert!(not(root()).signature().is_strict());
    }

    #[test]
    fn preserves_nulls() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let input = BoolArray::from_iter([Some(false), None, Some(true)]).into_array();

        let result = input.apply(&not(root()))?;

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(true), None, Some(false)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn invert_booleans() {
        let mut ctx = array_session().create_execution_ctx();
        let not_expr = not(root());
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        let result = bools
            .into_array()
            .apply(&not_expr)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        assert_eq!(
            result.to_bit_buffer().iter().collect::<Vec<_>>(),
            vec![true, false, true, true, false, false]
        );
    }

    #[test]
    fn test_display_order_of_operations() {
        let a = not(get_item("a", root()));
        let b = get_item("a", not(root()));
        assert_ne!(a.to_string(), b.to_string());
        assert_eq!(a.to_string(), "vortex.not($.a)");
        assert_eq!(b.to_string(), "vortex.not($).a");
    }

    #[test]
    fn dtype() {
        let not_expr = not(root());
        let dtype = DType::Bool(Nullability::NonNullable);
        assert_eq!(
            not_expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );

        let dtype = test_harness::struct_dtype();
        assert_eq!(
            not(col("bool1")).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }
}
