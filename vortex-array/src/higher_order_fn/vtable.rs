// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::BoundLambda;
use crate::expr::BoundLambdaArgs;
use crate::expr::Frame;
use crate::expr::Lambda;
use crate::expr::Scope;
use crate::expr::display::HigherOrderExprDisplay;
use crate::higher_order_fn::HigherOrderFunctionId;
use crate::higher_order_fn::HigherOrderFunctionRef;
use crate::higher_order_fn::LambdaCall;
use crate::higher_order_fn::TypedHigherOrderFunctionInstance;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::validity::Validity;

/// The implementation contract for a higher-order function.
///
/// Unlike a [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable), a higher-order function owns
/// one or more lambdas in addition to its ordinary expression children. It determines the lambda
/// parameter types while binding, then receives the resulting [`BoundLambda`]s at execution.
pub trait HigherOrderFunctionVTable: 'static + Sized + Clone + Send + Sync {
    /// Per-call options for this higher-order function.
    ///
    /// Lambdas and their captures are owned by the expression, not by these options. Options
    /// contain only declarative configuration of the higher-order function itself.
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    /// The globally unique identifier for this higher-order function.
    fn id(&self) -> HigherOrderFunctionId;

    /// The arity of the ordinary expression children.
    fn arity(&self, options: &Self::Options) -> Arity;

    /// The number of lambda arguments.
    fn lambda_arity(&self, options: &Self::Options) -> usize;

    /// The name of an ordinary expression child.
    fn child_name(&self, options: &Self::Options, child_idx: usize) -> ChildName;

    /// Serialize the per-call options.
    ///
    /// The lambda syntax is serialized by [`Expression`](crate::expr::Expression), not here.
    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        _ = options;
        Ok(None)
    }

    /// Deserialize per-call options.
    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_bail!("higher-order function {} is not deserializable", self.id());
    }

    /// Format an expression tree in a human-readable SQL-style format.
    ///
    /// The display exposes ordinary expression children separately from lambda arguments, so a
    /// function may use a custom syntax for either kind. The default prints ordinary children,
    /// then lambdas, followed by non-empty options.
    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &dyn HigherOrderExprDisplay,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}(", self.id())?;
        let mut wrote_argument = false;
        for index in 0..expr.display_children_count() {
            if wrote_argument {
                write!(f, ", ")?;
            }
            Display::fmt(expr.display_child(index), f)?;
            wrote_argument = true;
        }
        for index in 0..expr.display_lambdas_count() {
            if wrote_argument {
                write!(f, ", ")?;
            }
            Display::fmt(expr.display_lambda(index), f)?;
            wrote_argument = true;
        }
        let options = options.to_string();
        if !options.is_empty() {
            if wrote_argument {
                write!(f, ", ")?;
            }
            write!(f, "opts={options}")?;
        }
        write!(f, ")")
    }

    /// Coerce the ordinary input argument types for this higher-order function.
    ///
    /// Lambda bodies are not scalar children and are therefore deliberately excluded. Their
    /// parameter types are established by [`Self::lambda_signatures`] after these input types have
    /// been coerced.
    fn coerce_input_args(
        &self,
        options: &Self::Options,
        args: &[DType],
    ) -> VortexResult<Vec<DType>> {
        _ = options;
        Ok(args.to_vec())
    }

    /// Return the binding signatures for `lambdas` after ordinary children have been bound.
    ///
    /// The generic expression binder uses each signature to establish the lambda's parameter frame
    /// and root scope, then produces [`BoundLambda`] nodes. Keeping this declaration separate from
    /// the binding walk also lets scoped rewrites optimize lambda bodies with the same scope.
    fn lambda_signatures(
        &self,
        options: &Self::Options,
        arg_dtypes: &[DType],
        lambdas: &[&Lambda],
    ) -> VortexResult<Box<[LambdaSignature]>>;

    /// Return the dtype after both ordinary children and lambdas have been bound.
    fn return_dtype(
        &self,
        options: &Self::Options,
        arg_dtypes: &[DType],
        lambdas: BoundLambdaArgs<'_>,
    ) -> VortexResult<DType>;

    /// Return the result validity without executing the function.
    ///
    /// Most functions only know that a nullable result is potentially valid everywhere, but a
    /// function such as `list_transform` can preserve validity from one of its inputs exactly.
    fn validity(
        &self,
        options: &Self::Options,
        args: &[ArrayRef],
        lambdas: BoundLambdaArgs<'_>,
    ) -> VortexResult<Validity> {
        let arg_dtypes = args
            .iter()
            .map(|arg| arg.dtype().clone())
            .collect::<Vec<_>>();
        Ok(Validity::from(
            self.return_dtype(options, &arg_dtypes, lambdas)?
                .nullability(),
        ))
    }

    /// Lower the function over its already-applied ordinary arguments and closed lambdas.
    ///
    /// Lambdas build their bodies as normal scalar-function arrays. Implementations return an
    /// ordinary array graph such as a rebuilt list array, not a dedicated HOF array encoding.
    fn apply(
        &self,
        options: &Self::Options,
        args: &[ArrayRef],
        lambdas: &[LambdaCall],
        execution_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef>;

    /// Whether evaluation of the function can raise a semantic error.
    fn is_fallible(&self, options: &Self::Options, _lambdas: BoundLambdaArgs<'_>) -> bool {
        _ = options;
        true
    }
}

/// The root and parameter dtypes a higher-order function assigns to one lambda argument.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LambdaSignature {
    root_dtype: DType,
    parameter_dtypes: Box<[DType]>,
}

impl LambdaSignature {
    /// Create a lambda signature with its implicit root dtype and parameter dtypes.
    pub fn new(root_dtype: DType, parameter_dtypes: impl Into<Box<[DType]>>) -> Self {
        Self {
            root_dtype,
            parameter_dtypes: parameter_dtypes.into(),
        }
    }

    /// The dtype to which [`crate::expr::root`] resolves in the lambda body.
    pub fn root_dtype(&self) -> &DType {
        &self.root_dtype
    }

    /// The parameter dtypes in declaration order.
    pub fn parameter_dtypes(&self) -> &[DType] {
        &self.parameter_dtypes
    }

    /// Construct this lambda's lexical scope from its parent scope and syntax.
    pub fn scope(&self, lambda: &Lambda, parent: &Scope) -> VortexResult<Scope> {
        vortex_error::vortex_ensure!(
            lambda.params().len() == self.parameter_dtypes.len(),
            "lambda takes {} parameters but its higher-order function supplied {} parameter types",
            lambda.params().len(),
            self.parameter_dtypes.len(),
        );
        let frame = Frame::try_new(
            lambda
                .params()
                .iter()
                .cloned()
                .zip(self.parameter_dtypes.iter().cloned()),
        )?;
        Ok(parent.with_root(self.root_dtype.clone()).push_frame(frame))
    }

    /// Bind `lambda` in this signature's lexical scope.
    pub fn bind(&self, lambda: &Lambda, parent: &Scope) -> VortexResult<BoundLambda> {
        BoundLambda::bind(lambda, &self.scope(lambda, parent)?)
    }
}

/// Empty higher-order-function options.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EmptyOptions;

impl Display for EmptyOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("")
    }
}

/// Factory methods for higher-order function vtables.
pub trait HigherOrderFunctionVTableExt: HigherOrderFunctionVTable {
    /// Bind this vtable with the given options into a [`HigherOrderFunctionRef`].
    fn bind(&self, options: Self::Options) -> HigherOrderFunctionRef {
        TypedHigherOrderFunctionInstance::new(self.clone(), options).erased()
    }

    /// Create a higher-order expression from ordinary children and lambda syntax.
    fn new_expr(
        &self,
        options: Self::Options,
        children: impl IntoIterator<Item = crate::expr::Expression>,
        lambdas: impl IntoIterator<Item = crate::expr::Expression>,
    ) -> crate::expr::Expression {
        Self::try_new_expr(self, options, children, lambdas)
            .vortex_expect("failed to create higher-order expression")
    }

    /// Try to create a higher-order expression from ordinary children and lambda syntax.
    fn try_new_expr(
        &self,
        options: Self::Options,
        children: impl IntoIterator<Item = crate::expr::Expression>,
        lambdas: impl IntoIterator<Item = crate::expr::Expression>,
    ) -> VortexResult<crate::expr::Expression> {
        crate::expr::Expression::try_new_higher_order(self.bind(options), children, lambdas)
    }
}

impl<V: HigherOrderFunctionVTable> HigherOrderFunctionVTableExt for V {}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::fmt::Display;
    use std::fmt::Formatter;

    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_session::registry::CachedId;

    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::lambda;
    use crate::expr::root;
    use crate::expr::transform::coerce_expression;
    use crate::expr::var;

    #[derive(Clone, Debug)]
    struct TestHigherOrder;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct TestOptions(&'static str);

    impl Display for TestOptions {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    impl HigherOrderFunctionVTable for TestHigherOrder {
        type Options = TestOptions;

        fn id(&self) -> HigherOrderFunctionId {
            static ID: CachedId = CachedId::new("vortex.test.higher_order");
            *ID
        }

        fn arity(&self, _options: &Self::Options) -> Arity {
            Arity::Exact(1)
        }

        fn lambda_arity(&self, _options: &Self::Options) -> usize {
            1
        }

        fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
            ChildName::from("input")
        }

        fn coerce_input_args(
            &self,
            _options: &Self::Options,
            args: &[DType],
        ) -> VortexResult<Vec<DType>> {
            let [input] = args else {
                vortex_bail!("test higher-order function expects one input");
            };
            Ok(vec![DType::Primitive(PType::I64, input.nullability())])
        }

        fn lambda_signatures(
            &self,
            _options: &Self::Options,
            arg_dtypes: &[DType],
            lambdas: &[&Lambda],
        ) -> VortexResult<Box<[LambdaSignature]>> {
            let [input] = arg_dtypes else {
                vortex_bail!("test higher-order function expects one input");
            };
            let [_lambda] = lambdas else {
                vortex_bail!("test higher-order function expects one lambda");
            };
            Ok(Box::new([LambdaSignature::new(
                input.clone(),
                [input.clone()],
            )]))
        }

        fn return_dtype(
            &self,
            _options: &Self::Options,
            arg_dtypes: &[DType],
            _lambdas: BoundLambdaArgs<'_>,
        ) -> VortexResult<DType> {
            let [input] = arg_dtypes else {
                vortex_bail!("test higher-order function expects one input");
            };
            Ok(input.clone())
        }

        fn apply(
            &self,
            _options: &Self::Options,
            _args: &[ArrayRef],
            _lambdas: &[LambdaCall],
            _execution_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            vortex_bail!("test higher-order function is not executable")
        }
    }

    #[test]
    fn options_participate_in_sql_display_and_identity() -> VortexResult<()> {
        let expr = TestHigherOrder.try_new_expr(
            TestOptions("strategy=checked"),
            [root()],
            [lambda(["value"], var("value"))?],
        )?;

        assert_eq!(
            expr.to_string(),
            "vortex.test.higher_order($, (value) -> $value, opts=strategy=checked)"
        );
        assert_ne!(
            TestHigherOrder.bind(TestOptions("left")),
            TestHigherOrder.bind(TestOptions("right"))
        );
        Ok(())
    }

    #[test]
    fn coercion_applies_only_to_ordinary_higher_order_inputs() -> VortexResult<()> {
        let expr = TestHigherOrder.try_new_expr(
            TestOptions("strategy=checked"),
            [crate::expr::lit(1_i32)],
            [lambda(["value"], var("value"))?],
        )?;

        let coerced = coerce_expression(expr, &DType::Bool(Nullability::NonNullable))?;
        assert_eq!(
            coerced
                .child(0)
                .return_dtype(&DType::Bool(Nullability::NonNullable))?,
            DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_eq!(coerced.lambdas()[0].to_string(), "(value) -> $value");
        let bound = coerced.bind(&DType::Bool(Nullability::NonNullable))?;
        let lambda = bound.lambdas()[0]
            .as_lambda()
            .vortex_expect("higher-order lambdas bind as lambda nodes");
        assert_eq!(
            lambda.param_dtypes(),
            &[DType::Primitive(PType::I64, Nullability::NonNullable)]
        );
        Ok(())
    }
}
