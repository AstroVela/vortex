// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::MaskedArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::expr::root;
use crate::scalar::Scalar;
use crate::scalar_fn::row::execute::row_null_handling;
use crate::scalar_fn::*;

/// Builds `scalar_fn` over `args` and executes it end to end, which is what every test below does.
fn apply<F: RowFn<Options = EmptyOptions>>(
    scalar_fn: F,
    args: impl IntoIterator<Item = ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let args = args.into_iter().collect::<Vec<_>>();
    let rows = args.first().map_or(0, |arg| arg.len());

    Ok(scalar_fn
        .try_new_array(rows, EmptyOptions, args)?
        .execute::<Canonical>(ctx)?
        .into_array())
}

/// A binary row function over fixed primitive types: `hypot(x, y)`.
#[derive(Clone)]
struct Hypot;

impl RowFn for Hypot {
    type Options = EmptyOptions;
    type ArgsWitness = (f64, f64);
    type RetWitness = f64;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.hypot");
        *ID
    }

    fn arg_name(&self, idx: usize) -> ChildName {
        ChildName::from(["x", "y"][idx])
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit::<(f64, f64), f64>(|(x, y)| x.hypot(y))
    }
}

/// A unary row function over strings: uppercased text, exercising [`Bytes`] input and
/// [`String`] output.
#[derive(Clone)]
struct Shout;

impl RowFn for Shout {
    type Options = EmptyOptions;
    type ArgsWitness = (Bytes,);
    type RetWitness = String;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.shout");
        *ID
    }

    fn arg_name(&self, _idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit::<(Bytes,), String>(|(text,)| String::from_utf8_lossy(text).to_uppercase())
    }
}

/// A fallible row function: integer division, undefined at a zero divisor.
#[derive(Clone)]
struct CheckedDiv;

impl RowFn for CheckedDiv {
    type Options = EmptyOptions;
    type ArgsWitness = (i64, i64);
    type RetWitness = VortexResult<i64>;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.checked_div");
        *ID
    }

    fn arg_name(&self, idx: usize) -> ChildName {
        ChildName::from(["lhs", "rhs"][idx])
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit::<(i64, i64), VortexResult<i64>>(|(lhs, rhs)| {
            if rhs == 0 {
                vortex_bail!("division by zero");
            }
            Ok(lhs / rhs)
        })
    }
}

#[test]
fn hypot_columns() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let x = buffer![3.0f64, 5.0].into_array();
    let y = buffer![4.0f64, 12.0].into_array();

    let result = apply(Hypot, [x, y], &mut ctx)?;

    assert_arrays_eq!(result, PrimitiveArray::from_iter([5.0f64, 13.0]), &mut ctx);
    Ok(())
}

#[test]
fn hypot_propagates_nulls_and_constants() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let x = PrimitiveArray::from_option_iter([Some(3.0f64), None, Some(8.0)]).into_array();
    let y = ConstantArray::new(Scalar::from(4.0f64), 3).into_array();

    let result = apply(Hypot, [x, y], &mut ctx)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(5.0f64), None, Some((80.0f64).sqrt())]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn shout_strings() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input =
        VarBinViewArray::from_iter_nullable_str([Some("hello"), None, Some("Vortex")]).into_array();

    let result = apply(Shout, [input], &mut ctx)?;

    let expected =
        VarBinViewArray::from_iter_nullable_str([Some("HELLO"), None, Some("VORTEX")]).into_array();
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn display_names_the_function_id() {
    let expr = Hypot.new_expr(EmptyOptions, [root(), root()]);
    assert_eq!(expr.to_string(), "vortex.test.hypot($, $)");
}

mod nullable_outputs {
    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    struct NullableI64(i64);

    impl OutputElement for NullableI64 {
        fn element_dtype() -> DType {
            DType::Primitive(PType::I64, Nullability::Nullable)
        }

        fn build(values: Vec<Self>) -> ArrayRef {
            PrimitiveArray::from_option_iter(values.into_iter().map(|value| Some(value.0)))
                .into_array()
        }

        fn placeholder() -> Self {
            Self(0)
        }
    }

    struct NullableSink(usize);

    impl OutputSink for NullableSink {
        type Rows<'a> = usize;
        type Row<'a> = ();

        fn sink_dtype(_args: &[DType]) -> VortexResult<DType> {
            Ok(DType::Primitive(PType::I64, Nullability::Nullable))
        }

        fn with_capacity(rows: usize, _dtype: &DType) -> VortexResult<Self> {
            Ok(Self(rows))
        }

        fn rows(&mut self) -> Self::Rows<'_> {
            self.0
        }

        fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
            *rows == row_count
        }

        fn row<'a>(_rows: &'a mut Self::Rows<'_>, _index: usize) -> Self::Row<'a> {}

        fn finish(self) -> VortexResult<ArrayRef> {
            Ok(PrimitiveArray::from_option_iter(Vec::<Option<i64>>::new()).into_array())
        }
    }

    #[derive(Clone)]
    struct NullableElementFn;

    impl RowFn for NullableElementFn {
        type Options = EmptyOptions;
        type ArgsWitness = (i64,);
        type RetWitness = NullableI64;

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("vortex.test.nullable_element");
            *ID
        }

        fn arg_name(&self, _idx: usize) -> ChildName {
            ChildName::from("input")
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::Out> {
            visitor.visit::<(i64,), NullableI64>(|(value,)| NullableI64(value))
        }
    }

    #[derive(Clone)]
    struct NullableSinkFn;

    impl RowFn for NullableSinkFn {
        type Options = EmptyOptions;
        type ArgsWitness = (i64,);
        type RetWitness = ();

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("vortex.test.nullable_sink");
            *ID
        }

        fn arg_name(&self, _idx: usize) -> ChildName {
            ChildName::from("input")
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::Out> {
            visitor.visit_into::<(i64,), NullableSink, ()>(|_, ()| {})
        }
    }

    #[test]
    fn nullable_element_dtype_is_rejected() {
        let input = DType::Primitive(PType::I64, Nullability::NonNullable);
        let error =
            ScalarFnVTable::return_dtype(&NullableElementFn, &EmptyOptions, &[input]).unwrap_err();

        assert!(error.to_string().contains("non-nullable dtype"), "{error}");
    }

    #[test]
    fn nullable_sink_dtype_is_rejected() {
        let input = DType::Primitive(PType::I64, Nullability::NonNullable);
        let error =
            ScalarFnVTable::return_dtype(&NullableSinkFn, &EmptyOptions, &[input]).unwrap_err();

        assert!(error.to_string().contains("non-nullable dtype"), "{error}");
    }
}

#[derive(Clone)]
struct WrongLength;

impl RowFn for WrongLength {
    type Options = EmptyOptions;
    type ArgsWitness = (i64,);
    type RetWitness = i64;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.wrong_length");
        *ID
    }

    fn arg_name(&self, _idx: usize) -> ChildName {
        ChildName::from("input")
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit::<(i64,), i64>(|(value,)| value)
    }

    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        _args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(PrimitiveArray::from_iter([0i64]).into_array()))
    }
}

#[test]
fn kernel_result_length_is_validated() {
    let mut ctx = array_session().create_execution_ctx();
    let input = buffer![1i64, 2, 3].into_array();

    let error = apply(WrongLength, [input], &mut ctx).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("produced 1 rows for 3 input rows"),
        "{error}"
    );
}

#[derive(Clone)]
struct FortyTwo;

impl RowFn for FortyTwo {
    type Options = EmptyOptions;
    type ArgsWitness = ();
    type RetWitness = i64;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.forty_two");
        *ID
    }

    fn arg_name(&self, _idx: usize) -> ChildName {
        ChildName::from("unused")
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit::<(), i64>(|()| 42)
    }
}

#[test]
fn nullary_row_fn_executes_requested_rows() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let result = FortyTwo
        .try_new_array(3, EmptyOptions, [])?
        .execute::<Canonical>(&mut ctx)?;

    assert_arrays_eq!(result, PrimitiveArray::from_iter([42i64; 3]), &mut ctx);
    Ok(())
}

#[derive(Clone)]
struct SumFour;

impl RowFn for SumFour {
    type Options = EmptyOptions;
    type ArgsWitness = (i64, i64, i64, i64);
    type RetWitness = i64;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.sum_four");
        *ID
    }

    fn arg_name(&self, idx: usize) -> ChildName {
        ChildName::from(["a", "b", "c", "d"][idx])
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit::<(i64, i64, i64, i64), i64>(|(a, b, c, d)| a + b + c + d)
    }
}

#[test]
fn four_argument_row_fn_executes() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let result = apply(
        SumFour,
        [
            buffer![1i64, 2].into_array(),
            buffer![10i64, 20].into_array(),
            buffer![100i64, 200].into_array(),
            buffer![1000i64, 2000].into_array(),
        ],
        &mut ctx,
    )?;

    assert_arrays_eq!(result, PrimitiveArray::from_iter([1111i64, 2222]), &mut ctx);
    Ok(())
}

#[test]
fn tuples_are_supported_through_arity_twelve() {
    type TwelveI64s = (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64);

    assert_eq!(<() as ElementTuple>::ARITY, 0);
    assert_eq!(<TwelveI64s as ElementTuple>::ARITY, 12);
}

#[test]
fn ret_type_decides_fallibility() {
    assert!(!ScalarFnVTable::is_fallible(&Hypot, &EmptyOptions));
    assert!(ScalarFnVTable::is_fallible(&CheckedDiv, &EmptyOptions));
}

#[test]
fn fallible_apply_propagates_its_error() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = buffer![10i64, 10].into_array();
    let rhs = buffer![2i64, 0].into_array();

    let error = apply(CheckedDiv, [lhs, rhs], &mut ctx)
        .expect_err("a zero divisor must fail the execution");

    assert!(
        error.to_string().contains("division by zero"),
        "unexpected error: {error}"
    );
    Ok(())
}

/// The divisor's null slot holds a zero, which a dense pass would divide by. Filtering keeps the
/// fallible kernel away from it.
#[test]
fn fallible_apply_never_sees_rows_behind_nulls() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = buffer![10i64, 10].into_array();
    let rhs = PrimitiveArray::from_option_iter([Some(2i64), None]).into_array();

    let result = apply(CheckedDiv, [lhs, rhs], &mut ctx)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(5i64), None]),
        &mut ctx
    );
    Ok(())
}

/// The [`NullHandling`] the framework derives for `F`, which no other API exposes: a row function
/// never declares one.
fn null_handling<F: RowFn>() -> NullHandling {
    row_null_handling::<F::ArgsWitness, F::RetWitness>()
}

/// Neither `Dense` nor `Filter` is ever written down: the arguments and the return type decide.
/// `Dense` is chosen whenever it is sound, because it is cheaper and preserves input encodings.
#[test]
fn null_handling_follows_from_args_and_ret() {
    // Primitive arguments, infallible: nothing behind a null row can fault.
    assert_eq!(null_handling::<Hypot>(), NullHandling::Dense);
    // `Bytes` resolves a view into a data buffer, which is only meaningful for valid rows.
    assert_eq!(null_handling::<Shout>(), NullHandling::Filter);
    // Fallible: a garbage row could raise an error of its own.
    assert_eq!(null_handling::<CheckedDiv>(), NullHandling::Filter);
}

/// What the lifting adds around every row loop: null propagation, constant folding, nullability
/// widening, and options serde, none of which a row function writes.
///
/// The kernel is the same wrapping addition either way, and only its argument element decides which
/// null-handling path the lifting takes, so every case here runs both.
mod lifting {
    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    /// An `i32` element that is [dense-safe] iff `DENSE`, and otherwise the plain `i32` element in
    /// every respect. Dense-safety is what decides the null-handling path, so a pair of these is
    /// how one kernel gets run under both.
    ///
    /// [dense-safe]: InputElement::DENSE_SAFE
    struct MaybeDenseI32<const DENSE: bool>;

    impl<const DENSE: bool> InputElement for MaybeDenseI32<DENSE> {
        type Column = <i32 as InputElement>::Column;
        type Varying<'a> = <i32 as InputElement>::Varying<'a>;
        type Elem<'a> = i32;

        const DENSE_SAFE: bool = DENSE;
        const DECODE_FALLIBLE: bool = false;

        fn validate(dtype: &DType) -> VortexResult<()> {
            <i32 as InputElement>::validate(dtype)
        }

        fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
            <i32 as InputElement>::decode(array, ctx)
        }

        fn get(column: &Self::Column, index: usize) -> i32 {
            <i32 as InputElement>::get(column, index)
        }

        fn varying(column: &Self::Column) -> Self::Varying<'_> {
            <i32 as InputElement>::varying(column)
        }

        fn varying_len(column: &Self::Varying<'_>) -> usize {
            <i32 as InputElement>::varying_len(column)
        }

        fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> i32
        where
            Self: 'a,
        {
            <i32 as InputElement>::get_varying(column, index)
        }
    }

    /// Wrapping addition over two [`MaybeDenseI32`] columns.
    #[derive(Clone)]
    struct Add<const DENSE: bool>;

    impl<const DENSE: bool> RowFn for Add<DENSE> {
        type Options = EmptyOptions;
        type ArgsWitness = (MaybeDenseI32<DENSE>, MaybeDenseI32<DENSE>);
        type RetWitness = i32;

        fn id(&self) -> ScalarFnId {
            if DENSE {
                static ID: CachedId = CachedId::new("vortex.test.add.dense");
                *ID
            } else {
                static ID: CachedId = CachedId::new("vortex.test.add.filter");
                *ID
            }
        }

        fn arg_name(&self, idx: usize) -> ChildName {
            ChildName::from(["lhs", "rhs"][idx])
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::Out> {
            visitor.visit::<(MaybeDenseI32<DENSE>, MaybeDenseI32<DENSE>), i32>(|(lhs, rhs)| {
                lhs.wrapping_add(rhs)
            })
        }
    }

    /// Adds `lhs` to `rhs` under both null-handling paths and asserts each result equals
    /// `expected`, which is what every case below does.
    ///
    /// Forcing a *strategy* within the filter contract is a separate axis, covered in
    /// [`null_strategies`](super::null_strategies).
    fn assert_add(lhs: ArrayRef, rhs: ArrayRef, expected: ArrayRef) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let dense = apply(Add::<true>, [lhs.clone(), rhs.clone()], &mut ctx)?;
        let filtered = apply(Add::<false>, [lhs, rhs], &mut ctx)?;

        assert_eq!(null_handling::<Add<true>>(), NullHandling::Dense);
        assert_eq!(null_handling::<Add<false>>(), NullHandling::Filter);
        assert_arrays_eq!(dense, expected, &mut ctx);
        assert_arrays_eq!(filtered, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn no_nulls() -> VortexResult<()> {
        assert_add(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            PrimitiveArray::from_iter([11i32, 22, 33]).into_array(),
        )
    }

    #[test]
    fn nulls_propagate() -> VortexResult<()> {
        assert_add(
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None]).into_array(),
            PrimitiveArray::from_option_iter([Some(10i32), Some(20), None, None]).into_array(),
            PrimitiveArray::from_option_iter([Some(11i32), None, None, None]).into_array(),
        )
    }

    /// Strictness: a null constant makes the whole output null without the kernel running at all.
    #[test]
    fn null_constant_short_circuits() -> VortexResult<()> {
        let null = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));

        assert_add(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            ConstantArray::new(null, 3).into_array(),
            PrimitiveArray::from_option_iter([Option::<i32>::None, None, None]).into_array(),
        )
    }

    /// All-constant inputs evaluate one row and broadcast it.
    #[test]
    fn all_constants_broadcast() -> VortexResult<()> {
        assert_add(
            ConstantArray::new(Scalar::from(2i32), 4).into_array(),
            ConstantArray::new(Scalar::from(40i32), 4).into_array(),
            PrimitiveArray::from_iter([42i32, 42, 42, 42]).into_array(),
        )
    }

    #[test]
    fn mixed_constant_and_column() -> VortexResult<()> {
        assert_add(
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
            ConstantArray::new(Scalar::from(10i32), 3).into_array(),
            PrimitiveArray::from_option_iter([Some(11i32), None, Some(13)]).into_array(),
        )
    }

    /// An empty batch is neither all-valid nor all-null, and a zero-length non-nullable execution
    /// keeps its non-nullable dtype.
    #[test]
    fn empty_input_keeps_dtype() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let empty = || PrimitiveArray::from_iter(Vec::<i32>::new()).into_array();

        let result = apply(Add::<false>, [empty(), empty()], &mut ctx)?;

        assert_eq!(result.len(), 0);
        assert!(!result.dtype().is_nullable());
        Ok(())
    }

    /// The output element dtype is non-nullable, and the lifting widens it iff an input is
    /// nullable, which is what makes strictness's dtype contract hold by construction.
    #[test]
    fn return_dtype_unions_nullability() -> VortexResult<()> {
        let non_nullable = DType::Primitive(PType::I32, Nullability::NonNullable);
        let nullable = non_nullable.as_nullable();

        assert_eq!(
            ScalarFnVTable::return_dtype(
                &Add::<true>,
                &EmptyOptions,
                &[non_nullable.clone(), non_nullable.clone()]
            )?,
            non_nullable
        );
        assert_eq!(
            ScalarFnVTable::return_dtype(
                &Add::<true>,
                &EmptyOptions,
                &[non_nullable, nullable.clone()]
            )?,
            nullable
        );
        Ok(())
    }

    #[test]
    fn a_row_fn_is_strict() {
        assert!(ScalarFnVTable::is_strict(&Add::<true>, &EmptyOptions));
    }

    /// Both output forms build an all-valid column, so the output validity is exactly the child
    /// conjunction and the planner never has to execute the function to learn which rows are null.
    #[test]
    fn validity_is_the_child_conjunction() -> VortexResult<()> {
        let expr = Add::<true>.new_expr(EmptyOptions, [root(), root()]);

        assert!(ScalarFnVTable::validity(&Add::<true>, &EmptyOptions, &expr)?.is_some());
        Ok(())
    }

    /// Options serde comes from [`PersistableOptions`], so a row function needs none of its own.
    #[test]
    fn options_round_trip_without_per_function_serde() -> VortexResult<()> {
        let metadata = ScalarFnVTable::serialize(&Add::<true>, &EmptyOptions)?
            .expect("EmptyOptions is serializable");

        let options = ScalarFnVTable::deserialize(&Add::<true>, &metadata, &array_session())?;

        assert_eq!(options, EmptyOptions);
        Ok(())
    }
}

/// A [`RowFn`] choosing its element types per batch: `max(a, b)` over whichever integer width the
/// inputs are, every width under one ID.
mod dispatched {
    use vortex_error::vortex_ensure;

    use super::*;
    use crate::match_each_integer_ptype;

    #[derive(Clone)]
    struct Max;

    impl RowFn for Max {
        type Options = EmptyOptions;
        type ArgsWitness = (i64, i64);
        type RetWitness = i64;

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("vortex.test.int_max");
            *ID
        }

        fn arg_name(&self, idx: usize) -> ChildName {
            ChildName::from(["lhs", "rhs"][idx])
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &Self::Options,
            args: &[DType],
            visitor: V,
        ) -> VortexResult<V::Out> {
            let DType::Primitive(ptype, _) = args[0] else {
                vortex_bail!("int_max requires primitive inputs, got {}", args[0]);
            };
            vortex_ensure!(
                ptype.is_int(),
                "int_max requires integer inputs, got {ptype}"
            );

            match_each_integer_ptype!(ptype, |T| { visitor.visit::<(T, T), T>(|(a, b)| a.max(b)) })
        }
    }

    #[rstest]
    #[case::i16(buffer![1i16, 9, 3].into_array(), buffer![4i16, 2, 3].into_array(), buffer![4i16, 9, 3].into_array())]
    #[case::i64(buffer![1i64, 9, 3].into_array(), buffer![4i64, 2, 3].into_array(), buffer![4i64, 9, 3].into_array())]
    #[case::u8(buffer![1u8, 9, 3].into_array(), buffer![4u8, 2, 3].into_array(), buffer![4u8, 9, 3].into_array())]
    fn dispatches_at_each_integer_width(
        #[case] lhs: ArrayRef,
        #[case] rhs: ArrayRef,
        #[case] expected: ArrayRef,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let result = apply(Max, [lhs, rhs], &mut ctx)?;

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn rejects_a_float_width() {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = buffer![1.0f64].into_array();
        let rhs = buffer![2.0f64].into_array();

        let error = apply(Max, [lhs, rhs], &mut ctx)
            .expect_err("a float width must be rejected at construction");

        assert!(
            error.to_string().contains("integer inputs"),
            "unexpected error: {error}"
        );
    }
}

/// A constant operand holds one distinct value, so the framework decodes it once and every row reads
/// that single row. Without this an element with an expensive decode pays for it per row.
mod constant_operands {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_buffer::Buffer;

    use super::*;

    /// Total rows handed to [`CountedI64::decode`] across one execution. Sound as a global because
    /// each test binary runs one test per process.
    static DECODED_ROWS: AtomicUsize = AtomicUsize::new(0);

    /// Stands in for an element whose decode is expensive per row, recording how wide a column each
    /// decode was actually given.
    struct CountedI64;

    impl InputElement for CountedI64 {
        type Column = Buffer<i64>;
        type Varying<'a> = <i64 as InputElement>::Varying<'a>;
        type Elem<'a> = i64;

        const DENSE_SAFE: bool = true;
        const DECODE_FALLIBLE: bool = false;

        fn validate(dtype: &DType) -> VortexResult<()> {
            <i64 as InputElement>::validate(dtype)
        }

        fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
            DECODED_ROWS.fetch_add(array.len(), Ordering::Relaxed);
            <i64 as InputElement>::decode(array, ctx)
        }

        fn get(column: &Self::Column, index: usize) -> i64 {
            <i64 as InputElement>::get(column, index)
        }

        fn varying(column: &Self::Column) -> Self::Varying<'_> {
            <i64 as InputElement>::varying(column)
        }

        fn varying_len(column: &Self::Varying<'_>) -> usize {
            <i64 as InputElement>::varying_len(column)
        }

        fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> i64
        where
            Self: 'a,
        {
            <i64 as InputElement>::get_varying(column, index)
        }
    }

    #[derive(Clone)]
    struct AddCounted;

    impl RowFn for AddCounted {
        type Options = EmptyOptions;
        type ArgsWitness = (CountedI64, CountedI64);
        type RetWitness = i64;

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("vortex.test.add_counted");
            *ID
        }

        fn arg_name(&self, idx: usize) -> ChildName {
            ChildName::from(["lhs", "rhs"][idx])
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::Out> {
            visitor.visit::<(CountedI64, CountedI64), i64>(|(lhs, rhs)| lhs + rhs)
        }
    }

    #[test]
    fn a_constant_operand_is_decoded_once() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let column = PrimitiveArray::from_iter(0..64i64).into_array();
        let constant = ConstantArray::new(Scalar::from(10i64), 64).into_array();

        let result = apply(AddCounted, [column, constant], &mut ctx)?;

        // 64 rows for the real column, plus exactly one for the constant.
        assert_eq!(DECODED_ROWS.load(Ordering::Relaxed), 65);
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter((0..64i64).map(|value| value + 10)),
            &mut ctx
        );
        Ok(())
    }
}

/// The prepared visit: per-batch state computed once from the batch-constant operands, handed to
/// every row by shared reference.
mod prepared {
    use std::cell::Cell;

    use super::*;
    use crate::validity::Validity;

    thread_local! {
        /// Which operands the last `prepare` saw as constant, as a bitmask (bit 0 for `x`, bit 1
        /// for `y`). Thread-local rather than a process global so concurrent tests in one process
        /// (plain `cargo test`) cannot race it; execution runs on the calling thread.
        static SEEN_CONSTANTS: Cell<u8> = const { Cell::new(u8::MAX) };
    }

    /// `sqrt(x^2 + y^2)` through [`RowVisitor::visit_prepared`]: the square of any constant
    /// operand is hoisted out of the row loop, and recorded in [`SEEN_CONSTANTS`].
    #[derive(Clone)]
    struct PreparedHypot;

    impl RowFn for PreparedHypot {
        type Options = EmptyOptions;
        type ArgsWitness = (f64, f64);
        type RetWitness = f64;

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("vortex.test.prepared_hypot");
            *ID
        }

        fn arg_name(&self, idx: usize) -> ChildName {
            ChildName::from(["x", "y"][idx])
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::Out> {
            visitor.visit_prepared::<(f64, f64), (Option<f64>, Option<f64>), f64>(
                |(x, y)| {
                    SEEN_CONSTANTS.set(u8::from(x.is_some()) | (u8::from(y.is_some()) << 1));
                    (x.map(|x| x * x), y.map(|y| y * y))
                },
                |&(x_sq, y_sq), (x, y)| (x_sq.unwrap_or(x * x) + y_sq.unwrap_or(y * y)).sqrt(),
            )
        }
    }

    /// A constant operand reaches `prepare` as `Some`, and the result is identical to the same
    /// value expanded into a full column, which reaches `prepare` as `None`.
    #[test]
    fn a_constant_operand_matches_its_expanded_column() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let x = buffer![3.0f64, 5.0, 8.0].into_array();

        let constant = ConstantArray::new(Scalar::from(4.0f64), 3).into_array();
        let from_constant = apply(PreparedHypot, [x.clone(), constant], &mut ctx)?;
        assert_eq!(SEEN_CONSTANTS.get(), 0b10);

        let expanded = buffer![4.0f64, 4.0, 4.0].into_array();
        let from_expanded = apply(PreparedHypot, [x, expanded], &mut ctx)?;
        assert_eq!(SEEN_CONSTANTS.get(), 0b00);

        assert_arrays_eq!(from_constant, from_expanded, &mut ctx);
        Ok(())
    }

    /// A masked constant (the same value in every row, some rows null, how the compressor spells
    /// an all-same-with-nulls chunk) is a batch constant too: the wrapper carries only validity,
    /// which the lifting owns, so `prepare` sees the child's value and the null rows stay
    /// null in the result.
    #[test]
    fn a_masked_constant_operand_is_seen_as_constant() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let x = buffer![3.0f64, 5.0, 8.0].into_array();

        let masked_constant = MaskedArray::try_new(
            ConstantArray::new(Scalar::from(4.0f64), 3).into_array(),
            Validity::from_iter([true, false, true]),
        )?
        .into_array();
        let result = apply(PreparedHypot, [x, masked_constant], &mut ctx)?;

        assert_eq!(SEEN_CONSTANTS.get(), 0b10);
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(5.0f64), None, Some((80.0f64).sqrt())]),
            &mut ctx
        );
        Ok(())
    }

    /// With no constant operand every `ConstElems` slot is `None` and the loop computes exactly
    /// what [`RowVisitor::visit`] would.
    #[test]
    fn all_varying_operands_prepare_nothing() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let x = buffer![3.0f64, 5.0].into_array();
        let y = buffer![4.0f64, 12.0].into_array();

        let result = apply(PreparedHypot, [x, y], &mut ctx)?;

        assert_eq!(SEEN_CONSTANTS.get(), 0b00);
        assert_arrays_eq!(result, PrimitiveArray::from_iter([5.0f64, 13.0]), &mut ctx);
        Ok(())
    }

    /// Two constant operands are folded to a single-row execution by the lifting, and that
    /// row still goes through `prepare`, seeing both constants.
    #[test]
    fn all_constant_operands_fold_and_still_prepare() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let x = ConstantArray::new(Scalar::from(3.0f64), 4).into_array();
        let y = ConstantArray::new(Scalar::from(4.0f64), 4).into_array();

        let result = apply(PreparedHypot, [x, y], &mut ctx)?;

        assert_eq!(SEEN_CONSTANTS.get(), 0b11);
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([5.0f64, 5.0, 5.0, 5.0]),
            &mut ctx
        );
        Ok(())
    }

    /// Null rows pass through the prepared path exactly as through [`RowVisitor::visit`].
    #[test]
    fn nulls_propagate_through_the_prepared_path() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let x = PrimitiveArray::from_option_iter([Some(3.0f64), None, Some(8.0)]).into_array();
        let y = ConstantArray::new(Scalar::from(4.0f64), 3).into_array();

        let result = apply(PreparedHypot, [x, y], &mut ctx)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(5.0f64), None, Some((80.0f64).sqrt())]),
            &mut ctx
        );
        Ok(())
    }
}

/// Fallibility has two sources, the return type and an argument whose decode can fail on legal data,
/// and the framework has to derive it from both.
mod decode_fallibility {
    use super::*;

    /// Stands in for an element that *parses* its bytes, like a WKB geometry: malformed bytes in a
    /// valid row are a domain error, so decoding can fail on otherwise legal input.
    struct ParsedBytes;

    impl InputElement for ParsedBytes {
        type Column = VarBinViewArray;
        type Varying<'a> = &'a VarBinViewArray;
        type Elem<'a> = usize;

        const DENSE_SAFE: bool = true;
        const DECODE_FALLIBLE: bool = true;

        fn validate(_dtype: &DType) -> VortexResult<()> {
            Ok(())
        }

        fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
            array.execute::<VarBinViewArray>(ctx)
        }

        fn get(column: &Self::Column, index: usize) -> usize {
            column.views()[index].len() as usize
        }

        fn varying(column: &Self::Column) -> Self::Varying<'_> {
            column
        }

        fn varying_len(column: &Self::Varying<'_>) -> usize {
            column.len()
        }

        fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> usize
        where
            Self: 'a,
        {
            Self::get(column, index)
        }
    }

    /// Its row computation is total; only the decode can fail.
    #[derive(Clone)]
    struct TotalKernelOverParsedInput;

    impl RowFn for TotalKernelOverParsedInput {
        type Options = EmptyOptions;
        type ArgsWitness = (ParsedBytes,);
        type RetWitness = u64;

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("vortex.test.total_over_parsed");
            *ID
        }

        fn arg_name(&self, _idx: usize) -> ChildName {
            ChildName::from("input")
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::Out> {
            visitor.visit::<(ParsedBytes,), u64>(|(len,)| len as u64)
        }
    }

    /// `Ret = u64` is infallible, so reading fallibility off the return type alone would report
    /// `false` and let dict pushdown speculatively evaluate the parse over unreferenced values.
    #[test]
    fn a_fallible_decode_makes_the_function_fallible() {
        assert!(ScalarFnVTable::is_fallible(
            &TotalKernelOverParsedInput,
            &EmptyOptions
        ));
    }

    /// And it must not run densely: rows behind nulls would be parsed too.
    #[test]
    fn a_fallible_decode_forces_filtering() {
        assert_eq!(
            null_handling::<TotalKernelOverParsedInput>(),
            NullHandling::Filter
        );
    }
}

/// The sink visit: output written per row rather than returned.
///
/// The sink here has the same shape as `vortex-tensor`'s tensor sink, which is what the mechanism was
/// built for: a per-row handle that is a *slice* of one flat buffer allocated for the whole batch, and
/// an output dtype read off the input dtypes rather than fixed by a Rust type.
mod sink {
    use std::sync::Arc;

    use vortex_buffer::BufferMut;
    use vortex_error::VortexExpect;
    use vortex_error::vortex_ensure_eq;
    use vortex_error::vortex_err;

    use super::*;
    use crate::arrays::FixedSizeListArray;
    use crate::dtype::NativePType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    /// Builds a `FixedSizeList<T, W>` column, presenting each row as the `&mut [T]` slice to fill.
    ///
    /// Its element dtype comes from the input rather than from `T` alone, so it exercises
    /// [`OutputSink::sink_dtype`] actually reading `args`.
    struct SpreadSink<T, const W: usize> {
        dtype: DType,
        rows: usize,
        elements: BufferMut<T>,
    }

    impl<T: NativePType, const W: usize> OutputSink for SpreadSink<T, W> {
        type Rows<'a> = (&'a mut [T], usize);
        type Row<'a> = &'a mut [T];

        fn sink_dtype(args: &[DType]) -> VortexResult<DType> {
            let element = args.first().ok_or_else(|| {
                vortex_err!("a spread sink takes its element dtype from its input")
            })?;
            <T as InputElement>::validate(element)?;
            Ok(DType::FixedSizeList(
                Arc::new(element.as_nonnullable()),
                u32::try_from(W).vortex_expect("test width fits in u32"),
                Nullability::NonNullable,
            ))
        }

        fn with_capacity(rows: usize, dtype: &DType) -> VortexResult<Self> {
            Ok(Self {
                dtype: dtype.clone(),
                rows,
                elements: BufferMut::zeroed(rows * W),
            })
        }

        fn rows(&mut self) -> Self::Rows<'_> {
            (self.elements.as_mut_slice(), self.rows)
        }

        fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
            rows.1 == row_count && row_count.checked_mul(W) == Some(rows.0.len())
        }

        fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> &'a mut [T] {
            &mut rows.0[index * W..][..W]
        }

        fn finish(self) -> VortexResult<ArrayRef> {
            vortex_ensure_eq!(
                self.dtype,
                Self::sink_dtype(&[DType::Primitive(T::PTYPE, self.dtype.nullability())])?,
                "the sink must build the dtype it named",
            );
            Ok(FixedSizeListArray::try_new(
                PrimitiveArray::new(self.elements.freeze(), Validity::NonNullable).into_array(),
                u32::try_from(W).vortex_expect("test width fits in u32"),
                Validity::NonNullable,
                self.rows,
            )?
            .into_array())
        }
    }

    /// Broadcasts each input value across a fixed-size list row: `spread(x) == [x, x, x]`.
    #[derive(Clone)]
    struct Spread;

    impl RowFn for Spread {
        type Options = EmptyOptions;
        type ArgsWitness = (i64,);
        type RetWitness = ();

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("vortex.test.spread");
            *ID
        }

        fn arg_name(&self, _idx: usize) -> ChildName {
            ChildName::from("input")
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::Out> {
            visitor.visit_into::<(i64,), SpreadSink<i64, 3>, ()>(|(x,), out| out.fill(x))
        }
    }

    /// The same, but refusing negative inputs, so its row closure returns `VortexResult<()>`.
    #[derive(Clone)]
    struct SpreadNonNegative;

    impl RowFn for SpreadNonNegative {
        type Options = EmptyOptions;
        type ArgsWitness = (i64,);
        type RetWitness = VortexResult<()>;

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("vortex.test.spread_non_negative");
            *ID
        }

        fn arg_name(&self, _idx: usize) -> ChildName {
            ChildName::from("input")
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::Out> {
            visitor.visit_into::<(i64,), SpreadSink<i64, 3>, VortexResult<()>>(|(x,), out| {
                if x < 0 {
                    vortex_bail!("negative input {x}");
                }
                out.fill(x);
                Ok(())
            })
        }
    }

    /// `SpreadSink`'s three-element rows, built from `values`.
    fn spread_rows(values: impl IntoIterator<Item = Option<i64>>) -> VortexResult<ArrayRef> {
        let values = values.into_iter().collect::<Vec<_>>();
        let rows = values.len();
        let flat = values
            .iter()
            .flat_map(|value| [value.unwrap_or(0); 3])
            .collect::<Vec<_>>();
        let validity = if values.iter().all(Option::is_some) {
            Validity::NonNullable
        } else {
            Validity::from_iter(values.iter().map(Option::is_some))
        };

        Ok(FixedSizeListArray::try_new(
            PrimitiveArray::new(flat, Validity::NonNullable).into_array(),
            3,
            validity,
            rows,
        )?
        .into_array())
    }

    /// The output dtype is the sink's, with its width, and every row holds the written slice.
    #[test]
    fn writes_one_row_at_a_time() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let input = buffer![7i64, -2, 0].into_array();

        let result = apply(Spread, [input], &mut ctx)?;

        assert_arrays_eq!(result, spread_rows([Some(7), Some(-2), Some(0)])?, &mut ctx);
        Ok(())
    }

    /// A null input row is written densely and masked away afterwards, exactly as on the value path.
    #[test]
    fn nulls_are_masked_after_the_sink() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let input = PrimitiveArray::from_option_iter([Some(7i64), None, Some(4)]).into_array();

        let result = apply(Spread, [input], &mut ctx)?;

        assert!(result.dtype().is_nullable());
        assert_arrays_eq!(result, spread_rows([Some(7), None, Some(4)])?, &mut ctx);
        Ok(())
    }

    /// A sink whose closure cannot fail is dense, and one whose closure can is filtered, from the
    /// return type alone. This is the fact the split of `RetWitness` down to [`RowResult`] preserves.
    #[test]
    fn null_handling_follows_from_the_sink_return_type() {
        assert_eq!(null_handling::<Spread>(), NullHandling::Dense);
        assert!(!ScalarFnVTable::is_fallible(&Spread, &EmptyOptions));

        assert_eq!(null_handling::<SpreadNonNegative>(), NullHandling::Filter);
        assert!(ScalarFnVTable::is_fallible(
            &SpreadNonNegative,
            &EmptyOptions
        ));
    }

    /// An error from a writing closure aborts the batch rather than being written into the sink.
    #[test]
    fn a_failing_row_propagates() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let input = buffer![1i64, -5, 3].into_array();

        let error = apply(SpreadNonNegative, [input], &mut ctx).unwrap_err();

        assert!(error.to_string().contains("negative input -5"), "{error}");
        Ok(())
    }

    /// Being fallible, `SpreadNonNegative` is filtered, so its closure never sees the value behind a
    /// null row. A negative payload there must therefore not raise.
    #[test]
    fn a_failing_row_is_never_reached_behind_a_null() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let input = PrimitiveArray::new(
            buffer![1i64, -5, 3],
            Validity::from_iter([true, false, true]),
        )
        .into_array();

        let result = apply(SpreadNonNegative, [input], &mut ctx)?;

        assert_arrays_eq!(result, spread_rows([Some(1), None, Some(3)])?, &mut ctx);
        Ok(())
    }

    /// The sink names its output dtype from the input, so a wrong input dtype is rejected at plan
    /// time rather than producing a mis-typed column.
    #[test]
    fn the_sink_dtype_validates_its_input() {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(ScalarFnVTable::return_dtype(&Spread, &EmptyOptions, &[dtype]).is_err());
    }

    /// The width the sink declares is the width it builds, over the element dtype it read off the
    /// input.
    #[test]
    fn the_return_dtype_is_the_sinks() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        assert_eq!(
            ScalarFnVTable::return_dtype(&Spread, &EmptyOptions, std::slice::from_ref(&dtype))?,
            DType::FixedSizeList(Arc::new(dtype), 3, Nullability::NonNullable),
        );
        Ok(())
    }
}

/// The branch-and-skip null strategy must agree with the filter strategy bit for bit, must never
/// run `apply` (nor resolve an element) behind a row unset in the conjoined mask, and must be
/// selected exactly per the measured rule.
mod null_strategies {
    use std::sync::Arc;

    use vortex_buffer::ByteBuffer;

    use super::*;
    use crate::arrays::varbinview::BinaryView;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    /// Executes `scalar_fn` over `args` with `strategy` forced, canonicalized like [`apply`].
    fn apply_forced<F: RowFn<Options = EmptyOptions>>(
        scalar_fn: &F,
        args: &[ArrayRef],
        strategy: NullStrategy,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let rows = args.first().map_or(0, |arg| arg.len());

        Ok(execute_row_fn_with_strategy(
            scalar_fn,
            &EmptyOptions,
            args.to_vec(),
            rows,
            strategy,
            ctx,
        )?
        .execute::<Canonical>(ctx)?
        .into_array())
    }

    /// Runs `scalar_fn` under forced filter, forced branch-and-skip, and the automatic per-batch
    /// selection, and asserts all three produce identical arrays.
    fn assert_strategies_agree<F: RowFn<Options = EmptyOptions>>(
        scalar_fn: F,
        args: Vec<ArrayRef>,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let filtered = apply_forced(&scalar_fn, &args, NullStrategy::Filter, &mut ctx)?;
        let branched = apply_forced(&scalar_fn, &args, NullStrategy::BranchAndSkip, &mut ctx)?;
        let auto = apply(scalar_fn, args, &mut ctx)?;

        assert_arrays_eq!(branched, filtered, &mut ctx);
        assert_arrays_eq!(auto, filtered, &mut ctx);
        Ok(())
    }

    /// A `Utf8` column whose null rows carry views naming a buffer that does not exist, at
    /// offsets far out of bounds. Resolving such a row's bytes panics, so strategy agreement
    /// proves the branch loop never calls `get` behind a null.
    fn hostile_nullable_strings() -> VortexResult<ArrayRef> {
        let views = buffer![
            BinaryView::make_view(b"a longer string here", 0, 0),
            BinaryView::new_ref(64, *b"junk", 9, 4096),
            BinaryView::make_view(b"another non-inlined string", 1, 0),
            BinaryView::new_ref(64, *b"junk", 7, 1 << 20),
        ];

        Ok(VarBinViewArray::try_new(
            views,
            Arc::from([
                ByteBuffer::copy_from(b"a longer string here"),
                ByteBuffer::copy_from(b"another non-inlined string"),
            ]),
            DType::Utf8(Nullability::Nullable),
            Validity::from_iter([true, false, true, false]),
        )?
        .into_array())
    }

    /// `Bytes` is not dense-safe, so `Shout` runs under [`NullHandling::Filter`]; both strategies
    /// must produce the same array without resolving the hostile views behind the nulls.
    #[test]
    fn branch_matches_filter_for_bytes() -> VortexResult<()> {
        assert_strategies_agree(Shout, vec![hostile_nullable_strings()?])
    }

    /// A fallible kernel with a poison value (zero divisor) behind every null: the branch loop
    /// must skip those rows rather than spuriously failing on them.
    #[test]
    fn branch_never_applies_a_fallible_kernel_behind_nulls() -> VortexResult<()> {
        let lhs = buffer![10i64, 10, 12, 9].into_array();
        let rhs = PrimitiveArray::new(
            buffer![2i64, 0, 3, 0],
            Validity::from_iter([true, false, true, false]),
        )
        .into_array();

        assert_strategies_agree(CheckedDiv, vec![lhs, rhs])
    }

    /// Nulls in both operands: the branch loop must honor the *conjoined* mask, not either
    /// input's own validity.
    #[test]
    fn branch_conjoins_validities() -> VortexResult<()> {
        let lhs = PrimitiveArray::from_option_iter([Some(10i64), None, Some(12), Some(9), None])
            .into_array();
        let rhs = PrimitiveArray::new(
            buffer![2i64, 0, 3, 0, 0],
            Validity::from_iter([true, true, true, false, false]),
        )
        .into_array();

        assert_strategies_agree(CheckedDiv, vec![lhs, rhs])
    }

    /// A constant operand under the branch strategy still hoists through the stride-0 decode.
    #[test]
    fn branch_handles_constant_operands() -> VortexResult<()> {
        let lhs = PrimitiveArray::from_option_iter([Some(10i64), None, Some(12)]).into_array();
        let rhs = ConstantArray::new(Scalar::from(2i64), 3).into_array();

        assert_strategies_agree(CheckedDiv, vec![lhs, rhs])
    }

    /// An error from a *valid* row still propagates under the branch strategy.
    #[test]
    fn branch_propagates_real_errors() {
        let mut ctx = array_session().create_execution_ctx();
        let lhs = buffer![10i64, 10, 12].into_array();
        let rhs = PrimitiveArray::new(
            buffer![2i64, 3, 0],
            Validity::from_iter([true, false, true]),
        )
        .into_array();

        let error = apply_forced(
            &CheckedDiv,
            &[lhs, rhs],
            NullStrategy::BranchAndSkip,
            &mut ctx,
        )
        .expect_err("a zero divisor in a valid row must fail");

        assert!(
            error.to_string().contains("division by zero"),
            "unexpected error: {error}"
        );
    }

    /// The automatic per-batch selection, observed through elements that record which decode ran
    /// on how many rows: the branch strategy decodes null-tolerantly at full length, the filter
    /// strategy decodes ordinarily over the survivors.
    mod selection {
        use std::cell::Cell;

        use vortex_buffer::Buffer;
        use vortex_error::vortex_err;
        use vortex_mask::Mask;

        use super::*;
        use crate::scalar_fn::row::lift::branch_beats_filter;

        thread_local! {
            /// What the last varying-column decode did: `(null_tolerant, rows)`. Thread-local so
            /// concurrent tests in one process cannot race it; execution runs on the calling
            /// thread.
            static LAST_DECODE: Cell<Option<(bool, usize)>> = const { Cell::new(None) };
        }

        /// An i64 element that records its decodes and claims `SHRINKS` for
        /// [`InputElement::DECODE_SHRINKS_WHEN_FILTERED`]. Declared not dense-safe so the
        /// function runs under [`NullHandling::Filter`] and the strategy selection actually
        /// happens.
        struct TrackedI64<const SHRINKS: bool>;

        impl<const SHRINKS: bool> InputElement for TrackedI64<SHRINKS> {
            type Column = Buffer<i64>;
            type Varying<'a> = <i64 as InputElement>::Varying<'a>;
            type Elem<'a> = i64;

            const DENSE_SAFE: bool = false;
            const DECODE_FALLIBLE: bool = false;
            const DECODE_SHRINKS_WHEN_FILTERED: bool = SHRINKS;

            fn validate(dtype: &DType) -> VortexResult<()> {
                <i64 as InputElement>::validate(dtype)
            }

            fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
                LAST_DECODE.set(Some((false, array.len())));
                <i64 as InputElement>::decode(array, ctx)
            }

            fn decode_null_tolerant(
                array: ArrayRef,
                ctx: &mut ExecutionCtx,
            ) -> VortexResult<Option<Self::Column>> {
                LAST_DECODE.set(Some((true, array.len())));
                <i64 as InputElement>::decode(array, ctx).map(Some)
            }

            fn get(column: &Self::Column, index: usize) -> i64 {
                <i64 as InputElement>::get(column, index)
            }

            fn varying(column: &Self::Column) -> Self::Varying<'_> {
                <i64 as InputElement>::varying(column)
            }

            fn varying_len(column: &Self::Varying<'_>) -> usize {
                <i64 as InputElement>::varying_len(column)
            }

            fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> i64
            where
                Self: 'a,
            {
                <i64 as InputElement>::get_varying(column, index)
            }
        }

        /// Negation over one tracked column.
        #[derive(Clone)]
        struct TrackedNegate<const SHRINKS: bool>;

        impl<const SHRINKS: bool> RowFn for TrackedNegate<SHRINKS> {
            type Options = EmptyOptions;
            type ArgsWitness = (TrackedI64<SHRINKS>,);
            type RetWitness = i64;

            fn id(&self) -> ScalarFnId {
                if SHRINKS {
                    static ID: CachedId = CachedId::new("vortex.test.tracked_negate.per_row");
                    *ID
                } else {
                    static ID: CachedId = CachedId::new("vortex.test.tracked_negate.bulk");
                    *ID
                }
            }

            fn arg_name(&self, _idx: usize) -> ChildName {
                ChildName::from("input")
            }

            fn dispatch<V: RowVisitor>(
                &self,
                _options: &Self::Options,
                _args: &[DType],
                visitor: V,
            ) -> VortexResult<V::Out> {
                visitor.visit::<(TrackedI64<SHRINKS>,), i64>(|(value,)| -value)
            }
        }

        /// A 32-row nullable i64 column whose first `valid_count` rows are valid.
        fn column_with_survivors(valid_count: usize) -> ArrayRef {
            PrimitiveArray::from_option_iter(
                (0..32u16).map(|i| (usize::from(i) < valid_count).then_some(i64::from(i))),
            )
            .into_array()
        }

        /// Executes the tracked function through the full pipeline and returns what the decode
        /// recorded: whether it was null-tolerant, and how many rows it saw.
        fn run<const SHRINKS: bool>(valid_count: usize) -> VortexResult<(bool, usize)> {
            let mut ctx = array_session().create_execution_ctx();
            LAST_DECODE.set(None);

            apply(
                TrackedNegate::<SHRINKS>,
                [column_with_survivors(valid_count)],
                &mut ctx,
            )?;

            LAST_DECODE
                .get()
                .ok_or_else(|| vortex_err!("no decode ran"))
        }

        /// A bulk-decoded element takes branch-and-skip on a mixed mask however sparse the
        /// survivors: the decode is null-tolerant and full length.
        #[test]
        fn bulk_decode_branches_at_any_density() -> VortexResult<()> {
            assert_eq!(run::<false>(31)?, (true, 32));
            assert_eq!(run::<false>(4)?, (true, 32));
            Ok(())
        }

        /// A per-row decode branches only while at least 75% of the rows survive; below that the
        /// filter strategy shrinks the decode to the survivors.
        #[test]
        fn per_row_decode_filters_when_sparse() -> VortexResult<()> {
            // 30/32 surviving: branch, full-length null-tolerant decode.
            assert_eq!(run::<true>(30)?, (true, 32));
            // 24/32 = 75% surviving sits exactly on the threshold: still branch.
            assert_eq!(run::<true>(24)?, (true, 32));
            // 16/32 = 50% surviving: filter, ordinary decode over the survivors.
            assert_eq!(run::<true>(16)?, (false, 16));
            Ok(())
        }

        /// An all-true mask short-circuits to the plain kernel and an all-false mask to an
        /// all-null constant, before any strategy is selected.
        #[test]
        fn degenerate_masks_bypass_the_selection() -> VortexResult<()> {
            assert_eq!(run::<true>(32)?, (false, 32));

            let mut ctx = array_session().create_execution_ctx();
            LAST_DECODE.set(None);
            apply(TrackedNegate::<true>, [column_with_survivors(0)], &mut ctx)?;
            assert_eq!(LAST_DECODE.get(), None);
            Ok(())
        }

        /// An i64 element that omits `decode_null_tolerant`: the conservative default refuses, so
        /// the batch must fall back to the filter strategy even though the selection preferred
        /// branch.
        struct RefusesNullTolerant;

        impl InputElement for RefusesNullTolerant {
            type Column = Buffer<i64>;
            type Varying<'a> = <i64 as InputElement>::Varying<'a>;
            type Elem<'a> = i64;

            const DENSE_SAFE: bool = false;
            const DECODE_FALLIBLE: bool = false;

            fn validate(dtype: &DType) -> VortexResult<()> {
                <i64 as InputElement>::validate(dtype)
            }

            fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
                LAST_DECODE.set(Some((false, array.len())));
                <i64 as InputElement>::decode(array, ctx)
            }

            fn get(column: &Self::Column, index: usize) -> i64 {
                <i64 as InputElement>::get(column, index)
            }

            fn varying(column: &Self::Column) -> Self::Varying<'_> {
                <i64 as InputElement>::varying(column)
            }

            fn varying_len(column: &Self::Varying<'_>) -> usize {
                <i64 as InputElement>::varying_len(column)
            }

            fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> i64
            where
                Self: 'a,
            {
                <i64 as InputElement>::get_varying(column, index)
            }
        }

        #[derive(Clone)]
        struct RefusingNegate;

        impl RowFn for RefusingNegate {
            type Options = EmptyOptions;
            type ArgsWitness = (RefusesNullTolerant,);
            type RetWitness = i64;

            fn id(&self) -> ScalarFnId {
                static ID: CachedId = CachedId::new("vortex.test.refusing_negate");
                *ID
            }

            fn arg_name(&self, _idx: usize) -> ChildName {
                ChildName::from("input")
            }

            fn dispatch<V: RowVisitor>(
                &self,
                _options: &Self::Options,
                _args: &[DType],
                visitor: V,
            ) -> VortexResult<V::Out> {
                visitor.visit::<(RefusesNullTolerant,), i64>(|(value,)| -value)
            }
        }

        /// The fallback is silent and correct: the ordinary decode runs over the survivors and
        /// the result matches the expected negation.
        #[test]
        fn missing_null_tolerant_decode_falls_back_to_filter() -> VortexResult<()> {
            let mut ctx = array_session().create_execution_ctx();
            LAST_DECODE.set(None);

            let result = apply(
                RefusingNegate,
                [PrimitiveArray::from_option_iter([Some(3i64), None, Some(5)]).into_array()],
                &mut ctx,
            )?;

            assert_eq!(LAST_DECODE.get(), Some((false, 2)));
            assert_arrays_eq!(
                result,
                PrimitiveArray::from_option_iter([Some(-3i64), None, Some(-5)]),
                &mut ctx
            );
            Ok(())
        }

        /// The rule itself, at and around the threshold, without going through an execution.
        ///
        /// [`BRANCH_MIN_SURVIVING_FRACTION`]: crate::scalar_fn::row::lift::BRANCH_MIN_SURVIVING_FRACTION
        #[rstest]
        #[case::bulk_dense_mask(false, 99, 100, true)]
        #[case::bulk_sparse_mask(false, 1, 100, true)]
        #[case::per_row_dense_mask(true, 99, 100, true)]
        #[case::per_row_at_threshold(true, 75, 100, true)]
        #[case::per_row_below_threshold(true, 74, 100, false)]
        #[case::per_row_sparse_mask(true, 10, 100, false)]
        fn selects_branch_per_the_measured_rule(
            #[case] decode_shrinks_when_filtered: bool,
            #[case] true_count: usize,
            #[case] len: usize,
            #[case] expect_branch: bool,
        ) {
            let valid = Mask::from_indices(len, 0..true_count);
            assert_eq!(
                branch_beats_filter(decode_shrinks_when_filtered, &valid),
                expect_branch,
            );
        }
    }
}

/// Every [`InputElement`] in this crate run through [`assert_element_conforms`].
///
/// Each case feeds an array whose payload *behind the nulls* is deliberately hostile, since a
/// vacuous payload would let a wrong [`InputElement::DENSE_SAFE`] pass.
mod conformance {
    use std::sync::Arc;

    use vortex_buffer::BitBuffer;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbinview::BinaryView;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar_fn::Bytes;
    use crate::scalar_fn::BytesLen;
    use crate::scalar_fn::assert_element_conforms;
    use crate::validity::Validity;

    /// A `Utf8` column whose single null row carries a view naming a buffer that does not exist, at
    /// an offset far past the end of the data. Reading its *bytes* densely panics; reading its
    /// *length* does not, which is exactly the distinction `DENSE_SAFE` encodes.
    fn hostile_views() -> VortexResult<crate::ArrayRef> {
        let views = buffer![
            BinaryView::make_view(b"a longer string here", 0, 0),
            BinaryView::new_ref(64, *b"junk", 9, 4096),
        ];
        Ok(VarBinViewArray::try_new(
            views,
            Arc::from([ByteBuffer::copy_from(b"a longer string here")]),
            DType::Utf8(Nullability::Nullable),
            Validity::from_iter([true, false]),
        )?
        .into_array())
    }

    #[test]
    fn primitive_element_conforms() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // The extremes sit at the rows that are then marked null.
        let array = PrimitiveArray::new(
            buffer![i32::MAX, 1, i32::MIN, 2],
            Validity::from_iter([false, true, false, true]),
        )
        .into_array();

        assert_element_conforms::<i32>(array, &DType::Utf8(Nullability::NonNullable), &mut ctx)
    }

    #[test]
    fn bool_element_conforms() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = BoolArray::new(
            BitBuffer::from(vec![true, true, false, true]),
            Validity::from_iter([false, true, true, false]),
        )
        .into_array();

        assert_element_conforms::<bool>(array, &DType::Utf8(Nullability::NonNullable), &mut ctx)
    }

    #[test]
    fn bytes_len_element_conforms() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        assert_element_conforms::<BytesLen>(
            hostile_views()?,
            &DType::Bool(Nullability::NonNullable),
            &mut ctx,
        )
    }

    #[test]
    fn bytes_element_conforms() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        assert_element_conforms::<Bytes>(
            hostile_views()?,
            &DType::Bool(Nullability::NonNullable),
            &mut ctx,
        )
    }
}
