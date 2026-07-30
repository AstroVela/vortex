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
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::expr::root;
use crate::scalar::Scalar;
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

/// Neither `Dense` nor `Filter` is ever written down: the arguments and the return type decide.
/// `Dense` is chosen whenever it is sound, because it is cheaper and preserves input encodings.
#[test]
fn null_handling_follows_from_args_and_ret() {
    // Primitive arguments, infallible: nothing behind a null row can fault.
    assert_eq!(
        StrictScalarFnVTable::null_handling(&Hypot, &EmptyOptions),
        NullHandling::Dense
    );
    // `Bytes` resolves a view into a data buffer, which is only meaningful for valid rows.
    assert_eq!(
        StrictScalarFnVTable::null_handling(&Shout, &EmptyOptions),
        NullHandling::Filter
    );
    // Fallible: a garbage row could raise an error of its own.
    assert_eq!(
        StrictScalarFnVTable::null_handling(&CheckedDiv, &EmptyOptions),
        NullHandling::Filter
    );
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

/// Fallibility has two sources, the return type and an argument whose decode can fail on legal data,
/// and the framework has to derive it from both.
mod decode_fallibility {
    use super::*;

    /// Stands in for an element that *parses* its bytes, like a WKB geometry: malformed bytes in a
    /// valid row are a domain error, so decoding can fail on otherwise legal input.
    struct ParsedBytes;

    impl InputElement for ParsedBytes {
        type Column = VarBinViewArray;
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
            StrictScalarFnVTable::null_handling(&TotalKernelOverParsedInput, &EmptyOptions),
            NullHandling::Filter
        );
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
