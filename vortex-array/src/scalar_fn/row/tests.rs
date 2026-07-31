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
                    SEEN_CONSTANTS
                        .set(u8::from(x.is_some()) | (u8::from(y.is_some()) << 1));
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
    /// which the strict lifting owns, so `prepare` sees the child's value and the null rows stay
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

    /// Two constant operands are folded to a single-row execution by the strict lifting, and that
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

        fn row(&mut self, index: usize) -> &mut [T] {
            &mut self.elements.as_mut_slice()[index * W..][..W]
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
        assert_eq!(
            StrictScalarFnVTable::null_handling(&Spread, &EmptyOptions),
            NullHandling::Dense
        );
        assert!(!ScalarFnVTable::is_fallible(&Spread, &EmptyOptions));

        assert_eq!(
            StrictScalarFnVTable::null_handling(&SpreadNonNegative, &EmptyOptions),
            NullHandling::Filter
        );
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
        assert!(
            StrictScalarFnVTable::return_element_dtype(&Spread, &EmptyOptions, &[dtype]).is_err()
        );
    }

    /// The width the sink declares is the width it builds, over the element dtype it read off the
    /// input.
    #[test]
    fn the_return_dtype_is_the_sinks() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        assert_eq!(
            StrictScalarFnVTable::return_element_dtype(
                &Spread,
                &EmptyOptions,
                std::slice::from_ref(&dtype)
            )?,
            DType::FixedSizeList(Arc::new(dtype), 3, Nullability::NonNullable),
        );
        Ok(())
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
