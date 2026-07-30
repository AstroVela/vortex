// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The element types a row function can read and produce, and the argument tuples built
//! from them.
//!
//! Both traits are open: adding a type family (`&str`, decimals, a tensor row) is one impl, and
//! every row function gains it. See `vortex-tensor`'s `TensorRow` for one that drills through an
//! extension wrapper into its storage.

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::scalar_fn::ExecutionArgs;
use crate::validity::Validity;

/// An element type that can be read row-wise out of an input column.
pub trait InputElement: 'static {
    /// The decoded column representation supporting `O(1)` row access.
    type Column;

    /// The borrowed element value handed to the row closure a [`RowFn`](crate::scalar_fn::RowFn)
    /// visits with.
    type Elem<'a>;

    /// Whether [`get`](Self::get) may be called for a row that is null in the input.
    ///
    /// Arrays only guarantee their contents for *valid* rows, so this is `false` for any element
    /// that follows an offset or pointer stored in the array: behind a null row that value is
    /// arbitrary and may not address anything. Reading a whole value out of a flat buffer is `true`,
    /// since the value is garbage but the read cannot fault.
    ///
    /// [`NullHandling::Dense`](crate::scalar_fn::NullHandling::Dense) requires this of every
    /// argument, and the row layers reject the combination when it does not hold.
    const DENSE_SAFE: bool;

    /// Whether [`decode`](Self::decode) can fail on *legal* input data.
    ///
    /// `false` for an element read straight out of a buffer: decoding can still fail for
    /// infrastructural reasons (IO, allocation), but never because of the values. `true` for an
    /// element that parses its bytes, since a malformed WKB geometry in a *valid* row is a domain
    /// error, which makes a function over that element
    /// [fallible](crate::scalar_fn::ScalarFnVTable::is_fallible) however infallible its own row
    /// computation is.
    const DECODE_FALLIBLE: bool;

    /// Validate that `dtype` is an acceptable input column dtype for this element type.
    fn validate(dtype: &DType) -> VortexResult<()>;

    /// Decode `array` into its column representation. Called once per batch.
    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column>;

    /// Read the element at `index`. Must be `O(1)`: it is called in the row loop.
    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_>;
}

impl<T: NativePType> InputElement for T {
    type Column = Buffer<T>;
    type Elem<'a> = T;

    // Every lane of the buffer holds a `T`, valid or not.
    const DENSE_SAFE: bool = true;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        let expected = T::PTYPE;
        let DType::Primitive(ptype, _) = dtype else {
            vortex_bail!("expected a {expected} column, got {dtype}");
        };
        vortex_ensure_eq!(
            *ptype,
            expected,
            "expected a {expected} column, got {dtype}"
        );
        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        Ok(array.execute::<PrimitiveArray>(ctx)?.into_buffer::<T>())
    }

    fn get(column: &Self::Column, index: usize) -> T {
        column[index]
    }
}

impl InputElement for bool {
    type Column = BitBuffer;
    type Elem<'a> = bool;

    // Every bit of the buffer is readable, valid or not.
    const DENSE_SAFE: bool = true;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        vortex_ensure!(
            matches!(dtype, DType::Bool(_)),
            "expected a Bool column, got {dtype}",
        );
        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        Ok(array.execute::<BoolArray>(ctx)?.into_bit_buffer())
    }

    fn get(column: &Self::Column, index: usize) -> bool {
        column.value(index)
    }
}

/// Marker for byte-string input elements: accepts `Utf8` or `Binary` columns and presents each
/// row as `&[u8]`.
///
/// Resolving a row means following the offset in its view into a data buffer, which is only
/// meaningful for valid rows, so this element forces
/// [`NullHandling::Filter`](crate::scalar_fn::NullHandling::Filter). Use [`BytesLen`] instead when
/// only the length is needed.
pub struct Bytes;

/// Decoded column form of a [`Bytes`] input: the canonical views array plus its resolved data
/// buffers, supporting cheap per-row byte access.
pub struct BytesColumn {
    /// The canonical views array, read one view per row.
    array: VarBinViewArray,

    /// The array's data buffers, hoisted out of the row loop. These could be re-derived per row
    /// from `array`, but [`Bytes::get`](InputElement::get) runs once per row and resolving a buffer
    /// by index **must** stay a slice index rather than a lookup.
    buffers: Vec<ByteBuffer>,
}

impl InputElement for Bytes {
    type Column = BytesColumn;
    type Elem<'a> = &'a [u8];

    // A view behind a null row may point outside its buffer, or name a buffer that does not
    // exist: `VarBinViewArray` only validates the views of valid rows.
    const DENSE_SAFE: bool = false;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        validate_byte_column(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        let array = array.execute::<VarBinViewArray>(ctx)?;
        let buffers = (0..array.data_buffers().len())
            .map(|idx| array.buffer(idx).clone())
            .collect();
        Ok(BytesColumn { array, buffers })
    }

    fn get(column: &Self::Column, index: usize) -> &[u8] {
        let view = &column.array.views()[index];
        if view.is_inlined() {
            view.as_inlined().value()
        } else {
            let view = view.as_view();
            &column.buffers[view.buffer_index as usize].as_slice()[view.as_range()]
        }
    }
}

/// Marker for the byte *length* of a `Utf8` or `Binary` row, presented as `usize`.
///
/// Every view stores its own length, so this reads one field and never resolves the row's bytes:
/// cheaper than [`Bytes`], and safe to read densely. Prefer it whenever the length is all a
/// function needs.
pub struct BytesLen;

impl InputElement for BytesLen {
    type Column = VarBinViewArray;
    type Elem<'a> = usize;

    // The length lives in the view itself, so a view behind a null row yields an arbitrary length
    // rather than an unresolvable pointer.
    const DENSE_SAFE: bool = true;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        validate_byte_column(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        array.execute::<VarBinViewArray>(ctx)
    }

    fn get(column: &Self::Column, index: usize) -> usize {
        column.views()[index].len() as usize
    }
}

/// Shared dtype check for the byte-string element types.
fn validate_byte_column(dtype: &DType) -> VortexResult<()> {
    vortex_ensure!(
        matches!(dtype, DType::Utf8(_) | DType::Binary(_)),
        "expected a Utf8 or Binary column, got {dtype}",
    );
    Ok(())
}

/// An element type that a row computation can produce, buildable into an all-valid column.
pub trait OutputElement: 'static + Sized {
    /// The dtype of columns built from this element type. Must be non-nullable: nullability is
    /// derived from the inputs by the strict lifting.
    fn element_dtype() -> DType;

    /// Build a column from one value per row. Called once per batch.
    fn build(values: Vec<Self>) -> ArrayRef;
}

impl<T: NativePType> OutputElement for T {
    fn element_dtype() -> DType {
        DType::Primitive(T::PTYPE, Nullability::NonNullable)
    }

    fn build(values: Vec<Self>) -> ArrayRef {
        PrimitiveArray::new(values, Validity::NonNullable).into_array()
    }
}

impl OutputElement for bool {
    fn element_dtype() -> DType {
        DType::Bool(Nullability::NonNullable)
    }

    fn build(values: Vec<Self>) -> ArrayRef {
        // `From<Vec<bool>>` packs through the multiversioned SIMD path; `from_iter` would set one
        // bit at a time, which measures 6.6-7.9x slower on the packing step alone.
        BoolArray::new(BitBuffer::from(values), Validity::NonNullable).into_array()
    }
}

impl OutputElement for String {
    fn element_dtype() -> DType {
        DType::Utf8(Nullability::NonNullable)
    }

    fn build(values: Vec<Self>) -> ArrayRef {
        VarBinViewArray::from_iter_str(values).into_array()
    }
}

/// What a row computation may return: an [`OutputElement`] directly, or a [`VortexResult`] of one.
///
/// Implementing it for both forms is what lets one row function trait serve infallible and fallible
/// kernels without a second trait or a wrapper. A kernel returning `f64` and one returning
/// `VortexResult<f64>` agree on [`Out`](Self::Out) and differ only in
/// [`FALLIBLE`](Self::FALLIBLE), which is what the framework reads.
pub trait ApplyResult: 'static {
    /// The element this computation produces.
    type Out: OutputElement;

    /// Whether this return type can carry an error.
    const FALLIBLE: bool;

    /// Convert into a result, so one code path can handle both forms.
    fn into_result(self) -> VortexResult<Self::Out>;
}

impl<T: OutputElement> ApplyResult for T {
    type Out = T;

    const FALLIBLE: bool = false;

    fn into_result(self) -> VortexResult<T> {
        Ok(self)
    }
}

impl<T: OutputElement> ApplyResult for VortexResult<T> {
    type Out = T;

    const FALLIBLE: bool = true;

    fn into_result(self) -> VortexResult<T> {
        self
    }
}

/// One decoded input column of an [`ElementTuple`], together with the stride used to read it.
///
/// A constant operand holds the same value in every row, so it is decoded once as a single row and
/// read at index 0 forever. That is what stops a constant argument costing one decode per row, which
/// matters whenever the decode is more than a buffer read: parsing a geometry from WKB, or
/// canonicalizing an extension row.
pub struct ArgColumn<T: InputElement> {
    /// The decoded column, holding either every row or the single row of a constant operand.
    column: T::Column,

    /// `1` for a real column and `0` for a constant operand, so `index * stride` pins a constant to
    /// its only row. A multiplier rather than a branch, to leave the row loop's shape unconditional.
    stride: usize,
}

impl<T: InputElement> ArgColumn<T> {
    /// Decode one input column, collapsing a constant operand to its single distinct row.
    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        // An empty input has no row 0 to slice, and its row loop runs zero times either way.
        if array.as_constant().is_some() && !array.is_empty() {
            return Ok(Self {
                column: T::decode(array.slice(0..1)?, ctx)?,
                stride: 0,
            });
        }

        Ok(Self {
            column: T::decode(array, ctx)?,
            stride: 1,
        })
    }

    /// Read the element at `index`, which for a constant operand is always its single row.
    fn get(&self, index: usize) -> T::Elem<'_> {
        T::get(&self.column, index * self.stride)
    }
}

/// Tuples of [`InputElement`]s forming the typed argument list a [`RowFn`](crate::scalar_fn::RowFn)
/// visits with.
pub trait ElementTuple: 'static {
    /// The decoded column representations.
    type Columns;

    /// The borrowed row of element values.
    type Elems<'a>;

    /// The number of arguments.
    const ARITY: usize;

    /// Whether every argument is [`InputElement::DENSE_SAFE`].
    const DENSE_SAFE: bool;

    /// Whether *any* argument is [`InputElement::DECODE_FALLIBLE`].
    const DECODE_FALLIBLE: bool;

    /// Validate the input dtypes. `dtypes` has exactly `ARITY` entries (checked by the
    /// expression layer against [`Arity`](crate::scalar_fn::Arity)).
    fn validate(dtypes: &[DType]) -> VortexResult<()>;

    /// Decode every input column once. Called once per batch.
    fn decode(args: &dyn ExecutionArgs, ctx: &mut ExecutionCtx) -> VortexResult<Self::Columns>;

    /// Read the row of elements at `index`. Must be `O(1)`: it is called in the row loop.
    fn get(columns: &Self::Columns, index: usize) -> Self::Elems<'_>;
}

macro_rules! element_tuple {
    ($arity:literal; $($t:ident : $idx:tt),+) => {
        impl<$($t: InputElement),+> ElementTuple for ($($t,)+) {
            type Columns = ($(ArgColumn<$t>,)+);
            type Elems<'a> = ($($t::Elem<'a>,)+);

            const ARITY: usize = $arity;
            const DENSE_SAFE: bool = $($t::DENSE_SAFE &&)+ true;
            const DECODE_FALLIBLE: bool = $($t::DECODE_FALLIBLE ||)+ false;

            fn validate(dtypes: &[DType]) -> VortexResult<()> {
                $($t::validate(&dtypes[$idx])?;)+
                Ok(())
            }

            fn decode(
                args: &dyn ExecutionArgs,
                ctx: &mut ExecutionCtx,
            ) -> VortexResult<Self::Columns> {
                Ok(($(ArgColumn::<$t>::decode(args.get($idx)?, ctx)?,)+))
            }

            fn get(columns: &Self::Columns, index: usize) -> Self::Elems<'_> {
                ($(columns.$idx.get(index),)+)
            }
        }
    };
}

element_tuple!(1; A:0);
element_tuple!(2; A:0, B:1);
element_tuple!(3; A:0, B:1, C:2);
