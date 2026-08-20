// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::rules::RULES;

const METADATA_VERSION: u8 = 1;
const METADATA_LEN: usize = 9;

/// An integer transform represented as multiplied and additive child arrays.
pub type IntMultArray = Array<IntMult>;

#[array_slots(IntMult)]
pub struct IntMultSlots {
    /// Values multiplied by the array base.
    #[slot(0)]
    pub primary: ArrayRef,
    /// Values added after multiplication.
    #[slot(1)]
    pub secondary: ArrayRef,
}

#[derive(Clone, Debug)]
pub struct IntMultData {
    base: u64,
}

impl Display for IntMultData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "base: {}", self.base)
    }
}

impl ArrayHash for IntMultData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.base.hash(state);
    }
}

impl ArrayEq for IntMultData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.base == other.base
    }
}

#[derive(Clone, Debug)]
pub struct IntMult;

impl VTable for IntMult {
    type TypedArrayData = IntMultData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.int_mult");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        validate_children(data.base, dtype, len, IntMultSlotsView::from_slots(slots))
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, index: usize) -> BufferHandle {
        vortex_panic!("IntMultArray buffer index {index} is invalid")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, index: usize) -> Option<String> {
        vortex_panic!("IntMultArray buffer index {index} is invalid")
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let mut metadata = Vec::with_capacity(METADATA_LEN);
        metadata.push(METADATA_VERSION);
        metadata.extend_from_slice(&array.data().base.to_le_bytes());
        Ok(Some(metadata))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            metadata.len() == METADATA_LEN,
            "IntMult metadata requires {METADATA_LEN} bytes"
        );
        vortex_ensure!(
            metadata[0] == METADATA_VERSION,
            "unsupported IntMult metadata version {}",
            metadata[0]
        );
        vortex_ensure!(children.len() == 2, "IntMult requires two children");

        let mut base_bytes = [0_u8; size_of::<u64>()];
        base_bytes.copy_from_slice(&metadata[1..]);
        let base = u64::from_le_bytes(base_bytes);
        let ptype = PType::try_from(dtype)?;
        ensure_base_fits(base, ptype)?;
        let primary = children.get(0, dtype, len)?;
        let secondary_dtype = DType::Primitive(ptype, NonNullable);
        let secondary = children.get(1, &secondary_dtype, len)?;
        let slots = IntMultSlots { primary, secondary }.into_slots();
        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, IntMultData { base })
                .with_slots(slots),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, index: usize) -> String {
        IntMultSlots::NAMES[index].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let primary = array.primary().clone().execute::<PrimitiveArray>(ctx)?;
        let secondary = array.secondary().clone().execute::<PrimitiveArray>(ctx)?;
        Ok(ExecutionResult::done(
            decode_primitive(primary, secondary, array.base())?.into_array(),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<IntMult> for IntMult {
    fn scalar_at(
        array: ArrayView<'_, IntMult>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let primary = array.primary().execute_scalar(index, ctx)?;
        if primary.is_null() {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        let secondary = array.secondary().execute_scalar(index, ctx)?;
        let primary = primary.as_primitive();
        let secondary = secondary.as_primitive();
        let base = array.base();
        let nullability = array.dtype().nullability();
        Ok(match_each_integer_ptype!(primary.ptype(), |T| {
            let primary = primary
                .typed_value::<T>()
                .vortex_expect("validated IntMult primary scalar");
            let secondary = secondary
                .typed_value::<T>()
                .vortex_expect("validated IntMult secondary scalar");
            let base = T::try_from(base).vortex_expect("validated IntMult base");
            let value = if array.base() == 1 {
                <T as WrappingMulAdd>::wrapping_add(primary, secondary)
            } else {
                T::wrapping_mul_add(base, primary, secondary)
            };
            Scalar::primitive(value, nullability)
        }))
    }
}

impl ValidityChild<IntMult> for IntMult {
    fn validity_child(array: ArrayView<'_, IntMult>) -> ArrayRef {
        array.primary().clone()
    }
}

pub trait IntMultArrayExt: TypedArrayRef<IntMult> + IntMultArraySlotsExt {
    /// Return the multiplier that reconstructs each value.
    fn base(&self) -> u64 {
        self.deref().base
    }
}

impl<T: TypedArrayRef<IntMult>> IntMultArrayExt for T {}

impl IntMult {
    /// Construct an integer multiplication array from generic child arrays.
    pub fn try_new(
        primary: ArrayRef,
        secondary: ArrayRef,
        base: u64,
    ) -> VortexResult<IntMultArray> {
        let dtype = primary.dtype().clone();
        let len = primary.len();
        let slots = IntMultSlots { primary, secondary }.into_slots();
        Array::try_from_parts(
            ArrayParts::new(IntMult, dtype, len, IntMultData { base }).with_slots(slots),
        )
    }

    /// Split integers into quotient and remainder child arrays.
    pub fn from_primitive(
        array: ArrayView<'_, Primitive>,
        base: u64,
    ) -> VortexResult<IntMultArray> {
        let ptype = array.ptype();
        ensure_base_fits(base, ptype)?;
        let validity = array.validity()?;
        let (primary, secondary) = match_each_integer_ptype!(ptype, |T| {
            let base = T::try_from(base).vortex_expect("validated IntMult base");
            let (primary, secondary) = split_buffer(array.as_slice::<T>(), base);
            (
                PrimitiveArray::new(primary, validity).into_array(),
                PrimitiveArray::new(secondary, NonNullable.into()).into_array(),
            )
        });
        Self::try_new(primary, secondary, base)
    }
}

fn validate_children(
    base: u64,
    dtype: &DType,
    len: usize,
    slots: IntMultSlotsView<'_>,
) -> VortexResult<()> {
    let ptype = PType::try_from(dtype)?;
    ensure_base_fits(base, ptype)?;
    vortex_ensure!(
        slots.primary.dtype() == dtype,
        "IntMult primary dtype {} differs from {dtype}",
        slots.primary.dtype()
    );
    vortex_ensure!(slots.primary.len() == len, "IntMult primary length differs");
    let secondary_dtype = DType::Primitive(ptype, NonNullable);
    vortex_ensure!(
        slots.secondary.dtype() == &secondary_dtype,
        "IntMult secondary dtype {} differs from {secondary_dtype}",
        slots.secondary.dtype()
    );
    vortex_ensure!(
        slots.secondary.len() == len,
        "IntMult secondary length differs"
    );
    Ok(())
}

fn ensure_base_fits(base: u64, ptype: PType) -> VortexResult<()> {
    vortex_ensure!(ptype.is_int(), "IntMult requires integers");
    vortex_ensure!(base >= 1, "IntMult base must be positive");
    let maximum = match ptype {
        PType::U8 => u64::from(u8::MAX),
        PType::U16 => u64::from(u16::MAX),
        PType::U32 => u64::from(u32::MAX),
        PType::U64 => u64::MAX,
        PType::I8 => i8::MAX as u64,
        PType::I16 => i16::MAX as u64,
        PType::I32 => i32::MAX as u64,
        PType::I64 => i64::MAX as u64,
        _ => vortex_bail!("IntMult requires integers"),
    };
    vortex_ensure!(base <= maximum, "IntMult base does not fit {ptype}");
    Ok(())
}

fn split_buffer<T>(values: &[T], base: T) -> (Buffer<T>, Buffer<T>)
where
    T: NativePType + Copy + std::ops::Div<Output = T> + std::ops::Rem<Output = T>,
{
    let mut primary = BufferMut::with_capacity(values.len());
    let mut secondary = BufferMut::with_capacity(values.len());
    for value in values {
        primary.push(*value / base);
        secondary.push(*value % base);
    }
    (primary.freeze(), secondary.freeze())
}

fn decode_primitive(
    primary: PrimitiveArray,
    secondary: PrimitiveArray,
    base: u64,
) -> VortexResult<PrimitiveArray> {
    let ptype = primary.ptype();
    ensure_base_fits(base, ptype)?;
    let validity = primary.validity()?;
    Ok(match_each_integer_ptype!(ptype, |T| {
        let base = T::try_from(base).vortex_expect("validated IntMult base");
        let values = compose_buffer(
            primary.into_buffer_mut::<T>(),
            secondary.into_buffer::<T>(),
            base,
        );
        PrimitiveArray::new(values, validity)
    }))
}

fn compose_buffer<T>(mut primary: BufferMut<T>, secondary: Buffer<T>, base: T) -> Buffer<T>
where
    T: NativePType + WrappingMulAdd,
{
    if <T as WrappingMulAdd>::is_one(base) {
        for (primary, secondary) in primary.as_mut_slice().iter_mut().zip(secondary.as_slice()) {
            *primary = <T as WrappingMulAdd>::wrapping_add(*primary, *secondary);
        }
        return primary.freeze();
    }

    for (primary, secondary) in primary.as_mut_slice().iter_mut().zip(secondary.as_slice()) {
        *primary = T::wrapping_mul_add(base, *primary, *secondary);
    }
    primary.freeze()
}

trait WrappingMulAdd: Copy {
    fn is_one(value: Self) -> bool;

    fn wrapping_add(lhs: Self, rhs: Self) -> Self;

    fn wrapping_mul_add(base: Self, primary: Self, secondary: Self) -> Self;
}

macro_rules! impl_wrapping_mul_add {
    ($type:ty) => {
        impl WrappingMulAdd for $type {
            #[inline]
            fn is_one(value: Self) -> bool {
                value == 1
            }

            #[inline]
            fn wrapping_add(lhs: Self, rhs: Self) -> Self {
                lhs.wrapping_add(rhs)
            }

            #[inline]
            fn wrapping_mul_add(base: Self, primary: Self, secondary: Self) -> Self {
                base.wrapping_mul(primary).wrapping_add(secondary)
            }
        }
    };
}

impl_wrapping_mul_add!(u8);
impl_wrapping_mul_add!(u16);
impl_wrapping_mul_add!(u32);
impl_wrapping_mul_add!(u64);
impl_wrapping_mul_add!(i8);
impl_wrapping_mul_add!(i16);
impl_wrapping_mul_add!(i32);
impl_wrapping_mul_add!(i64);

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_session::registry::ReadContext;

    use super::*;

    fn round_trip<T>(values: Vec<T>, base: u64) -> VortexResult<()>
    where
        T: NativePType + Copy,
    {
        let input = PrimitiveArray::from_iter(values);
        let encoded = IntMult::from_primitive(input.as_view(), base)?;
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut array_session().create_execution_ctx())?;
        assert_arrays_eq!(decoded, input, &mut array_session().create_execution_ctx());
        Ok(())
    }

    #[rstest]
    #[case(vec![0_u8, 1, 6, 7, 8, 254, 255], 7)]
    #[case(vec![0_u16, 1, 6, 7, 8, 65_534, 65_535], 7)]
    #[case(vec![0_u32, 1, 6, 7, 8, u32::MAX - 1, u32::MAX], 7)]
    #[case(vec![0_u64, 1, 6, 7, 8, u64::MAX - 1, u64::MAX], 7)]
    #[case(vec![i8::MIN, -8, -7, -1, 0, 1, 7, 8, i8::MAX], 7)]
    #[case(vec![i16::MIN, -8, -7, -1, 0, 1, 7, 8, i16::MAX], 7)]
    #[case(vec![i32::MIN, -8, -7, -1, 0, 1, 7, 8, i32::MAX], 7)]
    #[case(vec![i64::MIN, -8, -7, -1, 0, 1, 7, 8, i64::MAX], 7)]
    fn round_trip_unsigned<T>(#[case] values: Vec<T>, #[case] base: u64) -> VortexResult<()>
    where
        T: NativePType + Copy,
    {
        round_trip(values, base)
    }

    #[test]
    fn exposes_exact_children() -> VortexResult<()> {
        let input = PrimitiveArray::from_iter([0_u32, 11, 22, 33, 44]);
        let encoded = IntMult::from_primitive(input.as_view(), 10)?;
        assert_arrays_eq!(
            encoded.primary(),
            PrimitiveArray::from_iter([0_u32, 1, 2, 3, 4]),
            &mut array_session().create_execution_ctx()
        );
        assert_arrays_eq!(
            encoded.secondary(),
            PrimitiveArray::from_iter([0_u32, 1, 2, 3, 4]),
            &mut array_session().create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn base_one_adds_generic_children() -> VortexResult<()> {
        let primary = PrimitiveArray::from_iter([100_u32, 200, 300]).into_array();
        let secondary = PrimitiveArray::from_iter([1_u32, 2, 3]).into_array();
        let encoded = IntMult::try_new(primary, secondary, 1)?;
        assert_arrays_eq!(
            encoded,
            PrimitiveArray::from_iter([101_u32, 202, 303]),
            &mut array_session().create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn preserves_validity() -> VortexResult<()> {
        let input = PrimitiveArray::new(
            buffer![10_u32, 20, 30, 40],
            Validity::from_iter([true, false, true, false]),
        );
        let encoded = IntMult::from_primitive(input.as_view(), 10)?;
        let mut ctx = array_session().create_execution_ctx();
        assert!(
            encoded
                .clone()
                .into_array()
                .execute_scalar(1, &mut ctx)?
                .is_null()
        );
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, input, &mut ctx);
        Ok(())
    }

    #[test]
    fn slices_children() -> VortexResult<()> {
        let input = PrimitiveArray::from_iter([10_u32, 21, 32, 43, 54]);
        let encoded = IntMult::from_primitive(input.as_view(), 10)?;
        let sliced = encoded.into_array().slice(1..4)?;
        assert_arrays_eq!(
            sliced,
            PrimitiveArray::from_iter([21_u32, 32, 43]),
            &mut array_session().create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn serialized_generic_children_round_trip() -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);
        let primary = PrimitiveArray::from_option_iter([Some(100_u32), None, Some(300)]);
        let secondary = PrimitiveArray::from_iter([1_u32, 2, 3]);
        let encoded =
            IntMult::try_new(primary.into_array(), secondary.into_array(), 1)?.into_array();
        let dtype = encoded.dtype().clone();
        let len = encoded.len();
        let array_ctx = ArrayContext::empty();
        let serialized = encoded.serialize(&array_ctx, &session, &SerializeOptions::default())?;
        let mut bytes = ByteBufferMut::empty();
        for buffer in serialized {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let parts = SerializedArray::try_from(bytes.freeze())?;
        let decoded = parts.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &session)?;
        let expected: ArrayRef =
            PrimitiveArray::from_option_iter([Some(101_u32), None, Some(303)]).into_array();
        assert_arrays_eq!(decoded, expected, &mut session.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    #[case(0)]
    #[case(256)]
    fn rejects_invalid_u8_base(#[case] base: u64) {
        let input = PrimitiveArray::from_iter([1_u8, 2, 3]);
        assert!(IntMult::from_primitive(input.as_view(), base).is_err());
    }
}
