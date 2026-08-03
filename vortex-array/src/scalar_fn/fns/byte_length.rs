// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::dtype::DType;
use crate::kernel::ExecuteParentKernel;
use crate::scalar_fn::BytesLen;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ElementSink;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::RowFn;
use crate::scalar_fn::RowVisitor;
use crate::scalar_fn::ScalarFnId;

pub trait ByteLengthKernel: VTable {
    fn byte_length(
        array: ArrayView<'_, Self>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

#[derive(Default, Debug)]
pub struct ByteLengthExecuteAdaptor<V>(pub V);

impl<V: ByteLengthKernel> ExecuteParentKernel<V> for ByteLengthExecuteAdaptor<V> {
    type Parent = ExactScalarFn<ByteLength>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        _parent: ScalarFnArrayView<'_, ByteLength>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        vortex_ensure!(child_idx == 0);
        V::byte_length(array, ctx)
    }
}

/// Byte length of each element in a Utf8 or Binary array.
///
/// This is a [`RowFn`] over fixed element types, so the definition below is the whole function:
/// arity, dtype validation, the return dtype, constant handling and null propagation all follow from
/// it. Reading [`BytesLen`] rather than [`Bytes`](crate::scalar_fn::Bytes) is what lets the rows
/// behind nulls be computed densely and masked afterwards, since a view carries its own length.
#[derive(Clone, Default)]
pub struct ByteLength;

impl RowFn for ByteLength {
    type Options = EmptyOptions;
    type ArgsWitness = (BytesLen,);

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.byte_length");
        *ID
    }

    fn arg_name(&self, idx: usize) -> ChildName {
        match idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {idx} for byte_length()"),
        }
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(BytesLen,), ElementSink<u64>, _, _>(
            |_| (),
            |&(), (len,), output| output.write(len as u64),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbinview::BinaryView;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::byte_length;
    use crate::expr::root;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[rstest]
    #[case(VarBinArray::from_strs(vec!["hello", "world", ""]).into_array(), vec![5u64, 5, 0])]
    #[case(VarBinArray::from_bytes(vec![b"ab".as_ref(), b"cde"]).into_array(), vec![2u64, 3])]
    #[case(VarBinArray::from_strs(vec!["Пуховички"]).into_array(), vec![18u64])]
    #[case(VarBinArray::from_bytes(vec!["Пуховички".as_ref()]).into_array(), vec![18u64])]
    fn test_bytes_byte_length(
        #[case] array: ArrayRef,
        #[case] expected_lens: Vec<u64>,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let result = array.apply(&byte_length(root()))?;
        let expected = PrimitiveArray::from_iter(expected_lens);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_varbinview_byte_length() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = VarBinViewArray::from_iter_str(["short", "a longer string here"]).into_array();
        let result = array.apply(&byte_length(root()))?;
        let expected = PrimitiveArray::from_iter(vec![5u64, 20]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_nullable_string_byte_length() -> VortexResult<()> {
        let array = VarBinArray::from_nullable_strs(vec![Some("hello"), None, Some("Пуховички")])
            .into_array();
        let result = array.apply(&byte_length(root()))?;

        let mut ctx = array_session().create_execution_ctx();
        assert!(result.is_valid(0, &mut ctx)?);
        assert!(!result.is_valid(1, &mut ctx)?);
        assert!(result.is_valid(2, &mut ctx)?);
        assert_eq!(
            result.execute_scalar(0, &mut array_session().create_execution_ctx())?,
            Scalar::primitive(5u64, Nullability::Nullable),
        );
        assert_eq!(
            result.execute_scalar(2, &mut array_session().create_execution_ctx())?,
            Scalar::primitive(18u64, Nullability::Nullable),
        );
        Ok(())
    }

    #[test]
    fn test_null_scalar_byte_length() -> VortexResult<()> {
        let null_scalar = Scalar::null(DType::Utf8(Nullability::Nullable));
        let array = ConstantArray::new(null_scalar, 2).into_array();
        let result = array.apply(&byte_length(root()))?;
        let mut ctx = array_session().create_execution_ctx();
        assert!(!result.is_valid(0, &mut ctx)?);
        assert!(!result.is_valid(1, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn test_constant_byte_length() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = ConstantArray::new(Scalar::from("hello"), 3).into_array();
        let result = array.apply(&byte_length(root()))?;
        let expected = ConstantArray::new(Scalar::primitive(5u64, Nullability::NonNullable), 3);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    /// `VarBinViewArray` only validates the views of its *valid* rows, so a legal array may hold a
    /// view behind a null row that names a buffer that does not exist. Byte length is a function of
    /// the view alone, so it must read the length rather than resolve the row.
    #[test]
    fn test_byte_length_ignores_unresolvable_views_behind_nulls() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let views = buffer![
            BinaryView::make_view(b"a longer string here", 0, 0),
            // Null row: buffer 9 does not exist and the offset is far past the end of the data.
            BinaryView::new_ref(64, *b"junk", 9, 4096),
        ];
        let array = VarBinViewArray::try_new(
            views,
            Arc::from([ByteBuffer::copy_from(b"a longer string here")]),
            DType::Utf8(Nullability::Nullable),
            Validity::from_iter([true, false]),
        )?
        .into_array();

        let result = array.apply(&byte_length(root()))?;
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(20u64), None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_display() {
        let expr = byte_length(root());
        assert_eq!(expr.to_string(), "vortex.byte_length($)");
    }
}
