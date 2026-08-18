// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::ReadContext;

use crate::ArrayContext;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::VarBinViewData;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

/// A single 17 byte buffer, referenced by the views built below.
fn data_buffers() -> Arc<[ByteBuffer]> {
    Arc::new([ByteBuffer::from(b"hello world ololo".to_vec())])
}

/// A view referencing the first 14 bytes of buffer 0.
fn valid_view() -> BinaryView {
    BinaryView::new_ref(14, *b"hell", 0, 0)
}

/// A view referencing a buffer that does not exist, at an out of bounds offset.
fn invalid_view() -> BinaryView {
    BinaryView::new_ref(13, *b"AAAA", 0xDEAD_BEEF, 0xF000_0000)
}

fn nullable_utf8() -> DType {
    DType::Utf8(Nullability::Nullable)
}

fn validity_from(values: impl IntoIterator<Item = bool>) -> Validity {
    Validity::from_bit_buffer(BitBuffer::from_iter(values), Nullability::Nullable)
}

#[test]
pub fn varbin_view() {
    let mut ctx = array_session().create_execution_ctx();
    let binary_arr =
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"]);
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"]),
        &mut ctx
    );
}

#[test]
pub fn slice_array() {
    let mut ctx = array_session().create_execution_ctx();
    let binary_arr =
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"])
            .slice(1..2)
            .unwrap();
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["hello world this is a long string"]),
        &mut ctx
    );
}

#[test]
pub fn flatten_array() {
    let mut ctx = array_session().create_execution_ctx();
    let binary_arr = VarBinViewArray::from_iter_str(["string1", "string2"]);
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["string1", "string2"]),
        &mut ctx
    );
}

#[test]
pub fn binary_view_size_and_alignment() {
    assert_eq!(size_of::<BinaryView>(), 16);
    assert_eq!(align_of::<BinaryView>(), 16);
}

#[test]
fn validate_or_fix_owned_views_in_place() -> VortexResult<()> {
    let views = Buffer::copy_from(vec![valid_view(), invalid_view()]);
    let views_ptr = views.as_ptr();

    let fixed = VarBinViewData::validate_or_fix(
        views,
        &data_buffers(),
        &nullable_utf8(),
        &validity_from([true, false]),
    )?;

    assert_eq!(fixed.as_ptr(), views_ptr, "views should be fixed in place");
    assert_eq!(fixed[0], valid_view());
    assert_eq!(fixed[1], BinaryView::empty_view());
    Ok(())
}

#[test]
fn validate_or_fix_shared_views_copies() -> VortexResult<()> {
    let views = Buffer::copy_from(vec![valid_view(), invalid_view(), valid_view()]);

    let fixed = VarBinViewData::validate_or_fix(
        views.clone(),
        &data_buffers(),
        &nullable_utf8(),
        &validity_from([true, false, true]),
    )?;

    assert_ne!(
        fixed.as_ptr(),
        views.as_ptr(),
        "shared views must be copied"
    );
    assert_eq!(
        fixed.as_slice(),
        [valid_view(), BinaryView::empty_view(), valid_view()]
    );
    assert_eq!(
        views[1],
        invalid_view(),
        "the shared views are left untouched"
    );
    Ok(())
}

#[test]
fn validate_or_fix_shared_valid_views_are_not_copied() -> VortexResult<()> {
    let views = Buffer::copy_from(vec![valid_view(), valid_view()]);

    let fixed = VarBinViewData::validate_or_fix(
        views.clone(),
        &data_buffers(),
        &nullable_utf8(),
        &validity_from([true, false]),
    )?;

    assert_eq!(
        fixed.as_ptr(),
        views.as_ptr(),
        "valid views must not be copied"
    );
    Ok(())
}

#[rstest]
#[case::owned(false)]
#[case::shared(true)]
fn validate_or_fix_rejects_invalid_non_null_view(#[case] shared: bool) {
    let views = Buffer::copy_from(vec![valid_view(), invalid_view()]);
    let _shared = shared.then(|| views.clone());

    let result = VarBinViewData::validate_or_fix(
        views,
        &data_buffers(),
        &nullable_utf8(),
        &validity_from([true, true]),
    );

    assert!(matches!(result, Err(VortexError::InvalidArgument(_, _))));
}

#[rstest]
#[case::owned(false)]
#[case::shared(true)]
fn validate_or_fix_all_invalid(#[case] shared: bool) -> VortexResult<()> {
    let views = Buffer::copy_from(vec![invalid_view(), invalid_view()]);
    let _shared = shared.then(|| views.clone());

    let fixed = VarBinViewData::validate_or_fix(
        views,
        &data_buffers(),
        &nullable_utf8(),
        &Validity::AllInvalid,
    )?;

    assert!(fixed.iter().all(|view| *view == BinaryView::empty_view()));
    Ok(())
}

#[rstest]
#[case::owned(false)]
#[case::shared(true)]
fn validate_or_fix_alternating_validity_runs(#[case] shared: bool) -> VortexResult<()> {
    let views = Buffer::copy_from(vec![
        valid_view(),
        invalid_view(),
        valid_view(),
        valid_view(),
        invalid_view(),
        valid_view(),
    ]);
    let _shared = shared.then(|| views.clone());

    let fixed = VarBinViewData::validate_or_fix(
        views,
        &data_buffers(),
        &nullable_utf8(),
        &validity_from([true, false, false, true, false, true]),
    )?;

    assert_eq!(
        fixed.as_slice(),
        [
            valid_view(),
            BinaryView::empty_view(),
            valid_view(),
            valid_view(),
            BinaryView::empty_view(),
            valid_view(),
        ]
    );
    Ok(())
}

#[rstest]
#[case::owned(false)]
#[case::shared(true)]
fn validate_or_fix_rejects_invalid_non_null_view_after_a_fix(#[case] shared: bool) {
    let views = Buffer::copy_from(vec![invalid_view(), valid_view(), invalid_view()]);
    let _shared = shared.then(|| views.clone());

    // The first malformed view is under a null and is fixed, the second one is not.
    let result = VarBinViewData::validate_or_fix(
        views,
        &data_buffers(),
        &nullable_utf8(),
        &validity_from([false, true, true]),
    );

    assert!(matches!(result, Err(VortexError::InvalidArgument(_, _))));
}

#[test]
fn decode_fixes_invalid_null_views() -> VortexResult<()> {
    let views = Buffer::copy_from(vec![valid_view(), invalid_view()]);
    let dtype = nullable_utf8();
    let session = array_session();

    // SAFETY: the view of the null slot is deliberately malformed, mimicking an array written by
    // another producer. It is never read as a value, only serialized and decoded again.
    let array = unsafe {
        VarBinViewArray::new_unchecked(
            views.clone(),
            data_buffers(),
            dtype.clone(),
            validity_from([true, false]),
        )
    };

    let array_ctx = ArrayContext::empty();
    let serialized =
        array
            .clone()
            .into_array()
            .serialize(&array_ctx, &session, &SerializeOptions::default())?;

    let mut concat = ByteBufferMut::empty();
    for buf in serialized {
        concat.extend_from_slice(buf.as_ref());
    }
    let parts = SerializedArray::try_from(concat.freeze())?;
    let decoded = parts.decode(
        &dtype,
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &session,
    )?;

    let decoded = decoded
        .as_opt::<VarBinView>()
        .ok_or_else(|| vortex_err!("expected VarBinView"))?;
    assert_eq!(decoded.views()[0], views[0]);
    assert_eq!(decoded.views()[1], BinaryView::empty_view());
    Ok(())
}
