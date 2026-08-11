// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnionVariants;
use vortex_array::scalar::Scalar;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::DenseUnion;
use crate::DenseUnionArray;
use crate::DenseUnionArraySlotsExt;
use crate::initialize;

fn variants() -> VortexResult<UnionVariants> {
    UnionVariants::try_new(
        ["number", "flag"].into(),
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Bool(Nullability::NonNullable),
        ],
        vec![5, 9],
    )
}

fn dense_union() -> VortexResult<DenseUnionArray> {
    DenseUnion::try_new(
        PrimitiveArray::from_iter([5u8, 9, 5]).into_array(),
        PrimitiveArray::from_iter([0i32, 0, 1]).into_array(),
        variants()?,
        vec![
            PrimitiveArray::from_iter([10i32, 30]).into_array(),
            BoolArray::from_iter([true]).into_array(),
        ],
    )
}

fn nullable_variants() -> VortexResult<UnionVariants> {
    UnionVariants::try_new(
        ["number", "optional"].into(),
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Primitive(PType::I64, Nullability::Nullable),
        ],
        vec![5, 9],
    )
}

fn nullable_dense_union() -> VortexResult<DenseUnionArray> {
    DenseUnion::try_new(
        PrimitiveArray::from_option_iter([Some(5u8), None, Some(9), Some(9)]).into_array(),
        PrimitiveArray::from_iter([0i32, 0, 0, 1]).into_array(),
        nullable_variants()?,
        vec![
            PrimitiveArray::from_iter([10i32]).into_array(),
            PrimitiveArray::from_option_iter([None, Some(40i64)]).into_array(),
        ],
    )
}

fn session() -> VortexSession {
    let session = vortex_array::array_session();
    initialize(&session);
    session
}

fn assert_rows(
    array: &ArrayRef,
    expected: Vec<Scalar>,
    session: &VortexSession,
) -> VortexResult<()> {
    let mut ctx = session.create_execution_ctx();
    for (index, expected) in expected.into_iter().enumerate() {
        assert_eq!(array.execute_scalar(index, &mut ctx)?, expected);
    }
    Ok(())
}

fn assert_same_rows(
    left: &ArrayRef,
    right: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<()> {
    assert_eq!(left.dtype(), right.dtype());
    assert_eq!(left.len(), right.len());
    let mut ctx = session.create_execution_ctx();
    for index in 0..left.len() {
        assert_eq!(
            left.execute_scalar(index, &mut ctx)?,
            right.execute_scalar(index, &mut ctx)?
        );
    }
    Ok(())
}

#[test]
fn scalar_at_uses_type_id_and_offset() -> VortexResult<()> {
    let session = session();
    let array = dense_union()?.into_array();
    assert_rows(
        &array,
        vec![
            Scalar::union(variants()?, 5, 10i32.into(), Nullability::NonNullable)?,
            Scalar::union(variants()?, 9, true.into(), Nullability::NonNullable)?,
            Scalar::union(variants()?, 5, 30i32.into(), Nullability::NonNullable)?,
        ],
        &session,
    )
}

#[test]
fn outer_and_selected_child_nulls_are_distinct() -> VortexResult<()> {
    let session = session();
    let variants = nullable_variants()?;
    let array = nullable_dense_union()?.into_array();
    assert_rows(
        &array,
        vec![
            Scalar::union(variants.clone(), 5, 10i32.into(), Nullability::Nullable)?,
            Scalar::null(DType::Union(variants.clone(), Nullability::Nullable)),
            Scalar::union(
                variants.clone(),
                9,
                Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
                Nullability::Nullable,
            )?,
            Scalar::union(
                variants,
                9,
                Scalar::primitive(40i64, Nullability::Nullable),
                Nullability::Nullable,
            )?,
        ],
        &session,
    )
}

#[test]
fn slice_filter_take_and_mask_preserve_dense_encoding() -> VortexResult<()> {
    let session = session();
    let array = dense_union()?.into_array();

    let sliced = array.slice(1..3)?;
    let filtered = array.filter(Mask::from_iter([true, false, true]))?;
    let taken = array.take(PrimitiveArray::from_iter([2u32, 0, 1]).into_array())?;
    let masked = array.mask(BoolArray::from_iter([true, false, true]).into_array())?;

    assert!(sliced.is::<DenseUnion>());
    assert!(filtered.is::<DenseUnion>());
    assert!(taken.is::<DenseUnion>());
    assert!(masked.is::<DenseUnion>());
    assert_eq!(sliced.as_::<DenseUnion>().children()[0].len(), 2);
    assert_eq!(filtered.as_::<DenseUnion>().children()[0].len(), 2);
    assert_eq!(taken.as_::<DenseUnion>().children()[0].len(), 2);
    assert_eq!(masked.as_::<DenseUnion>().children()[0].len(), 2);

    assert_rows(
        &sliced,
        vec![
            Scalar::union(variants()?, 9, true.into(), Nullability::NonNullable)?,
            Scalar::union(variants()?, 5, 30i32.into(), Nullability::NonNullable)?,
        ],
        &session,
    )?;
    assert_rows(
        &filtered,
        vec![
            Scalar::union(variants()?, 5, 10i32.into(), Nullability::NonNullable)?,
            Scalar::union(variants()?, 5, 30i32.into(), Nullability::NonNullable)?,
        ],
        &session,
    )?;
    assert_rows(
        &taken,
        vec![
            Scalar::union(variants()?, 5, 30i32.into(), Nullability::NonNullable)?,
            Scalar::union(variants()?, 5, 10i32.into(), Nullability::NonNullable)?,
            Scalar::union(variants()?, 9, true.into(), Nullability::NonNullable)?,
        ],
        &session,
    )?;
    assert_rows(
        &masked,
        vec![
            Scalar::union(variants()?, 5, 10i32.into(), Nullability::Nullable)?,
            Scalar::null(DType::Union(variants()?, Nullability::Nullable)),
            Scalar::union(variants()?, 5, 30i32.into(), Nullability::Nullable)?,
        ],
        &session,
    )
}

#[test]
fn nullable_take_indices_become_outer_nulls() -> VortexResult<()> {
    let session = session();
    let taken = dense_union()?
        .into_array()
        .take(PrimitiveArray::from_option_iter([Some(2u32), None, Some(0)]).into_array())?;
    assert!(taken.is::<DenseUnion>());
    assert_rows(
        &taken,
        vec![
            Scalar::union(variants()?, 5, 30i32.into(), Nullability::Nullable)?,
            Scalar::null(DType::Union(variants()?, Nullability::Nullable)),
            Scalar::union(variants()?, 5, 10i32.into(), Nullability::Nullable)?,
        ],
        &session,
    )
}

#[test]
fn canonicalization_uses_sparse_dictionary_children() -> VortexResult<()> {
    let session = session();
    let array = dense_union()?.into_array();
    let mut ctx = session.create_execution_ctx();
    let canonical = array.clone().execute::<UnionArray>(&mut ctx)?;

    assert!(canonical.children()[0].is::<Dict>());
    assert!(canonical.children()[1].is::<Dict>());
    assert_same_rows(&array, &canonical.into_array(), &session)
}

#[test]
fn canonicalization_handles_unselected_empty_child() -> VortexResult<()> {
    let session = session();
    let array = DenseUnion::try_new(
        PrimitiveArray::from_iter([5u8, 5]).into_array(),
        PrimitiveArray::from_iter([0i32, 1]).into_array(),
        variants()?,
        vec![
            PrimitiveArray::from_iter([10i32, 20]).into_array(),
            BoolArray::from_iter(Vec::<bool>::new()).into_array(),
        ],
    )?
    .into_array();
    let mut ctx = session.create_execution_ctx();
    let canonical = array.clone().execute::<Canonical>(&mut ctx)?.into_array();

    assert_same_rows(&array, &canonical, &session)
}

#[test]
fn invalid_type_id_and_offsets_return_errors() -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();

    let unknown_type_id = DenseUnion::try_new(
        PrimitiveArray::from_iter([7u8]).into_array(),
        PrimitiveArray::from_iter([0i32]).into_array(),
        variants()?,
        vec![
            PrimitiveArray::from_iter([10i32]).into_array(),
            BoolArray::from_iter([true]).into_array(),
        ],
    )?;
    assert!(unknown_type_id.execute_scalar(0, &mut ctx).is_err());

    let negative_offset = DenseUnion::try_new(
        PrimitiveArray::from_iter([5u8]).into_array(),
        PrimitiveArray::from_iter([-1i32]).into_array(),
        variants()?,
        vec![
            PrimitiveArray::from_iter([10i32]).into_array(),
            BoolArray::from_iter([true]).into_array(),
        ],
    )?;
    assert!(negative_offset.execute_scalar(0, &mut ctx).is_err());

    let out_of_bounds = DenseUnion::try_new(
        PrimitiveArray::from_iter([9u8]).into_array(),
        PrimitiveArray::from_iter([1i32]).into_array(),
        variants()?,
        vec![
            PrimitiveArray::from_iter([10i32]).into_array(),
            BoolArray::from_iter([true]).into_array(),
        ],
    )?;
    assert!(out_of_bounds.execute_scalar(0, &mut ctx).is_err());
    Ok(())
}

#[test]
fn validates_structural_components() -> VortexResult<()> {
    assert!(
        DenseUnion::try_new(
            PrimitiveArray::from_iter([5u16]).into_array(),
            PrimitiveArray::from_iter([0i32]).into_array(),
            variants()?,
            vec![
                PrimitiveArray::from_iter([10i32]).into_array(),
                BoolArray::from_iter([true]).into_array(),
            ],
        )
        .is_err()
    );
    assert!(
        DenseUnion::try_new(
            PrimitiveArray::from_iter([5u8]).into_array(),
            PrimitiveArray::from_iter([0u32]).into_array(),
            variants()?,
            vec![
                PrimitiveArray::from_iter([10i32]).into_array(),
                BoolArray::from_iter([true]).into_array(),
            ],
        )
        .is_err()
    );
    assert!(
        DenseUnion::try_new(
            PrimitiveArray::from_iter([5u8]).into_array(),
            PrimitiveArray::from_iter([0i32]).into_array(),
            variants()?,
            vec![PrimitiveArray::from_iter([10i32]).into_array()],
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn serde_roundtrip() -> VortexResult<()> {
    let session = session();
    let array = nullable_dense_union()?.into_array();
    let dtype = array.dtype().clone();
    let len = array.len();
    let array_ctx = ArrayContext::empty();
    let serialized = array.serialize(&array_ctx, &session, &SerializeOptions::default())?;
    let mut concat = ByteBufferMut::empty();
    for buffer in serialized {
        concat.extend_from_slice(buffer.as_ref());
    }
    let decoded = SerializedArray::try_from(concat.freeze())?.decode(
        &dtype,
        len,
        &ReadContext::new(array_ctx.to_ids()),
        &session,
    )?;

    assert!(decoded.is::<DenseUnion>());
    assert_same_rows(&array, &decoded, &session)
}
