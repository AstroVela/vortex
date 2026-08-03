// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A custom [`ArrayPlugin`] that lets you load in and deserialize a `Sparse` array of primitive
//! values as a `PatchedArray` wrapping a `Constant` array of the fill value.
//!
//! A `SparseArray` is already "a constant plus patches": every position takes the fill value
//! unless it appears in the patch indices. That is exactly the shape [`Patched`] encodes, so the
//! two are interchangeable whenever the patches can be transposed into the data-parallel layout.
//!
//! This enables zero-cost backward compatibility with previously written datasets.

use vortex_array::Array;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::Sparse;
use crate::SparseExt;

/// Custom deserialization plugin that converts a `Sparse` array of primitive values into a
/// [`Patched`] array holding a [`ConstantArray`] of the fill value.
#[derive(Debug, Clone)]
pub(crate) struct SparsePatchedPlugin;

impl ArrayPlugin for SparsePatchedPlugin {
    fn id(&self) -> ArrayId {
        // We reuse the existing `Sparse` ID so that we can take over its
        // deserialization pathway.
        ArrayVTable::id(&Sparse)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // Delegate to Sparse's metadata serde.
        Sparse.serialize(array, session)
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let sparse = Array::<Sparse>::try_from_parts(ArrayVTable::deserialize(
            &Sparse, dtype, len, metadata, buffers, children, session,
        )?)
        .map_err(|_| vortex_err!("Sparse plugin should only deserialize vortex.sparse"))?;

        let mut ctx = session.create_execution_ctx();
        let patches = sparse.patches();

        if !is_patchable(dtype, sparse.fill_scalar(), &patches, &mut ctx)? {
            return Ok(sparse.into_array());
        }

        let fill_value = sparse.fill_scalar().clone();
        let inner = ConstantArray::new(fill_value, len).into_array();

        Ok(Patched::from_array_and_patches(inner, &patches, &mut ctx)?.into_array())
    }

    fn is_supported_encoding(&self, id: &ArrayId) -> bool {
        id == ArrayVTable::id(&Patched) || id == ArrayVTable::id(&Sparse)
    }
}

/// Whether a sparse array with these parts can be represented as `Patched(Constant, patches)`.
///
/// [`Patched`] derives its validity entirely from its inner array and only rewrites the *values*
/// buffer when applying patches, so the conversion is only valid when every element of the
/// reconstructed array has the validity of the fill value. That means both the fill value and
/// every patch value must be non-null.
fn is_patchable(
    dtype: &DType,
    fill_value: &Scalar,
    patches: &Patches,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    // `Patched` is only defined over primitive values.
    if !dtype.is_primitive() {
        return Ok(false);
    }

    // A null fill would leave the inner constant all-invalid, and applying patches would not
    // clear those nulls.
    if fill_value.is_null() {
        return Ok(false);
    }

    // The transposed layout addresses patches with `u32` offsets.
    if patches.num_patches() > u32::MAX as usize {
        return Ok(false);
    }

    // Nullable patch values are only convertible if they happen to be entirely non-null.
    if patches.values().dtype().is_nullable() && !patches.values().all_valid(ctx)? {
        return Ok(false);
    }

    Ok(true)
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Patched;
    use vortex_array::arrays::PatchedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::patched::PatchedArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::Nullability::Nullable;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::SparsePatchedPlugin;
    use crate::Sparse;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session.arrays().register(SparsePatchedPlugin);
        session
    });

    /// Serialize `array` and feed the parts back through the plugin's deserializer.
    fn round_trip(array: &ArrayRef) -> VortexResult<ArrayRef> {
        let metadata = SESSION
            .array_serialize(array)?
            .ok_or_else(|| vortex_err!("expected Sparse metadata"))?;
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        SparsePatchedPlugin.deserialize(
            array.dtype(),
            array.len(),
            &metadata,
            &buffers,
            &children,
            &SESSION,
        )
    }

    fn sparse_i32(
        len: usize,
        stride: usize,
        fill: Scalar,
        values_nullability: vortex_array::dtype::Nullability,
    ) -> VortexResult<ArrayRef> {
        let indices: Buffer<u32> = (0..len as u32).step_by(stride).collect();
        let values = PrimitiveArray::new(
            indices
                .iter()
                .map(|i| -(*i as i32))
                .collect::<Buffer<i32>>(),
            match values_nullability {
                NonNullable => Validity::NonNullable,
                Nullable => Validity::AllValid,
            },
        )
        .into_array();

        Ok(Sparse::try_new(indices.into_array(), values, len, fill)?.into_array())
    }

    #[rstest]
    #[case(NonNullable)]
    #[case(Nullable)]
    fn primitive_sparse_decodes_as_patched(
        #[case] nullability: vortex_array::dtype::Nullability,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let fill = Scalar::primitive(7i32, nullability);
        let sparse = sparse_i32(4_096, 17, fill, nullability)?;

        let deserialized = round_trip(&sparse)?;

        let patched: PatchedArray = deserialized
            .clone()
            .try_downcast()
            .map_err(|a| vortex_err!("Expected Patched, got {}", a.encoding_id()))?;
        assert_eq!(
            patched.inner().encoding_id(),
            vortex_array::ArrayVTable::id(&vortex_array::arrays::Constant)
        );

        // The patched array must execute to exactly what the sparse array executes to.
        assert_arrays_eq!(&deserialized, &sparse, &mut ctx);

        Ok(())
    }

    #[test]
    fn null_fill_stays_sparse() -> VortexResult<()> {
        let fill = Scalar::null(DType::Primitive(PType::I32, Nullable));
        let sparse = sparse_i32(1_024, 9, fill, Nullable)?;

        let deserialized = round_trip(&sparse)?;

        assert_eq!(deserialized.encoding_id(), Sparse.id());
        Ok(())
    }

    #[test]
    fn null_patch_values_stay_sparse() -> VortexResult<()> {
        let len = 1_024;
        let indices: Buffer<u32> = (0..len as u32).step_by(9).collect();
        let values = PrimitiveArray::new(
            indices.iter().map(|i| *i as i32).collect::<Buffer<i32>>(),
            Validity::from_iter(indices.iter().enumerate().map(|(i, _)| i % 3 != 0)),
        )
        .into_array();
        let sparse = Sparse::try_new(
            indices.into_array(),
            values,
            len,
            Scalar::primitive(0i32, Nullable),
        )?
        .into_array();

        let deserialized = round_trip(&sparse)?;

        assert_eq!(deserialized.encoding_id(), Sparse.id());
        Ok(())
    }

    #[test]
    fn non_primitive_sparse_stays_sparse() -> VortexResult<()> {
        let indices: Buffer<u32> = (0..1_024u32).step_by(11).collect();
        let values = vortex_array::arrays::VarBinViewArray::from_iter_str(
            indices.iter().map(|i| format!("v{i}")),
        )
        .into_array();
        let sparse = Sparse::try_new(
            indices.into_array(),
            values,
            1_024,
            Scalar::utf8("fill", NonNullable),
        )?
        .into_array();

        let deserialized = round_trip(&sparse)?;

        assert_eq!(deserialized.encoding_id(), Sparse.id());
        Ok(())
    }

    #[test]
    fn supported_encodings() {
        assert!(SparsePatchedPlugin.is_supported_encoding(&Sparse.id()));
        assert!(
            SparsePatchedPlugin.is_supported_encoding(&vortex_array::ArrayVTable::id(&Patched))
        );
    }
}
