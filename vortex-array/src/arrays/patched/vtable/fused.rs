// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fused decompression for a [`Patched`] array over a [`Constant`] inner array.
//!
//! A `Patched(Constant, patches)` array is a `Sparse` array in transposed form: every position
//! takes the fill value unless it is patched. The generic [`Patched`] execution path has to
//! materialize the inner array first, which routes the constant through a [`PrimitiveBuilder`]
//! and produces an intermediate `PrimitiveArray` before the patches are scattered over its
//! buffer.
//!
//! This module fuses those two steps the way `Sparse` execution already does: allocate the
//! output buffer pre-filled with the fill value, then scatter directly into it. That skips the
//! builder, the intermediate array, and the extra trip through the executor to run the child.
//!
//! [`PrimitiveBuilder`]: crate::builders::PrimitiveBuilder

use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::apply_patches_primitive;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedSlots;
use crate::arrays::patched::PatchedSlotsView;
use crate::arrays::patched::vtable::Patched;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Execute a [`Patched`] array whose inner array is a non-null [`Constant`].
///
/// The caller must have already executed the `lane_offsets`, `patch_indices` and `patch_values`
/// slots to [`Primitive`], and must have checked that `fill_value` is non-null.
///
/// [`Constant`]: crate::arrays::Constant
pub(super) fn fused_decompress_constant(
    array: Array<Patched>,
    fill_value: &Scalar,
) -> VortexResult<ExecutionResult> {
    debug_assert!(!fill_value.is_null(), "fill value must be non-null");

    let len = array.len();
    let n_lanes = array.n_lanes();
    let offset = array.offset();

    // A non-null constant is valid at every position, and the patch values are required to be
    // non-null, so the whole output is valid.
    let validity = if array.dtype().is_nullable() {
        Validity::AllValid
    } else {
        Validity::NonNullable
    };

    let slots = match array.try_into_parts() {
        Ok(parts) => PatchedSlots::from_slots(parts.slots),
        Err(array) => PatchedSlotsView::from_slots(array.slots()).to_owned(),
    };

    let values = slots.patch_values.downcast::<Primitive>();
    let lane_offsets = slots.lane_offsets.downcast::<Primitive>();
    let patch_indices = slots.patch_indices.downcast::<Primitive>();

    let patched_values = match_each_native_ptype!(values.ptype(), |V| {
        let fill = fill_value
            .as_primitive()
            .typed_value::<V>()
            .vortex_expect("fill value must be non-null and match the patch value type");

        // Allocate the output already filled with the fill value, rather than executing the
        // inner constant into a separate array and then taking ownership of its buffer.
        let mut output = BufferMut::<V>::full(fill, len);

        apply_patches_primitive::<V>(
            &mut output,
            offset,
            len,
            n_lanes,
            lane_offsets.as_slice::<u32>(),
            patch_indices.as_slice::<u16>(),
            values.as_slice::<V>(),
        );

        PrimitiveArray::new(output.freeze(), validity)
    });

    Ok(ExecutionResult::done(patched_values.into_array()))
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Constant;
    use crate::arrays::ConstantArray;
    use crate::arrays::Patched;
    use crate::arrays::PatchedArray;
    use crate::arrays::patched::PatchedArraySlotsExt;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::patches::Patches;
    use crate::scalar::Scalar;

    const LEN: usize = 4_096;

    /// `Patched(Constant(fill), patches)` with patches at every 7th position, valued `i * 10`.
    fn patched_over_constant(fill: Scalar) -> VortexResult<PatchedArray> {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let indices: Buffer<u32> = (0..LEN as u32).step_by(7).collect();
        let values: Buffer<i32> = indices.iter().map(|i| (*i as i32) * 10).collect();
        let patches = Patches::new(LEN, 0, indices.into_array(), values.into_array(), None)?;

        let inner = ConstantArray::new(fill, LEN).into_array();
        Patched::from_array_and_patches(inner, &patches, &mut ctx)
    }

    fn expected_values(fill: i32) -> Vec<i32> {
        let mut expected = vec![fill; LEN];
        for i in (0..LEN).step_by(7) {
            expected[i] = (i as i32) * 10;
        }
        expected
    }

    #[rstest]
    #[case(NonNullable)]
    #[case(Nullable)]
    fn fused_matches_expected(#[case] nullability: Nullability) -> VortexResult<()> {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let array = patched_over_constant(Scalar::primitive(3i32, nullability))?;
        assert!(array.inner().is::<Constant>());

        let executed = array
            .into_array()
            .execute::<Canonical>(&mut ctx)?
            .into_primitive();

        assert_eq!(executed.as_slice::<i32>(), expected_values(3).as_slice());
        assert!(executed.all_valid(&mut ctx)?);

        Ok(())
    }

    /// Slicing shifts the array offset, which the fused path must honour just like the generic
    /// one: the patch at absolute index 7 lands at position 2 of a slice starting at 5.
    #[test]
    fn fused_honours_offset() -> VortexResult<()> {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let array = patched_over_constant(Scalar::primitive(3i32, NonNullable))?
            .into_array()
            .slice(5..LEN)?;
        let executed = array.execute::<Canonical>(&mut ctx)?.into_primitive();

        let expected = expected_values(3)[5..].to_vec();
        assert_eq!(executed.as_slice::<i32>(), expected.as_slice());

        Ok(())
    }

    /// A null constant keeps its all-invalid validity, so it must fall through to the generic
    /// path rather than being fused into an all-valid output.
    #[test]
    fn null_constant_falls_back() -> VortexResult<()> {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let fill = Scalar::null(DType::Primitive(PType::I32, Nullable));
        let array = patched_over_constant(fill)?;
        let executed = array
            .into_array()
            .execute::<Canonical>(&mut ctx)?
            .into_primitive();

        // Every position reads as null, patched ones included. This is exactly why the Sparse
        // deserialize plugin refuses to convert null-filled sparse arrays.
        assert!(
            executed
                .validity()?
                .execute_mask(LEN, &mut ctx)?
                .all_false()
        );

        Ok(())
    }

    /// The fused path must not engage for a non-constant inner array.
    #[test]
    fn non_constant_inner_unaffected() -> VortexResult<()> {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let inner = buffer![0i32; 1024].into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![9i32; 3].into_array(),
            None,
        )?;
        let array = Patched::from_array_and_patches(inner, &patches, &mut ctx)?.into_array();

        let executed = array.execute::<Canonical>(&mut ctx)?.into_primitive();

        let mut expected = vec![0i32; 1024];
        expected[1] = 9;
        expected[2] = 9;
        expected[3] = 9;
        assert_eq!(executed.as_slice::<i32>(), expected.as_slice());

        Ok(())
    }
}
