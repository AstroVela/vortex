// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operations;
mod slice;

use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::ValidityChild;
use crate::array::ValidityVTableFromChild;
use crate::array::with_empty_buffers;
use crate::arrays::Dict;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::patches::PATCH_BLOCK_SIZE;
use crate::arrays::patches::PatchFn;
use crate::arrays::patches::PatchesArrayData;
use crate::arrays::patches::PatchesArrayExt;
use crate::arrays::patches::PatchesArraySlotsExt;
use crate::arrays::patches::PatchesSlots;
use crate::arrays::patches::PatchesSlotsView;
use crate::arrays::patches::apply_patches_primitive;
use crate::arrays::patches::compute::rules::PARENT_RULES;
use crate::arrays::primitive::PrimitiveDataParts;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::require_child;
use crate::serde::ArrayChildren;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Dict.id(), Patches, TakeExecuteAdaptor(Patches));
}

#[derive(Clone, Debug)]
pub struct Patches;

impl ValidityChild<Patches> for Patches {
    fn validity_child(array: ArrayView<'_, Patches>) -> ArrayRef {
        array.inner().clone()
    }
}

#[derive(Clone, prost::Message)]
pub struct PatchesArrayMetadata {
    /// The total number of patches, and the length of the indices and values child arrays.
    #[prost(uint32, tag = "1")]
    pub(crate) n_patches: u32,

    /// An offset into the first block's patches that should be considered in-view.
    ///
    /// Always between 0 and 1023.
    #[prost(uint32, tag = "2")]
    pub(crate) offset: u32,

    /// The combine function applied between base and patch values, see [`PatchFn`].
    #[prost(uint32, tag = "3")]
    pub(crate) patch_fn: u32,
}

impl ArrayHash for PatchesArrayData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
        self.patch_fn.hash(state);
    }
}

impl ArrayEq for PatchesArrayData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset && self.patch_fn == other.patch_fn
    }
}

impl VTable for Patches {
    type TypedArrayData = PatchesArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.patches");
        *ID
    }

    fn validate(
        &self,
        data: &PatchesArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        data.validate(dtype, len, &PatchesSlotsView::from_slots(slots))
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("invalid buffer index for PatchesArray: {idx}");
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("invalid buffer index for PatchesArray: {idx}");
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            PatchesArrayMetadata {
                n_patches: u32::try_from(array.indices().len())?,
                offset: u32::try_from(array.offset())?,
                patch_fn: array.patch_fn() as u32,
            }
            .encode_to_vec(),
        ))
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
        let metadata = PatchesArrayMetadata::decode(metadata)?;
        let n_patches = metadata.n_patches as usize;
        let offset = metadata.offset as usize;
        let patch_fn = PatchFn::try_from(metadata.patch_fn)?;

        // After slicing when offset > 0, there may be additional blocks.
        let n_blocks = (len + offset).div_ceil(PATCH_BLOCK_SIZE);

        let inner = children.get(0, dtype, len)?;
        let skip_indices = children.get(1, PType::U32.into(), n_blocks + 1)?;
        let indices = children.get(2, PType::U16.into(), n_patches)?;
        let values = children.get(3, dtype, n_patches)?;

        let data = PatchesArrayData { offset, patch_fn };
        let slots = PatchesSlots {
            inner,
            skip_indices,
            indices,
            values,
        }
        .into_slots();
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let dtype = array.array().dtype();

        if !dtype.is_primitive() {
            // Default pathway: canonicalize and propagate.
            let canonical = array
                .array()
                .clone()
                .execute::<Canonical>(ctx)?
                .into_array();
            return canonical.append_to_builder(builder, ctx);
        }

        let ptype = dtype.as_ptype();
        let len = array.len();

        array.inner().append_to_builder(builder, ctx)?;

        let offset = array.offset();
        let patch_fn = array.patch_fn();
        let skip_indices = array
            .skip_indices()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let indices = array.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = array.values().clone().execute::<PrimitiveArray>(ctx)?;

        match_each_native_ptype!(ptype, |V| {
            let typed_builder = builder
                .as_any_mut()
                .downcast_mut::<PrimitiveBuilder<V>>()
                .vortex_expect("correctly typed builder");

            // Combine into the last `len` elements of the builder. These would have been
            // populated by the inner.append_to_builder() call above.
            let output = typed_builder.values_mut();
            let trailer = output.len() - len;

            apply_patches_primitive::<V>(
                &mut output[trailer..],
                offset,
                len,
                skip_indices.as_slice::<u32>(),
                indices.as_slice::<u16>(),
                values.as_slice::<V>(),
                patch_fn,
            );
        });

        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        PatchesSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = require_child!(array, array.inner(), PatchesSlots::INNER => Primitive);
        let array =
            require_child!(array, array.skip_indices(), PatchesSlots::SKIP_INDICES => Primitive);
        let array = require_child!(array, array.indices(), PatchesSlots::INDICES => Primitive);
        let array = require_child!(array, array.values(), PatchesSlots::VALUES => Primitive);

        let len = array.len();
        let offset = array.offset;
        let patch_fn = array.patch_fn;
        let slots = match array.try_into_parts() {
            Ok(parts) => PatchesSlots::from_slots(parts.slots),
            Err(array) => PatchesSlotsView::from_slots(array.slots()).to_owned(),
        };

        let PrimitiveDataParts {
            buffer,
            ptype,
            validity,
        } = slots.inner.downcast::<Primitive>().into_data_parts();

        let skip_indices = slots.skip_indices.downcast::<Primitive>();
        let indices = slots.indices.downcast::<Primitive>();
        let values = slots.values.downcast::<Primitive>();

        let patched_values = match_each_native_ptype!(values.ptype(), |V| {
            let mut output = Buffer::<V>::from_byte_buffer(buffer.unwrap_host()).into_mut();

            apply_patches_primitive::<V>(
                &mut output,
                offset,
                len,
                skip_indices.as_slice::<u32>(),
                indices.as_slice::<u16>(),
                values.as_slice::<V>(),
                patch_fn,
            );

            let output = output.freeze();

            PrimitiveArray::from_byte_buffer(output.into_byte_buffer(), ptype, validity)
        });

        Ok(ExecutionResult::done(patched_values.into_array()))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_buffer::buffer_mut;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use crate::ArrayContext;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Patches;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::patches::PatchFn;
    use crate::arrays::patches::PatchesArray;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
    use crate::session::ArraySessionExt;
    use crate::validity::Validity;

    fn make_patches_array(
        inner: impl IntoIterator<Item = u16>,
        patch_indices: &[u32],
        patch_values: &[u16],
        patch_fn: PatchFn,
    ) -> VortexResult<PatchesArray> {
        let values: Vec<u16> = inner.into_iter().collect();
        let len = values.len();
        let array = PrimitiveArray::from_iter(values).into_array();

        let indices = PrimitiveArray::from_iter(patch_indices.iter().copied()).into_array();
        let patch_vals = PrimitiveArray::from_iter(patch_values.iter().copied()).into_array();

        let patches = crate::patches::Patches::new(len, 0, indices, patch_vals, None)?;

        let mut ctx = array_session().create_execution_ctx();
        Patches::from_array_and_patches(array, &patches, patch_fn, &mut ctx)
    }

    #[test]
    fn test_execute() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = make_patches_array(
            vec![0u16; 1024],
            &[1, 2, 3],
            &[10, 20, 30],
            PatchFn::Overwrite,
        )?
        .into_array();

        let executed = array
            .execute::<Canonical>(&mut ctx)?
            .into_primitive()
            .into_buffer::<u16>();

        let mut expected = buffer_mut![0u16; 1024];
        expected[1] = 10;
        expected[2] = 20;
        expected[3] = 30;

        assert_eq!(executed, expected.freeze());
        Ok(())
    }

    #[rstest]
    #[case::add(PatchFn::Add, [5u16, 15, 5, 5])]
    #[case::or(PatchFn::Or, [5u16, 15, 5, 5])]
    #[case::overwrite(PatchFn::Overwrite, [5u16, 10, 5, 5])]
    fn test_execute_combine(
        #[case] patch_fn: PatchFn,
        #[case] expected: [u16; 4],
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Base 5 = 0b101, patch 10 = 0b1010: add => 15, or => 15, overwrite => 10.
        let array = make_patches_array([5u16, 5, 5, 5], &[1], &[10], patch_fn)?.into_array();

        let executed = array.execute::<Canonical>(&mut ctx)?.into_primitive();
        assert_eq!(executed.as_slice::<u16>(), &expected);
        Ok(())
    }

    #[test]
    fn test_execute_multi_block_sliced() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = make_patches_array(
            vec![0u16; 4096],
            &[100, 1500, 2500, 3500],
            &[11, 22, 33, 44],
            PatchFn::Overwrite,
        )?
        .into_array()
        .slice(1200..2600)?;

        let executed = array.execute::<Canonical>(&mut ctx)?.into_primitive();

        let mut expected = buffer_mut![0u16; 1400];
        expected[1500 - 1200] = 22;
        expected[2500 - 1200] = 33;

        assert_eq!(executed.into_buffer::<u16>(), expected.freeze());
        Ok(())
    }

    #[test]
    fn test_append_to_builder() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = make_patches_array(
            vec![0u16; 1024],
            &[1, 2, 3],
            &[10, 20, 30],
            PatchFn::Overwrite,
        )?
        .into_array()
        .slice(3..1024)?;

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array.append_to_builder(builder.as_mut(), &mut ctx)?;
        let result = builder.finish();

        let mut expected = buffer_mut![0u16; 1021];
        expected[0] = 30;
        let expected = expected.into_array();

        assert_arrays_eq!(expected, result, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_scalar_at_combine() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = make_patches_array([5u16, 5, 5, 5], &[1], &[10], PatchFn::Add)?.into_array();

        assert_eq!(array.execute_scalar(0, &mut ctx)?, 5u16.into());
        assert_eq!(array.execute_scalar(1, &mut ctx)?, 15u16.into());
        Ok(())
    }

    #[test]
    fn test_scalar_at_multi_block() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = make_patches_array(
            vec![0u16; 4096],
            &[100, 1500, 2500, 3500],
            &[11, 22, 33, 44],
            PatchFn::Overwrite,
        )?
        .into_array();

        for (index, expected) in [(100, 11u16), (1500, 22), (2500, 33), (3500, 44), (99, 0)] {
            assert_eq!(array.execute_scalar(index, &mut ctx)?, expected.into());
        }
        Ok(())
    }

    #[test]
    fn test_scalar_at_sliced() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = make_patches_array(
            vec![0u16; 4096],
            &[100, 1500, 2500, 3500],
            &[11, 22, 33, 44],
            PatchFn::Overwrite,
        )?
        .into_array()
        .slice(1200..2600)?;

        assert_eq!(array.execute_scalar(1500 - 1200, &mut ctx)?, 22u16.into());
        assert_eq!(array.execute_scalar(2500 - 1200, &mut ctx)?, 33u16.into());
        assert_eq!(array.execute_scalar(0, &mut ctx)?, 0u16.into());
        Ok(())
    }

    #[test]
    fn test_execute_with_validity() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let validity = Validity::from_iter((0..10).map(|i| i != 0 && i != 5));
        let inner = PrimitiveArray::new(buffer![0u16; 10], validity).into_array();

        let patches = crate::patches::Patches::new(
            10,
            0,
            buffer![1u32, 2, 3].into_array(),
            PrimitiveArray::new(buffer![10u16, 20, 30], Validity::AllValid).into_array(),
            None,
        )?;

        let array = Patches::from_array_and_patches(inner, &patches, PatchFn::Overwrite, &mut ctx)?
            .into_array();

        let expected = PrimitiveArray::from_option_iter([
            None,
            Some(10u16),
            Some(20),
            Some(30),
            Some(0),
            None,
            Some(0),
            Some(0),
            Some(0),
            Some(0),
        ])
        .into_array();

        assert_arrays_eq!(expected, array, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::basic(
        make_patches_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30], PatchFn::Overwrite).unwrap().into_array()
    )]
    #[case::multi_block(
        make_patches_array(vec![0u16; 4096], &[100, 1500, 2500, 3500], &[11, 22, 33, 44], PatchFn::Add).unwrap().into_array()
    )]
    #[case::sliced({
        let arr = make_patches_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30], PatchFn::Overwrite).unwrap();
        arr.into_array().slice(2..1024).unwrap()
    })]
    fn test_serde_roundtrip(#[case] array: crate::ArrayRef) {
        let dtype = array.dtype().clone();
        let len = array.len();

        let session = array_session();
        session.arrays().register(Patches);

        let ctx = ArrayContext::empty().with_allowed_ids(
            session
                .arrays()
                .registry()
                .read(|map| map.keys().copied().collect()),
        );
        let serialized = array
            .serialize(&ctx, &session, &SerializeOptions::default())
            .unwrap();

        // Concat into a single buffer.
        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = SerializedArray::try_from(concat).unwrap();
        let decoded = parts
            .decode(&dtype, len, &ReadContext::new(ctx.to_ids()), &session)
            .unwrap();

        assert!(decoded.is::<Patches>());
        assert_eq!(
            array.display_values().to_string(),
            decoded.display_values().to_string()
        );
    }

    #[test]
    fn test_slice_execute_equivalence() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = make_patches_array(
            vec![1u16; 10_000],
            &[0, 1, 2, 3, 4, 16, 17, 18, 19, 1024, 2048, 2049],
            &[u16::MAX; 12],
            PatchFn::Overwrite,
        )?
        .into_array();

        let slice_first = array
            .slice(1024..5000)?
            .execute::<Canonical>(&mut ctx)?
            .into_array();
        let slice_last = array
            .execute::<Canonical>(&mut ctx)?
            .into_primitive()
            .slice(1024..5000)?;

        assert_arrays_eq!(slice_first, slice_last, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_stacked_slices() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let inner = PrimitiveArray::from_iter(0u16..10_000).into_array();

        let patches = crate::patches::Patches::new(
            10_000,
            0,
            buffer![1u32, 2, 1024, 2048, 3072, 3088].into_array(),
            buffer![0u16, 1, 2, 3, 4, 5].into_array(),
            None,
        )?;

        let array = Patches::from_array_and_patches(inner, &patches, PatchFn::Overwrite, &mut ctx)?
            .into_array();

        let sliced = array
            .slice(1024..5000)?
            .slice(1..2065)?
            .execute::<Canonical>(&mut ctx)?
            .into_array();

        let mut expected = vortex_buffer::BufferMut::from_iter(1025u16..=3088);
        expected[1023] = 3;
        expected[2047] = 4;
        expected[2063] = 5;
        let expected = expected.into_array();

        assert_arrays_eq!(expected, sliced, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_patch_fn_rejected_for_floats() {
        let mut ctx = array_session().create_execution_ctx();
        let inner = PrimitiveArray::from_iter([1.0f32, 2.0, 3.0]).into_array();
        let patches = crate::patches::Patches::new(
            3,
            0,
            buffer![1u32].into_array(),
            buffer![10.0f32].into_array(),
            None,
        )
        .unwrap();

        let result = Patches::from_array_and_patches(inner, &patches, PatchFn::Add, &mut ctx);
        assert!(result.is_err());
    }
}
