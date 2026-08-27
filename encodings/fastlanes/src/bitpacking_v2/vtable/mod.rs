// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
use vortex_array::patches::PatchesMetadata;
use vortex_array::require_patches;
use vortex_array::require_validity;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BitPackedV2ArrayExt;
use crate::bitpacking_v2::array::BitPackedV2Data;
use crate::bitpacking_v2::array::BitPackedV2DataParts;
use crate::bitpacking_v2::array::BitPackedV2Slots;
use crate::bitpacking_v2::array::BitPackedV2SlotsView;
use crate::bitpacking_v2::array::PATCH_SLOTS;
use crate::bitpacking_v2::array::decompress::unpack_v2_array;
use crate::bitpacking_v2::array::decompress::unpack_v2_into_primitive_builder;
use crate::bitpacking_v2::vtable::rules::RULES;

mod kernels;
mod operations;
mod rules;
mod validity;

/// A [`BitPackedV2`]-encoded Vortex array.
pub type BitPackedV2Array = Array<BitPackedV2>;

pub(crate) fn initialize(session: &VortexSession) {
    kernels::initialize(session);
}

#[derive(Clone, prost::Message)]
pub struct BitPackedV2Metadata {
    #[prost(uint32, tag = "1")]
    pub(crate) offset: u32, // must be <1024
    #[prost(message, optional, tag = "2")]
    pub(crate) patches: Option<PatchesMetadata>,
}

impl ArrayHash for BitPackedV2Data {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.offset.hash(state);
        self.bit_widths.array_hash(state, accuracy);
        self.packed.array_hash(state, accuracy);
        self.patches_data.hash(state);
    }
}

impl ArrayEq for BitPackedV2Data {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.offset == other.offset
            && self.bit_widths.array_eq(&other.bit_widths, accuracy)
            && self.packed.array_eq(&other.packed, accuracy)
            && self.patches_data == other.patches_data
    }
}

impl VTable for BitPackedV2 {
    type TypedArrayData = BitPackedV2Data;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("fastlanes.bitpacked_v2");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let bp_slots = BitPackedV2SlotsView::from_slots(slots);

        let validity = child_to_validity(bp_slots.validity_child, dtype.nullability());
        let patches =
            PatchesData::patches_from_slots(data.patches_data.as_ref(), len, slots, PATCH_SLOTS);
        data.validate(dtype.as_ptype(), &validity, patches.as_ref(), len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        2
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.packed().clone(),
            1 => array.bit_widths_buffer().clone(),
            _ => vortex_panic!("BitPackedV2Array buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("packed".to_string()),
            1 => Some("bit_widths".to_string()),
            _ => None,
        }
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 2,
            "Expected 2 buffers, got {}",
            buffers.len()
        );
        let data = BitPackedV2Data::try_new(
            buffers[0].clone(),
            buffers[1].clone(),
            array.patches(),
            array.offset(),
        )?;
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            BitPackedV2Metadata {
                offset: array.offset() as u32,
                patches: array
                    .patches()
                    .map(|p| p.to_metadata(array.len(), array.dtype()))
                    .transpose()?,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = BitPackedV2Metadata::decode(metadata)?;
        if buffers.len() != 2 {
            vortex_bail!("Expected 2 buffers, got {}", buffers.len());
        }
        let packed = buffers[0].clone();
        let bit_widths = buffers[1].clone();

        let load_validity = |child_idx: usize| {
            if children.len() == child_idx {
                Ok(Validity::from(dtype.nullability()))
            } else if children.len() == child_idx + 1 {
                let validity = children.get(child_idx, &Validity::DTYPE, len)?;
                Ok(Validity::Array(validity))
            } else {
                vortex_bail!(
                    "Expected {} or {} children, got {}",
                    child_idx,
                    child_idx + 1,
                    children.len()
                );
            }
        };

        let validity_idx = match &metadata.patches {
            None => 0,
            Some(patches_meta) if patches_meta.chunk_offsets_dtype()?.is_some() => 3,
            Some(_) => 2,
        };

        let validity = load_validity(validity_idx)?;

        let patches = metadata
            .patches
            .map(|p| {
                let indices = children.get(0, &p.indices_dtype()?, p.len()?)?;
                let values = children.get(1, dtype, p.len()?)?;
                let chunk_offsets = p
                    .chunk_offsets_dtype()?
                    .map(|dtype| children.get(2, &dtype, p.chunk_offsets_len() as usize))
                    .transpose()?;

                Patches::new(len, p.offset()?, indices, values, chunk_offsets)
            })
            .transpose()?;

        let slots = {
            let mut s = ArraySlots::with_capacity(BitPackedV2Slots::COUNT);
            PatchesData::push_slots(&mut s, patches.as_ref());
            s.push(validity_to_child(&validity, len));
            s
        };
        let data = BitPackedV2Data::try_new(
            packed,
            bit_widths,
            patches,
            u16::try_from(metadata.offset).map_err(|_| {
                vortex_err!(
                    "BitPackedV2Metadata offset {} does not fit in u16",
                    metadata.offset
                )
            })?,
        )?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
            unpack_v2_into_primitive_builder::<T>(
                array,
                builder
                    .as_any_mut()
                    .downcast_mut()
                    .vortex_expect("bit packed array must canonicalize into a primitive array"),
                ctx,
            )
        })
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        BitPackedV2Slots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        require_patches!(
            array,
            BitPackedV2Slots::PATCH_INDICES,
            BitPackedV2Slots::PATCH_VALUES,
            BitPackedV2Slots::PATCH_CHUNK_OFFSETS
        );
        require_validity!(array, BitPackedV2Slots::VALIDITY_CHILD);

        Ok(ExecutionResult::done(
            unpack_v2_array(array.as_view(), ctx)?.into_array(),
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

/// Bit-packing that gives every 1024-element FastLanes chunk its own bit width.
///
/// [`BitPacked`](crate::BitPacked) packs a whole array at a single width, so one wide region
/// forces every chunk to pay for it (or to patch around it). `BitPackedV2` instead stores one
/// width per chunk and concatenates the variable-width blocks, so chunks whose values are small
/// shrink independently of the rest of the array. Chunk blocks are always a whole multiple of 128
/// bytes, which keeps every chunk aligned for its packed type.
#[derive(Clone, Debug)]
pub struct BitPackedV2;

impl BitPackedV2 {
    #[expect(clippy::too_many_arguments)]
    pub fn try_new(
        packed: BufferHandle,
        bit_widths: BufferHandle,
        ptype: PType,
        validity: Validity,
        patches: Option<Patches>,
        len: usize,
        offset: u16,
    ) -> VortexResult<BitPackedV2Array> {
        let dtype = DType::Primitive(ptype, validity.nullability());
        let slots = {
            let mut s = ArraySlots::with_capacity(BitPackedV2Slots::COUNT);
            PatchesData::push_slots(&mut s, patches.as_ref());
            s.push(validity_to_child(&validity, len));
            s
        };
        let data = BitPackedV2Data::try_new(packed, bit_widths, patches, offset)?;
        Array::try_from_parts(ArrayParts::new(BitPackedV2, dtype, len, data).with_slots(slots))
    }

    pub fn into_parts(array: BitPackedV2Array) -> BitPackedV2DataParts {
        let len = array.len();
        let patches = array.patches();
        let validity = array.validity().vortex_expect("BitPackedV2 validity");
        let data = array.into_data();
        BitPackedV2DataParts {
            offset: data.offset,
            len,
            packed: data.packed,
            bit_widths: data.bit_widths,
            patches,
            validity,
        }
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::test_harness::check_metadata;

    use super::BitPackedV2Metadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_bitpacked_v2_metadata() {
        check_metadata(
            "bitpacked_v2.metadata",
            &BitPackedV2Metadata {
                offset: u32::MAX,
                patches: None,
            }
            .encode_to_vec(),
        );
    }
}
