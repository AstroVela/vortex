// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::EmptyArrayData;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::OperationsVTable;
use vortex_array::TypedArrayRef;
use vortex_array::VTable;
use vortex_array::ValidityVTable;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnionVariants;
use vortex_array::require_child;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::with_empty_buffers;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::canonical::canonicalize;
use crate::rules::PARENT_RULES;

const OFFSETS_DTYPE: DType = DType::Primitive(PType::I32, Nullability::NonNullable);

/// A [`DenseUnion`]-encoded Vortex array.
pub type DenseUnionArray = Array<DenseUnion>;

/// Slot layout of a dense union array.
#[array_slots(DenseUnion)]
pub struct DenseUnionSlots {
    /// The row-aligned type IDs selecting union variants.
    #[slot(0)]
    pub type_ids: ArrayRef,
    /// The row-aligned offsets into the selected compact child.
    #[slot(1)]
    pub offsets: ArrayRef,
    /// The compact children in variant order.
    #[slot(2..)]
    pub children: Vec<ArrayRef>,
}

#[derive(Clone, prost::Message)]
struct DenseUnionMetadata {
    #[prost(uint64, repeated, tag = "1")]
    child_lengths: Vec<u64>,
}

/// Concrete parts of a [`DenseUnionArray`].
pub struct DenseUnionDataParts {
    /// The union variant schema.
    pub variants: UnionVariants,
    /// The row-aligned type IDs.
    pub type_ids: ArrayRef,
    /// The row-aligned compact-child offsets.
    pub offsets: ArrayRef,
    /// The compact children in variant order.
    pub children: Vec<ArrayRef>,
}

fn make_parts(
    type_ids: ArrayRef,
    offsets: ArrayRef,
    variants: UnionVariants,
    children: impl IntoIterator<Item = ArrayRef>,
) -> ArrayParts<DenseUnion> {
    let len = type_ids.len();
    let nullability = type_ids.dtype().nullability();
    let children = children.into_iter();
    let (lower, _) = children.size_hint();
    let mut slots = ArraySlots::with_capacity(DenseUnionSlots::CHILDREN_OFFSET + lower);
    slots.push(Some(type_ids));
    slots.push(Some(offsets));
    slots.extend(children.map(Some));

    ArrayParts::new(
        DenseUnion,
        DType::Union(variants, nullability),
        len,
        EmptyArrayData,
    )
    .with_slots(slots)
}

/// Accessors for a dense union array.
pub trait DenseUnionArrayExt: DenseUnionArraySlotsExt {
    /// Return the union's variant schema.
    fn variants(&self) -> &UnionVariants {
        match self.as_ref().dtype() {
            DType::Union(variants, _) => variants,
            _ => unreachable!("DenseUnionArrayExt requires a union dtype"),
        }
    }

    /// Iterate over compact children in variant order.
    fn iter_children(&self) -> impl ExactSizeIterator<Item = &ArrayRef> + '_ {
        self.children().iter()
    }

    /// Return a compact child by variant index.
    fn child(&self, index: usize) -> Option<&ArrayRef> {
        self.children().get(index)
    }

    /// Return a compact child selected by a data-level type ID.
    fn child_by_type_id(&self, type_id: u8) -> Option<&ArrayRef> {
        self.child(self.variants().tag_to_child_index(type_id)?)
    }

    /// Return a compact child selected by variant name.
    fn child_by_name_opt(&self, name: impl AsRef<str>) -> Option<&ArrayRef> {
        self.child(self.variants().find(name)?)
    }

    /// Return a compact child selected by variant name.
    fn child_by_name(&self, name: impl AsRef<str>) -> VortexResult<&ArrayRef> {
        let name = name.as_ref();
        self.child_by_name_opt(name).ok_or_else(|| {
            vortex_err!(
                "Variant {name} not found in dense union array with names {:?}",
                self.variants().names()
            )
        })
    }
}

impl<T: TypedArrayRef<DenseUnion>> DenseUnionArrayExt for T {}

/// The dense physical encoding for the logical [`DType::Union`] type.
#[derive(Clone, Debug)]
pub struct DenseUnion;

impl DenseUnion {
    /// Construct a dense union array.
    ///
    /// # Panics
    ///
    /// Panics if the components do not satisfy the invariants documented by [`Self::try_new`].
    pub fn new(
        type_ids: ArrayRef,
        offsets: ArrayRef,
        variants: UnionVariants,
        children: impl IntoIterator<Item = ArrayRef>,
    ) -> DenseUnionArray {
        Self::try_new(type_ids, offsets, variants, children)
            .vortex_expect("DenseUnion construction failed")
    }

    /// Try to construct a dense union array.
    ///
    /// Type IDs and offsets are structurally validated, but their individual values are checked
    /// only when the array is accessed or converted to its canonical sparse representation.
    pub fn try_new(
        type_ids: ArrayRef,
        offsets: ArrayRef,
        variants: UnionVariants,
        children: impl IntoIterator<Item = ArrayRef>,
    ) -> VortexResult<DenseUnionArray> {
        Array::try_from_parts(make_parts(type_ids, offsets, variants, children))
    }
}

/// Owned accessors for a dense union array.
pub trait DenseUnionArrayOwnedExt {
    /// Deconstruct this array into its type IDs, offsets, schema, and compact children.
    fn into_data_parts(self) -> DenseUnionDataParts;
}

impl DenseUnionArrayOwnedExt for Array<DenseUnion> {
    fn into_data_parts(self) -> DenseUnionDataParts {
        let variants = self.variants().clone();
        let type_ids = self.type_ids().clone();
        let offsets = self.offsets().clone();
        let children = self.iter_children().cloned().collect();
        DenseUnionDataParts {
            variants,
            type_ids,
            offsets,
            children,
        }
    }
}

fn validate_components(
    type_ids: &ArrayRef,
    offsets: &ArrayRef,
    children: &[&ArrayRef],
    dtype: &DType,
    len: usize,
) -> VortexResult<()> {
    let DType::Union(variants, nullability) = dtype else {
        return Err(vortex_err!(
            "DenseUnion requires a union dtype, got {dtype}"
        ));
    };
    vortex_ensure_eq!(
        children.len(),
        variants.len(),
        "DenseUnion has {} compact children but expected {}",
        children.len(),
        variants.len()
    );
    vortex_ensure_eq!(
        type_ids.dtype(),
        &DType::Primitive(PType::U8, *nullability),
        "DenseUnion type_ids have incompatible dtype"
    );
    vortex_ensure_eq!(
        type_ids.len(),
        len,
        "DenseUnion type_ids length does not match its logical length"
    );
    vortex_ensure_eq!(
        offsets.dtype(),
        &OFFSETS_DTYPE,
        "DenseUnion offsets must be non-nullable i32"
    );
    vortex_ensure_eq!(
        offsets.len(),
        len,
        "DenseUnion offsets length does not match its logical length"
    );

    for (index, (variant_dtype, child)) in variants.variants().zip(children).enumerate() {
        vortex_ensure_eq!(
            child.dtype(),
            &variant_dtype,
            "DenseUnion child {index} has incompatible dtype"
        );
    }

    Ok(())
}

impl VTable for DenseUnion {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.dense_union");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Union(variants, _) = dtype else {
            return Err(vortex_err!(
                "DenseUnion requires a union dtype, got {dtype}"
            ));
        };
        vortex_ensure_eq!(
            slots.len(),
            DenseUnionSlots::CHILDREN_OFFSET + variants.len(),
            "DenseUnion has an unexpected number of slots"
        );
        let type_ids = slots[DenseUnionSlots::TYPE_IDS]
            .as_ref()
            .ok_or_else(|| vortex_err!("DenseUnion is missing its type_ids slot"))?;
        let offsets = slots[DenseUnionSlots::OFFSETS]
            .as_ref()
            .ok_or_else(|| vortex_err!("DenseUnion is missing its offsets slot"))?;
        let children = slots[DenseUnionSlots::CHILDREN_OFFSET..]
            .iter()
            .enumerate()
            .map(|(index, child)| {
                child
                    .as_ref()
                    .ok_or_else(|| vortex_err!("DenseUnion is missing compact child {index}"))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        validate_components(type_ids, offsets, &children, dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("DenseUnion buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("DenseUnion buffer_name index {idx} out of bounds")
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
        let child_lengths = array
            .iter_children()
            .map(|child| {
                u64::try_from(child.len())
                    .map_err(|_| vortex_err!("DenseUnion child length does not fit in u64"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Some(DenseUnionMetadata { child_lengths }.encode_to_vec()))
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
        vortex_ensure!(buffers.is_empty(), "DenseUnion expects no buffers");
        let DType::Union(variants, nullability) = dtype else {
            return Err(vortex_err!(
                "DenseUnion requires a union dtype, got {dtype}"
            ));
        };
        let metadata = DenseUnionMetadata::decode(metadata)?;
        vortex_ensure_eq!(
            metadata.child_lengths.len(),
            variants.len(),
            "DenseUnion metadata has an unexpected number of child lengths"
        );
        vortex_ensure_eq!(
            children.len(),
            DenseUnionSlots::CHILDREN_OFFSET + variants.len(),
            "DenseUnion has an unexpected number of serialized children"
        );

        let type_ids = children.get(
            DenseUnionSlots::TYPE_IDS,
            &DType::Primitive(PType::U8, *nullability),
            len,
        )?;
        let offsets = children.get(DenseUnionSlots::OFFSETS, &OFFSETS_DTYPE, len)?;
        let compact_children = variants
            .variants()
            .zip(metadata.child_lengths)
            .enumerate()
            .map(|(index, (variant_dtype, child_len))| {
                let child_len = usize::try_from(child_len)
                    .map_err(|_| vortex_err!("DenseUnion child length does not fit in usize"))?;
                children.get(
                    DenseUnionSlots::CHILDREN_OFFSET + index,
                    &variant_dtype,
                    child_len,
                )
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(make_parts(
            type_ids,
            offsets,
            variants.clone(),
            compact_children,
        ))
    }

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        match idx {
            DenseUnionSlots::TYPE_IDS => "type_ids".to_string(),
            DenseUnionSlots::OFFSETS => "offsets".to_string(),
            _ => array.variants().names()[idx - DenseUnionSlots::CHILDREN_OFFSET].to_string(),
        }
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = require_child!(
            array,
            array.type_ids(),
            DenseUnionSlots::TYPE_IDS => Primitive
        );
        let array = require_child!(
            array,
            array.offsets(),
            DenseUnionSlots::OFFSETS => Primitive
        );
        canonicalize(array, ctx).map(ExecutionResult::done)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<DenseUnion> for DenseUnion {
    fn scalar_at(
        array: ArrayView<'_, DenseUnion>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let type_id_scalar = array.type_ids().execute_scalar(index, ctx)?;
        let Some(type_id) = type_id_scalar.as_primitive().typed_value::<u8>() else {
            return Ok(Scalar::null(array.dtype().clone()));
        };
        let child_index = array
            .variants()
            .tag_to_child_index(type_id)
            .ok_or_else(|| vortex_err!("DenseUnion contains unknown type ID {type_id}"))?;
        let offset = array
            .offsets()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .typed_value::<i32>()
            .ok_or_else(|| vortex_err!("DenseUnion contains a null offset at row {index}"))?;
        let offset = usize::try_from(offset).map_err(|_| {
            vortex_err!("DenseUnion contains negative offset {offset} at row {index}")
        })?;
        let child = array
            .child(child_index)
            .ok_or_else(|| vortex_err!("DenseUnion is missing compact child {child_index}"))?;
        vortex_ensure!(
            offset < child.len(),
            "DenseUnion offset {offset} is out of bounds for child {child_index} of length {}",
            child.len()
        );

        Scalar::union(
            array.variants().clone(),
            type_id,
            child.execute_scalar(offset, ctx)?,
            array.dtype().nullability(),
        )
    }
}

impl ValidityVTable<DenseUnion> for DenseUnion {
    fn validity(array: ArrayView<'_, DenseUnion>) -> VortexResult<Validity> {
        array.type_ids().validity()
    }
}
