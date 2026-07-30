// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::num::NonZeroU32;
use std::sync::Arc;

use prost::Message;
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
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::require_child;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::TileGeometry;
use crate::transpose::decode_elements;
use crate::transpose::encode_elements;

/// A tiled fixed-size-list Vortex array.
pub type TiledFixedSizeListArray = Array<TiledFixedSizeList>;

/// Wire-format metadata for [`TiledFixedSizeListArray`].
#[derive(Clone, prost::Message)]
pub struct TiledFixedSizeListMetadata {
    /// The nonzero row capacity of each physical tile.
    #[prost(uint32, tag = "1")]
    pub tile_rows: u32,
    /// The nonzero dimension capacity of each physical tile.
    #[prost(uint32, tag = "2")]
    pub tile_dimensions: u32,
}

#[array_slots(TiledFixedSizeList)]
/// Child slots owned by a tiled fixed-size-list array.
pub struct TiledFixedSizeListSlots {
    /// The primitive physical elements in tiled order.
    #[slot(0)]
    pub elements: ArrayRef,
    /// The optional outer-list validity bitmap.
    #[slot(1)]
    pub validity: Option<ArrayRef>,
}

/// Encoding-specific state for [`TiledFixedSizeListArray`].
#[derive(Clone, Debug)]
pub struct TiledFixedSizeListData {
    geometry: TileGeometry,
}

impl TiledFixedSizeListData {
    fn make_slots(
        elements: &ArrayRef,
        validity: &Validity,
        len: usize,
    ) -> vortex_array::ArraySlots {
        TiledFixedSizeListSlots {
            elements: elements.clone(),
            validity: validity_to_child(validity, len),
        }
        .into_slots()
    }
}

impl Display for TiledFixedSizeListData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tile_rows: {}, tile_dimensions: {}",
            self.geometry.rows(),
            self.geometry.dimensions()
        )
    }
}

impl ArrayHash for TiledFixedSizeListData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.geometry.hash(state);
    }
}

impl ArrayEq for TiledFixedSizeListData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.geometry == other.geometry
    }
}

/// A two-dimensional tiled physical layout for primitive fixed-size-list values.
#[derive(Clone, Debug)]
pub struct TiledFixedSizeList;

impl TiledFixedSizeList {
    /// Encodes a canonical primitive fixed-size-list array into tiled physical order.
    pub fn encode(
        array: ArrayView<'_, FixedSizeList>,
        geometry: TileGeometry,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<TiledFixedSizeListArray> {
        let elements = array.elements().clone().execute::<PrimitiveArray>(ctx)?;
        let tiled_elements = encode_elements(
            elements.as_view(),
            array.len(),
            array.list_size() as usize,
            geometry,
            ctx,
        )?;
        Self::try_new(
            tiled_elements.into_array(),
            array.list_size(),
            array.fixed_size_list_validity(),
            array.len(),
            geometry,
        )
    }

    /// Constructs a tiled fixed-size-list array from primitive physical elements.
    ///
    /// The physical child must contain exactly `len * list_size` elements in tiled order.
    pub fn try_new(
        elements: ArrayRef,
        list_size: u32,
        validity: Validity,
        len: usize,
        geometry: TileGeometry,
    ) -> VortexResult<TiledFixedSizeListArray> {
        let dtype = DType::FixedSizeList(
            Arc::new(elements.dtype().clone()),
            list_size,
            validity.nullability(),
        );
        let data = TiledFixedSizeListData { geometry };
        let slots = TiledFixedSizeListData::make_slots(&elements, &validity, len);
        Array::try_from_parts(
            ArrayParts::new(TiledFixedSizeList, dtype, len, data).with_slots(slots),
        )
    }
}

impl TryFrom<&TiledFixedSizeListMetadata> for TileGeometry {
    type Error = vortex_error::VortexError;

    fn try_from(metadata: &TiledFixedSizeListMetadata) -> VortexResult<Self> {
        let rows = NonZeroU32::new(metadata.tile_rows)
            .ok_or_else(|| vortex_err!(InvalidArgument: "tile_rows must be nonzero"))?;
        let dimensions = NonZeroU32::new(metadata.tile_dimensions)
            .ok_or_else(|| vortex_err!(InvalidArgument: "tile_dimensions must be nonzero"))?;
        Ok(Self::new(rows, dimensions))
    }
}

impl VTable for TiledFixedSizeList {
    type TypedArrayData = TiledFixedSizeListData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.tiled_fsl");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::FixedSizeList(element_dtype, list_size, nullability) = dtype else {
            vortex_bail!(InvalidArgument: "tiled fixed-size-list dtype must be FixedSizeList, got {dtype}");
        };
        vortex_ensure!(
            matches!(element_dtype.as_ref(), DType::Primitive(..)),
            InvalidArgument: "tiled fixed-size-list elements must have a primitive dtype, got {element_dtype}"
        );
        vortex_ensure!(
            slots.len() == TiledFixedSizeListSlots::COUNT,
            InvalidArgument: "tiled fixed-size-list expected {} slots, got {}",
            TiledFixedSizeListSlots::COUNT,
            slots.len()
        );

        let elements = slots.first().and_then(Option::as_ref).ok_or_else(
            || vortex_err!(InvalidArgument: "tiled fixed-size-list elements slot is missing"),
        )?;
        let validity_child = slots
            .get(TiledFixedSizeListSlots::VALIDITY)
            .and_then(Option::as_ref);

        vortex_ensure!(
            elements.dtype() == element_dtype.as_ref(),
            InvalidArgument: "tiled fixed-size-list physical child dtype {} does not match logical element dtype {}",
            elements.dtype(),
            element_dtype
        );

        let expected_len = len.checked_mul(*list_size as usize).ok_or_else(|| {
            vortex_err!(InvalidArgument: "tiled fixed-size-list length {len} times list size {list_size} overflows usize")
        })?;
        vortex_ensure!(
            elements.len() == expected_len,
            InvalidArgument: "tiled fixed-size-list physical child length {} does not match expected {expected_len}",
            elements.len()
        );

        if let Some(validity) = validity_child {
            vortex_ensure!(
                validity.dtype() == &Validity::DTYPE,
                InvalidArgument: "tiled fixed-size-list outer validity must have dtype {}",
                Validity::DTYPE
            );
            vortex_ensure!(
                validity.len() == len,
                InvalidArgument: "tiled fixed-size-list outer validity length {} does not match {len}",
                validity.len()
            );
        }

        let validity = child_to_validity(validity_child, *nullability);
        vortex_ensure!(
            validity.nullability() == *nullability,
            InvalidArgument: "tiled fixed-size-list outer validity does not match dtype nullability"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("TiledFixedSizeListArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("TiledFixedSizeListArray buffer_name index {idx} out of bounds")
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
        Ok(Some(
            TiledFixedSizeListMetadata {
                tile_rows: array.geometry.rows().get(),
                tile_dimensions: array.geometry.dimensions().get(),
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
        vortex_ensure!(
            buffers.is_empty(),
            InvalidArgument: "tiled fixed-size-list expects 0 buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            matches!(children.len(), 1 | 2),
            InvalidArgument: "tiled fixed-size-list expects one elements child and an optional validity child, got {} children",
            children.len()
        );

        let metadata = TiledFixedSizeListMetadata::decode(metadata)?;
        let geometry = TileGeometry::try_from(&metadata)?;
        let DType::FixedSizeList(element_dtype, list_size, nullability) = dtype else {
            vortex_bail!(InvalidArgument: "tiled fixed-size-list dtype must be FixedSizeList, got {dtype}");
        };
        vortex_ensure!(
            matches!(element_dtype.as_ref(), DType::Primitive(..)),
            InvalidArgument: "tiled fixed-size-list elements must have a primitive dtype, got {element_dtype}"
        );
        let physical_len = len.checked_mul(*list_size as usize).ok_or_else(|| {
            vortex_err!(InvalidArgument: "tiled fixed-size-list length {len} times list size {list_size} overflows usize")
        })?;
        let elements = children.get(0, element_dtype.as_ref(), physical_len)?;
        let validity = match children.len() {
            1 => Validity::from(*nullability),
            2 => Validity::Array(children.get(1, &Validity::DTYPE, len)?),
            _ => unreachable!("validated tiled fixed-size-list child count"),
        };
        let slots = TiledFixedSizeListData::make_slots(&elements, &validity, len);
        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            TiledFixedSizeListData { geometry },
        )
        .with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        TiledFixedSizeListSlots::NAMES
            .get(idx)
            .map(ToString::to_string)
            .unwrap_or_else(|| {
                vortex_panic!("TiledFixedSizeListArray slot index {idx} out of bounds")
            })
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = require_child!(
            array,
            array.elements(),
            TiledFixedSizeListSlots::ELEMENTS => Primitive
        );
        let elements = array
            .elements()
            .clone()
            .try_downcast::<Primitive>()
            .map_err(|elements| {
                vortex_err!(
                    "tiled fixed-size-list physical child must execute to primitive, got {}",
                    elements.encoding_id()
                )
            })?;
        let decoded_elements = decode_elements(
            elements.as_view(),
            array.len(),
            array.list_size() as usize,
            array.geometry(),
            ctx,
        )?;
        Ok(ExecutionResult::done(
            FixedSizeListArray::new(
                decoded_elements.into_array(),
                array.list_size(),
                array.array_validity(),
                array.len(),
            )
            .into_array(),
        ))
    }
}

impl ValidityVTable<TiledFixedSizeList> for TiledFixedSizeList {
    fn validity(array: ArrayView<'_, TiledFixedSizeList>) -> VortexResult<Validity> {
        let validity = array
            .slots()
            .get(TiledFixedSizeListSlots::VALIDITY)
            .and_then(Option::as_ref);
        Ok(child_to_validity(validity, array.dtype().nullability()))
    }
}

/// Typed accessors for [`TiledFixedSizeListArray`].
pub trait TiledFixedSizeListArrayExt:
    TypedArrayRef<TiledFixedSizeList> + TiledFixedSizeListArraySlotsExt
{
    /// Returns the tile geometry that defines the physical layout.
    fn geometry(&self) -> TileGeometry {
        self.geometry
    }

    /// Returns the number of elements in each logical fixed-size list.
    fn list_size(&self) -> u32 {
        match self.as_ref().dtype() {
            DType::FixedSizeList(_, list_size, _) => *list_size,
            _ => unreachable!("validated tiled fixed-size-list dtype"),
        }
    }

    /// Returns the number of row tiles needed for the logical array length.
    fn row_tile_count(&self) -> usize {
        self.as_ref()
            .len()
            .div_ceil(self.geometry().rows().get() as usize)
    }

    /// Returns the number of dimension tiles needed for each logical list.
    fn dimension_tile_count(&self) -> usize {
        (self.list_size() as usize).div_ceil(self.geometry().dimensions().get() as usize)
    }

    /// Returns the outer-list validity derived from the optional validity slot.
    fn array_validity(&self) -> Validity {
        let validity = self
            .as_ref()
            .slots()
            .get(TiledFixedSizeListSlots::VALIDITY)
            .and_then(Option::as_ref);
        child_to_validity(validity, self.as_ref().dtype().nullability())
    }
}

impl<T> TiledFixedSizeListArrayExt for T where
    T: TypedArrayRef<TiledFixedSizeList> + TiledFixedSizeListArraySlotsExt
{
}
