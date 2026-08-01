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
use crate::geometry::TileBounds;
use crate::geometry::TileBoundsIter;
use crate::geometry::geometry_usizes;
use crate::transpose::decode_visible_elements;
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
    /// The logical row offset within the first retained physical tile.
    #[prost(uint32, tag = "3")]
    pub row_offset: u32,
    /// The number of rows represented by the retained physical child.
    #[prost(uint64, tag = "4")]
    pub backing_rows: u64,
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
    row_offset: usize,
    backing_rows: usize,
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
            "tile_rows: {}, tile_dimensions: {}, row_offset: {}, backing_rows: {}",
            self.geometry.rows(),
            self.geometry.dimensions(),
            self.row_offset,
            self.backing_rows
        )
    }
}

impl ArrayHash for TiledFixedSizeListData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.geometry.hash(state);
        self.row_offset.hash(state);
        self.backing_rows.hash(state);
    }
}

impl ArrayEq for TiledFixedSizeListData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.geometry == other.geometry
            && self.row_offset == other.row_offset
            && self.backing_rows == other.backing_rows
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
        Self::try_new_view(elements, list_size, validity, len, geometry, 0, len)
    }

    pub(crate) fn try_new_view(
        elements: ArrayRef,
        list_size: u32,
        validity: Validity,
        len: usize,
        geometry: TileGeometry,
        row_offset: usize,
        backing_rows: usize,
    ) -> VortexResult<TiledFixedSizeListArray> {
        let dtype = DType::FixedSizeList(
            Arc::new(elements.dtype().clone()),
            list_size,
            validity.nullability(),
        );
        let data = TiledFixedSizeListData {
            geometry,
            row_offset,
            backing_rows,
        };
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
        let geometry = Self::new(rows, dimensions);
        geometry_usizes(geometry)?;
        Ok(geometry)
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
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::FixedSizeList(element_dtype, list_size, nullability) = dtype else {
            vortex_bail!(InvalidArgument: "tiled fixed-size-list dtype must be FixedSizeList, got {dtype}");
        };
        let (tile_rows, tile_dimensions) = geometry_usizes(data.geometry)?;
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

        vortex_ensure!(
            data.row_offset < tile_rows,
            InvalidArgument: "tiled fixed-size-list row offset {} must be less than tile rows {tile_rows}",
            data.row_offset
        );
        let logical_end = data.row_offset.checked_add(len).ok_or_else(|| {
            vortex_err!(InvalidArgument: "tiled fixed-size-list row offset {} plus length {len} overflows usize", data.row_offset)
        })?;
        vortex_ensure!(
            logical_end <= data.backing_rows,
            InvalidArgument: "tiled fixed-size-list row window {}..{logical_end} exceeds {} backing rows",
            data.row_offset,
            data.backing_rows
        );
        if len > 0 {
            let remainder = logical_end % tile_rows;
            let max_backing_rows = if remainder == 0 {
                logical_end
            } else {
                logical_end.saturating_add(tile_rows - remainder)
            };
            vortex_ensure!(
                data.backing_rows <= max_backing_rows,
                InvalidArgument: "tiled fixed-size-list backing rows {} exceeds the retained tile extent {max_backing_rows}",
                data.backing_rows
            );
        }
        if *list_size == 0 || (*list_size as usize) > tile_dimensions {
            vortex_ensure!(
                data.row_offset == 0 && data.backing_rows == len,
                InvalidArgument: "tiled fixed-size-list zero-width and multi-slab arrays require row offset 0 and backing rows equal to length {len}"
            );
        }

        let expected_len = data
            .backing_rows
            .checked_mul(*list_size as usize)
            .ok_or_else(|| {
                vortex_err!(InvalidArgument: "tiled fixed-size-list backing rows {} times list size {list_size} overflows usize", data.backing_rows)
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
        let row_offset = u32::try_from(array.row_offset()).map_err(|error| {
            vortex_err!(InvalidArgument: "tiled fixed-size-list row offset cannot be serialized as u32: {error}")
        })?;
        let backing_rows = u64::try_from(array.backing_rows()).map_err(|error| {
            vortex_err!(InvalidArgument: "tiled fixed-size-list backing row count cannot be serialized as u64: {error}")
        })?;
        Ok(Some(
            TiledFixedSizeListMetadata {
                tile_rows: array.geometry.rows().get(),
                tile_dimensions: array.geometry.dimensions().get(),
                row_offset,
                backing_rows,
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
        let row_offset = usize::try_from(metadata.row_offset).map_err(|error| {
            vortex_err!(InvalidArgument: "tiled fixed-size-list row offset cannot be represented as usize: {error}")
        })?;
        let backing_rows = usize::try_from(metadata.backing_rows).map_err(|error| {
            vortex_err!(InvalidArgument: "tiled fixed-size-list backing row count cannot be represented as usize: {error}")
        })?;
        let DType::FixedSizeList(element_dtype, list_size, nullability) = dtype else {
            vortex_bail!(InvalidArgument: "tiled fixed-size-list dtype must be FixedSizeList, got {dtype}");
        };
        vortex_ensure!(
            matches!(element_dtype.as_ref(), DType::Primitive(..)),
            InvalidArgument: "tiled fixed-size-list elements must have a primitive dtype, got {element_dtype}"
        );
        vortex_ensure!(
            nullability.is_nullable() || children.len() == 1,
            InvalidArgument: "non-nullable tiled fixed-size-list dtype cannot have an outer validity child"
        );
        let physical_len = backing_rows
            .checked_mul(*list_size as usize)
            .ok_or_else(|| {
                vortex_err!(InvalidArgument: "tiled fixed-size-list backing rows {backing_rows} times list size {list_size} overflows usize")
        })?;
        let elements = children.get(0, element_dtype.as_ref(), physical_len)?;
        let validity = match children.len() {
            1 => Validity::from(*nullability),
            2 => Validity::Array(children.get(1, &Validity::DTYPE, len)?),
            _ => unreachable!("validated tiled fixed-size-list child count"),
        };
        let array = Self::try_new_view(
            elements,
            *list_size,
            validity,
            len,
            geometry,
            row_offset,
            backing_rows,
        )?;
        vortex_ensure!(
            array.dtype() == dtype,
            InvalidArgument: "deserialized tiled fixed-size-list dtype {} does not match supplied dtype {dtype}",
            array.dtype()
        );
        array.try_into_parts().map_err(|_| {
            vortex_err!(InvalidArgument: "deserialized tiled fixed-size-list array is unexpectedly shared")
        })
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
        let decoded_elements = decode_visible_elements(
            elements.as_view(),
            array.len(),
            array.list_size() as usize,
            array.geometry(),
            array.row_offset(),
            array.backing_rows(),
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

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::rules::RULES.evaluate(array, parent, child_idx)
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

    /// Returns the logical row offset within the first retained physical tile.
    fn row_offset(&self) -> usize {
        self.row_offset
    }

    /// Returns the number of rows represented by the retained physical child.
    fn backing_rows(&self) -> usize {
        self.backing_rows
    }

    /// Returns whether every logical row is stored in a single dimension slab.
    fn is_full_width(&self) -> bool {
        let (_, dimensions) = validated_geometry_usizes(self.geometry());
        self.list_size() as usize <= dimensions
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
        let (rows, _) = validated_geometry_usizes(self.geometry());
        if self.as_ref().is_empty() {
            0
        } else {
            (self.row_offset() + self.as_ref().len()).div_ceil(rows)
        }
    }

    /// Returns the number of dimension tiles needed for each logical list.
    fn dimension_tile_count(&self) -> usize {
        let (_, dimensions) = validated_geometry_usizes(self.geometry());
        (self.list_size() as usize).div_ceil(dimensions)
    }

    /// Returns the bounds for one checked logical tile.
    fn tile(&self, row_tile: usize, dimension_tile: usize) -> VortexResult<TileBounds> {
        crate::geometry::tile_bounds_view(
            self.as_ref().len(),
            self.list_size() as usize,
            self.geometry(),
            self.row_offset(),
            self.backing_rows(),
            row_tile,
            dimension_tile,
        )
    }

    /// Returns the logical and physical bounds of every tile in physical storage order.
    fn tiles(&self) -> TileBoundsIter {
        TileBoundsIter::new_view(
            self.as_ref().len(),
            self.list_size() as usize,
            self.geometry(),
            self.row_offset(),
            self.backing_rows(),
            self.row_tile_count(),
            self.dimension_tile_count(),
        )
    }

    /// Slices the physical child for inspection of one tile's elements.
    ///
    /// This is a cold-path convenience; scoring kernels should retain the child once and index it
    /// with [`TileBounds::physical_range`].
    #[cold]
    fn tile_elements(&self, bounds: &TileBounds) -> VortexResult<ArrayRef> {
        self.elements().slice(bounds.physical_range.clone())
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

fn validated_geometry_usizes(geometry: TileGeometry) -> (usize, usize) {
    match geometry_usizes(geometry) {
        Ok(geometry) => geometry,
        Err(_) => unreachable!("validated tiled fixed-size-list geometry must fit usize"),
    }
}
