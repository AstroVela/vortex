// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::patches::PatchSlotIndices;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
use vortex_array::validity::Validity;
use vortex_array::vtable::child_to_validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::FL_CHUNK_SIZE;

pub mod compress;
pub mod decompress;

/// Number of bytes a single FastLanes chunk occupies per bit of width.
///
/// A chunk holds [`FL_CHUNK_SIZE`] values, so packing it at `w` bits per value takes
/// `1024 * w / 8 == 128 * w` bytes. Because every chunk is a whole multiple of 128 bytes, chunk
/// boundaries are always aligned for any packed primitive type, which is what lets chunks of
/// differing widths be concatenated into a single buffer.
pub const BYTES_PER_CHUNK_BIT: usize = FL_CHUNK_SIZE / 8;

#[array_slots(crate::BitPackedV2)]
pub struct BitPackedV2Slots {
    /// The indices of exception values that don't fit in their chunk's bit width.
    #[slot(0)]
    pub patch_indices: Option<ArrayRef>,
    /// The exception values that don't fit in their chunk's bit width.
    #[slot(1)]
    pub patch_values: Option<ArrayRef>,
    /// Chunk offsets for the patch indices/values.
    #[slot(2)]
    pub patch_chunk_offsets: Option<ArrayRef>,
    /// The validity bitmap indicating which elements are non-null.
    #[slot(3)]
    pub validity_child: Option<ArrayRef>,
}

pub(crate) const PATCH_SLOTS: PatchSlotIndices = PatchSlotIndices {
    indices: BitPackedV2Slots::PATCH_INDICES,
    values: BitPackedV2Slots::PATCH_VALUES,
    chunk_offsets: BitPackedV2Slots::PATCH_CHUNK_OFFSETS,
};

/// The decomposed parts of a [`BitPackedV2Array`](crate::BitPackedV2Array).
pub struct BitPackedV2DataParts {
    pub offset: u16,
    pub len: usize,
    pub packed: BufferHandle,
    pub bit_widths: BufferHandle,
    pub patches: Option<Patches>,
    pub validity: Validity,
}

#[derive(Clone, Debug)]
pub struct BitPackedV2Data {
    /// The offset within the first chunk (created with a slice). `0 <= offset < 1024`.
    pub(super) offset: u16,
    /// The packed values, one variable-width FastLanes block per chunk.
    pub(super) packed: BufferHandle,
    /// One bit width per FastLanes chunk.
    pub(super) bit_widths: BufferHandle,
    /// Byte offset of each chunk within `packed`, with a trailing total. Derived from
    /// `bit_widths` at construction so that chunk lookup is O(1).
    pub(super) chunk_byte_offsets: Buffer<u64>,
    /// Patch metadata for reconstructing Patches from slots.
    pub(super) patches_data: Option<PatchesData>,
}

impl Display for BitPackedV2Data {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let widths = self.bit_widths();
        write!(
            f,
            "chunks: {}, bit_width: {}..={}, offset: {}",
            widths.len(),
            widths.iter().min().copied().unwrap_or(0),
            widths.iter().max().copied().unwrap_or(0),
            self.offset
        )
    }
}

/// Byte offsets of each chunk within the packed buffer, with a trailing total length.
fn chunk_byte_offsets(bit_widths: &[u8]) -> Buffer<u64> {
    let mut offsets = BufferMut::<u64>::with_capacity(bit_widths.len() + 1);
    let mut offset = 0u64;
    offsets.push(offset);
    for &width in bit_widths {
        offset += (BYTES_PER_CHUNK_BIT * width as usize) as u64;
        offsets.push(offset);
    }
    offsets.freeze()
}

impl BitPackedV2Data {
    /// Create a bit-packed array whose FastLanes chunks each carry their own bit width.
    ///
    /// `packed` is the concatenation of one FastLanes block per chunk, where chunk `c` occupies
    /// `128 * bit_widths[c]` bytes. `bit_widths` holds one width per chunk of the array,
    /// including the chunk the leading `offset` values were sliced away from.
    ///
    /// # Safety
    ///
    /// For signed arrays, it is the caller's responsibility to ensure that no packed value is
    /// interpreted as negative once unpacked to the provided PType. This invariant is upheld by
    /// [`bitpack_v2_encode`](compress::bitpack_v2_encode).
    pub fn try_new(
        packed: BufferHandle,
        bit_widths: BufferHandle,
        patches: Option<Patches>,
        offset: u16,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            (offset as usize) < FL_CHUNK_SIZE,
            "Offset must be less than the full chunk i.e., {FL_CHUNK_SIZE}, got {offset}"
        );
        let widths = bit_widths.as_host();
        vortex_ensure!(
            widths.iter().all(|&w| w <= 64),
            "Unsupported bit width, all widths must be <= 64"
        );

        Ok(Self {
            offset,
            chunk_byte_offsets: chunk_byte_offsets(widths.as_slice()),
            packed,
            bit_widths,
            patches_data: patches.as_ref().map(PatchesData::from_patches),
        })
    }

    pub(crate) fn validate(
        &self,
        ptype: PType,
        validity: &Validity,
        patches: Option<&Patches>,
        length: usize,
    ) -> VortexResult<()> {
        vortex_ensure!(ptype.is_int(), MismatchedTypes: "integer", ptype);

        let bit_widths = self.bit_widths();
        vortex_ensure!(
            bit_widths.iter().all(|&w| w as usize <= ptype.bit_width()),
            "BitPackedV2 bit widths must not exceed the {} bits of {ptype}",
            ptype.bit_width(),
        );

        let expected_chunks = (length + self.offset as usize).div_ceil(FL_CHUNK_SIZE);
        vortex_ensure!(
            bit_widths.len() == expected_chunks,
            "BitPackedV2 expected {expected_chunks} bit widths, got {}",
            bit_widths.len(),
        );

        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == length,
                "BitPackedV2Array validity length {validity_len} != array length {length}",
            );
        }

        if let Some(patches) = patches {
            vortex_ensure!(
                patches.dtype().eq_ignore_nullability(ptype.into()),
                "Patches DType {} does not match BitPackedV2Array dtype {ptype}",
                patches.dtype().as_nonnullable(),
            );
            vortex_ensure!(
                patches.array_len() == length,
                "BitPackedV2Array patches length {} != expected {length}",
                patches.array_len(),
            );
        }

        let expected_packed_len = self.packed_len();
        vortex_ensure!(
            self.packed.len() == expected_packed_len,
            "Expected {expected_packed_len} packed bytes, got {}",
            self.packed.len()
        );

        Ok(())
    }

    pub fn ptype(&self, dtype: &DType) -> PType {
        dtype.as_ptype()
    }

    /// The packed values of every chunk, concatenated.
    #[inline]
    pub fn packed(&self) -> &BufferHandle {
        &self.packed
    }

    /// The bit width of every chunk, as a buffer.
    #[inline]
    pub fn bit_widths_buffer(&self) -> &BufferHandle {
        &self.bit_widths
    }

    /// The bit width of every chunk.
    #[inline]
    pub fn bit_widths(&self) -> &[u8] {
        // SAFETY: reconstructed from raw parts only to reinterpret the lifetime as `self`, which
        // outlives the borrow of the buffer.
        let bytes = self.bit_widths.as_host();
        unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) }
    }

    /// The total number of packed bytes across all chunks.
    #[inline]
    pub fn packed_len(&self) -> usize {
        self.chunk_byte_offsets.last().copied().unwrap_or_default() as usize
    }

    /// Access the packed block of `chunk_idx` as a slice of `T`.
    #[inline]
    pub fn packed_chunk<T>(&self, chunk_idx: usize) -> &[T] {
        let start = self.chunk_byte_offsets[chunk_idx] as usize;
        let end = self.chunk_byte_offsets[chunk_idx + 1] as usize;
        let packed_bytes = self.packed.as_host();

        // SAFETY: chunk boundaries are multiples of 128 bytes, so both the pointer offset and the
        // length are whole multiples of `size_of::<T>()` and the pointer stays aligned to `T`. The
        // slice is reconstructed from raw parts only to reinterpret the lifetime as `self`.
        unsafe {
            std::slice::from_raw_parts(
                packed_bytes.as_ptr().add(start).cast::<T>(),
                (end - start) / size_of::<T>(),
            )
        }
    }

    /// Byte offset of every chunk within the packed buffer, with a trailing total.
    #[inline]
    pub fn chunk_byte_offsets(&self) -> &[u64] {
        &self.chunk_byte_offsets
    }

    /// Offset of the first element within the first chunk.
    #[inline]
    pub fn offset(&self) -> u16 {
        self.offset
    }
}

pub trait BitPackedV2ArrayExt: BitPackedV2ArraySlotsExt {
    #[inline]
    fn packed(&self) -> &BufferHandle {
        BitPackedV2Data::packed(self)
    }

    #[inline]
    fn bit_widths(&self) -> &[u8] {
        BitPackedV2Data::bit_widths(self)
    }

    #[inline]
    fn bit_widths_buffer(&self) -> &BufferHandle {
        BitPackedV2Data::bit_widths_buffer(self)
    }

    #[inline]
    fn offset(&self) -> u16 {
        BitPackedV2Data::offset(self)
    }

    #[inline]
    fn chunk_byte_offsets(&self) -> &[u64] {
        BitPackedV2Data::chunk_byte_offsets(self)
    }

    #[inline]
    fn patches(&self) -> Option<Patches> {
        PatchesData::patches_from_slots(
            self.patches_data.as_ref(),
            self.as_ref().len(),
            self.as_ref().slots(),
            PATCH_SLOTS,
        )
    }

    #[inline]
    fn validity(&self) -> Validity {
        child_to_validity(self.validity_child(), self.as_ref().dtype().nullability())
    }
}

impl<T: TypedArrayRef<crate::BitPackedV2>> BitPackedV2ArrayExt for T {}
