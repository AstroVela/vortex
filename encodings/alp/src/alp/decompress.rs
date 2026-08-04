// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::chunk_range;
use vortex_array::arrays::primitive::patch_chunk;
use vortex_array::dtype::DType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::Patches;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ALPArray;
use crate::ALPArrayOwnedExt;
use crate::ALPFloat;
use crate::Exponents;
use crate::match_each_alp_float_ptype;

/// Decompresses an ALP-encoded array using `to_primitive` (legacy path).
///
/// # Returns
///
/// A `PrimitiveArray` containing the decompressed floating-point values with all patches applied.
pub fn decompress_into_array(
    array: ALPArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let dtype = array.dtype().clone();
    let (encoded, exponents, patches) = ALPArrayOwnedExt::into_parts(array);
    if let Some(p) = &patches
        && let Some(chunk_offsets) = p.chunk_offsets()
    {
        let prim_encoded = encoded.execute::<PrimitiveArray>(ctx)?;
        let patches_chunk_offsets = chunk_offsets.clone().execute::<PrimitiveArray>(ctx)?;
        let patches_indices = p.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let patches_values = p.values().clone().execute::<PrimitiveArray>(ctx)?;
        Ok(decompress_chunked_core(
            prim_encoded,
            exponents,
            &patches_indices,
            &patches_values,
            &patches_chunk_offsets,
            p,
            dtype,
        ))
    } else {
        let encoded_prim = encoded.execute::<PrimitiveArray>(ctx)?;
        decompress_unchunked_core(encoded_prim, exponents, patches, dtype, ctx)
    }
}

/// Decompresses an ALP-encoded array using `execute` (execution path).
///
/// This version uses `execute` on child arrays instead of `to_primitive`,
/// ensuring proper recursive execution through the execution context.
///
/// # Returns
///
/// A `PrimitiveArray` containing the decompressed floating-point values with all patches applied.
pub fn execute_decompress(array: ALPArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let dtype = array.dtype().clone();
    let (encoded, exponents, patches) = ALPArrayOwnedExt::into_parts(array);
    if let Some(p) = &patches
        && let Some(chunk_offsets) = p.chunk_offsets()
    {
        // TODO(joe): have into parts.
        let encoded = encoded.execute::<PrimitiveArray>(ctx)?;
        let patches_chunk_offsets = chunk_offsets.clone().execute::<PrimitiveArray>(ctx)?;
        let patches_indices = p.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let patches_values = p.values().clone().execute::<PrimitiveArray>(ctx)?;
        Ok(decompress_chunked_core(
            encoded,
            exponents,
            &patches_indices,
            &patches_values,
            &patches_chunk_offsets,
            p,
            dtype,
        ))
    } else {
        let encoded = encoded.execute::<PrimitiveArray>(ctx)?;
        decompress_unchunked_core(encoded, exponents, patches, dtype, ctx)
    }
}

/// Reinterprets a run of ALP integers that [`ALPFloat::decode_slice_inplace`] has already decoded
/// as the floats it now holds.
///
/// `transmute` cannot do this job safely: between two slice references it only checks the size of
/// the fat pointer, never the element, so it would happily carry an element count from a narrow
/// integer over to a wider float and hand back a slice spanning twice the memory it owns. Building
/// the slice from a pointer behind a compile-time width check makes the count provably right, and
/// turns any future `ALPFloat` impl whose integer does not match its float into a build failure
/// rather than silent out-of-bounds access.
///
/// # Safety
///
/// Every element of `decoded` must already hold the bit pattern of a `T`.
unsafe fn decoded_as_floats<T: ALPFloat>(decoded: &mut [T::ALPInt]) -> &mut [T] {
    const {
        assert!(size_of::<T>() == size_of::<T::ALPInt>());
        assert!(align_of::<T>() == align_of::<T::ALPInt>());
    }

    let len = decoded.len();
    // SAFETY: the assertions above pin `T` and `T::ALPInt` to one size and alignment, so `len`
    // values of `T` occupy exactly the bytes `decoded` already owns and are correctly aligned. The
    // caller guarantees each of those slots holds a valid `T`, and the returned slice inherits the
    // exclusive borrow it consumed.
    unsafe { std::slice::from_raw_parts_mut(decoded.as_mut_ptr().cast::<T>(), len) }
}

/// Core decompression logic for chunked ALP arrays.
///
/// Takes pre-resolved `PrimitiveArray` inputs to avoid duplication between
/// the `to_primitive` and `execute` paths.
#[expect(
    clippy::cognitive_complexity,
    reason = "complexity is from nested match_each_* macros"
)]
fn decompress_chunked_core(
    encoded: PrimitiveArray,
    exponents: Exponents,
    patches_indices: &PrimitiveArray,
    patches_values: &PrimitiveArray,
    patches_chunk_offsets: &PrimitiveArray,
    patches: &Patches,
    dtype: DType,
) -> PrimitiveArray {
    let validity = encoded
        .validity()
        .vortex_expect("ALP validity should be derivable");
    let ptype = dtype.as_ptype();
    let array_len = encoded.len();
    let offset_within_chunk = patches.offset_within_chunk().unwrap_or(0);

    match_each_alp_float_ptype!(ptype, |T| {
        let patches_values = patches_values.as_slice::<T>();
        let mut alp_buffer = encoded.into_buffer_mut::<<T as ALPFloat>::ALPInt>();
        match_each_unsigned_integer_ptype!(patches_chunk_offsets.ptype(), |C| {
            let patches_chunk_offsets = patches_chunk_offsets.as_slice::<C>();

            match_each_unsigned_integer_ptype!(patches_indices.ptype(), |I| {
                let patches_indices = patches_indices.as_slice::<I>();

                for chunk_idx in 0..patches_chunk_offsets.len() {
                    let chunk_range = chunk_range(chunk_idx, patches.offset(), array_len);
                    let chunk_slice = &mut alp_buffer.as_mut_slice()[chunk_range];

                    <T>::decode_slice_inplace(chunk_slice, exponents);

                    // SAFETY: `decode_slice_inplace` just overwrote every element of the chunk
                    // with the bit pattern of the `T` it decodes to.
                    let decoded_chunk: &mut [T] = unsafe { decoded_as_floats::<T>(chunk_slice) };
                    patch_chunk(
                        decoded_chunk,
                        patches_indices,
                        patches_values,
                        patches.offset(),
                        patches_chunk_offsets,
                        chunk_idx,
                        offset_within_chunk,
                    );
                }

                // SAFETY: every bit pattern of `T::ALPInt` is a valid `T` of the same width.
                // `BufferMut::transmute` asserts that size and alignment match and rebuilds the
                // buffer, rather than assuming two monomorphizations of a `repr(Rust)` generic
                // share a layout the way a bare `transmute` of the whole struct would.
                let decoded_buffer = unsafe { alp_buffer.transmute::<T>() };
                PrimitiveArray::new::<T>(decoded_buffer.freeze(), validity)
            })
        })
    })
}

/// Core decompression logic for unchunked ALP arrays.
///
/// Takes a pre-resolved `PrimitiveArray` to avoid duplication between
/// the `to_primitive` and `execute` paths.
fn decompress_unchunked_core(
    encoded: PrimitiveArray,
    exponents: Exponents,
    patches: Option<Patches>,
    dtype: DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let validity = encoded.validity()?;
    let ptype = dtype.as_ptype();

    let decoded = match_each_alp_float_ptype!(ptype, |T| {
        let mut alp_buffer = encoded.into_buffer_mut::<<T as ALPFloat>::ALPInt>();
        <T>::decode_slice_inplace(alp_buffer.as_mut_slice(), exponents);
        // SAFETY: as above — `decode_slice_inplace` left every element holding the bit pattern of
        // a `T`, and `BufferMut::transmute` checks the widths match.
        let decoded_buffer = unsafe { alp_buffer.transmute::<T>() };
        PrimitiveArray::new::<T>(decoded_buffer.freeze(), validity)
    });

    if let Some(patches) = patches {
        decoded.patch(&patches, ctx)
    } else {
        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use std::sync::LazyLock;

    use ::alp::ENCODE_CHUNK_SIZE;
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::alp::array::ALPArrayExt;
    use crate::alp_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    /// Drives `decompress_chunked_core`, which reinterprets each decoded chunk of ALP integers as
    /// the floats it now holds. Seeding a patch in every chunk keeps all of them on that path, and
    /// a trailing partial chunk covers the case where the last run is shorter than the rest.
    #[rstest]
    #[case::whole_chunks(3 * ENCODE_CHUNK_SIZE)]
    #[case::trailing_partial_chunk(2 * ENCODE_CHUNK_SIZE + 17)]
    fn chunked_decode_round_trips(#[case] len: usize) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<f64> = (0..len)
            .map(|i| if i % ENCODE_CHUNK_SIZE == 7 { PI } else { 1.0 })
            .collect();
        let original = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);

        let encoded = alp_encode(original.as_view(), None, &mut ctx)?;
        let patches = encoded
            .patches()
            .vortex_expect("PI does not round-trip through ALP, so it must be patched");
        assert!(
            patches.chunk_offsets().is_some(),
            "test must reach the chunked decode path"
        );

        let decoded = decompress_into_array(encoded, &mut ctx)?;
        assert_arrays_eq!(decoded, original, &mut ctx);

        // `assert_arrays_eq!` compares logically; check the bit patterns the reinterpretation
        // produced land exactly on the originals too.
        for (decoded, original) in decoded
            .as_slice::<f64>()
            .iter()
            .zip(original.as_slice::<f64>())
        {
            assert!(NativePType::is_eq(*decoded, *original));
        }
        Ok(())
    }
}
