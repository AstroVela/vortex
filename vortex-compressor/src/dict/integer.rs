// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted
//! for external compatibility.

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::estimate::CompressionEstimate;
use crate::estimate::EstimateVerdict;
use crate::stats::ArrayAndStats;
use crate::stats::IntegerErasedStats;
use crate::stats::IntegerStats;

/// Estimates dictionary compression for low-cardinality integer values.
pub(super) fn estimate(data: &ArrayAndStats, exec_ctx: &mut ExecutionCtx) -> CompressionEstimate {
    let bit_width = data.array_as_primitive().ptype().bit_width();
    let stats = data.integer_stats(exec_ctx);

    if stats.value_count() == 0 {
        return CompressionEstimate::Verdict(EstimateVerdict::Skip);
    }

    let distinct_values_count = stats.distinct_count().vortex_expect(
        "this must be present since the built-in dict compression declared that we need distinct values",
    );

    // If > 50% of the values are distinct, skip dictionary compression.
    if distinct_values_count > stats.value_count() / 2 {
        return CompressionEstimate::Verdict(EstimateVerdict::Skip);
    }

    // Ignore nulls encoding for the estimate. We only focus on values.

    let values_size = bit_width * distinct_values_count as usize;

    // TODO(connor): Should we just hardcode this instead of let the compressor choose?
    // Assume codes are compressed RLE + BitPacking.
    let codes_bw = u32::BITS - distinct_values_count.leading_zeros();

    let n_runs = (stats.value_count() / stats.average_run_length()) as usize;

    // Assume that codes will either be BitPack or RLE-BitPack.
    let codes_size_bp = codes_bw as usize * stats.value_count() as usize;
    let codes_size_rle_bp = usize::checked_mul(codes_bw as usize + 32, n_runs);

    let codes_size = usize::min(codes_size_bp, codes_size_rle_bp.unwrap_or(usize::MAX));

    let before = stats.value_count() as usize * bit_width;

    CompressionEstimate::Verdict(EstimateVerdict::Ratio(
        before as f64 / (values_size + codes_size) as f64,
    ))
}

/// Encodes a typed integer array into a [`DictArray`] using the pre-computed distinct values.
macro_rules! typed_encode {
    ($source_array:ident, $stats:ident, $typed:ident, $typ:ty) => {{
        let distinct = $typed.distinct().vortex_expect(
            "this must be present since the built-in dict compression declared that we need \
             distinct values",
        );

        let values_validity = match $source_array.validity()? {
            Validity::NonNullable => Validity::NonNullable,
            _ => Validity::AllValid,
        };
        let codes_validity = $source_array.validity()?;

        let values: Buffer<$typ> = distinct.distinct_values().keys().map(|x| x.0).collect();

        let max_code = values.len();
        let codes = if max_code <= u8::MAX as usize {
            let buf = <DictEncoder as Encode<$typ, u8>>::encode(
                &values,
                $source_array.as_slice::<$typ>(),
            );
            PrimitiveArray::new(buf, codes_validity).into_array()
        } else if max_code <= u16::MAX as usize {
            let buf = <DictEncoder as Encode<$typ, u16>>::encode(
                &values,
                $source_array.as_slice::<$typ>(),
            );
            PrimitiveArray::new(buf, codes_validity).into_array()
        } else {
            let buf = <DictEncoder as Encode<$typ, u32>>::encode(
                &values,
                $source_array.as_slice::<$typ>(),
            );
            PrimitiveArray::new(buf, codes_validity).into_array()
        };

        let values = PrimitiveArray::new(values, values_validity).into_array();
        // SAFETY: invariants enforced in DictEncoder.
        Ok(unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) })
    }};
}

/// Compresses an integer array into a dictionary array according to attached stats.
///
/// # Errors
///
/// Returns an error if unable to compute validity.
#[expect(
    clippy::cognitive_complexity,
    reason = "complexity from match on all integer types"
)]
pub(crate) fn dictionary_encode(
    array: ArrayView<'_, Primitive>,
    stats: &IntegerStats,
) -> VortexResult<DictArray> {
    match stats.erased() {
        IntegerErasedStats::U8(typed) => typed_encode!(array, stats, typed, u8),
        IntegerErasedStats::U16(typed) => typed_encode!(array, stats, typed, u16),
        IntegerErasedStats::U32(typed) => typed_encode!(array, stats, typed, u32),
        IntegerErasedStats::U64(typed) => typed_encode!(array, stats, typed, u64),
        IntegerErasedStats::I8(typed) => typed_encode!(array, stats, typed, i8),
        IntegerErasedStats::I16(typed) => typed_encode!(array, stats, typed, i16),
        IntegerErasedStats::I32(typed) => typed_encode!(array, stats, typed, i32),
        IntegerErasedStats::I64(typed) => typed_encode!(array, stats, typed, i64),
    }
}

/// Stateless encoder that maps values to dictionary codes via a `HashMap`.
struct DictEncoder;

/// Trait for encoding values of type `T` into codes of type `I`.
trait Encode<T, I> {
    /// Using the distinct value set, turn the values into a set of codes.
    fn encode(distinct: &[T], values: &[T]) -> Buffer<I>;
}

/// Implements [`Encode`] for an integer type with all code width variants (u8, u16, u32).
macro_rules! impl_encode {
    ($typ:ty) => { impl_encode!($typ, u8, u16, u32); };
    ($typ:ty, $($ityp:ty),+) => {
        $(
        impl Encode<$typ, $ityp> for DictEncoder {
            #[expect(clippy::cast_possible_truncation)]
            fn encode(distinct: &[$typ], values: &[$typ]) -> Buffer<$ityp> {
                let mut codes =
                    vortex_utils::aliases::hash_map::HashMap::<$typ, $ityp>::with_capacity(
                        distinct.len(),
                    );
                for (code, &value) in distinct.iter().enumerate() {
                    codes.insert(value, code as $ityp);
                }

                let mut output = vortex_buffer::BufferMut::with_capacity(values.len());
                for value in values {
                    // Any code lookups which fail are for nulls, so their value does not matter.
                    // SAFETY: we have exactly sized output to be as large as values.
                    unsafe { output.push_unchecked(codes.get(value).copied().unwrap_or_default()) };
                }

                output.freeze()
            }
        }
        )*
    };
}

impl_encode!(u8);
impl_encode!(u16);
impl_encode!(u32);
impl_encode!(u64);
impl_encode!(i8);
impl_encode!(i16);
impl_encode!(i32);
impl_encode!(i64);

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::dict::DictArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::dictionary_encode;
    use crate::stats::IntegerStats;

    #[test]
    fn test_dict_encode_integer_stats() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let data = buffer![100i32, 200, 100, 0, 100];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(data, validity);

        let stats = IntegerStats::generate_opts(
            &array,
            crate::stats::GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );
        let dict_array = dictionary_encode(array.as_view(), &stats)?;
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);

        let expected = PrimitiveArray::new(
            buffer![100i32, 200, 100, 100, 100],
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array()),
        )
        .into_array();
        let undict = dict_array
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();
        assert_arrays_eq!(undict, expected, &mut ctx);
        Ok(())
    }
}
