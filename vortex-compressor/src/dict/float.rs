// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for
//! external compatibility.

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::half::f16;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::estimate::CompressionEstimate;
use crate::estimate::DeferredEstimate;
use crate::estimate::EstimateVerdict;
use crate::stats::ArrayAndStats;
use crate::stats::FloatErasedStats;
use crate::stats::FloatStats;

/// Estimates dictionary compression for low-cardinality float values.
pub(super) fn estimate(data: &ArrayAndStats, exec_ctx: &mut ExecutionCtx) -> CompressionEstimate {
    let stats = data.float_stats(exec_ctx);

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

    // Let sampling determine the expected ratio.
    CompressionEstimate::Deferred(DeferredEstimate::Sample)
}

/// Encodes a typed float array into a [`DictArray`] using the pre-computed distinct values.
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

        let values: Buffer<$typ> = distinct.distinct_values().iter().map(|x| x.0).collect();

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
        // SAFETY: enforced by the DictEncoder.
        Ok(unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) })
    }};
}

/// Compresses a floating-point array into a dictionary array according to attached stats.
///
/// # Errors
///
/// Returns an error if unable to compute validity.
pub(crate) fn dictionary_encode(
    array: ArrayView<'_, Primitive>,
    stats: &FloatStats,
) -> VortexResult<DictArray> {
    match stats.erased() {
        FloatErasedStats::F16(typed) => typed_encode!(array, stats, typed, f16),
        FloatErasedStats::F32(typed) => typed_encode!(array, stats, typed, f32),
        FloatErasedStats::F64(typed) => typed_encode!(array, stats, typed, f64),
    }
}

/// Stateless encoder that maps values to dictionary codes via a `HashMap`.
struct DictEncoder;

/// Trait for encoding values of type `T` into codes of type `I`.
trait Encode<T, I> {
    /// Using the distinct value set, turn the values into a set of codes.
    fn encode(distinct: &[T], values: &[T]) -> Buffer<I>;
}

/// Implements [`Encode`] for a float type using its bit representation as the hash key.
macro_rules! impl_encode {
    ($typ:ty, $utyp:ty) => { impl_encode!($typ, $utyp, u8, u16, u32); };
    ($typ:ty, $utyp:ty, $($ityp:ty),+) => {
        $(
        impl Encode<$typ, $ityp> for DictEncoder {
            #[expect(clippy::cast_possible_truncation)]
            fn encode(distinct: &[$typ], values: &[$typ]) -> Buffer<$ityp> {
                let mut codes =
                    vortex_utils::aliases::hash_map::HashMap::<$utyp, $ityp>::with_capacity(
                        distinct.len(),
                    );
                for (code, &value) in distinct.iter().enumerate() {
                    codes.insert(value.to_bits(), code as $ityp);
                }

                let mut output = vortex_buffer::BufferMut::with_capacity(values.len());
                for value in values {
                    // Any code lookups which fail are for nulls, so their value does not matter.
                    output.push(codes.get(&value.to_bits()).copied().unwrap_or_default());
                }

                output.freeze()
            }
        }
        )*
    };
}

impl_encode!(f16, u16);
impl_encode!(f32, u32);
impl_encode!(f64, u64);

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
    use crate::stats::FloatStats;
    use crate::stats::GenerateStatsOptions;

    #[test]
    fn test_float_dict_encode() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let values = buffer![1f32, 2f32, 2f32, 0f32, 1f32];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(values, validity);

        let stats = FloatStats::generate_opts(
            &array,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );
        let dict_array = dictionary_encode(array.as_view(), &stats)?;
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);

        let expected = PrimitiveArray::new(
            buffer![1f32, 2f32, 2f32, 1f32, 1f32],
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
