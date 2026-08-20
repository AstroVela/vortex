// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_array::expr::Expression;
use vortex_array::expr::arbitrary::filter_expr;
use vortex_array::expr::arbitrary::projection_expr;

use crate::FUZZ_FILE_ARRAY_MAX_LEN;
use crate::array::CompressorStrategy;

#[derive(Debug)]
pub struct FuzzFileAction {
    pub array: ArrayRef,
    pub projection_expr: Option<Expression>,
    pub filter_expr: Option<Expression>,
    pub compressor_strategy: CompressorStrategy,
}

impl<'a> Arbitrary<'a> for FuzzFileAction {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let array = ArbitraryArray::arbitrary_with_config(
            u,
            &ArbitraryArrayConfig {
                dtype: None,
                len: 0..=FUZZ_FILE_ARRAY_MAX_LEN,
            },
        )?
        .0;
        let dtype = array.dtype().clone();
        Ok(FuzzFileAction {
            array,
            projection_expr: projection_expr(u, &dtype)?,
            filter_expr: filter_expr(u, &dtype)?,
            compressor_strategy: CompressorStrategy::arbitrary(u)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::MapBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::MapDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_file::OpenOptionsSessionExt;
    use vortex_file::WriteOptionsSessionExt;

    use crate::RUNTIME;
    use crate::SESSION;

    /// The fuzz session must permit every component the arbitrary array generator produces.
    /// Map arrays are a draft member of the `core` edition family, so the default write policy
    /// rejects them and every generated map would fail the write instead of exercising file IO.
    #[test]
    fn roundtrips_map_arrays_through_a_file() -> VortexResult<()> {
        let map_dtype = MapDType::try_new(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Primitive(PType::I32, Nullability::NonNullable),
            false,
        )?;
        let dtype = DType::Map(map_dtype.clone(), Nullability::NonNullable);

        let mut builder = MapBuilder::<u64, u64>::new(map_dtype, Nullability::NonNullable);
        for key in 0i32..4 {
            builder.append_scalar(&Scalar::try_map(
                dtype.clone(),
                [(Scalar::from(key), Scalar::from(key * 2))],
            )?)?;
        }
        let array = builder.finish_into_map().into_array();

        let mut buffer = ByteBufferMut::empty();
        SESSION
            .write_options()
            .blocking(&*RUNTIME)
            .write(&mut buffer, array.to_array_iterator())?;

        let chunks = SESSION
            .open_options()
            .open_buffer(buffer)?
            .scan()?
            .into_array_iter(&*RUNTIME)?
            .try_collect::<_, Vec<_>, _>()?;
        let read = ChunkedArray::try_new(chunks, dtype)?.into_array();

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(array, read, &mut ctx);

        Ok(())
    }
}
