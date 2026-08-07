// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decode native geometry columns for row-oriented `geo` algorithms.
//!
//! A native `take` can represent repeated variable-sized geometries as shared dictionary values
//! or repeated list views. Preserve that sharing across the `geo_types` boundary by decoding one
//! representative of each physical value and retaining only a cheap logical-row mapping.

use std::hash::Hash;

use geo_types::Geometry;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ListView;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::extension::geometries;

/// A logically dense geometry column whose decoded values may retain native physical sharing.
pub(super) enum DecodedGeometries {
    Dense(Vec<Geometry<f64>>),
    Shared {
        values: Vec<Geometry<f64>>,
        row_to_value: Vec<usize>,
    },
}

impl DecodedGeometries {
    pub(super) fn decode(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let Some(extension) = array.as_opt::<Extension>() else {
            return Ok(Self::Dense(geometries(array, ctx)?));
        };
        let Some(shared) = shared_rows(extension.storage_array(), ctx)? else {
            return Ok(Self::Dense(geometries(array, ctx)?));
        };

        let representatives = PrimitiveArray::from_iter(shared.representatives).into_array();
        Ok(Self::Shared {
            values: geometries(&array.take(representatives)?, ctx)?,
            row_to_value: shared.row_to_value,
        })
    }

    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = &Geometry<f64>> {
        (0..self.len()).map(move |row| match self {
            Self::Dense(values) => &values[row],
            Self::Shared {
                values,
                row_to_value,
            } => &values[row_to_value[row]],
        })
    }

    fn len(&self) -> usize {
        match self {
            Self::Dense(values) => values.len(),
            Self::Shared { row_to_value, .. } => row_to_value.len(),
        }
    }
}

struct SharedRows {
    representatives: Vec<u64>,
    row_to_value: Vec<usize>,
}

/// Find repeated physical values without executing the reduced extension and losing its encoding.
fn shared_rows(storage: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<SharedRows>> {
    if let Some(dict) = storage.as_opt::<Dict>() {
        let codes = dict
            .codes()
            .clone()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<Buffer<u64>>(ctx)?;
        return deduplicate(codes.iter().copied());
    }

    if let Some(list) = storage.as_opt::<ListView>() {
        let offsets = list
            .offsets()
            .clone()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<Buffer<u64>>(ctx)?;
        let sizes = list
            .sizes()
            .clone()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<Buffer<u64>>(ctx)?;

        // Ordinary canonical list views are contiguous and therefore cannot repeat a non-empty
        // physical geometry. Keep their existing dense decode path without hashing every row.
        let contiguous = offsets
            .iter()
            .zip(sizes.iter())
            .zip(offsets.iter().skip(1))
            .all(|((&offset, &size), &next)| size != 0 && offset.checked_add(size) == Some(next));
        if contiguous {
            return Ok(None);
        }

        return deduplicate(offsets.iter().copied().zip(sizes.iter().copied()));
    }

    Ok(None)
}

fn deduplicate<K>(keys: impl IntoIterator<Item = K>) -> VortexResult<Option<SharedRows>>
where
    K: Eq + Hash,
{
    let mut value_by_key = HashMap::new();
    let mut representatives = Vec::new();
    let mut row_to_value = Vec::new();

    for (row, key) in keys.into_iter().enumerate() {
        let value = match value_by_key.entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let value = representatives.len();
                entry.insert(value);
                representatives.push(
                    u64::try_from(row)
                        .map_err(|_| vortex_err!("geo: geometry row index exceeds u64"))?,
                );
                value
            }
        };
        row_to_value.push(value);
    }

    Ok(
        (representatives.len() < row_to_value.len()).then_some(SharedRows {
            representatives,
            row_to_value,
        }),
    )
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::RecursiveCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use super::DecodedGeometries;
    use crate::test_harness::geo_session;
    use crate::test_harness::polygon_column;

    fn polygons() -> VortexResult<vortex_array::ArrayRef> {
        polygon_column(vec![
            vec![vec![(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, 0.0)]],
            vec![vec![(10.0, 10.0), (11.0, 10.0), (10.0, 11.0), (10.0, 10.0)]],
        ])
    }

    fn assert_shared(decoded: &DecodedGeometries) -> VortexResult<()> {
        let DecodedGeometries::Shared {
            values,
            row_to_value,
        } = decoded
        else {
            vortex_bail!("expected shared geometry decoding");
        };
        assert_eq!(values.len(), 2);
        assert_eq!(row_to_value, &[0, 0, 1, 0]);

        let rows = decoded.iter().collect::<Vec<_>>();
        assert!(std::ptr::eq(rows[0], rows[1]));
        assert!(std::ptr::eq(rows[0], rows[3]));
        assert_ne!(rows[0], rows[2]);
        Ok(())
    }

    #[test]
    fn keeps_contiguous_list_views_dense() -> VortexResult<()> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();
        let polygons = polygons()?
            .execute::<RecursiveCanonical>(&mut ctx)?
            .0
            .into_array();

        let DecodedGeometries::Dense(values) = DecodedGeometries::decode(&polygons, &mut ctx)?
        else {
            vortex_bail!("expected dense geometry decoding");
        };
        assert_eq!(values.len(), 2);
        Ok(())
    }

    #[test]
    fn reuses_taken_list_views() -> VortexResult<()> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();
        let polygons = polygons()?
            .execute::<RecursiveCanonical>(&mut ctx)?
            .0
            .into_array();
        let taken = polygons.take(PrimitiveArray::from_iter([0u64, 0, 1, 0]).into_array())?;

        assert_shared(&DecodedGeometries::decode(&taken, &mut ctx)?)?;
        Ok(())
    }

    #[test]
    fn reuses_dictionary_values() -> VortexResult<()> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();
        let values = polygons()?
            .execute::<RecursiveCanonical>(&mut ctx)?
            .0
            .into_array()
            .execute::<ExtensionArray>(&mut ctx)?;
        let dictionary = DictArray::try_new(
            PrimitiveArray::from_iter([0u64, 0, 1, 0]).into_array(),
            values.storage_array().clone(),
        )?;
        let polygons =
            ExtensionArray::try_new(values.ext_dtype().clone(), dictionary.into_array())?
                .into_array();

        assert_shared(&DecodedGeometries::decode(&polygons, &mut ctx)?)?;
        Ok(())
    }
}
