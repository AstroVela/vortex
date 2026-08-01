// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Differential property testing for tiled primitive fixed-size lists.

use std::num::NonZeroU32;
use std::ops::ControlFlow;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_tiled_fsl::TileGeometry;
use vortex_tiled_fsl::TiledFixedSizeList;
use vortex_tiled_fsl::TiledFixedSizeListArrayExt;
use vortex_tiled_fsl::TiledFixedSizeListArraySlotsExt;

use crate::array::assert_array_eq;
use crate::array::assert_scalar_eq;
use crate::array::slice_canonical_array;
use crate::array::take_canonical_array;
use crate::error::Backtrace;
use crate::error::VortexFuzzError;
use crate::error::VortexFuzzResult;

static TILED_FSL_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_tiled_fsl::initialize(&session);
    session
});

#[derive(Clone, Debug)]
pub enum TiledFslAction {
    CheckTiles,
    ScalarAt(u16),
    Slice { start: u16, stop: u16 },
    Take(Vec<Option<u16>>),
    Reconstruct,
    ReconstructSerde,
}

#[derive(Debug)]
pub struct FuzzTiledFsl {
    canonical: ArrayRef,
    geometry: TileGeometry,
    actions: Vec<TiledFslAction>,
}

impl<'a> Arbitrary<'a> for FuzzTiledFsl {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let ptype: PType = u.arbitrary()?;
        let element_nullability: Nullability = u.arbitrary()?;
        let outer_nullability: Nullability = u.arbitrary()?;
        let list_size = u.int_in_range(0u32..=64)?;
        let row_count = u.int_in_range(0usize..=128)?;
        let dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(ptype, element_nullability)),
            list_size,
            outer_nullability,
        );
        let canonical = ArbitraryArray::arbitrary_with_config(
            u,
            &ArbitraryArrayConfig {
                dtype: Some(dtype),
                len: row_count..=row_count,
            },
        )?
        .0;

        let geometry = TileGeometry::new(
            NonZeroU32::new(u.int_in_range(1u32..=128)?)
                .ok_or(arbitrary::Error::IncorrectFormat)?,
            NonZeroU32::new(u.int_in_range(1u32..=128)?)
                .ok_or(arbitrary::Error::IncorrectFormat)?,
        );
        let action_count = u.int_in_range(1usize..=8)?;
        let mut actions = Vec::with_capacity(action_count);
        for _ in 0..action_count {
            actions.push(match u.int_in_range(0u8..=5)? {
                0 => TiledFslAction::CheckTiles,
                1 => TiledFslAction::ScalarAt(u.arbitrary()?),
                2 => TiledFslAction::Slice {
                    start: u.arbitrary()?,
                    stop: u.arbitrary()?,
                },
                3 => {
                    let take_len = u.int_in_range(0usize..=64)?;
                    let mut seeds = Vec::with_capacity(take_len);
                    for _ in 0..take_len {
                        seeds.push(u.arbitrary()?);
                    }
                    TiledFslAction::Take(seeds)
                }
                4 => TiledFslAction::Reconstruct,
                5 => TiledFslAction::ReconstructSerde,
                _ => unreachable!("action tag is bounded"),
            });
        }

        Ok(Self {
            canonical,
            geometry,
            actions,
        })
    }
}

#[expect(clippy::result_large_err)]
fn fuzz<T>(result: VortexResult<T>) -> VortexFuzzResult<T> {
    result.map_err(|error| VortexFuzzError::VortexError(error, Backtrace::capture()))
}

fn assert_tiled_geometry(array: &ArrayRef, expected: TileGeometry) -> VortexResult<()> {
    if !array.is::<TiledFixedSizeList>() {
        vortex_bail!("expected nondegenerate operation to retain tiled FSL");
    }
    let actual = array.as_::<TiledFixedSizeList>().geometry();
    if actual != expected {
        vortex_bail!("expected geometry {expected:?}, found {actual:?}");
    }
    Ok(())
}

struct SerializedChildren(Vec<ArrayRef>);

impl ArrayChildren for SerializedChildren {
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        let child = self
            .0
            .as_slice()
            .get(index)
            .ok_or_else(|| vortex_err!(InvalidArgument: "missing serialized child {index}"))?;
        vortex_ensure!(
            child.dtype() == dtype && child.len() == len,
            InvalidArgument:
            "serialized child {index} has dtype {} and len {}, expected {dtype} and {len}",
            child.dtype(),
            child.len()
        );
        Ok(child.clone())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

fn validate_array_tree(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    for descendant in array.depth_first_traversal() {
        descendant.validity()?.execute_mask(descendant.len(), ctx)?;
    }
    Ok(())
}

fn reconstruct_serde(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let tiled = array.as_::<TiledFixedSizeList>();
    let expected_row_offset = tiled.row_offset();
    let expected_backing_rows = tiled.backing_rows();
    let metadata = <TiledFixedSizeList as ArrayVTable>::serialize(tiled, ctx.session())?
        .ok_or_else(|| vortex_err!("tiled fixed-size list did not serialize metadata"))?;
    let children = SerializedChildren(array.slots().iter().flatten().cloned().collect());
    let parts = <TiledFixedSizeList as ArrayVTable>::deserialize(
        &TiledFixedSizeList,
        array.dtype(),
        array.len(),
        &metadata,
        &[] as &[BufferHandle],
        &children,
        ctx.session(),
    )?;
    let reconstructed = Array::<TiledFixedSizeList>::try_from_parts(parts)?.into_array();
    let reconstructed_tiled = reconstructed.as_::<TiledFixedSizeList>();
    vortex_ensure!(
        reconstructed_tiled.row_offset() == expected_row_offset,
        "serde changed row offset from {expected_row_offset} to {}",
        reconstructed_tiled.row_offset()
    );
    vortex_ensure!(
        reconstructed_tiled.backing_rows() == expected_backing_rows,
        "serde changed backing rows from {expected_backing_rows} to {}",
        reconstructed_tiled.backing_rows()
    );
    validate_array_tree(&reconstructed, ctx)?;
    Ok(reconstructed)
}

fn expected_tile_indices(
    list_size: usize,
    rows: Range<usize>,
    dimensions: Range<usize>,
) -> ArrayRef {
    PrimitiveArray::from_iter(dimensions.flat_map(|dimension| {
        rows.clone()
            .map(move |row| (row * list_size + dimension) as u64)
    }))
    .into_array()
}

#[expect(clippy::result_large_err)]
fn check_tiles(
    canonical: &ArrayRef,
    tiled: &ArrayRef,
    step: usize,
    ctx: &mut ExecutionCtx,
) -> VortexFuzzResult<()> {
    fuzz((|| {
        vortex_ensure!(
            tiled.is::<TiledFixedSizeList>(),
            "tile check requires a tiled fixed-size list"
        );
        Ok(())
    })())?;
    let canonical_fsl = fuzz(canonical.clone().execute::<FixedSizeListArray>(ctx))?;
    let tiled_fsl = tiled.as_::<TiledFixedSizeList>();
    let rows = canonical.len();
    let list_size = canonical_fsl.list_size() as usize;
    let tile_rows = tiled_fsl.geometry().rows().get() as usize;
    let tile_dimensions = tiled_fsl.geometry().dimensions().get() as usize;
    let expected_row_tile_count = rows.div_ceil(tile_rows);
    let expected_dimension_tile_count = list_size.div_ceil(tile_dimensions);

    fuzz((|| {
        vortex_ensure!(
            tiled_fsl.row_tile_count() == expected_row_tile_count,
            "row tile count mismatch: expected {expected_row_tile_count}, found {}",
            tiled_fsl.row_tile_count()
        );
        vortex_ensure!(
            tiled_fsl.dimension_tile_count() == expected_dimension_tile_count,
            "dimension tile count mismatch: expected {expected_dimension_tile_count}, found {}",
            tiled_fsl.dimension_tile_count()
        );
        Ok(())
    })())?;

    let outer_mask = fuzz(
        canonical
            .validity()
            .and_then(|validity| validity.execute_mask(rows, ctx)),
    )?;
    let mut physical_cursor = 0usize;
    let mut tile_count = 0usize;

    for (tile_index, bounds) in tiled_fsl.tiles().enumerate() {
        let dimension_tile = tile_index / expected_row_tile_count;
        let row_tile = tile_index % expected_row_tile_count;
        let row_start = row_tile * tile_rows;
        let row_end = row_start.saturating_add(tile_rows).min(rows);
        let dimension_start = dimension_tile * tile_dimensions;
        let dimension_end = dimension_start
            .saturating_add(tile_dimensions)
            .min(list_size);
        let physical_len = (row_end - row_start) * (dimension_end - dimension_start);
        let expected_physical_end = physical_cursor + physical_len;
        let expected_rows = row_start..row_end;
        let expected_dimensions = dimension_start..dimension_end;
        let expected_physical = physical_cursor..expected_physical_end;

        fuzz((|| {
            vortex_ensure!(
                dimension_tile < expected_dimension_tile_count,
                "tile iterator yielded unexpected tile {tile_index}"
            );
            vortex_ensure!(
                bounds.row_range == expected_rows,
                "tile {tile_index} row range mismatch: expected {expected_rows:?}, found {:?}",
                bounds.row_range
            );
            vortex_ensure!(
                bounds.dimension_range == expected_dimensions,
                "tile {tile_index} dimension range mismatch: expected {expected_dimensions:?}, found {:?}",
                bounds.dimension_range
            );
            vortex_ensure!(
                bounds.physical_range == expected_physical,
                "tile {tile_index} physical range mismatch: expected {expected_physical:?}, found {:?}",
                bounds.physical_range
            );
            vortex_ensure!(
                bounds.physical_range.len()
                    == bounds.row_range.len() * bounds.dimension_range.len(),
                "tile {tile_index} physical cardinality mismatch"
            );
            Ok(())
        })())?;

        let expected_tile = fuzz(canonical_fsl.elements().clone().take(expected_tile_indices(
            list_size,
            expected_rows.clone(),
            expected_dimensions.clone(),
        )))?;
        let actual_tile = fuzz(tiled_fsl.tile_elements(&bounds))?;
        let selected_positions = expected_dimensions
            .clone()
            .flat_map(|dimension| {
                let expected_row_start = expected_rows.start;
                let expected_row_count = expected_rows.len();
                let expected_dimension_start = expected_dimensions.start;
                expected_rows.clone().filter_map({
                    let outer_mask = &outer_mask;
                    move |row| {
                        outer_mask.value(row).then_some(
                            (dimension - expected_dimension_start) * expected_row_count
                                + (row - expected_row_start),
                        )
                    }
                })
            })
            .map(|position| position as u64);
        let selection = PrimitiveArray::from_iter(selected_positions).into_array();
        let selected_expected = fuzz(expected_tile.take(selection.clone()))?;
        let selected_actual = fuzz(actual_tile.take(selection))?;
        assert_array_eq(&selected_expected, &selected_actual, step, ctx)?;

        physical_cursor = expected_physical_end;
        tile_count += 1;
    }

    fuzz((|| {
        vortex_ensure!(
            tile_count == expected_row_tile_count * expected_dimension_tile_count,
            "tile iterator count mismatch: expected {}, found {tile_count}",
            expected_row_tile_count * expected_dimension_tile_count
        );
        vortex_ensure!(
            physical_cursor == rows * list_size,
            "physical tile ranges cover {physical_cursor} values, expected {}",
            rows * list_size
        );
        Ok(())
    })())
}

#[expect(clippy::result_large_err)]
fn execute_action(
    action: TiledFslAction,
    canonical: &mut ArrayRef,
    tiled: &mut ArrayRef,
    geometry: TileGeometry,
    step: usize,
    ctx: &mut ExecutionCtx,
) -> VortexFuzzResult<ControlFlow<()>> {
    match action {
        TiledFslAction::CheckTiles => {
            if tiled.is::<TiledFixedSizeList>() {
                check_tiles(canonical, tiled, step, ctx)?;
            }
        }
        TiledFslAction::ScalarAt(seed) => {
            if canonical.is_empty() {
                return Ok(ControlFlow::Continue(()));
            }
            let row = usize::from(seed) % canonical.len();
            let expected = fuzz(canonical.execute_scalar(row, ctx))?;
            let actual = fuzz(tiled.execute_scalar(row, ctx))?;
            assert_scalar_eq(&expected, &actual, step)?;
        }
        TiledFslAction::Slice { start, stop } => {
            let source_is_empty = canonical.is_empty();
            let source_len = canonical.len();
            let source_is_tiled = tiled.is::<TiledFixedSizeList>();
            let source_is_full_width =
                source_is_tiled && tiled.as_::<TiledFixedSizeList>().is_full_width();
            let first = usize::from(start).min(usize::from(stop));
            let last = usize::from(start).max(usize::from(stop));
            let start = first % (canonical.len() + 1);
            let stop = start + last % (canonical.len() - start + 1);
            *canonical = fuzz(slice_canonical_array(canonical, start, stop, ctx))?;
            *tiled = fuzz(tiled.clone().slice(start..stop))?;
            if tiled.is_empty() {
                if source_is_empty {
                    fuzz(assert_tiled_geometry(tiled, geometry))?;
                    assert_array_eq(canonical, tiled, step, ctx)?;
                    return Ok(ControlFlow::Continue(()));
                }
                fuzz((|| {
                    vortex_ensure!(
                        tiled.is::<FixedSizeList>(),
                        "expected an empty slice of a nonempty source to be canonical FSL"
                    );
                    Ok(())
                })())?;
                assert_array_eq(canonical, tiled, step, ctx)?;
                return Ok(ControlFlow::Break(()));
            }
            let tile_rows = geometry.rows().get() as usize;
            let aligned_multi_slab =
                start % tile_rows == 0 && (stop % tile_rows == 0 || stop == source_len);
            if source_is_tiled && (source_is_full_width || aligned_multi_slab) {
                fuzz(assert_tiled_geometry(tiled, geometry))?;
            } else {
                fuzz((|| {
                    vortex_ensure!(
                        tiled.is::<Slice>(),
                        "expected a non-preserving slice to remain a lazy Slice, found {}",
                        tiled.encoding_id()
                    );
                    Ok(())
                })())?;
            }
            assert_array_eq(canonical, tiled, step, ctx)?;
        }
        TiledFslAction::Take(seeds) => {
            let source_is_empty = canonical.is_empty();
            let indices = seeds
                .into_iter()
                .map(|seed| {
                    seed.and_then(|seed| {
                        (!source_is_empty).then(|| usize::from(seed) % canonical.len())
                    })
                })
                .collect::<Vec<_>>();
            let index_array = if indices.contains(&None) {
                PrimitiveArray::from_option_iter(
                    indices.iter().map(|index| index.map(|index| index as u64)),
                )
                .into_array()
            } else {
                PrimitiveArray::from_iter(indices.iter().flatten().map(|index| *index as u64))
                    .into_array()
            };
            *canonical = fuzz(take_canonical_array(canonical, &indices, ctx))?;
            let lazy = fuzz(tiled.clone().take(index_array))?;
            *tiled = fuzz(lazy.execute::<Canonical>(ctx))?.into_array();
            assert_array_eq(canonical, tiled, step, ctx)?;
            return Ok(ControlFlow::Break(()));
        }
        TiledFslAction::Reconstruct => {
            if !tiled.is::<TiledFixedSizeList>() {
                return Ok(ControlFlow::Continue(()));
            }
            let array = tiled.as_::<TiledFixedSizeList>();
            if array.row_offset() != 0 || array.backing_rows() != array.len() {
                return Ok(ControlFlow::Continue(()));
            }
            *tiled = fuzz(TiledFixedSizeList::try_new(
                array.elements().clone(),
                array.list_size(),
                array.array_validity(),
                array.len(),
                geometry,
            ))?
            .into_array();
            assert_array_eq(canonical, tiled, step, ctx)?;
        }
        TiledFslAction::ReconstructSerde => {
            if !tiled.is::<TiledFixedSizeList>() {
                return Ok(ControlFlow::Continue(()));
            }
            *tiled = fuzz(reconstruct_serde(tiled, ctx))?;
            fuzz(assert_tiled_geometry(tiled, geometry))?;
            assert_array_eq(canonical, tiled, step, ctx)?;
        }
    }
    Ok(ControlFlow::Continue(()))
}

#[expect(clippy::result_large_err)]
pub fn run_tiled_fsl(input: FuzzTiledFsl) -> VortexFuzzResult<()> {
    let mut ctx = TILED_FSL_SESSION.create_execution_ctx();
    let mut canonical = fuzz(
        input
            .canonical
            .execute::<FixedSizeListArray>(&mut ctx)
            .map(IntoArray::into_array),
    )?;
    let canonical_fsl = canonical.as_::<FixedSizeList>();
    let mut tiled = fuzz(TiledFixedSizeList::encode(
        canonical_fsl,
        input.geometry,
        &mut ctx,
    ))?
    .into_array();

    fuzz(assert_tiled_geometry(&tiled, input.geometry))?;
    assert_array_eq(&canonical, &tiled, 0, &mut ctx)?;

    if !canonical.is_empty() {
        let tile_rows = input.geometry.rows().get() as usize;
        let mut probe_rows = vec![0, canonical.len() - 1];
        probe_rows.extend((tile_rows..canonical.len()).step_by(tile_rows));
        probe_rows.sort_unstable();
        probe_rows.dedup();
        for row in probe_rows {
            let expected = fuzz(canonical.execute_scalar(row, &mut ctx))?;
            let actual = fuzz(tiled.execute_scalar(row, &mut ctx))?;
            assert_scalar_eq(&expected, &actual, 0)?;
        }
    }

    let expected_row_tiles = canonical
        .len()
        .div_ceil(input.geometry.rows().get() as usize);
    let expected_dimension_tiles =
        (canonical_fsl.list_size() as usize).div_ceil(input.geometry.dimensions().get() as usize);
    let tiled_fsl = tiled.as_::<TiledFixedSizeList>();
    fuzz((|| {
        vortex_ensure!(
            tiled_fsl.row_tile_count() == expected_row_tiles,
            "row tile count mismatch: expected {expected_row_tiles}, found {}",
            tiled_fsl.row_tile_count()
        );
        vortex_ensure!(
            tiled_fsl.dimension_tile_count() == expected_dimension_tiles,
            "dimension tile count mismatch: expected {expected_dimension_tiles}, found {}",
            tiled_fsl.dimension_tile_count()
        );
        Ok(())
    })())?;

    for (step, action) in input.actions.into_iter().enumerate() {
        if execute_action(
            action,
            &mut canonical,
            &mut tiled,
            input.geometry,
            step,
            &mut ctx,
        )?
        .is_break()
        {
            break;
        }
    }
    Ok(())
}

fn geometry(rows: u32, dimensions: u32) -> TileGeometry {
    let Some(rows) = NonZeroU32::new(rows) else {
        unreachable!("deterministic geometry rows are nonzero");
    };
    let Some(dimensions) = NonZeroU32::new(dimensions) else {
        unreachable!("deterministic geometry dimensions are nonzero");
    };
    TileGeometry::new(rows, dimensions)
}

#[expect(clippy::result_large_err)]
pub fn deterministic_tiled_fsl_cases() -> VortexFuzzResult<Vec<FuzzTiledFsl>> {
    let zero_by_zero = FixedSizeListArray::new(
        PrimitiveArray::from_iter::<[i32; 0]>([]).into_array(),
        0,
        Validity::NonNullable,
        0,
    )
    .into_array();

    let nullable_u16 = FixedSizeListArray::new(
        PrimitiveArray::from_option_iter((0u16..15).map(|value| (value % 4 != 1).then_some(value)))
            .into_array(),
        5,
        Validity::from_iter([true, false, true]),
        3,
    )
    .into_array();

    let nullable_f32 = FixedSizeListArray::new(
        PrimitiveArray::from_option_iter(
            (0..65 * 129).map(|index| (index % 7 != 3).then_some((index as f32) * 0.25 - 1000.0)),
        )
        .into_array(),
        129,
        Validity::from_iter((0..65).map(|row| row % 5 != 2)),
        65,
    )
    .into_array();

    let full_width_row_view = FixedSizeListArray::new(
        PrimitiveArray::new(
            Buffer::from_iter((0..200 * 128).map(|index| index as u32)),
            Validity::from_iter((0..200 * 128).map(|index| index % 11 != 0)),
        )
        .into_array(),
        128,
        Validity::from_iter((0..200).map(|row| row % 7 != 0)),
        200,
    )
    .into_array();

    let multi_slab_slices = FixedSizeListArray::new(
        PrimitiveArray::new(
            Buffer::from_iter((0..200 * 129).map(|index| index as u32)),
            Validity::from_iter((0..200 * 129).map(|index| index % 13 != 0)),
        )
        .into_array(),
        129,
        Validity::from_iter((0..200).map(|row| row % 5 != 0)),
        200,
    )
    .into_array();

    Ok(vec![
        FuzzTiledFsl {
            canonical: zero_by_zero,
            geometry: geometry(128, 128),
            actions: vec![
                TiledFslAction::CheckTiles,
                TiledFslAction::Slice { start: 0, stop: 0 },
                TiledFslAction::CheckTiles,
                TiledFslAction::Reconstruct,
            ],
        },
        FuzzTiledFsl {
            canonical: nullable_u16,
            geometry: geometry(2, 3),
            actions: vec![
                TiledFslAction::CheckTiles,
                TiledFslAction::Take(vec![Some(2), None, Some(2), Some(0)]),
                TiledFslAction::Reconstruct,
            ],
        },
        FuzzTiledFsl {
            canonical: nullable_f32,
            geometry: geometry(32, 64),
            actions: vec![
                TiledFslAction::Slice {
                    start: 31,
                    stop: 38,
                },
                TiledFslAction::Take(vec![Some(2), Some(1), Some(0)]),
                TiledFslAction::ScalarAt(1),
                TiledFslAction::CheckTiles,
            ],
        },
        FuzzTiledFsl {
            canonical: full_width_row_view,
            geometry: geometry(64, 128),
            actions: vec![
                TiledFslAction::CheckTiles,
                TiledFslAction::Slice {
                    start: 10,
                    stop: 150,
                },
                TiledFslAction::ScalarAt(53),
                TiledFslAction::ScalarAt(54),
                TiledFslAction::ReconstructSerde,
                TiledFslAction::Slice {
                    start: 60,
                    stop: 70,
                },
                TiledFslAction::ScalarAt(9),
                TiledFslAction::Take(vec![Some(9), None, Some(0)]),
            ],
        },
        FuzzTiledFsl {
            canonical: multi_slab_slices,
            geometry: geometry(64, 64),
            actions: vec![
                TiledFslAction::CheckTiles,
                TiledFslAction::Slice {
                    start: 64,
                    stop: 192,
                },
                TiledFslAction::ReconstructSerde,
                TiledFslAction::ScalarAt(63),
                TiledFslAction::Slice { start: 1, stop: 66 },
                TiledFslAction::ScalarAt(0),
                TiledFslAction::Slice { start: 1, stop: 64 },
                TiledFslAction::Take(vec![Some(62), None, Some(0)]),
            ],
        },
    ])
}

#[cfg(test)]
mod tests {
    use std::ops::ControlFlow;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::FixedSizeList;
    use vortex_tiled_fsl::TiledFixedSizeList;
    use vortex_tiled_fsl::TiledFixedSizeListArrayExt;

    use super::TILED_FSL_SESSION;
    use super::TiledFslAction;
    use super::deterministic_tiled_fsl_cases;
    use super::execute_action;
    use super::fuzz;
    use super::run_tiled_fsl;
    use crate::error::VortexFuzzResult;

    #[test]
    #[expect(clippy::result_large_err)]
    fn empty_full_range_slice_continues_with_tiled_geometry() -> VortexFuzzResult<()> {
        let mut cases = deterministic_tiled_fsl_cases()?;
        let input = cases.remove(0);
        let geometry = input.geometry;
        let mut canonical = input.canonical;
        let mut ctx = TILED_FSL_SESSION.create_execution_ctx();
        let mut tiled = fuzz(TiledFixedSizeList::encode(
            canonical.as_::<FixedSizeList>(),
            geometry,
            &mut ctx,
        ))?
        .into_array();

        let control = execute_action(
            TiledFslAction::Slice { start: 0, stop: 0 },
            &mut canonical,
            &mut tiled,
            geometry,
            0,
            &mut ctx,
        )?;

        assert_eq!(control, ControlFlow::Continue(()));
        assert!(tiled.is::<TiledFixedSizeList>());
        assert_eq!(tiled.as_::<TiledFixedSizeList>().geometry(), geometry);
        Ok(())
    }

    #[test]
    #[expect(clippy::result_large_err)]
    fn deterministic_tiled_fsl_smoke() -> VortexFuzzResult<()> {
        let cases = deterministic_tiled_fsl_cases()?;
        assert_eq!(cases.len(), 5);
        for input in cases {
            run_tiled_fsl(input)?;
        }
        Ok(())
    }

    #[test]
    #[expect(clippy::result_large_err)]
    fn serde_reconstruction_validates_offset_view_tree() -> VortexFuzzResult<()> {
        let mut cases = deterministic_tiled_fsl_cases()?;
        let input = cases.remove(3);
        let geometry = input.geometry;
        let mut canonical = input.canonical;
        let mut ctx = TILED_FSL_SESSION.create_execution_ctx();
        let mut tiled = fuzz(TiledFixedSizeList::encode(
            canonical.as_::<FixedSizeList>(),
            geometry,
            &mut ctx,
        ))?
        .into_array();

        let slice_control = execute_action(
            TiledFslAction::Slice {
                start: 10,
                stop: 150,
            },
            &mut canonical,
            &mut tiled,
            geometry,
            0,
            &mut ctx,
        )?;
        let control = execute_action(
            TiledFslAction::ReconstructSerde,
            &mut canonical,
            &mut tiled,
            geometry,
            1,
            &mut ctx,
        )?;

        assert_eq!(slice_control, ControlFlow::Continue(()));
        assert_eq!(control, ControlFlow::Continue(()));
        assert!(tiled.is::<TiledFixedSizeList>());
        Ok(())
    }
}
