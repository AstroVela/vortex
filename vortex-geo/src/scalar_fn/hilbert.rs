// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `hilbert`: a locality-preserving `u32` key for native geometry columns.
//!
//! Each key encodes the center of the geometry's XY envelope. Bounds are reduced directly from
//! native coordinate buffers and consumed immediately: no `geoarrow.box` array or WKB decode is
//! materialized along the way.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::Rect;
use crate::extension::coordinate::ordinates;
use crate::extension::for_each_row_coordinate_bounds;
use crate::extension::validate_geometry_operands;

/// A locality-preserving `u32` key for the center of each native geometry's XY envelope.
///
/// The coordinate mapping follows DuckDB Spatial's bounds-free convention: it orders the full
/// IEEE-754 `f32` coordinate range, then applies a 16-bit-per-axis Hilbert encoding. A null or
/// empty geometry has no envelope center and therefore yields null. In particular, GeoArrow's
/// `(NaN, NaN)` representation of `POINT EMPTY` yields null rather than a sortable key.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GeoHilbert;

impl GeoHilbert {
    /// A lazy `ScalarFnArray` computing one Hilbert key per row of `geometry`.
    pub fn try_new_array(geometry: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(GeoHilbert, EmptyOptions).erased(),
            vec![geometry],
        )
    }
}

/// Map an IEEE-754 `f32` into an unsigned integer that is monotonic over non-NaN values, retaining
/// the IEEE signed-zero distinction. All NaNs map to the final bucket. This is the bounds-free
/// coordinate mapping used by DuckDB's `ST_Hilbert` implementation.
#[inline]
fn f32_to_hilbert_u32(value: f32) -> u32 {
    if value.is_nan() {
        return u32::MAX;
    }
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        bits ^ u32::MAX
    } else {
        bits | 0x8000_0000
    }
}

/// Whether XY ordinates encode GeoArrow's `POINT EMPTY` sentinel for this XY key function.
///
/// Only the complete `(NaN, NaN)` pair is empty. A partial NaN remains a coordinate value rather
/// than being silently converted into an empty geometry.
#[inline]
fn is_empty_point(x: f64, y: f64) -> bool {
    x.is_nan() && y.is_nan()
}

/// Interleave the low 16 bits of `value` with zero bits.
#[inline]
fn hilbert_interleave(mut value: u32) -> u32 {
    value = (value | (value << 8)) & 0x00ff_00ff;
    value = (value | (value << 4)) & 0x0f0f_0f0f;
    value = (value | (value << 2)) & 0x3333_3333;
    (value | (value << 1)) & 0x5555_5555
}

/// Encode `x` and `y` as a 16-bit-per-axis Hilbert index.
///
/// This is the public-domain algorithm used by DuckDB Spatial, retained in structure so the stable
/// bounds-free mapping has the same orientation and bit ordering.
#[inline]
fn hilbert_encode_16(x: u32, y: u32) -> u32 {
    let input_x = x;
    let input_y = y;
    let mut state_a = x ^ y;
    let mut state_b = 0xffff ^ state_a;
    let mut state_c = 0xffff ^ (x | y);
    let mut state_d = x & (y ^ 0xffff);
    let mut next_a = state_a | (state_b >> 1);
    let mut next_b = (state_a >> 1) ^ state_a;
    let mut next_c = ((state_c >> 1) ^ (state_b & (state_d >> 1))) ^ state_c;
    let mut next_d = ((state_a & (state_c >> 1)) ^ (state_d >> 1)) ^ state_d;

    state_a = next_a;
    state_b = next_b;
    state_c = next_c;
    state_d = next_d;
    next_a = (state_a & (state_a >> 2)) ^ (state_b & (state_b >> 2));
    next_b = (state_a & (state_b >> 2)) ^ (state_b & ((state_a ^ state_b) >> 2));
    next_c ^= (state_a & (state_c >> 2)) ^ (state_b & (state_d >> 2));
    next_d ^= (state_b & (state_c >> 2)) ^ ((state_a ^ state_b) & (state_d >> 2));

    state_a = next_a;
    state_b = next_b;
    state_c = next_c;
    state_d = next_d;
    next_a = (state_a & (state_a >> 4)) ^ (state_b & (state_b >> 4));
    next_b = (state_a & (state_b >> 4)) ^ (state_b & ((state_a ^ state_b) >> 4));
    next_c ^= (state_a & (state_c >> 4)) ^ (state_b & (state_d >> 4));
    next_d ^= (state_b & (state_c >> 4)) ^ ((state_a ^ state_b) & (state_d >> 4));

    state_a = next_a;
    state_b = next_b;
    state_c = next_c;
    state_d = next_d;
    next_c ^= (state_a & (state_c >> 8)) ^ (state_b & (state_d >> 8));
    next_d ^= (state_b & (state_c >> 8)) ^ ((state_a ^ state_b) & (state_d >> 8));

    let state_a = next_c ^ (next_c >> 1);
    let state_b = next_d ^ (next_d >> 1);
    let i0 = input_x ^ input_y;
    let i1 = state_b | (0xffff ^ (i0 | state_a));

    (hilbert_interleave(i1) << 1) | hilbert_interleave(i0)
}

/// Encode the given XY center with the stable, bounds-free mapping.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the public bounds-free contract deliberately maps to the full f32 coordinate range"
)]
#[inline]
fn hilbert_key(x: f64, y: f64) -> u32 {
    hilbert_encode_16(f32_to_hilbert_u32(x as f32), f32_to_hilbert_u32(y as f32))
}

/// Return the center of `[xmin, ymin, xmax, ymax]` without constructing a box array.
#[inline]
fn bounds_center([xmin, ymin, xmax, ymax]: [f64; 4]) -> (f64, f64) {
    (xmin + (xmax - xmin) / 2.0, ymin + (ymax - ymin) / 2.0)
}

/// Materialize keys for a point column directly from its coordinate leaves.
///
/// GeoArrow represents `POINT EMPTY` as `(NaN, NaN)`. A partial NaN remains a coordinate value
/// and uses the stable float mapping; only the pair denotes an empty point.
fn point_keys(
    storage: ArrayRef,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let coords = storage.execute::<StructArray>(ctx)?;
    let xs = ordinates(&coords, "x", ctx)?;
    let ys = ordinates(&coords, "y", ctx)?;
    let mut keys = BufferMut::zeroed(xs.len());
    for row in 0..keys.len() {
        keys[row] = hilbert_key(xs[row], ys[row]);
    }
    let non_empty = Mask::from(BitBuffer::collect_bool(xs.len(), |row| {
        !is_empty_point(xs[row], ys[row])
    }));
    let valid = validity.execute_mask(xs.len(), ctx)?;
    let validity = Validity::from_mask(&valid & &non_empty, Nullability::Nullable);
    Ok(PrimitiveArray::new(keys.freeze(), validity).into_array())
}

/// Materialize keys for a native rectangle column directly from its bound fields.
fn rect_keys(
    storage: ArrayRef,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let boxes = storage.execute::<StructArray>(ctx)?;
    let xmins = ordinates(&boxes, "xmin", ctx)?;
    let ymins = ordinates(&boxes, "ymin", ctx)?;
    let xmaxs = ordinates(&boxes, "xmax", ctx)?;
    let ymaxs = ordinates(&boxes, "ymax", ctx)?;
    let mut keys = BufferMut::zeroed(xmins.len());
    for row in 0..keys.len() {
        let (x, y) = bounds_center([xmins[row], ymins[row], xmaxs[row], ymaxs[row]]);
        keys[row] = hilbert_key(x, y);
    }
    Ok(PrimitiveArray::new(keys.freeze(), validity.into_nullable()).into_array())
}

/// Materialize keys for a list-backed native geometry column. The shared traversal identifies a
/// row's coordinate slice through its list parents, calculates its bounds, and lets this kernel
/// consume the result immediately.
fn nested_geometry_keys(
    storage: ArrayRef,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = storage.len();
    let valid = validity.execute_mask(len, ctx)?;
    let mut keys = BufferMut::zeroed(len);
    let non_empty = for_each_row_coordinate_bounds(storage, ctx, |row, bounds| {
        let (x, y) = bounds_center(bounds);
        keys[row] = hilbert_key(x, y);
    })?;
    let validity = Validity::from_mask(&valid & &non_empty, Nullability::Nullable);
    Ok(PrimitiveArray::new(keys.freeze(), validity).into_array())
}

impl ScalarFnVTable for GeoHilbert {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.geo.hilbert");
        *ID
    }

    fn serialize(&self, _: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _: &[u8], _: &VortexSession) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("geometry"),
            _ => unreachable!("hilbert has exactly one child"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        validate_geometry_operands(dtypes)?;
        // Empty geometries have no envelope center, so the output is nullable even if the input
        // itself cannot contain nulls.
        Ok(DType::Primitive(PType::U32, Nullability::Nullable))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let geometry = args.get(0)?;
        let ext = geometry
            .dtype()
            .as_extension_opt()
            .ok_or_else(|| vortex_err!("geo: hilbert operand is not a geometry extension type"))?;
        let validity = geometry.validity()?;
        let storage = geometry
            .clone()
            .execute::<ExtensionArray>(ctx)?
            .storage_array()
            .clone();

        if ext.is::<Rect>() {
            rect_keys(storage, validity, ctx)
        } else if !storage.dtype().is_list() {
            point_keys(storage, validity, ctx)
        } else {
            nested_geometry_keys(storage, validity, ctx)
        }
    }

    fn is_strict(&self, _: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use super::GeoHilbert;
    use super::f32_to_hilbert_u32;
    use super::hilbert_encode_16;
    use super::hilbert_key;
    use crate::test_harness::multipoint_column;
    use crate::test_harness::multipolygon_column;
    use crate::test_harness::nullable_multipolygon_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::point_column;
    use crate::test_harness::rect_column;

    /// Execute `GeoHilbert` over `array`.
    fn keys(array: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(GeoHilbert::try_new_array(array)?.into_array())
    }

    /// A known reference vector for the public-domain 16-bit Hilbert encoder used by DuckDB.
    #[test]
    fn hilbert_curve_orientation_is_stable() {
        assert_eq!(hilbert_encode_16(0, 0), 0);
        assert_eq!(hilbert_encode_16(1, 0), 1);
        assert_eq!(hilbert_encode_16(1, 1), 2);
        assert_eq!(hilbert_encode_16(0, 1), 3);
    }

    /// The bounds-free mapping places floats in numeric order and uses the final value for NaNs.
    #[test]
    fn float_mapping_is_stable() {
        assert!(f32_to_hilbert_u32(-1.0) < f32_to_hilbert_u32(0.0));
        assert!(f32_to_hilbert_u32(0.0) < f32_to_hilbert_u32(1.0));
        assert_eq!(f32_to_hilbert_u32(f32::NAN), u32::MAX);
    }

    /// A fixed vector locks down the full float mapping and the curve orientation together.
    #[test]
    fn bounds_free_key_is_stable() {
        assert_eq!(hilbert_key(2.0, 3.0), 838_860_800);
    }

    /// A point and any geometry whose envelope has the same center yield the same key.
    #[test]
    fn keys_the_envelope_center() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let points = point_column(vec![2.0], vec![3.0])?;
        let multipoints = multipoint_column(vec![vec![(0.0, 1.0), (4.0, 5.0)]])?;
        let rects = rect_column(vec![(0.0, 1.0, 4.0, 5.0)])?;
        let expected =
            PrimitiveArray::new(vec![hilbert_key(2.0, 3.0)], Validity::from_iter([true]))
                .into_array();

        assert_arrays_eq!(keys(points)?, expected, &mut ctx);
        assert_arrays_eq!(keys(multipoints)?, expected, &mut ctx);
        assert_arrays_eq!(keys(rects)?, expected, &mut ctx);
        Ok(())
    }

    /// An empty geometry and a null geometry both have no envelope center, preserving their row
    /// positions as null keys.
    #[test]
    fn empty_and_null_geometries_are_null() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let geometries = nullable_multipolygon_column(vec![
            Some(vec![vec![vec![(0.0, 0.0), (2.0, 2.0)]]]),
            Some(vec![]),
            None,
        ])?;
        let expected = PrimitiveArray::new(
            vec![hilbert_key(1.0, 1.0), 0, 0],
            Validity::from_iter([true, false, false]),
        )
        .into_array();
        assert_arrays_eq!(keys(geometries)?, expected, &mut ctx);
        Ok(())
    }

    /// Point coordinates are read directly from their `x`/`y` leaves, while null rows remain
    /// null and never affect an adjacent key.
    #[test]
    fn nullable_points_keep_rows_aligned() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let points = nullable_point_column(vec![Some((1.0, 2.0)), None, Some((3.0, 4.0))])?;
        let expected = PrimitiveArray::new(
            vec![hilbert_key(1.0, 2.0), 0, hilbert_key(3.0, 4.0)],
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        assert_arrays_eq!(keys(points)?, expected, &mut ctx);
        Ok(())
    }

    /// GeoArrow represents `POINT EMPTY` as `(NaN, NaN)`, which has no envelope center and must
    /// therefore produce a null key like every other empty geometry.
    #[test]
    fn empty_points_are_null() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let points = point_column(vec![f64::NAN, 1.0], vec![f64::NAN, 2.0])?;
        let expected = PrimitiveArray::new(
            vec![0, hilbert_key(1.0, 2.0)],
            Validity::from_iter([false, true]),
        )
        .into_array();
        assert_arrays_eq!(keys(points)?, expected, &mut ctx);
        Ok(())
    }

    /// A partial NaN is not GeoArrow's `POINT EMPTY` sentinel, so it retains a deterministic
    /// bounds-free key instead of being silently converted to null.
    #[test]
    fn partial_nan_point_is_not_empty() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let points = point_column(vec![f64::NAN], vec![2.0])?;
        let expected = PrimitiveArray::new(
            vec![hilbert_key(f64::NAN, 2.0)],
            Validity::from_iter([true]),
        )
        .into_array();
        assert_arrays_eq!(keys(points)?, expected, &mut ctx);
        Ok(())
    }

    /// The result type remains nullable even for non-nullable input because an empty geometry can
    /// occur in a valid native list column.
    #[test]
    fn output_is_nullable_u32() -> VortexResult<()> {
        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let out = GeoHilbert.return_dtype(&EmptyOptions, &[dtype])?;
        assert_eq!(out, DType::Primitive(PType::U32, Nullability::Nullable));
        Ok(())
    }

    /// A non-geometry operand is rejected at planning time.
    #[test]
    fn non_geometry_operand_is_rejected() {
        let numeric = DType::Primitive(PType::I32, Nullability::NonNullable);
        assert!(GeoHilbert.return_dtype(&EmptyOptions, &[numeric]).is_err());
    }

    /// Slicing still maps nested coordinates to their outer geometry rows before encoding.
    #[test]
    fn sliced_nested_geometry_keeps_rows_aligned() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();

        let geometries = multipolygon_column(vec![
            vec![vec![vec![(-100.0, -100.0), (100.0, 100.0)]]],
            vec![vec![vec![(0.0, 0.0), (2.0, 2.0)]]],
            vec![vec![vec![(4.0, 4.0), (6.0, 6.0)]]],
        ])?;
        let expected = PrimitiveArray::new(
            vec![hilbert_key(1.0, 1.0), hilbert_key(5.0, 5.0)],
            Validity::from_iter([true, true]),
        )
        .into_array();
        assert_arrays_eq!(keys(geometries.slice(1..3)?)?, expected, &mut ctx);
        Ok(())
    }

    /// The scalar function executes to a canonical primitive array.
    #[test]
    fn result_is_primitive() -> VortexResult<()> {
        let session = crate::test_harness::geo_session();
        let mut ctx = session.create_execution_ctx();
        let result = keys(point_column(vec![1.0], vec![2.0])?)?
            .execute::<Canonical>(&mut ctx)?
            .into_primitive();
        assert_eq!(result.ptype(), PType::U32);
        Ok(())
    }
}
