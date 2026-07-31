# Tiled Fixed-Size List Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental Vortex encoding that stores primitive fixed-size-list elements in configurable two-dimensional tiles, preserves useful nondegenerate row operations through Vortex execution, and round-trips through files.

**Architecture:** A new `vortex-tiled-fsl` encoding crate owns tile geometry, the physical child order, transpose/inverse-transpose logic, and generic Vortex operations. The parent retains an ordinary logical `FixedSizeList<Primitive>` dtype, while its one primitive element child can itself use another Vortex encoding; outer validity remains a separate optional child. Vortex file and facade integration are feature-gated behind `unstable_encodings`; SpiralDB-specific scoring remains outside this plan.

**Tech Stack:** Rust, Vortex array vtables and execution contexts, Prost metadata, FastLanes composition tests, Divan benchmarks, Vortex file editions.

## Global Constraints

- Support only `DType::FixedSizeList` whose element dtype is `DType::Primitive`.
- Physical order is `dimension tile -> row tile -> dimension within tile -> row within tile`.
- `TileGeometry.rows` and `TileGeometry.dimensions` are both `NonZeroU32`.
- Store exactly `len * list_size` physical elements with checked arithmetic and no tail padding.
- Preserve outer row validity separately and transpose element validity with element values.
- Preserve the tiled parent and its geometry for nonempty `slice` results, the
  full `0..0` slice of an already-empty tiled source, and an executed
  nondegenerate `take` in a session that registers the encoding. Empty slices
  of nonempty sources are canonical; empty takes and takes from empty sources
  may use Vortex's canonical or constant degenerate results. No row operation
  may canonicalize the whole source.
- Use canonical `FixedSizeListArray` as the deterministic unit-test and fuzz-test oracle for every tiled operation.
- Keep production `vortex-tiled-fsl` independent of `vortex-fastlanes`; FastLanes is a dev-only composition dependency.
- Do not add the encoding to BtrBlocks or any automatic compression strategy.
- Register and re-export the encoding only through `unstable_encodings`.
- Do not add algorithm, arithmetic, padding, or layout-version metadata fields.
- Give every new public type, method, and field a rustdoc comment.
- Use `Array::try_from_parts` for construction; do not introduce unsafe array
  construction or unchecked indexing in this crate.
- Do not modify or remove the user-owned untracked `.agents/worktrees/` directory.
- Do not create an alternate Cargo target directory.
- Run Rust formatting after every Rust-editing task and use signed-off commits.
- This plan stops after the Vortex representation, file integration, and benchmarks. SpiralDB scorer integration gets a separate plan after this API and its measurements exist.

---

## File Map

The implementation is split by responsibility before task boundaries are
defined:

- `Cargo.toml`: workspace membership and workspace dependency declarations for
  `vortex-tiled-fsl`.
- `encodings/tiled-fsl/Cargo.toml`: production dependencies, test-only
  FastLanes dependency, and the Divan benchmark target.
- `encodings/tiled-fsl/src/lib.rs`: public re-exports and the single
  `initialize(&VortexSession)` registration entrypoint.
- `encodings/tiled-fsl/src/geometry.rs`: `TileGeometry`, `TileBounds`, checked
  logical-to-physical mapping, and the range-only iterator state.
- `encodings/tiled-fsl/src/array.rs`: array metadata, slots, validation,
  serialization, construction, validity, and public array accessors.
- `encodings/tiled-fsl/src/transpose.rs`: typed bulk transpose and inverse
  transpose for primitive values and element validity.
- `encodings/tiled-fsl/src/operations.rs`: canonical execution and bounded
  scalar access.
- `encodings/tiled-fsl/src/gather.rs`: output-proportional physical-index
  generation shared by row-selection operations.
- `encodings/tiled-fsl/src/slice.rs`: the preserving nonempty-slice rule.
- `encodings/tiled-fsl/src/take.rs`: checked index decoding and tiled take
  execution.
- `encodings/tiled-fsl/src/rules.rs`: parent reduction-rule registration.
- `encodings/tiled-fsl/src/kernel.rs`: execute-parent kernel registration.
- `encodings/tiled-fsl/src/tests.rs`: deterministic construction, operation,
  conformance, nullability, and composition tests using canonical FSL as the
  oracle.
- `encodings/tiled-fsl/goldenfiles/tiled_fsl.metadata`: two-field unstable
  metadata golden.
- `encodings/tiled-fsl/benches/tiled_fsl.rs`: representation and weighted-score
  proxy benchmarks.
- `vortex-file/Cargo.toml`, `vortex-file/src/lib.rs`, and
  `vortex-file/tests/tiled_fsl.rs`: unstable decoder registration and raw plus
  nested-FastLanes file round-trips.
- `vortex/Cargo.toml` and `vortex/src/lib.rs`: unstable facade dependency and
  re-export.
- `fuzz/src/tiled_fsl.rs`: bounded generator, independent canonical oracle, and
  composed action runner.
- `fuzz/fuzz_targets/tiled_fsl.rs`: native libFuzzer entrypoint.
- `fuzz/Cargo.toml`, `fuzz/src/lib.rs`, `fuzz/README.md`, and
  `.github/workflows/fuzz.yml`: fuzz target wiring, documentation, and scheduled
  corpus execution.

No SpiralDB source file changes in this plan. The later scorer integration is a
separate plan and consumes this Vortex API.

## Design Traceability

Every approved design requirement has one implementation owner and one
verification owner:

| Design contract | Implementation task | Verification task |
|---|---:|---:|
| Checked, unpadded two-dimensional geometry | 1 | 1, 9, 10 |
| Primitive-only FSL parent, exact child length, two-field metadata | 2 | 2, 8, 9 |
| Bulk value and element-validity transpose | 3 | 3, 9, 10 |
| Range-only traversal with no child ownership per tile | 4 | 4, 9, 12 |
| Output-proportional preserving nonempty slice | 5 | 5, 9, 10 |
| Output-proportional executed nondegenerate take | 6 | 6, 9, 10 |
| Independently encoded physical primitive child | 7 | 7, 8, 9, 12 |
| Experimental file and facade integration only | 8 | 8, 13 |
| Generic filter fallback, without a preserving filter kernel | none | 9 |
| Independent canonical FSL oracle | none | 9, 10 |
| Persistent scheduled fuzz corpus and crash reporting | 11 | 11, 13 |
| Representation and traversal measurements, without choosing a default | 12 | 12, 13 |
| SpiralDB-specific scoring remains downstream | none | 13 handoff |

---

### Task 1: Workspace crate and checked tile geometry

**Files:**
- Modify: `Cargo.toml:45-70`
- Modify: `Cargo.toml:290-320`
- Create: `encodings/tiled-fsl/Cargo.toml`
- Create: `encodings/tiled-fsl/src/lib.rs`
- Create: `encodings/tiled-fsl/src/geometry.rs`

**Interfaces:**
- Produces: `TileGeometry::new(NonZeroU32, NonZeroU32) -> TileGeometry`
- Produces: `TileGeometry::{rows, dimensions}() -> NonZeroU32`
- Produces: `TileBounds { row_range, dimension_range, physical_range }`
- Produces: crate-private `tile_bounds(len, list_size, geometry, row_tile, dimension_tile) -> VortexResult<TileBounds>`
- Produces: crate-private `physical_offset(len, list_size, geometry, row, dimension) -> VortexResult<usize>`

- [ ] **Step 1: Add the crate scaffold**

Add `"encodings/tiled-fsl"` to the workspace members and:

```toml
vortex-tiled-fsl = { version = "0.1.0", path = "./encodings/tiled-fsl", default-features = false }
```

Create the package manifest with only the dependencies used by this task:

```toml
[package]
name = "vortex-tiled-fsl"
authors = { workspace = true }
categories = { workspace = true }
description = "Two-dimensional tiled encoding for primitive Vortex fixed-size lists"
edition = { workspace = true }
homepage = { workspace = true }
include = { workspace = true }
keywords = { workspace = true }
license = { workspace = true }
readme = { workspace = true }
repository = { workspace = true }
rust-version = { workspace = true }
version = { workspace = true }

[lints]
workspace = true

[dependencies]
vortex-error = { workspace = true }
```

- [ ] **Step 2: Write failing geometry tests**

In `geometry.rs`, first add tests against the not-yet-defined API. The golden
case is the approved 3-row by 5-dimension example with geometry 2 by 3:

```rust
#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use vortex_error::VortexResult;

    use super::{TileGeometry, physical_offset, tile_bounds};

    fn geometry() -> TileGeometry {
        TileGeometry::new(
            NonZeroU32::new(2).unwrap(),
            NonZeroU32::new(3).unwrap(),
        )
    }

    #[test]
    fn physical_offsets_match_golden_layout() -> VortexResult<()> {
        let expected = [
            [0, 2, 4, 9, 11],
            [1, 3, 5, 10, 12],
            [6, 7, 8, 13, 14],
        ];
        for (row, offsets) in expected.into_iter().enumerate() {
            for (dimension, expected_offset) in offsets.into_iter().enumerate() {
                assert_eq!(
                    physical_offset(3, 5, geometry(), row, dimension)?,
                    expected_offset
                );
            }
        }
        Ok(())
    }

    #[test]
    fn tile_bounds_cover_unpadded_tails() -> VortexResult<()> {
        let bounds = [
            tile_bounds(3, 5, geometry(), 0, 0)?,
            tile_bounds(3, 5, geometry(), 1, 0)?,
            tile_bounds(3, 5, geometry(), 0, 1)?,
            tile_bounds(3, 5, geometry(), 1, 1)?,
        ];
        assert_eq!(bounds[0].physical_range, 0..6);
        assert_eq!(bounds[1].physical_range, 6..9);
        assert_eq!(bounds[2].physical_range, 9..13);
        assert_eq!(bounds[3].physical_range, 13..15);
        assert_eq!(bounds[3].row_range, 2..3);
        assert_eq!(bounds[3].dimension_range, 3..5);
        Ok(())
    }
}
```

- [ ] **Step 3: Run the focused test and confirm it fails**

Run:

```bash
cargo nextest run -p vortex-tiled-fsl geometry
```

Expected: compilation fails because `TileGeometry`, `physical_offset`, and
`tile_bounds` do not yet exist.

- [ ] **Step 4: Implement checked geometry and physical ranges**

Implement the domain types and the approved offset formula:

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TileGeometry {
    rows: NonZeroU32,
    dimensions: NonZeroU32,
}

impl TileGeometry {
    pub const fn new(rows: NonZeroU32, dimensions: NonZeroU32) -> Self {
        Self { rows, dimensions }
    }

    pub const fn rows(self) -> NonZeroU32 {
        self.rows
    }

    pub const fn dimensions(self) -> NonZeroU32 {
        self.dimensions
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TileBounds {
    pub row_range: Range<usize>,
    pub dimension_range: Range<usize>,
    pub physical_range: Range<usize>,
}
```

`tile_bounds` must:

1. compute `row_start = row_tile * rows` and
   `dimension_start = dimension_tile * dimensions` with `checked_mul`;
2. reject a tile index whose start is outside a nonempty logical extent;
3. allow no tile at all when either logical extent is zero;
4. clamp both range ends to their logical extents;
5. compute
   `physical_start = dimension_start * len + row_start * dimension_width`;
6. compute `physical_end = physical_start + row_height * dimension_width`;
7. return contextual `InvalidArgument` or overflow errors rather than panic.

`physical_offset` must reject out-of-bounds logical row/dimension indices and
implement:

```rust
dimension_start * len
    + row_start * dimension_width
    + dimension_within_tile * row_height
    + row_within_tile
```

Re-export `TileGeometry` and `TileBounds` from `lib.rs`.

- [ ] **Step 5: Format and run geometry tests**

Run:

```bash
cargo +nightly fmt --all
cargo nextest run -p vortex-tiled-fsl geometry
```

Expected: both golden tests pass, including the two unpadded tail tiles.

- [ ] **Step 6: Commit the geometry**

```bash
git add Cargo.toml Cargo.lock encodings/tiled-fsl
git commit -s -m "feat: add tiled fixed-size-list geometry"
```

---

### Task 2: Physical array and metadata

**Files:**
- Modify: `encodings/tiled-fsl/Cargo.toml`
- Modify: `encodings/tiled-fsl/src/lib.rs`
- Create: `encodings/tiled-fsl/src/array.rs`
- Create: `encodings/tiled-fsl/src/tests.rs`
- Create through the metadata golden test: `encodings/tiled-fsl/goldenfiles/tiled_fsl.metadata`

**Interfaces:**
- Consumes: `TileGeometry` from Task 1.
- Produces: `TiledFixedSizeList` and `TiledFixedSizeListArray = Array<TiledFixedSizeList>`.
- Produces: `TiledFixedSizeList::try_new(elements, list_size, validity, len, geometry)`.
- Produces: `TiledFixedSizeListArrayExt::{elements, geometry, list_size, array_validity, row_tile_count, dimension_tile_count}`.
- Produces: `initialize(&VortexSession)`.

- [ ] **Step 1: Add the array dependencies**

Add `prost`, `vortex-array`, `vortex-buffer`, `vortex-mask`, and
`vortex-session` as workspace dependencies. Add `_test-harness` to the
`vortex-array` dev-dependency and add `rstest`:

```toml
[dependencies]
prost = { workspace = true }
vortex-array = { workspace = true }
vortex-buffer = { workspace = true }
vortex-error = { workspace = true }
vortex-mask = { workspace = true }
vortex-session = { workspace = true }

[dev-dependencies]
rstest = { workspace = true }
vortex-array = { workspace = true, features = ["_test-harness"] }
```

- [ ] **Step 2: Write failing construction and metadata tests**

Create `src/tests.rs` with the shared session and constructor helper:

```rust
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

fn geometry(rows: u32, dimensions: u32) -> TileGeometry {
    TileGeometry::new(
        NonZeroU32::new(rows).unwrap(),
        NonZeroU32::new(dimensions).unwrap(),
    )
}

fn physical_fixture(
    rows: usize,
    dimensions: u32,
    geometry: TileGeometry,
) -> VortexResult<TiledFixedSizeListArray> {
    TiledFixedSizeList::try_new(
        PrimitiveArray::from_iter(
            (0..rows * dimensions as usize).map(|index| index as u16),
        ).into_array(),
        dimensions,
        Validity::NonNullable,
        rows,
        geometry,
    )
}

#[test]
fn try_new_derives_dtype_and_accessors() -> VortexResult<()> {
    let tiled = physical_fixture(3, 5, geometry(2, 3))?;
    assert_eq!(tiled.len(), 3);
    assert_eq!(tiled.list_size(), 5);
    assert_eq!(tiled.geometry(), geometry(2, 3));
    assert_eq!(tiled.row_tile_count(), 2);
    assert_eq!(tiled.dimension_tile_count(), 2);
    assert_eq!(
        tiled.dtype(),
        &DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U16, Nullability::NonNullable)),
            5,
            Nullability::NonNullable,
        )
    );
    Ok(())
}

#[rstest]
#[case(3, 5, 14)]
#[case(3, 5, 16)]
fn rejects_wrong_child_length(
    #[case] rows: usize,
    #[case] dimensions: u32,
    #[case] physical_len: usize,
) {
    let elements = PrimitiveArray::from_iter(
        (0..physical_len).map(|index| index as u16),
    ).into_array();
    assert!(
        TiledFixedSizeList::try_new(
            elements,
            dimensions,
            Validity::NonNullable,
            rows,
            geometry(2, 3),
        )
        .is_err()
    );
}

#[test]
fn tiled_fsl_metadata() {
    check_metadata(
        "tiled_fsl.metadata",
        &TiledFixedSizeListMetadata {
            tile_rows: 32,
            tile_dimensions: 64,
        }
        .encode_to_vec(),
    );
}
```

Add these invariant tests in the same module:

```rust
#[test]
fn rejects_non_primitive_child() {
    assert!(
        TiledFixedSizeList::try_new(
            VarBinViewArray::from_iter_str(["x"]).into_array(),
            1,
            Validity::NonNullable,
            1,
            geometry(1, 1),
        )
        .is_err()
    );
}

#[test]
fn rejects_wrong_outer_validity_length() {
    assert!(
        TiledFixedSizeList::try_new(
            PrimitiveArray::from_iter([1u8, 2, 3]).into_array(),
            1,
            Validity::from_iter([true, false]),
            3,
            geometry(2, 1),
        )
        .is_err()
    );
}

#[test]
fn rejects_len_times_list_size_overflow() {
    assert!(
        TiledFixedSizeList::try_new(
            PrimitiveArray::from_iter(std::iter::empty::<u8>()).into_array(),
            2,
            Validity::NonNullable,
            usize::MAX,
            geometry(1, 1),
        )
        .is_err()
    );
}

#[test]
fn rejects_zero_geometry_metadata() {
    let metadata = TiledFixedSizeListMetadata {
        tile_rows: 0,
        tile_dimensions: 64,
    };
    assert!(TileGeometry::try_from(&metadata).is_err());
}

#[test]
fn accepts_degenerate_and_oversized_geometry() -> VortexResult<()> {
    physical_fixture(0, 5, geometry(32, 64))?;
    physical_fixture(3, 0, geometry(32, 64))?;
    physical_fixture(3, 5, geometry(32, 64))?;
    Ok(())
}
```

- [ ] **Step 3: Run the tests and confirm the new API is missing**

Run:

```bash
cargo nextest run -p vortex-tiled-fsl
```

Expected: compilation fails because `TiledFixedSizeList` and the array
extension methods have not been implemented.

- [ ] **Step 4: Implement the physical array and metadata**

In `array.rs`, define:

```rust
pub type TiledFixedSizeListArray = Array<TiledFixedSizeList>;

#[derive(Clone, prost::Message)]
pub struct TiledFixedSizeListMetadata {
    #[prost(uint32, tag = "1")]
    pub tile_rows: u32,
    #[prost(uint32, tag = "2")]
    pub tile_dimensions: u32,
}

#[array_slots(TiledFixedSizeList)]
pub struct TiledFixedSizeListSlots {
    #[slot(0)]
    pub elements: ArrayRef,
    #[slot(1)]
    pub validity: Option<ArrayRef>,
}

#[derive(Clone, Debug)]
pub struct TiledFixedSizeListData {
    geometry: TileGeometry,
}

#[derive(Clone, Debug)]
pub struct TiledFixedSizeList;
```

Implement `ArrayHash`, `ArrayEq`, and `Display` using the geometry. Use encoding
ID `vortex.tiled_fsl`, zero top-level buffers, and
`vortex_array::vtable::with_empty_buffers`.

`try_new` derives its dtype rather than accepting redundant state:

```rust
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
        ArrayParts::new(TiledFixedSizeList, dtype, len, data)
            .with_slots(slots),
    )
}
```

Validation must reject each of these conditions explicitly:

```text
logical dtype is not FixedSizeList
FixedSizeList element dtype is not Primitive
physical child dtype differs from the logical element dtype
len.checked_mul(list_size) overflows
physical child length differs from len * list_size
outer validity length differs from len
slot count is not one element slot plus the optional validity slot
```

Serialization stores only the two geometry fields. Deserialization obtains
`list_size` and the element dtype from the logical dtype, requests one element
child of checked length `len * list_size`, accepts zero or one outer-validity
child, rejects all other child counts, and rejects zero geometry. Implement
`TryFrom<&TiledFixedSizeListMetadata> for TileGeometry` by applying
`NonZeroU32::new` to both fields and returning `InvalidArgument` if either is
zero. Set `type OperationsVTable = NotSupported` in this construction-only
checkpoint; Task 3 replaces it with the real operations vtable.

Implement `ValidityVTable<TiledFixedSizeList>` by converting the optional
validity slot with `child_to_validity`. Register the array from the crate's
single entrypoint:

```rust
pub fn initialize(session: &VortexSession) {
    session.arrays().register(TiledFixedSizeList);
}
```

- [ ] **Step 5: Materialize and verify the metadata golden**

Run:

```bash
cargo nextest run -p vortex-tiled-fsl tiled_fsl_metadata
xxd -g 1 encodings/tiled-fsl/goldenfiles/tiled_fsl.metadata
cargo nextest run -p vortex-tiled-fsl tiled_fsl_metadata
```

Expected: when no golden exists, the first run creates it and may report the
new file; its bytes are exactly `08 20 10 40`, encoding only row tile 32 and
dimension tile 64; the second run passes.

- [ ] **Step 6: Format and run the construction tests**

Run:

```bash
cargo +nightly fmt --all
cargo nextest run -p vortex-tiled-fsl
```

Expected: all construction, dtype, validity-length, overflow, degenerate, and
metadata tests pass.

- [ ] **Step 7: Commit the physical array**

```bash
git add Cargo.toml Cargo.lock encodings/tiled-fsl
git commit -s -m "feat: add tiled list array metadata"
```

---

### Task 3: Bulk transpose and canonical execution

**Files:**
- Modify: `encodings/tiled-fsl/src/array.rs`
- Modify: `encodings/tiled-fsl/src/lib.rs`
- Create: `encodings/tiled-fsl/src/transpose.rs`
- Create: `encodings/tiled-fsl/src/operations.rs`
- Modify: `encodings/tiled-fsl/src/tests.rs`

**Interfaces:**
- Consumes: `TileGeometry`, `tile_bounds`, and `physical_offset` from Task 1.
- Consumes: `TiledFixedSizeList::try_new` and physical accessors from Task 2.
- Produces: `TiledFixedSizeList::encode(ArrayView<FixedSizeList>, TileGeometry, &mut ExecutionCtx) -> VortexResult<TiledFixedSizeListArray>`.
- Produces: crate-private
  `encode_elements(ArrayView<Primitive>, usize, usize, TileGeometry, &mut ExecutionCtx) -> VortexResult<PrimitiveArray>`.
- Produces: crate-private
  `decode_elements(ArrayView<Primitive>, usize, usize, TileGeometry, &mut ExecutionCtx) -> VortexResult<PrimitiveArray>`.
- Produces: crate-private `TransposeDirection::{CanonicalToTiled, TiledToCanonical}` and
  `transpose_validity(Validity, usize, usize, TileGeometry, TransposeDirection, &mut ExecutionCtx) -> VortexResult<Validity>`.
- Produces: `OperationsVTable<TiledFixedSizeList>` with bounded scalar access
  and canonical execution through the array's execution vtable.

- [ ] **Step 1: Write failing physical-order and round-trip tests**

Add the canonical fixture and logical-equality helper:

```rust
fn fixture(
    rows: usize,
    dimensions: u32,
    geometry: TileGeometry,
) -> VortexResult<(FixedSizeListArray, TiledFixedSizeListArray, ExecutionCtx)> {
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::from_iter(
            (0..rows * dimensions as usize).map(|index| (index % 16) as u8),
        )
        .into_array(),
        dimensions,
        Validity::NonNullable,
        rows,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let tiled =
        TiledFixedSizeList::encode(canonical.as_view(), geometry, &mut ctx)?;
    Ok((canonical, tiled, ctx))
}

fn assert_fsl_equivalent(
    canonical: &ArrayRef,
    candidate: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    assert_eq!(candidate.dtype(), canonical.dtype());
    assert_eq!(candidate.len(), canonical.len());
    assert_arrays_eq!(canonical, candidate, ctx);
    Ok(())
}

#[test]
fn golden_physical_child_and_round_trip() -> VortexResult<()> {
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::from_iter([
            0u16, 1, 2, 3, 4,
            10, 11, 12, 13, 14,
            20, 21, 22, 23, 24,
        ]).into_array(),
        5,
        Validity::NonNullable,
        3,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let tiled =
        TiledFixedSizeList::encode(canonical.as_view(), geometry(2, 3), &mut ctx)?;
    let physical =
        tiled.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(
        physical.as_slice::<u16>(),
        &[0, 10, 1, 11, 2, 12, 20, 21, 22, 3, 13, 4, 14, 23, 24]
    );
    assert_fsl_equivalent(
        &canonical.into_array(),
        &tiled.into_array(),
        &mut ctx,
    )
}
```

Add this all-native-types test:

```rust
#[test]
fn all_native_ptypes_round_trip() -> VortexResult<()> {
    for ptype in [
        PType::U8, PType::U16, PType::U32, PType::U64,
        PType::I8, PType::I16, PType::I32, PType::I64,
        PType::F16, PType::F32, PType::F64,
    ] {
        match_each_native_ptype!(ptype, |T| {
            let canonical = FixedSizeListArray::new(
                PrimitiveArray::new(
                    Buffer::<T>::zeroed(15),
                    Validity::NonNullable,
                )
                .into_array(),
                5,
                Validity::NonNullable,
                3,
            );
            let mut ctx = SESSION.create_execution_ctx();
            let tiled = TiledFixedSizeList::encode(
                canonical.as_view(),
                geometry(2, 3),
                &mut ctx,
            )?;
            assert_fsl_equivalent(
                &canonical.into_array(),
                &tiled.into_array(),
                &mut ctx,
            )
        })?;
    }
    Ok(())
}
```

Add a four-case validity test with these exact pairs:

```rust
[
    (Validity::NonNullable, Validity::NonNullable),
    (Validity::AllValid, Validity::AllValid),
    (Validity::AllInvalid, Validity::AllInvalid),
    (
        Validity::from_iter([true, false, true, true, false, true]),
        Validity::from_iter([true, false, true]),
    ),
]
```

The first tuple member is element validity for a 3-by-2 child; the second is
outer validity for three rows. For every pair, assert both the encoded physical
element validity and the executed canonical result against the source.

Add a fixed-probe scalar test:

```rust
#[test]
fn scalar_at_matches_canonical_boundaries() -> VortexResult<()> {
    let (canonical, tiled, mut ctx) =
        fixture(65, 129, geometry(32, 64))?;
    for row in [0, 31, 32, 63, 64] {
        assert_eq!(
            canonical.execute_scalar(row, &mut ctx)?,
            tiled.execute_scalar(row, &mut ctx)?,
        );
    }
    Ok(())
}
```

- [ ] **Step 2: Run the execution tests and confirm they fail**

Run:

```bash
cargo nextest run -p vortex-tiled-fsl golden_physical_child_and_round_trip
cargo nextest run -p vortex-tiled-fsl scalar_at_matches_canonical_boundaries
```

Expected: compilation fails because `encode`, transpose helpers, and
`OperationsVTable<TiledFixedSizeList>` do not exist.

- [ ] **Step 3: Implement bulk transpose and inverse transpose**

In `transpose.rs`, keep dtype dispatch at the boundary:

```rust
pub(crate) fn encode_elements(
    elements: ArrayView<'_, Primitive>,
    len: usize,
    list_size: usize,
    geometry: TileGeometry,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    match_each_native_ptype!(elements.ptype(), |T| {
        let source = elements.as_slice::<T>();
        let mut output = BufferMut::<T>::with_capacity(source.len());
        for dimension_tile in 0..list_size.div_ceil(geometry.dimensions().get() as usize) {
            for row_tile in 0..len.div_ceil(geometry.rows().get() as usize) {
                let bounds = tile_bounds(len, list_size, geometry, row_tile, dimension_tile)?;
                for dimension in bounds.dimension_range.clone() {
                    for row in bounds.row_range.clone() {
                        output.push(source[row * list_size + dimension]);
                    }
                }
            }
        }
        let reordered_validity = transpose_validity(
            elements.validity()?,
            len,
            list_size,
            geometry,
            TransposeDirection::CanonicalToTiled,
            ctx,
        )?;
        Ok(PrimitiveArray::new(output.freeze(), reordered_validity))
    })
}
```

Define:

```rust
pub(crate) enum TransposeDirection {
    CanonicalToTiled,
    TiledToCanonical,
}

pub(crate) fn transpose_validity(
    validity: Validity,
    len: usize,
    list_size: usize,
    geometry: TileGeometry,
    direction: TransposeDirection,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Validity>
```

Return `NonNullable`, `AllValid`, and `AllInvalid` unchanged. Execute an array
validity to one mask, reorder it through the same coordinate mapping as the
values, and construct one mixed output validity. For a mixed mask, allocate one
unset `BitBufferMut`; obtain its `MaskValues`, call `bit_buffer().for_each_set_index`
with a closure that maps each set source index to its destination index, and set
that output bit. Never call `Validity::is_valid`, `Mask::value`, or
`execute_scalar` in an element-count-dependent loop.

The inverse function allocates `BufferMut::<T>::zeroed(len * list_size)` and
places every physical value at `row * list_size + dimension`. Reorder mixed
element validity through the identical mapping. Do not call `execute_scalar`
inside either element loop.

- [ ] **Step 4: Implement encoding, canonical execution, and scalar access**

`TiledFixedSizeList::encode` executes the canonical FSL element child exactly
once to `PrimitiveArray`, transposes it, and preserves the outer FSL validity.
Use this signature:

```rust
pub fn encode(
    array: ArrayView<'_, FixedSizeList>,
    geometry: TileGeometry,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TiledFixedSizeListArray>
```

Change `type OperationsVTable` from `NotSupported` to `Self`. The VTable
`execute` uses `require_child!` to request a canonical primitive element child,
inverse-transposes it, and returns:

```rust
Ok(ExecutionResult::done(
    FixedSizeListArray::new(
        decoded_elements.into_array(),
        array.list_size(),
        array.array_validity(),
        array.len(),
    )
    .into_array(),
))
```

Implement `OperationsVTable::scalar_at` by building the `list_size` physical
indices for one selected row, performing one bulk child `take`, executing that
result once to `PrimitiveArray`, restoring row-major dimension order, and
returning the single fixed-size-list scalar. It must not execute the entire
tiled source and must not call `execute_scalar` inside a dimension loop.

- [ ] **Step 5: Format and run all execution tests**

Run:

```bash
cargo +nightly fmt --all
cargo nextest run -p vortex-tiled-fsl golden_physical_child_and_round_trip
cargo nextest run -p vortex-tiled-fsl scalar_at_matches_canonical_boundaries
cargo nextest run -p vortex-tiled-fsl
```

Expected: physical bytes match the 3-by-5 golden, every native primitive and
validity combination round-trips, degenerate arrays round-trip, and scalar
access matches canonical FSL.

- [ ] **Step 6: Commit bulk execution**

```bash
git add encodings/tiled-fsl
git commit -s -m "feat: transpose tiled fixed-size lists"
```

---

### Task 4: Range-only tile traversal

**Files:**
- Modify: `encodings/tiled-fsl/src/geometry.rs`
- Modify: `encodings/tiled-fsl/src/array.rs`
- Modify: `encodings/tiled-fsl/src/lib.rs`
- Modify: `encodings/tiled-fsl/src/tests.rs`

**Interfaces:**
- Consumes: `TileGeometry`, `TileBounds`, and checked `tile_bounds` from Task 1.
- Consumes: array accessors from Task 2.
- Produces: `TileBoundsIter: Iterator<Item = TileBounds>`.
- Produces:
  `TiledFixedSizeListArrayExt::{tile, tiles, tile_elements}`.
- Guarantees: constructing or advancing `TileBoundsIter` performs no
  `ArrayRef` clone, child slice, heap allocation, or dynamic dispatch.

- [ ] **Step 1: Write failing traversal tests**

Add these exact checks:

```rust
#[test]
fn tiles_are_range_only_and_in_physical_order() -> VortexResult<()> {
    let (_, tiled, _) = fixture(3, 5, geometry(2, 3))?;
    let bounds: Vec<TileBounds> = tiled.tiles().collect();
    assert_eq!(
        bounds
            .iter()
            .map(|bounds| bounds.physical_range.clone())
            .collect::<Vec<_>>(),
        vec![0..6, 6..9, 9..13, 13..15],
    );
    assert_eq!(tiled.tile(1, 1)?, bounds[3]);
    assert_eq!(tiled.tile_elements(&bounds[3])?.len(), 2);
    assert!(tiled.tile(2, 0).is_err());
    assert!(tiled.tile(0, 2).is_err());
    Ok(())
}

#[test]
fn degenerate_arrays_have_no_tiles() -> VortexResult<()> {
    assert_eq!(physical_fixture(0, 5, geometry(2, 3))?.tiles().count(), 0);
    assert_eq!(physical_fixture(3, 0, geometry(2, 3))?.tiles().count(), 0);
    Ok(())
}
```

The explicit `Vec<TileBounds>` annotation fixes the iterator item contract at
compile time.

- [ ] **Step 2: Run the traversal tests and confirm the API is absent**

Run:

```bash
cargo nextest run -p vortex-tiled-fsl tiles_are_range_only_and_in_physical_order
cargo nextest run -p vortex-tiled-fsl degenerate_arrays_have_no_tiles
```

Expected: compilation fails because `TileBoundsIter`, `tile`, `tiles`, and
`tile_elements` do not exist.

- [ ] **Step 3: Implement range-only tile traversal**

Reuse `TileBounds` as the only item yielded by `tile` and `tiles`. A tile must
not own, clone, or borrow-wrap the physical child. Define one concrete iterator
whose state is exclusively scalar geometry and counters:

```rust
pub struct TileBoundsIter {
    len: usize,
    list_size: usize,
    geometry: TileGeometry,
    row_tile_count: usize,
    dimension_tile_count: usize,
    next_row_tile: usize,
    next_dimension_tile: usize,
}

impl Iterator for TileBoundsIter {
    type Item = TileBounds;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_tile_count == 0
            || self.next_dimension_tile == self.dimension_tile_count
        {
            return None;
        }

        let bounds = tile_bounds_for_validated_array(
            self.len,
            self.list_size,
            self.geometry,
            self.next_row_tile,
            self.next_dimension_tile,
        );
        self.next_row_tile += 1;
        if self.next_row_tile == self.row_tile_count {
            self.next_row_tile = 0;
            self.next_dimension_tile += 1;
        }
        Some(bounds)
    }
}

pub trait TiledFixedSizeListArrayExt: TiledFixedSizeListArraySlotsExt {
    fn tile(
        &self,
        row_tile: usize,
        dimension_tile: usize,
    ) -> VortexResult<TileBounds>;
    fn tiles(&self) -> TileBoundsIter;
    fn tile_elements(&self, bounds: &TileBounds) -> VortexResult<ArrayRef> {
        self.elements().slice(bounds.physical_range.clone())
    }
}

impl<T: TiledFixedSizeListArraySlotsExt> TiledFixedSizeListArrayExt for T {}
```

Implement the hand-written extension trait on top of the generated
`TiledFixedSizeListArraySlotsExt` trait. `tiles()` initializes
`TileBoundsIter` from the array's validated length, list size, and geometry.
The iterator advances row tiles inside dimension tiles, so bounds arrive in
physical order. A zero row-tile count or zero dimension-tile count yields no
items. `tile_bounds_for_validated_array` is private and infallible: it uses only
tile indices generated within the precomputed counts, with `debug_assert!`
checks documenting that precondition. Factor the range calculation once so this
helper and checked `tile_bounds` cannot drift. The public `tile(row_tile,
dimension_tile)` uses the checked entrypoint and returns an error for arbitrary
invalid indices.

Neither constructing the iterator nor yielding an item performs an `ArrayRef`
clone, child slice, heap allocation, or dynamic dispatch. `tile_elements` is an
explicit cold-path convenience for tests and inspection; scoring kernels
execute `elements()` once and index it with `physical_range`.

- [ ] **Step 4: Format and run traversal plus crate tests**

Run:

```bash
cargo +nightly fmt --all
cargo nextest run -p vortex-tiled-fsl tiles_are_range_only_and_in_physical_order
cargo nextest run -p vortex-tiled-fsl degenerate_arrays_have_no_tiles
cargo nextest run -p vortex-tiled-fsl
```

Expected: traversal tests and the complete crate suite pass.

- [ ] **Step 5: Commit range traversal**

```bash
git add encodings/tiled-fsl
git commit -s -m "feat: expose tiled list ranges"
```

---

### Task 5: Encoding-preserving slice

**Files:**
- Modify: `encodings/tiled-fsl/src/array.rs`
- Modify: `encodings/tiled-fsl/src/lib.rs`
- Create: `encodings/tiled-fsl/src/gather.rs`
- Create: `encodings/tiled-fsl/src/rules.rs`
- Create: `encodings/tiled-fsl/src/slice.rs`
- Modify: `encodings/tiled-fsl/src/tests.rs`

**Interfaces:**
- Consumes: geometry and constructors from Tasks 1-2 and execution from Task 3.
- Produces: crate-private `physical_indices_for_rows(source_len, list_size, geometry, rows) -> VortexResult<Buffer<u64>>`.
- Produces: crate-private `gather_tiled_rows(array, rows, validity) -> VortexResult<TiledFixedSizeListArray>`.
- Produces: `SliceReduce for TiledFixedSizeList`.
- Preserves: `TiledFixedSizeList` parent and exact `TileGeometry` for nonempty
  output ranges and for the full `0..0` range of an already-empty tiled source;
  Vortex's ordinary canonical empty result for an empty slice of a nonempty
  source.

- [ ] **Step 1: Add failing slice conformance and boundary tests**

Add tests for:

```rust
#[rstest]
#[case(0..0)]
#[case(0..1)]
#[case(0..31)]
#[case(0..32)]
#[case(0..33)]
#[case(1..64)]
#[case(31..65)]
#[case(64..65)]
fn slice_preserves_encoding_and_values(#[case] range: Range<usize>) -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(65, 129, geometry(32, 64))?;
    let expected = canonical.into_array().slice(range.clone())?;
    let actual = tiled.into_array().slice(range)?;
    if actual.is_empty() {
        assert!(actual.is::<FixedSizeList>());
    } else {
        assert!(actual.is::<TiledFixedSizeList>());
        assert_eq!(
            actual.as_::<TiledFixedSizeList>().geometry(),
            geometry(32, 64)
        );
    }
    assert_arrays_eq!(expected, actual, &mut ctx);
    Ok(())
}
```

Add one special-case test with these exact fixtures:

```rust
#[test]
fn slice_preserves_special_cases() -> VortexResult<()> {
    let cases = [
        (3, 0, geometry(2, 3), 1..3),
        (3, 5, geometry(32, 64), 1..3),
        (4_096, 128, geometry(32, 64), 2_048..2_049),
    ];
    for (rows, dimensions, tile_geometry, range) in cases {
        let (canonical, tiled, mut ctx) =
            fixture(rows, dimensions, tile_geometry)?;
        let expected = canonical.into_array().slice(range.clone())?;
        let actual = tiled.into_array().slice(range.clone())?;
        assert_fsl_equivalent(&expected, &actual, &mut ctx)?;
        if !actual.is_empty() {
            let actual = actual.as_::<TiledFixedSizeList>();
            assert_eq!(actual.geometry(), tile_geometry);
            assert_eq!(
                actual.elements().len(),
                range.len() * dimensions as usize,
            );
        }
    }
    Ok(())
}
```

Add the mixed-validity case explicitly:

```rust
#[test]
fn slice_preserves_mixed_validity() -> VortexResult<()> {
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::new(
            buffer![0i32, 1, 10, 11, 20, 21],
            Validity::from_iter([true, false, true, true, false, true]),
        )
        .into_array(),
        2,
        Validity::from_iter([true, false, true]),
        3,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let tiled =
        TiledFixedSizeList::encode(canonical.as_view(), geometry(2, 2), &mut ctx)?;
    let expected = canonical.into_array().slice(1..3)?;
    let actual = tiled.into_array().slice(1..3)?;
    assert_eq!(
        actual.as_::<TiledFixedSizeList>().geometry(),
        geometry(2, 2),
    );
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}
```

Encoded-child composition is tested separately in Task 7, after the FastLanes
dev-dependency exists.

- [ ] **Step 2: Run the focused slice tests and confirm fallback loses tiling**

Run:

```bash
cargo nextest run -p vortex-tiled-fsl slice
```

Expected: nonempty value fallbacks may work, but their encoding assertion fails
because no `SliceReduce` rule is registered. For the nonempty test source, the
`0..0` case already returns Vortex's canonical empty FSL and passes its
degenerate-result assertion.

- [ ] **Step 3: Implement output-proportional row gathering**

In `gather.rs`, implement:

```rust
pub(crate) fn physical_indices_for_rows(
    source_len: usize,
    list_size: usize,
    geometry: TileGeometry,
    rows: &[Option<usize>],
) -> VortexResult<Buffer<u64>>
```

The output index order must itself be tiled using the same geometry but
`rows.len()` as the output row count. For each output logical coordinate,
translate the selected source row plus dimension through `physical_offset`.
Reject any valid selected row outside `source_len`; use source row zero only as
the placeholder for a null selected row when the source is nonempty. Assert the
internal precondition that `source_len != 0 || rows.is_empty()`: Vortex's
generic take adaptor handles empty sources before the tiled take kernel runs,
and a slice of an empty source cannot be nonempty. Do not add an unreachable
default-elements path.

Define the shared row-operation boundary explicitly:

```rust
pub(crate) fn gather_tiled_rows(
    array: ArrayView<'_, TiledFixedSizeList>,
    rows: &[Option<usize>],
    validity: Validity,
) -> VortexResult<TiledFixedSizeListArray>
```

It builds the output-ordered physical indices, takes the element child once,
and reconstructs with `rows.len()` and the source geometry. The caller supplies
the already-derived outer validity so slice and take share physical gathering
without conflating their validity semantics.

- [ ] **Step 4: Implement and register `SliceReduce`**

`SliceReduce::slice` must:

1. create the selected logical rows from the requested range;
2. build only `range.len() * list_size` physical indices;
3. slice the outer validity with `array.array_validity().slice(range.clone())`;
4. call `gather_tiled_rows` with the selected rows and sliced validity;
5. return the resulting tiled array with unchanged geometry.

`ArrayRef::slice` returns a full-range slice before applying its canonical-empty
short-circuit. Thus `0..0` retains an already-empty tiled source, while an empty
slice of a nonempty source is canonical. This rule is invoked only for nonempty
output. Do not modify those generic short-circuits.

Register:

```rust
pub(crate) static RULES: ParentRuleSet<TiledFixedSizeList> =
    ParentRuleSet::new(&[
        ParentRuleSet::lift(&SliceReduceAdaptor(TiledFixedSizeList)),
    ]);
```

Have `VTable::reduce_parent` evaluate `RULES`. Do not add cast or compressor
rules in this task.

- [ ] **Step 5: Format and verify slice behavior**

Run:

```bash
cargo +nightly fmt --all
cargo nextest run -p vortex-tiled-fsl slice
cargo nextest run -p vortex-tiled-fsl
```

Expected: every nonempty slice and the full `0..0` slice of an already-empty
tiled source preserve the tiled parent; empty slices of nonempty sources are
canonical, and all cases match canonical FSL logically.

- [ ] **Step 6: Commit slice support**

```bash
git add encodings/tiled-fsl
git commit -s -m "feat: preserve tiled lists through slice"
```

---

### Task 6: Encoding-preserving take

**Files:**
- Modify: `encodings/tiled-fsl/src/lib.rs`
- Create: `encodings/tiled-fsl/src/kernel.rs`
- Create: `encodings/tiled-fsl/src/take.rs`
- Modify: `encodings/tiled-fsl/src/tests.rs`

**Interfaces:**
- Consumes: `physical_indices_for_rows` and `gather_tiled_rows` from Task 5.
- Produces: `TakeExecute for TiledFixedSizeList`.
- Produces: crate-private `collect_checked_rows<I: IntegerPType>(indices, mask, source_len) -> VortexResult<Vec<Option<usize>>>`.
- Produces: `kernel::initialize(&VortexSession)` registering `TakeExecuteAdaptor`.
- Preserves after execution: tiled parent and geometry for nonempty takes from a
  nonempty source, plus requested row order, duplicates, and nullable-index
  semantics. Empty indices and empty sources retain Vortex's generic
  canonical/constant short-circuits.

- [ ] **Step 1: Add failing take conformance and explicit preservation tests**

Use Vortex's conformance harness:

```rust
#[test]
fn take_conformance() {
    let (_, tiled, mut ctx) = fixture(65, 129, geometry(32, 64)).unwrap();
    test_take_conformance(&tiled.into_array(), &mut ctx);
}
```

Add an explicit nullable, duplicated, unsorted case:

```rust
let indices = PrimitiveArray::new(
    buffer![64u32, 1, 1, 32, 0],
    Validity::from_iter([true, true, false, true, true]),
).into_array();
let actual = tiled
    .into_array()
    .take(indices.clone())?
    .execute_until::<TiledFixedSizeList>(&mut ctx)?;
assert!(actual.is::<TiledFixedSizeList>());
assert_eq!(actual.as_::<TiledFixedSizeList>().geometry(), geometry(32, 64));
assert_arrays_eq!(canonical.into_array().take(indices)?, actual, &mut ctx);
```

Define the helper:

```rust
fn assert_take_indices(indices: ArrayRef) -> VortexResult<()> {
    let index_count = indices.len();
    let (canonical, tiled, mut ctx) = fixture(3, 5, geometry(2, 3))?;
    let expected = canonical.into_array().take(indices.clone())?;
    let actual = tiled
        .into_array()
        .take(indices)?
        .execute_until::<TiledFixedSizeList>(&mut ctx)?;
    assert_eq!(
        actual.as_::<TiledFixedSizeList>().elements().len(),
        index_count * 5,
    );
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}
```

Call it with these exact arrays so every integer index PType reaches the
dispatch:

```rust
assert_take_indices(PrimitiveArray::from_iter([2u8, 0, 1]).into_array())?;
assert_take_indices(PrimitiveArray::from_iter([2u16, 0, 1]).into_array())?;
assert_take_indices(PrimitiveArray::from_iter([2u32, 0, 1]).into_array())?;
assert_take_indices(PrimitiveArray::from_iter([2u64, 0, 1]).into_array())?;
assert_take_indices(PrimitiveArray::from_iter([2i8, 0, 1]).into_array())?;
assert_take_indices(PrimitiveArray::from_iter([2i16, 0, 1]).into_array())?;
assert_take_indices(PrimitiveArray::from_iter([2i32, 0, 1]).into_array())?;
assert_take_indices(PrimitiveArray::from_iter([2i64, 0, 1]).into_array())?;
```

Add these regression tests:

```rust
#[test]
fn take_all_null_indices_from_empty_source() -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(0, 5, geometry(2, 3))?;
    let indices =
        PrimitiveArray::from_option_iter([None::<u32>, None]).into_array();
    let expected = canonical.into_array().take(indices.clone())?;
    let actual = tiled.into_array().take(indices)?;
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}

#[test]
fn take_empty_indices_is_canonical_empty() -> VortexResult<()> {
    let (_, tiled, mut ctx) = fixture(3, 5, geometry(2, 3))?;
    let indices =
        PrimitiveArray::from_iter::<[u32; 0]>([]).into_array();
    let actual = tiled
        .into_array()
        .take(indices)?
        .execute_until::<FixedSizeList>(&mut ctx)?;
    assert!(actual.is::<FixedSizeList>());
    assert!(actual.is_empty());
    Ok(())
}

#[test]
fn take_zero_width_lists() -> VortexResult<()> {
    let (canonical, tiled, mut ctx) = fixture(3, 0, geometry(2, 3))?;
    let indices = PrimitiveArray::from_iter([2u32, 0]).into_array();
    let expected = canonical.into_array().take(indices.clone())?;
    let actual = tiled
        .into_array()
        .take(indices)?
        .execute_until::<TiledFixedSizeList>(&mut ctx)?;
    assert_eq!(actual.as_::<TiledFixedSizeList>().elements().len(), 0);
    assert_fsl_equivalent(&expected, &actual, &mut ctx)
}

#[test]
fn take_rejects_out_of_bounds_valid_index() -> VortexResult<()> {
    let (_, tiled, mut ctx) = fixture(3, 5, geometry(2, 3))?;
    let indices = PrimitiveArray::from_iter([3u32]).into_array();
    assert!(
        tiled
            .into_array()
            .take(indices)?
            .execute::<Canonical>(&mut ctx)
            .is_err()
    );
    Ok(())
}

#[test]
fn nullable_take_makes_outer_dtype_nullable() -> VortexResult<()> {
    let (_, tiled, mut ctx) = fixture(3, 5, geometry(2, 3))?;
    let indices =
        PrimitiveArray::from_option_iter([Some(2u32), None]).into_array();
    let actual = tiled
        .into_array()
        .take(indices)?
        .execute_until::<TiledFixedSizeList>(&mut ctx)?;
    assert!(actual.dtype().is_nullable());
    assert_eq!(actual.as_::<TiledFixedSizeList>().elements().len(), 10);
    Ok(())
}
```

- [ ] **Step 2: Run take tests and confirm the tiled assertion fails**

Run:

```bash
cargo nextest run -p vortex-tiled-fsl take
```

Expected: `ArrayRef::take` constructs a lazy `DictArray`. Executing it does not
reach a tiled parent because no encoding-specific take kernel is registered, so
the explicit `execute_until::<TiledFixedSizeList>` preservation assertion fails.

- [ ] **Step 3: Implement `TakeExecute` with one canonicalization of indices**

The implementation must:

1. reject non-integer index dtypes;
2. execute the index array once to `PrimitiveArray`;
3. execute its validity once to a mask;
4. convert valid indices to checked `usize` and null indices to `None`;
5. call `physical_indices_for_rows`;
6. take the physical element child once;
7. call `array.array_validity().take(indices)`;
8. construct a tiled output with the unchanged geometry.

Define index conversion once:

```rust
fn collect_checked_rows<I: IntegerPType>(
    indices: &[I],
    mask: &Mask,
    source_len: usize,
) -> VortexResult<Vec<Option<usize>>>
```

Zip values with the mask. Push `None` for a null index. Convert each valid
integer with checked `usize` conversion, reject negative or unrepresentable
values, bounds-check against `source_len`, then push `Some(row)`.

Keep PType dispatch at the edge:

```rust
impl TakeExecute for TiledFixedSizeList {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let indices_ref = indices.clone();
        let indices_array = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let mask = indices_array
            .validity()?
            .execute_mask(indices_array.len(), ctx)?;
        let rows = match_each_integer_ptype!(indices_array.ptype(), |I| {
            collect_checked_rows::<I>(indices_array.as_slice::<I>(), &mask, array.len())
        })?;
        let validity = array.array_validity().take(&indices_ref)?;
        Ok(Some(
            gather_tiled_rows(array, &rows, validity)?.into_array(),
        ))
    }
}
```

For an empty source, Vortex's generic `TakeExecuteAdaptor` handles zero output
rows and all-null index rows before calling this implementation. Likewise, it
handles empty indices against a nonempty source. Those results are canonical or
constant and are explicitly exempt from tiled preservation. Do not replace the
generic adaptor solely to retain a meaningless layout on degenerate output.

- [ ] **Step 4: Register the take parent kernel**

In `kernel.rs`:

```rust
pub(crate) fn initialize(session: &VortexSession) {
    session.kernels().register_execute_parent_kernel(
        Dict.id(),
        TiledFixedSizeList,
        TakeExecuteAdaptor(TiledFixedSizeList),
    );
}
```

Call `kernel::initialize(session)` from the crate's public `initialize`.

- [ ] **Step 5: Format and verify take plus the full crate**

Run:

```bash
cargo +nightly fmt --all
cargo nextest run -p vortex-tiled-fsl take
cargo nextest run -p vortex-tiled-fsl
```

Expected: conformance passes; executing each nondegenerate take produces a tiled
parent with unchanged geometry, while degenerate takes match canonical FSL
without a tiled assertion.

- [ ] **Step 6: Commit take support**

```bash
git add encodings/tiled-fsl
git commit -s -m "feat: preserve tiled lists through take"
```

---

### Task 7: FastLanes child composition

**Files:**
- Modify: `encodings/tiled-fsl/Cargo.toml`
- Modify: `encodings/tiled-fsl/src/tests.rs`

**Interfaces:**
- Consumes: `TiledFixedSizeList::{encode, try_new}` and slice/take from Tasks 2-6.
- Consumes only in tests: `vortex_fastlanes::bitpack_encode`.
- Proves: the tiled parent does not require a canonical primitive physical child.

- [ ] **Step 1: Add the failing bitpacked-child composition test**

Before adding the dev-dependency, write a test that references
`vortex_fastlanes` and exercises a 4-bit `u8` physical child:

```rust
#[test]
fn bitpacked_child_roundtrips_and_remains_tiled_after_row_ops() -> VortexResult<()> {
    let (canonical, raw_tiled, mut ctx) = u8_fixture(65, 128, geometry(32, 64))?;
    vortex_fastlanes::initialize(ctx.session());
    let physical = raw_tiled
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let bitpacked = bitpack_encode(&physical, 4, None, &mut ctx)?.into_array();
    let tiled = TiledFixedSizeList::try_new(
        bitpacked,
        128,
        raw_tiled.array_validity(),
        65,
        geometry(32, 64),
    )?;

    assert_arrays_eq!(canonical, tiled.clone(), &mut ctx);

    let sliced = tiled.clone().into_array().slice(1..64)?;
    assert!(sliced.is::<TiledFixedSizeList>());

    let taken = tiled
        .into_array()
        .take(PrimitiveArray::from_iter([64u32, 0, 32]).into_array())?
        .execute_until::<TiledFixedSizeList>(&mut ctx)?;
    assert!(taken.is::<TiledFixedSizeList>());
    Ok(())
}
```

- [ ] **Step 2: Run the test and confirm the missing dev dependency**

Run:

```bash
cargo nextest run -p vortex-tiled-fsl bitpacked_child
```

Expected: compilation fails because `vortex-fastlanes` is not a dependency of
the test target.

- [ ] **Step 3: Add the dev dependency and verify generic-child behavior**

Add only:

```toml
[dev-dependencies]
vortex-fastlanes = { workspace = true }
```

Do not add it to `[dependencies]`. Keep `try_new` validation limited to logical
dtype and length: only `encode` and canonical execution require a canonical
`PrimitiveArray`; `slice` and `take` call generic child operations and accept
whatever same-dtype encoding those operations return.

- [ ] **Step 4: Format and run composition plus full tests**

Run:

```bash
cargo +nightly fmt --all
cargo nextest run -p vortex-tiled-fsl bitpacked_child
cargo nextest run -p vortex-tiled-fsl
```

Expected: raw and bitpacked children both round-trip; parent slice/take remain
tiled.

- [ ] **Step 5: Commit the composition coverage**

```bash
git add encodings/tiled-fsl/Cargo.toml encodings/tiled-fsl/src/tests.rs Cargo.lock
git commit -s -m "test: cover bitpacked tiled list children"
```

---

### Task 8: Unstable file registration and facade re-export

**Files:**
- Modify: `vortex-file/Cargo.toml:35-95`
- Modify: `vortex-file/src/lib.rs:155-205`
- Create: `vortex-file/tests/tiled_fsl.rs`
- Modify: `vortex/Cargo.toml:20-105`
- Modify: `vortex/src/lib.rs:235-285`
- Create: `vortex/src/editions/unstable/v2026_07.rs`
- Modify: `vortex/src/editions/unstable/mod.rs`
- Modify: `vortex/src/editions/mod.rs`
- Modify: `vortex/src/editions/tests.rs`

**Interfaces:**
- Consumes: `vortex_tiled_fsl::initialize`.
- Produces: decoder registration under `vortex-file/unstable_encodings`.
- Produces: `vortex::encodings::tiled_fsl` under `vortex/unstable_encodings`.
- Produces: writer permission in the default unstable edition without changing a
  frozen `core` edition.
- Proves: explicit experimental writer-edition file round-trip retains the
  tiled parent for both raw and FastLanes-bitpacked children, retains the
  bitpacked child encoding, and preserves logical values.
- Proves: the ordinary default unstable session permits tiled FSL output.

- [ ] **Step 1: Add a failing feature-gated file round-trip test**

Create an integration test beginning with:

```rust
#![cfg(feature = "unstable_encodings")]
#![expect(clippy::tests_outside_test_module)]

mod common;

use common::enable_all_registered_array_encodings;
```

Build one raw tiled FSL and a second tiled FSL reconstructed around a four-bit
FastLanes child. Store them as `raw` and `bitpacked` fields in one
`StructArray`. Create a session with `LayoutSession` and `RuntimeSession`, call
`vortex_file::register_default_encodings`, then import and call
`common::enable_all_registered_array_encodings` from
`vortex-file/tests/common/mod.rs` so the writer edition explicitly permits
every registered ID. Write with `FlatLayoutStrategy` to avoid automatic
compression policy.

After scanning one chunk:

```rust
let result = chunk.execute::<StructArray>(&mut ctx)?;
let raw = result.unmasked_field(0).clone();
let bitpacked = result.unmasked_field(1).clone();

assert!(raw.is::<TiledFixedSizeList>());
assert!(bitpacked.is::<TiledFixedSizeList>());
assert_eq!(
    raw.as_::<TiledFixedSizeList>().geometry(),
    geometry(32, 64)
);
assert_eq!(
    bitpacked.as_::<TiledFixedSizeList>().geometry(),
    geometry(32, 64)
);
assert!(
    bitpacked
        .as_::<TiledFixedSizeList>()
        .elements()
        .is::<vortex_fastlanes::BitPacked>()
);
assert_arrays_eq!(input, result, &mut ctx);
```

- [ ] **Step 2: Run the test and confirm registration is missing**

Run:

```bash
cargo nextest run -p vortex-file --features unstable_encodings --test tiled_fsl
```

Expected: compilation fails because `vortex-file` does not yet depend on or
register `vortex-tiled-fsl`.

- [ ] **Step 3: Wire the optional file dependency and decoder registration**

In `vortex-file/Cargo.toml`, add:

```toml
vortex-tiled-fsl = { workspace = true, optional = true }
```

and include `"dep:vortex-tiled-fsl"` in `unstable_encodings`.

In `register_default_encodings`:

```rust
#[cfg(feature = "unstable_encodings")]
vortex_tiled_fsl::initialize(session);
```

Keep the private permissive test edition: it remains useful for exhaustive
nested round-trips. Also declare tiled FSL in the next unstable edition and add
an ordinary-default-session test to prove the normal writer policy. Do not add
tiled FSL to a frozen `core` edition.

- [ ] **Step 4: Wire the optional facade dependency and re-export**

In `vortex/Cargo.toml`, add the optional dependency and include it in
`unstable_encodings`. In `vortex/src/lib.rs`:

```rust
#[cfg(feature = "unstable_encodings")]
/// Experimental two-dimensional tiled fixed-size-list encoding.
pub mod tiled_fsl {
    pub use vortex_tiled_fsl::*;
}
```

Place it under the existing `encodings` module.

- [ ] **Step 5: Verify feature-off and feature-on builds**

Run:

```bash
cargo +nightly fmt --all
cargo nextest run -p vortex-file --features unstable_encodings --test tiled_fsl
cargo check -p vortex-file --no-default-features
cargo check -p vortex --features unstable_encodings
```

Expected: the file test retains both tiled parents, retains `BitPacked` beneath
the composed parent, and preserves values; feature-off builds do not compile or
register the new crate.

- [ ] **Step 6: Commit unstable integration**

```bash
git add Cargo.toml Cargo.lock vortex-file vortex
git commit -s -m "feat: register tiled lists as unstable encoding"
```

---

### Task 9: Canonical-FSL unit conformance suite

**Files:**
- Modify: `encodings/tiled-fsl/src/tests.rs`

**Interfaces:**
- Consumes: complete array, scalar, slice, take, tile, and FastLanes-composition APIs from Tasks 1-8.
- Consumes: `test_array_consistency`, `test_filter_conformance`, and
  `test_take_conformance` from Vortex's `_test-harness`.
- Uses: canonical `FixedSizeListArray` as the oracle; it never computes expected logical values through tiled helpers.
- Produces: deterministic regression coverage shared conceptually with the fuzz oracle in Task 10.

- [ ] **Step 1: Add the standard Vortex conformance checks**

Add a typed constructor so every conformance fixture is explicit:

```rust
fn encode_fixture<T: NativePType>(
    values: Buffer<T>,
    element_validity: Validity,
    list_size: u32,
    outer_validity: Validity,
    len: usize,
    geometry: TileGeometry,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let canonical = FixedSizeListArray::new(
        PrimitiveArray::new(values, element_validity).into_array(),
        list_size,
        outer_validity,
        len,
    );
    Ok(TiledFixedSizeList::encode(canonical.as_view(), geometry, ctx)?
        .into_array())
}
```

Build exactly these arrays:

```rust
let fixtures = vec![
    encode_fixture(
        buffer![0u8, 1, 2, 3, 4, 5],
        Validity::NonNullable,
        2,
        Validity::NonNullable,
        3,
        geometry(2, 2),
        &mut ctx,
    )?,
    encode_fixture(
        buffer![0i32, 1, 2, 3, 4, 5],
        Validity::NonNullable,
        2,
        Validity::from_iter([true, false, true]),
        3,
        geometry(2, 2),
        &mut ctx,
    )?,
    encode_fixture(
        buffer![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
        Validity::from_iter([true, false, true, true, false, true]),
        2,
        Validity::NonNullable,
        3,
        geometry(2, 2),
        &mut ctx,
    )?,
    encode_fixture(
        buffer![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0],
        Validity::from_iter([true, false, true, true, false, true]),
        2,
        Validity::from_iter([true, false, true]),
        3,
        geometry(2, 2),
        &mut ctx,
    )?,
    encode_fixture(
        buffer![0u8; 0],
        Validity::NonNullable,
        5,
        Validity::NonNullable,
        0,
        geometry(32, 64),
        &mut ctx,
    )?,
    encode_fixture(
        buffer![0u8; 0],
        Validity::NonNullable,
        0,
        Validity::NonNullable,
        65,
        geometry(32, 64),
        &mut ctx,
    )?,
    encode_fixture(
        buffer![7u8; 65 * 129],
        Validity::NonNullable,
        129,
        Validity::NonNullable,
        65,
        geometry(32, 64),
        &mut ctx,
    )?,
    encode_fixture(
        buffer![7u8; 3 * 5],
        Validity::NonNullable,
        5,
        Validity::NonNullable,
        3,
        geometry(32, 64),
        &mut ctx,
    )?,
];

for tiled in fixtures {
    test_array_consistency(&tiled, &mut ctx);
    test_filter_conformance(&tiled, &mut ctx);
    test_take_conformance(&tiled, &mut ctx);
}
```

Add the bitpacked parent explicitly:

```rust
let (_, raw_tiled, _) = fixture(65, 128, geometry(32, 64))?;
let physical = raw_tiled
    .elements()
    .clone()
    .execute::<PrimitiveArray>(&mut ctx)?;
let bitpacked = bitpack_encode(&physical, 4, None, &mut ctx)?.into_array();
let bitpacked_tiled = TiledFixedSizeList::try_new(
    bitpacked,
    128,
    raw_tiled.array_validity(),
    65,
    geometry(32, 64),
)?
.into_array();
test_array_consistency(&bitpacked_tiled, &mut ctx);
test_filter_conformance(&bitpacked_tiled, &mut ctx);
test_take_conformance(&bitpacked_tiled, &mut ctx);
```

The standard harness exercises cross-operation consistency, generic filter
fallback, and nullable-index semantics. Filter results need only match the
canonical oracle; this plan deliberately adds no preserving filter rule. Keep
the explicit oracle tests below because the standard harness does not require
slice/take to preserve this particular encoding.

- [ ] **Step 2: Add one table-driven canonical-oracle test for every tiled operation**

Define cases around both tile boundaries:

```rust
const ROW_COUNTS: &[usize] = &[0, 1, 15, 16, 31, 32, 33, 63, 64, 65];
const DIMENSION_COUNTS: &[u32] = &[0, 1, 3, 4, 63, 64, 65, 129];
```

For geometries 16-by-4, 32-by-64, 64-by-64, and 64-by-full-width, construct one
canonical FSL and its tiled encoding, then check the following. For a zero-width
list, represent "full-width" with tile dimension one so geometry remains
nonzero.

1. `encode` followed by canonical execution equals the canonical oracle;
2. `try_new` from the encoded physical parts equals the canonical oracle;
3. `execute_scalar` at row zero, the last row, and each row immediately around
   a row-tile boundary equals canonical FSL;
4. `row_tile_count` and `dimension_tile_count` equal independent `div_ceil`
   calculations;
5. each bounds value from `tile` and `tiles` has the expected logical and
   physical ranges, and `tile_elements(&bounds)` equals a canonical child
   `take` built from `row * list_size + dimension`;
6. slices `0..0`, full range, each boundary-adjacent range, and ranges crossing
   row tiles equal canonical FSL; nonempty results remain tiled with identical
   geometry, empty slices of nonempty sources are canonical, and the full
   `0..0` range of an already-empty tiled source retains tiling;
7. empty, identity, reverse, duplicated, unsorted, and nullable takes equal
   canonical FSL after execution; nondegenerate results become tiled with
   identical geometry, while empty takes and takes from empty sources use
   Vortex's ordinary degenerate representation.

Use the existing `assert_fsl_equivalent` helper after every operation:

```rust
let source_is_empty = canonical.is_empty();
let expected = canonical.clone().into_array().slice(range.clone())?;
let actual = tiled.clone().into_array().slice(range)?;
if source_is_empty {
    assert!(actual.is::<TiledFixedSizeList>());
    assert_eq!(actual.as_::<TiledFixedSizeList>().geometry(), geometry);
} else if actual.is_empty() {
    assert!(actual.is::<FixedSizeList>());
} else {
    assert!(actual.is::<TiledFixedSizeList>());
    assert_eq!(actual.as_::<TiledFixedSizeList>().geometry(), geometry);
}
assert_fsl_equivalent(&expected, &actual, &mut ctx)?;
```

For tile element views, derive expected indices directly from canonical
row-major coordinates; do not call `physical_offset`, `tile_bounds`, or another
tiled helper when constructing the oracle. Collect each iterator output into an
explicitly typed `Vec<TileBounds>` so the test fixes the hot iterator's
range-only item contract at compile time.

- [ ] **Step 3: Run the focused conformance suite**

Run:

```bash
cargo nextest run -p vortex-tiled-fsl conformance
```

Expected: all standard consistency checks and canonical-oracle comparisons
pass. Treat any mismatch or lost tiled parent as a failing test requiring a
minimal implementation fix before continuing.

- [ ] **Step 4: Run the complete crate tests**

Run:

```bash
cargo +nightly fmt --all
cargo nextest run -p vortex-tiled-fsl
```

Expected: the full crate suite passes after the conformance matrix.

- [ ] **Step 5: Commit the conformance suite**

```bash
git add encodings/tiled-fsl
git commit -s -m "test: check tiled lists against canonical FSL"
```

---

### Task 10: Differential fuzzing against canonical FSL

**Files:**
- Modify: `fuzz/Cargo.toml`
- Modify: `fuzz/src/lib.rs`
- Create: `fuzz/src/tiled_fsl.rs`
- Create: `fuzz/fuzz_targets/tiled_fsl.rs`
- Modify: `fuzz/README.md`

**Interfaces:**
- Consumes: the complete tiled array API and Vortex's canonical FSL helpers.
- Produces: `FuzzTiledFsl`, `TiledFslAction`, and
  `run_tiled_fsl(FuzzTiledFsl) -> VortexFuzzResult<()>`.
- Produces:
  `deterministic_tiled_fsl_cases() -> VortexFuzzResult<Vec<FuzzTiledFsl>>`.
- Produces: native cargo-fuzz target `tiled_fsl`.
- Checks: encode/execute, `try_new`, scalar access, tile bounds and tile element
  views, slice, take, outer validity, element validity, empty arrays, zero-width
  lists, and composed operation sequences.

- [ ] **Step 1: Add a non-vacuous deterministic fuzz-harness smoke test**

Add `vortex-tiled-fsl = { workspace = true }` to `fuzz/Cargo.toml`, declare the
module and public re-exports in `fuzz/src/lib.rs`, and add a unit test beside
the harness that calls it with explicitly constructed inputs:

```rust
#[cfg(test)]
mod tests {
    use crate::error::VortexFuzzResult;

    use super::{deterministic_tiled_fsl_cases, run_tiled_fsl};

    #[test]
    fn deterministic_tiled_fsl_smoke() -> VortexFuzzResult<()> {
        let cases = deterministic_tiled_fsl_cases()?;
        assert_eq!(cases.len(), 3);
        for input in cases {
            run_tiled_fsl(input)?;
        }
        Ok(())
    }
}
```

`deterministic_tiled_fsl_cases` constructs exactly:

1. zero rows, zero-width nonnullable `i32`, geometry 128-by-128, with
   `CheckTiles` and `Reconstruct`;
2. three rows by five nullable `u16` elements, mixed outer validity, geometry
   2-by-3, with `CheckTiles`, nullable duplicated `Take`, and `Reconstruct`;
3. 65 rows by 129 `f32` elements with independently mixed element and outer
   validity, geometry 32-by-64, with boundary-crossing `Slice`, reverse `Take`,
   `ScalarAt`, and `CheckTiles`.

Construct values and validity through ordinary `PrimitiveArray` and
`FixedSizeListArray` constructors, not `Arbitrary`. There is no conditional
parse or skipped case: the test fails if any fixture cannot be constructed or
run.

Give the helper this exact signature:

```rust
pub fn deterministic_tiled_fsl_cases() -> VortexFuzzResult<Vec<FuzzTiledFsl>>
```

- [ ] **Step 2: Run the smoke test and confirm the harness is absent**

Run:

```bash
cargo nextest run -p vortex-fuzz deterministic_tiled_fsl_smoke
```

Expected: compilation fails because `FuzzTiledFsl` and `run_tiled_fsl` do not
exist.

- [ ] **Step 3: Implement bounded arbitrary canonical FSL input generation**

In `fuzz/src/tiled_fsl.rs`, define:

```rust
#[derive(Clone, Debug)]
pub enum TiledFslAction {
    CheckTiles,
    ScalarAt(u16),
    Slice { start: u16, stop: u16 },
    Take(Vec<Option<u16>>),
    Reconstruct,
}

#[derive(Debug)]
pub struct FuzzTiledFsl {
    canonical: ArrayRef,
    geometry: TileGeometry,
    actions: Vec<TiledFslAction>,
}
```

Implement `Arbitrary` manually so inputs remain useful and bounded:

- choose any arbitrary `PType`;
- choose element and outer `Nullability` independently;
- choose `list_size` in `0..=64` and row count in `0..=128`;
- build
  `DType::FixedSizeList(Arc::new(DType::Primitive(ptype, element_nullability)), list_size, outer_nullability)`;
- generate the canonical oracle through `ArbitraryArray::arbitrary_with_config`
  with the exact dtype and row count;
- choose both tile sizes in `1..=128`, independently of the logical extents;
- generate `1..=8` actions;
- bound every take action to at most 64 nullable indices.

The generator must include empty arrays, zero-width lists, tile sizes larger
than the input, both tail shapes, every primitive PType, and both validity
layers without allocating from untrusted sizes.

- [ ] **Step 4: Implement independent canonical-oracle checks**

Create a session that registers `vortex_tiled_fsl` exactly once. At the start
of `run_tiled_fsl`:

1. execute the generated oracle to canonical `FixedSizeListArray`;
2. encode it to tiled form;
3. compare logical dtype, length, and values with `array::assert_array_eq`;
4. compare bounded scalar probes at row zero, the last row, and each row-tile
   boundary with canonical FSL;
5. validate `row_tile_count` and `dimension_tile_count` against independent
   `div_ceil` calculations.

Use these shared boundaries:

```rust
static TILED_FSL_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_tiled_fsl::initialize(&session);
    session
});

fn fuzz<T>(result: VortexResult<T>) -> VortexFuzzResult<T> {
    result.map_err(|error| {
        VortexFuzzError::VortexError(error, Backtrace::capture())
    })
}

fn assert_tiled_geometry(
    array: &ArrayRef,
    expected: TileGeometry,
) -> VortexResult<()> {
    if !array.is::<TiledFixedSizeList>() {
        vortex_bail!("expected nondegenerate operation to retain tiled FSL");
    }
    let actual = array.as_::<TiledFixedSizeList>().geometry();
    if actual != expected {
        vortex_bail!(
            "expected geometry {expected:?}, found {actual:?}"
        );
    }
    Ok(())
}
```

Check physical order independently of tiled offset helpers:

```rust
fn expected_tile_indices(
    list_size: usize,
    rows: Range<usize>,
    dimensions: Range<usize>,
) -> ArrayRef {
    PrimitiveArray::from_iter(
        dimensions.flat_map(|dimension| {
            rows.clone().map(move |row| (row * list_size + dimension) as u64)
        }),
    )
    .into_array()
}
```

Execute the canonical FSL outer validity once to a mask. For every bounds value
from `tiles()`:

- assert logical ranges do not overlap incorrectly and jointly cover every
  row/dimension tile;
- assert `physical_range.len() == row_range.len() * dimension_range.len()`;
- build the expected tile by passing clones of `bounds.row_range` and
  `bounds.dimension_range` to `expected_tile_indices`, then taking those
  returned indices from the canonical FSL element child;
- construct tile-relative positions only for coordinates whose outer row is
  valid, then take those same positions from both the expected tile and
  `tile_elements(&bounds)`;
- compare the selected expected and actual values, including element validity;
- assert physical ranges are adjacent from zero through exactly
  `len * list_size`.

Put those checks behind the helper consumed by the action runner:

```rust
fn check_tiles(
    canonical: &ArrayRef,
    tiled: &ArrayRef,
    step: usize,
    ctx: &mut ExecutionCtx,
) -> VortexFuzzResult<()>
```

This oracle deliberately derives expected values from canonical row-major
coordinates and must not call `physical_offset` or `tile_bounds`. It must not
compare placeholder child values beneath an invalid outer row: canonical
builders may append defaults there while tiled nullable take may copy source row
zero, and those physical values are deliberately not part of logical equality.

- [ ] **Step 5: Differentially execute and compose every tiled operation**

Normalize action seeds against the current oracle length:

- `ScalarAt`: skip on empty arrays; otherwise use `seed % len` and compare
  canonical and tiled scalars.
- `Slice`: map both seeds into one valid `start..stop`, use
  `array::slice_canonical_array` for the oracle and `tiled.slice` for the
  candidate, and assert logical equality. A nonempty result must be tiled with
  unchanged geometry. An empty result from a nonempty source must be canonical
  and ends this composed action sequence after its logical check. For an
  already-empty tiled source, `0..0` is the full range and must retain the tiled
  representation and geometry.
- `Take`: map every valid seed modulo nonzero `len`, turn every valid seed into
  `None` when `len == 0`, use `array::take_canonical_array` for the oracle and
  `tiled.take` for the candidate. For nonempty indices from a nonempty source,
  execute the lazy candidate with
  `execute_until::<TiledFixedSizeList>(&mut ctx)`, then assert equality, tiled
  encoding, and unchanged geometry. Empty indices or a take from an empty source
  may execute to Vortex's canonical/constant degenerate result and end the
  action sequence after logical equality is checked.
- `Reconstruct`: call `try_new` with the current physical child and parts, then
  assert equality.
- `CheckTiles`: rerun the independent tile check from Step 4.

After every mutating action, run `assert_array_eq`; the explicit `ScalarAt`
action covers scalar dispatch without turning every fuzz step into a
row-at-a-time scan. `run_tiled_fsl` returns `Ok(())` on success. `Arbitrary`
decoding occurs before a typed libFuzzer target invokes this function, so
malformed bytes do not need a second `Ok(false)` rejection channel.

Implement the state transition in one helper with this exact boundary:

```rust
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
            check_tiles(canonical, tiled, step, ctx)?;
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
            let first = usize::from(start).min(usize::from(stop));
            let last = usize::from(start).max(usize::from(stop));
            let start = first % (canonical.len() + 1);
            let stop = start + last % (canonical.len() - start + 1);
            *canonical =
                fuzz(slice_canonical_array(canonical, start, stop, ctx))?;
            *tiled = fuzz(tiled.clone().slice(start..stop))?;
            assert_array_eq(canonical, tiled, step, ctx)?;
            if tiled.is_empty() {
                if source_is_empty {
                    fuzz(assert_tiled_geometry(tiled, geometry))?;
                    return Ok(ControlFlow::Continue(()));
                }
                fuzz((|| {
                    vortex_ensure!(
                        tiled.is::<FixedSizeList>(),
                        "expected an empty slice of a nonempty source to be canonical FSL"
                    );
                    Ok(())
                })())?;
                return Ok(ControlFlow::Break(()));
            }
            fuzz(assert_tiled_geometry(tiled, geometry))?;
        }
        TiledFslAction::Take(seeds) => {
            let source_is_empty = canonical.is_empty();
            let indices = seeds
                .into_iter()
                .map(|seed| {
                    seed.and_then(|seed| {
                        (!source_is_empty)
                            .then(|| usize::from(seed) % canonical.len())
                    })
                })
                .collect::<Vec<_>>();
            let index_array = if indices.contains(&None) {
                PrimitiveArray::from_option_iter(
                    indices.iter().map(|index| index.map(|index| index as u64)),
                )
                .into_array()
            } else {
                PrimitiveArray::from_iter(
                    indices.iter().map(|index| index.unwrap() as u64),
                )
                .into_array()
            };
            *canonical =
                fuzz(take_canonical_array(canonical, &indices, ctx))?;
            let lazy = fuzz(tiled.clone().take(index_array))?;
            *tiled = if indices.is_empty() || source_is_empty {
                fuzz(lazy.execute::<Canonical>(ctx))?.into_array()
            } else {
                fuzz(lazy.execute_until::<TiledFixedSizeList>(ctx))?
            };
            assert_array_eq(canonical, tiled, step, ctx)?;
            if indices.is_empty() || source_is_empty {
                return Ok(ControlFlow::Break(()));
            }
            fuzz(assert_tiled_geometry(tiled, geometry))?;
        }
        TiledFslAction::Reconstruct => {
            let array = tiled.as_::<TiledFixedSizeList>();
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
    }
    Ok(ControlFlow::Continue(()))
}
```

Import `std::ops::ControlFlow` and the existing
`crate::array::{assert_array_eq, assert_scalar_eq}` helpers. Step 4 defines
`assert_tiled_geometry` and `check_tiles`. In `run_tiled_fsl`, enumerate
actions and stop only on `ControlFlow::Break(())`:

```rust
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
```

- [ ] **Step 6: Add the native fuzz entrypoint**

In `fuzz/Cargo.toml`:

```toml
[[bin]]
bench = false
doc = false
name = "tiled_fsl"
path = "fuzz_targets/tiled_fsl.rs"
test = false
required-features = ["native"]
```

The target follows the existing error/Corpus convention:

```rust
#![no_main]

use libfuzzer_sys::{Corpus, fuzz_target};
use vortex_error::vortex_panic;
use vortex_fuzz::{FuzzTiledFsl, run_tiled_fsl};

fuzz_target!(|input: FuzzTiledFsl| -> Corpus {
    match run_tiled_fsl(input) {
        Ok(()) => Corpus::Keep,
        Err(error) => vortex_panic!("{error}"),
    }
});
```

Document `cargo +nightly fuzz run tiled_fsl` and one-input crash replay in
`fuzz/README.md`.

- [ ] **Step 7: Build and run a bounded local fuzz campaign**

Run:

```bash
cargo +nightly fmt --all
cargo nextest run -p vortex-fuzz deterministic_tiled_fsl_smoke
cargo +nightly fuzz build --dev --sanitizer=none tiled_fsl
cargo +nightly fuzz run --dev --sanitizer=none tiled_fsl -- -runs=10000 -max_len=4096
```

Expected: all three deterministic cases run, the target builds, and 10,000
inputs produce no logical divergence, panic, sanitizer failure, or unbounded
allocation.

- [ ] **Step 8: Commit the differential fuzz harness**

```bash
git add fuzz Cargo.lock
git commit -s -m "test: fuzz tiled lists against canonical FSL"
```

---

### Task 11: Scheduled tiled-FSL fuzzing

**Files:**
- Modify: `.github/workflows/fuzz.yml`

**Interfaces:**
- Consumes: native cargo-fuzz target `tiled_fsl` from Task 10.
- Produces: scheduled four-worker corpus execution and automatic crash
  reporting using the repository's shared fuzz workflows.

- [ ] **Step 1: Add the target to scheduled fuzzing**

Add these exact jobs to `.github/workflows/fuzz.yml`:

```yaml
  # ============================================================================
  # Tiled Fixed-Size List Fuzzer
  # ============================================================================
  tiled_fsl_fuzz:
    name: "Tiled Fixed-Size List Fuzz"
    uses: ./.github/workflows/run-fuzzer.yml
    with:
      fuzz_target: tiled_fsl
      jobs: 4
    secrets:
      R2_FUZZ_ACCESS_KEY_ID: ${{ secrets.R2_FUZZ_ACCESS_KEY_ID }}
      R2_FUZZ_SECRET_ACCESS_KEY: ${{ secrets.R2_FUZZ_SECRET_ACCESS_KEY }}

  report-tiled-fsl-fuzz-failures:
    name: "Report Tiled Fixed-Size List Fuzz Failures"
    needs: tiled_fsl_fuzz
    if: always() && needs.tiled_fsl_fuzz.outputs.crashes_found == 'true'
    permissions:
      issues: write
      contents: read
      id-token: write
      pull-requests: read
    uses: ./.github/workflows/report-fuzz-crash.yml
    with:
      fuzz_target: tiled_fsl
      crash_file: ${{ needs.tiled_fsl_fuzz.outputs.first_crash_name }}
      artifact_url: ${{ needs.tiled_fsl_fuzz.outputs.artifact_url }}
      artifact_name: tiled_fsl-crash-artifacts
      logs_artifact_name: tiled_fsl-logs
      branch: ${{ github.ref_name }}
      commit: ${{ github.sha }}
    secrets:
      claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
      gh_token: ${{ secrets.GITHUB_TOKEN }}
      incident_io_alert_token: ${{ secrets.INCIDENT_IO_ALERT_TOKEN }}
```

Use `tiled_fsl` consistently as target, fuzz name, corpus key, crash artifact
prefix, and logs prefix. Do not add `vortex/unstable_encodings`: the target
registers the direct encoding crate explicitly.

- [ ] **Step 2: Validate the workflow**

Run:

```bash
yamllint --strict -c .yamllint.yaml .github/workflows/fuzz.yml
git diff --check -- .github/workflows/fuzz.yml
```

Expected: the YAML and whitespace checks pass, and both jobs reference the
existing reusable workflow inputs exactly.

- [ ] **Step 3: Commit scheduled fuzzing**

```bash
git add .github/workflows/fuzz.yml
git commit -s -m "ci: schedule tiled list fuzzing"
```

---

### Task 12: Representation benchmarks

**Files:**
- Modify: `encodings/tiled-fsl/Cargo.toml`
- Create: `encodings/tiled-fsl/benches/tiled_fsl.rs`

**Interfaces:**
- Consumes: complete public encoding API.
- Produces: Divan benchmarks named `encode`, `execute`, `slice`, `take`,
  `score_canonical`, `score_prepared`, and `score_end_to_end`.
- Produces: benchmark arguments covering row tiles 32/64, full-width/64-dimension bands, and a 16-row by 4-dimension microtile.

- [ ] **Step 1: Add the benchmark target and exact argument matrix**

Add:

```toml
[dev-dependencies]
divan = { workspace = true }
mimalloc = { workspace = true }

[[bench]]
name = "tiled_fsl"
harness = false
```

Create the benchmark argument type:

```rust
use std::fmt;

#[derive(Clone, Copy)]
struct Args {
    rows: usize,
    dimensions: usize,
    tile_rows: u32,
    tile_dimensions: TileDimensions,
}

#[derive(Clone, Copy)]
enum TileDimensions {
    Full,
    Fixed(u32),
}

impl fmt::Display for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dimension_tile = match self.tile_dimensions {
            TileDimensions::Full => "full".to_owned(),
            TileDimensions::Fixed(dimensions) => dimensions.to_string(),
        };
        write!(
            f,
            "rows{}_dims{}_tile{}x{}",
            self.rows,
            self.dimensions,
            self.tile_rows,
            dimension_tile,
        )
    }
}

fn args() -> Vec<Args> {
    let mut args = Vec::new();
    let geometries = [
        (32, TileDimensions::Full),
        (64, TileDimensions::Full),
        (32, TileDimensions::Fixed(64)),
        (64, TileDimensions::Fixed(64)),
        (16, TileDimensions::Fixed(4)),
    ];
    for rows in [1_024, 16_384] {
        for dimensions in [128, 768, 1_536] {
            for (tile_rows, tile_dimensions) in geometries {
                args.push(Args {
                    rows,
                    dimensions,
                    tile_rows,
                    tile_dimensions,
                });
            }
        }
    }
    for rows in [31, 33, 63, 65] {
        for dimensions in [31, 33, 63, 65] {
            for tile_rows in [32, 64] {
                args.push(Args {
                    rows,
                    dimensions,
                    tile_rows,
                    tile_dimensions: TileDimensions::Fixed(64),
                });
            }
        }
    }
    args
}
```

- [ ] **Step 2: Implement deterministic raw and bitpacked fixtures**

Use deterministic low-bit `u8` values for bitpacked cases, deterministic
dimension-specific `u8` query weights, and `f32` values for floating-point
encode/execute cases. Construct inputs and query weights outside timed loops.
Initialize both `vortex_tiled_fsl` and `vortex_fastlanes` in the benchmark
session.

Use these fixture helpers:

```rust
fn tile_geometry(args: Args) -> TileGeometry {
    let dimensions = match args.tile_dimensions {
        TileDimensions::Full => u32::try_from(args.dimensions).unwrap(),
        TileDimensions::Fixed(dimensions) => dimensions,
    };
    TileGeometry::new(
        NonZeroU32::new(args.tile_rows).unwrap(),
        NonZeroU32::new(dimensions).unwrap(),
    )
}

fn canonical_u8(args: Args) -> FixedSizeListArray {
    FixedSizeListArray::new(
        PrimitiveArray::from_iter(
            (0..args.rows * args.dimensions)
                .map(|index| ((index * 17 + index / args.dimensions) & 0x0f) as u8),
        )
        .into_array(),
        u32::try_from(args.dimensions).unwrap(),
        Validity::NonNullable,
        args.rows,
    )
}

fn query(args: Args) -> Vec<u8> {
    (0..args.dimensions)
        .map(|dimension| ((dimension * 13 + 7) & 0x0f) as u8)
        .collect()
}

fn raw_tiled(
    args: Args,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TiledFixedSizeListArray> {
    TiledFixedSizeList::encode(
        canonical_u8(args).as_view(),
        tile_geometry(args),
        ctx,
    )
}

fn bitpacked_tiled(
    args: Args,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TiledFixedSizeListArray> {
    let raw = raw_tiled(args, ctx)?;
    let physical = raw
        .elements()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let bitpacked = bitpack_encode(&physical, 4, None, ctx)?.into_array();
    TiledFixedSizeList::try_new(
        bitpacked,
        u32::try_from(args.dimensions)?,
        raw.array_validity(),
        args.rows,
        tile_geometry(args),
    )
}
```

Use this floating-point fixture for encode/execute measurements:

```rust
fn canonical_f32(args: Args) -> FixedSizeListArray {
    FixedSizeListArray::new(
        PrimitiveArray::from_iter(
            (0..args.rows * args.dimensions)
                .map(|index| ((index * 17) % 1_009) as f32 / 1_009.0),
        )
        .into_array(),
        u32::try_from(args.dimensions).unwrap(),
        Validity::NonNullable,
        args.rows,
    )
}
```

- [ ] **Step 3: Implement operation benchmarks with honest preparation boundaries**

Measure:

- `encode`: canonical FSL to tiled;
- `execute`: tiled to canonical FSL;
- `slice`: a small, a half-width, and a tile-boundary-crossing row range;
- `take`: sorted sparse and unsorted duplicated row indices;
- `score_canonical`: score the row-major canonical child with one deterministic
  query weight per dimension;
- `score_prepared`: execute the tiled physical child outside the timed loop,
  then perform the identical weighted row accumulation through `array.tiles()`
  and each supplied `physical_range`;
- `score_end_to_end`: execute the raw or bitpacked physical child once inside
  each timed sample, then perform the same tiled scoring kernel.

The two scoring bodies are:

```rust
fn score_canonical(
    values: &[u8],
    rows: usize,
    dimensions: usize,
    query: &[u8],
) -> Vec<u64> {
    let mut scores = vec![0; rows];
    for (row, score) in scores.iter_mut().enumerate() {
        let row_values = &values[row * dimensions..(row + 1) * dimensions];
        *score = row_values
            .iter()
            .zip(query)
            .map(|(&value, &weight)| u64::from(value) * u64::from(weight))
            .sum();
    }
    scores
}

fn score_tiled(
    array: ArrayView<'_, TiledFixedSizeList>,
    values: &[u8],
    query: &[u8],
) -> Vec<u64> {
    let mut scores = vec![0; array.len()];
    for bounds in array.tiles() {
        let tile_rows = bounds.row_range.len();
        for (dimension_offset, dimension) in bounds.dimension_range.clone().enumerate() {
            let physical_start =
                bounds.physical_range.start + dimension_offset * tile_rows;
            let weight = u64::from(query[dimension]);
            for (row_offset, row) in bounds.row_range.clone().enumerate() {
                scores[row] +=
                    u64::from(values[physical_start + row_offset]) * weight;
            }
        }
    }
    scores
}
```

Before timing, assert that both bodies return identical scores for every
fixture and geometry. Use `divan::black_box` on results. Do not call
`tile_elements` per tile in the prepared benchmark; the benchmark exists to
prove that range-only traversal supports one-time child execution. A plain sum
of all physical values is forbidden as a geometry comparison: because the
physical ranges are adjacent, it degenerates to the same sequential scan for
every layout and measures only tile-count overhead.

- [ ] **Step 4: Compile and run the benchmark families**

Run:

```bash
cargo +nightly fmt --all
cargo bench -p vortex-tiled-fsl --bench tiled_fsl --no-run
cargo bench -p vortex-tiled-fsl --bench tiled_fsl -- score_canonical
cargo bench -p vortex-tiled-fsl --bench tiled_fsl -- score_prepared
cargo bench -p vortex-tiled-fsl --bench tiled_fsl -- score_end_to_end
```

Expected: the target compiles, the canonical baseline runs, and both
raw/bitpacked tiled scoring families produce measurements for every candidate
geometry.

- [ ] **Step 5: Commit benchmarks**

```bash
git add encodings/tiled-fsl/Cargo.toml encodings/tiled-fsl/benches Cargo.lock
git commit -s -m "bench: measure tiled fixed-size lists"
```

---

### Task 13: Final Vortex verification and implementation evidence

**Files:**
- Modify only if checks expose defects: files introduced or changed in Tasks 1-12.
- Do not create a SpiralDB path override in this task.

**Interfaces:**
- Consumes: complete Vortex implementation from Tasks 1-12.
- Produces: a clean, reviewable Vortex branch and benchmark command/output summary for the later SpiralDB plan.

- [ ] **Step 1: Run formatting and targeted tests**

Run:

```bash
cargo +nightly fmt --all -- --check
cargo nextest run -p vortex-tiled-fsl
cargo nextest run -p vortex-file --features unstable_encodings --test tiled_fsl
cargo nextest run -p vortex --features unstable_encodings
cargo nextest run -p vortex-fuzz deterministic_tiled_fsl_smoke
```

Expected: all pass.

- [ ] **Step 2: Run debug and release lints**

Run:

```bash
cargo clippy -p vortex-tiled-fsl --all-targets --all-features -- -D warnings
cargo clippy -p vortex-file --all-targets --features unstable_encodings -- -D warnings
cargo clippy -p vortex-fuzz --bin tiled_fsl --features native -- -D warnings
cargo clippy --release -p vortex-tiled-fsl --all-targets --all-features -- -D warnings
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy --release --workspace --all-targets --all-features -- -D warnings
```

Expected: no warnings.

- [ ] **Step 3: Verify public documentation and dependency hygiene**

Run:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p vortex-tiled-fsl
cargo test --doc -p vortex-tiled-fsl
taplo fmt --check
cargo shear
reuse lint
cargo deny check licenses
cargo audit
yamllint --strict -c .yamllint.yaml .github/workflows/fuzz.yml
```

Expected: rustdoc, doctests, TOML formatting, unused-dependency, REUSE
source-file licensing, cargo-deny dependency licensing, advisory, and workflow
YAML checks pass.

- [ ] **Step 4: Re-run benchmark compilation and capture selected evidence**

Run:

```bash
cargo bench -p vortex-tiled-fsl --bench tiled_fsl --no-run
cargo bench -p vortex-tiled-fsl --bench tiled_fsl -- score_canonical
cargo bench -p vortex-tiled-fsl --bench tiled_fsl -- score_prepared
cargo bench -p vortex-tiled-fsl --bench tiled_fsl -- score_end_to_end
cargo +nightly fuzz build --dev --sanitizer=none tiled_fsl
```

Record, without committing generated artifacts:

- canonical row-major scoring throughput;
- prepared tiled scoring throughput for 32/64-by-full, 32/64-by-64, and
  16-by-4;
- end-to-end raw versus bitpacked tiled scoring;
- encode and execute costs for 1,024-by-128 and at least one wider fixture;
- the relative overheads that the later SpiralDB scoring benchmark must explain.

Do not eliminate a geometry based on this representation-level proxy. Carry all
candidate geometries into the first SpiralDB scorer experiment; only the real
RaBitQ kernel can establish that a geometry is dominated for the target
workload.

- [ ] **Step 5: Inspect the complete branch diff and repository state**

Run:

```bash
git diff --check develop...HEAD
git diff --stat develop...HEAD
git status --short --branch
```

Expected: no whitespace errors; only the user-owned `.agents/worktrees/` may
remain untracked.

- [ ] **Step 6: Commit any verification fixes separately**

When Steps 1-5 modify a tracked file, commit those verification fixes
separately:

```bash
git add Cargo.toml Cargo.lock encodings/tiled-fsl vortex-file vortex fuzz .github/workflows/fuzz.yml
git commit -s -m "fix: harden tiled fixed-size-list encoding"
```

If no files changed, do not create an empty commit.

- [ ] **Step 7: Hand off to the separate SpiralDB integration plan**

The next plan must point SpiralDB locally at `/Users/will/git/vortex`, add the
RaBitQ scoring kernel outside Vortex, differential-test every score kind and
code depth, benchmark the candidate geometries against the scalar scorer, and
remove the path override before the dependent SpiralDB merge. Do not start that
work until this Vortex API and its benchmark evidence have passed review.
