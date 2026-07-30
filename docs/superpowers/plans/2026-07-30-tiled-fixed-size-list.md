# Tiled Fixed-Size List Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental Vortex encoding that stores primitive fixed-size-list elements in configurable two-dimensional tiles and preserves that representation through slice, take, and file round-trips.

**Architecture:** A new `vortex-tiled-fsl` encoding crate owns tile geometry, the physical child order, transpose/inverse-transpose logic, and generic Vortex operations. The parent retains an ordinary logical `FixedSizeList<Primitive>` dtype, while its one primitive element child can itself use another Vortex encoding; outer validity remains a separate optional child. Vortex file and facade integration are feature-gated behind `unstable_encodings`; SpiralDB-specific scoring remains outside this plan.

**Tech Stack:** Rust, Vortex array vtables and execution contexts, Prost metadata, FastLanes composition tests, Divan benchmarks, Vortex file editions.

## Global Constraints

- Support only `DType::FixedSizeList` whose element dtype is `DType::Primitive`.
- Physical order is `dimension tile -> row tile -> dimension within tile -> row within tile`.
- `TileGeometry.rows` and `TileGeometry.dimensions` are both `NonZeroU32`.
- Store exactly `len * list_size` physical elements with checked arithmetic and no tail padding.
- Preserve outer row validity separately and transpose element validity with element values.
- Preserve the tiled parent and its geometry through `slice` and `take`; neither operation may canonicalize the whole source.
- Use canonical `FixedSizeListArray` as the deterministic unit-test and fuzz-test oracle for every tiled operation.
- Keep production `vortex-tiled-fsl` independent of `vortex-fastlanes`; FastLanes is a dev-only composition dependency.
- Do not add the encoding to BtrBlocks or any automatic compression strategy.
- Register and re-export the encoding only through `unstable_encodings`.
- Do not add algorithm, arithmetic, padding, or layout-version metadata fields.
- Do not modify or remove the user-owned untracked `.agents/worktrees/` directory.
- Do not create an alternate Cargo target directory.
- Run Rust formatting after every Rust-editing task and use signed-off commits.
- This plan stops after the Vortex representation, file integration, and benchmarks. SpiralDB scorer integration gets a separate plan after this API and its measurements exist.

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

- [ ] **Step 1: Add the crate scaffold and failing geometry tests**

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

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
cargo test -p vortex-tiled-fsl geometry
```

Expected: compilation fails because `TileGeometry`, `physical_offset`, and
`tile_bounds` do not yet exist.

- [ ] **Step 3: Implement checked geometry and physical ranges**

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

- [ ] **Step 4: Format and run geometry tests**

Run:

```bash
cargo +nightly fmt --all
cargo test -p vortex-tiled-fsl geometry
```

Expected: both golden tests pass, including the two unpadded tail tiles.

- [ ] **Step 5: Commit the geometry**

```bash
git add Cargo.toml Cargo.lock encodings/tiled-fsl
git commit -s -m "feat: add tiled fixed-size-list geometry"
```

---

### Task 2: Tiled array, transpose, tile traversal, and canonical execution

**Files:**
- Modify: `encodings/tiled-fsl/Cargo.toml`
- Modify: `encodings/tiled-fsl/src/lib.rs`
- Modify: `encodings/tiled-fsl/src/geometry.rs`
- Create: `encodings/tiled-fsl/src/array.rs`
- Create: `encodings/tiled-fsl/src/transpose.rs`
- Create: `encodings/tiled-fsl/src/operations.rs`
- Create: `encodings/tiled-fsl/src/tests.rs`
- Create through the metadata golden test: `encodings/tiled-fsl/goldenfiles/tiled_fsl.metadata`

**Interfaces:**
- Consumes: `TileGeometry`, `TileBounds`, `tile_bounds`, and `physical_offset` from Task 1.
- Produces: `TiledFixedSizeList` and `TiledFixedSizeListArray = Array<TiledFixedSizeList>`.
- Produces: `TiledFixedSizeList::try_new(elements, list_size, validity, len, geometry)`.
- Produces: `TiledFixedSizeList::encode(ArrayView<FixedSizeList>, geometry, &mut ExecutionCtx)`.
- Produces: `TiledFixedSizeListArrayExt::{elements, geometry, list_size, array_validity, row_tile_count, dimension_tile_count, tile, tiles}`.
- Produces: `TiledFixedSizeListTile` with logical ranges, `physical_range`, and `elements()`.
- Produces: crate-private `TransposeDirection::{CanonicalToTiled, TiledToCanonical}` and `transpose_validity`.
- Produces: `initialize(&VortexSession)`.

- [ ] **Step 1: Add failing layout, round-trip, validity, and invariant tests**

Add `prost`, `vortex-array`, `vortex-buffer`, `vortex-mask`, and
`vortex-session` as workspace dependencies. Add `_test-harness` to the
`vortex-array` dev-dependency and add `rstest`.

Create tests that establish all of these behaviors:

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
    let tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry, &mut ctx)?;
    Ok((canonical, tiled, ctx))
}

fn u8_fixture(
    rows: usize,
    dimensions: u32,
    geometry: TileGeometry,
) -> VortexResult<(FixedSizeListArray, TiledFixedSizeListArray, ExecutionCtx)> {
    fixture(rows, dimensions, geometry)
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
fn golden_physical_child_and_tiles() -> VortexResult<()> {
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
    let tiled = TiledFixedSizeList::encode(canonical.as_view(), geometry(2, 3), &mut ctx)?;
    let physical = tiled.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(
        physical.as_slice::<u16>(),
        &[0, 10, 1, 11, 2, 12, 20, 21, 22, 3, 13, 4, 14, 23, 24]
    );
    assert_eq!(
        tiled.tiles().map(|tile| tile.physical_range).collect::<Vec<_>>(),
        vec![0..6, 6..9, 9..13, 13..15]
    );
    assert_arrays_eq!(canonical, tiled, &mut ctx);
    Ok(())
}
```

Also add:

- a table-driven round-trip for every `PType` accepted by
  `match_each_native_ptype!`;
- independent outer validity and element validity cases using
  `Validity::NonNullable`, `AllValid`, `AllInvalid`, and mixed masks;
- `len == 0`, `list_size == 0`, and tile sizes larger than both extents;
- rejection of a non-primitive child;
- rejection of `elements.len() != len * list_size`;
- rejection of an outer validity length different from `len`;
- checked multiplication overflow;
- zero tile geometry when decoding metadata;
- tile index bounds;
- `execute_scalar` on every row compared directly with canonical FSL, without
  canonicalizing unrelated rows;
- `test_array_consistency` from
  `vortex_array::compute::conformance::consistency` on nonnullable, outer
  nullable, and element-nullable tiled fixtures;
- a metadata snapshot using `vortex_array::test_harness::check_metadata`.

The metadata snapshot input is exactly:

```rust
check_metadata(
    "tiled_fsl.metadata",
    &TiledFixedSizeListMetadata {
        tile_rows: 32,
        tile_dimensions: 64,
    }
    .encode_to_vec(),
);
```

- [ ] **Step 2: Run the tests and confirm the new API is missing**

Run:

```bash
cargo test -p vortex-tiled-fsl
```

Expected: compilation fails because `TiledFixedSizeList` and the array
extension methods have not been implemented.

- [ ] **Step 3: Implement the physical array and metadata**

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
    // Validate before Array::from_parts_unchecked.
}
```

Validation must enforce every invariant in the design. Serialization stores
only the two geometry fields. Deserialization obtains `list_size` and the
element dtype from the logical dtype, requests one element child of checked
length `len * list_size`, accepts zero or one outer-validity child, rejects all
other child counts, and rejects zero geometry.

Implement `ValidityVTable<TiledFixedSizeList>` by converting the optional
validity slot with `child_to_validity`.

- [ ] **Step 4: Implement bulk transpose and inverse transpose**

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
values, and construct one mixed output validity; never probe validity once per
element through `execute_scalar`.

The inverse function allocates `BufferMut::<T>::zeroed(len * list_size)` and
places every physical value at `row * list_size + dimension`. Reorder mixed
element validity through the identical mapping. Do not call `execute_scalar`
inside either element loop.

`TiledFixedSizeList::encode` executes the canonical FSL element child exactly
once to `PrimitiveArray`, transposes it, and preserves the outer FSL validity.
The VTable `execute` uses `require_child!` to request a canonical primitive
element child, inverse-transposes it, and returns a canonical
`FixedSizeListArray`.

- [ ] **Step 5: Implement efficient tile descriptors and scalar access**

Define a descriptor that carries the parent child once and slices only on
demand:

```rust
#[derive(Clone, Debug)]
pub struct TiledFixedSizeListTile {
    pub row_range: Range<usize>,
    pub dimension_range: Range<usize>,
    pub physical_range: Range<usize>,
    elements: ArrayRef,
}

impl TiledFixedSizeListTile {
    pub fn elements(&self) -> VortexResult<ArrayRef> {
        self.elements.slice(self.physical_range.clone())
    }
}
```

`tiles()` must iterate dimension tiles outside row tiles so descriptors arrive
in physical order. `tile(row_tile, dimension_tile)` uses `tile_bounds` and
returns an error for invalid indices.

Implement `OperationsVTable::scalar_at` by producing the physical indices for
one selected row, performing one bulk child `take`, and constructing a
one-row canonical FSL scalar. It must not execute the entire tiled source and
must not call `execute_scalar` once per source row.

Register the array vtable in:

```rust
pub fn initialize(session: &VortexSession) {
    session.arrays().register(TiledFixedSizeList);
}
```

- [ ] **Step 6: Materialize and verify the metadata golden**

Run:

```bash
cargo test -p vortex-tiled-fsl tiled_fsl_metadata
xxd -g 1 encodings/tiled-fsl/goldenfiles/tiled_fsl.metadata
cargo test -p vortex-tiled-fsl tiled_fsl_metadata
```

Expected: when no golden exists, the first run creates it and may report the
new file; its bytes are exactly `08 20 10 40`, encoding only row tile 32 and
dimension tile 64; the second run passes.

- [ ] **Step 7: Format and run all crate tests**

Run:

```bash
cargo +nightly fmt --all
cargo test -p vortex-tiled-fsl
```

Expected: all physical-order, round-trip, dtype, validity, degenerate, metadata,
and scalar tests pass.

- [ ] **Step 8: Commit the working tiled array**

```bash
git add Cargo.toml Cargo.lock encodings/tiled-fsl
git commit -s -m "feat: add tiled fixed-size-list array"
```

---

### Task 3: Encoding-preserving slice

**Files:**
- Modify: `encodings/tiled-fsl/src/array.rs`
- Modify: `encodings/tiled-fsl/src/lib.rs`
- Create: `encodings/tiled-fsl/src/gather.rs`
- Create: `encodings/tiled-fsl/src/rules.rs`
- Create: `encodings/tiled-fsl/src/slice.rs`
- Modify: `encodings/tiled-fsl/src/tests.rs`

**Interfaces:**
- Consumes: geometry and constructors from Tasks 1-2.
- Produces: crate-private `physical_indices_for_rows(source_len, list_size, geometry, rows) -> VortexResult<Buffer<u64>>`.
- Produces: crate-private `gather_tiled_rows(array, rows, validity) -> VortexResult<TiledFixedSizeListArray>`.
- Produces: `SliceReduce for TiledFixedSizeList`.
- Preserves: `TiledFixedSizeList` parent and exact `TileGeometry`.

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
    assert!(actual.is::<TiledFixedSizeList>());
    assert_eq!(actual.as_::<TiledFixedSizeList>().geometry(), geometry(32, 64));
    assert_arrays_eq!(expected, actual, &mut ctx);
    Ok(())
}
```

Repeat representative cases with:

- row and element mixed validity;
- zero-width lists;
- geometry larger than the source;
- an encoded primitive child supplied through `try_new`;
- a source much larger than the output, asserting the child gather length is
  exactly `output_len * list_size`.

- [ ] **Step 2: Run the focused slice tests and confirm fallback loses tiling**

Run:

```bash
cargo test -p vortex-tiled-fsl slice
```

Expected: the value fallback may work, but the encoding assertion fails because
no `SliceReduce` rule is registered.

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
the placeholder for a null selected row when the source is nonempty.

Add a separate helper for the empty-source/all-null case that builds
`len * list_size` default primitive values with `builder_with_capacity`; never
invent an index into an empty source.

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
cargo test -p vortex-tiled-fsl slice
cargo test -p vortex-tiled-fsl
```

Expected: slice preserves the tiled parent for every boundary and validity case.

- [ ] **Step 6: Commit slice support**

```bash
git add encodings/tiled-fsl
git commit -s -m "feat: preserve tiled lists through slice"
```

---

### Task 4: Encoding-preserving take

**Files:**
- Modify: `encodings/tiled-fsl/src/lib.rs`
- Create: `encodings/tiled-fsl/src/kernel.rs`
- Create: `encodings/tiled-fsl/src/take.rs`
- Modify: `encodings/tiled-fsl/src/tests.rs`

**Interfaces:**
- Consumes: `physical_indices_for_rows` and empty-source defaults from Task 3.
- Produces: `TakeExecute for TiledFixedSizeList`.
- Produces: crate-private `collect_checked_rows<I: IntegerPType>(indices, mask, source_len) -> VortexResult<Vec<Option<usize>>>`.
- Produces: `kernel::initialize(&VortexSession)` registering `TakeExecuteAdaptor`.
- Preserves: tiled parent, geometry, requested row order, duplicates, and nullable-index semantics.

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
let actual = tiled.into_array().take(indices.clone())?;
assert!(actual.is::<TiledFixedSizeList>());
assert_eq!(actual.as_::<TiledFixedSizeList>().geometry(), geometry(32, 64));
assert_arrays_eq!(canonical.into_array().take(indices)?, actual, &mut ctx);
```

Also test:

- every integer index PType accepted by Vortex take;
- all-null indices against an empty source;
- zero-width lists;
- out-of-bounds valid indices;
- nullable indices applied to a nonnullable source produce nullable outer FSL;
- output child length is exactly `indices.len() * list_size`.

- [ ] **Step 2: Run take tests and confirm the tiled assertion fails**

Run:

```bash
cargo test -p vortex-tiled-fsl take
```

Expected: generic execution does not return a tiled parent because no
encoding-specific take kernel is registered.

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

For an empty source, permit only zero output rows or all-null index rows.

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
cargo test -p vortex-tiled-fsl take
cargo test -p vortex-tiled-fsl
```

Expected: conformance and explicit preservation tests pass.

- [ ] **Step 6: Commit take support**

```bash
git add encodings/tiled-fsl
git commit -s -m "feat: preserve tiled lists through take"
```

---

### Task 5: FastLanes child composition

**Files:**
- Modify: `encodings/tiled-fsl/Cargo.toml`
- Modify: `encodings/tiled-fsl/src/tests.rs`

**Interfaces:**
- Consumes: `TiledFixedSizeList::{encode, try_new}` and slice/take from Tasks 2-4.
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

    let taken = tiled.into_array().take(
        PrimitiveArray::from_iter([64u32, 0, 32]).into_array(),
    )?;
    assert!(taken.is::<TiledFixedSizeList>());
    Ok(())
}
```

- [ ] **Step 2: Run the test and confirm the missing dev dependency**

Run:

```bash
cargo test -p vortex-tiled-fsl bitpacked_child
```

Expected: compilation fails because `vortex-fastlanes` is not a dependency of
the test target.

- [ ] **Step 3: Add the dev dependency and verify generic-child behavior**

Add `vortex-fastlanes = { workspace = true }` under `[dev-dependencies]`.
Run the new test. If it exposes an encoding assumption, keep `try_new`
validation limited to logical dtype and length: only `encode` and canonical
execution may require a canonical `PrimitiveArray`; `slice` and `take` call
generic child operations and accept whatever same-dtype encoding they return.

Do not add `vortex-fastlanes` to `[dependencies]`.

- [ ] **Step 4: Format and run composition plus full tests**

Run:

```bash
cargo +nightly fmt --all
cargo test -p vortex-tiled-fsl bitpacked_child
cargo test -p vortex-tiled-fsl
```

Expected: raw and bitpacked children both round-trip; parent slice/take remain
tiled.

- [ ] **Step 5: Commit the composition coverage**

```bash
git add encodings/tiled-fsl/Cargo.toml encodings/tiled-fsl/src/tests.rs Cargo.lock
git commit -s -m "test: cover bitpacked tiled list children"
```

---

### Task 6: Unstable file registration and facade re-export

**Files:**
- Modify: `vortex-file/Cargo.toml:35-95`
- Modify: `vortex-file/src/lib.rs:155-205`
- Create: `vortex-file/tests/tiled_fsl.rs`
- Modify: `vortex/Cargo.toml:20-105`
- Modify: `vortex/src/lib.rs:235-285`

**Interfaces:**
- Consumes: `vortex_tiled_fsl::initialize`.
- Produces: decoder registration under `vortex-file/unstable_encodings`.
- Produces: `vortex::encodings::tiled_fsl` under `vortex/unstable_encodings`.
- Proves: explicit experimental writer-edition file round-trip retains the tiled encoding and logical values.

- [ ] **Step 1: Add a failing feature-gated file round-trip test**

Create an integration test beginning with:

```rust
#![cfg(feature = "unstable_encodings")]
#![expect(clippy::tests_outside_test_module)]
```

Build a tiled FSL as a field in a one-field `StructArray`. Create a session with
`LayoutSession` and `RuntimeSession`, call
`vortex_file::register_default_encodings`, then call the existing test helper
`enable_all_registered_array_encodings` so the writer edition explicitly
permits the experimental ID. Write with `FlatLayoutStrategy` to avoid automatic
compression policy.

After scanning one chunk:

```rust
let result = chunk.execute::<StructArray>(&mut ctx)?;
let field = result.unmasked_field(0).clone();
assert!(field.is::<TiledFixedSizeList>());
assert_eq!(
    field.as_::<TiledFixedSizeList>().geometry(),
    geometry(32, 64)
);
assert_arrays_eq!(input, result, &mut ctx);
```

- [ ] **Step 2: Run the test and confirm registration is missing**

Run:

```bash
cargo test -p vortex-file --features unstable_encodings --test tiled_fsl
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

Do not declare inclusion in a production edition. The integration test's test
edition is the only new writer permission.

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
cargo test -p vortex-file --features unstable_encodings --test tiled_fsl
cargo check -p vortex-file --no-default-features
cargo check -p vortex --features unstable_encodings
```

Expected: the file test retains the encoding and values; feature-off builds do
not compile or register the new crate.

- [ ] **Step 6: Commit unstable integration**

```bash
git add Cargo.toml Cargo.lock vortex-file vortex
git commit -s -m "feat: register tiled lists as unstable encoding"
```

---

### Task 7: Canonical-FSL unit conformance suite

**Files:**
- Modify: `encodings/tiled-fsl/src/tests.rs`

**Interfaces:**
- Consumes: complete array, scalar, slice, take, tile, and FastLanes-composition APIs from Tasks 1-6.
- Consumes: `test_array_consistency` and `test_take_conformance` from Vortex's `_test-harness`.
- Uses: canonical `FixedSizeListArray` as the oracle; it never computes expected logical values through tiled helpers.
- Produces: deterministic regression coverage shared conceptually with the fuzz oracle in Task 8.

- [ ] **Step 1: Add the standard Vortex conformance checks**

For each representative tiled fixture, run:

```rust
test_array_consistency(&tiled.clone().into_array(), &mut ctx);
test_take_conformance(&tiled.clone().into_array(), &mut ctx);
```

Cover at least:

- nonnullable `u8`;
- nullable outer `i32`;
- nullable elements `f32`;
- nullable outer and nullable elements `f64`;
- empty rows;
- zero-width lists;
- row and dimension tails;
- tile sizes larger than both logical extents;
- one tiled parent with a bitpacked integer child.

The standard harness exercises cross-operation consistency and nullable-index
semantics. Keep the explicit oracle tests below because the standard harness
does not require slice/take to preserve this particular encoding.

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
3. every legal `execute_scalar(row)` equals canonical FSL;
4. `row_tile_count` and `dimension_tile_count` equal independent `div_ceil`
   calculations;
5. each `tile` and `tiles` element view equals a canonical child `take` built
   from `row * list_size + dimension`;
6. slices `0..0`, full range, each boundary-adjacent range, and ranges crossing
   row tiles equal canonical FSL and remain tiled with identical geometry;
7. empty, identity, reverse, duplicated, unsorted, and nullable takes equal
   canonical FSL and remain tiled with identical geometry.

Use the existing `assert_fsl_equivalent` helper after every operation:

```rust
let expected = canonical.clone().into_array().slice(range.clone())?;
let actual = tiled.clone().into_array().slice(range)?;
assert!(actual.is::<TiledFixedSizeList>());
assert_eq!(actual.as_::<TiledFixedSizeList>().geometry(), geometry);
assert_fsl_equivalent(&expected, &actual, &mut ctx)?;
```

For tile element views, derive expected indices directly from canonical
row-major coordinates; do not call `physical_offset`, `tile_bounds`, or another
tiled helper when constructing the oracle.

- [ ] **Step 3: Run the focused conformance suite**

Run:

```bash
cargo test -p vortex-tiled-fsl conformance
```

Expected: all standard consistency checks and canonical-oracle comparisons
pass. Treat any mismatch or lost tiled parent as a failing test requiring a
minimal implementation fix before continuing.

- [ ] **Step 4: Run the complete crate tests and commit**

Run:

```bash
cargo +nightly fmt --all
cargo test -p vortex-tiled-fsl
```

Then:

```bash
git add encodings/tiled-fsl
git commit -s -m "test: check tiled lists against canonical FSL"
```

---

### Task 8: Differential fuzzing against canonical FSL

**Files:**
- Modify: `fuzz/Cargo.toml`
- Modify: `fuzz/src/lib.rs`
- Create: `fuzz/src/tiled_fsl.rs`
- Create: `fuzz/fuzz_targets/tiled_fsl.rs`
- Modify: `fuzz/README.md`
- Modify: `.github/workflows/fuzz.yml`

**Interfaces:**
- Consumes: the complete tiled array API and Vortex's canonical FSL helpers.
- Produces: `FuzzTiledFsl`, `TiledFslAction`, and `run_tiled_fsl`.
- Produces: native cargo-fuzz target `tiled_fsl`.
- Checks: encode/execute, `try_new`, scalar access, tile descriptors and tile element views, slice, take, outer validity, element validity, empty arrays, zero-width lists, and composed operation sequences.

- [ ] **Step 1: Add a failing deterministic fuzz-harness smoke test**

Add `vortex-tiled-fsl = { workspace = true }` to `fuzz/Cargo.toml`, declare the
module and public re-exports in `fuzz/src/lib.rs`, and add a unit test that
calls the currently absent harness with fixed bytes:

```rust
#[cfg(test)]
mod tests {
    use arbitrary::Arbitrary;
    use arbitrary::Unstructured;

    use super::{FuzzTiledFsl, run_tiled_fsl};

    #[test]
    fn deterministic_tiled_fsl_smoke() {
        let zeros = [0u8; 256];
        let ones = [0xffu8; 256];
        let ascending = (0u8..=255).collect::<Vec<_>>();
        for seed in [
            zeros.as_slice(),
            ones.as_slice(),
            ascending.as_slice(),
        ] {
            let mut unstructured = Unstructured::new(seed);
            if let Ok(input) = FuzzTiledFsl::arbitrary(&mut unstructured) {
                run_tiled_fsl(input).unwrap();
            }
        }
    }
}
```

- [ ] **Step 2: Run the smoke test and confirm the harness is absent**

Run:

```bash
cargo test -p vortex-fuzz deterministic_tiled_fsl_smoke
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
- build a `DType::FixedSizeList(Primitive(...), list_size, outer_nullability)`;
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
4. compare every row's scalar with canonical FSL;
5. validate `row_tile_count` and `dimension_tile_count` against independent
   `div_ceil` calculations.

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

For every descriptor from `tiles()`:

- assert logical ranges do not overlap incorrectly and jointly cover every
  row/dimension tile;
- assert `physical_range.len() == row_range.len() * dimension_range.len()`;
- take `expected_tile_indices` from the canonical FSL element child;
- compare those expected values and element validity with `tile.elements()`;
- assert physical ranges are adjacent from zero through exactly
  `len * list_size`.

This oracle deliberately derives expected values from canonical row-major
coordinates and must not call `physical_offset` or `tile_bounds`.

- [ ] **Step 5: Differentially execute and compose every tiled operation**

Normalize action seeds against the current oracle length:

- `ScalarAt`: skip on empty arrays; otherwise use `seed % len` and compare
  canonical and tiled scalars.
- `Slice`: map both seeds into one valid `start..stop`, use
  `array::slice_canonical_array` for the oracle and `tiled.slice` for the
  candidate, assert logical equality, tiled encoding, and unchanged geometry.
- `Take`: map every valid seed modulo nonzero `len`, turn every valid seed into
  `None` when `len == 0`, use `array::take_canonical_array` for the oracle and
  `tiled.take` for the candidate, then assert equality, tiled encoding, and
  unchanged geometry.
- `Reconstruct`: call `try_new` with the current physical child and parts, then
  assert equality.
- `CheckTiles`: rerun the independent tile check from Step 4.

After every mutating action, run `assert_array_eq` and compare every legal
scalar. Return `Ok(false)` only for bytes that cannot generate the bounded
input; a logical mismatch is an error or panic so libFuzzer records it.

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
        Ok(true) => Corpus::Keep,
        Ok(false) => Corpus::Reject,
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
cargo test -p vortex-fuzz deterministic_tiled_fsl_smoke
cargo +nightly fuzz build --dev --sanitizer=none tiled_fsl
cargo +nightly fuzz run --dev --sanitizer=none tiled_fsl -- -runs=10000 -max_len=4096
```

Expected: the deterministic smoke test passes, the target builds, and 10,000
inputs produce no logical divergence, panic, sanitizer failure, or unbounded
allocation.

- [ ] **Step 8: Add the target to scheduled fuzzing**

Add a `tiled_fsl_fuzz` job to `.github/workflows/fuzz.yml` using
`./.github/workflows/run-fuzzer.yml`, target `tiled_fsl`, and four jobs. Add the
matching `report-tiled-fsl-fuzz-failures` job with the same permissions,
artifact naming, tokens, branch, and commit fields as the array-operations
reporter.

Use `tiled_fsl` consistently as target, fuzz name, corpus key, crash artifact
prefix, and logs prefix. Do not add `vortex/unstable_encodings`: the target
registers the direct encoding crate explicitly.

- [ ] **Step 9: Commit differential fuzzing**

```bash
git add fuzz .github/workflows/fuzz.yml Cargo.lock
git commit -s -m "test: fuzz tiled lists against canonical FSL"
```

---

### Task 9: Representation benchmarks

**Files:**
- Modify: `encodings/tiled-fsl/Cargo.toml`
- Create: `encodings/tiled-fsl/benches/tiled_fsl.rs`

**Interfaces:**
- Consumes: complete public encoding API.
- Produces: Divan benchmarks named `encode`, `execute`, `slice`, `take`, `traverse_prepared`, and `traverse_end_to_end`.
- Produces: benchmark arguments covering row tiles 32/64, full-width/64-dimension bands, and a 16-row by 4-dimension microtile.

- [ ] **Step 1: Add the benchmark target and compile-failing skeleton**

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
```

The argument set must include:

- rows 1,024 and 16,384;
- dimensions 128, 768, and 1,536;
- geometries 32-by-full, 64-by-full, 32-by-64, 64-by-64, and 16-by-4;
- boundary fixtures with rows/dimensions one below and one above 32 and 64.

Start with benchmark functions calling the not-yet-written fixture and traversal
helpers so the target fails to compile.

- [ ] **Step 2: Confirm the benchmark target fails before helper implementation**

Run:

```bash
cargo bench -p vortex-tiled-fsl --bench tiled_fsl --no-run
```

Expected: compilation fails on the missing fixture/traversal helpers.

- [ ] **Step 3: Implement deterministic raw and bitpacked fixtures**

Use deterministic low-bit `u8` values for bitpacked cases and `f32` values for
floating-point encode/execute cases. Construct inputs outside timed loops.
Initialize both `vortex_tiled_fsl` and `vortex_fastlanes` in the benchmark
session.

For bitpacked input, encode the already-transposed physical `u8` child at four
bits, then reconstruct with `try_new`.

- [ ] **Step 4: Implement operation benchmarks with honest preparation boundaries**

Measure:

- `encode`: canonical FSL to tiled;
- `execute`: tiled to canonical FSL;
- `slice`: a small, a half-width, and a tile-boundary-crossing row range;
- `take`: sorted sparse and unsorted duplicated row indices;
- `traverse_prepared`: execute the physical child outside the timed loop, then
  sum values through `array.tiles()` and each supplied `physical_range`;
- `traverse_end_to_end`: execute the raw or bitpacked physical child once
  inside each timed sample, then traverse the same public ranges.

The representative traversal body is:

```rust
fn sum_tiles(array: ArrayView<'_, TiledFixedSizeList>, values: &[u8]) -> u64 {
    array
        .tiles()
        .flat_map(|tile| values[tile.physical_range].iter())
        .map(|value| u64::from(*value))
        .sum()
}
```

Use `divan::black_box` on results. Do not call `tile.elements()` per tile in the
prepared benchmark; the benchmark exists to prove that the public physical
ranges support one-time child execution.

- [ ] **Step 5: Compile and smoke-run representative benchmark filters**

Run:

```bash
cargo +nightly fmt --all
cargo bench -p vortex-tiled-fsl --bench tiled_fsl --no-run
cargo bench -p vortex-tiled-fsl --bench tiled_fsl -- traverse_prepared
cargo bench -p vortex-tiled-fsl --bench tiled_fsl -- traverse_end_to_end
```

Expected: the target compiles and both raw/bitpacked traversal families produce
measurements for the candidate geometries.

- [ ] **Step 6: Commit benchmarks**

```bash
git add encodings/tiled-fsl/Cargo.toml encodings/tiled-fsl/benches Cargo.lock
git commit -s -m "bench: measure tiled fixed-size lists"
```

---

### Task 10: Final Vortex verification and implementation evidence

**Files:**
- Modify only if checks expose defects: files introduced or changed in Tasks 1-9.
- Do not create a SpiralDB path override in this task.

**Interfaces:**
- Consumes: complete Vortex implementation from Tasks 1-9.
- Produces: a clean, reviewable Vortex branch and benchmark command/output summary for the later SpiralDB plan.

- [ ] **Step 1: Run formatting and targeted tests**

Run:

```bash
cargo +nightly fmt --all -- --check
cargo test -p vortex-tiled-fsl
cargo test -p vortex-file --features unstable_encodings --test tiled_fsl
cargo test -p vortex --features unstable_encodings
cargo test -p vortex-fuzz deterministic_tiled_fsl_smoke
```

Expected: all pass.

- [ ] **Step 2: Run debug and release lints**

Run:

```bash
cargo clippy -p vortex-tiled-fsl --all-targets --all-features -- -D warnings
cargo clippy -p vortex-file --all-targets --features unstable_encodings -- -D warnings
cargo clippy -p vortex-fuzz --bin tiled_fsl --features native -- -D warnings
cargo clippy --release -p vortex-tiled-fsl --all-targets --all-features -- -D warnings
```

Expected: no warnings.

- [ ] **Step 3: Verify public documentation and dependency hygiene**

Run:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p vortex-tiled-fsl
taplo fmt --check
cargo shear
```

Expected: rustdoc, TOML formatting, and unused-dependency checks pass.

- [ ] **Step 4: Re-run benchmark compilation and capture representative evidence**

Run:

```bash
cargo bench -p vortex-tiled-fsl --bench tiled_fsl --no-run
cargo bench -p vortex-tiled-fsl --bench tiled_fsl -- traverse_prepared
cargo bench -p vortex-tiled-fsl --bench tiled_fsl -- traverse_end_to_end
cargo +nightly fuzz build --dev --sanitizer=none tiled_fsl
```

Record, without committing generated artifacts:

- prepared traversal throughput for 32/64-by-full and 32/64-by-64;
- end-to-end raw versus bitpacked traversal;
- encode and execute costs for 1,024-by-128 and at least one wider fixture;
- any geometry that is consistently dominated and can be dropped from the
  first SpiralDB experiment.

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

If verification required code changes:

```bash
git add Cargo.toml Cargo.lock encodings/tiled-fsl vortex-file vortex
git commit -s -m "fix: harden tiled fixed-size-list encoding"
```

If no files changed, do not create an empty commit.

- [ ] **Step 7: Hand off to the separate SpiralDB integration plan**

The next plan must point SpiralDB locally at `/Users/will/git/vortex`, add the
RaBitQ scoring kernel outside Vortex, differential-test every score kind and
code depth, benchmark the candidate geometries against the scalar scorer, and
remove the path override before the dependent SpiralDB merge. Do not start that
work until this Vortex API and its benchmark evidence have passed review.
