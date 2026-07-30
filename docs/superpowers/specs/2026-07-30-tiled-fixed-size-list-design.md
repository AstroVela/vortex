# Tiled Fixed-Size List Encoding

**Status:** Approved for implementation planning

**Date:** 2026-07-30

## Summary

Add an experimental, scheme-neutral Vortex encoding for fixed-size lists whose
primitive elements are stored in two-dimensional tiles. The encoding makes
small blocks of dimensions across small blocks of rows contiguous so specialized
compute kernels can consume SIMD-friendly data without first materializing a
row-major `FixedSizeListArray`.

The encoding belongs in a dedicated `encodings/tiled-fsl` crate rather than in
SpiralDB or a quantizer-specific crate. SpiralDB will initially use it for
RaBitQ scoring, but neither the representation nor its public API assumes
RaBitQ, TurboQuant, or any particular vector index.

The first version is explicitly selected by its caller and remains behind
Vortex's `unstable_encodings` feature. It is not selected by the default
compressor and does not carry a stable wire-format promise.

## Motivation

Canonical Vortex fixed-size lists store every row contiguously. That is a good
general representation, but it is a poor match for vector kernels that evaluate
the same dimensions across a block of candidate rows. Those kernels otherwise
gather or transpose data in their hot loop.

RaBitQ makes this particularly visible. The existing SpiralDB implementation is
logically correct and composes through Vortex arrays, but its scalar scorer is
substantially slower than specialized reference implementations. A vertically
or partially vertically arranged physical representation is necessary to test
how much of that difference can be recovered by specialized compute.

The same data-layout problem is not specific to RaBitQ. TurboQuant, future
vector indexes, and other fixed-width matrix-like workloads may want different
row and dimension tile shapes. The representation must therefore retain both
axes as explicit geometry rather than hard-code one layout discovered on one
machine.

## Goals

- Represent an ordinary logical `FixedSizeList<Primitive>` using a tiled
  physical primitive child.
- Support independent, nonzero row and dimension tile sizes.
- Keep every rectangular tile contiguous.
- Preserve both row-level and element-level validity.
- Avoid serialized padding for tail tiles.
- Preserve the tiled parent encoding across `slice` and `take`.
- Allow the physical child to use another Vortex encoding such as FastLanes.
- Expose enough tile structure for downstream specialized compute without
  embedding a particular compute algorithm in this crate.
- Validate the representation against SpiralDB's RaBitQ scorer before making it
  a default there.

## Non-goals

- Supporting boolean, variable-width, or nested element dtypes in the first
  version.
- Implementing RaBitQ, TurboQuant, distance scoring, or vector-index policy in
  Vortex.
- Teaching BtrBlocks or another default compressor to select the encoding.
- Choosing one permanent tile geometry.
- Promising a stable encoding ID, metadata schema, or file compatibility while
  the encoding is experimental.
- Adding a specialized `filter` implementation in the first version. Generic
  fallback remains available; a preserving implementation can be added from
  measured need.
- Solving incremental vector-index maintenance or IVF clustering.

## Alternatives considered

### Dedicated scheme-neutral encoding crate

Place the encoding in `encodings/tiled-fsl`, with package name
`vortex-tiled-fsl`. The crate owns only the representation and general Vortex
operations.

This is the selected approach. It gives RaBitQ and TurboQuant a common building
block without coupling Vortex to either algorithm, and it lets the physical
primitive child be compressed independently.

### FastLanes-specific tiled encoding

Combine tiling and bitpacking in one encoding. This could simplify one initial
RaBitQ layout, but it would prevent raw, constant, or future primitive child
encodings and would make the outer representation needlessly scheme-specific.

This is rejected. Tiling describes element order; FastLanes describes how the
ordered primitive stream is encoded. Vortex can compose the two.

### File-layout-only transposition

Keep the in-memory array canonical and transpose only in file layout metadata.
This avoids a new array encoding, but specialized compute could not reliably
downcast to and traverse the representation. Slice, take, and execution would
also lack one explicit contract.

This is rejected because the tiled arrangement is a physical array
representation with useful compute semantics, not merely file placement.

## Logical and physical model

The logical dtype remains:

```text
FixedSizeList<Primitive>
```

For a logical array with `len` rows and `list_size` dimensions, the physical
primitive child contains exactly `len * list_size` elements. It contains the
same logical element values as the canonical child, but in tiled order.

The geometry is:

```rust
pub struct TileGeometry {
    pub rows: NonZeroU32,
    pub dimensions: NonZeroU32,
}
```

The physical traversal order is fixed:

```text
dimension tile
  -> row tile
    -> dimension within tile
      -> row within tile
```

Each rectangular tile is consequently one contiguous range in the physical
child. Dimension tiles are the outermost bands. Within a dimension band, row
tiles are consecutive. Values within a tile are vertical by dimension.

For logical row `r`, dimension `d`, row tile height `R`, and dimension tile
width `D`:

```text
dimension_start = floor(d / D) * D
row_start       = floor(r / R) * R
dimension_width = min(D, list_size - dimension_start)
row_height      = min(R, len - row_start)

physical_offset =
    dimension_start * len
    + row_start * dimension_width
    + (d - dimension_start) * row_height
    + (r - row_start)
```

All arithmetic used to construct or index the representation is checked.

### Example

For three rows, five dimensions, and geometry `{ rows: 2, dimensions: 3 }`,
name the canonical values by row and dimension:

```text
row 0: 00 01 02 03 04
row 1: 10 11 12 13 14
row 2: 20 21 22 23 24
```

The physical child is:

```text
00 10 01 11 02 12 | 20 21 22 | 03 13 04 14 | 23 24
```

The separators show the four rectangular tiles. Neither the short row tail nor
the short dimension tail is padded.

This geometry can express several useful families without changing the
encoding:

- original PDX-like vertical row blocks:
  `{ rows: 64, dimensions: list_size }`;
- dimension-banded blocks:
  `{ rows: 32 or 64, dimensions: 64 }`;
- small instruction-oriented microtiles:
  `{ rows: 16, dimensions: 4 }`.

## Validity

Row-level fixed-size-list validity remains a separate validity child with
logical length `len`.

Element validity belongs to the primitive element dtype and is transposed in
lockstep with element values. The physical primitive child must have exactly the
element dtype declared by the logical fixed-size-list dtype, including
nullability.

Null parent rows do not remove or shorten their physical elements, matching the
canonical fixed-size-list representation.

## Representation and invariants

The proposed array type is `TiledFixedSizeList`, with experimental encoding ID
`vortex.tiled_fsl`.

Construction validates:

- the logical dtype is `DType::FixedSizeList`;
- its element dtype is `DType::Primitive`;
- row and dimension tile sizes are nonzero;
- the physical child dtype exactly matches the logical element dtype;
- the physical child length is exactly `len.checked_mul(list_size)`;
- row validity has logical length `len`;
- all sizes and offsets are representable without overflow.

`len == 0` and `list_size == 0` are valid. A tile size may exceed its logical
extent, producing at most one short tile on that axis. A zero-width logical list
has an empty physical child but still retains its requested nonzero geometry.

No tail padding is stored or included in physical child length.

## Public API

The public surface should be small and follow established Vortex array
conventions. Exact names may adjust during implementation to match generated
array APIs, but the semantic surface is:

```rust
TiledFixedSizeList::encode(fsl, geometry, ctx)
TiledFixedSizeList::try_new(elements, dtype, len, geometry, validity)

array.geometry()
array.elements()
array.validity()
array.row_tile_count()
array.dimension_tile_count()
array.tile(row_tile, dimension_tile)
```

`encode` is the normal construction path. It executes the canonical primitive
child once and transposes values and element validity in bulk.

`try_new` validates structural invariants. Like other encoded-array
constructors, it cannot prove that caller-provided physical values are the
correct transpose.

`tile` returns a view containing the logical row range, logical dimension
range, and the corresponding contiguous physical element range. Tile traversal
does not know about RaBitQ scoring or any other consuming algorithm.

There is no `with_elements` convenience method. A caller that wants a
FastLanes-compressed physical child obtains `elements()`, compresses that
unchanged physical sequence, and reconstructs the tiled array through
`try_new`. This keeps composition explicit and avoids implying that an
arbitrary replacement child preserves semantics.

## Execution

Executing the array to its canonical form:

1. executes the physical primitive child once;
2. allocates an ordinary row-major primitive output;
3. inverse-transposes values and element validity by tile;
4. constructs a canonical `FixedSizeListArray` with the unchanged row
   validity.

The work is proportional to the number of logical elements and uses typed,
bulk access rather than per-element scalar execution.

## Slice and take

`slice` and `take` return `TiledFixedSizeList` with the same geometry.

They generate the selected rows directly in output tiled order and gather the
corresponding physical child elements. They do not canonicalize the entire
source first. Work and temporary index storage are proportional to output
elements, not source elements.

Because dimension bands are outermost, even a row-tile-aligned row slice spans
multiple disjoint physical bands. The first implementation therefore does not
promise zero-copy slices. The parent remains tiled, while the encoding returned
for its physical child depends on the child `take` implementation. Callers may
recompress that child when appropriate.

`take` follows Vortex's ordinary ordering, duplicate, nullable-index, and bounds
semantics. Output row validity follows the selected logical rows. Physical
element placeholders for null indices follow the canonical fixed-size-list
take contract and do not alter the null output row's semantics.

A future chunked child or layout may optimize aligned bands without changing
the tiled array contract.

## File integration and feature gating

The new crate is added to the workspace and registered with Vortex file
sessions only when `unstable_encodings` is enabled. The top-level `vortex`
crate may re-export it under the same feature.

A file writer may emit this encoding only when its input already contains a
tiled array and the session's writer-edition policy permits the encoding. Tests
and local SpiralDB validation explicitly enable an experimental edition; this
work does not silently widen a production writer edition or add the encoding to
a default compression strategy.

Experimental readers and writers must round-trip files exactly when the feature
is enabled. Files are not promised to remain readable after changes to the
experimental metadata or layout. There is no algorithm, arithmetic, padding,
or layout-version field in the initial metadata; those can be added if and when
the unstable encoding evolves toward a stable contract.

## Specialized compute boundary

Vortex provides representation-level tile traversal. SpiralDB owns the first
RaBitQ scoring kernels:

```text
TiledFixedSizeList
        |
        +-- generic Vortex execute/slice/take
        |
        +-- SpiralDB RaBitQ scorer
        |
        +-- possible future TurboQuant or index kernels
```

The scorer may downcast to `TiledFixedSizeList`, inspect its geometry, and
consume its tiles directly. It must not depend on private child-offset
knowledge that bypasses the public tile contract.

Different RaBitQ components may choose different geometries. In particular,
sign codes and extra magnitude bits need not share one row/dimension tile
shape.

## Correctness testing

Vortex tests include:

- the hand-calculated physical-layout example above;
- round-trips across every fixed-width primitive type;
- independent row and element nullability;
- empty arrays and zero-width lists;
- row and dimension counts immediately below, at, and above tile boundaries;
- row and dimension tail tiles;
- tile sizes larger than logical extents;
- ordered, duplicated, unsorted, and nullable take indices;
- bounds failures and preservation of geometry through slice and take;
- malformed logical dtypes, child dtypes, child lengths, validity lengths,
  zero geometry, and overflow;
- composition with a raw primitive child and, for supported integer dtypes, a
  FastLanes-compressed child;
- experimental Vortex file round-trips.

Property-style differential tests compare canonical values before encoding,
after execution, and after slice/take combinations.

## Benchmarks

Vortex benchmarks measure:

- canonical-to-tiled encoding;
- tiled-to-canonical execution;
- slice;
- take;
- direct tile traversal through a small representative arithmetic kernel.

The benchmark matrix includes:

- 32- and 64-row tiles;
- full-width dimension tiles;
- 64-dimension bands;
- at least one microtile such as 4 dimensions by 16 rows;
- representative widths such as 128, 768, and 1536;
- representative row counts such as 1,024 and 16,384;
- sizes immediately around tile boundaries;
- raw primitive children and FastLanes-compressed children for supported
  integer dtypes.

These benchmarks characterize the representation; they do not select a global
default.

## SpiralDB validation

Before either repository relies on the new representation, SpiralDB's local
Vortex dependencies point at the Vortex checkout. SpiralDB then:

1. stores RaBitQ fixed-size-list code components in the tiled representation;
2. implements specialized scorers outside Vortex;
3. differential-tests every score kind and code depth against the existing
   scalar scorer;
4. benchmarks sign-only and multibit scoring across the candidate geometries;
5. measures end-to-end scoring as well as isolated tile traversal, encoding
   cost, and file size.

A tiled geometry becomes a SpiralDB default only if it:

- remains logically identical after Vortex execution and file round-trip;
- passes differential scoring;
- materially improves end-to-end scoring;
- avoids unacceptable encoding, slicing, or storage-size regressions.

The experiment may choose different layouts for different RaBitQ components or
decide that some components should remain canonical.

## Delivery sequence

1. Implement and validate the experimental Vortex encoding.
2. Point SpiralDB locally at the Vortex checkout.
3. Implement and benchmark the SpiralDB RaBitQ tiled scorer.
4. Open and merge the Vortex PR.
5. Update SpiralDB to the merged Vortex revision and merge the dependent
   SpiralDB work.

The local path override is validation scaffolding and is not committed as the
long-term SpiralDB dependency.

## References

- [Product Quantization with Duality-based Encoding (PDX)](https://arxiv.org/abs/2503.04422)
- [CWI PDX data-layout notes](https://github.com/cwida/PDX#the-data-layout)
