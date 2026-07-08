# FastLanes Encodings — Byte Layout

The **FastLanes** family is the SIMD-oriented integer-compression group: bit-packing,
frame-of-reference, delta, and run-length encodings that share the FastLanes block structure. This
page is the per-encoding byte-layout reference for that family, one section per encoding.

It is part of the [Encoding Format](../encoding-format.md) specification; null handling for every
encoding here follows the cross-cutting [Validity](../encoding-format.md#validity) contract on that
page and is not restated per section.

Encodings covered on this page: `BitPacked`, `FoR`, `Delta`, `RLE`.

Throughout, `n` is the node's logical length (the row count carried by the array container, after any
slicing).

## FastLanes fundamentals

All four encodings on this page share the FastLanes block machinery. This section defines it once; the
per-encoding sections reference it. It is precise enough to reimplement.

### Blocks of 1024

Data is processed in fixed **blocks of 1024 values** (`FL_CHUNK_SIZE = 1024`, `lib.rs` `FL_CHUNK_SIZE`). A writer
splits the input into consecutive 1024-value blocks and **zero-pads the final block** up to 1024. All
stored child/buffer lengths are therefore multiples of 1024 (or of a per-block count derived from it),
and the array's logical length `n` — plus, when sliced, an `offset` — selects the live window out of
that padded extent.

### Element width and lanes

Let `T` be the **unsigned** integer type whose width matches the logical primitive type (`u8` for
`i8`/`u8`, `u16` for `i16`/`u16`, … `u64` for `i64`/`u64`). Write `T_BITS` for its bit width
(8/16/32/64) and define

```text
LANES = 1024 / T_BITS        # u8→128, u16→64, u32→32, u64→16
```

(`fastlanes` 0.5.1 `lib.rs` `FastLanes`). A block is viewed as `LANES` interleaved **lanes**, each holding
`T_BITS` values.

### The order permutation `FL_ORDER`

```text
FL_ORDER = [0, 4, 2, 6, 1, 5, 3, 7]        # an involution: FL_ORDER[FL_ORDER[k]] == k
```

(`fastlanes` 0.5.1 `lib.rs` `FL_ORDER`).

(fastlanes-packing-order)=
### Bit-packing layout (`W` bits per value)

Bit-packing stores one block of 1024 values at a uniform width `W` bits each (`0 ≤ W ≤ T_BITS`),
occupying exactly `1024·W` bits = `128·W` bytes. View that region as `1024·W / T_BITS` words of type
`T`, laid out little-endian in memory. Index the words as `word[w·LANES + lane]` for `w` in `0..W`,
`lane` in `0..LANES`.

For a fixed `lane`, concatenate its `W` words from `w = 0` (least-significant) to `w = W−1`
(most-significant), each contributing `T_BITS` bits low-bit-first, to form that lane's `W·T_BITS`-bit
stream. The value for **row** `r` (`r` in `0..T_BITS`) is the `W`-bit field at bit offset `r·W` in the
lane stream. That value's position **within the block's logical (row) order** is

```text
pack_index(r, lane) = FL_ORDER[r / 8] * 16 + (r % 8) * 128 + lane
```

(`fastlanes` 0.5.1 `macros.rs` `index`, the `unpack!` kernel `macros.rs` `index`). `pack_index` is a
bijection of `{0..T_BITS} × {0..LANES}` onto `0..1024`. To unpack a block directly into logical order:

```text
for lane in 0 .. LANES:
    for r in 0 .. T_BITS:
        v = W-bit field at bit offset r*W of lane `lane`'s stream
        block[pack_index(r, lane)] = v
```

Special cases: `W = 0` ⇒ the region is empty and every value is `0`; `W = T_BITS` ⇒ word
`word[r·LANES + lane]` holds `block[pack_index(r, lane)]` verbatim. Unpacking yields **row order** —
the internal transposition is invisible in the output of plain bit-packing. (Values are unpacked as
unsigned `T`; reinterpreting the block as a signed logical type is a no-op on the bit pattern, valid
because packed values are non-negative and any out-of-range value is carried as a patch.)

(fastlanes-transpose)=
### The 1024-element transpose

Delta stores its values in a **transposed** block order that a reader must invert. The permutation is

```text
transpose(i) = (i % 16) * 64 + FL_ORDER[(i / 16) % 8] * 8 + (i / 128)      # i in 0..1024
```

(`fastlanes` 0.5.1 `transpose.rs` `transpose`). The forward transform is `output[i] = input[transpose(i)]`;
the inverse (**untranspose**) is

```text
untranspose(input)[transpose(i)] = input[i]     for i in 0..1024
```

(`transpose.rs` `Transpose`). The same permutation applied at bit granularity is used to untranspose Delta's
validity bitmap; that mechanism is specified in [Validity](../encoding-format.md#validity) and is not
restated here.

(encoding-layout-BitPacked)=
## `fastlanes.bitpacked` (BitPacked) — byte layout

BitPacked stores an integer column at a reduced, uniform bit width `W`, using the FastLanes
[bit-packing layout](#fastlanes-packing-order). Values too wide to fit in `W` bits (and, for signed
columns, any that would not round-trip) are carried verbatim as **patches**.

**Wire ID:** `fastlanes.bitpacked` (`bitpacking/vtable/mod.rs` `BitPacked::id`). **Logical dtype:** any integer
primitive (`i8`…`i64`, `u8`…`u64`).

**Buffers:** exactly 1 — buffer 0, named `packed` (`bitpacking/vtable/mod.rs` `BitPacked::buffer_name`). Its length is
`ceil((n + offset) / 1024) * 128 * W` bytes (`bitpacking/array/mod.rs` `BitPackedData::validate`) — one `128·W`-byte
[block](#fastlanes-packing-order) per 1024 padded values. `W = 0` ⇒ empty buffer.

### Metadata (`BitPackedMetadata`, Protobuf)

| Field | Protobuf type / tag | Meaning |
|-------|---------------------|---------|
| `bit_width` | `uint32`, tag 1 | The pack width `W` (`0 ≤ W ≤ 64`; always `< T_BITS` for a genuinely packed column). Stored as `u32`, narrowed to `u8` (`bitpacking/vtable/mod.rs` `BitPacked::deserialize`). |
| `offset` | `uint32`, tag 2 | Physical start position within the **first** block, `0 ≤ offset < 1024` (`bitpacking/array/mod.rs` `BitPackedData.offset`, `BitPackedData::try_new`). Applies to the `packed` buffer **only** — see below. |
| `patches` | `PatchesMetadata`, tag 3, optional | Present iff the column has exceptions. Describes the patch children ([shared patch layout](#fastlanes-patch-metadata)). |

### Child slots

Slots are positional; the validity slot's index depends on whether patches (and patch chunk-offsets)
are present (`bitpacking/vtable/mod.rs` `BitPacked::deserialize`, `bitpacking/array/mod.rs` `BitPackedSlots`):

| Idx | Name | Present when | Logical dtype |
|-----|------|--------------|---------------|
| 0 | `patch_indices` | `patches` set | non-nullable unsigned int (ptype from `PatchesMetadata`) |
| 1 | `patch_values` | `patches` set | the node's own integer dtype (the true exception values; **all-valid**) |
| 2 | `patch_chunk_offsets` | `patches` set **and** `chunk_offsets_ptype` present | non-nullable unsigned int (O(1) lookup accelerator) |
| *k* | `validity_child` | **only when the array stores per-position validity** — omitted for `NonNullable`/`AllValid` per the [Validity](../encoding-format.md#validity) contract | non-nullable `Bool`; when present it is the **last** slot, at index `k = 0` (no patches), `2` (patches, no chunk offsets), or `3` (patches + chunk offsets) |

`patch_values` is decoded recursively like any node; it holds the real values and is never bit-packed
against `W`.

### The `offset` (most common mistake)

`offset` is a physical cursor into the first packed block, produced by slicing: a slice of `[a..b]`
sets `offset = (old_offset + a) % 1024` and keeps only the covering blocks (`bitpacking/compute/slice.rs` `slice_bitpacked`).
It is applied to the **packed buffer only** and is **never** applied to the validity slot, which stays
row-aligned. Over-applying it to validity corrupts every null after the slice — see the warning in
[Validity](../encoding-format.md#validity).

### Decode

Given `packed`, `W`, `offset`, `n`, and optional patches:

1. **Unpack blocks.** `num_blocks = ceil((offset + n) / 1024)`. Unpack each block into 1024 logical
   (row-order) values per [the bit-packing layout](#fastlanes-packing-order), concatenating into a
   padded buffer of `num_blocks · 1024` values (`bitpacking/array/unpack_iter.rs`,
   `bitpacking/array/bitpack_decompress.rs` `unpack_into_primitive_builder`).
2. **Take the window.** Keep logical positions `offset .. offset + n`.
3. **Reinterpret** the unsigned block values as the node's integer ptype (identity on the bits).
4. **Apply patches** (if present). For each patch `j` in `0 .. len`, overwrite
   ```text
   value[ patch_indices[j] − patch_offset ] = patch_values[j]
   ```
   where `patch_offset` is the patches' **own** `offset` field ([shared patch layout](#fastlanes-patch-metadata)) —
   distinct from the block `offset` above (`bitpacking/array/bitpack_decompress.rs` `apply_patches_to_uninit_range`). A patch
   position holds a meaningless unpacked placeholder until this step overwrites it.
5. **Validity** is a stored, row-aligned slot; see [Validity](../encoding-format.md#validity).

**Example (no patches, no slice).** `W = 3`, `offset = 0`, `n = 4`, a single block whose row-order
values are `[5, 0, 1, 7, 0, …]` decodes to `[5, 0, 1, 7]`.

(encoding-layout-FoR)=
## `fastlanes.for` (FoR) — byte layout

Frame-of-Reference stores each value as its offset from a single **reference** integer (typically the
column minimum): `value[i] = encoded[i] + reference`. Subtracting a common base shrinks the magnitudes
so the `encoded` child bit-packs to a narrow width.

**Wire ID:** `fastlanes.for` (`for/vtable/mod.rs` `FoR::id`). **Logical dtype:** any integer primitive.

**Buffers:** none (`nbuffers = 0`, `for/vtable/mod.rs` `FoR::nbuffers`).

### Metadata (raw `ScalarValue`, Protobuf)

FoR's metadata is **not** a wrapper message. It is the reference scalar's **value** alone, serialized
as `ScalarValue` protobuf bytes **without** its dtype (`for/vtable/mod.rs` `FoR::serialize`). On read, decode
the bytes as a `ScalarValue` against the node's own `DType` to recover the reference
(`for/vtable/mod.rs` `FoR::deserialize`). The reference is non-null and shares the node's integer type
(`for/array/mod.rs` `FoRData::try_new`).

### Child slots

| Idx | Name | Present when | Logical dtype | Length |
|-----|------|--------------|---------------|--------|
| 0 | `encoded` | always | the node's own integer dtype | `n` |

`encoded` is decoded recursively; it is commonly (but not necessarily) `fastlanes.bitpacked`
(`for/vtable/mod.rs` `FoR::deserialize`).

### Decode

1. Decode `encoded` to an integer array of length `n`, and decode the `reference` scalar from metadata.
2. For each position `i`: `value[i] = encoded[i] wrapping_add reference` — modular (wrapping) integer
   addition (`for/array/for_decompress.rs` `decompress`, `decompress_primitive`). If `reference == 0`, `encoded` is the
   result unchanged.
3. **Validity** is delegated to the `encoded` child; see [Validity](../encoding-format.md#validity).

Wrapping addition means the reference may be added correctly even when the true values wrap the type;
the reference/encoded split is purely arithmetic.

**Example.** `reference = 100`, `encoded = [0, 5, 3]` ⇒ `[100, 105, 103]`.

(encoding-layout-Delta)=
## `fastlanes.delta` (Delta) — byte layout

Delta stores per-lane running differences over a **transposed** block, so each lane holds a contiguous
run of the original sequence and the deltas within a lane are small. Reconstructing a block is a
per-lane prefix sum followed by an [untranspose](#fastlanes-transpose).

**Wire ID:** `fastlanes.delta` (`delta/vtable/mod.rs` `Delta::id`). **Logical dtype:** any integer primitive.

**Buffers:** none (`nbuffers = 0`, `delta/vtable/mod.rs` `Delta::nbuffers`).

### Metadata (`DeltaMetadata`, Protobuf)

| Field | Protobuf type / tag | Meaning |
|-------|---------------------|---------|
| `deltas_len` | `uint64`, tag 1 | Length of the `deltas` child; always a multiple of 1024 (`delta/vtable/mod.rs` `validate_parts`). |
| `offset` | `uint32`, tag 2 | Physical start position within the first block, `0 ≤ offset < 1024` (`delta/array/mod.rs` `DeltaData::try_new`). |

### Child slots

Both children share the node's integer type; the node's nullability comes from `deltas`
(`delta/vtable/mod.rs` `Delta::try_new`). Let `num_blocks = deltas_len / 1024`.

| Idx | Name | Logical dtype | Length |
|-----|------|---------------|--------|
| 0 | `bases` | the node's integer dtype (node nullability; always all-valid — validity lives in `deltas`) | `num_blocks · LANES` (`delta/vtable/mod.rs` `Delta::deserialize`) |
| 1 | `deltas` | the node's integer dtype (carries validity) | `deltas_len` |

`deltas` holds the per-lane differences in [bit-packing (`pack_index`) order](#fastlanes-packing-order),
padded to whole 1024-blocks; `bases[c·LANES + lane]` is the seed value for lane `lane` of block `c`
(the first transposed element of that lane; the corresponding stored delta is `0`). Both are decoded
recursively (`deltas` is commonly `fastlanes.bitpacked`).

### Decode

Reinterpret `bases` and `deltas` as the **unsigned** type `T` for the arithmetic, then per block
`c` in `0 .. num_blocks` (`delta/array/delta_decompress.rs` `decompress_primitive`; `fastlanes` 0.5.1
`delta.rs` `Delta::undelta`):

```text
# 1. per-lane prefix sum, in pack_index order (produces a transposed block)
for lane in 0 .. LANES:
    acc = bases[c*LANES + lane]
    for r in 0 .. T_BITS:
        idx = pack_index(r, lane)                 # see FastLanes fundamentals
        acc = acc wrapping_add deltas[c*1024 + idx]
        trans[idx] = acc

# 2. untranspose the block back into row order
for i in 0 .. 1024:
    out[c*1024 + transpose(i)] = trans[i]         # see FastLanes fundamentals
```

Then reinterpret `out` back to the node's (possibly signed) ptype and **slice**
`out[offset .. offset + n]` (`delta/array/delta_decompress.rs` `delta_decompress`). Skipping either the
untranspose or the offset slice yields scrambled or shifted values.

**Validity** is delegated to `deltas`, then untransposed and sliced `offset .. offset + n`; see
[Validity](../encoding-format.md#validity).

:::{note}
A worked numeric example spans a full 1024-value block, so it is impractical to tabulate here. The two
permutation functions (`pack_index`, `transpose`) plus the per-lane `wrapping_add` recurrence above are
a complete, self-contained reconstruction. As a spot check of `pack_index` for `T = u16`
(`T_BITS = 16`, `LANES = 64`): `pack_index(0, lane) = lane`, and `pack_index(1, 0) = 128`.
:::

(encoding-layout-RLE)=
## `fastlanes.rle` (RLE) — byte layout

FastLanes run-length encoding is **per-block**: within each 1024-value block, it stores that block's
run values and, for each of the 1024 rows, a **chunk-local index** into those run values. (This is
distinct from `vortex.runend`, a combine encoding on the [Dict/RunEnd/Sparse page](dict-runend-sparse.md).)

**Wire ID:** `fastlanes.rle` (`rle/vtable/mod.rs` `RLE::id`). **Logical dtype:** integer primitive; the ptype
comes from `values`, the nullability from `indices` (`rle/vtable/mod.rs` `RLE::try_new`).

**Buffers:** none (`nbuffers = 0`, `rle/vtable/mod.rs` `RLE::nbuffers`).

### Metadata (`RLEMetadata`, Protobuf)

| Field | Protobuf type / tag | Meaning |
|-------|---------------------|---------|
| `values_len` | `uint64`, tag 1 | Length of the `values` child (total run values across all blocks). |
| `indices_len` | `uint64`, tag 2 | Length of the `indices` child; a multiple of 1024 (`rle/vtable/mod.rs` `validate_parts`). |
| `indices_ptype` | `PType` enum, tag 3 | Ptype of `indices` (`u8` or `u16`). |
| `values_idx_offsets_len` | `uint64`, tag 4 | Length of the `values_idx_offsets` child = `indices_len / 1024` (one per block). |
| `values_idx_offsets_ptype` | `PType` enum, tag 5 | Ptype of `values_idx_offsets` (unsigned int). |
| `offset` | `uint64`, tag 6, default 0 | Physical start position within the first block, `0 ≤ offset < 1024` (`rle/array/mod.rs` `RLEData::try_new`). |

### Child slots

| Idx | Name | Logical dtype | Length |
|-----|------|---------------|--------|
| 0 | `values` | `Primitive(ptype, NonNullable)` — the run values | `values_len` |
| 1 | `indices` | `Primitive(u8 or u16, node nullability)` — per-row chunk-local run index (**carries validity**) | `indices_len` |
| 2 | `values_idx_offsets` | non-nullable unsigned int — start index into `values` for each block | `indices_len / 1024` |

The `values_idx_offsets` entries are **absolute** offsets into `values`; a reader rebases them by
subtracting `values_idx_offsets[0]`, because slicing may drop leading blocks and re-slice `values`
without rewriting the offsets (`rle/array/mod.rs` `RLEArrayExt::values_idx_offset`, `rle/kernel.rs` `RLE::slice`).

### Decode

Decode `values` → `V`, `indices` → `I` (u8/u16), `values_idx_offsets` → `O`. If `n == 0` the array
is empty (`O`, `I`, and `V` are all empty) — return the empty array **without** reading `O[0]`.
Otherwise let `num_blocks = ceil((offset + n) / 1024)` and `base = O[0]`
(`rle/array/rle_decompress.rs` `rle_decode_typed`):

```text
for b in 0 .. num_blocks:
    start = O[b] - base
    end   = (b + 1 < num_blocks) ? (O[b+1] - base) : V.len()   # exclusive; run values for block b
    for row in 0 .. 1024:
        out[b*1024 + row] = V[start + I[b*1024 + row]]
# then take the live window:
result = out[offset .. offset + n]
```

Each block's run values are `V[start .. end]`; row `row` of block `b` picks run value
`V[start + I[b*1024 + row]]`.

**Robustness at null positions.** Where `indices` is null, its stored index value is a don't-care and
may be out of range (a byproduct of further-compressing the indices). Because RLE's validity is
sourced from `indices`, such rows are null regardless of the value read; a defensive reader clamps the
index into `0 .. (end − start)` before the lookup to avoid an out-of-bounds access
(`rle/array/rle_decompress.rs` `rle_decode_typed`).

**Validity** is delegated to `indices` sliced `offset .. offset + n` *before* taking its validity; see
[Validity](../encoding-format.md#validity).

**Example (single block).** `V = [10, 20, 30]`, `O = [0]`, `offset = 0`, `n = 5`, and
`I[0..5] = [0, 0, 1, 2, 2]` ⇒ `[10, 10, 20, 30, 30]`.

(fastlanes-patch-metadata)=
## Shared patch layout

`fastlanes.bitpacked` reuses the workspace-wide `PatchesMetadata` message (`patches.rs` `PatchesMetadata`) and its
three patch children.

| Field | Protobuf type / tag | Meaning |
|-------|---------------------|---------|
| `len` | `uint64`, tag 1 | Number of patches `P` = length of both `patch_indices` and `patch_values`. |
| `offset` | `uint64`, tag 2 | Absolute offset subtracted from every stored index; `0` for an unsliced array. Array-local position = `patch_indices[j] − offset`. |
| `indices_ptype` | `PType` enum, tag 3 | Unsigned-int ptype of `patch_indices`. |
| `chunk_offsets_len` | `uint64`, tag 4, optional | Length of `patch_chunk_offsets`, when present. |
| `chunk_offsets_ptype` | `PType` enum, tag 5, optional | Ptype of `patch_chunk_offsets`; its presence signals child slot 2 exists. |
| `offset_within_chunk` | `uint64`, tag 6, optional | Rows sliced off the front of the first chunk (slice bookkeeping). |

`patch_indices` is a strictly-ascending non-nullable unsigned-int array of length `P`; `patch_values`
(length `P`, non-null) holds the true value for each. `patch_chunk_offsets` is an **optional O(1)
lookup index** (one entry per 1024-row chunk, `PATCH_CHUNK_SIZE = 1024`, `patches.rs` `PATCH_CHUNK_SIZE`); a reader
that scans all `P` patches linearly may ignore it.
