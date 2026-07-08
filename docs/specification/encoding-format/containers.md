# Container Encodings — Byte Layout

The **container** encodings are the nested and structural forms — structs, the list variants, and
the wrapper encodings that compose, concatenate, extend, or mask other arrays. This page is the
per-encoding byte-layout reference for that family, one section per encoding.

It is part of the [Encoding Format](../encoding-format.md) specification; null handling for every
encoding here follows the cross-cutting [Validity](../encoding-format.md#validity) contract on that
page and is not restated per section.

Encodings covered on this page: `Struct`, `List`, `ListView`, `FixedSizeList`, `Chunked`,
`Extension`, `Variant`, `Masked`.

```{contents}
:local:
:depth: 1
```

## Common structure for this family

Every encoding on this page shares three structural conventions. They are stated once here and
assumed by each section below.

**No self-describing dtype or length.** A serialized array node does **not** store its own `DType`
or logical length. The parent supplies both, top-down, when it decodes a child: the child is fetched
by *(child index, expected dtype, expected length)*. Consequently the child dtypes of a container
are **derived by the parent**, never read from the child node — struct field dtypes come from the
struct's dtype, a list's element dtype from the list's dtype, the offsets/sizes integer types from
the parent's metadata, and a validity child is always a non-nullable `Bool` of length `n`.

**Optional child slots are omitted, not null-padded.** The child list carried by a node is the
sequence of *present* child slots, packed with no gaps. When an optional slot (a stored validity
child, or Variant's `shredded` child) is absent, it is simply not written, and the following
children shift down. A reader therefore distinguishes the layout by the **child count**: each
section below tabulates the exact count cases. (A stored validity child is present only when it
carries information — an `Array` mask, or an all-null `Constant(false)`; an all-valid or
non-nullable validity produces no child. See [Validity](../encoding-format.md#validity).)

**No data buffers.** Every encoding on this page reports **zero buffers**: all their state lives in
child arrays and metadata. Bytes are held by the (leaf) children these containers wrap.

Metadata, where non-empty, is a Protocol Buffers message (proto3 field defaults apply to absent
fields). An encoding whose metadata is specified as *empty* must be rejected by a reader if the
metadata segment is non-empty.

(encoding-layout-Struct)=
## `vortex.struct` — Struct

A struct array stores a fixed set of named fields as parallel columns; every field child has the
same length `n` as the struct, and field `f`'s value for row `i` is child-column `f` at position
`i`. It carries an independent row-level validity that composes with each field's own validity.

- **Wire encoding ID:** `vortex.struct` (`vortex-array/src/arrays/struct_/vtable/mod.rs` `Struct::id`).
- **Buffers:** none (`nbuffers` = 0, `struct_/vtable/mod.rs` `Struct::nbuffers`).
- **Metadata:** empty. A reader must reject non-empty metadata
  (`struct_/vtable/mod.rs` `Struct::deserialize`). Field names, field dtypes, and the field count all come from the
  node's `DType::Struct(fields, nullability)`, supplied top-down — not from metadata
  (`struct_/vtable/mod.rs` `Struct::deserialize`).

**Child slots** — let `N` = number of struct fields (`fields.nfields()`):

| Child count | Child 0 | Children `1..=N` (or `0..N`) | Meaning |
|-------------|---------|------------------------------|---------|
| `N` | field `0` | fields `1..N` | no stored validity child → validity from nullability |
| `N + 1` | validity | fields `0..N` (at child indices `1..=N`) | child 0 is the stored validity |

Every field child `f` has dtype `fields[f]` and length `n` (`struct_/vtable/mod.rs` `Struct::validate`,
`Struct::deserialize`). Fields are **row-aligned**: struct row `i` is the tuple `(field_0[i], …, field_{N-1}[i])`.
Field order is positional and matches the dtype's field order; names are not stored per node. Field
names need not be unique — accessors resolve a name to the first matching field
(`struct_/array.rs` `StructDataParts`).

**Decode.** Read the field count `N` from the dtype. If the node has `N` children, they are the
fields and validity is all-valid/non-nullable; if `N + 1`, child 0 is validity and children
`1..=N` are the fields. Decode each field child at its dtype and length `n`. Struct row `i` reads
position `i` from every field child.

**Validity.** Stored slot (Mechanism 1), row-aligned. The struct carries a *top-level* row validity
that **composes** with field validity: a null struct row makes the whole row null regardless of any
field's own value. See [Validity → Nested containers](../encoding-format.md#validity) — do not treat
field validity as the row validity or vice versa.

(encoding-layout-List)=
## `vortex.list` — List

A list array stores variable-length lists via a single **offsets** child that indexes into a flat
**elements** child. It mirrors the Apache Arrow `List` layout.

- **Wire encoding ID:** `vortex.list` (`vortex-array/src/arrays/list/vtable/mod.rs` `List::id`).
- **Buffers:** none (`nbuffers` = 0, `list/vtable/mod.rs` `List::nbuffers`).
- **Metadata** (`ListMetadata`, `list/vtable/mod.rs` `ListMetadata`):
  - `elements_len: u64` (tag 1) — logical length of the `elements` child.
  - `offset_ptype: PType` (tag 2) — integer type of the `offsets` child (see the `PType`
    enumeration in [Canonical encodings](canonical.md) or [DType Format](../dtype-format.md)).

**Child slots** (`list/array.rs` `ELEMENTS_SLOT`):

| Index | Name | Present | DType | Length | Role |
|-------|------|---------|-------|--------|------|
| 0 | `elements` | always | list element dtype (from `DType::List`) | `elements_len` | flat concatenation of all list contents |
| 1 | `offsets` | always | `Primitive(offset_ptype, NonNullable)` | `n + 1` | list boundaries |
| 2 | `validity` | optional | non-nullable `Bool` | `n` | row-level nulls |

Child count is **2** (no stored validity) or **3** (child 2 is validity)
(`list/vtable/mod.rs` `List::deserialize`). `n` is the outer length; `offsets.len() == n + 1`
(`list/vtable/mod.rs` `List::validate`).

**Offset semantics.** `offsets` is a **non-nullable integer** array of length `n + 1`, monotonically
non-decreasing, with `offsets[0] >= 0` and `offsets[n] <= elements_len`. List `i` occupies the
half-open range `elements[offsets[i] .. offsets[i+1]]`; its length is `offsets[i+1] - offsets[i]`
(zero-length lists allowed) (`list/array.rs` `ListData`, `ListData::validate`, `ListArrayExt::list_elements_at`).

**Decode.** Read `elements_len` and `offset_ptype` from metadata. Decode `elements` (child 0) at the
element dtype and length `elements_len`; decode `offsets` (child 1) as `Primitive(offset_ptype,
NonNullable)` of length `n + 1`. For row `i`, slice `elements[offsets[i] .. offsets[i+1]]`. If a
third child is present it is the validity mask of length `n`.

**Validity.** Stored slot (Mechanism 1), top-level and row-aligned; composes with element validity
exactly as for Struct. See [Validity → Nested containers](../encoding-format.md#validity).

(encoding-layout-ListView)=
## `vortex.listview` — ListView

A list-view array is the canonical `DType::List` encoding. Unlike `List`, each row carries **both**
an offset and a size, so the views may be out of order, may overlap, and need not tile the elements
child. It mirrors Apache Arrow's column-major `ListView`.

- **Wire encoding ID:** `vortex.listview` (`vortex-array/src/arrays/listview/vtable/mod.rs` `ListView::id`).
- **Buffers:** none (`nbuffers` = 0, `listview/vtable/mod.rs` `ListView::nbuffers`).
- **Metadata** (`ListViewMetadata`, `listview/vtable/mod.rs` `ListViewMetadata`):
  - `elements_len: u64` (tag 1) — logical length of the `elements` child.
  - `offset_ptype: PType` (tag 2) — integer type of the `offsets` child (see the `PType`
    enumeration in [Canonical encodings](canonical.md) or [DType Format](../dtype-format.md)).
  - `size_ptype: PType` (tag 3) — integer type of the `sizes` child (see the `PType` enumeration in
    [Canonical encodings](canonical.md) or [DType Format](../dtype-format.md)).

**Child slots** (`listview/array.rs` `ELEMENTS_SLOT`):

| Index | Name | Present | DType | Length | Role |
|-------|------|---------|-------|--------|------|
| 0 | `elements` | always | list element dtype (from `DType::List`) | `elements_len` | flat pool of list contents |
| 1 | `offsets` | always | `Primitive(offset_ptype, NonNullable)` | `n` | per-row start into `elements` |
| 2 | `sizes` | always | `Primitive(size_ptype, NonNullable)` | `n` | per-row length |
| 3 | `validity` | optional | non-nullable `Bool` | `n` | row-level nulls |

Child count is **3** (no stored validity) or **4** (child 3 is validity)
(`listview/vtable/mod.rs` `ListView::deserialize`). Both `offsets` and `sizes` have length exactly `n` — **not**
`n + 1` as in `List` (`listview/vtable/mod.rs` `ListView::validate`, `ListView::deserialize`).

**Offset/size semantics.** `offsets` and `sizes` are **non-nullable integer** arrays of equal length
`n`. Row `i` is the range `elements[offsets[i] .. offsets[i] + sizes[i]]`
(`listview/array.rs` `ListViewArrayExt::list_elements_at`). Constraints (`listview/array.rs` `ListViewData::new_unchecked`, `validate_offsets_and_sizes`): every
`offsets[i] + sizes[i] <= elements_len` (and no overflow); if `offsets[i] == elements_len` then
`sizes[i]` must be 0. (`offset_ptype` and `size_ptype` may be any integer widths independently — the
decode reads each child at its own type; there is no width relationship between them.) Offsets are **not** required
to be sorted, gaps and overlaps are permitted, and these constraints hold even for rows the validity
marks null.

**Decode.** Read the three metadata fields. Decode `elements` (child 0) at `elements_len`; decode
`offsets` (child 1) and `sizes` (child 2) as non-nullable primitives of their metadata ptypes, each
length `n`. For row `i`, slice `elements[offsets[i] .. offsets[i] + sizes[i]]`. A fourth child, if
present, is the validity mask of length `n`.

:::{note}
The in-memory `is_zero_copy_to_list` optimization flag (`listview/array.rs` `ListViewData`) is **not**
serialized. On decode it is reset to `false` (`ListViewData::try_new`, `listview/vtable/mod.rs` `ListView::deserialize`);
it is a runtime hint, never a wire fact.
:::

**Validity.** Stored slot (Mechanism 1), top-level and row-aligned; composes with element validity
as for Struct/List. See [Validity → Nested containers](../encoding-format.md#validity).

(encoding-layout-FixedSizeList)=
## `vortex.fixed_size_list` — FixedSizeList

A fixed-size-list array stores lists that all have the same length `list_size`, so no offsets are
needed: the elements are laid out contiguously and row `i` is a fixed-width slice.

- **Wire encoding ID:** `vortex.fixed_size_list`
  (`vortex-array/src/arrays/fixed_size_list/vtable/mod.rs` `FixedSizeList::id`).
- **Buffers:** none (`nbuffers` = 0, `fixed_size_list/vtable/mod.rs` `FixedSizeList::nbuffers`).
- **Metadata:** empty; a reader must reject non-empty metadata
  (`fixed_size_list/vtable/mod.rs` `FixedSizeList::deserialize`). `list_size` (a `u32`) and the element dtype come from the
  node's `DType::FixedSizeList(element_dtype, list_size, nullability)`, supplied top-down
  (`fixed_size_list/vtable/mod.rs` `FixedSizeList::deserialize`).

**Child slots** (`fixed_size_list/array.rs` `ELEMENTS_SLOT`):

| Index | Name | Present | DType | Length | Role |
|-------|------|---------|-------|--------|------|
| 0 | `elements` | always | element dtype (from dtype) | `n * list_size` | flat contiguous elements |
| 1 | `validity` | optional | non-nullable `Bool` | `n` | row-level nulls |

Child count is **1** (no stored validity) or **2** (child 1 is validity)
(`fixed_size_list/vtable/mod.rs` `FixedSizeList::deserialize`). `elements.len() == n * list_size`
(`fixed_size_list/array.rs` `FixedSizeListData::validate`).

**Element semantics.** Row `i` is the slice `elements[i * list_size .. (i + 1) * list_size]`
(`fixed_size_list/array.rs` `FixedSizeListArrayExt::fixed_size_list_elements_at`). Degenerate case `list_size == 0`: `elements` is empty and the
row count `n` cannot be recovered from the children — it is supplied by the parent as the node length
(`fixed_size_list/array.rs` `FixedSizeListData.degenerate_len`, `FixedSizeListData::validate`).

**Decode.** Read `list_size` and element dtype from the dtype. Decode `elements` (child 0) at the
element dtype and length `n * list_size`. For row `i`, slice `elements[i*list_size .. (i+1)*list_size]`.
A second child, if present, is the validity mask of length `n`.

**Validity.** Stored slot (Mechanism 1), top-level and row-aligned; composes with element validity as
for the other list containers. See [Validity → Nested containers](../encoding-format.md#validity).

(encoding-layout-Chunked)=
## `vortex.chunked` — Chunked

A chunked array is the logical concatenation, in order, of a sequence of same-dtype child chunks. A
leading child holds the chunk boundary offsets.

- **Wire encoding ID:** `vortex.chunked` (`vortex-array/src/arrays/chunked/vtable/mod.rs` `Chunked::id`).
- **Buffers:** none (`nbuffers` = 0, `chunked/vtable/mod.rs` `Chunked::nbuffers`).
- **Metadata:** empty; a reader must reject non-empty metadata (`chunked/vtable/mod.rs` `Chunked::deserialize`). The
  number of chunks is `child_count - 1`.

**Child slots** (`chunked/array.rs` `CHUNK_OFFSETS_SLOT`) — let `K` = number of chunks:

| Index | Name | DType | Length | Role |
|-------|------|-------|--------|------|
| 0 | `chunk_offsets` | `Primitive(U64, NonNullable)` | `K + 1` | exclusive prefix-sum of chunk lengths |
| `1 + j` | `chunks[j]` | the node's own dtype | `chunk_offsets[j+1] - chunk_offsets[j]` | the `j`-th chunk |

Every chunk child has the **same dtype as the chunked node itself** (`chunked/vtable/mod.rs` `Chunked::validate`).
There is always at least the `chunk_offsets` child (`chunked/vtable/mod.rs` `Chunked::deserialize`).

**Offset semantics.** `chunk_offsets` is a non-nullable `u64` array of length `K + 1`, built by the
writer as an exclusive prefix sum (`chunked/array.rs` `ChunkedData::compute_chunk_offsets`) — so
`chunk_offsets[0] == 0` **by construction**, it is non-decreasing, and `chunk_offsets[j+1] -
chunk_offsets[j]` is chunk `j`'s length. The invariant enforced on read is `chunk_offsets[K] == n`
(the total length) (`chunked/vtable/mod.rs` `Chunked::validate`, `Chunked::deserialize`). Row `i` lives in the unique
chunk `j` with `chunk_offsets[j] <= i < chunk_offsets[j+1]`, at chunk-local position
`i - chunk_offsets[j]` (`chunked/array.rs` `ChunkedArrayExt::find_chunk_idx`).

**Decode.** Let `K = child_count - 1`. Decode child 0 as `Primitive(U64, NonNullable)` of length
`K + 1` to get `chunk_offsets`. For `j` in `0..K`, decode child `1 + j` at the node's dtype and
length `chunk_offsets[j+1] - chunk_offsets[j]`. The logical array is the chunks concatenated in
order; map a global row `i` to `(chunk j, i - chunk_offsets[j])` by binary search over
`chunk_offsets`.

**Validity.** Chunked does **not** carry its own validity slot. It uses the **combine** mechanism:
its validity is the ordered concatenation of each chunk's validity. See
[Validity → `vortex.chunked`](../encoding-format.md#validity).

(encoding-layout-Extension)=
## `vortex.ext` — Extension

An extension array attaches a semantic extension type to a single storage child without changing the
storage bytes. All values and validity are those of the storage child, reinterpreted through the
extension type.

- **Wire encoding ID:** `vortex.ext` (`vortex-array/src/arrays/extension/vtable/mod.rs` `Extension::id`).
- **Buffers:** none (`nbuffers` = 0, `extension/vtable/mod.rs` `Extension::nbuffers`).
- **Metadata:** empty; a reader must reject non-empty metadata (`extension/vtable/mod.rs` `Extension::deserialize`). The
  extension identity, the storage dtype, and any extension-specific metadata are carried by the
  node's `DType::Extension(ext_dtype)` (supplied top-down), **not** in the array metadata segment
  (`extension/vtable/mod.rs` `Extension::deserialize`).

**Child slots** (`extension/array.rs` `STORAGE_SLOT`):

| Index | Name | Present | DType | Length | Role |
|-------|------|---------|-------|--------|------|
| 0 | `storage` | always (exactly 1 child) | `ext_dtype.storage_dtype()` | `n` | the underlying values |

The node has exactly one child (`extension/vtable/mod.rs` `Extension::deserialize`); its dtype is the extension type's
declared storage dtype and its length equals `n` (`extension/vtable/mod.rs` `Extension::validate`, `Extension::deserialize`).

**Decode.** Take the storage dtype from the extension dtype. Decode child 0 at that dtype and length
`n`. Row `i` of the extension array is storage row `i`, interpreted according to the extension type.

**Validity.** Delegate (Mechanism 2) to the `storage` child — the extension array's validity **is**
its storage child's validity, unchanged (`ValidityVTableFromChild`, `extension/vtable/mod.rs` `Extension::ValidityVTable`).
See [Validity → delegate](../encoding-format.md#validity).

(encoding-layout-Variant)=
## `vortex.variant` — Variant

A variant array stores semi-structured (`DType::Variant`) values. Every row's full value lives in the
`core_storage` child; an optional row-aligned `shredded` child holds typed values for selected paths
to accelerate typed access.

- **Wire encoding ID:** `vortex.variant` (`vortex-array/src/arrays/variant/vtable/mod.rs` `Variant::id`).
- **Buffers:** none (`nbuffers` = 0, `variant/vtable/mod.rs` `Variant::nbuffers`).
- **Metadata** (`VariantMetadataProto`, `variant/vtable/mod.rs` `VariantMetadataProto`):
  - `shredded_dtype: Option<DType>` (tag 1) — the dtype of the `shredded` child, as an encoded
    `vortex-proto` `DType`. Present **iff** a `shredded` child exists.

**Child slots** (`variant/mod.rs` `CORE_STORAGE_SLOT`):

| Index | Name | Present | DType | Length | Role |
|-------|------|---------|-------|--------|------|
| 0 | `core_storage` | always | the node's own `DType::Variant` | `n` | full variant value per row |
| 1 | `shredded` | optional | `shredded_dtype` (from metadata) | `n` | typed values for selected paths |

Child count is **1** when `shredded_dtype` is absent, **2** when present
(`variant/vtable/mod.rs` `Variant::deserialize`). `core_storage` has the **same** `DType::Variant` as the node and
length `n`; it is a logical variant array and may itself be any variant-typed encoding (chunked,
constant, …) — do not assume a physical layout for it (`variant/mod.rs` `VariantArrayExt`, `Array<Variant>::try_new`). When present,
`shredded` is row-aligned (length `n`) and may have any dtype (`variant/vtable/mod.rs` `Variant::validate`, `Variant::deserialize`).

**Decode.** Decode metadata; if `shredded_dtype` is set, expect 2 children, else 1. Decode child 0
(`core_storage`) at the node's variant dtype and length `n`. If present, decode child 1
(`shredded`) at `shredded_dtype` and length `n`. Row `i`'s logical value is `core_storage[i]`;
`shredded[i]`, when present and non-null at a path, supplies the typed value for that path (a
reader that ignores shredding can decode the full value from `core_storage` alone).

:::{note}
`core_storage` always retains the complete value for every row, so it is authoritative for
structural decoding. The exact rules for **merging** a `shredded` typed tree back over `core_storage`
(field-by-field precedence, null fallback) are a compute-layer concern
(`variant/vtable/mod.rs` `merge_typed_scalar_as_variant`) beyond this byte-layout section.
:::

**Validity.** Delegate (Mechanism 2) to the `core_storage` child — Variant's validity is
`core_storage`'s validity. See [Validity → delegate](../encoding-format.md#validity).

(encoding-layout-Masked)=
## `vortex.masked` — Masked

A masked array superimposes a nullability mask onto a single non-null child, producing a nullable
view without rewriting the child. It is always nullable, and its child must contain no **top-level**
nulls (a nested child — e.g. a `List` — may still carry inner-element nulls; only top-level row nulls
are forbidden).

- **Wire encoding ID:** `vortex.masked` (`vortex-array/src/arrays/masked/vtable/mod.rs` `Masked::id`).
- **Buffers:** none (`nbuffers` = 0, `masked/vtable/mod.rs` `Masked::nbuffers`; a reader must reject any buffer,
  `masked/vtable/mod.rs` `Masked::deserialize`).
- **Metadata:** empty; a reader must reject non-empty metadata (`masked/vtable/mod.rs` `Masked::deserialize`).

**Child slots** (`masked/array.rs` `MaskedSlots`):

| Index | Name | Present | DType | Length | Role |
|-------|------|---------|-------|--------|------|
| 0 | `child` | always | the node's dtype made **non-nullable** | `n` | the underlying values (no nulls) |
| 1 | `validity` | optional | non-nullable `Bool` | `n` | which rows are null |

Child count is **1** (no stored validity child) or **2** (child 1 is validity)
(`masked/vtable/mod.rs` `Masked::deserialize`). The `child` dtype is the node dtype with nullability removed
(`masked/vtable/mod.rs` `Masked::validate`, `Masked::deserialize`); the node dtype is always nullable
(`masked/array.rs` `MaskedData::try_new`, `Array<Masked>::try_new`). Row-aligned: `child.len() == n` (`masked/vtable/mod.rs` `Masked::validate`).

**Decode.** Decode child 0 at `dtype.as_nonnullable()` and length `n`. If a second child is present,
it is the validity mask (length `n`); otherwise validity comes from the node's nullability. Row `i`
is null iff validity says so, and equals `child[i]` otherwise. (Because the child is guaranteed
top-level null-free, the mask is the sole source of top-level nulls.)

**Validity.** Stored slot (Mechanism 1), row-aligned — the `validity` child directly gives per-row
validity of length `n` (`masked/array.rs` `MaskedArrayExt::masked_validity`). See [Validity](../encoding-format.md#validity).
