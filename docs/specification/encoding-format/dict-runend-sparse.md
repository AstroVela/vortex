# Dictionary, Run-End, and Sparse Encodings — Byte Layout

The **dictionary**, **run-end**, and **sparse** encodings derive their logical values — and their
validity — by combining child arrays (codes and values, runs and ends, or fill value and patches).
This page is the per-encoding byte-layout reference for that family, one section per encoding.

It is part of the [Encoding Format](../encoding-format.md) specification; null handling for every
encoding here follows the cross-cutting [Validity](../encoding-format.md#validity) contract on that
page and is not restated per section.

Encodings covered on this page: `Dict`, `RunEnd`, `Sparse`.

Throughout, `n` is the node's logical length (the row count carried by the array container, after any
slicing). Each encoding's validity is sourced by the **combine** mechanism; the exact combine rule is
given once in [Validity](../encoding-format.md#validity) and is **not** restated here. These sections
specify the **data** decode — how row `i` resolves to a value — only.

(encoding-layout-Dict)=
## `vortex.dict` (Dictionary) — byte layout

A dictionary stores each logical row as an integer **code** that indexes a shared **values** table.
The decode identity is exact:

> **row `i` = `values[codes[i]]`.**

**Wire ID:** `vortex.dict` (`vortex-array/src/arrays/dict/vtable/mod.rs` `Dict::id`). **Buffers:** none
(0) (`dict/vtable/mod.rs` `Dict::nbuffers`). **Children:** exactly 2 — `codes` (slot 0) and
`values` (slot 1) (`dict/array.rs` `DictSlots`); a reader must reject any node that does not present
exactly these two children (`dict/vtable/mod.rs` `Dict::deserialize`).

### Metadata

Protobuf message `DictMetadata` (`dict/array.rs` `DictMetadata`):

| Proto tag | Field | Type | Meaning |
|-----------|-------|------|---------|
| 1 | `values_len` | `uint32` | Length (row count) of the `values` child. |
| 2 | `codes_ptype` | `PType` enum | Primitive type of the `codes` child (an integer type). |
| 3 | `is_nullable_codes` | optional `bool` | Whether the `codes` child's dtype is nullable. Absent (added after stabilisation) ⇒ fall back to the node's own nullability (`dict/vtable/mod.rs` `Dict::deserialize`). |
| 4 | `all_values_referenced` | optional `bool` | Optimisation hint (`true` ⇒ every value is referenced by some code). **Not needed to decode**; never rely on it for correctness. |

### Child slots

| Slot | Name | dtype | Length |
|------|------|-------|--------|
| 0 | `codes` | `Primitive(codes_ptype, nullable?)` — integer; `nullable?` from `is_nullable_codes` (else the node's nullability) | `n` |
| 1 | `values` | the node's own logical dtype | `values_len` |

`codes[i]` is a non-negative integer used directly as a 0-based index into `values`; every code
satisfies `0 <= codes[i] < values_len`. Both children are ordinary Vortex nodes and are decoded
recursively.

### Decode

1. Decode the `codes` child to an integer array of length `n`, and the `values` child to an array of
   length `values_len` (`dict/vtable/mod.rs` `Dict::deserialize`).
2. For each row `i` in `0..n`: read `c = codes[i]`, then emit `values[c]`
   (`dict/vtable/mod.rs` `Dict::execute`).

Nullability composes per the combine rule (a null code, or a code pointing at a null value, yields a
null row); see [Validity](../encoding-format.md#validity).

**Example.** `codes = [0, 2, 1, 0]`, `values = ["a", "b", "c"]` ⇒ `["a", "c", "b", "a"]`.

(encoding-layout-RunEnd)=
## `vortex.runend` (Run-End) — byte layout

A run-end array stores one **value** per run together with the **exclusive end position** of each
run. Row values are recovered by expanding each run across the positions it covers. The encoding is
**offset-aware**: a slice is expressed by an `offset` into the run coordinate space plus the logical
length `n`, without rewriting `ends` or `values`.

**Wire ID:** `vortex.runend` (`encodings/runend/src/array.rs` `RunEnd::id`). **Buffers:** none (0)
(`encodings/runend/src/array.rs` `RunEnd::nbuffers`). **Children:** exactly 2 — `ends` (slot 0) and
`values` (slot 1) (`encodings/runend/src/array.rs` `NUM_SLOTS`, `SLOT_NAMES`).

### Metadata

Protobuf message `RunEndMetadata` (`encodings/runend/src/array.rs` `RunEndMetadata`):

| Proto tag | Field | Type | Meaning |
|-----------|-------|------|---------|
| 1 | `ends_ptype` | `PType` enum | Primitive type of the `ends` child (an **unsigned** integer type). |
| 2 | `num_runs` | `uint64` | Number of runs = length of **both** the `ends` and `values` children (`encodings/runend/src/array.rs` `RunEnd::serialize`, `RunEndData::validate_parts`). |
| 3 | `offset` | `uint64` | Logical slice offset into the unsliced run coordinate space (see below). Unsliced arrays have `offset = 0`. |

### Child slots

| Slot | Name | dtype | Length |
|------|------|-------|--------|
| 0 | `ends` | `Primitive(ends_ptype, NonNullable)` — unsigned integer (`encodings/runend/src/array.rs` `RunEnd::deserialize`, `RunEndData::validate_parts`) | `num_runs` |
| 1 | `values` | the node's own logical dtype (`encodings/runend/src/array.rs` `RunEnd::deserialize`, `RunEnd::validate`) | `num_runs` |

`ends` is **strictly increasing** (`encodings/runend/src/compress.rs` `runend_encode`). `ends[j]` is
the **exclusive** end of run `j` in the *unsliced*
coordinate space: run `j` covers unsliced positions `[ends[j-1], ends[j])`, with `ends[-1] = 0`
(`encodings/runend/src/compress.rs` `runend_encode`). The
total unsliced extent is `ends[num_runs-1]` (`encodings/runend/src/array.rs`
`RunEndData::logical_len_from_ends`). Validity of a well-formed node satisfies
`ends[num_runs-1] >= offset + n` (and, when `offset != 0`, `ends[0] >= offset`)
(`encodings/runend/src/array.rs` `RunEndData::validate_parts`).

:::{note}
"Children: exactly 2" and "`ends` strictly increasing" are **writer requirements** (the format
contract). The reference *reader* does not hard-enforce them — `RunEnd::deserialize` fetches children
by index without rejecting a spurious third, and strict-increase is only a debug-assertion — so a
conformant reader SHOULD validate both itself rather than assume them. (`vortex.dict` and
`vortex.sparse`, by contrast, do reject a wrong child count on read.)
:::

### Decode (offset-aware run expansion)

A sliced run-end array selects the logical window `offset .. offset + n` of the unsliced coordinate
space. Logical row `i` (`0 <= i < n`) corresponds to unsliced position `i + offset`.

**Random access (single row `i`):**

1. Let `p = i + offset`.
2. Find the run `j` = the smallest index with `ends[j] > p` (equivalently a right-side binary search
   for `p` over `ends`) (`encodings/runend/src/array.rs` `RunEndArrayExt::find_physical_index`).
   Because `ends[num_runs-1] > p` for every in-range `i`, this always lands
   inside `0..num_runs`.
3. Emit `values[j]` (`encodings/runend/src/ops.rs` `RunEnd::scalar_at`).

**Bulk expansion (all `n` rows):** first map every raw end into the sliced window by clamping —
`trimmed[j] = min(ends[j] - offset, n)` (`encodings/runend/src/iter.rs` `trimmed_ends_iter`) — then
fill left to right (`encodings/runend/src/compress.rs` `runend_decode_slice`):

```text
cursor = 0
for j in 0 .. num_runs:
    e = min(ends[j] - offset, n)      # trimmed end; ends[j] >= offset is guaranteed
    for row in cursor .. e:           # empty span if e <= cursor
        out[row] = values[j]
    cursor = e
    if e == n: break                  # window filled
```

Runs whose end equals `offset` (`ends[j] == offset`; the strict `ends[j] < offset` case is precluded
by the `ends[0] >= offset` invariant (`encodings/runend/src/array.rs` `RunEndData::validate_parts`))
trim to `0` and emit nothing; runs whose end lies beyond
`offset + n` clamp to `n` and terminate expansion (`encodings/runend/src/iter.rs` `trimmed_ends_iter`).
A reader that ignores
`offset`/`n` and expands `ends` verbatim produces a shifted, wrong-length result on any sliced array.

**Example.** `ends = [2, 5, 10]`, `values = [1, 2, 3]`.
- `offset = 0, n = 10` ⇒ `[1, 1, 2, 2, 2, 3, 3, 3, 3, 3]`.
- `offset = 2, n = 5` (the `2..7` window) ⇒ `trimmed = [0, 3, 5]` ⇒ `[2, 2, 2, 3, 3]`.

(encoding-layout-Sparse)=
## `vortex.sparse` (Sparse) — byte layout

A sparse array stores a single **fill value** for the overwhelming majority of positions plus a
sorted set of **patches** — `(index, value)` pairs — for the exceptional positions. Row `i` is the
patch value if `i` is a patch index, else the fill value.

**Wire ID:** `vortex.sparse` (`encodings/sparse/src/lib.rs` `Sparse::id`). **Buffers:** 1 — buffer 0,
named `fill_value` (`encodings/sparse/src/lib.rs` `Sparse::nbuffers`, `Sparse::buffer_name`).
**Children:** exactly
2 — `patch_indices` (slot 0) and `patch_values` (slot 1) (`encodings/sparse/src/lib.rs` `SparseSlots`,
`Sparse::deserialize`).

### Buffer 0 — `fill_value`

The fill scalar's *value* serialized as `ScalarValue` protobuf bytes (`encodings/sparse/src/lib.rs`
`Sparse::buffer`). It is **not** in the metadata (`encodings/sparse/src/lib.rs` `Sparse::serialize`).
Decode it against the node's own `DType` (the fill scalar's dtype equals the node dtype)
(`encodings/sparse/src/lib.rs` `Sparse::deserialize`). If the fill
scalar is null, every non-patched position is null.

### Metadata

Protobuf message `SparseMetadata` (`encodings/sparse/src/lib.rs` `SparseMetadata`) wraps a single
required `PatchesMetadata` (tag 1) (`vortex-array/src/patches.rs` `PatchesMetadata`):

| Proto tag | Field | Type | Meaning |
|-----------|-------|------|---------|
| 1 | `len` | `uint64` | Number of patches = length of **both** the `patch_indices` and `patch_values` children. |
| 2 | `offset` | `uint64` | Absolute slice offset subtracted from each stored index (see below). Unsliced arrays have `offset = 0`. |
| 3 | `indices_ptype` | `PType` enum | Primitive type of `patch_indices` (an **unsigned** integer type). |
| 4 | `chunk_offsets_len` | optional `uint64` | In-memory lookup accelerator only — see note. |
| 5 | `chunk_offsets_ptype` | optional `PType` enum | In-memory lookup accelerator only — see note. |
| 6 | `offset_within_chunk` | optional `uint64` | In-memory lookup accelerator only — see note. |

:::{note}
The `chunk_offsets` accelerator — the `patch_chunk_offsets` child (a potential third patches child)
and `PatchesMetadata` tags 4–6 — is an in-memory `O(1)`-lookup index. It is **never attached to a
serialized Sparse node** by any production writer path: `Sparse::try_new`
(`encodings/sparse/src/lib.rs` `Sparse::try_new`), `encode` (`encodings/sparse/src/lib.rs`
`SparseData::encode`), and `slice` (`encodings/sparse/src/slice.rs` `Sparse::slice`) all
pass or preserve `None`. A conformant reader therefore resolves every row purely by `offset` + a
binary search over `patch_indices`, treating tags 4–6 and a third child as absent — and rejecting a
node that carries them (the deserializer requires *exactly* two children, `patch_indices` and
`patch_values`) (`encodings/sparse/src/lib.rs` `Sparse::deserialize`).
:::

### Child slots

| Slot | Name | dtype | Length |
|------|------|-------|--------|
| 0 | `patch_indices` | `Primitive(indices_ptype, NonNullable)` — unsigned integer (`vortex-array/src/patches.rs` `PatchesMetadata::indices_dtype`), **strictly sorted** | `len` |
| 1 | `patch_values` | the node's own logical dtype (`encodings/sparse/src/lib.rs` `Sparse::deserialize`) | `len` |

`patch_indices` are stored in the **absolute (pre-slice) coordinate space**: `offset` is how many
leading positions were sliced off the front (`vortex-array/src/patches.rs` `Patches::slice`). Each
stored index satisfies
`stored_index - offset < n` (`vortex-array/src/patches.rs` `Patches::new`).

### Decode (patch resolution)

Logical row `i` (`0 <= i < n`) maps to absolute position `a = i + offset`
(`vortex-array/src/patches.rs` `Patches::search_index`).

1. Binary-search `patch_indices` for the value `a` (`vortex-array/src/patches.rs`
   `Patches::search_index_binary_search`).
2. If found at patch position `p`, emit `patch_values[p]` (`vortex-array/src/patches.rs`
   `Patches::get_patched`).
3. Otherwise emit the `fill_value` scalar (`encodings/sparse/src/ops.rs` `Sparse::scalar_at`).

Nullability composes per the combine rule (patched positions take the patch value's validity;
non-patched positions take `fill_value`'s validity); see
[Validity](../encoding-format.md#validity).

**Example.** `patch_indices = [2, 5, 8]`, `patch_values = [100, 200, 300]`, `fill_value = null`,
`n = 10`.
- `offset = 0` ⇒ `[null, null, 100, null, null, 200, null, null, 300, null]`.
- `offset = 2, n = 5` (the `2..7` window) ⇒ absolute positions `2..7` ⇒
  `[100, null, null, 200, null]`.
