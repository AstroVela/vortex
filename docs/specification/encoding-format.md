# Encoding Format

Every Vortex array node names an **encoding** — the concrete physical layout of its buffers,
metadata, and child slots. The [Array Format](array-format.md) page describes the container that
carries a node (the FlatBuffer tree plus the buffer table); this page is the authoritative
reference for what the bytes inside a node *mean*, encoding by encoding.

This page is written for reader/writer implementers porting Vortex to a new language. Its acceptance
bar is deliberately strict: each section must be complete and precise enough that an implementer
working **from this text alone**, with no access to the Rust reference, can decode the encoding
correctly for every case. The exceptions are the encodings that explicitly wrap an external codec —
`vortex.pco`, `vortex.zstd`, and `vortex.parquet.variant` — whose component byte streams are defined
by their named upstream specifications (at the versions Vortex pins) rather than restated here. Where
a rule has a subtle exception, the exception is stated inline rather than left to be inferred.

:::{note}
Scope. This page covers the **stable** encodings — those available without the `unstable_encodings`
build feature. Encodings gated behind `unstable_encodings` (e.g. `vortex.zstd_buffers` and
`vortex.onpair` — the btrblocks cascade registers as `vortex.onpair`), and any third-party plugin
encoding, are out of scope: a conformant reader that does not recognise an encoding ID must fail
loudly with a clear error, never guess a layout or return silently-wrong data.
:::

```{contents}
:local:
:depth: 2
```

## Validity

**Validity** answers one question for every logical position `i` in an array of length `n`: *is the
value at `i` present, or is it null?* It is the single most error-prone part of a clean-room reader,
because validity is not stored in one uniform place — each encoding *sources* it differently, and
several encodings store it in a transformed or unsliced form that a naive reader will mis-apply.
Getting this wrong produces silently-corrupted nulls rather than a loud failure, so this section
specifies the full contract exactly.

Throughout, "**valid**" means the value is present (non-null) and "**invalid**" (or "null") means it
is absent. The convention is uniform: wherever validity is materialised as booleans, **`true` = valid,
`false` = null**. A null at position `i` means the logical value there is null regardless of whatever
bytes the encoding's data buffers happen to hold at that position — the data may be an arbitrary
placeholder.

### The logical result: four forms

Decoding validity for a node yields one of four logical forms. They are representations of the same
thing — a per-position valid/null decision — differing only in how compactly the all-valid and
all-null cases are expressed:

| Form | Meaning |
|------|---------|
| `NonNullable` | The type forbids nulls. Every position is valid, by construction. |
| `AllValid` | The type permits nulls, but every position happens to be valid. |
| `AllInvalid` | Every position is null. |
| `Array` | Per-position validity given by a boolean array (see below). |

`NonNullable` and `AllValid` are indistinguishable when *reading a value* (both mean "position `i` is
present"); they differ only in the type's declared nullability. `AllValid` is exactly equivalent to
an `Array` of all-`true`; `AllInvalid` is exactly equivalent to an `Array` of all-`false`. An
implementation may collapse to the compact forms as an optimisation, but it is never *wrong* to
decode a validity as a full boolean array of length `n` — the compact forms are semantic shorthand,
not a distinct wire encoding.

#### The `Array` form is itself an encoded array

When validity is an `Array`, that array is a **non-nullable Boolean array of length exactly `n`**,
where `true` = valid. Critically, it is a *normal Vortex array node* and may use **any encoding** —
it is commonly `Dict`-, `RunEnd`-, or `Sparse`-encoded rather than a flat bitmap. A reader **must
decode it recursively** through the same encoding dispatch used for data arrays; it must **not**
assume a flat, packed bitmap.

:::{important}
The two invariants a validity `Array` must satisfy, and which a reader should enforce:

1. Its length equals the parent array's length `n`.
2. Its dtype is **non-nullable** `Bool`.

(A validity array that were itself nullable would be a contradiction — validity has no validity.)
:::

### Rule 0 — the nullability gate comes first

Before any encoding-specific logic runs, apply the **nullability gate**, which is universal across
every encoding:

> If the node's `DType` is **non-nullable**, its validity is `NonNullable`. Stop — do not read any
> validity slot, child, or metadata. The per-encoding rules below apply **only** when the dtype is
> nullable.

This gate is not per-encoding; it wraps the encoding dispatch. A writer is free to omit a validity
slot entirely on a non-nullable array, and a nullable array with no stored nulls is likewise valid.
Consequently every per-encoding rule in this section may assume the dtype is nullable.

### The three sourcing mechanisms

Once past the gate (dtype is nullable), an encoding produces its validity by one of three mechanisms
(or, for a few encodings, a constant-validity rule — see [Constant-validity encodings](#constant-validity-encodings)).
Each stable encoding is assigned to exactly one; the [reference table](#validity-reference) below is authoritative.

#### Mechanism 1 — stored slot

The encoding carries validity **directly**, in a dedicated child slot (its "validity slot"). The
slot is decoded to the logical form as follows — this is the **physical carriage rule**:

| Physical state of the validity slot | Decoded validity |
|-------------------------------------|------------------|
| Slot absent (no child present) | `AllValid` |
| Slot present, a `Constant` bool `true` | `AllValid` |
| Slot present, a `Constant` bool `false` | `AllInvalid` |
| Slot present, any other array | `Array` (decode that array recursively) |

In all cases the decoded meaning is the same: *decode the slot to a boolean mask of length `n` and
read position `i`; an absent slot means all-valid.* The `Constant`-collapse rows are an
optimisation — a writer that stores an all-true validity as a `Constant(true)` and one that stores
it as a flat all-`true` `Bool` array are semantically identical; the former merely decodes to
`AllValid` directly. (The collapse fires **only** for the literal `Constant` encoding carrying a
boolean scalar. An all-true array in some other encoding is carried as an `Array` and decoded
normally; the resulting bits are the same.)

##### The offset rule (most common mistake)

The stored validity slot is **row-aligned to the array's logical rows**: slot position `i`
corresponds to array row `i`, directly. When an array is sliced, its stored validity slot is sliced
to match. Therefore, for a stored-slot encoding, a reader must **not** apply any array-level offset
to the validity — doing so shifts every null and corrupts the result after any slice.

Two encodings look like exceptions but are **not** — on the wire a reader applies **no** offset to
their validity slot either:

- **`vortex.pco` and `vortex.zstd`** hold an in-memory `slice_start .. slice_stop` range, but that
  range is **runtime-only state for lazy in-place slicing and is not serialized**: it is absent from
  the on-wire `PcoMetadata`/`ZstdMetadata`, and `deserialize` sets it to `0 .. len`. An on-wire
  pco/zstd node is therefore **always unsliced** — its validity slot is a normal row-aligned stored
  slot of length `n`, decoded like any other, with **no** offset applied. (The slice range matters
  only for in-place slicing of an already-decoded array, never for a file reader.)
- **`fastlanes.bitpacked`** has an `offset` field (`0 ≤ offset < 1024`). That offset applies to the
  **packed data buffer only** — it is the start position within the first FastLanes block of 1024
  values. It is **never** applied to the validity slot, which remains row-aligned like every other
  stored slot.

:::{warning}
Never apply `fastlanes.bitpacked`'s `offset` to its stored validity slot: that offset is
packed-buffer-only, and over-applying it to validity is the single most damaging validity bug — it
corrupts every null after the slice point. On the wire **every** stored validity slot is row-aligned
and takes no offset — pco and zstd included, since their in-memory slice range is never serialized.
:::

#### Mechanism 2 — delegate to a child

The encoding stores no validity of its own; its validity **is** the validity of one specific child
array. Decode that child's validity (recursively, through this same contract) and use it unchanged.

The exact delegate child differs per encoding and must be matched precisely — delegating to the wrong
child yields a validity of the wrong length or the wrong values:

| Encoding | Delegates to child |
|----------|--------------------|
| `vortex.alp` | `encoded` |
| `vortex.alprd` | `left_parts` |
| `fastlanes.for` | `encoded` |
| `vortex.zigzag` | `encoded` |
| `vortex.datetimeparts` | `days` |
| `vortex.decimal_byte_parts` | `msp` (most-significant-parts, child 0) |
| `vortex.ext` (Extension) | `storage` |
| `vortex.variant` | `core_storage` |
| `fastlanes.delta` | `deltas` — **with transform**, see below |
| `fastlanes.rle` (FastLanes RLE) | `indices` — **offset-aware**, see below |

Most delegates are direct: the child is row-aligned to the parent, so the child's validity is the
parent's validity verbatim. Two carry an extra transform that a reader must reproduce:

- **`fastlanes.delta`** stores its `deltas` values — and therefore the `deltas` child's validity — in
  **FastLanes bit-transposed order** (blocks of 1024, transposed for SIMD unpacking). To recover
  row-order validity: take `deltas`' validity; if it is an `Array`, materialise it to a bit buffer,
  **untranspose** the bit buffer (inverse of the FastLanes transpose), and wrap it back into a
  boolean array; the compact forms (`AllValid` / `AllInvalid`) pass through untouched. Then slice
  the result to `offset .. offset + n`, where `offset` is the Delta array's block offset. Skipping
  either the untranspose or the offset slice yields scrambled or shifted nulls.
- **`fastlanes.rle`** (the FastLanes run-length encoding — *distinct from* `vortex.runend`, which is a
  combine encoding, see below) stores its `indices` child unsliced. Slice `indices` to
  `offset .. offset + n` first, *then* take that slice's validity.

:::{note}
`vortex.variant`'s validity delegate is implemented as a direct call to `core_storage`'s validity
rather than via the shared "from child" helper, but the observable rule is identical: Variant's
validity is its `core_storage` child's validity.
:::

#### Mechanism 3 — combine

The encoding synthesises validity by **combining** the validity of its children according to its
semantics. There are four such encodings; each has an exact rule.

##### `vortex.dict` (Dictionary)

A dictionary has a `codes` child (one code per logical row) and a `values` child (the dictionary
entries). Row `i` points at dictionary entry `codes[i]`. Row `i` is **valid iff `codes[i]` is itself
valid AND the dictionary value it points to (`values[codes[i]]`) is valid.** Equivalently, row `i`
is null iff its code is null, or the code resolves to a null dictionary value.

Concretely, from `codes.validity()` and `values.validity()`:

- both all-valid (`NonNullable`/`AllValid`) → `AllValid`.
- either is `AllInvalid` → `AllInvalid`.
- `codes` per-position `Array`, `values` all-valid → the codes' validity array (a null code is a
  null row; every value is valid).
- `codes` all-valid, `values` per-position `Array` → gather the values' validity through the codes:
  a `Dict` array of (`codes`, `values_validity`). Row `i`'s validity is the validity of the value
  its code selects.
- both per-position `Array` → gather the values' validity through the codes as above, then set to
  `false` every position whose code is itself null (`fill_null(false)`), because a null code is a
  null row irrespective of which value it would have named.

##### `vortex.runend` (Run-End)

A run-end array has an `ends` child (exclusive run boundaries) and a `values` child (one value per
run), plus an `offset` and length `n`. Its validity is the **run-expansion of the run values'
validity**: run value `r` covers a contiguous span of rows, and every row in that span shares run
`r`'s validity.

From `values.validity()`:

- all-valid (`NonNullable`/`AllValid`) → `AllValid`.
- `AllInvalid` → `AllInvalid`.
- per-position `Array` → a `RunEnd` array built from the *same* `ends`, the run values' validity as
  its values, and the *same* `offset` and `n`. The run-expansion is **offset-aware**: `offset` and
  `n` select the sliced logical window, exactly as for the data. A reader must carry `offset` and
  `n` through, or validity will not line up with a sliced run-end array.

##### `vortex.sparse` (Sparse)

A sparse array stores a `fill_value` scalar (the value at every non-patched position) and a set of
patches (`indices` + patch `values`). Its validity combines:

- **default** (all non-patched positions): the fill value's own validity — `fill_value.is_valid()`.
  If the fill scalar is null, unpatched positions are null; otherwise they are valid.
- **patched positions**: the patch **values'** validity, at their indices.

Result: a `Sparse` validity array whose fill is `fill_value.is_valid()` and whose patches are the
patch values' validity placed at the same `indices`. The patch offset metadata (`offset`,
`offset_within_chunk`, `chunk_offsets`) is preserved unchanged, so the validity is sliced-consistent
with the data. Row `i` is valid iff: `i` is a patch index and that patch value is valid; or `i` is
not a patch index and the fill value is valid.

##### `vortex.chunked` (Chunked)

A chunked array concatenates child chunks in order. Its validity is the **ordered concatenation of
each chunk's validity**, each chunk contributing its own length. Row `i` falls in exactly one chunk;
its validity is that chunk's validity at the chunk-local position. If every chunk shares the same
compact form (all `AllValid`, or all `AllInvalid`, or all `NonNullable`), the result collapses to
that form; otherwise it is an `Array` formed by concatenating the chunks' validities. A chunked array
with no chunks (`n = 0`) is `AllValid` (its declared nullable dtype with no rows).

### Constant-validity encodings

Three encodings determine validity from their type or a scalar alone, with no child or slot to read:

| Encoding | Validity |
|----------|----------|
| `vortex.constant` | `AllInvalid` if the constant scalar is null, else `AllValid`. |
| `vortex.null` | Always `AllInvalid`. (The `Null` type is null at every position by definition; dtype is `Null`.) |
| `vortex.sequence` | Always `AllValid`. (A sequence `A[i] = base + i·step` has a value at every position.) |

(For `vortex.constant`, recall Rule 0 has already excluded the non-nullable case; a non-nullable
constant is `NonNullable`.)

### Nested containers: top-level validity composes with field validity

A container (`vortex.struct`, `vortex.list`, `vortex.listview`, `vortex.fixed_size_list`) carries its
**own top-level validity** (via a stored slot — Mechanism 1) that is **independent of, and composes
with, the validity of its fields/elements**:

- The container's `validity()` returns only the **top-level, row-level** validity — whether the whole
  container row (the struct row, or the list at row `i`) is present.
- Each field/element child array carries its **own** validity, decoded independently.
- The two compose by masking: **a null container row makes the entire row null regardless of any
  field's own validity.** When reading field `f` at row `i`, the logical result is null if the
  container row `i` is null, *even if* field `f`'s child array reports that position as valid.
  Field-level validity only distinguishes present-vs-null *within* rows whose container is valid.

A reader must therefore combine the two levels (container-row null OR field null ⇒ null); it must not
treat a struct's field validity as the row's validity, nor vice versa.

(The row-alignment and length relationships between a container and its field/element children — for
example that struct fields are row-aligned to the struct while a list's element child is a separate
flattened buffer indexed by offsets — are specified in the canonical/container layout section.)

### Validity reference

The authoritative per-encoding assignment. "Mechanism" is one of the three above, or a
constant-validity rule. All rules assume Rule 0 has passed (dtype is nullable); a non-nullable dtype
is `NonNullable` for every encoding.

| Encoding ID | Mechanism | Source / detail |
|-------------|-----------|-----------------|
| `vortex.primitive` | Stored slot | Validity slot, row-aligned. |
| `vortex.bool` | Stored slot | Validity slot, row-aligned. |
| `vortex.varbin` | Stored slot | Validity slot, row-aligned. |
| `vortex.varbinview` | Stored slot | Validity slot, row-aligned. |
| `vortex.decimal` | Stored slot | Validity slot (child 0), row-aligned. |
| `vortex.struct` | Stored slot | Top-level validity slot; composes with field validity. |
| `vortex.list` | Stored slot | Top-level validity slot; composes with element validity. |
| `vortex.listview` | Stored slot | Top-level validity slot; composes with element validity. |
| `vortex.fixed_size_list` | Stored slot | Top-level validity slot; composes with element validity. |
| `vortex.masked` | Stored slot | Validity slot, row-aligned. |
| `vortex.bytebool` | Stored slot | Validity slot, row-aligned. |
| `vortex.fsst` | Stored slot | Validity slot, row-aligned. |
| `vortex.parquet.variant` | Stored slot | Validity slot (child 0), row-aligned. |
| `fastlanes.bitpacked` | Stored slot | Validity slot, row-aligned. `offset` applies to the packed buffer **only**, never to validity. |
| `vortex.pco` | Stored slot | Validity slot, row-aligned; on-wire always unsliced, **no** offset (the `slice_start..slice_stop` range is runtime-only, not serialized). |
| `vortex.zstd` | Stored slot | Validity slot, row-aligned; on-wire always unsliced, **no** offset (the `slice_start..slice_stop` range is runtime-only, not serialized). |
| `vortex.alp` | Delegate | child `encoded`. |
| `vortex.alprd` | Delegate | child `left_parts`. |
| `fastlanes.for` | Delegate | child `encoded`. |
| `vortex.zigzag` | Delegate | child `encoded`. |
| `vortex.datetimeparts` | Delegate | child `days`. |
| `vortex.decimal_byte_parts` | Delegate | child `msp` (child 0). |
| `vortex.ext` | Delegate | child `storage`. |
| `vortex.variant` | Delegate | child `core_storage`. |
| `fastlanes.delta` | Delegate | child `deltas`, then **untranspose** (FastLanes) and slice `offset..offset+n`. |
| `fastlanes.rle` | Delegate | child `indices`, sliced `offset..offset+n` **before** taking its validity. |
| `vortex.dict` | Combine | `codes` valid AND `values[codes]` valid. |
| `vortex.runend` | Combine | run-expand `values`' validity (offset- and length-aware). |
| `vortex.sparse` | Combine | `fill_value.is_valid()` default, patched positions take patch values' validity. |
| `vortex.chunked` | Combine | ordered concatenation of each chunk's validity. |
| `vortex.constant` | Constant rule | `AllInvalid` if scalar null, else `AllValid`. |
| `vortex.null` | Constant rule | Always `AllInvalid`. |
| `vortex.sequence` | Constant rule | Always `AllValid`. |

### Decoding procedure

The full validity-decode of a node, expressed as a recursive procedure returning a boolean decision
for each of the `n` positions (or one of the compact forms):

```text
decode_validity(node) -> validity of length n:
    # Rule 0 — nullability gate (universal, first).
    if not node.dtype.is_nullable():
        return NonNullable            # stop; read nothing further

    match node.encoding:                     # arms key on the encoding ID from the Validity reference table

        # --- Mechanism 1: stored slot ---
        stored-slot encoding:
            slot = node.validity_slot        # a child array, or absent
            if slot is absent:
                v = AllValid
            else:
                v = decode_array(slot)        # recursive; non-nullable Bool, length n
            # No offset is ever applied to a stored validity slot. (bitpacked's `offset` is
            # packed-buffer-only; pco/zstd track an in-memory slice range that is NOT serialized,
            # so an on-wire node is always unsliced and its validity slot is already length n.)
            return v

        # --- Mechanism 2: delegate to child ---
        delegate encoding:
            child = node.child(delegate_name)     # per reference table
            v = decode_validity(child)
            if node.encoding == delta:
                v = untranspose(v); v = v.slice(node.offset .. node.offset + n)
            if node.encoding == rle:
                v = decode_validity(child.slice(node.offset .. node.offset + n))
            return v

        # --- Mechanism 3: combine ---
        dict:    return combine_dict(decode_validity(codes), decode_validity(values), codes)
        runend:  return runexpand(decode_validity(values), ends, node.offset, n)
        sparse:  return sparse_validity(fill_value.is_valid(), decode_validity(patch_values),
                                        indices, node.patch_offsets)   # offset metadata preserved
        chunked: return concat(decode_validity(chunk) for chunk in chunks)

        # --- constant-validity encodings ---
        constant: return AllInvalid if node.scalar.is_null() else AllValid
        null:     return AllInvalid
        sequence: return AllValid

        # --- unrecognised (experimental / third-party) ---
        _: error("unknown encoding <id>")     # never guess; fail loudly
```

Where `decode_array` decodes an arbitrary Vortex array node to values (here, a non-nullable boolean
mask), through the same encoding dispatch — validity `Array`s are ordinary encoded arrays.

## Per-encoding byte layouts

The per-encoding byte-layout reference — buffer table, metadata fields, and child slots for each
stable encoding — is split into family pages, one per encoding family:

```{toctree}
:maxdepth: 1

encoding-format/canonical
encoding-format/containers
encoding-format/fastlanes
encoding-format/alp
encoding-format/dict-runend-sparse
encoding-format/misc
```
