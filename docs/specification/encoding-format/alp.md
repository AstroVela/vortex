# ALP Encodings — Byte Layout

The **ALP** family is the floating-point-compression group: Adaptive Lossless floating-Point (ALP)
and its real-double variant (ALPRD). This page is the per-encoding byte-layout reference for that
family, one section per encoding.

It is part of the [Encoding Format](../encoding-format.md) specification; null handling for every
encoding here follows the cross-cutting [Validity](../encoding-format.md#validity) contract on that
page and is not restated per section.

Encodings covered on this page: `ALP`, `ALPRD`.

(encoding-layout-ALP)=
## `vortex.alp` — Byte layout

ALP (Adaptive Lossless floating-Point) stores an `f32`/`f64` column as scaled integers. Each value
`x` is represented by an integer `i` chosen so that `x = i × 10^(f − e)`, for two small per-array
exponents `e` and `f`. Values that do not round-trip through that scaling — the **exceptions** — are
kept verbatim as floating-point *patches*. The scaled integers live in the `encoded` child; the
exceptions live in patch children.

The logical dtype is `f32` (with `i32` scaled integers) or `f64` (with `i64` scaled integers); the
integer width is fixed by the float width.

**Wire ID:** `vortex.alp`.

**Buffers:** none. `vortex.alp` owns no buffers (`nbuffers = 0`); everything is carried in metadata
and children (`array.rs` (`ALP::nbuffers`)).

### Metadata (`ALPMetadata`, Protobuf)

| Field | Protobuf type / tag | Meaning |
|-------|---------------------|---------|
| `exp_e` | `uint32`, tag 1 | The exponent `e`. Bounded to `0 ≤ e ≤ 10` for `f32` and `0 ≤ e ≤ 18` for `f64`; always fits a `u8`. |
| `exp_f` | `uint32`, tag 2 | The factor `f`, same bounds as `e`. |
| `patches` | `PatchesMetadata`, tag 3, optional | Present iff the array has exceptions. Describes the patch children — see [Patch metadata and children](#alp-patch-metadata). |

The bounds and their per-type values come from `ALPFloat::MAX_EXPONENT` (`10` for `f32`, `18` for
`f64`) checked in `array.rs` (`ALPData::validate_components`); the fields are stored as `u32` and narrowed to `u8` in
`array.rs` (`ALP::deserialize`).

### Child slots

Children are positional (`array.rs` (`ALP::deserialize`)):

| Idx | Name | Present when | Logical dtype |
|-----|------|--------------|---------------|
| 0 | `encoded` | always | `i32` (for an `f32` array) / `i64` (for an `f64` array), **same nullability** as the ALP array. |
| 1 | `patch_indices` | `patches` set | non-nullable unsigned int (ptype from `PatchesMetadata`). |
| 2 | `patch_values` | `patches` set | the ALP array's float dtype (`f32`/`f64`). |
| 3 | `patch_chunk_offsets` | `patches` set **and** metadata `chunk_offsets_ptype` present | non-nullable unsigned int (acceleration index). |

**Child lengths.** `encoded` has length `n` (row-aligned to the ALP array); the patch children
`patch_indices` and `patch_values` have length `P` (the patch count), and `patch_chunk_offsets` has
length `chunk_offsets_len` — all per the shared [Patch metadata](#alp-patch-metadata). (The ALP-RD
children follow the same convention: `left_parts` / `right_parts` are length `n`, patch children
length `P`.)

`encoded` is decoded recursively like any node (it is typically itself `fastlanes`-packed). At an
exception position the integer stored in `encoded` is a **meaningless placeholder** — the "fill
value", the first non-exception encoded integer, substituted for compressibility
(`mod.rs` (`encode_chunk_unchecked`)). The true value is supplied by the patch, so a reader **must** apply patches and
must never trust `encoded` at a patched position.

### Decode

Given `encoded` decoded to an `i32`/`i64` slice, the exponents `e`, `f`, and the optional patches:

1. **Integer → float.** For each position `k`, compute in the target float type as a **two-step
   multiply by the tabulated constants** — this exact sequence is normative:
   ```text
   value[k] = (float) encoded[k] × F10[f] × IF10[e]
   ```
   where `F10[f] = 10^f` and `IF10[e] = 10^(−e)` are the per-float-type exact table entries; the
   multiply order matches `decode_single` in `mod.rs` (`ALPFloat::decode_single`). Example: `e = 2`, `f = 0`,
   `encoded[k] = 1234` → `1234 × 1 × 10^(−2) = 12.34`.

   :::{warning}
   Do **not** collapse this to a single `powi(10, f − e)`. ALP is lossless: the encoder only omits
   (rather than patches) values that round-trip through this *exact two-step* sequence in the target
   float type. A single combined power double-rounds and can differ by 1 ULP on non-exception
   values — a silent lossy divergence. Use the two tabulated constants, in this order.
   :::
2. **Apply patches** (if `patches` present). For each patch `j` in `0 .. patches.len`, overwrite
   ```text
   value[ indices[j] − offset ] = patch_values[j]
   ```
   Patch values are already floats and replace the decoded placeholder outright (`decompress.rs` (`decompress_unchunked_core`);
   the `index − offset` write is `patch.rs` (`PrimitiveArray::patch_typed`)).
3. **Validity** is delegated (below).

(alp-patch-metadata)=
### Patch metadata and children

`PatchesMetadata` (Protobuf, `patches.rs` (`PatchesMetadata`)) describes the patch children. This layout is shared
verbatim by `vortex.alprd`.

| Field | Protobuf type / tag | Meaning |
|-------|---------------------|---------|
| `len` | `uint64`, tag 1 | Number of patches `P` (= length of both the `patch_indices` and `patch_values` children). |
| `offset` | `uint64`, tag 2 | Absolute offset subtracted from every stored index; `0` for an unsliced array. Array-local position = `index − offset`. |
| `indices_ptype` | `PType` enum, tag 3 | Unsigned-int ptype of the `patch_indices` child. |
| `chunk_offsets_len` | `uint64`, tag 4, optional | Length of the `patch_chunk_offsets` child, when present. |
| `chunk_offsets_ptype` | `PType` enum, tag 5, optional | Unsigned-int ptype of `patch_chunk_offsets`; its presence signals that child 3 exists. |
| `offset_within_chunk` | `uint64`, tag 6, optional | Rows sliced off the front of the first chunk (slice bookkeeping). |

`patch_indices` is a strictly-ascending non-nullable unsigned-int array of length `P`; each entry is
the absolute logical position of an exception. `patch_values` (length `P`, non-null) holds the value
for that position. The pair `(indices[j], values[j])` reads as: *logical row `indices[j] − offset`
takes value `values[j]`.* `patch_chunk_offsets` is an **optional O(1) lookup index** — one entry per
1024-row chunk (`PATCH_CHUNK_SIZE = 1024`, `patches.rs` (`PATCH_CHUNK_SIZE`)) pointing into the patch arrays; a decoder
that scans all `P` patches linearly does not need it and may ignore it.

### Validity

`vortex.alp` sources validity by **delegation to its `encoded` child**: decode `encoded`'s validity
recursively and use it unchanged (`array.rs` (`ALP::ValidityVTable`), `array.rs` (`ALP::validity_child`)). The full contract — including
the universal nullability gate — is specified once in [Validity](../encoding-format.md#validity); it
is not restated here.

(encoding-layout-ALPRD)=
## `vortex.alprd` — Byte layout

ALP-RD ("real doubles") targets floats that do not scale to integers cleanly. It splits each value's
IEEE-754 bit pattern into a low part of `right_bit_width` bits (`right_parts`) and a high part of at
most 16 bits (the "left" bits). Because the high parts repeat heavily across a vector, they are
**dictionary-encoded**: the `left_parts` child stores a small dictionary *code* per row, and the
metadata `dict` maps each code back to its 16-bit pattern. High parts not in the dictionary are held
as **exceptions** in left-parts patches. Recombining a left pattern with its right part reproduces
the original IEEE bits exactly (lossless).

The logical dtype is `f32` or `f64`.

**Wire ID:** `vortex.alprd`.

**Buffers:** none (`nbuffers = 0`, `array.rs` (`ALPRD::nbuffers`)); all data is in metadata and children.

### Metadata (`ALPRDMetadata`, Protobuf)

| Field | Protobuf type / tag | Meaning |
|-------|---------------------|---------|
| `right_bit_width` | `uint32`, tag 1 | Number of low bits `R` per value; fits a `u8` (`array.rs` (`ALPRD::deserialize`)). The left part occupies the remaining `BITS − R` bits (`≤ 16`; `BITS` is 32 for `f32`, 64 for `f64`). |
| `dict_len` | `uint32`, tag 2 | Number of valid dictionary entries `D` (`D ≤ 8`, `MAX_DICT_SIZE`, `mod.rs` (`MAX_DICT_SIZE`)). |
| `dict` | repeated `uint32`, tag 3 | The dictionary. Entry `dict[c]` is the 16-bit left bit-pattern for code `c`. Only the first `dict_len` entries are used; each fits a `u16` (`array.rs` (`ALPRD::deserialize`)). |
| `left_parts_ptype` | `PType` enum, tag 4 | Unsigned-int ptype of the `left_parts` child (`u16` in the reference encoder). |
| `patches` | `PatchesMetadata`, tag 5, optional | Left-parts exceptions. Layout is exactly the [Patch metadata](#alp-patch-metadata) above, with the type note below. |

### Child slots

Children are positional (`array.rs` (`SLOT_NAMES`)):

| Idx | Name | Present when | Logical dtype |
|-----|------|--------------|---------------|
| 0 | `left_parts` | always | unsigned int (ptype = `left_parts_ptype`), **same nullability** as the ALP-RD array. One dictionary code per row, `0 ≤ code < dict_len`. |
| 1 | `right_parts` | always | `u32` (for `f32`) / `u64` (for `f64`), **non-nullable**. The low `R` bits of each value. |
| 2 | `patch_indices` | `patches` set | non-nullable unsigned int; see [Patch metadata](#alp-patch-metadata). |
| 3 | `patch_values` | `patches` set | unsigned int (`left_parts_ptype`, non-nullable). The **true 16-bit left patterns** of the exceptions — *not* codes (`array.rs` (`ALPRD::deserialize`)). |

`left_parts` and `right_parts` are decoded recursively (both are typically `fastlanes`-bit-packed).
At an exception position, `left_parts` stores code `0` as a placeholder; the real left pattern comes
from the patch. ALP-RD patches never carry chunk offsets, and a **freshly encoded** array's patch
`offset` is `0` (`mod.rs` (`RDEncoder::encode_generic`)). Slicing, however, updates the patch
`offset` and re-serializes it (`array.rs` (`ALPRD::serialize`)), so a **sliced** ALP-RD array
persists a non-zero `offset`; a reader must always apply the `indices[j] − offset` rule below and
never assume `offset == 0`.

### Decode

Given `left_parts` codes, `right_parts`, the `dict`, `R = right_bit_width`, and optional patches, for
each row `k`:

1. **Dictionary-decode** the left bits: `L[k] = dict[ left_parts[k] ]` (`mod.rs` (`alp_rd_decode`)).
2. **Apply patches** (if present). For each patch `j`, overwrite the left bits with the exception's
   true pattern:
   ```text
   L[ indices[j] − offset ] = patch_values[j]
   ```
   This runs *after* the dictionary decode, so it replaces the placeholder `dict[0]` at exception
   positions (`mod.rs` (`alp_rd_decode`), `mod.rs` (`alp_rd_apply_patches`)).
3. **Recombine** into the value's unsigned integer type (`u32` for `f32`, `u64` for `f64`):
   ```text
   bits[k] = ( (UINT) L[k] << R ) | right_parts[k]
   ```
   (`mod.rs` (`alp_rd_decode`); equivalently `ops.rs` (`ALPRD::scalar_at`)).
4. **Reinterpret** `bits[k]` as the IEEE-754 float via a bit-cast (`u32 → f32`, `u64 → f64`;
   `mod.rs` (`alp_rd_combine_inplace`)).
5. **Validity** is delegated (below).

Worked example (`f64`, `R = 52`, so the left part is the top 12 bits): if `dict = [0x3FF]` and row `k`
has `left_parts[k] = 0` (selecting `dict[0] = 0x3FF`) and `right_parts[k] = 0x0000000000000000`, then
`L[k] = 0x3FF`, `bits[k] = 0x3FF << 52 = 0x3FF0000000000000`, which bit-casts to `1.0`. (`R = 52` keeps
the left part `BITS − R = 12` bits, within the ≤16-bit dictionary-code width.)

### Validity

`vortex.alprd` sources validity by **delegation to its `left_parts` child** (`array.rs` (`ALPRD::ValidityVTable`),
`array.rs` (`ALPRD::validity_child`)): decode `left_parts`' validity recursively and use it unchanged. `right_parts`
is non-nullable and contributes no validity. See [Validity](../encoding-format.md#validity).
