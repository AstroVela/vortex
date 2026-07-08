# Canonical Encodings — Byte Layout

The **canonical** encodings are the base, largely uncompressed forms — the Arrow-style flat layouts
that logical values canonicalise to, plus the trivial constant and generative encodings. This page
is the per-encoding byte-layout reference for that family, one section per encoding.

It is part of the [Encoding Format](../encoding-format.md) specification; null handling for every
encoding here follows the cross-cutting [Validity](../encoding-format.md#validity) contract on that
page and is not restated per section.

Encodings covered on this page: `Primitive`, `Bool`, `VarBin`, `VarBinView`, `Decimal`, `Null`,
`Constant`, `Sequence`.

```{contents}
:local:
:depth: 1
```

## Conventions used on this page

Each section below decodes one encoding from its **serialized components**: the array-specific
**metadata** bytes, the **data buffers** it references (by position in the node's buffer table), and
its **child** array nodes. The [Array Format](../array-format.md) page describes the container that
carries these; this page defines what each encoding's components *mean*. Terms used uniformly:

- **`n`** is the node's logical length. Vortex arrays do **not** store their own length or dtype —
  both are supplied by the parent (ultimately the file/IPC schema), so `n` and the `DType` are
  inputs to every decode here, never parsed from the node. See [Array Format](../array-format.md)
  and [DType Format](../dtype-format.md).
- **Metadata** is a Protobuf message (the reference implementation uses `prost`), decoded by field
  tag. An encoding whose metadata is *empty* serializes zero metadata bytes; its section says so.
  Beyond that, a metadata blob may be **zero bytes even for an encoding that defines metadata
  fields** — whenever every field holds its proto3 default, `prost` omits it, so nothing is written.
  For example a `Bool` with `offset = 0`, or a `Decimal` with `values_type = I8` (the `= 0` enum
  value), serialises no metadata bytes. A reader must therefore decode absent or empty metadata to
  the proto3 defaults for every field and must **not** error on "expected metadata".
  Integer/scalar values inside buffers and metadata are **little-endian** throughout.
- **Buffers** are referenced positionally: "buffer 0", "buffer 1", … index into this node's slice of
  the file's buffer table, in the order each section lists.
- **Child slots.** Each encoding has a fixed, named, ordered list of *slots*; a slot may be
  mandatory or optional. On the wire the children are a **dense list of the present slots, in slot
  order** — an absent optional slot is simply omitted, not written as a placeholder. A reader
  therefore reconstructs slot identity positionally from how many children are present (each section
  states the exact mapping). The most common optional slot is `validity`, present when validity is
  materialised as a **stored child** — an `Array`, or the `Constant(false)` child a writer emits for
  `AllInvalid` — and absent only for `NonNullable`/`AllValid` (see below).
- **Validity** (null handling) is **not** restated per encoding: it follows the cross-cutting
  [Validity](../encoding-format.md#validity) contract, and each section names which mechanism the
  encoding uses and links the [Validity reference](../encoding-format.md#validity-reference) table
  row. Recall Rule 0: if the node's `DType` is non-nullable, validity is `NonNullable` and no
  validity slot is read at all.

(encoding-layout-Primitive)=

## Primitive — `vortex.primitive`

A flat array of fixed-width numeric values, one per logical row. Mirrors the Arrow primitive layout.

**Wire encoding ID:** `vortex.primitive` (`vortex-array/src/arrays/primitive/vtable/mod.rs` `Primitive::id`).

**Buffers** (`primitive/vtable/mod.rs` `Primitive::nbuffers`, `Primitive::buffer_name`; buffer length checked in `Primitive::deserialize`):

| # | Name | Contents | Length |
|---|------|----------|--------|
| 0 | `values` | `n` values laid out contiguously, each `w` bytes, little-endian (two's-complement for integers, IEEE-754 for floats). | exactly `w · n` bytes |

The element width `w` and interpretation come from the node's `DType::Primitive(ptype, …)`; the
`ptype` is one of the 11 `PType`s (`U8`/`U16`/`U32`/`U64`/`I8`/`I16`/`I32`/`I64`/`F16`/`F32`/`F64`),
whose byte widths are 1/2/4/8 as per the width in the name (`F16` = 2 bytes)
(`vortex-array/src/dtype/ptype.rs` `PType`, `PType::byte_width`). See the `PType` enum in
[DType Format](../dtype-format.md#flatbuffer-definition). The buffer is aligned to `w`
(`primitive/vtable/mod.rs` `Primitive::deserialize`). `F16` holds
the raw IEEE-754 half-precision bits.

**Metadata:** none (zero bytes). The `ptype` is taken from the `DType`, not the metadata
(`primitive/vtable/mod.rs` `Primitive::deserialize`).

**Child slots** (`primitive/vtable/mod.rs` `Primitive::deserialize`; slot names `primitive/array/mod.rs` `SLOT_NAMES`):

| slot | name | role | presence |
|------|------|------|----------|
| 0 | `validity` | validity mask | optional — present when validity is a stored child (an `Array`, or `Constant(false)` for `AllInvalid`) |

So the child list is empty (validity sourced from the dtype's nullability) or holds exactly the
validity array as child 0.

**Decode:** for each `i` in `0..n`, the value occupies `values[i·w .. (i+1)·w]`, decoded as the
`ptype`. A row that validity marks null carries an arbitrary placeholder in these bytes — read the
value only after consulting validity.

**Validity:** stored slot (Mechanism 1), row-aligned; no offset is applied. See the
[Validity reference](../encoding-format.md#validity-reference) row for `vortex.primitive`.

(encoding-layout-Bool)=

## Bool — `vortex.bool`

A bit-packed boolean array — one bit per logical row. Mirrors the Arrow boolean layout, and is also
the flat form a `validity` `Array` takes.

**Wire encoding ID:** `vortex.bool` (`vortex-array/src/arrays/bool/vtable/mod.rs` `Bool::id`).

**Buffers** (`bool/vtable/mod.rs` `Bool::nbuffers`, `Bool::buffer_name`; length bound checked in `Bool::validate`):

| # | Name | Contents | Length |
|---|------|----------|--------|
| 0 | `bits` | packed bits, **LSB-first within each byte** (`0` = false, `1` = true) | at least `⌈(offset + n) / 8⌉` bytes |

**Metadata:** Protobuf `BoolMetadata` (`bool/vtable/mod.rs` `BoolMetadata`; the `0 ≤ offset < 8` bound is asserted in `Bool::serialize`):

| tag | field | type | meaning |
|-----|-------|------|---------|
| 1 | `offset` | `uint32` | starting bit offset within the first byte of `bits`; always `0 ≤ offset < 8` |

**Child slots:** identical shape to Primitive — slot 0 `validity`, optional (present when validity
is a stored child: an `Array`, or `Constant(false)` for `AllInvalid`) (`bool/vtable/mod.rs`
`Bool::deserialize`; slot names `bool/array.rs` `SLOT_NAMES`).

**Decode:** the value at logical row `i` is the bit at absolute bit index `offset + i`: byte
`(offset + i) / 8`, bit `(offset + i) % 8` counting the least-significant bit as bit 0. That is,
`value(i) = (bits[(offset+i) >> 3] >> ((offset+i) & 7)) & 1 == 1`. The `offset` (a sub-byte slice
start, `< 8`; whole leading bytes are dropped at write time) applies to `bits` **only**
(`bool/array.rs` `BoolData::to_bit_buffer`). Bits at
null positions are undefined.

**Validity:** stored slot (Mechanism 1), row-aligned. See the
[Validity reference](../encoding-format.md#validity-reference) row for `vortex.bool`.

(encoding-layout-VarBin)=

## VarBin — `vortex.varbin`

Variable-length binary/UTF-8 values stored as one concatenated byte buffer plus an offsets array.
Mirrors the Arrow variable-binary layout (offsets + data), except the offsets are a **child array**,
not a buffer.

**Wire encoding ID:** `vortex.varbin` (`vortex-array/src/arrays/varbin/vtable/mod.rs` `VarBin::id`).

**Buffers** (`varbin/vtable/mod.rs` `VarBin::nbuffers`, `VarBin::buffer_name`; `offsets[n] ≤ len(bytes)` checked in `varbin/array.rs` `VarBinData::validate`):

| # | Name | Contents | Length |
|---|------|----------|--------|
| 0 | `bytes` | all element bytes concatenated in row order | `≥ offsets[n]` (leading bytes before `offsets[0]` and trailing bytes after `offsets[n]` both permitted) |

**Metadata:** Protobuf `VarBinMetadata` (`varbin/vtable/mod.rs` `VarBinMetadata`):

| tag | field | type | meaning |
|-----|-------|------|---------|
| 1 | `offsets_ptype` | `PType` enum | integer `ptype` of the `offsets` child (e.g. `I32`, `I64`) |

The `PType` enum values are `U8=0, U16=1, U32=2, U64=3, I8=4, I16=5, I32=6, I64=7, F16=8, F32=9,
F64=10` (`vortex-array/src/dtype/ptype.rs` `PType`; see [DType Format](../dtype-format.md#flatbuffer-definition)).

**Child slots** (`varbin/array.rs` `OFFSETS_SLOT`, `VALIDITY_SLOT`):

| slot | name | role | presence |
|------|------|------|----------|
| 0 | `offsets` | element boundary offsets | **mandatory** |
| 1 | `validity` | validity mask | optional — present when validity is a stored child (an `Array`, or `Constant(false)` for `AllInvalid`) |

`offsets` is itself a Vortex array node of dtype `Primitive(offsets_ptype, NonNullable)` and length
`n + 1`; decode it recursively (it may use any encoding). The child list is therefore `[offsets]`
(no stored validity) or `[offsets, validity]` (`varbin/vtable/mod.rs` `VarBin::deserialize`).

**Decode:** decode `offsets` to `n + 1` integers `O[0..=n]`. They are monotonically non-decreasing
and satisfy `O[n] ≤ len(bytes)` (`varbin/array.rs` `VarBinData::validate`). `O[0]` is **not necessarily 0**: slicing a VarBin does not rebase
the offsets and does not trim the `bytes` buffer (`varbin/compute/slice.rs` `VarBin::_slice`), so after a slice `O[0]` may be `> 0` and `bytes`
may carry bytes both before `O[0]` and after `O[n]`. Element `i` is the byte range
`bytes[O[i] .. O[i+1]]` (`varbin/array.rs` `VarBinArrayExt::bytes_at`) — decode strictly this way; do not assume element 0 starts at byte 0, and do
not normalise or trim the buffer. If the node's `DType` is `Utf8`, those bytes are a UTF-8 string; if
`Binary`, raw bytes. Null rows still occupy an offsets slot (typically an empty range).

**Validity:** stored slot (Mechanism 1, slot 1), row-aligned. See the
[Validity reference](../encoding-format.md#validity-reference) row for `vortex.varbin`.

(encoding-layout-VarBinView)=

## VarBinView — `vortex.varbinview`

Variable-length binary/UTF-8 stored as fixed-width 16-byte **views** that either inline short values
or reference a data buffer. Mirrors the Arrow `StringView`/`BinaryView` layout.

**Wire encoding ID:** `vortex.varbinview` (`vortex-array/src/arrays/varbinview/vtable/mod.rs` `VarBinView::id`).

**Buffers:** a variable count, `k + 1` total, where the **last** buffer is the views and the
preceding `k ≥ 0` buffers hold spilled long-value bytes (`varbinview/vtable/mod.rs`
`VarBinView::nbuffers`, `VarBinView::buffer_name`):

| # | Name | Contents | Length |
|---|------|----------|--------|
| `0 .. k-1` | `buffer_0` … `buffer_{k-1}` | data blocks holding the bytes of values longer than 12 bytes | arbitrary |
| `k` (last) | `views` | `n` fixed 16-byte view entries | exactly `16 · n` bytes |

A decoder splits the buffer list as *(all-but-last = data buffers, last = views)*
(`varbinview/vtable/mod.rs` `VarBinView::deserialize`; the views buffer is validated to be exactly
`16 · n` bytes, and each view entry is 16 bytes, `varbinview/view.rs` `BinaryView`).

**Metadata:** none (zero bytes) (`varbinview/vtable/mod.rs` `VarBinView::serialize`).

**Child slots:** slot 0 `validity`, optional (present when validity is a stored child: an `Array`,
or `Constant(false)` for `AllInvalid`) (`varbinview/vtable/mod.rs` `VarBinView::deserialize`; slot
names `varbinview/array.rs` `SLOT_NAMES`).

**Decode:** the `views` buffer is `n` entries of 16 bytes each (`BinaryView`). For entry `i`, read
the first 4 bytes as `size: u32` (little-endian), the total value length. Then:

- **Inlined** (`size ≤ 12`): the value is bytes `4 .. 4+size` of the entry; bytes `4+size .. 16` are
  zero padding (`varbinview/view.rs` `Inlined`, `BinaryView::MAX_INLINED_SIZE`).
- **Reference** (`size > 12`): the remaining 12 bytes are three little-endian `u32`s —
  `prefix` occupies bytes `4..8` (the first 4 bytes of the value, a comparison shortcut),
  `buffer_index` bytes `8..12`, and `offset` bytes `12..16`. The value is
  `data_buffer[buffer_index][offset .. offset + size]` (`varbinview/view.rs` `Ref`).

| entry bytes | inlined (`size ≤ 12`) | reference (`size > 12`) |
|-------------|-----------------------|--------------------------|
| `0..4` | `size` (`u32` LE) | `size` (`u32` LE) |
| `4..8` | value bytes `0..4` | `prefix` = value bytes `0..4` |
| `8..12` | value bytes `4..8` | `buffer_index` (`u32` LE) |
| `12..16` | value bytes `8..12` | `offset` (`u32` LE) |

`Utf8` dtype → the assembled bytes are a UTF-8 string; `Binary` → raw bytes.

**Validity:** stored slot (Mechanism 1), row-aligned. See the
[Validity reference](../encoding-format.md#validity-reference) row for `vortex.varbinview`.

(encoding-layout-Decimal)=

## Decimal — `vortex.decimal`

Fixed-precision decimals stored as scaled integers of a chosen backing width.

**Wire encoding ID:** `vortex.decimal` (`vortex-array/src/arrays/decimal/vtable/mod.rs` `Decimal::id`).

**Buffers** (`decimal/vtable/mod.rs` `Decimal::nbuffers`, `Decimal::buffer_name`):

| # | Name | Contents | Length |
|---|------|----------|--------|
| 0 | `values` | `n` unscaled integers, each `w` bytes, little-endian two's-complement | exactly `w · n` bytes |

**Metadata:** Protobuf `DecimalMetadata` (`decimal/vtable/mod.rs` `DecimalMetadata`):

| tag | field | type | meaning |
|-----|-------|------|---------|
| 1 | `values_type` | `DecimalType` enum | physical integer width of `values` |

`DecimalType` values and their widths `w`: `I8 = 0` (1 byte), `I16 = 1` (2), `I32 = 2` (4),
`I64 = 3` (8), `I128 = 4` (16), `I256 = 5` (32) (`vortex-array/src/dtype/decimal/types.rs`
`DecimalType`, `DecimalType::byte_width`). `I256` is a 32-byte little-endian two's-complement
integer. The buffer is aligned to `w` for `I8`–`I128` (i.e. 1/2/4/8/16), but **`I256` is aligned to
16, not 32**: Vortex's `i256` is `#[repr(transparent)]` over `arrow_buffer::i256 { low: u128, high:
i128 }` (`vortex-array/src/dtype/bigint/mod.rs` `i256`), whose alignment is 16, and the reader
enforces exactly that (`Alignment::of::<D>()` = 16) (`decimal/vtable/mod.rs` `Decimal::deserialize`). A
reader that demands 32-byte alignment for `I256` would reject valid buffers.

The logical **precision** (`u8`) and **scale** (`i8`) come from the node's `DType::Decimal`, *not*
from the metadata (`decimal/array.rs` `DecimalArrayExt::decimal_dtype`). `values_type` is only the storage width and need not be the smallest that fits
the precision — a writer may store precision-38 values in an `I8` buffer if they all fit
(`decimal/array.rs` `DecimalData`).

**Child slots:** slot 0 `validity`, optional (present when validity is a stored child: an `Array`,
or `Constant(false)` for `AllInvalid`) (`decimal/vtable/mod.rs` `Decimal::deserialize`; slot names
`decimal/array.rs` `SLOT_NAMES`).

**Decode:** reinterpret `values` as `n` integers of width `w`. Row `i`'s logical value is
`unscaled[i] / 10^scale` (i.e. the integer with the decimal point moved left `scale` places; a
negative `scale` moves it right) (`decimal/array.rs` `DecimalData`). Null rows carry an arbitrary placeholder integer.

**Validity:** stored slot (Mechanism 1, child 0), row-aligned. See the
[Validity reference](../encoding-format.md#validity-reference) row for `vortex.decimal`.

(encoding-layout-Null)=

## Null — `vortex.null`

An all-null column of the `Null` type. Carries no data — only its length.

**Wire encoding ID:** `vortex.null` (`vortex-array/src/arrays/null/mod.rs` `Null::id`).

**Buffers:** none (`null/mod.rs` `Null::nbuffers`).

**Metadata:** none (zero bytes) (`null/mod.rs` `Null::serialize`).

**Child slots:** none.

**Decode:** the node's `DType` **must** be `Null` (`null/mod.rs` `Null::validate`). The array is `n` null values; there is nothing
else to read. Every logical position is null.

**Validity:** constant-validity rule — always `AllInvalid` (`null/mod.rs` `Null::validity`). See
[Constant-validity encodings](../encoding-format.md#constant-validity-encodings).

(encoding-layout-Constant)=

## Constant — `vortex.constant`

`n` repetitions of a single scalar value.

**Wire encoding ID:** `vortex.constant` (`vortex-array/src/arrays/constant/vtable/mod.rs` `Constant::id`).

**Buffers** (`constant/vtable/mod.rs` `Constant::nbuffers`, `Constant::buffer_name`):

| # | Name | Contents | Length |
|---|------|----------|--------|
| 0 | `scalar` | the Protobuf encoding of one `ScalarValue` (the value only; no dtype) | variable |

The buffer holds a serialized `ScalarValue` — the same value-only Protobuf message described in
[Scalar Format](../scalar-format.md) (a `null_value`, `bool_value`, primitive, decimal, string,
list, etc.) (`vortex-proto/proto/scalar.proto` `ScalarValue`). It is deliberately carried in a
buffer rather than metadata (`constant/vtable/mod.rs` `Constant::buffer`).

**Metadata:** none (zero bytes) (`constant/vtable/mod.rs` `Constant::serialize`).

**Child slots:** none.

**Decode:** decode the `ScalarValue` from buffer 0, interpreting it with the node's `DType` (supplied
externally) (`constant/vtable/mod.rs` `Constant::deserialize`). The array is that one scalar repeated for all `n` rows. If the decoded scalar is null,
every row is null; otherwise every row holds the scalar's value.

:::{warning}
A `vortex.constant` node's `DType` **MUST** be one the `ScalarValue` protobuf can represent. In
particular, `Struct` and `FixedSizeList` scalars do **not** round-trip through `ScalarValue` (see
[Scalar Format](../scalar-format.md)), so a writer **MUST NOT** emit a Constant with a `Struct` or
`FixedSizeList` dtype — the reference writer would serialize a `ListValue` that no reader (including
itself) can decode back. Constants of primitive, `Bool`, `Utf8`/`Binary`, `Decimal`, `List`, `Null`,
and `Extension` (over a representable storage) dtypes are fine.
:::

**Validity:** constant-validity rule — `AllInvalid` if the scalar is null, else `AllValid` (the
non-nullable case is already `NonNullable` via Rule 0) (`constant/vtable/validity.rs` `Constant::validity`). See
[Constant-validity encodings](../encoding-format.md#constant-validity-encodings).

(encoding-layout-Sequence)=

## Sequence — `vortex.sequence`

A generated arithmetic sequence: `A[i] = base + i · step`, materialised lazily. Values are integers.

**Wire encoding ID:** `vortex.sequence` (`encodings/sequence/src/array.rs` `Sequence::id`).

**Buffers:** none (`encodings/sequence/src/array.rs` `Sequence::nbuffers`).

**Metadata:** Protobuf `SequenceMetadata` (`sequence/src/array.rs` `SequenceMetadata`):

| tag | field | type | meaning |
|-----|-------|------|---------|
| 1 | `base` | `ScalarValue` | value at index 0 (`A[0]`) |
| 2 | `multiplier` | `ScalarValue` | the per-step increment (the `step`) |

Both are `ScalarValue` messages (see [Scalar Format](../scalar-format.md)) and are decoded as
non-nullable primitives of the node's `ptype`. Both fields are required (`sequence/src/array.rs` `Sequence::deserialize`).

**Child slots:** none.

**Decode:** the node's `DType` must be `Primitive(ptype, …)` with an **integer** `ptype`
(`sequence/src/array.rs` `SequenceData::validate`). Decode
`base` and `multiplier` to that `ptype`. Row `i` (for `i` in `0..n`) is `base + i · multiplier`,
evaluated in the `ptype`'s integer arithmetic (`sequence/src/array.rs` `SequenceData::index_value`). `n ≥ 1` (`sequence/src/array.rs` `SequenceData::validate`).

**Validity:** constant-validity rule — always `AllValid` (every index has a value) (`sequence/src/array.rs` `Sequence::validity`). See
[Constant-validity encodings](../encoding-format.md#constant-validity-encodings).
