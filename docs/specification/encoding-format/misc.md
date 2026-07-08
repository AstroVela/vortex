# Miscellaneous Encodings — Byte Layout

The remaining stable encodings that do not fall into the other families: string compression
(`FSST`), byte-wide booleans, split-representation encodings for temporal and decimal values, integer
transforms, and the external-codec encodings. This page is the per-encoding byte-layout reference for
that group, one section per encoding.

It is part of the [Encoding Format](../encoding-format.md) specification; null handling for every
encoding here follows the cross-cutting [Validity](../encoding-format.md#validity) contract on that
page and is not restated per section.

Encodings covered on this page: `FSST`, `ByteBool`, `DateTimeParts`, `DecimalByteParts`, `ZigZag`,
`Pco`, `Zstd`, `ParquetVariant`.

```{contents}
:local:
:depth: 1
```

Throughout, `n` is the array's logical length, a **child** is a nested Vortex array node decoded
recursively through the same encoding dispatch, and a **buffer** is a raw byte buffer. Children are
carried in named **slots**; on the wire an optional slot (notably the validity slot) may be omitted,
so the number of serialised children is smaller than the slot count. Each section states both the
slot layout and the exact on-wire child ordering a reader must map back onto it. All multi-byte
integers are little-endian. Metadata is a Protocol Buffers message; the `tag` column gives the
protobuf field number.

(encoding-layout-FSST)=
## `vortex.fsst` — Byte layout

FSST (Fast Static Symbol Table) string compression, per the FSST scheme (Boncz, Neumann, Leis,
*VLDB 2020*). Each row's bytes are compressed to a sequence of 8-bit **codes**; a code either names
an entry in a shared **symbol table** (expanding to 1–8 bytes) or is the **escape code**, which emits
the following raw byte verbatim.

- **Wire ID:** `vortex.fsst` (`array.rs` `FSST::id`).
- **DType:** `Utf8` or `Binary` (either nullability, enforced in `array.rs` `FSSTData::validate_parts`).

This section specifies the **current** 3-buffer format. A legacy 2-buffer form also exists and is
covered under *Compatibility* below.

**Buffers** (3, `array.rs` `FSST::nbuffers`; names from `array.rs` `FSST::buffer_name`):

| Index | Name | Contents |
|-------|------|----------|
| 0 | `symbols` | The symbol table: 8 bytes per symbol, `n_sym` symbols (`n_sym ≤ 255`), so `8 · n_sym` bytes. Symbol `s` occupies bytes `[8·s, 8·s+8)`; its content is the first `lengths[s]` of those bytes, in order. (`n_sym ≤ 255` checked in `array.rs` `FSSTData::validate_parts`.) |
| 1 | `symbol_lengths` | One `u8` per symbol (`n_sym` bytes). `lengths[s]` ∈ `1..=8` is the number of content bytes of symbol `s` (`array.rs` `FSSTData::validate_symbol_lengths`). |
| 2 | `compressed_codes` | The concatenated code bytes for all rows. Row `i` occupies `compressed_codes[codes_offsets[i] .. codes_offsets[i+1]]`. |

**Metadata** (`FSSTMetadata`, `array.rs` `FSSTMetadata`):

| Field | Tag | Type | Meaning |
|-------|-----|------|---------|
| `uncompressed_lengths_ptype` | 1 | `PType` enum | Integer type of the `uncompressed_lengths` child. |
| `codes_offsets_ptype` | 2 | `PType` enum | Integer type of the `codes_offsets` child. |

**Child slots** (2–3; wire child order equals slot order; `array.rs` `FSST::deserialize`, slot names `array.rs` `SLOT_NAMES`):

| Slot | Name | Role | DType / length |
|------|------|------|----------------|
| 0 | `uncompressed_lengths` | Decoded byte length of each row (used to size/split the decoded heap). | Non-nullable integer (`uncompressed_lengths_ptype`), length `n`. |
| 1 | `codes_offsets` | VarBin-style offsets into `compressed_codes`. | Non-nullable integer (`codes_offsets_ptype`), length `n + 1` (`array.rs` `FSSTData::validate_parts`). |
| 2 | `codes_validity` | Validity slot. | Non-nullable `Bool`, length `n`. Omitted when the array carries no per-position nulls. |

**Decode** (row `i`, when valid):

1. Take the compressed byte range `R = compressed_codes[codes_offsets[i] .. codes_offsets[i+1]]`.
2. Walk `R` with a cursor `p` starting at 0, emitting to an output buffer:
   - Let `c = R[p]`. If `c == 255` (the escape code, `fsst-rs` 0.5.11 `lib.rs` `ESCAPE_CODE`), emit `R[p+1]` verbatim and advance `p += 2`.
   - Otherwise emit the first `symbol_lengths[c]` bytes of symbol `c` (bytes `symbols[8·c .. 8·c + symbol_lengths[c]]`, in order) and advance `p += 1`.
   - Stop when `p` reaches the end of `R`.
3. The emitted byte count equals `uncompressed_lengths[i]`. The emitted bytes are row `i`'s value (interpreted as UTF-8 for `Utf8`, raw for `Binary`) (`canonical.rs` `fsst_decode_views`).

**Compatibility (legacy 2-buffer form).** Older files store buffers `[symbols, symbol_lengths]` and
children `[codes, uncompressed_lengths]`, where `codes` is a full `VarBin` array (its bytes are the
`compressed_codes` heap, its offsets are `codes_offsets`, and its validity is the codes validity).
A reader recognises this form by the buffer count being 2 rather than 3; decode is identical after
lifting the bytes/offsets/validity out of the `codes` child (`array.rs` `FSST::deserialize`, `FSST::deserialize_legacy`).

**Validity.** Stored slot (`codes_validity`), row-aligned (`array.rs` `FSST::validity`). See
[Validity](../encoding-format.md#validity).

(encoding-layout-ByteBool)=
## `vortex.bytebool` — Byte layout

A boolean array stored as **one byte per value** (rather than a packed bitmap). Trades space for
branch-free, byte-addressable access.

- **Wire ID:** `vortex.bytebool` (`array.rs` `ByteBool::id`).
- **DType:** `Bool` (either nullability).

**Buffers** (1, `array.rs` `ByteBool::nbuffers`; name from `array.rs` `ByteBool::buffer_name`):

| Index | Name | Contents |
|-------|------|----------|
| 0 | `values` | Exactly `n` bytes, one per row (`array.rs` `ByteBoolData::validate`). Writers emit `0x00` for `false` and `0x01` for `true` (`array.rs` `ByteBool::from_vec`). |

**Metadata:** empty (a reader must reject non-empty metadata, `array.rs` `ByteBool::deserialize`).

**Child slots** (0–1, `array.rs` `ByteBool::deserialize`; slot name `array.rs` `SLOT_NAMES`):

| Slot | Name | Role | DType / length |
|------|------|------|----------------|
| 0 | `validity` | Validity slot. | Non-nullable `Bool`, length `n`. Omitted when the array carries no per-position nulls. |

**Decode:** row `i` (when valid) is `values[i] != 0` — the canonical decoder treats byte `0` as
`false` and any non-zero byte as `true` (`array.rs` `ByteBoolData::truthy_bytes`).

**Validity.** Stored slot, row-aligned (`array.rs` `ByteBool::validity`). See [Validity](../encoding-format.md#validity).

(encoding-layout-DateTimeParts)=
## `vortex.datetimeparts` — Byte layout

Splits a timestamp column into three integer children — whole **days**, whole **seconds within the
day**, and a **sub-second remainder** — which compress far better independently than the combined
64-bit instant. Decoding recombines them into a single instant in the timestamp's time unit.

- **Wire ID:** `vortex.datetimeparts` (`array.rs` `DateTimeParts::id`).
- **DType:** an `Extension` type whose `id` is `vortex.timestamp` and whose metadata is the
  **Timestamp extension metadata**, laid out as `[unit_tag: u8][tz_len: u16 LE][tz: UTF-8]`
  (`vortex-array/src/extension/datetime/timestamp.rs` `serialize_metadata`): a one-byte `unit_tag`,
  then a little-endian `u16` timezone-name length `tz_len`, then `tz_len` UTF-8 bytes of timezone
  name (`tz_len = 0` means no timezone). The `unit_tag` selects the time unit — `0` nanoseconds,
  `1` microseconds, `2` milliseconds, `3` seconds, `4` days (`vortex-array/src/extension/datetime/unit.rs`
  `TimeUnit`). DateTimeParts uses one of ns/µs/ms/s; unit `Days` is not decodable here (a
  day-resolution timestamp does not split into sub-day parts). The Timestamp extension's **storage
  dtype is `Primitive(I64, <array nullability>)`** (`vortex-array/src/extension/datetime/timestamp.rs`
  `validate_dtype`) — the decoded instant is an `i64` in the selected unit.

**Buffers:** none (`array.rs` `DateTimeParts::nbuffers`).

**Metadata** (`DateTimePartsMetadata`, `array.rs` `DateTimePartsMetadata`):

| Field | Tag | Type | Meaning |
|-------|-----|------|---------|
| `days_ptype` | 1 | `PType` enum | Integer type of the `days` child. |
| `seconds_ptype` | 2 | `PType` enum | Integer type of the `seconds` child. |
| `subseconds_ptype` | 3 | `PType` enum | Integer type of the `subseconds` child. |

**Child slots** (3; wire child order equals slot order; `array.rs` `DateTimeParts::deserialize`, `DateTimePartsData::validate`):

| Slot | Name | Role | DType / length |
|------|------|------|----------------|
| 0 | `days` | Whole days since the epoch. | Integer (`days_ptype`), nullability equal to the array's; length `n`. |
| 1 | `seconds` | Whole seconds within the day. | Non-nullable integer (`seconds_ptype`), length `n`. |
| 2 | `subseconds` | Sub-second remainder, already expressed in the timestamp's time unit. | Non-nullable integer (`subseconds_ptype`), length `n`. |

**Decode** (row `i`): let `divisor` be the number of time-unit ticks per second —
`1_000_000_000` (ns), `1_000_000` (µs), `1_000` (ms), or `1` (s) (`canonical.rs`
`decode_to_temporal`). The instant, in the timestamp's time unit, is

```text
value[i] = days[i] · 86_400 · divisor  +  seconds[i] · divisor  +  subseconds[i]
```

(`seconds` is scaled by `divisor`; `subseconds` is added unscaled because it is already in the
target unit). Wrap `value` as a `Timestamp` with the extension's `unit` and `tz`. The `seconds` or
`subseconds` child may be a `Constant` array (a common optimisation, e.g. day-granularity data has a
constant `0`); decode it as any other array.

**Validity.** Delegates to the `days` child (`array.rs` `DateTimeParts::validity_child`). See [Validity](../encoding-format.md#validity).

(encoding-layout-DecimalByteParts)=
## `vortex.decimal_byte_parts` — Byte layout

Stores a decimal column as a small number of signed-integer "part" columns, most-significant part
first, so each part compresses independently. The current format emits a **single part** (`msp`, the
most-significant part) that holds the whole unscaled integer; the metadata reserves room for lower
parts that are not yet produced.

- **Wire ID:** `vortex.decimal_byte_parts` (`decimal_byte_parts/mod.rs` `DecimalByteParts::id`).
- **DType:** `Decimal(precision, scale)` (either nullability).

**Buffers:** none (`decimal_byte_parts/mod.rs` `DecimalByteParts::nbuffers`).

**Metadata** (`DecimalBytesPartsMetadata`, `decimal_byte_parts/mod.rs` `DecimalBytesPartsMetadata`):

| Field | Tag | Type | Meaning |
|-------|-----|------|---------|
| `zeroth_child_ptype` | 1 | `PType` enum | Signed-integer type of the `msp` child. |
| `lower_part_count` | 2 | `uint32` | Number of additional lower-part children. **Currently always 0**; a reader must reject any other value (`decimal_byte_parts/mod.rs` `DecimalByteParts::deserialize`). |

**Child slots** (1, `decimal_byte_parts/mod.rs` `DecimalByteParts::deserialize`; slot name `decimal_byte_parts/mod.rs` `SLOT_NAMES`):

| Slot | Name | Role | DType / length |
|------|------|------|----------------|
| 0 | `msp` | Most-significant part; holds the entire unscaled decimal integer. | Signed integer (`zeroth_child_ptype`, i.e. `i8`/`i16`/`i32`/`i64`), nullability equal to the array's; length `n` (`decimal_byte_parts/mod.rs` `DecimalBytePartsData::validate`). |

**Decode** (row `i`, when valid): the unscaled integer is `msp[i]` (sign-preserved). The logical
decimal value is `msp[i] · 10^(−scale)`, with `precision`/`scale` taken from the `Decimal` dtype.
Materialised canonically, this is a `Decimal` array whose storage integer buffer is `msp` and whose
decimal type is the array's dtype (`decimal_byte_parts/mod.rs` `to_canonical_decimal`).

*(Reserved multi-part design, for reference: with `lower_part_count > 0`, `msp` would hold the
high-order bits and the lower-part children the successively less-significant bits, concatenated to
form the full unscaled integer — e.g. an `i128` split as `msp = bits[127:64]`, `lower[0] =
bits[63:0]`. No writer emits this today.)*

**Validity.** Delegates to the `msp` child (child 0) (`decimal_byte_parts/mod.rs` `DecimalByteParts::validity_child`). See
[Validity](../encoding-format.md#validity).

(encoding-layout-ZigZag)=
## `vortex.zigzag` — Byte layout

ZigZag maps signed integers to unsigned so that small-magnitude values (of either sign) become
small unsigned values — friendlier to downstream unsigned-friendly codecs. The array holds a single
unsigned child; decoding applies the ZigZag inverse.

- **Wire ID:** `vortex.zigzag` (`array.rs` `ZigZag::id`).
- **DType:** a signed-integer `Primitive` (`I8`/`I16`/`I32`/`I64`), either nullability. The child is
  the unsigned integer of the **same bit width** and the same nullability (`array.rs`
  `ZigZag::deserialize`, `ZigZagData::dtype_from_encoded_dtype`).

**Buffers:** none (`array.rs` `ZigZag::nbuffers`).

**Metadata:** empty (a reader must reject non-empty metadata, `array.rs` `ZigZag::deserialize`).

**Child slots** (1, `array.rs` `ZigZag::deserialize`; slot name `array.rs` `SLOT_NAMES`):

| Slot | Name | Role | DType / length |
|------|------|------|----------------|
| 0 | `encoded` | ZigZag-encoded values. | Unsigned integer of the same bit width as the array's signed type, same nullability; length `n`. |

**Decode** (row `i`, when valid): with `u = encoded[i]` a `W`-bit unsigned value, the signed value is

```text
signed = (u >> 1) ^ (0 - (u & 1))
```

— a **logical** right shift of `u`, XOR-ed with the all-ones pattern when `u` is odd (i.e. two's
complement negation of the low bit). Equivalently: if `u` is even, `signed = u / 2`; if `u` is odd,
`signed = −(u + 1) / 2`. The result type is the signed integer of the same width
(`compress.rs` `zigzag_decode_primitive`; `zigzag` 0.1.0 `lib.rs` `ZigZag::decode`).

**Validity.** Delegates to the `encoded` child (`array.rs` `ZigZag::validity_child`). See [Validity](../encoding-format.md#validity).

(encoding-layout-Pco)=
## `vortex.pco` — Byte layout

Wraps the [pcodec](https://github.com/mwlon/pcodec) (pco) numeric codec. Values are compressed with
pco's *wrapped* format into one or more **chunks**, each split into **pages**; Vortex stores the pco
components as buffers and the framing (page value-counts) in metadata. Crucially, **only the valid
(non-null) values are compressed** — nulls are dropped before compression and re-inserted by
scattering on decode.

The byte layout of the pco components themselves — the wrapped-format `header`, each chunk's
`ChunkMeta`, and the page bytes — is defined **only** by the pcodec library and is not restated here;
it is the one encoding on these pages not fully decodable from the material referenced elsewhere in
this spec. Vortex pins `pco = "1.0.1"`; decode these components per the
[pcodec wrapped-format documentation](https://github.com/mwlon/pcodec/blob/main/docs/format.md) at
that version — just as `vortex.zstd` defers to [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878) and
`vortex.parquet.variant` defers to the versioned Parquet Variant spec.

- **Wire ID:** `vortex.pco` (`array.rs` `Pco::id`).
- **DType:** a `Primitive` pco supports: `F16`/`F32`/`F64`, `I16`/`I32`/`I64`, `U16`/`U32`/`U64`
  (either nullability; `array.rs` `number_type_from_ptype`). Note the pco stream is keyed by the
  array's ptype, not stored in metadata.

**Buffers** (dynamic): the `n_chunks` **chunk-meta** buffers first (one per chunk, in order),
followed by all **page** buffers in order (chunk 0's pages, then chunk 1's pages, …). Total buffer
count = `n_chunks + Σ pages_per_chunk` (`array.rs` `Pco::nbuffers`, `Pco::buffer_name`, `Pco::deserialize`).

| Range | Name | Contents |
|-------|------|----------|
| `0 .. n_chunks` | `chunk_meta_{k}` | pco `ChunkMeta` bytes for chunk `k`. |
| `n_chunks ..` | `page_{j}` | pco page bytes, flattened across chunks in order. |

**Metadata** (`PcoMetadata`, `lib.rs` `PcoMetadata`):

| Field | Tag | Type | Meaning |
|-------|-----|------|---------|
| `header` | 1 | `bytes` | The pco wrapped-format file header (needed to construct the decompressor). |
| `chunks` | 2 | repeated `PcoChunkInfo` | One entry per chunk. Each `PcoChunkInfo.pages` (tag 1) is a repeated `PcoPageInfo`, and each `PcoPageInfo.n_values` (tag 1, `uint32`) is the count of values encoded in that page. (`lib.rs` `PcoChunkInfo`, `PcoPageInfo`.) |

`n_chunks = len(chunks)`, and chunk `k` has `len(chunks[k].pages)` page buffers
(`array.rs` `Pco::deserialize`).

**Child slots** (0–1, `array.rs` `Pco::deserialize`; slot name `array.rs` `SLOT_NAMES`):

| Slot | Name | Role | DType / length |
|------|------|------|----------------|
| 0 | `validity` | Validity slot, stored **unsliced**. | Non-nullable `Bool`, length `n`. Omitted when the array carries no per-position nulls. |

**Decode:**

1. Decode validity (length `n`) per the [Validity](../encoding-format.md#validity) contract; let `V`
   be the number of valid positions.
2. Construct the pco decompressor from `header`. For each chunk `k` in order, initialise a chunk
   decompressor from `chunk_meta_{k}`; for each of its pages `j`, decode exactly
   `chunks[k].pages[j].n_values` numbers from the corresponding page buffer. Concatenating across all
   chunks/pages yields exactly `V` decoded numbers (the valid values, in row order).
3. **Scatter** into an output of length `n`: the `m`-th decoded number goes to the `m`-th valid
   position (per the validity mask); null positions receive a placeholder (e.g. zero)
   (`array.rs` `PcoData::decompress`, `PcoData::decompress_values_typed`; only valid values are
   compressed per `array.rs` `collect_valid`).

**Slice metadata & validity.** `Pco` (with `Zstd`) is one of the two encodings whose validity slot is
stored **unsliced** and paired with a `slice_start .. slice_stop` range, handled by the
[Validity](../encoding-format.md#validity) offset rule. That slice range is **in-memory state used
for lazy slicing and is not part of the serialised metadata**: a deserialised array is always
unsliced (`slice_start = 0`, `slice_stop = n`), so for a file reader the validity slot has length `n`
and the slice is the identity (`array.rs` `Pco::validity`, `PcoData::new`).

(encoding-layout-Zstd)=
## `vortex.zstd` — Byte layout

Wraps the Zstandard codec ([RFC 8878](https://www.rfc-editor.org/rfc/rfc8878)). Values are
compressed into one or more independently decompressible **frames**, optionally sharing a trained
**dictionary**; the frame lengths live in metadata. Like `Pco`, **only valid values are
compressed**, and nulls are re-inserted by scattering on decode. This mirrors the `Pco` layout.

- **Wire ID:** `vortex.zstd` (`array.rs` `Zstd::id`).
- **DType:** `Primitive`, `Binary`, or `Utf8` (either nullability; enforced in `array.rs`
  `ZstdData::validate`).

:::{note}
Do not confuse this with `vortex.zstd_buffers`, a separate `unstable_encodings`-gated encoding that
is out of scope for the stable spec.
:::

**Buffers** (dynamic): if a dictionary is present (`dictionary_size > 0`), buffer 0 is the
`dictionary`; the remaining buffers are the `frame_{f}` buffers in order. If no dictionary, all
buffers are frames (`array.rs` `Zstd::nbuffers`, `Zstd::buffer_name`, `Zstd::deserialize`).

| Position | Name | Contents |
|----------|------|----------|
| 0 (only if `dictionary_size > 0`) | `dictionary` | The Zstd dictionary bytes. |
| rest | `frame_{f}` | Zstd-compressed frame `f`, in order. |

**Metadata** (`ZstdMetadata`, `lib.rs` `ZstdMetadata`):

| Field | Tag | Type | Meaning |
|-------|-----|------|---------|
| `dictionary_size` | 1 | `uint32` | Byte length of the dictionary buffer, or `0` if none. |
| `frames` | 2 | repeated `ZstdFrameMetadata` | One entry per frame. `ZstdFrameMetadata.uncompressed_size` (tag 1, `uint64`) is the frame's decompressed byte length; `ZstdFrameMetadata.n_values` (tag 2, `uint64`) is the number of values it encodes. (`lib.rs` `ZstdFrameMetadata`.) |

`n_values` must be present in the current format. (The reader keeps a legacy fallback — `n_values ==
0` ⇒ derive the count from `uncompressed_size / byte_width` — but current writers always set it;
`array.rs` `ZstdData::decompress`.)

**Child slots** (0–1, `array.rs` `Zstd::deserialize`; slot name `array.rs` `SLOT_NAMES`):

| Slot | Name | Role | DType / length |
|------|------|------|----------------|
| 0 | `validity` | Validity slot, stored **unsliced**. | Non-nullable `Bool`, length `n`. Omitted when the array carries no per-position nulls. |

**Decode:**

1. Decode validity (length `n`); let `V` be the number of valid positions.
2. Decompress each frame with Zstd (initialising the decompressor with the `dictionary` buffer when
   present). Concatenating the frames' output in order yields the uncompressed value bytes for all
   `V` valid values, in row order.
3. Interpret the uncompressed bytes by dtype, then **scatter** into an output of length `n` (the
   `m`-th decoded value to the `m`-th valid position; null positions get a placeholder):
   - **`Primitive`:** the bytes are the raw little-endian values, `V · byte_width` bytes total
     (`array.rs` `ZstdData::decompress`).
   - **`Binary`/`Utf8`:** the bytes are length-prefixed — for each of the `V` valid values, a `u32`
     little-endian byte length followed by exactly that many data bytes, concatenated in order (the
     same length-prefixed layout Parquet uses) (`array.rs` `collect_valid_vbv`, `reconstruct_views`).

**Slice metadata & validity.** As for `Pco`: the validity slot is stored **unsliced** with a
`slice_start .. slice_stop` range applied per the [Validity](../encoding-format.md#validity) offset
rule, and that range is in-memory-only — a deserialised array is unsliced (`slice_start = 0`,
`slice_stop = n`), so a file reader sees a length-`n` validity slot and an identity slice
(`array.rs` `Zstd::validity`, `ZstdData::new`).

(encoding-layout-ParquetVariant)=
## `vortex.parquet.variant` — Byte layout

A lossless carrier for Arrow's canonical `arrow.parquet.variant` extension storage: semi-structured
[Parquet Variant](https://github.com/apache/parquet-format/blob/master/VariantEncoding.md) values,
with optional [shredding](https://github.com/apache/parquet-format/blob/master/VariantShredding.md).
Vortex stores the canonical `metadata` / `value` / `typed_value` components as child arrays and
carries their bytes verbatim; the binary interpretation of the `metadata` and `value` blobs is
defined by the Parquet Variant spec, not by Vortex.

- **Wire ID:** `vortex.parquet.variant` (`vtable.rs` `ParquetVariant::id`).
- **DType:** `Variant(nullability)`.

**Buffers:** none (`vtable.rs` `ParquetVariant::nbuffers`; a reader must reject any buffer,
`vtable.rs` `ParquetVariant::deserialize`).

**Metadata** (`ParquetVariantMetadataProto`, `vtable.rs` `ParquetVariantMetadataProto`):

| Field | Tag | Type | Meaning |
|-------|-----|------|---------|
| `has_value` | 1 | `bool` | Whether the `value` child is present. |
| `typed_value_dtype` | 2 | optional `DType` (proto) | The dtype of the `typed_value` child; present iff `typed_value` is present. |
| `value_nullable` | 3 | `bool` | Whether the `value` child's dtype is nullable. |

**Child slots** (4 slots; 1–4 serialised children; slot names `array.rs` `SLOT_NAMES`, dtype/length
constraints `vtable.rs` `ParquetVariant::validate`):

| Slot | Name | Role | DType / length |
|------|------|------|----------------|
| 0 | `validity` | Top-level row validity slot. | Non-nullable `Bool`, length `n`. Omitted when there are no per-position nulls. |
| 1 | `metadata` | Parquet Variant metadata (field-name dictionary) per row. **Always present.** | Non-nullable `Binary`, length `n`. |
| 2 | `value` | Unshredded Variant value bytes. Present iff `has_value`. | `Binary` (nullable = `value_nullable`), length `n`. |
| 3 | `typed_value` | Shredded representation. Present iff `typed_value_dtype` is set. | Any type (`typed_value_dtype`); may be primitive, `List`, or `Struct` with recursively shredded children; length `n`. |

**On-wire child ordering.** Children are packed in slot order, skipping absent optional slots:
`[validity?] , metadata , [value?] , [typed_value?]`. A reader determines the layout as follows. Let
`expected = 1 + has_value + (typed_value_dtype present)`. If the serialised child count equals
`expected`, there is **no** validity child (validity derives from the dtype's nullability) and the
first child is `metadata`. If the count equals `expected + 1`, the **first** child is the validity
array and `metadata` follows. `value` (if `has_value`) and then `typed_value` (if present) follow
`metadata` (`vtable.rs` `ParquetVariant::deserialize`).

**Row semantics.** Each row is a Parquet Variant value, interpreted from the present components by
the (`value`, `typed_value`) presence combination:

| `value` | `typed_value` | Meaning |
|---------|---------------|---------|
| null | null | Missing value (valid only for a shredded object field). |
| non-null | null | Unshredded — decode from `metadata` + `value`. |
| null | non-null | Perfectly shredded — decode from `typed_value`. |
| non-null | non-null | Partially shredded object — merge shredded fields (`typed_value`) with raw-only fields (`value`); duplicate field names are invalid writer output. |

At least one of `value` / `typed_value` is always present (`vtable.rs` `ParquetVariant::validate`,
`ParquetVariant::deserialize`).

**Validity.** Stored slot (`validity`, child 0), row-aligned — the array's top-level row validity is
read directly from this slot (mechanism 1, `validity.rs` `ParquetVariant::validity`), with **no**
cross-component combine step. The component
children (`metadata` / `value` / `typed_value`) carry their own validity for their own values,
independently of the top-level row validity. See [Validity](../encoding-format.md#validity).
