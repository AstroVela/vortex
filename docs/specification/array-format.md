# Array Format

Vortex uses the **same binary representation for arrays in memory, on disk, and over the wire**. A
serialized array is the unit that a [file](file-format.md) segment or an [IPC](ipc-format.md) message
ultimately carries, and it is laid out so that its buffers can be used as in-memory arrays without
copying.

This page specifies that representation precisely enough to locate and decode any buffer from the
bytes alone. What the bytes *inside* a node mean — encoding by encoding — is the separate
[Encoding Format](encoding-format.md) reference; this page defines the container those encodings sit
in.

## Structure

A serialized array consists of two parts: a FlatBuffer describing the array tree, and a sequence of
data buffers.

The FlatBuffer is a single `Array` message. It holds a tree of `ArrayNode`s (the `root` node plus
its transitively-nested `children`) and one flat `buffers` table shared by the whole tree. Each
`ArrayNode` records:

- The **encoding** — an interned `u16` index into the encoding registry (see
  [Decoding a node](#decoding-a-node)).
- Array-specific **metadata** bytes, interpreted according to the node's encoding.
- Its **children**, embedded inline as further `ArrayNode`s.
- Its **buffers** — a list of `u16` indices into the shared buffer table.
- Optional **statistics** (min, max, null count, sort order, …).

A node stores **neither its length nor its dtype** — both are supplied top-down by the parent,
ultimately from the file or IPC schema. This is why decoding always takes the logical length `n` and
the `DType` as inputs rather than parsing them from the node.

The data buffers are written **first**; the `Array` FlatBuffer is appended as a suffix after them,
followed by a little-endian `u32` giving the FlatBuffer's length (see [Wire layout](#wire-layout)).
Serialization takes a `SerializeOptions`: when zero-copy padding is enabled (the default for files),
padding is inserted before each buffer to meet its alignment; serializers that disable it — such as
the default IPC encoder — omit the padding.

## FlatBuffer schema

The array tree and buffer table are defined by `array.fbs`:

:::{literalinclude} ../../vortex-flatbuffers/flatbuffers/vortex-array/array.fbs
:start-at: /// An Array describes
:::

`Array` is the schema's `root_type`: `root` is the top `ArrayNode`, and `buffers` is the single,
tree-global buffer table (see [The buffer table](#the-buffer-table)). `ArrayNode` is a FlatBuffer
**table** — its fields are optional and the format can add new ones without breaking older readers.
`Buffer` is a FlatBuffer **struct**: a fixed, inline, 8-byte record with no vtable, which by
construction cannot gain or lose fields (see [The Buffer struct](#the-buffer-struct)).

## Wire layout

On the wire, a serialized array is:

```
[pad?] [buffer 0] [pad?] [buffer 1] ... [pad?] [Array flatbuffer] [u32 flatbuffer length, LE]
```

The data-buffer region comes first, the `Array` FlatBuffer is the suffix, and the final 4 bytes are
the little-endian `u32` length of that FlatBuffer. A reader recovers the pieces from the end inward:
read the trailing `u32` length `L`; the FlatBuffer occupies the `L` bytes ending 4 bytes before the
end; everything before it is the data-buffer region (`vortex-array/src/serde.rs` `SerializedArray::try_from`).

Each `[pad?]` is alignment padding inserted **only when zero-copy padding is enabled**: before each
data buffer to meet its alignment, and once more before the FlatBuffer to 8-align it. When padding is
disabled (the IPC encoder's default), the `[pad?]` slots are empty and every `Buffer.padding` is `0`.

## The buffer table

`Array.buffers` is a single vector of `Buffer` structs describing every data buffer in the whole node
tree. It is a two-level indirection: a node names its buffers by index into this shared table, and
each entry says where its bytes live in the data-buffer region.

### The Buffer struct

Because `Buffer` is a FlatBuffer *struct* rather than a table, each entry is a fixed **8-byte** inline
record — the schema comment calls it a "packed 64-bit struct"
(`vortex-flatbuffers/flatbuffers/vortex-array/array.fbs` `Buffer`) — stored back-to-back with no offsets or
vtable. Its four fields, in declaration order, describe one data buffer:

| Byte offset | Field | Type | Size | Meaning |
|-------------|-------|------|------|---------|
| 0 | `padding` | `uint16` | 2 | Number of pad bytes written **immediately before** this buffer. |
| 2 | `alignment_exponent` | `uint8` | 1 | The buffer's minimum alignment, as an exponent of 2 (alignment = `2^alignment_exponent` bytes). |
| 3 | `compression` | `Compression` (`uint8`) | 1 | Compression codec applied to the buffer (see [Buffer compression](#buffer-compression)). |
| 4 | `length` | `uint32` | 4 | The buffer's *stored* length in bytes (the compressed length if compressed). |

All scalars are little-endian. The record's total size (8 bytes) is fixed by the schema, so a reader
indexes the table by multiplying the index by 8 rather than following an offset.

### Locating buffer bytes

A `Buffer` descriptor gives a buffer's length and its immediately-preceding padding, but **not** an
absolute offset — a reader recovers the offset by walking the table in order. Within the data-buffer
region (the prefix identified in [Wire layout](#wire-layout)), buffers are laid out
`[pad_0][buffer_0][pad_1][buffer_1]…`, where each `pad_j` is exactly `buffers[j].padding` bytes. To
locate buffer `k` (an absolute index into `Array.buffers`):

```
offset = 0
for j in 0 .. k:
    offset += buffers[j].padding      # pad bytes written before buffer j
    offset += buffers[j].length       # buffer j's own bytes
offset += buffers[k].padding          # pad bytes written before buffer k
bytes = region[offset .. offset + buffers[k].length]
```

Equivalently: accumulate `padding_j + length_j` over every `j < k`, then add `padding_k`. This is
exactly the reference reader's walk, which advances a running `offset` by each descriptor's `padding`
then `length` (`vortex-array/src/serde.rs` `SerializedArray::from_flatbuffer_and_segment_with_overrides`), mirroring the writer that emits `padding` zero
bytes before each buffer and records that count in the descriptor
(`vortex-array/src/serde.rs` `ArrayRef::serialize`). When padding is disabled every `padding` is `0`, so the buffers
are simply concatenated.

Each buffer is then aligned to `2^alignment_exponent`. On the padded (file) path the padding already
places the buffer at an aligned offset, so its bytes can be used in place; otherwise a reader may have
to copy to satisfy alignment (see [Zero-copy design](#zero-copy-design)).

### Tree-global indices

There is **one** buffer table for the entire node tree — `Array.buffers` — not one per node. Buffer
indices are assigned **pre-order**: a node's own buffers come first, immediately followed by all of
its first child's buffers (recursively), then the next child's, and so on. Concretely, a node
occupying the index range `[b, b + m)` for its `m` own buffers hands its first child the starting
index `b + m`, and each subsequent child starts after its predecessor's entire *recursive* buffer
count (`vortex-array/src/serde.rs` `ArrayNodeFlatBuffer::try_write_flatbuffer`; the writer collects the actual buffer bytes by the same
pre-order tree walk, `vortex-array/src/serde.rs` `ArrayRef::serialize`).

Each `ArrayNode.buffers` list therefore holds that node's **absolute** indices into the shared table.
This indirection is the bridge between the per-encoding pages and the bytes:

:::{important}
When an [Encoding Format](encoding-format.md) section says "**buffer `i`**", it means the *node-local*
position `i` — that is, `ArrayNode.buffers[i]`. Resolve that to an absolute index into
`Array.buffers`, then locate its bytes with the walk in [Locating buffer bytes](#locating-buffer-bytes).
:::

A node's own indices are contiguous and ascending, which lets a reader slice all of its buffers as one
range (`vortex-array/src/serde.rs` `SerializedArray::collect_buffers`), but the format only requires that each
`ArrayNode.buffers[i]` be a valid index into `Array.buffers`.

### Buffer compression

`Buffer.compression` selects a per-buffer compression codec from the `Compression` enum
(`vortex-flatbuffers/flatbuffers/vortex-array/array.fbs` `Compression`):

| Value | Name | Meaning |
|-------|------|---------|
| 0 | `None` | The buffer's bytes are stored verbatim. |
| 1 | `LZ4` | The buffer's bytes are LZ4-compressed. |

This is distinct from Vortex's array *encodings* (which are lossless logical re-representations of the
values): it is an opaque byte-level codec applied to an already-encoded buffer's bytes. When
`compression = LZ4`, `Buffer.length` is the **compressed** byte count — the value the offset walk in
[Locating buffer bytes](#locating-buffer-bytes) uses — and a reader must LZ4-decompress those bytes
into a freshly allocated, aligned buffer before using them. Such a buffer therefore **cannot be used
zero-copy**: the in-place slice holds compressed bytes, so a decode-and-allocate step is unavoidable
for it. `None` buffers keep the zero-copy property.

:::{note}
Reference-implementation status. Buffer-level `LZ4` is defined and reserved in the schema, but the
reference writer currently emits `None` for every buffer (`vortex-array/src/serde.rs` `ArrayRef::serialize`) and the
reference slice path does not decompress (`vortex-array/src/serde.rs` `SerializedArray::from_flatbuffer_and_segment_with_overrides`). The framing a decoder
would need for `LZ4` — in particular where the *uncompressed* length comes from — is not yet pinned by
the schema (the node-level `uncompressed_size_in_bytes` statistic is advisory, not a per-buffer
field). Treat `LZ4` as forward-compatible reservation: a reader **must** recognise the enum value and
**must** fail loudly rather than hand compressed bytes to a decoder expecting raw ones.
:::

## Node statistics

Each `ArrayNode` may carry an `ArrayStats` table
(`vortex-flatbuffers/flatbuffers/vortex-array/array.fbs` `ArrayStats`) summarising the node's values:

- `min`, `max`, `sum` — each a **Protobuf-serialized `ScalarValue`** (a bare value, no dtype). `min`
  and `max` decode against the node's `DType`; `sum` decodes against a **widened accumulator dtype
  derived from** the node's `DType` (not the `DType` verbatim). The [Scalar Format](scalar-format.md)
  specifies how to decode the `ScalarValue` bytes **once that dtype is known** — it does not itself
  define the widening; because statistics are advisory, a reader that does not reproduce the `sum`
  widening may skip `sum`. `min` and `max` each pair with a `Precision`
  (`vortex-flatbuffers/flatbuffers/vortex-array/array.fbs` `Precision`): `Exact` means the bound is the
  true extreme, while `Inexact` means it is only a valid lower/upper bound that may not be tight — a
  reader pruning on it must treat it conservatively.
- `is_sorted`, `is_strict_sorted`, `is_constant` — tri-state booleans whose default is `null`
  (unknown).
- `null_count`, `nan_count`, `uncompressed_size_in_bytes` — `uint64`s whose default is `null`
  (unknown).

Every field is optional; an absent field — or an absent `ArrayStats` altogether — means "unknown",
never a computed value. **Statistics are advisory.** They drive pruning and pushdown decisions (for
example skipping a node whose `max` is below a filter bound, or short-circuiting a `min`/`max`
aggregate), but a reader may ignore them entirely and still decode correct data — the reference reader
only *populates* a node's statistics set from them when they are present
(`vortex-array/src/serde.rs` `SerializedArray::decode`). A writer must never store a statistic it cannot prove.

## Decoding a node

Decoding turns a parsed node into a typed array. The node carries no length or dtype, so both are
inputs, supplied top-down from the schema. For one node:

1. **Resolve the encoding.** `ArrayNode.encoding` is a `u16` index into the encoding registry carried
   by the enclosing [file](file-format.md) or [IPC stream](ipc-format.md) — the interned list that
   maps indices to registry encoding IDs. Look the index up to obtain the ID (for example
   `vortex.primitive`). An index with no registry entry is an error, and an ID the reader does not
   implement **must** fail loudly rather than be guessed — unless the reader explicitly opts into
   opaque foreign passthrough (`session.allows_unknown()`), which preserves the node's metadata,
   buffers, and children verbatim without interpreting them (`vortex-array/src/serde.rs` `SerializedArray::decode`).
2. **Interpret the components per that encoding.** The [Encoding Format](encoding-format.md) page for
   the resolved ID defines how to read the node's `metadata` bytes, its referenced buffers (via the
   [tree-global index resolution](#tree-global-indices) above), and its children — using the supplied
   `n` and `DType`. Children are decoded recursively by the same procedure; the parent encoding
   derives each child's dtype and length.
3. **Optionally apply statistics.** If `stats` is present a reader may load it (see
   [Node statistics](#node-statistics)) or ignore it.

This contract is language-neutral: the FlatBuffer supplies structure and byte locations, and the
per-encoding pages supply meaning. No step depends on a particular runtime's types.

## Why FlatBuffers

Vortex uses FlatBuffers rather than Protocol Buffers or a custom binary format because FlatBuffers
support O(1) random access into the serialized data without parsing the entire message. This matters
for wide schemas where only a few columns are accessed per query — the reader can jump directly to
the relevant node without deserializing the rest.

All FlatBuffers in Vortex are aligned to 8 bytes. Schema definitions live in the `vortex-flatbuffers`
crate and cover arrays, layouts, the file footer, and IPC messages.

## Zero-copy design

The alignment and padding system is designed so that serialized buffers can be used directly as
in-memory arrays without copying — *when the padding is present and the buffer is uncompressed*. This
is what the on-disk [File Format](file-format.md) is built around: it writes the padding each buffer's
alignment requires, so when the I/O subsystem reads a segment into an aligned buffer, a
[file reader](reading-a-file.md) can hand the fetched bytes straight to an array — no reallocation,
no copy.

The [IPC format](ipc-format.md) reuses the same array representation but its default encoder omits the
inter-buffer padding, so a receiver may need to copy or realign some buffers rather than use them in
place. Zero-copy is therefore a property of the *padded, uncompressed* layout (the file path), not of
the array representation in every context.
