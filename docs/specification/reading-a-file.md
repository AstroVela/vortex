# Reading a Vortex File

This page walks through how a reader turns the bytes of a `.vortex` file into arrays. It is the
procedural companion to the [File Format](file-format.md) byte-layout reference: the File Format page
tells you *what the bytes are*, this page tells you *what to do with them*.

:::{seealso}
If you only want to read or write Vortex files from a program, you don't need any of this — use the
language bindings ([Python](../api/python/index.rst), [Rust](https://docs.rs/vortex),
[Java](../api/java/index.rst)) or a [query-engine integration](../user-guide/index.md). This page is
for people implementing a reader, debugging the format, or porting Vortex to a new language.
:::

## The easy part and the hard part

Here is the honest framing, because it shapes the whole format:

**Reading the bytes is not hard.** A Vortex file is a magic number, a small *postscript* that points
at a footer, a footer that describes where every data segment lives, a root *dtype* (the schema), and
a tree of *layouts* that says how those segments compose into arrays. Open the file, read ~64&nbsp;KiB
from the end, follow a handful of offsets, and you can materialize any column. A competent implementer
can write a correct-but-naive reader in an afternoon.

**Getting *high performance* out of the file is the hard part** — and it is the entire reason the
format is shaped the way it is. A fast reader must:

- **prune** whole row ranges it can prove are irrelevant, using the per-zone statistics stored
  alongside the data (min/max/null-count zone maps);
- **push down projections** so it only fetches the columns a query touches — cheap when the writer
  has laid out columns as independent layouts (the default), so a column's segments can be fetched
  without reading its siblings;
- **push down filters** and evaluate them against *compressed* data where possible, before
  decompressing;
- **schedule I/O** well: coalesce nearby segment reads, issue them concurrently, and prefetch — the
  difference between one large sequential read and thousands of tiny random ones.

Everything below describes the naive read path. The pruning, pushdown, and I/O-scheduling machinery
that makes it fast lives in [Layouts](../concepts/layouts.md), the
[Scan API](../concepts/scanning.md), and the [I/O subsystem](../developer-guide/internals/io.md).

## Step 1 — Read the trailer

A Vortex file begins and ends with the 4-byte magic number `VTXF`. The last 8 bytes form the
`EndOfFile` struct:

```
...        segments of binary data (+ optional inter-segment padding)
...        postscript
<2 bytes>  u16 version tag        (little-endian)
<2 bytes>  u16 postscript length  (little-endian)
<4 bytes>  magic number 'VTXF'
```

All manually-framed integers in the format — the trailer `u16`s here, the `u32` length suffix on a
serialized array, and the `u32` header length in an [IPC](ipc-format.md) message — are
**little-endian**. (The FlatBuffers themselves are little-endian by the FlatBuffer spec.)

The postscript is guaranteed never to exceed `MAX_POSTSCRIPT_SIZE` = `u16::MAX - 8` = **65527** bytes.
Because of that bound, a reader's **first read defaults to `MAX_POSTSCRIPT_SIZE + EOF_SIZE` = 65535
bytes from the end of the file** (the `EndOfFile` trailer is 8 bytes), which is guaranteed to cover
both the trailer and the entire postscript in a single round trip. Validate the trailing magic, read
the version tag, and slice out the postscript using its length.

## Step 2 — Parse the postscript

The postscript is a FlatBuffer locating four segments by offset/length (encryption and compression
specs are inlined here so they never require a prior fetch):

:::{literalinclude} ../../vortex-flatbuffers/flatbuffers/vortex-file/footer.fbs
:start-after: [postscript]
:end-before: [postscript]
:::

- **`layout`** (required) — the root [Layout](../concepts/layouts.md) FlatBuffer.
- **`footer`** (required) — the dictionary-encoded *registry* (segments, encodings, layouts,
  compression, encryption).
- **`dtype`** (optional) — the root [DType](../concepts/dtypes.md). Optional because large schemas can
  be shared or fetched externally rather than embedded.
- **`statistics`** (optional) — file-level per-field statistics for whole-file pruning.

:::{important}
The root `DType` is required to bind the layout and decode arrays, even though the `dtype` *segment*
is optional. If the postscript carries a `dtype` segment, fetch and parse it now; if it does not, the
reader **must** obtain the root `DType` from the caller (or an external catalog) before proceeding.
A file with no embedded dtype and no externally-supplied dtype cannot be read.
:::

The two-round-trip design is deliberate: keeping the postscript small and fixed-bounded means the
footer and the (possibly large) dtype live in their own segments, so a reader needs at most two round
trips — one for the tail, one for the segments it points to — rather than three.

## Step 3 — Load the footer registry

The footer is the lookup table the rest of the read depends on:

:::{literalinclude} ../../vortex-flatbuffers/flatbuffers/vortex-file/footer.fbs
:start-after: [footer]
:end-before: [footer]
:::

It holds dictionary-encoded specs that the layout tree references by index:

- **`segment_specs`** — for each `SegmentId`, the `offset`, `length`, and `alignment_exponent` of a
  byte range in the file. This is the indirection that lets the same layout tree be backed by a local
  file, an object store, or an in-memory cache.
- **`array_specs`** / **`layout_specs`** — globally-unique string IDs resolved against the Vortex
  registry at read time to find the encoding/layout implementation.
- **`compression_specs`** / **`encryption_specs`** — per-segment schemes (reserved; see
  [File Format](file-format.md)).

## Step 4 — Bind the layout tree to a segment source

Deserialize the root `Layout` FlatBuffer into a tree of nodes — each carries a layout ID (into
`layout_specs`), a row count, metadata, child layouts, and `SegmentId`s. Binding this tree to a
*segment source* (a thing that can fetch a `SegmentId`'s bytes) yields a `LayoutReader` that fetches
data lazily. The shape of the tree — struct-of-columns, chunked row groups, zone maps, dictionaries —
is entirely the writer's choice; see [Layouts](../concepts/layouts.md).

:::{note}
The `Layout` FlatBuffer's byte-level wire format (the `layout.fbs` schema — a node's `encoding`
index, `row_count`, opaque `metadata`, child layouts, and `segments`) is **not yet specified in this
Specification section**; it is a deferred follow-up. [Layouts](../concepts/layouts.md) describes the
tree conceptually, but a byte-exact clean-room reader of the layout tree cannot be built from the
spec alone today. This is the one boundary **required to decode data** still defined only by the
reference implementation. (The advisory `statistics` segment is likewise not byte-specified here, but
a reader never needs it to decode values.)
:::

## Step 5 — Resolve segments and decode

To materialize a region:

1. Walk the layout tree to the nodes covering the requested columns and row range, resolving each
   node's layout encoding through **`layout_specs`** (this is the layout registry — distinct from
   the array registry in the next step).
2. For each referenced `SegmentId`, look up its `SegmentSpec` in the footer to get `offset`,
   `length`, and required alignment.
3. Fetch those byte ranges. A leaf (flat) layout's segment is not a bare buffer — it holds a
   **serialized array**: the data buffers followed by an Array FlatBuffer and a trailing `u32`
   length (see [Array Format](array-format.md)). The buffers within it are aligned so they can be
   used **zero-copy** as in-memory arrays.
4. Parse that serialized array, resolving each `ArrayNode`'s encoding through **`array_specs`** (the
   array registry), and decode, recursing into child nodes and child layouts.

A naive reader fetches one segment per read and stops here — and it will be correct. A fast reader
overlaps steps 1–3 across columns and chunks, prunes nodes using zone statistics before fetching, and
coalesces adjacent segments into larger reads. That is the work the [Scan API](../concepts/scanning.md)
exists to do.
