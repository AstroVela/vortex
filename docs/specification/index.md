# Specification

Vortex defines serialization formats for columnar data on disk and over the wire. The on-disk
**File Format** is stable (since 0.36.0); the over-the-wire **IPC Format** is documented but still
unstable and subject to change. This section specifies both — the byte layouts, FlatBuffer and Protobuf schemas, and the procedure
for reading a file — with one documented exception: the layout tree's wire format is not yet
specified here (see [Reading a File](reading-a-file.md)).

If you just want to read or write Vortex data from a program, you don't need these pages — reach for
the [language bindings](../api/index.md) or a [query-engine integration](../user-guide/index.md)
instead. The specification is for reader/writer implementers, format debuggers, and anyone porting
Vortex to a new language.

## The format stack

The formats build on one another:

- **[Reading a File](reading-a-file.md)** — the end-to-end procedure for turning a `.vortex` file
  into arrays. Start here if you want to understand how a read actually works.
- **[File Format](file-format.md)** — the on-disk container: magic numbers, postscript, footer, and
  compatibility guarantees.
- **[Array Format](array-format.md)** — the shared binary representation of a single array, used
  identically in memory, on disk, and over the wire.
- **[Encoding Format](encoding-format.md)** — what the bytes inside a single array node *mean*,
  encoding by encoding: the validity contract, and (as they land) each encoding's buffer/metadata layout.
- **[IPC Format](ipc-format.md)** — the message-oriented wire protocol that streams arrays between
  processes.
- **[DType Format](dtype-format.md)** and **[Scalar Format](scalar-format.md)** — the schema and
  scalar-value encodings referenced by the above.
- **[Row Encoding](row-encoding.md)** — a separate, **experimental** byte-sortable row-key format
  (used internally for sort keys), *not* part of the file/IPC format stack above.

```{toctree}
---
maxdepth: 2
hidden:
---

reading-a-file
file-format
array-format
encoding-format
ipc-format
dtype-format
scalar-format
row-encoding
```
