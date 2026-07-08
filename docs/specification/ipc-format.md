# IPC Format

The IPC format wraps [serialized arrays](array-format.md) in a message-oriented protocol for
streaming between processes. It is used for inter-process communication and can serve as the wire
protocol for remote source execution in the [Scan API](../concepts/scanning.md).

:::{note}
The IPC format is unstable and subject to change. It does not yet support shared arrays (e.g. a
dictionary shared across multiple chunked arrays), which limits its efficiency for certain workloads.
This is an area of active development. Unlike the IPC format, the [File Format](file-format.md) is
stable.
:::

## Message framing

Each message is a length-prefixed FlatBuffer header followed by a body:

```
[u32 header length] [flatbuffer Message header] [body bytes]
```

The `u32` header length is little-endian. The `Message` header carries the fields an implementer
needs to consume the stream:

- **`version`** (`MessageVersion`, default `V0`) — the message format version.
- **`header`** — a union selecting one of the message types below.
- **`body_size`** (`uint64`) — the exact number of body bytes that follow the header. A reader uses
  this to know how much to consume before the next message.

## Message types

The `header` union selects one of three message types:

- **`ArrayMessage`** — a [serialized array](array-format.md) (in the body) with its `row_count` and
  the list of `encodings` (encoding IDs) the array references.
- **`BufferMessage`** — a raw buffer with an `alignment_exponent`, used for transferring individual
  segments.
- **`DTypeMessage`** — signals that the body is a serialized [dtype](dtype-format.md), used to
  communicate the schema before data transfer begins.
