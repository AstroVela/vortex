# DType Format

This page is the serialization reference for Vortex's logical type system. For what the types
*mean* — the distinction between logical and physical types, the full list of built-in types, and
how Vortex compares to Arrow — see the [DTypes concept page](../concepts/dtypes.md).

## Model

A `DType` is a single logical type drawn from a closed `union` of type variants. Three properties of
the encoding are worth calling out, because they differ from many other columnar formats:

- **Nullability is part of the type.** Each built-in value variant carries its own `nullable` flag,
  rather than nullability living on a separate schema/field object. `Null` is its own variant. The
  exception is `Extension`, which has no `nullable` field — its nullability is inherited from its
  `storage_dtype`.
- **There is no separate schema type.** Columnar data is just a `Struct_` whose `names` and
  `dtypes` line up positionally. A file can equally hold a bare `Primitive` or `Utf8` at the root —
  see the [File Format](file-format.md).
- **Nested types embed their children.** `List`, `FixedSizeList`, and `Struct_` contain child
  `DType`s directly, so a type is a self-contained tree. `Extension` wraps a `storage_dtype` with an
  `id` and `metadata` bytes that together narrow its domain (for example, `vortex.date` over an
  `I32`). The `metadata` bytes are opaque **at the DType layer**, but Vortex's built-in extensions
  define concrete metadata formats and required storage dtypes — see
  [Built-in extension types](#built-in-extension-types).

## Serializations

Vortex serializes DTypes two ways, for two different jobs:

- **FlatBuffer** (`dtype.fbs`) is the canonical, zero-copy serialization. It is what the
  [File Format](file-format.md) stores in its root `DType` segment (when the schema is embedded
  rather than supplied externally) and what the [IPC format](ipc-format.md) sends in a
  `DTypeMessage`. It is the schema-on-the-wire encoding.
- **Protobuf** (`dtype.proto`) mirrors the same model for contexts where a DType is embedded as
  *metadata* inside something else — notably compute expressions and the metadata of arrays and
  extension types. The two schemas describe the same logical types.

### FlatBuffer definition

The `Type` union is the heart of the schema; `DType` is just a wrapper around it. New variants are
appended to the union to preserve backward compatibility — note that `FixedSizeList` sits after
`Extension` for exactly this reason.

:::{literalinclude} ../../vortex-flatbuffers/flatbuffers/vortex-dtype/dtype.fbs
:start-at: enum PType
:::

### Protobuf definition

:::{literalinclude} ../../vortex-proto/proto/dtype.proto
:language: protobuf
:start-at: syntax =
:::

## Built-in extension types

An `Extension`'s `metadata` bytes are opaque at the DType layer, but Vortex registers four
**built-in** extensions by default (`vortex-array/src/dtype/session.rs` `DTypeSession::default`),
each with a concrete metadata format and a required `storage_dtype`. A reader needs both to
*interpret* such a column (as opposed to merely decoding the raw storage values):

| `id` | Metadata bytes | Storage dtype | Meaning |
|------|----------------|---------------|---------|
| `vortex.date` | `[unit_tag: u8]` | `Days` → `I32`, `Milliseconds` → `I64` | days / ms since the Unix epoch; only these two `TimeUnit`s are valid (`vortex-array/src/extension/datetime/date.rs` `date_ptype`, `serialize_metadata`) |
| `vortex.time` | `[unit_tag: u8]` | `Nanoseconds`/`Microseconds` → `I64`, `Milliseconds`/`Seconds` → `I32` | time-of-day (`vortex-array/src/extension/datetime/time.rs` `time_ptype`) |
| `vortex.timestamp` | `[unit_tag: u8][tz_len: u16 LE][tz: UTF-8]` | `I64` | instant; full layout in the [DateTimeParts encoding](encoding-format/misc.md) (`vortex-array/src/extension/datetime/timestamp.rs` `serialize_metadata`) |
| `vortex.uuid` | empty (any version) or `[version: u8]` | `FixedSizeList(non-nullable U8, 16)` | 16 raw bytes per value (`vortex-array/src/extension/uuid/vtable.rs` `serialize_metadata`) |

The `unit_tag` byte is the `TimeUnit` enum — `0` nanoseconds, `1` microseconds, `2` milliseconds,
`3` seconds, `4` days (`vortex-array/src/extension/datetime/unit.rs` `TimeUnit`).

**Unknown extension ids.** An `Extension` whose `id` is not registered is preserved as an opaque
**foreign** extension (`vortex-array/src/dtype/extension/foreign.rs`): a reader keeps the
`storage_dtype`, `id`, and `metadata` bytes verbatim and can decode the raw storage values, but
cannot interpret the extension's domain semantics — the same opaque-passthrough posture the array
layer takes for an unrecognised encoding.

## Field names

A `Struct_` stores its field names as a list of FlatBuffer strings (`names: [string]` in the
FlatBuffer definition above), and in memory a field name is just an `Arc<str>` whose contents are
unconstrained. A FlatBuffer string is *length-prefixed*: it records an explicit byte length and may
therefore hold any UTF-8 bytes, including an interior NUL (`U+0000`) or other control characters.
That is strictly more permissive than what a `DType` can portably carry across every serialization
and interop boundary, so the format constrains field names as follows.

:::{important}
A struct field name **MUST NOT** contain a NUL byte (`U+0000`) — a hard, structural rule (the Arrow C
Data Interface cannot represent it; see below). A writer **MUST** enforce this when a `DType` is
constructed or serialized, and a reader or exporter that encounters a NUL-bearing name **MUST** surface
a clean error — it **MUST NOT** abort the process. Other control characters **SHOULD** be avoided for
portability, but are not structurally forbidden — the reference implementation round-trips them today.
:::

### The Arrow C Data Interface constraint

The binding driving this rule is the [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html),
which every Vortex language binding uses to hand columnar data to Arrow-based consumers. In that
interface a field or schema name is a **NUL-terminated `const char*`** — the string ends at the
first NUL byte and has no separate length. A Vortex FlatBuffer string, by contrast, is
length-prefixed. The length-prefixed form is the more expressive of the two: it can represent a
name containing an interior NUL, which the NUL-terminated form cannot. A whole class of otherwise
well-formed names is therefore *not round-trippable* through the C Data Interface, and Vortex adopts
the more restrictive rule so that any `DType` a writer produces can be exported without loss.

NUL is the hard case. When a name carrying an interior NUL reaches the C Data export path, the
conversion into a NUL-terminated C string fails, and — because the failure happens inside a
non-unwinding FFI callback (the Arrow C stream's schema callback is an `extern "C"` function) — the
panic cannot unwind across the boundary and instead aborts the whole process with `SIGABRT`. A
single malformed field name can therefore crash any embedding of a Vortex binding, which is why the
constraint is enforced when the name is constructed. Other control characters are byte-representable in a
`const char*` (a non-NUL control byte does not terminate the string), so they do **not** trigger this
failure — the reference implementation round-trips them today. They remain a portability hazard across
the polyglot read surface, and Vortex's own display path escapes them, so the format **recommends
against** them (`SHOULD` avoid) without structurally forbidding them.

### Enforcement and clean failure

Enforcement belongs at the writer: an implementation **MUST** reject an illegal name when a `DType`
is constructed or serialized, before it can be persisted or sent on the wire, so that no downstream
reader is ever handed a name it cannot represent. Where a name nevertheless reaches an
export/conversion boundary — for example converting a `DType` to an Arrow schema or a target
engine's type — the exporter **MUST** validate the name and return a recoverable error naming the
offending field, never panic or abort. The DuckDB struct-type exporter is the model here: it maps a
NUL-bearing name to a `VortexError` rather than unwrapping the conversion failure.

### Metadata-key constraints (out of scope)

The analogous constraints on *metadata keys* (which are subject to the same C-Data and
interoperability limits) are intentionally not specified here. They will be defined alongside the
in-flight column-and-file metadata proposal (`vortex-data/rfcs#62`), which introduces the metadata
mechanism they would govern.
