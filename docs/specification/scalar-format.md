# Scalar Format

A **scalar** is a single typed value: a [`DType`](dtype-format.md) paired with a value drawn from
that type's domain. Scalars appear wherever Vortex needs to represent one value rather than an
array — in compute and pushdown **expressions** (constants and literals) and as **array metadata**
(for example the base and step of a sequence array, or the value of a constant array). Statistics
(per-zone minima and maxima, etc.) store a bare `ScalarValue` rather than a full `Scalar`: the
`dtype` is supplied by the array/field the statistic belongs to, so it is not repeated per value.

## Model

A `Scalar` carries its `dtype` alongside its `value`, so it is self-describing. The `ScalarValue` is
a `oneof` over the possible value kinds, paralleling the logical types in the
[DType format](dtype-format.md). A few encoding details:

- **Floats.** `f32_value` and `f64_value` store IEEE-754 values directly; `f16_value` stores the
  raw 16-bit half-precision bits in a `uint64`, since Protobuf has no native half type.
- **Nested values.** `ListValue` holds a repeated sequence of `ScalarValue`s. Note the protobuf has
  no dedicated struct value, and the current deserializer accepts `ListValue` only for a `List`
  dtype — `FixedSizeList` and `Struct` scalar round-trips through this protobuf are not yet
  supported.
- **Variant.** A variant scalar carries a row-specific nested `Scalar`.
- **Null.** Absence is represented by `null_value`, regardless of the declared `dtype`.

Scalars are serialized with Protobuf rather than FlatBuffers, because they travel as embedded
metadata inside arrays and expressions rather than on the zero-copy file/IPC path — the same split
described in the [DType format](dtype-format.md#serializations).

## Protobuf definition

:::{literalinclude} ../../vortex-proto/proto/scalar.proto
:language: protobuf
:start-at: syntax =
:::

## Value encoding

The `oneof` above is deliberately *narrower* than the [DType](dtype-format.md) system it serves:
several logical types share a single arm, and one arm (`bytes_value`) serves three unrelated types.
A `ScalarValue` is therefore **not self-describing** — the same wire bytes decode to different
values depending on the `DType`. Decoding is well-defined only *with* that `DType`, which is supplied
by context — the enclosing `Scalar`'s `dtype`, or the array/field that a statistic (or a `Constant`,
`Sequence`, or `FoR` metadata value) belongs to. The overloaded arms — `int64_value`, `uint64_value`,
`string_value`, and `bytes_value`, each shared across several logical types — are **not** recoverable
from the `ScalarValue` alone; the type-specific arms are self-identifying, but a reader still validates
them against the supplied `DType`. The rest of this section is the arm-by-arm mapping a reader needs
to turn a `ScalarValue` plus its `DType` back into a concrete value.

### Primitives

All integers collapse onto two arms by signedness, and `F16` rides in a `uint64`:

| Vortex `PType` | `oneof` arm | Carriage |
|----------------|-------------|----------|
| `I8`, `I16`, `I32`, `I64` | `int64_value` | value sign-extended to 64-bit |
| `U8`, `U16`, `U32`, `U64` | `uint64_value` | value zero-extended to 64-bit |
| `F16` | `f16_value` | raw 16-bit bits carried in a `uint64` |
| `F32` | `f32_value` | IEEE-754 single |
| `F64` | `f64_value` | IEEE-754 double |

`int64_value` is a Protobuf `sint64`, so on the wire it is **ZigZag-varint-coded** — not a plain
two's-complement varint. A decoder must ZigZag-decode it before the width narrowing below; reading it
as a plain `int64` varint mis-reads every negative value. `uint64_value` and `f16_value` are plain
`uint64` varints.

On decode the `DType`'s concrete `PType` narrows the wide arm back to its declared width: an
`int64_value` read against an `I16` dtype is range-checked and narrowed to `i16`, and likewise for
`uint64_value`. A value outside the target type's range is a hard error, never a silent wrap — the
narrowing is checked, not a wrapping truncation.

:::{note}
For backward compatibility a reader also accepts an integer carried under the *opposite* signedness
arm (for example a `U32` decoded from `int64_value`), and an `F16` carried in `uint64_value` — the
legacy pre-`f16_value` encoding, whose payload is `f16::to_bits() as u64`. A writer **MUST** use the
arm in the table above; a reader **SHOULD** accept these legacy forms so that older statistics remain
readable.
:::

### Strings, binary, and decimals

The `bytes_value` arm is overloaded across three logical types, and `Utf8`/`Binary` may also travel
in `string_value`. A writer emits `Utf8` as `string_value` and both `Binary` and `Decimal` as
`bytes_value`; a reader disambiguates **solely** by the supplied `DType`:

| `DType` | Decodes from | Interpreted as |
|---------|--------------|----------------|
| `Utf8` | `string_value` or `bytes_value` | UTF-8 string bytes |
| `Binary` | `string_value` or `bytes_value` | raw bytes, verbatim |
| `Decimal` | `bytes_value` | little-endian two's-complement integer (below) |

A `Decimal` scalar has no arm of its own. Its backing integer is serialized into `bytes_value` as
**little-endian two's-complement** bytes, and the integer width is recovered on decode from the byte
length alone:

| `bytes_value` length | Backing integer |
|----------------------|-----------------|
| 1 | `i8` |
| 2 | `i16` |
| 4 | `i32` |
| 8 | `i64` |
| 16 | `i128` |
| 32 | `i256` |

Any other length is a hard error. The decimal's precision and scale are **not** carried in the
`ScalarValue`; they come from the `Decimal` `DType`.

### Booleans, lists, and variants

Three arms decode directly against their matching `DType`; a type mismatch is a hard error, never a
reinterpretation.

- **`bool_value`** — a Protobuf `bool`, valid only against a `Bool` `DType` (`proto.rs` `bool_from_proto`).
- **`list_value`** — a `ListValue` carrying a `repeated ScalarValue`. Each element is decoded
  recursively against the **element** type of the enclosing `List` `DType` (`proto.rs` `list_from_proto`). As
  the Model section notes above, the deserializer accepts `list_value` only for a `List` dtype today;
  `FixedSizeList` and `Struct` scalars do not yet round-trip through this Protobuf.
- **`variant_value`** — a fully nested `Scalar` (its own `dtype` and `value`), valid only against a
  `Variant` `DType` (`proto.rs` `ScalarValue::from_proto`). Because it carries its own `dtype` it is self-describing: a
  reader decodes it by recursing into the ordinary `Scalar` decode.

### Extension types

Before matching any arm, a decoder handed an `Extension` `DType` substitutes the extension's
`storage_dtype` and decodes the value against that instead. An extension scalar is therefore encoded
exactly as its storage value would be; the extension identity lives only in the `DType`.

### Null

`null_value` uses the well-known `google.protobuf.NullValue` enum (imported from
`google/protobuf/struct.proto`), not a Vortex-defined type — a non-Rust reader must resolve it from
the standard Protobuf descriptors. It denotes absence regardless of the declared `DType`.
